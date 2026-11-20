// Schedule Device work issued by the Udon Host without a frame per command.
//
// On the Udon target a vrc.dispatch, vrc.buffer_copy or vrc.buffer_fill is a
// Blit recorded on Unity's in-order graphics queue when the Host issues it;
// its event is complete at issue (the scalar lowering stores CL_SUCCESS).
// Awaiting such an event therefore observes nothing, yet every vrc.await
// suspends the Host for at least one frame. Only Host transfers
// (upload/readback commands executed by the generated binding) and other
// external events need a suspension.
//
// This pass
//  1. folds vrc.await of an event produced by such a Device command to
//     CL_SUCCESS, so issuing Device work is enqueue-only, and
//  2. bounds the work one Udon event issues: after each Device command in a
//     may-suspend function it charges the command's static cost to a slice
//     budget held in instance globals, and each folded await becomes a
//     cooperative checkpoint taken only when the slice is full. Yields
//     therefore stay at points where the Host could already suspend, so no
//     new value becomes live across a suspension. Any suspension starts a new
//     slice.
//
// A slice has two budgets.
//  - Device cost, in texel-load units: every launched fragment costs
//    `fragment-weight` units (raster, output write and Blit overhead) and
//    every texel load one unit. Loads per fragment come from the kernel's
//    declared `vrc.fragment_cost` (fixed loads plus loads per unit of a
//    scalar argument, over the fragments selected by a product of scalar
//    arguments), evaluated with the dispatch site's constant arguments or the
//    declared argument maxima; without a declaration every launched fragment
//    reads each read resource once.
//  - Host cost, in estimated Udon instructions: issuing a command binds every
//    kernel argument and Blits (kUdonPerArgument per argument plus one), and
//    the Host operations its block runs since the previous command cost
//    kUdonPerHostOperation each. The constants were fitted to the Udon VM
//    emulator, where the budget slices of the generated TinyLlama Device
//    graph spend about 430 Udon instructions per dispatch of 8 to 14
//    arguments.
// Ordering is unchanged: commands stay in program order on the in-order
// queue, a Host read of Device memory remains a readback command that is
// awaited, and transfers still drain the single command slot before the next
// one is published. A checkpoint carries no Device dependence.
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_SCHEDULEUDONDEVICEWORKPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace host = ::vrc::udonhl;
namespace runtime = ::vrc::vrc;

// Estimated Udon instructions to bind one kernel argument (or Blit), and per
// Host operation preceding a command; see the file comment.
constexpr int64_t kUdonPerArgument = 32;
constexpr int64_t kUdonPerHostOperation = 8;

constexpr StringLiteral kDeviceCostSymbol = "__udoncl_slice_device_cost";
constexpr StringLiteral kHostCostSymbol = "__udoncl_slice_host_cost";

Value stripCasts(Value value) {
  while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  return value;
}

/// A Device command is recorded as one Blit and is complete at issue.
bool isDeviceCommand(Operation *operation) {
  return isa<runtime::DispatchOp, runtime::BufferCopyOp, runtime::BufferFillOp>(
      operation);
}

bool completesAtIssue(Value event) {
  Operation *producer = stripCasts(event).getDefiningOp();
  return producer && isDeviceCommand(producer);
}

std::optional<int64_t> constantI32(Value value) {
  value = stripCasts(value);
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  return std::nullopt;
}

int64_t saturatingMultiply(int64_t left, int64_t right) {
  constexpr int64_t limit = int64_t(1) << 50;
  if (left <= 0 || right <= 0)
    return 0;
  return left > limit / right ? limit : left * right;
}

/// Value of kernel argument `index` at a dispatch site: the constant scalar
/// operand when it is one, otherwise the declared maximum `fallback`.
std::optional<int64_t> argumentValue(runtime::DispatchOp dispatch,
                                     int64_t index, int64_t fallback) {
  ArrayRef<Attribute> kinds = dispatch.getArgumentKinds().getValue();
  if (index < 0 || index >= static_cast<int64_t>(kinds.size()))
    return std::nullopt;
  auto kind = dyn_cast<StringAttr>(kinds[index]);
  if (!kind || kind.getValue() == "buffer")
    return std::nullopt;
  int64_t scalar = 0;
  for (Attribute previous : kinds.take_front(index))
    if (cast<StringAttr>(previous).getValue() != "buffer")
      ++scalar;
  auto descriptor = stripCasts(dispatch.getDesc())
                        .getDefiningOp<runtime::DispatchDescCreateOp>();
  if (descriptor &&
      scalar < static_cast<int64_t>(descriptor.getScalars().size()))
    if (std::optional<int64_t> value =
            constantI32(descriptor.getScalars()[scalar]))
      return std::max<int64_t>(*value, 0);
  return std::max<int64_t>(fallback, 0);
}

/// Static Device cost of one command in texel-load units. Returns failure
/// for a malformed `vrc.fragment_cost` declaration.
FailureOr<int64_t> deviceCost(Operation *command, int64_t fragmentWeight) {
  if (auto copy = dyn_cast<runtime::BufferCopyOp>(command)) {
    // A copy is a Blit of the destination words: one load per fragment.
    std::optional<int64_t> bytes = constantI32(copy.getByteSize());
    int64_t words = bytes ? (*bytes + 3) / 4 : 131072;
    return saturatingMultiply(std::max<int64_t>(words, 1), fragmentWeight + 1);
  }
  if (auto fill = dyn_cast<runtime::BufferFillOp>(command)) {
    // A fill is a Blit of the destination words that loads nothing.
    std::optional<int64_t> bytes = constantI32(fill.getByteSize());
    int64_t words = bytes ? (*bytes + 3) / 4 : 131072;
    return saturatingMultiply(std::max<int64_t>(words, 1), fragmentWeight);
  }
  auto dispatch = cast<runtime::DispatchOp>(command);
  int64_t fragments = 1;
  for (int32_t extent : dispatch.getGlobalSize())
    fragments = saturatingMultiply(fragments, std::max<int32_t>(extent, 1));
  int64_t active = fragments;
  int64_t loads = llvm::count_if(
      dispatch.getResourceAccesses().getValue(), [](Attribute access) {
        return cast<StringAttr>(access).getValue() == "read";
      });
  if (auto declared =
          dispatch->getAttrOfType<DictionaryAttr>("vrc.fragment_cost")) {
    // {loads, active = [{index, max}], per_argument = [{index, loads, max}]}
    auto fixed = declared.getAs<IntegerAttr>("loads");
    auto activeTerms = declared.getAs<ArrayAttr>("active");
    auto perArgument = declared.getAs<ArrayAttr>("per_argument");
    if (!fixed || fixed.getInt() < 0)
      return failure();
    loads = fixed.getInt();
    if (activeTerms) {
      int64_t product = 1;
      for (Attribute term : activeTerms) {
        auto entry = dyn_cast<DictionaryAttr>(term);
        auto index = entry ? entry.getAs<IntegerAttr>("index") : IntegerAttr();
        auto maximum = entry ? entry.getAs<IntegerAttr>("max") : IntegerAttr();
        std::optional<int64_t> value =
            index && maximum
                ? argumentValue(dispatch, index.getInt(), maximum.getInt())
                : std::nullopt;
        if (!value)
          return failure();
        product = saturatingMultiply(product, std::max<int64_t>(*value, 1));
      }
      active = std::min(active, product);
    }
    for (Attribute term :
         perArgument ? perArgument.getValue() : ArrayRef<Attribute>()) {
      auto entry = dyn_cast<DictionaryAttr>(term);
      auto index = entry ? entry.getAs<IntegerAttr>("index") : IntegerAttr();
      auto unit = entry ? entry.getAs<IntegerAttr>("loads") : IntegerAttr();
      auto maximum = entry ? entry.getAs<IntegerAttr>("max") : IntegerAttr();
      std::optional<int64_t> value =
          index && unit && maximum && unit.getInt() >= 0
              ? argumentValue(dispatch, index.getInt(), maximum.getInt())
              : std::nullopt;
      if (!value)
        return failure();
      loads += saturatingMultiply(*value, unit.getInt());
    }
  }
  return saturatingMultiply(fragments, fragmentWeight) +
         saturatingMultiply(active, loads);
}

struct ScheduleUdonDeviceWorkPass
    : impl::ScheduleUdonDeviceWorkPassBase<ScheduleUdonDeviceWorkPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    // Counters are i32 globals that saturate at their budget, so a budget of
    // at most 2^30 keeps every partial sum representable.
    constexpr int64_t kMaxBudget = int64_t(1) << 30;
    if (deviceCostBudget < 1 || hostCostBudget < 1 || fragmentWeight < 0 ||
        deviceCostBudget > kMaxBudget || hostCostBudget > kMaxBudget) {
      module.emitError("Device work slice budgets must be in [1, 2^30] and "
                       "the fragment weight non-negative");
      return signalPassFailure();
    }

    // Charge every Device command issued by a may-suspend function, and turn
    // every await of a complete-at-issue event into a budget yield point at
    // the same position. Keeping yields exactly where the Host already could
    // suspend preserves the values live across suspension points.
    SmallVector<Operation *> commands;
    SmallVector<runtime::AwaitOp> yieldPoints;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      auto effect = function->getAttrOfType<StringAttr>("vrc.async_effect");
      if (function.isDeclaration() || !effect ||
          effect.getValue() != "may_suspend")
        continue;
      WalkResult result = function.walk([&](Operation *operation) {
        bool command = isDeviceCommand(operation);
        auto await = dyn_cast<runtime::AwaitOp>(operation);
        bool foldable = await &&
                        !await->hasAttr("vrc.cooperative_checkpoint") &&
                        completesAtIssue(await.getEvent());
        if (!command && !foldable)
          return WalkResult::advance();
        // The Host reaches this pass with unstructured control flow; a yield
        // edge cannot be added inside a structured region.
        if (operation->getParentOp() != function.getOperation()) {
          operation->emitError("Device work in a structured region cannot be "
                               "scheduled");
          return WalkResult::interrupt();
        }
        if (command)
          commands.push_back(operation);
        else
          yieldPoints.push_back(await);
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return signalPassFailure();
    }
    // Programs without Device work in may-suspend code are left byte-for-byte
    // unchanged.
    if (!yieldPoints.empty())
      module->setAttr(
          "vrc.device_work_folded_awaits",
          IntegerAttr::get(IntegerType::get(context, 32),
                           static_cast<int32_t>(yieldPoints.size())));
    if (commands.empty() && yieldPoints.empty())
      return;

    // Static costs, measured before any charging code is inserted. The Host
    // cost of a command is the number of operations its block issues since
    // the previous command (or the block entry), the command included.
    SmallVector<std::pair<int64_t, int64_t>> costs;
    int64_t totalDevice = 0, totalHost = 0;
    for (Operation *command : commands) {
      FailureOr<int64_t> device = deviceCost(command, fragmentWeight);
      if (failed(device)) {
        command->emitError("malformed vrc.fragment_cost declaration");
        return signalPassFailure();
      }
      int64_t operations = 1;
      for (Operation *previous = command->getPrevNode();
           previous && !isDeviceCommand(previous);
           previous = previous->getPrevNode())
        ++operations;
      int64_t arguments = 2; // a copy binds its source and destination
      if (auto dispatch = dyn_cast<runtime::DispatchOp>(command))
        arguments = dispatch.getArgumentKinds().size();
      else if (isa<runtime::BufferFillOp>(command))
        arguments = 8; // destination, range, width, period and pattern words
      int64_t host = kUdonPerArgument * (arguments + 1) +
                     kUdonPerHostOperation * operations;
      costs.push_back({*device, host});
      totalDevice = std::min<int64_t>(totalDevice + *device, int64_t(1) << 50);
      totalHost += host;
    }
    // Static totals over all issued command sites, for reports and tests.
    module->setAttr(
        "vrc.device_work_static_cost",
        DictionaryAttr::get(
            context,
            {NamedAttribute(StringAttr::get(context, "commands"),
                            IntegerAttr::get(IntegerType::get(context, 64),
                                             commands.size())),
             NamedAttribute(
                 StringAttr::get(context, "device"),
                 IntegerAttr::get(IntegerType::get(context, 64), totalDevice)),
             NamedAttribute(
                 StringAttr::get(context, "host"),
                 IntegerAttr::get(IntegerType::get(context, 64), totalHost))}));

    SymbolTable symbols(module);
    for (StringRef symbol : {kDeviceCostSymbol, kHostCostSymbol}) {
      if (symbols.lookup(symbol)) {
        module.emitError("Device work budget symbol already exists: ")
            << symbol;
        return signalPassFailure();
      }
      OpBuilder builder(module.getBodyRegion());
      builder.setInsertionPointToEnd(module.getBody());
      OperationState state(module.getLoc(), host::GlobalOp::getOperationName());
      state.addAttribute("sym_name", builder.getStringAttr(symbol));
      state.addAttribute("storage_type", TypeAttr::get(builder.getI32Type()));
      state.addAttribute("initial_value", builder.getI32IntegerAttr(0));
      state.addAttribute("linkage", builder.getStringAttr("internal"));
      state.addAttribute("address_space", builder.getI32IntegerAttr(0));
      state.addAttribute("alignment", builder.getI64IntegerAttr(4));
      builder.create(state);
    }

    Type i32 = IntegerType::get(context, 32);
    Type pointer = host::PtrType::get(context, 0);
    auto constant = [&](OpBuilder &builder, Location location, int64_t value) {
      return arith::ConstantIntOp::create(builder, location, value, 32)
          .getResult();
    };
    auto addressOf = [&](OpBuilder &builder, Location location,
                         StringRef symbol) {
      OperationState state(location, host::AddressOfOp::getOperationName());
      state.addAttribute("global", FlatSymbolRefAttr::get(context, symbol));
      state.addTypes(pointer);
      return builder.create(state)->getResult(0);
    };
    auto load = [&](OpBuilder &builder, Location location, StringRef symbol) {
      OperationState state(location, host::LoadOp::getOperationName());
      state.addOperands(addressOf(builder, location, symbol));
      state.addTypes(i32);
      state.addAttribute("alignment", builder.getI64IntegerAttr(4));
      return builder.create(state)->getResult(0);
    };
    auto store = [&](OpBuilder &builder, Location location, Value value,
                     StringRef symbol) {
      Value address = addressOf(builder, location, symbol);
      OperationState state(location, host::StoreOp::getOperationName());
      state.addOperands({value, address});
      state.addAttribute("alignment", builder.getI64IntegerAttr(4));
      builder.create(state);
    };

    // Every remaining await suspends the Host until a later frame, so the
    // work issued after it starts a new slice.
    SmallVector<runtime::AwaitOp> suspensions;
    module.walk([&](runtime::AwaitOp await) {
      if (!llvm::is_contained(yieldPoints, await))
        suspensions.push_back(await);
    });
    for (runtime::AwaitOp await : suspensions) {
      OpBuilder builder(context);
      builder.setInsertionPointAfter(await);
      Value zero = constant(builder, await.getLoc(), 0);
      store(builder, await.getLoc(), zero, kDeviceCostSymbol);
      store(builder, await.getLoc(), zero, kHostCostSymbol);
    }

    // Charge each command after it is issued. Each counter saturates at its
    // budget so it cannot overflow.
    auto charge = [&](OpBuilder &builder, Location location, StringRef symbol,
                      int64_t cost, int64_t budget) {
      Value used = arith::AddIOp::create(
          builder, location, load(builder, location, symbol),
          constant(builder, location, std::min(cost, budget)));
      store(builder, location,
            arith::MinSIOp::create(builder, location, used,
                                   constant(builder, location, budget)),
            symbol);
    };
    for (auto [command, cost] : llvm::zip_equal(commands, costs)) {
      OpBuilder builder(context);
      builder.setInsertionPointAfter(command);
      charge(builder, command->getLoc(), kDeviceCostSymbol, cost.first,
             deviceCostBudget);
      charge(builder, command->getLoc(), kHostCostSymbol, cost.second,
             hostCostBudget);
    }

    // At each former await of a complete-at-issue event, yield only when
    // the slice is full; the event itself is CL_SUCCESS.
    for (runtime::AwaitOp await : yieldPoints) {
      Location location = await.getLoc();
      Block *before = await->getBlock();
      Block *after = before->splitBlock(await->getNextNode());
      Region *region = before->getParent();
      Block *yield = new Block();
      region->getBlocks().insert(Region::iterator(after), yield);

      // Materialize the folded status after the yield point so no constant
      // becomes live across the possible suspension.
      OpBuilder builder(context);
      builder.setInsertionPointToStart(after);
      Value success = constant(builder, location, 0);
      await.getStatus().replaceAllUsesWith(success);
      builder.setInsertionPoint(await);
      Value full = arith::OrIOp::create(
          builder, location,
          arith::CmpIOp::create(builder, location, arith::CmpIPredicate::sge,
                                load(builder, location, kDeviceCostSymbol),
                                constant(builder, location, deviceCostBudget)),
          arith::CmpIOp::create(builder, location, arith::CmpIPredicate::sge,
                                load(builder, location, kHostCostSymbol),
                                constant(builder, location, hostCostBudget)));
      cf::CondBranchOp::create(builder, location, full, yield, ValueRange{},
                               after, ValueRange{});
      // The event is now unused: drop the casts that only carried it to the
      // await so no dead value reaches liveness and storage planning.
      Value event = await.getEvent();
      await.erase();
      while (auto cast = event.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (!cast->use_empty() || cast.getNumOperands() != 1)
          break;
        event = cast.getOperand(0);
        cast.erase();
      }

      builder.setInsertionPointToEnd(yield);
      Value checkpointEvent =
          UnrealizedConversionCastOp::create(
              builder, location, TypeRange{runtime::EventType::get(context)},
              ValueRange{constant(builder, location, 0)})
              .getResult(0);
      auto checkpoint =
          runtime::AwaitOp::create(builder, location, i32, checkpointEvent);
      checkpoint->setAttr("vrc.cooperative_checkpoint", builder.getUnitAttr());
      checkpoint->setAttr("vrc.device_work_budget_checkpoint",
                          builder.getUnitAttr());
      Value zero = constant(builder, location, 0);
      store(builder, location, zero, kDeviceCostSymbol);
      store(builder, location, zero, kHostCostSymbol);
      cf::BranchOp::create(builder, location, after);
    }
  }
};
} // namespace
} // namespace vrc
