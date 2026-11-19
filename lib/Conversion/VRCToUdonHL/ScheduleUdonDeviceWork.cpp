// Schedule Device work issued by the Udon Host without a frame per command.
//
// On the Udon target a vrc.dispatch or vrc.buffer_copy is a Blit recorded on
// Unity's in-order graphics queue when the Host issues it; its event is
// complete at issue (the scalar lowering stores CL_SUCCESS). Awaiting such an
// event therefore observes nothing, yet every vrc.await suspends the Host for
// at least one frame. Only Host transfers (upload/readback commands executed
// by the generated binding) and other external events need a suspension.
//
// This pass
//  1. folds vrc.await of an event produced by vrc.dispatch or vrc.buffer_copy
//     to CL_SUCCESS, so issuing Device work is enqueue-only, and
//  2. bounds the work one Udon event issues: after each Device command in a
//     may-suspend function it charges the command to a slice budget (command
//     count and fragments from the static launch size) held in instance
//     globals, and each folded await becomes a cooperative checkpoint taken
//     only when the slice is full. Yields therefore stay at points where the
//     Host could already suspend, so no new value becomes live across a
//     suspension. Any suspension starts a new slice.
//
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

constexpr StringLiteral kCommandBudgetSymbol = "__udoncl_slice_device_commands";
constexpr StringLiteral kFragmentBudgetSymbol =
    "__udoncl_slice_device_fragments";

Value stripCasts(Value value) {
  while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  return value;
}

bool completesAtIssue(Value event) {
  Operation *producer = stripCasts(event).getDefiningOp();
  return producer && isa<runtime::DispatchOp, runtime::BufferCopyOp>(producer);
}

int64_t fragmentCost(Operation *command) {
  auto dispatch = dyn_cast<runtime::DispatchOp>(command);
  if (!dispatch)
    return 1;
  auto size = dispatch->getAttrOfType<DenseI32ArrayAttr>("global_size");
  if (!size)
    return 1;
  int64_t fragments = 1;
  for (int32_t extent : size.asArrayRef())
    fragments *= std::max<int32_t>(extent, 1);
  return std::max<int64_t>(fragments, 1);
}

struct ScheduleUdonDeviceWorkPass
    : impl::ScheduleUdonDeviceWorkPassBase<ScheduleUdonDeviceWorkPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    if (commandBudget < 1 || fragmentBudget < 1) {
      module.emitError("Device work slice budgets must be positive");
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
        bool command =
            isa<runtime::DispatchOp, runtime::BufferCopyOp>(operation);
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

    SymbolTable symbols(module);
    for (StringRef symbol : {kCommandBudgetSymbol, kFragmentBudgetSymbol}) {
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
      store(builder, await.getLoc(), zero, kCommandBudgetSymbol);
      store(builder, await.getLoc(), zero, kFragmentBudgetSymbol);
    }

    // Charge each command after it is issued. Fragments saturate at the
    // budget so the counter cannot overflow.
    for (Operation *command : commands) {
      OpBuilder builder(context);
      builder.setInsertionPointAfter(command);
      Location location = command->getLoc();
      Value commandsUsed = arith::AddIOp::create(
          builder, location, load(builder, location, kCommandBudgetSymbol),
          constant(builder, location, 1));
      store(builder, location, commandsUsed, kCommandBudgetSymbol);
      Value fragments = arith::AddIOp::create(
          builder, location, load(builder, location, kFragmentBudgetSymbol),
          constant(builder, location,
                   std::min<int64_t>(fragmentCost(command), fragmentBudget)));
      store(builder, location,
            arith::MinSIOp::create(builder, location, fragments,
                                   constant(builder, location, fragmentBudget)),
            kFragmentBudgetSymbol);
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
                                load(builder, location, kCommandBudgetSymbol),
                                constant(builder, location, commandBudget)),
          arith::CmpIOp::create(builder, location, arith::CmpIPredicate::sge,
                                load(builder, location, kFragmentBudgetSymbol),
                                constant(builder, location, fragmentBudget)));
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
      store(builder, location, zero, kCommandBudgetSymbol);
      store(builder, location, zero, kFragmentBudgetSymbol);
      cf::BranchOp::create(builder, location, after);
    }
  }
};
} // namespace
} // namespace vrc
