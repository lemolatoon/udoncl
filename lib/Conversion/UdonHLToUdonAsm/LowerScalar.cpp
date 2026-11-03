#include "VRC/Conversion/Passes.h"
#include "VRC/ABI/Manifest/UdonSharpABI.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_LOWERUDONHLTOUDONASMPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace vm = ::vrc::udonasm;
namespace host = ::vrc::udonhl;

struct Slot {
  std::string name;
  std::string type;
  std::string initial;
  bool isPublic = false;
};

struct FunctionPlan {
  std::string prefix;
  SmallVector<std::string> parameterSlots;
  std::optional<std::string> resultSlot;
  std::optional<std::string> returnSlot;
  std::optional<uint64_t> entryAddress;
};

static FailureOr<StringRef> udonType(Type type) {
  if (type.isInteger(1))
    return StringRef("SystemBoolean");
  if (type.isInteger(32))
    return StringRef("SystemInt32");
  if (type.isF32())
    return StringRef("SystemSingle");
  if (type.isIndex())
    return StringRef("SystemInt32");
  if (isa<::vrc::vrc::BufferType>(type))
    return StringRef("UnityEngineRenderTexture");
  if (isa<::vrc::vrc::EventType>(type))
    return StringRef("SystemInt32");
  if (isa<host::FrameType>(type))
    return StringRef("SystemInt32");
  if (auto memref = dyn_cast<MemRefType>(type)) {
    if (memref.getRank() != 1 || !memref.getLayout().isIdentity())
      return failure();
    if (memref.getElementType().isF32())
      return StringRef("SystemSingleArray");
    if (memref.getElementType().isInteger(32))
      return StringRef("SystemInt32Array");
  }
  return failure();
}

static StringRef arrayLoadExtern(MemRefType type) {
  if (type.getElementType().isF32())
    return "SystemSingleArray.__Get__SystemInt32__SystemSingle";
  if (type.getElementType().isInteger(32))
    return "SystemInt32Array.__Get__SystemInt32__SystemInt32";
  return {};
}

static StringRef arrayStoreExtern(MemRefType type) {
  if (type.getElementType().isF32())
    return "SystemSingleArray.__Set__SystemInt32_SystemSingle__SystemVoid";
  if (type.getElementType().isInteger(32))
    return "SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid";
  return {};
}

static FailureOr<UdonSharpMethodABI>
methodABI(func::FuncOp function, UdonSharpABIBuilder &abiBuilder) {
  auto publicNameAttr = function->getAttrOfType<StringAttr>("vrc.public_name");
  StringRef publicName =
      publicNameAttr ? publicNameAttr.getValue() : function.getSymName();
  if (!isUdonSharpIdentifier(publicName))
    return failure();

  SmallVector<std::string> ownedNames;
  SmallVector<StringRef> parameterNames;
  if (auto names = function->getAttrOfType<ArrayAttr>("vrc.parameter_names")) {
    if (names.size() != function.getNumArguments())
      return failure();
    for (Attribute name : names) {
      auto string = dyn_cast<StringAttr>(name);
      if (!string || !isUdonSharpIdentifier(string.getValue()))
        return failure();
      parameterNames.push_back(string.getValue());
    }
  } else {
    ownedNames.reserve(function.getNumArguments());
    parameterNames.reserve(function.getNumArguments());
    for (unsigned index = 0; index < function.getNumArguments(); ++index)
      ownedNames.push_back(defaultUdonSharpParameterName(index));
    for (const std::string &name : ownedNames)
      parameterNames.push_back(name);
  }
  return abiBuilder.buildMethod(publicName, parameterNames,
                                function.getNumResults() != 0);
}

static StringRef binaryExtern(Operation *operation) {
  if (isa<arith::AddIOp>(operation))
    return "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32";
  if (isa<arith::SubIOp>(operation))
    return "SystemInt32.__op_Subtraction__SystemInt32_SystemInt32__SystemInt32";
  if (isa<arith::MulIOp>(operation))
    return "SystemInt32.__op_Multiply__SystemInt32_SystemInt32__SystemInt32";
  if (isa<arith::AddFOp>(operation))
    return "SystemSingle.__op_Addition__SystemSingle_SystemSingle__"
           "SystemSingle";
  if (isa<arith::SubFOp>(operation))
    return "SystemSingle.__op_Subtraction__SystemSingle_SystemSingle__"
           "SystemSingle";
  if (isa<arith::MulFOp>(operation))
    return "SystemSingle.__op_Multiplication__SystemSingle_SystemSingle__"
           "SystemSingle";
  if (isa<arith::DivFOp>(operation))
    return "SystemSingle.__op_Division__SystemSingle_SystemSingle__"
           "SystemSingle";
  return {};
}

static StringRef comparisonExtern(arith::CmpIOp operation) {
  StringRef name;
  switch (operation.getPredicate()) {
  case arith::CmpIPredicate::eq:
    name = "Equality";
    break;
  case arith::CmpIPredicate::ne:
    name = "Inequality";
    break;
  case arith::CmpIPredicate::slt:
    name = "LessThan";
    break;
  case arith::CmpIPredicate::sle:
    name = "LessThanOrEqual";
    break;
  case arith::CmpIPredicate::sgt:
    name = "GreaterThan";
    break;
  case arith::CmpIPredicate::sge:
    name = "GreaterThanOrEqual";
    break;
  default:
    return {};
  }
  static thread_local std::string signature;
  signature = "SystemInt32.__op_" + name.str() +
              "__SystemInt32_SystemInt32__SystemBoolean";
  return signature;
}

static StringRef comparisonExtern(arith::CmpFOp operation) {
  StringRef name;
  switch (operation.getPredicate()) {
  case arith::CmpFPredicate::OEQ:
    name = "Equality";
    break;
  case arith::CmpFPredicate::ONE:
    name = "Inequality";
    break;
  case arith::CmpFPredicate::OLT:
    name = "LessThan";
    break;
  case arith::CmpFPredicate::OLE:
    name = "LessThanOrEqual";
    break;
  case arith::CmpFPredicate::OGT:
    name = "GreaterThan";
    break;
  case arith::CmpFPredicate::OGE:
    name = "GreaterThanOrEqual";
    break;
  default:
    return {};
  }
  static thread_local std::string signature;
  signature = "SystemSingle.__op_" + name.str() +
              "__SystemSingle_SystemSingle__SystemBoolean";
  return signature;
}

class ScalarLowering {
public:
  ScalarLowering(ModuleOp module)
      : module(module), builder(module.getContext()) {
    builder.setInsertionPointToEnd(module.getBody());
  }

  LogicalResult run() {
    SmallVector<func::FuncOp> functions;
    SmallVector<func::FuncOp> exportsToLower;
    SmallVector<func::FuncOp> internalsToLower;
    llvm::SmallSet<int64_t, 8> exportIds;
    llvm::SmallSet<std::string, 8> eventNames;
    llvm::SmallSet<std::string, 8> publicNames;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      auto exportId = function->getAttrOfType<IntegerAttr>("vrc.export_id");
      if (exportId && (exportId.getInt() <= 0 ||
                       !exportIds.insert(exportId.getInt()).second))
        return function.emitError("requires a unique positive vrc.export_id");
      if (function.isExternal())
        return function.emitError(
            "Udon lowering requires runtime intrinsics to be substituted and "
            "every remaining function to be defined");
      if (exportId && function.isPrivate())
        return function.emitError("a Udon export cannot be private");
      if (function.getNumResults() > 1)
        return function.emitError(
            "initial scalar Udon ABI supports at most one result");
      for (Type type : function.getArgumentTypes())
        if (failed(udonType(type)))
          return function.emitError("unsupported scalar Udon argument type");
      for (Type type : function.getResultTypes())
        if (failed(udonType(type)))
          return function.emitError("unsupported scalar Udon result type");
      functions.push_back(function);
      (exportId ? exportsToLower : internalsToLower).push_back(function);
    }
    if (exportsToLower.empty())
      return module.emitError(
          "scalar Udon lowering requires at least one export");

    llvm::sort(exportsToLower, [](func::FuncOp lhs, func::FuncOp rhs) {
      return lhs->getAttrOfType<IntegerAttr>("vrc.export_id").getInt() <
             rhs->getAttrOfType<IntegerAttr>("vrc.export_id").getInt();
    });
    llvm::sort(internalsToLower, [](func::FuncOp lhs, func::FuncOp rhs) {
      return lhs.getSymName() < rhs.getSymName();
    });
    functions.clear();
    llvm::append_range(functions, exportsToLower);
    llvm::append_range(functions, internalsToLower);

    UdonSharpABIBuilder abiBuilder;
    for (func::FuncOp function : exportsToLower) {
      auto abi = methodABI(function, abiBuilder);
      if (failed(abi))
        return function.emitError(
            "invalid UdonSharp public or parameter identifier ABI");
      if (!publicNames.insert(abi->publicName).second)
        return function.emitError("duplicate UdonSharp public export name");
      if (auto explicitEvent =
              function->getAttrOfType<StringAttr>("vrc.udon_event"))
        if (explicitEvent.getValue() != abi->eventName)
          return function.emitError(
              "vrc.udon_event does not match the pinned UdonSharp ABI");
      if (!eventNames.insert(abi->eventName).second)
        return function.emitError("duplicate UdonSharp export event name");
      methodABIs.try_emplace(function.getOperation(), std::move(*abi));
    }

    for (auto [index, function] : llvm::enumerate(functions)) {
      FunctionPlan plan;
      if (auto exportId =
              function->getAttrOfType<IntegerAttr>("vrc.export_id")) {
        plan.prefix = "__vrc_f" + std::to_string(exportId.getInt()) + "_";
        const UdonSharpMethodABI &abi = methodABIs.at(function.getOperation());
        plan.parameterSlots.append(abi.parameterSlots.begin(),
                                   abi.parameterSlots.end());
        plan.resultSlot = abi.resultSlot;
      } else {
        plan.prefix = "__vrc_internal_" + std::to_string(index) + "_";
        for (unsigned argument = 0; argument < function.getNumArguments();
             ++argument)
          plan.parameterSlots.push_back(plan.prefix + "arg" +
                                        std::to_string(argument));
        if (function.getNumResults())
          plan.resultSlot = plan.prefix + "result";
        plan.returnSlot = plan.prefix + "return";
      }
      functionPlans.try_emplace(function.getOperation(), std::move(plan));
    }

    for (func::FuncOp function : functions)
      if (failed(lower(function)))
        return failure();

    for (auto [jump, callee] : callRelocations) {
      auto plan = functionPlans.find(callee.getOperation());
      if (plan == functionPlans.end() || !plan->second.entryAddress)
        return jump->emitError("call target was not lowered");
      jump->setAttr("target", builder.getI64IntegerAttr(static_cast<int64_t>(
                                  *plan->second.entryAddress)));
    }

    for (func::FuncOp function : functions)
      function.erase();
    module->setAttr("udonasm.data", dataMetadata());
    module->setAttr("udonasm.exports",
                    ArrayAttr::get(module.getContext(), exports));
    module->setAttr("udonasm.stage", builder.getStringAttr("scalar-v1"));
    module->removeAttr("vrc.host_stage");
    return success();
  }

private:
  LogicalResult lower(func::FuncOp function) {
    auto exportId = function->getAttrOfType<IntegerAttr>("vrc.export_id");
    FunctionPlan &plan = functionPlans.at(function.getOperation());
    std::string prefix = plan.prefix;
    plan.entryAddress = address;

    if (exportId) {
      const UdonSharpMethodABI &abi = methodABIs.at(function.getOperation());
      SmallVector<NamedAttribute> exportFields;
      exportFields.push_back(
          builder.getNamedAttr("name", builder.getStringAttr(abi.eventName)));
      exportFields.push_back(builder.getNamedAttr(
          "address", builder.getI64IntegerAttr(static_cast<int64_t>(address))));
      exportFields.push_back(builder.getNamedAttr(
          "function", builder.getStringAttr(function.getSymName())));
      exportFields.push_back(builder.getNamedAttr("export_id", exportId));
      exports.push_back(DictionaryAttr::get(module.getContext(), exportFields));
    }

    llvm::DenseMap<Value, std::string> valueSlots;
    llvm::DenseMap<Operation *, SmallVector<std::string>> frameSlots;
    for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
      auto type = *udonType(argument.getType());
      std::string name = plan.parameterSlots[index];
      addSlot(name, type, "null", false);
      valueSlots[argument] = name;
    }
    std::string resultSlot;
    if (function.getNumResults()) {
      auto type = *udonType(function.getResultTypes().front());
      resultSlot = *plan.resultSlot;
      addSlot(resultSlot, type, "null", false);
    }
    if (plan.returnSlot)
      addSlot(*plan.returnSlot, "SystemUInt32", "0", false);
    if (exportId)
      addSlot(udonclStatusSlot(exportId.getInt()), "SystemInt32", "0", false);

    // Allocate every SSA value before emitting code. This makes block argument
    // transfer and forward-branch relocation independent of traversal order.
    unsigned valueIndex = 0;
    for (auto [blockIndex, block] : llvm::enumerate(function.getBody())) {
      if (blockIndex)
        for (auto [argumentIndex, argument] :
             llvm::enumerate(block.getArguments())) {
          auto type = udonType(argument.getType());
          if (failed(type))
            return function.emitError("unsupported Udon block argument type");
          std::string name = prefix + "b" + std::to_string(blockIndex) + "_a" +
                             std::to_string(argumentIndex);
          addSlot(name, *type, "null", false);
          valueSlots[argument] = name;
        }
      for (Operation &operation : block) {
        if (!operation.getNumResults())
          continue;
        if (isa<::vrc::vrc::DispatchDescCreateOp>(operation))
          continue;
        if (auto allocation = dyn_cast<::vrc::vrc::BufferAllocOp>(operation)) {
          auto heapSlot =
              allocation->getAttrOfType<StringAttr>("vrc.unity_heap_slot");
          if (!heapSlot || heapSlot.getValue().empty())
            return allocation.emitError(
                "Unity buffer allocation requires a bound heap slot");
          addSlot(heapSlot.getValue(), "UnityEngineRenderTexture", "null",
                  true);
          valueSlots[allocation.getBuffer()] = heapSlot.getValue().str();
          std::string status = prefix + "v" + std::to_string(valueIndex++);
          addSlot(status, "SystemInt32", "0", false);
          valueSlots[allocation.getStatus()] = status;
          continue;
        }
        if (auto frame = dyn_cast<host::FrameCreateOp>(operation)) {
          std::string frameName =
              prefix + "frame_" + std::to_string(frameSlots.size());
          addSlot(frameName, "SystemInt32", "0", false);
          valueSlots[frame.getFrame()] = frameName;
          SmallVector<std::string> persistent;
          for (auto [slot, attribute] : llvm::enumerate(frame.getSlotTypes())) {
            auto type = dyn_cast<TypeAttr>(attribute);
            if (!type)
              return frame.emitError(
                  "static continuation frame requires type attributes");
            auto udon = udonType(type.getValue());
            if (failed(udon) ||
                (!type.getValue().isInteger(32) && !type.getValue().isF32()))
              return frame.emitError(
                  "static continuation frame supports only i32/f32 slots");
            std::string slotName = frameName + "_slot_" + std::to_string(slot);
            addSlot(slotName, *udon, "0", false);
            persistent.push_back(std::move(slotName));
          }
          frameSlots.try_emplace(frame.getOperation(), std::move(persistent));
          continue;
        }
        if (operation.getNumResults() != 1)
          return operation.emitError(
              "initial Udon lowering supports one result "
              "per non-allocation operation");
        auto type = udonType(operation.getResult(0).getType());
        if (failed(type))
          return operation.emitError("unsupported Udon result type");
        std::string initial = "null";
        std::string kind = "v";
        if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
          kind = "c";
          if (constant.getType().isInteger(1))
            return constant.emitError("boolean constants require "
                                      "target-verified runtime initialization");
          if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
            initial = std::to_string(integer.getInt());
          else if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) {
            llvm::SmallString<32> text;
            floating.getValue().toString(text);
            initial = text.str().str();
          } else
            return constant.emitError("unsupported scalar constant attribute");
        }
        std::string name = prefix + kind + std::to_string(valueIndex++);
        addSlot(name, *type, initial, false);
        valueSlots[operation.getResult(0)] = name;
      }
    }

    llvm::DenseMap<Block *, uint64_t> blockAddresses;
    SmallVector<std::pair<Operation *, Block *>> relocations;
    auto copies = [&](Location location, ValueRange sources,
                      Block *destination) -> LogicalResult {
      if (sources.size() != destination->getNumArguments())
        return emitError(location)
               << "branch operand count does not match destination";
      llvm::SmallSet<std::string, 8> destinations;
      for (BlockArgument argument : destination->getArguments())
        destinations.insert(valueSlots.lookup(argument));
      for (auto [source, target] :
           llvm::zip(sources, destination->getArguments())) {
        std::string sourceSlot = valueSlots.lookup(source);
        std::string targetSlot = valueSlots.lookup(target);
        if (sourceSlot.empty() || targetSlot.empty())
          return emitError(location)
                 << "branch value has no allocated Udon heap slot";
        if (sourceSlot == targetSlot)
          continue;
        if (destinations.contains(sourceSlot))
          return emitError(location)
                 << "parallel branch copy cycle requires a typed scratch slot";
        emitCopy(location, sourceSlot, targetSlot);
      }
      return success();
    };
    auto jumpTo = [&](Location location, Block *target) {
      Operation *jump = emitJump(location, 0);
      relocations.emplace_back(jump, target);
    };

    for (Block &block : function.getBody()) {
      blockAddresses[&block] = address;
      for (Operation &operation : block) {
        if (isa<arith::ConstantOp>(operation))
          continue;
        if (isa<::vrc::vrc::DispatchDescCreateOp>(operation))
          continue;
        if (isa<::vrc::vrc::BufferAllocOp>(operation))
          continue;
        if (auto frame = dyn_cast<host::FrameCreateOp>(operation)) {
          std::string execution = valueSlots.lookup(frame.getExecutionId());
          std::string handle = valueSlots.lookup(frame.getFrame());
          if (execution.empty() || handle.empty())
            return frame.emitError(
                "continuation frame has no allocated execution-id slots");
          emitCopy(frame.getLoc(), execution, handle);
          continue;
        }
        if (auto store = dyn_cast<host::FrameStoreOp>(operation)) {
          auto frame = store.getFrame().getDefiningOp<host::FrameCreateOp>();
          auto found =
              frame ? frameSlots.find(frame.getOperation()) : frameSlots.end();
          std::string value = valueSlots.lookup(store.getValue());
          if (found == frameSlots.end() || store.getSlot() < 0 ||
              static_cast<size_t>(store.getSlot()) >= found->second.size() ||
              value.empty())
            return store.emitError(
                "continuation store requires a direct static frame slot");
          emitCopy(store.getLoc(), value, found->second[store.getSlot()]);
          continue;
        }
        if (auto load = dyn_cast<host::FrameLoadOp>(operation)) {
          auto frame = load.getFrame().getDefiningOp<host::FrameCreateOp>();
          auto found =
              frame ? frameSlots.find(frame.getOperation()) : frameSlots.end();
          std::string result = valueSlots.lookup(load.getValue());
          if (found == frameSlots.end() || load.getSlot() < 0 ||
              static_cast<size_t>(load.getSlot()) >= found->second.size() ||
              result.empty())
            return load.emitError(
                "continuation load requires a direct static frame slot");
          emitCopy(load.getLoc(), found->second[load.getSlot()], result);
          continue;
        }
        if (auto dispatch = dyn_cast<host::ResumeDispatchOp>(operation)) {
          std::string state = valueSlots.lookup(dispatch.getState());
          if (state.empty())
            return dispatch.emitError(
                "continuation resume state has no allocated Udon heap slot");
          for (auto [index, destination] :
               llvm::enumerate(dispatch.getDestinations())) {
            int32_t resumeState = dispatch.getRegisteredStates()[index];
            std::string constant =
                prefix + "resume_" + std::to_string(resumeState);
            std::string matches =
                prefix + "resume_match_" + std::to_string(index);
            addSlot(constant, "SystemInt32", std::to_string(resumeState),
                    false);
            addSlot(matches, "SystemBoolean", "null", false);
            emitPush(dispatch.getLoc(), state);
            emitPush(dispatch.getLoc(), constant);
            emitPush(dispatch.getLoc(), matches);
            emitExtern(dispatch.getLoc(),
                       "SystemInt32.__op_Equality__SystemInt32_SystemInt32__"
                       "SystemBoolean");
            emitPush(dispatch.getLoc(), matches);
            Operation *next = create(
                dispatch.getLoc(), vm::JumpIfFalseOp::getOperationName(),
                {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
            address += 8;
            jumpTo(dispatch.getLoc(), destination);
            next->setAttr("target", builder.getI64IntegerAttr(
                                        static_cast<int64_t>(address)));
          }
          emitJump(dispatch.getLoc(), 0xfffffffcLL);
          continue;
        }
        if (auto suspend = dyn_cast<host::SuspendOp>(operation)) {
          host::ResumeDispatchOp owner;
          function.walk([&](host::ResumeDispatchOp candidate) {
            if (llvm::is_contained(candidate.getRegisteredStates(),
                                   suspend.getResumeState()))
              owner = candidate;
          });
          std::string state =
              owner ? valueSlots.lookup(owner.getState()) : std::string();
          if (state.empty())
            return suspend.emitError(
                "suspend has no static resume-state heap slot");
          std::string constant =
              prefix + "resume_" + std::to_string(suspend.getResumeState());
          addSlot(constant, "SystemInt32",
                  std::to_string(suspend.getResumeState()), false);
          emitCopy(suspend.getLoc(), constant, state);
          emitJump(suspend.getLoc(), 0xfffffffcLL);
          continue;
        }
        if (auto dispatch = dyn_cast<::vrc::vrc::DispatchOp>(operation)) {
          auto descriptor =
              dispatch.getDesc()
                  .getDefiningOp<::vrc::vrc::DispatchDescCreateOp>();
          auto materialSlot = dispatch->getAttrOfType<StringAttr>(
              "vrc.unity_dispatch_material_slot");
          auto outputIndex = dispatch->getAttrOfType<IntegerAttr>(
              "vrc.unity_output_resource_index");
          int64_t output = outputIndex ? outputIndex.getInt() : 0;
          std::string destination =
              output >= 0 && size_t(output) < dispatch.getResources().size()
                  ? valueSlots.lookup(dispatch.getResources()[output])
                  : std::string();
          if (!descriptor || !materialSlot || materialSlot.getValue().empty() ||
              destination.empty())
            return dispatch.emitError(
                "Unity dispatch requires a bound output, a copied scalar "
                "descriptor and a kernel material slot");
          addSlot(materialSlot.getValue(), "UnityEngineMaterial", "null", true);
          unsigned scalarIndex = 0, bufferIndex = 0;
          for (auto [argumentIndex, kindValue] :
               llvm::enumerate(dispatch.getArgumentKinds())) {
            auto kind = dyn_cast<StringAttr>(kindValue);
            if (!kind)
              return dispatch.emitError("invalid dispatch argument kind");
            if (kind.getValue() == "buffer") {
              if (bufferIndex >= dispatch.getResources().size())
                return dispatch.emitError(
                    "dispatch buffer descriptor is incomplete");
              std::string texture =
                  valueSlots.lookup(dispatch.getResources()[bufferIndex]);
              if (texture.empty())
                return dispatch.emitError(
                    "dispatch buffer has no allocated Udon heap slot");
              if (bufferIndex++ == size_t(output))
                continue;
              std::string propertySlot =
                  "__udoncl_kernel_" + std::to_string(dispatch.getKernelId()) +
                  "_arg_" + std::to_string(argumentIndex);
              addSlot(propertySlot, "SystemString",
                      argumentIndex == 0
                          ? "\"_VRC_Input\""
                          : "\"_VRC_Input" + std::to_string(argumentIndex) +
                                "\"",
                      false);
              emitPush(dispatch.getLoc(), materialSlot.getValue());
              emitPush(dispatch.getLoc(), propertySlot);
              emitPush(dispatch.getLoc(), texture);
              emitExtern(dispatch.getLoc(),
                         "UnityEngineMaterial.__SetTexture__SystemString_"
                         "UnityEngineTexture__SystemVoid");
              continue;
            }
            if (scalarIndex >= descriptor.getScalars().size())
              return dispatch.emitError("dispatch descriptor is incomplete");
            std::string scalar =
                valueSlots.lookup(descriptor.getScalars()[scalarIndex++]);
            if (scalar.empty())
              return dispatch.emitError(
                  "dispatch scalar has no allocated Udon heap slot");
            std::string propertySlot = "__udoncl_kernel_" +
                                       std::to_string(dispatch.getKernelId()) +
                                       "_arg_" + std::to_string(argumentIndex);
            addSlot(propertySlot, "SystemString",
                    "\"_VRC_Arg" + std::to_string(argumentIndex) + "\"", false);
            emitPush(dispatch.getLoc(), materialSlot.getValue());
            emitPush(dispatch.getLoc(), propertySlot);
            emitPush(dispatch.getLoc(), scalar);
            emitExtern(dispatch.getLoc(),
                       kind.getValue() == "f32"
                           ? "UnityEngineMaterial.__SetFloat__SystemString_"
                             "SystemSingle__SystemVoid"
                           : "UnityEngineMaterial.__SetInteger__SystemString_"
                             "SystemInt32__SystemVoid");
          }
          addSlot("__udoncl_null_texture", "UnityEngineTexture", "null", false);
          emitPush(dispatch.getLoc(), "__udoncl_null_texture");
          emitPush(dispatch.getLoc(), destination);
          emitPush(dispatch.getLoc(), materialSlot.getValue());
          emitExtern(
              dispatch.getLoc(),
              "VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_"
              "UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid");
          continue;
        }
        if (auto copy = dyn_cast<::vrc::vrc::BufferCopyOp>(operation)) {
          auto materialSlot =
              copy->getAttrOfType<StringAttr>("vrc.unity_copy_material_slot");
          std::string source = valueSlots.lookup(copy.getSource());
          std::string destination = valueSlots.lookup(copy.getDestination());
          if (!materialSlot || materialSlot.getValue().empty() ||
              source.empty() || destination.empty() || source == destination)
            return copy.emitError(
                "Unity buffer copy requires distinct bound buffers and a copy "
                "material slot");
          addSlot(materialSlot.getValue(), "UnityEngineMaterial", "null", true);
          emitPush(copy.getLoc(), source);
          emitPush(copy.getLoc(), destination);
          emitPush(copy.getLoc(), materialSlot.getValue());
          emitExtern(
              copy.getLoc(),
              "VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_"
              "UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid");
          continue;
        }
        if (auto call = dyn_cast<host::CallOp>(operation)) {
          auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
              call, call.getCalleeAttr());
          auto calleePlan = callee ? functionPlans.find(callee.getOperation())
                                   : functionPlans.end();
          if (!callee || calleePlan == functionPlans.end() ||
              !calleePlan->second.returnSlot)
            return call.emitError(
                "normal Udon call requires a defined internal callee");
          if (call.getNumOperands() != calleePlan->second.parameterSlots.size())
            return call.emitError("callee argument slot count mismatch");
          for (auto [operand, destination] : llvm::zip(
                   call.getOperands(), calleePlan->second.parameterSlots)) {
            std::string source = valueSlots.lookup(operand);
            if (source.empty())
              return call.emitError("call operand has no Udon heap slot");
            emitCopy(call.getLoc(), source, destination);
          }
          std::string returnAddress =
              prefix + "call_return_" + std::to_string(callReturnIndex++);
          addSlot(returnAddress, "SystemUInt32", std::to_string(address + 20),
                  false);
          emitCopy(call.getLoc(), returnAddress,
                   *calleePlan->second.returnSlot);
          Operation *jump = emitJump(call.getLoc(), 0);
          callRelocations.emplace_back(jump, callee);
          if (call.getNumResults()) {
            if (call.getNumResults() != 1 || !calleePlan->second.resultSlot)
              return call.emitError(
                  "normal Udon call supports at most one matching result");
            std::string destination = valueSlots.lookup(call.getResult(0));
            if (destination.empty())
              return call.emitError("call result has no Udon heap slot");
            emitCopy(call.getLoc(), *calleePlan->second.resultSlot,
                     destination);
          }
          continue;
        }
        if (auto returned = dyn_cast<func::ReturnOp>(operation)) {
          if (returned.getNumOperands()) {
            auto found = valueSlots.find(returned.getOperand(0));
            if (found == valueSlots.end())
              return returned.emitError(
                  "return value has no allocated Udon heap slot");
            emitCopy(returned.getLoc(), found->second, resultSlot);
          }
          if (plan.returnSlot) {
            create(returned.getLoc(), vm::JumpIndirectOp::getOperationName(),
                   {builder.getNamedAttr(
                       "slot", builder.getStringAttr(*plan.returnSlot))});
            address += 8;
          } else {
            emitJump(returned.getLoc(), 0xfffffffcLL);
          }
          continue;
        }
        if (auto branch = dyn_cast<cf::BranchOp>(operation)) {
          if (failed(copies(branch.getLoc(), branch.getDestOperands(),
                            branch.getDest())))
            return failure();
          jumpTo(branch.getLoc(), branch.getDest());
          continue;
        }
        if (auto branch = dyn_cast<cf::CondBranchOp>(operation)) {
          std::string condition = valueSlots.lookup(branch.getCondition());
          if (condition.empty())
            return branch.emitError(
                "condition has no allocated Udon heap slot");
          emitPush(branch.getLoc(), condition);
          Operation *falseJump = create(
              branch.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          if (failed(copies(branch.getLoc(), branch.getTrueDestOperands(),
                            branch.getTrueDest())))
            return failure();
          jumpTo(branch.getLoc(), branch.getTrueDest());
          falseJump->setAttr("target", builder.getI64IntegerAttr(
                                           static_cast<int64_t>(address)));
          if (failed(copies(branch.getLoc(), branch.getFalseDestOperands(),
                            branch.getFalseDest())))
            return failure();
          jumpTo(branch.getLoc(), branch.getFalseDest());
          continue;
        }
        if (auto load = dyn_cast<memref::LoadOp>(operation)) {
          if (load.getIndices().size() != 1)
            return load.emitError(
                "typed Udon array lowering requires one index");
          auto type = dyn_cast<MemRefType>(load.getMemRefType());
          StringRef signature = type ? arrayLoadExtern(type) : StringRef();
          std::string array = valueSlots.lookup(load.getMemRef());
          std::string index = valueSlots.lookup(load.getIndices().front());
          std::string result = valueSlots.lookup(load.getResult());
          if (signature.empty() || array.empty() || index.empty() ||
              result.empty())
            return load.emitError(
                "unsupported typed Udon array load or missing heap slot");
          emitPush(load.getLoc(), array);
          emitPush(load.getLoc(), index);
          emitPush(load.getLoc(), result);
          emitExtern(load.getLoc(), signature);
          continue;
        }
        if (auto store = dyn_cast<memref::StoreOp>(operation)) {
          if (store.getIndices().size() != 1)
            return store.emitError(
                "typed Udon array lowering requires one index");
          auto type = dyn_cast<MemRefType>(store.getMemRefType());
          StringRef signature = type ? arrayStoreExtern(type) : StringRef();
          std::string array = valueSlots.lookup(store.getMemRef());
          std::string index = valueSlots.lookup(store.getIndices().front());
          std::string value = valueSlots.lookup(store.getValueToStore());
          if (signature.empty() || array.empty() || index.empty() ||
              value.empty())
            return store.emitError(
                "unsupported typed Udon array store or missing heap slot");
          emitPush(store.getLoc(), array);
          emitPush(store.getLoc(), index);
          emitPush(store.getLoc(), value);
          emitExtern(store.getLoc(), signature);
          continue;
        }
        StringRef signature = binaryExtern(&operation);
        if (auto comparison = dyn_cast<arith::CmpIOp>(operation))
          signature = comparisonExtern(comparison);
        if (auto comparison = dyn_cast<arith::CmpFOp>(operation))
          signature = comparisonExtern(comparison);
        if (signature.empty() || operation.getNumOperands() != 2 ||
            operation.getNumResults() != 1)
          return operation.emitError("operation is not in the initial scalar "
                                     "Udon extern legalization set");
        auto lhs = valueSlots.find(operation.getOperand(0));
        auto rhs = valueSlots.find(operation.getOperand(1));
        if (lhs == valueSlots.end() || rhs == valueSlots.end())
          return operation.emitError("operand has no allocated Udon heap slot");
        std::string result = valueSlots.lookup(operation.getResult(0));
        emitPush(operation.getLoc(), lhs->second);
        emitPush(operation.getLoc(), rhs->second);
        emitPush(operation.getLoc(), result);
        emitExtern(operation.getLoc(), signature);
      }
    }
    for (auto [jump, target] : relocations) {
      auto found = blockAddresses.find(target);
      if (found == blockAddresses.end())
        return jump->emitError("branch target is outside the lowered function");
      jump->setAttr("target", builder.getI64IntegerAttr(
                                  static_cast<int64_t>(found->second)));
    }
    return success();
  }

  void addSlot(StringRef name, StringRef type, StringRef initial,
               bool isPublic) {
    if (llvm::any_of(slots,
                     [&](const Slot &slot) { return slot.name == name; }))
      return;
    slots.push_back({name.str(), type.str(), initial.str(), isPublic});
  }

  ArrayAttr dataMetadata() {
    SmallVector<Attribute> attributes;
    for (const Slot &slot : slots) {
      SmallVector<NamedAttribute> fields;
      fields.push_back(
          builder.getNamedAttr("name", builder.getStringAttr(slot.name)));
      fields.push_back(
          builder.getNamedAttr("type", builder.getStringAttr(slot.type)));
      fields.push_back(
          builder.getNamedAttr("initial", builder.getStringAttr(slot.initial)));
      if (slot.isPublic)
        fields.push_back(builder.getNamedAttr("public", builder.getUnitAttr()));
      attributes.push_back(DictionaryAttr::get(module.getContext(), fields));
    }
    return ArrayAttr::get(module.getContext(), attributes);
  }

  Operation *create(Location location, StringRef name,
                    ArrayRef<NamedAttribute> attributes = {}) {
    OperationState state(location, name);
    state.addAttributes(attributes);
    return builder.create(state);
  }

  void emitPush(Location location, StringRef slot) {
    create(location, vm::PushOp::getOperationName(),
           {builder.getNamedAttr("slot", builder.getStringAttr(slot))});
    address += 8;
  }

  void emitExtern(Location location, StringRef signature) {
    create(
        location, vm::ExternOp::getOperationName(),
        {builder.getNamedAttr("signature", builder.getStringAttr(signature))});
    address += 8;
  }

  void emitCopy(Location location, StringRef source, StringRef destination) {
    emitPush(location, source);
    emitPush(location, destination);
    create(location, vm::CopyOp::getOperationName());
    address += 4;
  }

  Operation *emitJump(Location location, int64_t target) {
    Operation *operation = create(
        location, vm::JumpOp::getOperationName(),
        {builder.getNamedAttr("target", builder.getI64IntegerAttr(target))});
    address += 8;
    return operation;
  }

  ModuleOp module;
  OpBuilder builder;
  uint64_t address = 0;
  SmallVector<Slot> slots;
  SmallVector<Attribute> exports;
  llvm::DenseMap<Operation *, UdonSharpMethodABI> methodABIs;
  llvm::DenseMap<Operation *, FunctionPlan> functionPlans;
  SmallVector<std::pair<Operation *, func::FuncOp>> callRelocations;
  uint64_t callReturnIndex = 0;
};

struct LowerUdonHLToUdonAsmPass
    : impl::LowerUdonHLToUdonAsmPassBase<LowerUdonHLToUdonAsmPass> {
  void runOnOperation() override {
    if (failed(ScalarLowering(getOperation()).run()))
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
