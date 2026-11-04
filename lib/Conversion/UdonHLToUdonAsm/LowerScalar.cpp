#include "VRC/Conversion/Passes.h"
#include "VRC/ABI/Manifest/UdonSharpABI.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Analysis/Liveness.h"
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
#include "llvm/ADT/StringMap.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_LOWERUDONHLTOUDONASMPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace vm = ::vrc::udonasm;
namespace host = ::vrc::udonhl;
namespace runtime = ::vrc::vrc;

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

struct GlobalPlan {
  std::string slot;
  MemRefType type;
  ElementsAttr initialValue;
};

struct UnityBufferPoolEntry {
  int64_t id;
  std::string heapSlot;
};

static FailureOr<StringRef> udonType(Type type) {
  if (type.isInteger(1))
    return StringRef("SystemBoolean");
  if (type.isInteger(8) || type.isInteger(16))
    return StringRef("SystemInt32");
  if (type.isInteger(32))
    return StringRef("SystemInt32");
  if (type.isInteger(64))
    return StringRef("SystemInt64");
  if (type.isF32())
    return StringRef("SystemSingle");
  if (type.isIndex())
    return StringRef("SystemInt32");
  if (isa<::vrc::vrc::BufferType>(type))
    return StringRef("SystemInt32");
  if (isa<::vrc::vrc::EventType>(type))
    return StringRef("SystemInt32");
  if (isa<host::FrameType>(type))
    return StringRef("SystemInt32");
  if (isa<host::PtrType>(type))
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
  auto integerSignature = [&](StringRef name) -> StringRef {
    Type type = operation->getOperand(0).getType();
    StringRef systemType = type.isInteger(64) ? "SystemInt64" : "SystemInt32";
    static thread_local std::string signature;
    signature = systemType.str() + ".__op_" + name.str() + "__" +
                systemType.str() + "_" + systemType.str() + "__" +
                systemType.str();
    return signature;
  };
  if (isa<arith::AddIOp>(operation))
    return integerSignature("Addition");
  if (isa<arith::SubIOp>(operation))
    return integerSignature("Subtraction");
  if (isa<arith::MulIOp>(operation))
    return integerSignature("Multiplication");
  if (isa<arith::DivSIOp>(operation))
    return integerSignature("Division");
  if (isa<arith::RemSIOp>(operation))
    return integerSignature("Modulus");
  if (isa<arith::AndIOp>(operation))
    return integerSignature("LogicalAnd");
  if (isa<arith::OrIOp>(operation))
    return integerSignature("LogicalOr");
  if (isa<arith::XOrIOp>(operation))
    return integerSignature("LogicalXor");
  if (isa<arith::ShLIOp>(operation)) {
    Type type = operation->getOperand(0).getType();
    return type.isInteger(64)
               ? "SystemInt64.__op_LeftShift__SystemInt64_SystemInt32__"
                 "SystemInt64"
               : "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__"
                 "SystemInt32";
  }
  if (isa<arith::ShRSIOp>(operation)) {
    Type type = operation->getOperand(0).getType();
    return type.isInteger(64)
               ? "SystemInt64.__op_RightShift__SystemInt64_SystemInt32__"
                 "SystemInt64"
               : "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__"
                 "SystemInt32";
  }
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
  case arith::CmpIPredicate::ult:
    name = "LessThan";
    break;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    name = "LessThanOrEqual";
    break;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    name = "GreaterThan";
    break;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    name = "GreaterThanOrEqual";
    break;
  default:
    return {};
  }
  static thread_local std::string signature;
  StringRef systemType = operation.getLhs().getType().isInteger(64)
                             ? "SystemInt64"
                             : "SystemInt32";
  signature = systemType.str() + ".__op_" + name.str() + "__" +
              systemType.str() + "_" + systemType.str() + "__SystemBoolean";
  return signature;
}

static StringRef comparisonExtern(arith::CmpFOp operation) {
  StringRef name;
  switch (operation.getPredicate()) {
  case arith::CmpFPredicate::OEQ:
    name = "Equality";
    break;
  case arith::CmpFPredicate::UNE:
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
      : module(module), builder(module.getContext()),
        externalRawHeapInitialization(
            module->hasAttr("vrc.external_raw_heap_initialization")) {
    builder.setInsertionPointToEnd(module.getBody());
  }

  LogicalResult run() {
    SmallVector<func::FuncOp> functions;
    SmallVector<func::FuncOp> exportsToLower;
    SmallVector<func::FuncOp> internalsToLower;
    llvm::SmallSet<int64_t, 8> exportIds;
    llvm::SmallSet<std::string, 8> eventNames;
    llvm::SmallSet<std::string, 8> publicNames;
    if (failed(planUnityBufferPool()) || failed(planRawMemory()))
      return failure();
    if (externalRawHeapInitialization && rawHeapBytes) {
      SmallVector<int8_t> image(rawHeapBytes, 0);
      for (auto &entry : rawGlobalInitializers) {
        auto base = rawGlobalBases.find(entry.getKey());
        if (base == rawGlobalBases.end())
          return module.emitError("raw global initializer has no heap base");
        llvm::copy(entry.getValue().asArrayRef(), image.begin() + base->second);
      }
      module->setAttr("udonasm.raw_heap_initializer",
                      DenseI8ArrayAttr::get(module.getContext(), image));
    }
    unsigned globalIndex = 0;
    for (memref::GlobalOp global : module.getOps<memref::GlobalOp>()) {
      auto type = global.getType();
      auto initial = global.getInitialValue();
      auto elements =
          initial ? dyn_cast<ElementsAttr>(*initial) : ElementsAttr();
      if (type.getRank() != 1 || !type.hasStaticShape() ||
          type.getNumElements() <= 0 || type.getNumElements() > INT32_MAX ||
          (!type.getElementType().isInteger(32) &&
           !type.getElementType().isF32()) ||
          !elements ||
          elements.getNumElements() !=
              static_cast<uint64_t>(type.getNumElements()))
        return global.emitError(
            "Udon global requires an initialized fixed i32/f32 array");
      std::string slot = "__vrc_global_" + std::to_string(globalIndex++);
      addSlot(slot, *udonType(type), "null", false);
      globalPlans.try_emplace(global.getSymName(),
                              GlobalPlan{std::move(slot), type, elements});
      globalOrder.push_back(global.getSymName().str());
    }
    if (!globalPlans.empty() ||
        (rawHeapBytes && !externalRawHeapInitialization)) {
      addSlot(globalInitializedSlot, "SystemInt32", "0", false);
      addSlot(globalOneSlot, "SystemInt32", "1", false);
      addSlot(globalReadySlot, "SystemBoolean", "null", false);
    }
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
    for (memref::GlobalOp global :
         llvm::make_early_inc_range(module.getOps<memref::GlobalOp>()))
      global.erase();
    for (host::GlobalOp global :
         llvm::make_early_inc_range(module.getOps<host::GlobalOp>()))
      global.erase();
    module->setAttr("udonasm.data", dataMetadata());
    module->setAttr("udonasm.exports",
                    ArrayAttr::get(module.getContext(), exports));
    module->setAttr("udonasm.stage", builder.getStringAttr("scalar-v1"));
    module->removeAttr("vrc.host_stage");
    return success();
  }

private:
  LogicalResult planUnityBufferPool() {
    auto pool = module->getAttrOfType<ArrayAttr>("vrc.unity_buffer_pool");
    if (!pool)
      return success();
    llvm::SmallSet<int64_t, 8> ids;
    llvm::SmallSet<std::string, 8> slots;
    for (Attribute attribute : pool) {
      auto entry = dyn_cast<DictionaryAttr>(attribute);
      auto id = entry ? entry.getAs<IntegerAttr>("id") : IntegerAttr();
      auto heapSlot =
          entry ? entry.getAs<StringAttr>("heap_slot") : StringAttr();
      if (!id || id.getInt() <= 0 || !heapSlot || heapSlot.getValue().empty() ||
          !ids.insert(id.getInt()).second ||
          !slots.insert(heapSlot.getValue().str()).second)
        return module.emitError("invalid Unity buffer pool metadata");
      unityBufferPool.push_back({id.getInt(), heapSlot.getValue().str()});
      addSlot(heapSlot.getValue(), "UnityEngineRenderTexture", "null", true);
    }
    return success();
  }

  LogicalResult planRawMemory() {
    uint64_t cursor = 4; // Keep the null pointer unmapped.
    for (host::GlobalOp global : module.getOps<host::GlobalOp>()) {
      if (global.getAddressSpace() != 0 || global.getLinkage() != "internal")
        return global.emitError(
            "raw Udon globals currently require internal address-space zero");
      Type type = global.getStorageType();
      auto bytes =
          dyn_cast_or_null<DenseI8ArrayAttr>(global.getInitialValueAttr());
      bool scalar =
          type.isInteger(32) || type.isF32() || isa<host::PtrType>(type);
      if (!scalar && !bytes)
        return global.emitError(
            "aggregate raw Udon globals require a serialized byte initializer");
      uint64_t alignment = static_cast<uint64_t>(global.getAlignment());
      cursor = llvm::alignTo(cursor, alignment);
      rawGlobalBases.try_emplace(global.getSymName(), cursor);
      cursor += bytes ? bytes.size() : 4;
      if (bytes)
        rawGlobalInitializers.try_emplace(global.getSymName(), bytes);
      Attribute initial = global.getInitialValueAttr();
      bool zero = !initial || isa<UnitAttr, DenseI8ArrayAttr>(initial);
      if (auto integer = dyn_cast_or_null<IntegerAttr>(initial))
        zero = integer.getValue().isZero();
      if (auto floating = dyn_cast_or_null<FloatAttr>(initial))
        zero = floating.getValue().isZero();
      if (!zero)
        return global.emitError(
            "nonzero raw global initialization is not implemented yet");
    }
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      WalkResult result = function.walk([&](host::AllocaOp allocation) {
        auto constant =
            allocation.getByteSize().getDefiningOp<arith::ConstantOp>();
        auto bytes = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                              : IntegerAttr();
        if (!bytes || bytes.getInt() <= 0 || bytes.getInt() > INT32_MAX) {
          allocation.emitError(
              "raw Udon alloca requires a positive constant byte size");
          return WalkResult::interrupt();
        }
        uint64_t alignment = static_cast<uint64_t>(allocation.getAlignment());
        cursor = llvm::alignTo(cursor, alignment);
        rawAllocaBases.try_emplace(allocation.getOperation(), cursor);
        cursor += static_cast<uint64_t>(bytes.getInt());
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return failure();
    }
    if (cursor > 4) {
      if (cursor > INT32_MAX)
        return module.emitError("raw Udon heap exceeds signed i32 capacity");
      rawHeapBytes = cursor;
      addSlot(rawHeapSlot, "SystemInt32Array", "null",
              externalRawHeapInitialization);
      addSlot(rawHeapLengthSlot, "SystemInt32", std::to_string(rawHeapBytes),
              false);
    }
    return success();
  }

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
    llvm::DenseMap<Value, unsigned> valueColors;
    llvm::DenseMap<Value, unsigned> valueOrdinals;
    llvm::DenseMap<Value, SmallVector<Value>> interference;
    llvm::DenseMap<Value, std::string> valueUdonTypes;
    llvm::DenseMap<Value, unsigned> descriptorColors;
    llvm::DenseMap<Value, unsigned> descriptorOrdinals;
    llvm::DenseMap<Value, SmallVector<Value>> descriptorInterference;
    SmallVector<Value> colorableValues;
    SmallVector<Value> descriptors;
    auto isColorableResult = [&](Operation &operation) {
      return !isa<arith::ConstantOp, memref::GetGlobalOp,
                  ::vrc::vrc::BufferAllocOp, ::vrc::vrc::DispatchDescCreateOp,
                  host::FrameCreateOp>(operation);
    };
    for (Block &block : function.getBody()) {
      if (&block != &function.getBody().front())
        for (BlockArgument argument : block.getArguments())
          colorableValues.push_back(argument);
      for (Operation &operation : block)
        if (auto descriptor =
                dyn_cast<::vrc::vrc::DispatchDescCreateOp>(operation))
          descriptors.push_back(descriptor.getResult());
        else if (isColorableResult(operation))
          llvm::append_range(colorableValues, operation.getResults());
    }
    for (auto [ordinal, value] : llvm::enumerate(colorableValues))
      valueOrdinals.try_emplace(value, ordinal);
    for (Value value : colorableValues) {
      auto type = udonType(value.getType());
      if (succeeded(type))
        valueUdonTypes.try_emplace(value, type->str());
    }
    for (auto [ordinal, descriptor] : llvm::enumerate(descriptors))
      descriptorOrdinals.try_emplace(descriptor, ordinal);

    Liveness liveness(function);
    function.walk([&](Operation *operation) {
      const LivenessBlockInfo *blockInfo =
          liveness.getLiveness(operation->getBlock());
      if (!blockInfo)
        return;
      SmallVector<Value> live;
      for (Value value : blockInfo->currentlyLiveValues(operation))
        if (valueOrdinals.contains(value))
          live.push_back(value);
      llvm::sort(live, [&](Value left, Value right) {
        return valueOrdinals.lookup(left) < valueOrdinals.lookup(right);
      });
      live.erase(std::unique(live.begin(), live.end()), live.end());
      for (auto [leftIndex, left] : llvm::enumerate(live))
        for (Value right : ArrayRef<Value>(live).drop_front(leftIndex + 1))
          if (valueUdonTypes.lookup(left) == valueUdonTypes.lookup(right)) {
            interference[left].push_back(right);
            interference[right].push_back(left);
          }

      SmallVector<Value> liveDescriptors;
      for (Value value : blockInfo->currentlyLiveValues(operation))
        if (descriptorOrdinals.contains(value))
          liveDescriptors.push_back(value);
      for (Value operand : operation->getOperands())
        if (descriptorOrdinals.contains(operand))
          liveDescriptors.push_back(operand);
      llvm::sort(liveDescriptors, [&](Value left, Value right) {
        return descriptorOrdinals.lookup(left) <
               descriptorOrdinals.lookup(right);
      });
      liveDescriptors.erase(
          std::unique(liveDescriptors.begin(), liveDescriptors.end()),
          liveDescriptors.end());
      for (auto [leftIndex, left] : llvm::enumerate(liveDescriptors))
        for (Value right :
             ArrayRef<Value>(liveDescriptors).drop_front(leftIndex + 1)) {
          descriptorInterference[left].push_back(right);
          descriptorInterference[right].push_back(left);
        }
    });
    for (auto &entry : interference) {
      SmallVector<Value> &neighbors = entry.second;
      llvm::sort(neighbors, [&](Value left, Value right) {
        return valueOrdinals.lookup(left) < valueOrdinals.lookup(right);
      });
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                      neighbors.end());
    }
    for (auto &entry : descriptorInterference) {
      SmallVector<Value> &neighbors = entry.second;
      llvm::sort(neighbors, [&](Value left, Value right) {
        return descriptorOrdinals.lookup(left) <
               descriptorOrdinals.lookup(right);
      });
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                      neighbors.end());
    }
    SmallVector<Value> coloringOrder(colorableValues);
    llvm::sort(coloringOrder, [&](Value left, Value right) {
      size_t leftDegree = interference[left].size();
      size_t rightDegree = interference[right].size();
      if (leftDegree != rightDegree)
        return leftDegree > rightDegree;
      return valueOrdinals.lookup(left) < valueOrdinals.lookup(right);
    });
    for (Value value : coloringOrder) {
      llvm::SmallSet<unsigned, 16> used;
      for (Value neighbor : interference[value]) {
        auto found = valueColors.find(neighbor);
        if (found != valueColors.end())
          used.insert(found->second);
      }
      unsigned color = 0;
      while (used.contains(color))
        ++color;
      valueColors.try_emplace(value, color);
    }
    SmallVector<Value> descriptorColoringOrder(descriptors);
    llvm::sort(descriptorColoringOrder, [&](Value left, Value right) {
      size_t leftDegree = descriptorInterference[left].size();
      size_t rightDegree = descriptorInterference[right].size();
      if (leftDegree != rightDegree)
        return leftDegree > rightDegree;
      return descriptorOrdinals.lookup(left) < descriptorOrdinals.lookup(right);
    });
    for (Value descriptor : descriptorColoringOrder) {
      llvm::SmallSet<unsigned, 16> used;
      for (Value neighbor : descriptorInterference[descriptor]) {
        auto found = descriptorColors.find(neighbor);
        if (found != descriptorColors.end())
          used.insert(found->second);
      }
      unsigned color = 0;
      while (used.contains(color))
        ++color;
      descriptorColors.try_emplace(descriptor, color);
    }
    auto coloredSlot = [&](Value value) -> std::string {
      auto type = udonType(value.getType());
      if (failed(type))
        return {};
      std::string name = prefix + "r_" + type->str() + "_" +
                         std::to_string(valueColors.lookup(value));
      addSlot(name, *type, "null", false);
      return name;
    };
    llvm::DenseMap<Operation *, SmallVector<std::string>> frameSlots;
    llvm::DenseMap<Operation *, SmallVector<std::string>> descriptorSlots;
    llvm::DenseMap<Operation *, std::string> allocaLengthSlots;
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

    SmallVector<runtime::AwaitOp> awaits;
    function.walk([&](runtime::AwaitOp await) { awaits.push_back(await); });
    if (!awaits.empty() && !exportId)
      return function.emitError()
             << "suspending helper " << function.getSymName()
             << " must be inlined into a Udon export";
    llvm::DenseMap<Operation *, int32_t> awaitStates;
    llvm::DenseMap<Operation *, Operation *> awaitResumeJumps;
    if (!awaits.empty()) {
      addSlot(asyncResumeStateSlot, "SystemInt32", "0", true);
      addSlot(asyncCompletionStatusSlot, "SystemInt32", "0", true);
      addSlot(asyncCommandKindSlot, "SystemInt32", "0", true);
      addSlot(asyncCommandBufferSlot, "SystemInt32", "0", true);
      addSlot(asyncCommandOffsetSlot, "SystemInt32", "0", true);
      addSlot(asyncCommandSizeSlot, "SystemInt32", "0", true);
      addSlot(asyncCommandPointerSlot, "SystemInt32", "0", true);
      addSlot(asyncEventSlot, "SystemInt32", "1", true);
      addSlot(asyncZeroSlot, "SystemInt32", "0", false);
      std::string stateSlot = prefix + "await_state_value";
      addSlot(stateSlot, "SystemInt32", "0", false);
      for (auto [index, await] : llvm::enumerate(awaits)) {
        int32_t state = static_cast<int32_t>(index + 1);
        awaitStates.try_emplace(await.getOperation(), state);
        constexpr StringLiteral matches = "__vrc_await_state_matches";
        addSlot(matches, "SystemBoolean", "null", false);
        emitI32Literal(function.getLoc(), static_cast<uint32_t>(state),
                       stateSlot);
        emitPush(function.getLoc(), asyncResumeStateSlot);
        emitPush(function.getLoc(), stateSlot);
        emitPush(function.getLoc(), matches);
        emitExtern(function.getLoc(),
                   "SystemInt32.__op_Equality__SystemInt32_SystemInt32__"
                   "SystemBoolean");
        emitPush(function.getLoc(), matches);
        Operation *next = create(
            function.getLoc(), vm::JumpIfFalseOp::getOperationName(),
            {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
        address += 8;
        emitCopy(function.getLoc(), asyncZeroSlot, asyncResumeStateSlot);
        awaitResumeJumps.try_emplace(await.getOperation(),
                                     emitJump(function.getLoc(), 0));
        next->setAttr("target",
                      builder.getI64IntegerAttr(static_cast<int64_t>(address)));
      }
    }

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
          std::string name = coloredSlot(argument);
          valueSlots[argument] = name;
        }
      for (Operation &operation : block) {
        if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(operation)) {
          auto found = globalPlans.find(getGlobal.getName());
          if (found == globalPlans.end() ||
              found->second.type != getGlobal.getType())
            return getGlobal.emitError(
                "Udon global reference does not match a defined array");
          valueSlots[getGlobal.getResult()] = found->second.slot;
          continue;
        }
        if (auto alloca = dyn_cast<memref::AllocaOp>(operation)) {
          auto type = alloca.getType();
          if (type.getRank() != 1 || !type.hasStaticShape() ||
              type.getNumElements() <= 0 || type.getNumElements() > INT32_MAX ||
              (!type.getElementType().isInteger(32) &&
               !type.getElementType().isF32()) ||
              !alloca.getDynamicSizes().empty() ||
              !alloca.getSymbolOperands().empty())
            return alloca.emitError(
                "Udon local array requires a fixed non-empty i32/f32 shape");
          std::string name = coloredSlot(alloca.getResult());
          valueSlots[alloca.getResult()] = name;
          std::string length = name + "_length";
          addSlot(length, "SystemInt32", std::to_string(type.getNumElements()),
                  false);
          allocaLengthSlots[alloca.getOperation()] = std::move(length);
          continue;
        }
        if (auto allocation = dyn_cast<host::AllocaOp>(operation)) {
          auto found = rawAllocaBases.find(allocation.getOperation());
          if (found == rawAllocaBases.end())
            return allocation.emitError("raw alloca has no static heap plan");
          std::string name = coloredSlot(allocation.getResult());
          valueSlots[allocation.getResult()] = name;
          continue;
        }
        if (auto addressOf = dyn_cast<host::AddressOfOp>(operation)) {
          auto found = rawGlobalBases.find(addressOf.getGlobal());
          if (found == rawGlobalBases.end())
            return addressOf.emitError("raw global has no static heap plan");
          std::string name = coloredSlot(addressOf.getResult());
          valueSlots[addressOf.getResult()] = name;
          continue;
        }
        if (!operation.getNumResults())
          continue;
        if (auto descriptor =
                dyn_cast<::vrc::vrc::DispatchDescCreateOp>(operation)) {
          SmallVector<std::string> snapshots;
          unsigned color = descriptorColors.lookup(descriptor.getResult());
          for (auto [index, scalar] :
               llvm::enumerate(descriptor.getScalars())) {
            auto type = udonType(scalar.getType());
            if (failed(type))
              return descriptor.emitError(
                  "dispatch scalar has unsupported Udon type");
            std::string name = prefix + "descriptor_" + std::to_string(color) +
                               "_" + std::to_string(index) + "_" + type->str();
            addSlot(name, *type, "null", false);
            snapshots.push_back(std::move(name));
          }
          descriptorSlots.try_emplace(descriptor.getOperation(),
                                      std::move(snapshots));
          continue;
        }
        if (auto cast = dyn_cast<UnrealizedConversionCastOp>(operation)) {
          if (cast.getInputs().size() != 1 || cast.getOutputs().size() != 1)
            return cast.emitError(
                "Udon token conversion requires one input and one output");
          std::string source = valueSlots.lookup(cast.getInputs().front());
          if (source.empty())
            return cast.emitError("Udon token conversion has no source slot");
          Type outputType = cast.getOutputs().front().getType();
          if (isa<::vrc::vrc::BufferType, ::vrc::vrc::EventType>(outputType)) {
            std::string token = coloredSlot(cast.getOutputs().front());
            if (token.empty())
              return cast.emitError(
                  "Udon token conversion has no colorable output slot");
            valueSlots[cast.getOutputs().front()] = token;
          } else {
            valueSlots[cast.getOutputs().front()] = source;
          }
          continue;
        }
        if (auto allocation = dyn_cast<::vrc::vrc::BufferAllocOp>(operation)) {
          std::string logicalId = valueSlots.lookup(allocation.getLogicalId());
          if (logicalId.empty())
            return allocation.emitError(
                "Unity buffer allocation requires a logical-id slot");
          std::string token = prefix + "buffer_" + std::to_string(valueIndex++);
          addSlot(token, "SystemInt32", "0", false);
          valueSlots[allocation.getBuffer()] = token;
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
        std::string constantIdentity;
        if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
          kind = "c";
          llvm::raw_string_ostream identityStream(constantIdentity);
          constant.getValue().print(identityStream);
          identityStream.flush();
          if (constant.getType().isInteger(1)) {
            initial = "null";
          } else if (auto integer =
                         dyn_cast<IntegerAttr>(constant.getValue())) {
            if (constant.getType().isInteger(64)) {
              initial = "null";
            } else {
              initial = std::to_string(integer.getInt());
            }
          } else if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) {
            llvm::SmallString<32> text;
            floating.getValue().toString(text);
            initial = text.str().str();
            // The Worlds SDK assembler does not accept APFloat's positive
            // scientific exponent spelling (E+NN or ENN). Reformat only
            // those large finite values with enough padding to force an
            // equivalent fixed-decimal literal.
            size_t exponent = initial.find_first_of("eE");
            if (exponent != std::string::npos &&
                exponent + 1 < initial.size() && initial[exponent + 1] == '+') {
              text.clear();
              floating.getValue().toString(text, 50, 400);
              initial = text.str().str();
              if (initial.find('.') == std::string::npos)
                initial += ".0";
            }
          } else
            return constant.emitError("unsupported scalar constant attribute");
        }
        std::string name;
        if (kind == "c") {
          std::string key = type->str() + "\n" + constantIdentity;
          // Every occurrence with this type and value materializes the same
          // immutable value. i1/i64 need runtime construction because the SDK
          // assembler cannot encode all literals directly, but writing the
          // same module-wide slot is still safe across helpers and exports.
          auto found = sharedConstantSlots.find(key);
          if (found == sharedConstantSlots.end()) {
            name = "__vrc_constant_" + std::to_string(sharedConstantIndex++);
            addSlot(name, *type, initial, false);
            sharedConstantSlots.try_emplace(key, name);
          } else {
            name = found->second;
          }
        } else {
          name = coloredSlot(operation.getResult(0));
        }
        valueSlots[operation.getResult(0)] = name;
      }
    }

    if (exportId && (!globalPlans.empty() ||
                     (rawHeapBytes && !externalRawHeapInitialization)))
      emitGlobalInitialization(function.getLoc());

    llvm::DenseMap<Block *, uint64_t> blockAddresses;
    SmallVector<std::pair<Operation *, Block *>> relocations;
    auto copies = [&](Location location, ValueRange sources,
                      Block *destination) -> LogicalResult {
      if (sources.size() != destination->getNumArguments())
        return emitError(location)
               << "branch operand count does not match destination";
      struct Move {
        std::string source;
        std::string destination;
        std::string type;
      };
      SmallVector<Move> moves;
      for (auto [source, target] :
           llvm::zip(sources, destination->getArguments())) {
        std::string sourceSlot = valueSlots.lookup(source);
        std::string targetSlot = valueSlots.lookup(target);
        if (sourceSlot.empty() || targetSlot.empty())
          return emitError(location)
                 << "branch value has no allocated Udon heap slot";
        if (sourceSlot == targetSlot)
          continue;
        auto type = udonType(target.getType());
        if (failed(type))
          return emitError(location) << "branch target has unsupported type";
        moves.push_back(
            {std::move(sourceSlot), std::move(targetSlot), type->str()});
      }
      while (!moves.empty()) {
        auto safe = llvm::find_if(moves, [&](const Move &move) {
          return llvm::none_of(moves, [&](const Move &candidate) {
            return candidate.source == move.destination;
          });
        });
        if (safe != moves.end()) {
          emitCopy(location, safe->source, safe->destination);
          moves.erase(safe);
          continue;
        }
        Move &cycle = moves.front();
        std::string scratch = "__vrc_parallel_copy_scratch_" + cycle.type;
        addSlot(scratch, cycle.type, "null", false);
        emitCopy(location, cycle.source, scratch);
        std::string captured = cycle.source;
        for (Move &move : moves)
          if (move.source == captured)
            move.source = scratch;
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
        if (isa<memref::GetGlobalOp>(operation))
          continue;
        if (auto alloca = dyn_cast<memref::AllocaOp>(operation)) {
          std::string length = allocaLengthSlots.lookup(&operation);
          std::string result = valueSlots.lookup(alloca.getResult());
          if (length.empty() || result.empty())
            return alloca.emitError(
                "Udon local array has no allocated heap slots");
          emitPush(alloca.getLoc(), length);
          emitPush(alloca.getLoc(), result);
          emitExtern(
              alloca.getLoc(),
              alloca.getType().getElementType().isF32()
                  ? "SystemSingleArray.__ctor__SystemInt32__SystemSingleArray"
                  : "SystemInt32Array.__ctor__SystemInt32__SystemInt32Array");
          continue;
        }
        if (auto allocation = dyn_cast<host::AllocaOp>(operation)) {
          auto found = rawAllocaBases.find(allocation.getOperation());
          if (found == rawAllocaBases.end())
            return allocation.emitError("raw alloca has no static heap plan");
          emitI32Literal(allocation.getLoc(), found->second,
                         valueSlots.lookup(allocation.getResult()));
          continue;
        }
        if (auto addressOf = dyn_cast<host::AddressOfOp>(operation)) {
          auto found = rawGlobalBases.find(addressOf.getGlobal());
          if (found == rawGlobalBases.end())
            return addressOf.emitError("raw global has no static heap plan");
          emitI32Literal(addressOf.getLoc(), found->second,
                         valueSlots.lookup(addressOf.getResult()));
          continue;
        }
        if (auto cast = dyn_cast<UnrealizedConversionCastOp>(operation)) {
          Type outputType = cast.getOutputs().front().getType();
          if (isa<::vrc::vrc::BufferType, ::vrc::vrc::EventType>(outputType)) {
            std::string source = valueSlots.lookup(cast.getInputs().front());
            std::string token = valueSlots.lookup(cast.getOutputs().front());
            if (source.empty() || token.empty())
              return cast.emitError("Udon token conversion has missing slots");
            emitCopy(cast.getLoc(), source, token);
          }
          continue;
        }
        if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
          if (constant.getType().isInteger(1) &&
              cast<IntegerAttr>(constant.getValue()).getInt()) {
            std::string one = valueSlots.lookup(constant.getResult()) + "_one";
            addSlot(one, "SystemInt32", "1", false);
            emitPush(constant.getLoc(), one);
            emitPush(constant.getLoc(), one);
            emitPush(constant.getLoc(),
                     valueSlots.lookup(constant.getResult()));
            emitExtern(constant.getLoc(),
                       "SystemInt32.__op_Equality__SystemInt32_SystemInt32__"
                       "SystemBoolean");
          } else if (constant.getType().isInteger(64)) {
            auto integer = cast<IntegerAttr>(constant.getValue());
            emitI64Constant(constant.getLoc(), integer.getValue(),
                            valueSlots.lookup(constant.getResult()));
          }
          continue;
        }
        if (auto descriptor =
                dyn_cast<::vrc::vrc::DispatchDescCreateOp>(operation)) {
          auto found = descriptorSlots.find(descriptor.getOperation());
          if (found == descriptorSlots.end() ||
              found->second.size() != descriptor.getScalars().size())
            return descriptor.emitError(
                "dispatch descriptor has no scalar snapshot slots");
          for (auto [scalar, snapshot] :
               llvm::zip(descriptor.getScalars(), found->second)) {
            std::string source = valueSlots.lookup(scalar);
            if (source.empty())
              return descriptor.emitError(
                  "dispatch scalar has no allocated Udon heap slot");
            emitCopy(descriptor.getLoc(), source, snapshot);
          }
          continue;
        }
        if (auto allocation = dyn_cast<::vrc::vrc::BufferAllocOp>(operation)) {
          std::string logicalId = valueSlots.lookup(allocation.getLogicalId());
          std::string token = valueSlots.lookup(allocation.getBuffer());
          if (logicalId.empty() || token.empty())
            return allocation.emitError(
                "Unity buffer allocation has missing handle slots");
          emitCopy(allocation.getLoc(), logicalId, token);
          continue;
        }
        if (isa<::vrc::vrc::BufferReleaseOp>(operation))
          // Unity buffers belong to the fixed physical pool. The C runtime
          // has already released its logical cl_mem reference; retaining the
          // RenderTexture backing and the result slot's initial zero models a
          // successful backend release without destroying the pool asset.
          continue;
        if (auto transfer = dyn_cast<runtime::BufferUploadOp>(operation)) {
          std::string buffer = valueSlots.lookup(transfer.getDestination());
          std::string offset =
              valueSlots.lookup(transfer.getDestinationOffset());
          std::string size = valueSlots.lookup(transfer.getByteSize());
          std::string pointer = valueSlots.lookup(transfer.getHostPointer());
          std::string event = valueSlots.lookup(transfer.getEvent());
          if (buffer.empty() || offset.empty() || size.empty() ||
              pointer.empty() || event.empty())
            return transfer.emitError(
                "Unity buffer upload has missing command slots");
          std::string kind = prefix + "upload_command";
          addSlot(kind, "SystemInt32", "1", false);
          emitCopy(transfer.getLoc(), buffer, asyncCommandBufferSlot);
          emitCopy(transfer.getLoc(), offset, asyncCommandOffsetSlot);
          emitCopy(transfer.getLoc(), size, asyncCommandSizeSlot);
          emitCopy(transfer.getLoc(), pointer, asyncCommandPointerSlot);
          emitCopy(transfer.getLoc(), kind, asyncCommandKindSlot);
          emitCopy(transfer.getLoc(), asyncEventSlot, event);
          continue;
        }
        if (auto transfer = dyn_cast<runtime::BufferReadbackOp>(operation)) {
          std::string buffer = valueSlots.lookup(transfer.getSource());
          std::string offset = valueSlots.lookup(transfer.getSourceOffset());
          std::string size = valueSlots.lookup(transfer.getByteSize());
          std::string pointer = valueSlots.lookup(transfer.getHostPointer());
          std::string event = valueSlots.lookup(transfer.getEvent());
          if (buffer.empty() || offset.empty() || size.empty() ||
              pointer.empty() || event.empty())
            return transfer.emitError(
                "Unity buffer readback has missing command slots");
          std::string kind = prefix + "readback_command";
          addSlot(kind, "SystemInt32", "2", false);
          emitCopy(transfer.getLoc(), buffer, asyncCommandBufferSlot);
          emitCopy(transfer.getLoc(), offset, asyncCommandOffsetSlot);
          emitCopy(transfer.getLoc(), size, asyncCommandSizeSlot);
          emitCopy(transfer.getLoc(), pointer, asyncCommandPointerSlot);
          emitCopy(transfer.getLoc(), kind, asyncCommandKindSlot);
          emitCopy(transfer.getLoc(), asyncEventSlot, event);
          continue;
        }
        if (auto await = dyn_cast<runtime::AwaitOp>(operation)) {
          auto state = awaitStates.find(await.getOperation());
          auto resumeJump = awaitResumeJumps.find(await.getOperation());
          std::string status = valueSlots.lookup(await.getStatus());
          if (state == awaitStates.end() ||
              resumeJump == awaitResumeJumps.end() || status.empty())
            return await.emitError("await has no static resume state");
          std::string stateSlot = prefix + "await_state_value";
          emitI32Literal(await.getLoc(), static_cast<uint32_t>(state->second),
                         stateSlot);
          emitCopy(await.getLoc(), stateSlot, asyncResumeStateSlot);
          emitJump(await.getLoc(), 0xfffffffcLL);
          resumeJump->second->setAttr(
              "target",
              builder.getI64IntegerAttr(static_cast<int64_t>(address)));
          emitCopy(await.getLoc(), asyncCompletionStatusSlot, status);
          continue;
        }
        if (auto status = dyn_cast<runtime::EventStatusOp>(operation)) {
          std::string result = valueSlots.lookup(status.getStatus());
          if (result.empty())
            return status.emitError("event status has no result slot");
          emitCopy(status.getLoc(), asyncCompletionStatusSlot, result);
          continue;
        }
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
          auto resourceHeapSlots = dispatch->getAttrOfType<ArrayAttr>(
              "vrc.unity_resource_heap_slots");
          int64_t output = outputIndex ? outputIndex.getInt() : 0;
          std::string destinationToken =
              output >= 0 && size_t(output) < dispatch.getResources().size()
                  ? valueSlots.lookup(dispatch.getResources()[output])
                  : std::string();
          if (!descriptor || !materialSlot || materialSlot.getValue().empty() ||
              destinationToken.empty())
            return dispatch.emitError(
                "Unity dispatch requires a bound output, a copied scalar "
                "descriptor and a kernel material slot");
          auto physicalBuffer = [&](size_t index, StringRef token,
                                    StringRef dynamicPrefix) {
            if (resourceHeapSlots && index < resourceHeapSlots.size())
              if (auto slot = dyn_cast<StringAttr>(resourceHeapSlots[index]);
                  slot && !slot.getValue().empty())
                return slot.getValue().str();
            return selectUnityBuffer(dispatch.getLoc(), token, dynamicPrefix);
          };
          std::string destination = physicalBuffer(
              static_cast<size_t>(output), destinationToken,
              prefix + "dispatch_" + std::to_string(dispatch.getKernelId()) +
                  "_output");
          if (destination.empty())
            return dispatch.emitError("Unity dispatch has no physical pool");
          addSlot(materialSlot.getValue(), "UnityEngineMaterial", "null", true);
          unsigned scalarIndex = 0, bufferIndex = 0;
          auto snapshots = descriptorSlots.find(descriptor.getOperation());
          if (snapshots == descriptorSlots.end())
            return dispatch.emitError(
                "Unity dispatch descriptor has no scalar snapshots");
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
              texture =
                  physicalBuffer(bufferIndex - 1, texture,
                                 prefix + "dispatch_" +
                                     std::to_string(dispatch.getKernelId()) +
                                     "_input_" + std::to_string(argumentIndex));
              if (texture.empty())
                return dispatch.emitError(
                    "Unity dispatch has no physical pool");
              std::string propertySlot =
                  "__udoncl_material_input_" + std::to_string(argumentIndex);
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
            if (scalarIndex >= snapshots->second.size())
              return dispatch.emitError("dispatch descriptor is incomplete");
            std::string scalar = snapshots->second[scalarIndex++];
            if (scalar.empty())
              return dispatch.emitError(
                  "dispatch scalar has no allocated Udon heap slot");
            std::string propertySlot =
                "__udoncl_material_arg_" + std::to_string(argumentIndex);
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
          std::string event = valueSlots.lookup(dispatch.getEvent());
          if (event.empty())
            return dispatch.emitError("Unity dispatch has no event slot");
          addSlot("__udoncl_completed_event", "SystemInt32", "0", false);
          emitCopy(dispatch.getLoc(), "__udoncl_completed_event", event);
          continue;
        }
        if (auto copy = dyn_cast<::vrc::vrc::BufferCopyOp>(operation)) {
          auto materialSlot =
              copy->getAttrOfType<StringAttr>("vrc.unity_copy_material_slot");
          std::string sourceToken = valueSlots.lookup(copy.getSource());
          std::string destinationToken =
              valueSlots.lookup(copy.getDestination());
          if (!materialSlot || materialSlot.getValue().empty() ||
              sourceToken.empty() || destinationToken.empty())
            return copy.emitError(
                "Unity buffer copy requires distinct bound buffers and a copy "
                "material slot");
          std::string source = selectUnityBuffer(
              copy.getLoc(), sourceToken,
              prefix + "copy_source_" + std::to_string(valueIndex++));
          std::string destination = selectUnityBuffer(
              copy.getLoc(), destinationToken,
              prefix + "copy_destination_" + std::to_string(valueIndex++));
          if (source.empty() || destination.empty())
            return copy.emitError("Unity copy has no physical buffer pool");
          addSlot(materialSlot.getValue(), "UnityEngineMaterial", "null", true);
          emitPush(copy.getLoc(), source);
          emitPush(copy.getLoc(), destination);
          emitPush(copy.getLoc(), materialSlot.getValue());
          emitExtern(
              copy.getLoc(),
              "VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_"
              "UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid");
          std::string event = valueSlots.lookup(copy.getEvent());
          if (event.empty())
            return copy.emitError("Unity buffer copy has no event slot");
          addSlot("__udoncl_completed_event", "SystemInt32", "0", false);
          emitCopy(copy.getLoc(), "__udoncl_completed_event", event);
          continue;
        }
        if (auto pointer = dyn_cast<host::PtrAddOp>(operation)) {
          std::string base = valueSlots.lookup(pointer.getPointer());
          std::string offset = valueSlots.lookup(pointer.getByteOffset());
          std::string result = valueSlots.lookup(pointer.getResult());
          if (base.empty() || offset.empty() || result.empty())
            return pointer.emitError("pointer addition has missing heap slots");
          emitPush(pointer.getLoc(), base);
          emitPush(pointer.getLoc(), offset);
          emitPush(pointer.getLoc(), result);
          emitExtern(pointer.getLoc(),
                     "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                     "SystemInt32");
          continue;
        }
        if (auto cast = dyn_cast<host::PtrToIntOp>(operation)) {
          emitCopy(cast.getLoc(), valueSlots.lookup(cast.getPointer()),
                   valueSlots.lookup(cast.getResult()));
          continue;
        }
        if (auto cast = dyn_cast<host::IntToPtrOp>(operation)) {
          emitCopy(cast.getLoc(), valueSlots.lookup(cast.getAddress()),
                   valueSlots.lookup(cast.getResult()));
          continue;
        }
        if (auto load = dyn_cast<host::LoadOp>(operation)) {
          std::string pointer = valueSlots.lookup(load.getPointer());
          std::string result = valueSlots.lookup(load.getResult());
          if (pointer.empty() || result.empty() ||
              failed(emitRawLoad(load.getLoc(), pointer, result,
                                 load.getResult().getType())))
            return load.emitError("unsupported raw Udon load");
          continue;
        }
        if (auto store = dyn_cast<host::StoreOp>(operation)) {
          std::string pointer = valueSlots.lookup(store.getPointer());
          std::string value = valueSlots.lookup(store.getValue());
          if (pointer.empty() || value.empty() ||
              failed(emitRawStore(store.getLoc(), pointer, value,
                                  store.getValue().getType())))
            return store.emitError("unsupported raw Udon store");
          continue;
        }
        if (auto memcpy = dyn_cast<host::MemcpyOp>(operation)) {
          std::string destination = valueSlots.lookup(memcpy.getDestination());
          std::string source = valueSlots.lookup(memcpy.getSource());
          std::string byteSize = valueSlots.lookup(memcpy.getByteSize());
          if (destination.empty() || source.empty() || byteSize.empty())
            return memcpy.emitError("memcpy pointer has no Udon heap slot");
          emitRawMemcpy(memcpy.getLoc(), destination, source, byteSize);
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
          // Two eight-byte PUSH instructions plus four-byte COPY store the
          // return address, then an eight-byte JUMP enters the callee.
          addSlot(returnAddress, "SystemUInt32", std::to_string(address + 28),
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
        if (auto cast = dyn_cast<arith::ExtSIOp>(operation)) {
          std::string source = valueSlots.lookup(cast.getIn());
          std::string result = valueSlots.lookup(cast.getOut());
          if (source.empty() || result.empty())
            return cast.emitError("integer extension has missing heap slots");
          if (cast.getIn().getType().isInteger(8) &&
              cast.getOut().getType().isInteger(32)) {
            constexpr StringLiteral threshold =
                "__vrc_i8_sign_extend_threshold";
            constexpr StringLiteral modulus = "__vrc_i8_sign_extend_modulus";
            constexpr StringLiteral negative = "__vrc_i8_sign_extend_negative";
            constexpr StringLiteral isNegative =
                "__vrc_i8_sign_extend_is_negative";
            addSlot(threshold, "SystemInt32", "128", false);
            addSlot(modulus, "SystemInt32", "256", false);
            addSlot(negative, "SystemInt32", "0", false);
            addSlot(isNegative, "SystemBoolean", "null", false);
            emitPush(cast.getLoc(), source);
            emitPush(cast.getLoc(), threshold);
            emitPush(cast.getLoc(), isNegative);
            emitExtern(cast.getLoc(),
                       "SystemInt32.__op_GreaterThanOrEqual__SystemInt32_"
                       "SystemInt32__SystemBoolean");
            emitPush(cast.getLoc(), isNegative);
            Operation *positive = create(
                cast.getLoc(), vm::JumpIfFalseOp::getOperationName(),
                {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
            address += 8;
            emitPush(cast.getLoc(), source);
            emitPush(cast.getLoc(), modulus);
            emitPush(cast.getLoc(), negative);
            emitExtern(cast.getLoc(),
                       "SystemInt32.__op_Subtraction__SystemInt32_SystemInt32__"
                       "SystemInt32");
            emitCopy(cast.getLoc(), negative, result);
            Operation *done = emitJump(cast.getLoc(), 0);
            positive->setAttr("target", builder.getI64IntegerAttr(
                                            static_cast<int64_t>(address)));
            emitCopy(cast.getLoc(), source, result);
            done->setAttr("target", builder.getI64IntegerAttr(
                                        static_cast<int64_t>(address)));
          } else if (cast.getIn().getType().isInteger(32) &&
                     cast.getOut().getType().isInteger(64)) {
            emitPush(cast.getLoc(), source);
            emitPush(cast.getLoc(), result);
            emitExtern(cast.getLoc(),
                       "SystemConvert.__ToInt64__SystemInt32__SystemInt64");
          } else {
            return cast.emitError("unsupported signed integer extension");
          }
          continue;
        }
        if (auto cast = dyn_cast<arith::ExtUIOp>(operation)) {
          std::string source = valueSlots.lookup(cast.getIn());
          std::string result = valueSlots.lookup(cast.getOut());
          if (source.empty() || result.empty() ||
              !cast.getIn().getType().isInteger(1) ||
              !cast.getOut().getType().isInteger(32))
            return cast.emitError("unsupported unsigned integer extension");
          constexpr StringLiteral zero = "__vrc_bool_to_i32_false";
          constexpr StringLiteral one = "__vrc_bool_to_i32_true";
          addSlot(zero, "SystemInt32", "0", false);
          addSlot(one, "SystemInt32", "1", false);
          emitPush(cast.getLoc(), source);
          Operation *useZero = create(
              cast.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          emitCopy(cast.getLoc(), one, result);
          Operation *done = emitJump(cast.getLoc(), 0);
          useZero->setAttr("target", builder.getI64IntegerAttr(
                                         static_cast<int64_t>(address)));
          emitCopy(cast.getLoc(), zero, result);
          done->setAttr("target", builder.getI64IntegerAttr(
                                      static_cast<int64_t>(address)));
          continue;
        }
        if (auto cast = dyn_cast<arith::TruncIOp>(operation)) {
          std::string source = valueSlots.lookup(cast.getIn());
          std::string result = valueSlots.lookup(cast.getOut());
          if (source.empty() || result.empty() ||
              !cast.getIn().getType().isInteger(64) ||
              !cast.getOut().getType().isInteger(32))
            return cast.emitError("unsupported integer truncation");
          emitPush(cast.getLoc(), source);
          emitPush(cast.getLoc(), result);
          emitExtern(cast.getLoc(),
                     "SystemConvert.__ToInt32__SystemInt64__SystemInt32");
          continue;
        }
        if (auto cast = dyn_cast<arith::SIToFPOp>(operation)) {
          std::string source = valueSlots.lookup(cast.getIn());
          std::string result = valueSlots.lookup(cast.getOut());
          if (source.empty() || result.empty() ||
              !cast.getIn().getType().isInteger(32) ||
              !cast.getOut().getType().isF32())
            return cast.emitError(
                "signed integer to float conversion supports only i32 to f32");
          emitPush(cast.getLoc(), source);
          emitPush(cast.getLoc(), result);
          emitExtern(cast.getLoc(),
                     "SystemConvert.__ToSingle__SystemInt32__SystemSingle");
          continue;
        }
        if (auto select = dyn_cast<arith::SelectOp>(operation)) {
          std::string condition = valueSlots.lookup(select.getCondition());
          std::string trueValue = valueSlots.lookup(select.getTrueValue());
          std::string falseValue = valueSlots.lookup(select.getFalseValue());
          std::string result = valueSlots.lookup(select.getResult());
          if (condition.empty() || trueValue.empty() || falseValue.empty() ||
              result.empty())
            return select.emitError("select has missing heap slots");
          emitPush(select.getLoc(), condition);
          Operation *useFalse = create(
              select.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          emitCopy(select.getLoc(), trueValue, result);
          Operation *done = emitJump(select.getLoc(), 0);
          useFalse->setAttr("target", builder.getI64IntegerAttr(
                                          static_cast<int64_t>(address)));
          emitCopy(select.getLoc(), falseValue, result);
          done->setAttr("target", builder.getI64IntegerAttr(
                                      static_cast<int64_t>(address)));
          continue;
        }
        if (auto maximum = dyn_cast<arith::MaxSIOp>(operation)) {
          std::string lhs = valueSlots.lookup(maximum.getLhs());
          std::string rhs = valueSlots.lookup(maximum.getRhs());
          std::string result = valueSlots.lookup(maximum.getResult());
          constexpr StringLiteral condition = "__vrc_i32_max_condition";
          if (lhs.empty() || rhs.empty() || result.empty() ||
              !maximum.getType().isInteger(32))
            return maximum.emitError("unsupported signed integer maximum");
          addSlot(condition, "SystemBoolean", "null", false);
          emitPush(maximum.getLoc(), lhs);
          emitPush(maximum.getLoc(), rhs);
          emitPush(maximum.getLoc(), condition);
          emitExtern(maximum.getLoc(),
                     "SystemInt32.__op_GreaterThanOrEqual__SystemInt32_"
                     "SystemInt32__SystemBoolean");
          emitPush(maximum.getLoc(), condition);
          Operation *useRhs = create(
              maximum.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          emitCopy(maximum.getLoc(), lhs, result);
          Operation *done = emitJump(maximum.getLoc(), 0);
          useRhs->setAttr("target", builder.getI64IntegerAttr(
                                        static_cast<int64_t>(address)));
          emitCopy(maximum.getLoc(), rhs, result);
          done->setAttr("target", builder.getI64IntegerAttr(
                                      static_cast<int64_t>(address)));
          continue;
        }
        if (auto minimum = dyn_cast<arith::MinSIOp>(operation)) {
          std::string lhs = valueSlots.lookup(minimum.getLhs());
          std::string rhs = valueSlots.lookup(minimum.getRhs());
          std::string result = valueSlots.lookup(minimum.getResult());
          std::string condition = result + "_min_condition";
          if (lhs.empty() || rhs.empty() || result.empty() ||
              !minimum.getType().isInteger(32))
            return minimum.emitError("unsupported signed integer minimum");
          addSlot(condition, "SystemBoolean", "null", false);
          emitPush(minimum.getLoc(), lhs);
          emitPush(minimum.getLoc(), rhs);
          emitPush(minimum.getLoc(), condition);
          emitExtern(minimum.getLoc(),
                     "SystemInt32.__op_LessThanOrEqual__SystemInt32_"
                     "SystemInt32__SystemBoolean");
          emitPush(minimum.getLoc(), condition);
          Operation *useRhs = create(
              minimum.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          emitCopy(minimum.getLoc(), lhs, result);
          Operation *done = emitJump(minimum.getLoc(), 0);
          useRhs->setAttr("target", builder.getI64IntegerAttr(
                                        static_cast<int64_t>(address)));
          emitCopy(minimum.getLoc(), rhs, result);
          done->setAttr("target", builder.getI64IntegerAttr(
                                      static_cast<int64_t>(address)));
          continue;
        }
        if (auto shift = dyn_cast<arith::ShRUIOp>(operation)) {
          std::string value = valueSlots.lookup(shift.getLhs());
          std::string amount = valueSlots.lookup(shift.getRhs());
          std::string result = valueSlots.lookup(shift.getResult());
          if (value.empty() || amount.empty() || result.empty() ||
              !shift.getType().isInteger(32))
            return shift.emitError(
                "logical right shift currently requires i32 operands");

          constexpr StringLiteral zero = "__vrc_lshr_i32_zero";
          constexpr StringLiteral width = "__vrc_lshr_i32_width";
          constexpr StringLiteral allOnes = "__vrc_lshr_i32_all_ones";
          constexpr StringLiteral isZero = "__vrc_lshr_i32_is_zero";
          constexpr StringLiteral arithmetic = "__vrc_lshr_i32_arithmetic";
          constexpr StringLiteral inverseAmount =
              "__vrc_lshr_i32_inverse_amount";
          constexpr StringLiteral highMask = "__vrc_lshr_i32_high_mask";
          constexpr StringLiteral lowMask = "__vrc_lshr_i32_low_mask";
          addSlot(zero, "SystemInt32", "0", false);
          addSlot(width, "SystemInt32", "32", false);
          addSlot(allOnes, "SystemInt32", "-1", false);
          addSlot(isZero, "SystemBoolean", "null", false);
          addSlot(arithmetic, "SystemInt32", "0", false);
          addSlot(inverseAmount, "SystemInt32", "0", false);
          addSlot(highMask, "SystemInt32", "0", false);
          addSlot(lowMask, "SystemInt32", "0", false);

          emitPush(shift.getLoc(), amount);
          emitPush(shift.getLoc(), zero);
          emitPush(shift.getLoc(), isZero);
          emitExtern(shift.getLoc(),
                     "SystemInt32.__op_Equality__SystemInt32_SystemInt32__"
                     "SystemBoolean");
          emitPush(shift.getLoc(), isZero);
          Operation *useMaskedShift = create(
              shift.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          emitCopy(shift.getLoc(), value, result);
          Operation *done = emitJump(shift.getLoc(), 0);
          useMaskedShift->setAttr("target", builder.getI64IntegerAttr(
                                                static_cast<int64_t>(address)));

          emitPush(shift.getLoc(), value);
          emitPush(shift.getLoc(), amount);
          emitPush(shift.getLoc(), arithmetic);
          emitExtern(shift.getLoc(),
                     "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__"
                     "SystemInt32");
          emitPush(shift.getLoc(), width);
          emitPush(shift.getLoc(), amount);
          emitPush(shift.getLoc(), inverseAmount);
          emitExtern(shift.getLoc(),
                     "SystemInt32.__op_Subtraction__SystemInt32_SystemInt32__"
                     "SystemInt32");
          emitPush(shift.getLoc(), allOnes);
          emitPush(shift.getLoc(), inverseAmount);
          emitPush(shift.getLoc(), highMask);
          emitExtern(shift.getLoc(),
                     "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__"
                     "SystemInt32");
          emitPush(shift.getLoc(), highMask);
          emitPush(shift.getLoc(), allOnes);
          emitPush(shift.getLoc(), lowMask);
          emitExtern(shift.getLoc(),
                     "SystemInt32.__op_LogicalXor__SystemInt32_SystemInt32__"
                     "SystemInt32");
          emitPush(shift.getLoc(), arithmetic);
          emitPush(shift.getLoc(), lowMask);
          emitPush(shift.getLoc(), result);
          emitExtern(shift.getLoc(),
                     "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__"
                     "SystemInt32");
          done->setAttr("target", builder.getI64IntegerAttr(
                                      static_cast<int64_t>(address)));
          continue;
        }
        if (auto remainder = dyn_cast<arith::RemSIOp>(operation)) {
          std::string lhs = valueSlots.lookup(remainder.getLhs());
          std::string rhs = valueSlots.lookup(remainder.getRhs());
          std::string result = valueSlots.lookup(remainder.getResult());
          constexpr StringLiteral quotient = "__vrc_remsi_quotient";
          constexpr StringLiteral product = "__vrc_remsi_product";
          if (!remainder.getType().isInteger(32) || lhs.empty() ||
              rhs.empty() || result.empty())
            return remainder.emitError(
                "signed remainder requires allocated i32 slots");
          addSlot(quotient, "SystemInt32", "0", false);
          addSlot(product, "SystemInt32", "0", false);
          emitPush(remainder.getLoc(), lhs);
          emitPush(remainder.getLoc(), rhs);
          emitPush(remainder.getLoc(), quotient);
          emitExtern(remainder.getLoc(),
                     "SystemInt32.__op_Division__SystemInt32_SystemInt32__"
                     "SystemInt32");
          emitPush(remainder.getLoc(), quotient);
          emitPush(remainder.getLoc(), rhs);
          emitPush(remainder.getLoc(), product);
          emitExtern(remainder.getLoc(),
                     "SystemInt32.__op_Multiplication__SystemInt32_"
                     "SystemInt32__SystemInt32");
          emitPush(remainder.getLoc(), lhs);
          emitPush(remainder.getLoc(), product);
          emitPush(remainder.getLoc(), result);
          emitExtern(remainder.getLoc(),
                     "SystemInt32.__op_Subtraction__SystemInt32_SystemInt32__"
                     "SystemInt32");
          continue;
        }
        StringRef signature = binaryExtern(&operation);
        if (auto comparison = dyn_cast<arith::CmpIOp>(operation))
          signature = comparisonExtern(comparison);
        if (auto comparison = dyn_cast<arith::CmpFOp>(operation))
          signature = comparisonExtern(comparison);
        if (signature.empty() || operation.getNumOperands() != 2 ||
            operation.getNumResults() != 1)
          return operation.emitError()
                 << "operation " << operation.getName()
                 << " is not in the initial scalar Udon extern legalization "
                    "set";
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
    if (type == "SystemInt64" && initial == "0")
      initial = "null";
    slots.push_back({name.str(), type.str(), initial.str(), isPublic});
  }

  std::string selectUnityBuffer(Location location, StringRef token,
                                StringRef prefix) {
    if (unityBufferPool.empty())
      return {};
    std::string selected = (prefix + "_texture").str();
    addSlot(selected, "UnityEngineRenderTexture", "null", false);
    SmallVector<Operation *> done;
    for (auto [index, entry] : llvm::enumerate(unityBufferPool)) {
      std::string id = "__vrc_unity_buffer_id_" + std::to_string(index);
      constexpr StringLiteral matches = "__vrc_unity_buffer_matches";
      addSlot(id, "SystemInt32", std::to_string(entry.id), false);
      addSlot(matches, "SystemBoolean", "null", false);
      emitPush(location, token);
      emitPush(location, id);
      emitPush(location, matches);
      emitExtern(location,
                 "SystemInt32.__op_Equality__SystemInt32_SystemInt32__"
                 "SystemBoolean");
      emitPush(location, matches);
      Operation *next = create(
          location, vm::JumpIfFalseOp::getOperationName(),
          {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
      address += 8;
      emitCopy(location, entry.heapSlot, selected);
      done.push_back(emitJump(location, 0));
      next->setAttr("target",
                    builder.getI64IntegerAttr(static_cast<int64_t>(address)));
    }
    for (Operation *jump : done)
      jump->setAttr("target",
                    builder.getI64IntegerAttr(static_cast<int64_t>(address)));
    return selected;
  }

  void emitI32Literal(Location location, uint64_t value, StringRef result) {
    assert(value <= UINT32_MAX && "i32 literal exceeds 32 bits");
    constexpr StringLiteral zero = "__vrc_i32_literal_zero";
    constexpr StringLiteral one = "__vrc_i32_literal_one";
    constexpr StringLiteral scratch = "__vrc_i32_literal_scratch";
    addSlot(zero, "SystemInt32", "0", false);
    addSlot(one, "SystemInt32", "1", false);
    addSlot(scratch, "SystemInt32", "0", false);
    emitCopy(location, zero, result);
    if (!value)
      return;
    unsigned highestBit = 0;
    for (uint64_t remaining = value; remaining >>= 1;)
      ++highestBit;
    for (int bit = static_cast<int>(highestBit); bit >= 0; --bit) {
      emitPush(location, result);
      emitPush(location, result);
      emitPush(location, scratch);
      emitExtern(location,
                 "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                 "SystemInt32");
      emitCopy(location, scratch, result);
      if ((value >> bit) & 1) {
        emitPush(location, result);
        emitPush(location, one);
        emitPush(location, scratch);
        emitExtern(location,
                   "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                   "SystemInt32");
        emitCopy(location, scratch, result);
      }
    }
  }

  void emitI64Constant(Location location, const llvm::APInt &value,
                       StringRef result) {
    constexpr StringLiteral wide = "__vrc_i64_constant_wide";
    constexpr StringLiteral shifted = "__vrc_i64_constant_shifted";
    constexpr StringLiteral combined = "__vrc_i64_constant_combined";
    addSlot(wide, "SystemInt64", "null", false);
    addSlot(shifted, "SystemInt64", "null", false);
    addSlot(combined, "SystemInt64", "null", false);
    for (unsigned byteIndex = 0; byteIndex < 8; ++byteIndex) {
      uint64_t byteValue = value.extractBitsAsZExtValue(8, byteIndex * 8);
      std::string seed = "__vrc_i64_constant_byte_" + std::to_string(byteValue);
      addSlot(seed, "SystemInt32", std::to_string(byteValue), false);
      emitPush(location, seed);
      emitPush(location, wide);
      emitExtern(location, "SystemConvert.__ToInt64__SystemInt32__SystemInt64");
      StringRef component = wide;
      if (byteIndex) {
        std::string shift =
            "__vrc_i64_constant_shift_" + std::to_string(byteIndex);
        addSlot(shift, "SystemInt32", std::to_string(byteIndex * 8), false);
        emitPush(location, component);
        emitPush(location, shift);
        emitPush(location, shifted);
        emitExtern(location,
                   "SystemInt64.__op_LeftShift__SystemInt64_SystemInt32__"
                   "SystemInt64");
        component = shifted;
      }
      if (!byteIndex) {
        emitCopy(location, component, result);
      } else {
        emitPush(location, result);
        emitPush(location, component);
        emitPush(location, combined);
        emitExtern(location,
                   "SystemInt64.__op_LogicalOr__SystemInt64_SystemInt64__"
                   "SystemInt64");
        emitCopy(location, combined, result);
      }
    }
  }

  void emitByteLiteral(Location location, uint8_t value, StringRef result) {
    uint8_t high = value >> 4;
    uint8_t low = value & 15;
    auto nibbleSlot = [&](uint8_t nibble) {
      std::string slot = "__vrc_byte_literal_" + std::to_string(nibble);
      addSlot(slot, "SystemInt32", std::to_string(nibble), false);
      return slot;
    };
    std::string lowSlot = nibbleSlot(low);
    if (!high) {
      emitCopy(location, lowSlot, result);
      return;
    }
    std::string highSlot = nibbleSlot(high);
    constexpr StringLiteral shift = "__vrc_byte_literal_shift";
    constexpr StringLiteral shifted = "__vrc_byte_literal_shifted";
    addSlot(shift, "SystemInt32", "4", false);
    addSlot(shifted, "SystemInt32", "0", false);
    emitPush(location, highSlot);
    emitPush(location, shift);
    emitPush(location, shifted);
    emitExtern(location, "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__"
                         "SystemInt32");
    if (!low) {
      emitCopy(location, shifted, result);
      return;
    }
    emitPush(location, shifted);
    emitPush(location, lowSlot);
    emitPush(location, result);
    emitExtern(location, "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                         "SystemInt32");
  }

  LogicalResult emitRawLoad(Location location, StringRef pointer,
                            StringRef result, Type type) {
    if (!type.isInteger(8) && !type.isInteger(32) && !type.isInteger(64) &&
        !type.isF32() && !isa<host::PtrType>(type))
      return failure();
    bool wide = type.isInteger(64);
    unsigned byteWidth = type.isInteger(8) ? 1 : wide ? 8 : 4;
    std::string prefix =
        wide ? "__vrc_raw_load_i64_scratch" : "__vrc_raw_load_i32_scratch";
    std::string bits = prefix + "_bits";
    std::string scratchIndex = prefix + "_index";
    std::string scratchByte = prefix + "_byte";
    std::string scratchWide = prefix + "_wide";
    std::string scratchComponent = prefix + "_component";
    std::string scratchCombined = prefix + "_combined";
    addSlot(bits, wide ? "SystemInt64" : "SystemInt32", "0", false);
    for (unsigned byteIndex = 0; byteIndex < byteWidth; ++byteIndex) {
      std::string index = pointer.str();
      if (byteIndex) {
        std::string offset = prefix + "_offset_" + std::to_string(byteIndex);
        index = scratchIndex;
        addSlot(offset, "SystemInt32", std::to_string(byteIndex), false);
        addSlot(index, "SystemInt32", "0", false);
        emitPush(location, pointer);
        emitPush(location, offset);
        emitPush(location, index);
        emitExtern(location,
                   "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                   "SystemInt32");
      }
      std::string byte = scratchByte;
      addSlot(byte, "SystemInt32", "0", false);
      emitPush(location, rawHeapSlot);
      emitPush(location, index);
      emitPush(location, byte);
      emitExtern(location, "SystemInt32Array.__Get__SystemInt32__SystemInt32");
      std::string component = byte;
      if (wide) {
        component = scratchWide;
        addSlot(component, "SystemInt64", "0", false);
        emitPush(location, byte);
        emitPush(location, component);
        emitExtern(location,
                   "SystemConvert.__ToInt64__SystemInt32__SystemInt64");
      }
      if (byteIndex) {
        std::string shift = prefix + "_shift_" + std::to_string(byteIndex);
        std::string shifted = scratchComponent;
        addSlot(shift, "SystemInt32", std::to_string(byteIndex * 8), false);
        addSlot(shifted, wide ? "SystemInt64" : "SystemInt32", "0", false);
        emitPush(location, component);
        emitPush(location, shift);
        emitPush(location, shifted);
        emitExtern(location, wide ? "SystemInt64.__op_LeftShift__SystemInt64_"
                                    "SystemInt32__SystemInt64"
                                  : "SystemInt32.__op_LeftShift__SystemInt32_"
                                    "SystemInt32__SystemInt32");
        component = shifted;
      }
      if (!byteIndex) {
        emitCopy(location, component, bits);
      } else {
        std::string combined = scratchCombined;
        addSlot(combined, wide ? "SystemInt64" : "SystemInt32", "0", false);
        emitPush(location, bits);
        emitPush(location, component);
        emitPush(location, combined);
        emitExtern(location,
                   wide
                       ? "SystemInt64.__op_LogicalOr__SystemInt64_SystemInt64__"
                         "SystemInt64"
                       : "SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__"
                         "SystemInt32");
        emitCopy(location, combined, bits);
      }
    }
    if (type.isF32()) {
      emitPush(location, bits);
      emitPush(location, result);
      emitExtern(location,
                 "SystemBitConverter.__Int32BitsToSingle__SystemInt32__"
                 "SystemSingle");
    } else {
      emitCopy(location, bits, result);
    }
    return success();
  }

  LogicalResult emitRawStore(Location location, StringRef pointer,
                             StringRef value, Type type) {
    if (!type.isInteger(8) && !type.isInteger(32) && !type.isInteger(64) &&
        !type.isF32() && !isa<host::PtrType>(type))
      return failure();
    bool wide = type.isInteger(64);
    unsigned byteWidth = type.isInteger(8) ? 1 : wide ? 8 : 4;
    std::string prefix =
        wide ? "__vrc_raw_store_i64_scratch" : "__vrc_raw_store_i32_scratch";
    std::string bits = prefix + "_bits";
    std::string mask = prefix + "_mask";
    std::string scratchComponent = prefix + "_component";
    std::string scratchByte = prefix + "_byte";
    std::string scratchWideByte = prefix + "_byte_wide";
    std::string scratchIndex = prefix + "_index";
    addSlot(bits, wide ? "SystemInt64" : "SystemInt32", "0", false);
    addSlot(mask, wide ? "SystemInt64" : "SystemInt32", wide ? "null" : "255",
            false);
    if (wide) {
      std::string maskSeed = mask + "_i32_seed";
      addSlot(maskSeed, "SystemInt32", "255", false);
      emitPush(location, maskSeed);
      emitPush(location, mask);
      emitExtern(location, "SystemConvert.__ToInt64__SystemInt32__SystemInt64");
    }
    if (type.isF32()) {
      emitPush(location, value);
      emitPush(location, bits);
      emitExtern(location,
                 "SystemBitConverter.__SingleToInt32Bits__SystemSingle__"
                 "SystemInt32");
    } else {
      emitCopy(location, value, bits);
    }
    for (unsigned byteIndex = 0; byteIndex < byteWidth; ++byteIndex) {
      std::string component = bits;
      if (byteIndex) {
        std::string shift = prefix + "_shift_" + std::to_string(byteIndex);
        component = scratchComponent;
        addSlot(shift, "SystemInt32", std::to_string(byteIndex * 8), false);
        addSlot(component, wide ? "SystemInt64" : "SystemInt32", "0", false);
        emitPush(location, bits);
        emitPush(location, shift);
        emitPush(location, component);
        emitExtern(location, wide ? "SystemInt64.__op_RightShift__SystemInt64_"
                                    "SystemInt32__SystemInt64"
                                  : "SystemInt32.__op_RightShift__SystemInt32_"
                                    "SystemInt32__SystemInt32");
      }
      std::string byte = scratchByte;
      std::string masked = wide ? scratchWideByte : byte;
      addSlot(masked, wide ? "SystemInt64" : "SystemInt32", "0", false);
      addSlot(byte, "SystemInt32", "0", false);
      emitPush(location, component);
      emitPush(location, mask);
      emitPush(location, masked);
      emitExtern(location,
                 wide ? "SystemInt64.__op_LogicalAnd__SystemInt64_SystemInt64__"
                        "SystemInt64"
                      : "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__"
                        "SystemInt32");
      if (wide) {
        emitPush(location, masked);
        emitPush(location, byte);
        emitExtern(location,
                   "SystemConvert.__ToInt32__SystemInt64__SystemInt32");
      }
      std::string index = pointer.str();
      if (byteIndex) {
        std::string offset = prefix + "_offset_" + std::to_string(byteIndex);
        index = scratchIndex;
        addSlot(offset, "SystemInt32", std::to_string(byteIndex), false);
        addSlot(index, "SystemInt32", "0", false);
        emitPush(location, pointer);
        emitPush(location, offset);
        emitPush(location, index);
        emitExtern(location,
                   "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                   "SystemInt32");
      }
      emitPush(location, rawHeapSlot);
      emitPush(location, index);
      emitPush(location, byte);
      emitExtern(location,
                 "SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid");
    }
    return success();
  }

  void emitRawMemcpy(Location location, StringRef destination, StringRef source,
                     StringRef byteCount) {
    std::string prefix = "__vrc_raw_memcpy_scratch";
    std::string counter = prefix + "_counter";
    std::string one = prefix + "_one";
    std::string condition = prefix + "_condition";
    std::string sourceIndex = prefix + "_source_index";
    std::string destinationIndex = prefix + "_destination_index";
    std::string byte = prefix + "_byte";
    std::string next = prefix + "_next";
    addSlot(counter, "SystemInt32", "0", false);
    addSlot(one, "SystemInt32", "1", false);
    addSlot(condition, "SystemBoolean", "null", false);
    addSlot(sourceIndex, "SystemInt32", "0", false);
    addSlot(destinationIndex, "SystemInt32", "0", false);
    addSlot(byte, "SystemInt32", "0", false);
    addSlot(next, "SystemInt32", "0", false);
    uint64_t loopAddress = address;
    emitPush(location, counter);
    emitPush(location, byteCount);
    emitPush(location, condition);
    emitExtern(location, "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__"
                         "SystemBoolean");
    emitPush(location, condition);
    Operation *done =
        create(location, vm::JumpIfFalseOp::getOperationName(),
               {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
    address += 8;
    emitPush(location, source);
    emitPush(location, counter);
    emitPush(location, sourceIndex);
    emitExtern(location, "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                         "SystemInt32");
    emitPush(location, destination);
    emitPush(location, counter);
    emitPush(location, destinationIndex);
    emitExtern(location, "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                         "SystemInt32");
    emitPush(location, rawHeapSlot);
    emitPush(location, sourceIndex);
    emitPush(location, byte);
    emitExtern(location, "SystemInt32Array.__Get__SystemInt32__SystemInt32");
    emitPush(location, rawHeapSlot);
    emitPush(location, destinationIndex);
    emitPush(location, byte);
    emitExtern(location,
               "SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid");
    emitPush(location, counter);
    emitPush(location, one);
    emitPush(location, next);
    emitExtern(location, "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                         "SystemInt32");
    emitCopy(location, next, counter);
    emitJump(location, static_cast<int64_t>(loopAddress));
    done->setAttr("target",
                  builder.getI64IntegerAttr(static_cast<int64_t>(address)));
  }

  void emitGlobalInitialization(Location location) {
    emitPush(location, globalInitializedSlot);
    emitPush(location, globalOneSlot);
    emitPush(location, globalReadySlot);
    emitExtern(location, "SystemInt32.__op_Equality__SystemInt32_SystemInt32__"
                         "SystemBoolean");
    emitPush(location, globalReadySlot);
    Operation *initialize =
        create(location, vm::JumpIfFalseOp::getOperationName(),
               {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
    address += 8;
    Operation *ready = emitJump(location, 0);
    uint64_t initializeAddress = address;
    if (rawHeapBytes && !externalRawHeapInitialization) {
      emitPush(location, rawHeapLengthSlot);
      emitPush(location, rawHeapSlot);
      emitExtern(location,
                 "SystemInt32Array.__ctor__SystemInt32__SystemInt32Array");
    }
    constexpr StringLiteral rawIndexSlot = "__vrc_raw_global_init_index";
    constexpr StringLiteral rawValueSlot = "__vrc_raw_global_init_value";
    constexpr StringLiteral rawNextSlot = "__vrc_raw_global_init_next";
    constexpr StringLiteral rawOneSlot = "__vrc_raw_global_init_one";
    addSlot(rawIndexSlot, "SystemInt32", "0", false);
    addSlot(rawValueSlot, "SystemInt32", "0", false);
    addSlot(rawNextSlot, "SystemInt32", "0", false);
    addSlot(rawOneSlot, "SystemInt32", "1", false);
    if (!externalRawHeapInitialization)
      for (host::GlobalOp global : module.getOps<host::GlobalOp>()) {
        auto initial = rawGlobalInitializers.find(global.getSymName());
        auto base = rawGlobalBases.find(global.getSymName());
        if (initial == rawGlobalInitializers.end() ||
            base == rawGlobalBases.end())
          continue;
        emitI32Literal(location, base->second, rawIndexSlot);
        for (int8_t byte : initial->second.asArrayRef()) {
          if (byte) {
            emitByteLiteral(location, static_cast<uint8_t>(byte), rawValueSlot);
            emitPush(location, rawHeapSlot);
            emitPush(location, rawIndexSlot);
            emitPush(location, rawValueSlot);
            emitExtern(
                location,
                "SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid");
          }
          emitPush(location, rawIndexSlot);
          emitPush(location, rawOneSlot);
          emitPush(location, rawNextSlot);
          emitExtern(location,
                     "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                     "SystemInt32");
          emitCopy(location, rawNextSlot, rawIndexSlot);
        }
      }
    constexpr StringLiteral globalLengthSlot = "__vrc_global_init_length";
    constexpr StringLiteral globalIndexSlot = "__vrc_global_init_index";
    constexpr StringLiteral globalIntValueSlot = "__vrc_global_init_int_value";
    constexpr StringLiteral globalFloatBitsSlot =
        "__vrc_global_init_float_bits";
    constexpr StringLiteral globalFloatValueSlot =
        "__vrc_global_init_float_value";
    addSlot(globalLengthSlot, "SystemInt32", "0", false);
    addSlot(globalIndexSlot, "SystemInt32", "0", false);
    addSlot(globalIntValueSlot, "SystemInt32", "0", false);
    addSlot(globalFloatBitsSlot, "SystemInt32", "0", false);
    addSlot(globalFloatValueSlot, "SystemSingle", "0", false);
    for (const std::string &name : globalOrder) {
      const GlobalPlan &plan = globalPlans.at(name);
      emitI32Literal(location, plan.type.getNumElements(), globalLengthSlot);
      emitPush(location, globalLengthSlot);
      emitPush(location, plan.slot);
      emitExtern(
          location,
          plan.type.getElementType().isF32()
              ? "SystemSingleArray.__ctor__SystemInt32__SystemSingleArray"
              : "SystemInt32Array.__ctor__SystemInt32__SystemInt32Array");
      unsigned index = 0;
      for (Attribute value : plan.initialValue.getValues<Attribute>()) {
        bool zero = false;
        bool floatingValue = false;
        uint32_t bits = 0;
        if (auto integer = dyn_cast<IntegerAttr>(value)) {
          zero = integer.getValue().isZero();
          bits = static_cast<uint32_t>(integer.getInt());
        } else if (auto floating = dyn_cast<FloatAttr>(value)) {
          zero = floating.getValue().isZero();
          floatingValue = true;
          bits = static_cast<uint32_t>(
              floating.getValue().bitcastToAPInt().getZExtValue());
        }
        if (!zero) {
          emitI32Literal(location, index, globalIndexSlot);
          StringRef valueSlot = globalIntValueSlot;
          if (floatingValue) {
            emitI32Literal(location, bits, globalFloatBitsSlot);
            emitPush(location, globalFloatBitsSlot);
            emitPush(location, globalFloatValueSlot);
            emitExtern(location,
                       "SystemBitConverter.__Int32BitsToSingle__SystemInt32__"
                       "SystemSingle");
            valueSlot = globalFloatValueSlot;
          } else {
            emitI32Literal(location, bits, globalIntValueSlot);
          }
          emitPush(location, plan.slot);
          emitPush(location, globalIndexSlot);
          emitPush(location, valueSlot);
          emitExtern(location,
                     plan.type.getElementType().isF32()
                         ? "SystemSingleArray.__Set__SystemInt32_SystemSingle__"
                           "SystemVoid"
                         : "SystemInt32Array.__Set__SystemInt32_SystemInt32__"
                           "SystemVoid");
        }
        ++index;
      }
    }
    emitCopy(location, globalOneSlot, globalInitializedSlot);
    initialize->setAttr("target", builder.getI64IntegerAttr(
                                      static_cast<int64_t>(initializeAddress)));
    ready->setAttr("target",
                   builder.getI64IntegerAttr(static_cast<int64_t>(address)));
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
  llvm::StringMap<GlobalPlan> globalPlans;
  llvm::StringMap<std::string> sharedConstantSlots;
  unsigned sharedConstantIndex = 0;
  SmallVector<std::string> globalOrder;
  llvm::StringMap<uint64_t> rawGlobalBases;
  llvm::StringMap<DenseI8ArrayAttr> rawGlobalInitializers;
  llvm::DenseMap<Operation *, uint64_t> rawAllocaBases;
  SmallVector<UnityBufferPoolEntry> unityBufferPool;
  uint64_t rawHeapBytes = 0;
  bool externalRawHeapInitialization = false;
  SmallVector<std::pair<Operation *, func::FuncOp>> callRelocations;
  uint64_t callReturnIndex = 0;
  static constexpr StringLiteral globalInitializedSlot =
      "__vrc_globals_initialized";
  static constexpr StringLiteral globalOneSlot = "__vrc_globals_one";
  static constexpr StringLiteral globalReadySlot = "__vrc_globals_ready";
  static constexpr StringLiteral rawHeapSlot = "__vrc_raw_heap";
  static constexpr StringLiteral rawHeapLengthSlot = "__vrc_raw_heap_length";
  static constexpr StringLiteral asyncResumeStateSlot =
      "__udoncl_async_resume_state";
  static constexpr StringLiteral asyncCompletionStatusSlot =
      "__udoncl_async_completion_status";
  static constexpr StringLiteral asyncCommandKindSlot =
      "__udoncl_async_command_kind";
  static constexpr StringLiteral asyncCommandBufferSlot =
      "__udoncl_async_command_buffer";
  static constexpr StringLiteral asyncCommandOffsetSlot =
      "__udoncl_async_command_offset";
  static constexpr StringLiteral asyncCommandSizeSlot =
      "__udoncl_async_command_size";
  static constexpr StringLiteral asyncCommandPointerSlot =
      "__udoncl_async_command_pointer";
  static constexpr StringLiteral asyncEventSlot = "__udoncl_async_event";
  static constexpr StringLiteral asyncZeroSlot = "__udoncl_async_zero";
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
