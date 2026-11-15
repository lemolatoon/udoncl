#include "VRC/Conversion/Passes.h"
#include "HostStorageLiveness.h"
#include "VRC/ABI/Manifest/UdonSharpABI.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "udoncl/array.h"
#include "udoncl/string.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"

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

struct AsyncResumeTarget {
  int64_t frameId;
  int64_t resumePc;
  func::FuncOp function;
  Block *block;
};

struct GlobalPlan {
  std::string slot;
  MemRefType type;
  ElementsAttr initialValue;
  uint64_t base;
};

struct GlobalBankPlan {
  std::string slot;
  Type elementType;
  uint64_t length = 0;
  SmallVector<std::pair<uint64_t, ElementsAttr>> initializers;
};

struct TypedSpillBankPlan {
  std::string slot;
  std::string scalarType;
  std::string arrayType;
  unsigned length = 0;
};

struct ConstantPoolLocation {
  unsigned index = 0;
};

struct UnityBufferPoolEntry {
  int64_t id;
  int64_t logicalBytes;
  std::string heapSlot;
  unsigned index;
};

struct RawObjectPlan {
  std::string name;
  std::string kind;
  std::string reason;
  uint64_t base;
  uint64_t bytes;
  uint64_t alignment;
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
  if (isa<::vrc::vrc::DispatchDescType>(type))
    return StringRef("SystemInt32");
  if (isa<host::FrameType>(type))
    return StringRef("SystemInt32");
  if (auto pointer = dyn_cast<host::PtrType>(type)) {
    if (pointer.getAddressSpace() == UDONCL_STRING_ADDRESS_SPACE)
      return StringRef("SystemString");
    if (pointer.getAddressSpace() == UDONCL_INT32_ARRAY_ADDRESS_SPACE)
      return StringRef("SystemInt32Array");
    return StringRef("SystemInt32");
  }
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

static FailureOr<std::string> slotStorageKey(Type type) {
  auto physicalType = udonType(type);
  if (failed(physicalType))
    return failure();
  std::string key = physicalType->str();
  if (auto memref = dyn_cast<MemRefType>(type)) {
    key += "_n";
    if (memref.hasStaticShape())
      key += std::to_string(memref.getNumElements());
    else
      key += "dynamic";
  }
  return key;
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

struct ScalarLoweringOptions {
  unsigned materializationLimit = 0;
  unsigned rawAddressCacheLimit = 0;
  unsigned unityBufferIdRematerializationCount = 0;
  unsigned unityBufferIdRematerializationAddressDelta = 0;
  unsigned unityBufferIdRematerializationAttempts = 0;
  bool enforceCapacity = true;
};

class ScalarLowering {
public:
  ScalarLowering(ModuleOp module, ScalarLoweringOptions options)
      : module(module), builder(module.getContext()),
        materializationLimit(options.materializationLimit),
        rawAddressCacheLimit(options.rawAddressCacheLimit),
        unityBufferIdRematerializationCount(
            options.unityBufferIdRematerializationCount),
        unityBufferIdRematerializationAddressDelta(
            options.unityBufferIdRematerializationAddressDelta),
        unityBufferIdRematerializationAttempts(
            options.unityBufferIdRematerializationAttempts),
        enforceCapacity(options.enforceCapacity),
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
    bool hasCompilerAsyncABI = false;
    if (failed(planUnityBufferPool()) || failed(planRawMemory()))
      return failure();
    if (module->hasAttr("vrc.debug_compiler_phases")) {
      llvm::errs() << "udonclc-scalar-lowering=raw-memory-planned"
                   << " raw_heap_bytes=" << rawHeapBytes << "\n";
      llvm::errs().flush();
    }
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
      std::string storageKey = type.getElementType().isF32() ? "f32" : "i32";
      auto bank = globalBankIndices.find(storageKey);
      unsigned bankIndex;
      if (bank == globalBankIndices.end()) {
        bankIndex = globalBanks.size();
        std::string slot = "__vrc_global_bank_" + storageKey;
        addSlot(slot,
                type.getElementType().isF32() ? "SystemSingleArray"
                                              : "SystemInt32Array",
                "null", false);
        globalBanks.push_back(
            GlobalBankPlan{std::move(slot), type.getElementType()});
        globalBankIndices.try_emplace(storageKey, bankIndex);
      } else {
        bankIndex = bank->second;
      }
      GlobalBankPlan &bankPlan = globalBanks[bankIndex];
      if (static_cast<uint64_t>(type.getNumElements()) >
          static_cast<uint64_t>(INT32_MAX) - bankPlan.length)
        return global.emitError(
            "packed Udon typed global bank exceeds SystemInt32 indexing");
      uint64_t base = bankPlan.length;
      bankPlan.length += static_cast<uint64_t>(type.getNumElements());
      bankPlan.initializers.emplace_back(base, elements);
      globalPlans.try_emplace(global.getSymName(),
                              GlobalPlan{bankPlan.slot, type, elements, base});
    }
    if (!globalPlans.empty() ||
        (rawHeapBytes && !externalRawHeapInitialization)) {
      addSlot(globalInitializedSlot, "SystemInt32", "0", false);
      addSlot(globalOneSlot, "SystemInt32", "1", false);
      addSlot(globalReadySlot, "SystemBoolean", "null", false);
    }
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      auto exportId = function->getAttrOfType<IntegerAttr>("vrc.export_id");
      bool compilerCompletion =
          function->hasAttr("vrc.compiler_async_completion");
      hasCompilerAsyncABI |= compilerCompletion;
      if (exportId && compilerCompletion)
        return function.emitError(
            "compiler async completion cannot be a public C export");
      if (compilerCompletion &&
          (function.getNumArguments() != 2 ||
           !function.getArgumentTypes()[0].isInteger(32) ||
           !function.getArgumentTypes()[1].isInteger(32) ||
           function.getNumResults() != 0))
        return function.emitError(
            "compiler async completion requires (i32, i32) -> ()");
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
      (exportId || compilerCompletion ? exportsToLower : internalsToLower)
          .push_back(function);
    }
    if (exportsToLower.empty())
      return module.emitError(
          "scalar Udon lowering requires at least one export");

    llvm::sort(exportsToLower, [](func::FuncOp lhs, func::FuncOp rhs) {
      auto left = lhs->getAttrOfType<IntegerAttr>("vrc.export_id");
      auto right = rhs->getAttrOfType<IntegerAttr>("vrc.export_id");
      if (bool(left) != bool(right))
        return bool(left);
      if (!left)
        return lhs.getSymName() < rhs.getSymName();
      return left.getInt() < right.getInt();
    });
    llvm::sort(internalsToLower, [](func::FuncOp lhs, func::FuncOp rhs) {
      return lhs.getSymName() < rhs.getSymName();
    });
    functions.clear();
    llvm::append_range(functions, exportsToLower);
    llvm::append_range(functions, internalsToLower);
    planMaterializedConstants(functions);
    if (externalRawHeapInitialization && rawHeapBytes) {
      SmallVector<int8_t> image(rawHeapBytes, 0);
      for (auto &entry : rawGlobalInitializers) {
        auto base = rawGlobalBases.find(entry.getKey());
        if (base == rawGlobalBases.end())
          return module.emitError("raw global initializer has no heap base");
        llvm::copy(entry.getValue().asArrayRef(), image.begin() + base->second);
      }
      for (auto [index, value] : llvm::enumerate(constantPoolValues)) {
        uint64_t byteOffset =
            (constantPoolWordBase + static_cast<uint64_t>(index)) * 4;
        for (unsigned byte = 0; byte < 4; ++byte)
          image[byteOffset + byte] =
              static_cast<int8_t>((value >> (byte * 8)) & 0xff);
      }
      module->setAttr("udonasm.raw_heap_initializer",
                      DenseI8ArrayAttr::get(module.getContext(), image));
    }
    bool hasAsyncTransfers = false;
    module.walk([&](Operation *operation) {
      hasAsyncTransfers |=
          isa<runtime::BufferUploadOp, runtime::BufferReadbackOp>(operation);
    });
    if (hasAsyncTransfers || hasCompilerAsyncABI) {
      addSlot(asyncCompletionStatusSlot, "SystemInt32", "0", true);
      addSlot(asyncCompletionObjectSlot, "SystemInt32", "0", true);
      addSlot(asyncPendingObjectSlot, "SystemInt32", "0", true);
      addSlot(asyncCommandKindSlot, "SystemInt32", "0", true);
      addSlot(asyncCommandBufferSlot, "SystemInt32", "0", true);
      addSlot(asyncCommandOffsetSlot, "SystemInt32", "0", true);
      addSlot(asyncCommandSizeSlot, "SystemInt32", "0", true);
      addSlot(asyncCommandPointerSlot, "SystemInt32", "0", true);
      addSlot(asyncEventSlot, "SystemInt32", "1", false);
    }

    UdonSharpABIBuilder abiBuilder;
    for (func::FuncOp function : exportsToLower) {
      if (function->hasAttr("vrc.compiler_async_completion")) {
        auto event = function->getAttrOfType<StringAttr>("vrc.udon_event");
        if (!event || event.getValue().empty() ||
            !eventNames.insert(event.getValue().str()).second)
          return function.emitError(
              "compiler async completion requires a unique Udon event");
        continue;
      }
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

    llvm::DenseMap<Operation *, unsigned> internalSlotBanks;
    {
      SmallVector<func::FuncOp> bankedFunctions;
      llvm::SmallPtrSet<Operation *, 32> bankedSet;
      for (func::FuncOp function : internalsToLower) {
        if (function->hasAttr("vrc.compiler_async_dispatch"))
          continue;
        bankedFunctions.push_back(function);
        bankedSet.insert(function.getOperation());
      }

      llvm::DenseMap<Operation *, SmallVector<Operation *>> callees;
      for (func::FuncOp function : bankedFunctions) {
        function.walk([&](Operation *operation) {
          FlatSymbolRefAttr calleeAttr;
          if (auto call = dyn_cast<func::CallOp>(operation))
            calleeAttr = call.getCalleeAttr();
          else if (auto call = dyn_cast<host::CallOp>(operation))
            calleeAttr = call.getCalleeAttr();
          if (!calleeAttr)
            return;
          auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
              operation, calleeAttr);
          if (callee && bankedSet.contains(callee.getOperation()))
            callees[function.getOperation()].push_back(callee.getOperation());
        });
      }

      llvm::DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 32>> reachable;
      for (func::FuncOp function : bankedFunctions) {
        llvm::SmallPtrSet<Operation *, 32> visited;
        SmallVector<Operation *> worklist(
            callees[function.getOperation()].begin(),
            callees[function.getOperation()].end());
        while (!worklist.empty()) {
          Operation *callee = worklist.pop_back_val();
          if (!visited.insert(callee).second)
            continue;
          llvm::append_range(worklist, callees[callee]);
        }
        if (visited.contains(function.getOperation())) {
          function.emitError(
              "recursive internal Udon call graph has no static slot plan");
          return failure();
        }
        reachable.try_emplace(function.getOperation(), std::move(visited));
      }

      SmallVector<func::FuncOp> coloringOrder(bankedFunctions);
      llvm::sort(coloringOrder, [&](func::FuncOp left, func::FuncOp right) {
        auto conflictDegree = [&](func::FuncOp function) {
          unsigned degree = 0;
          for (func::FuncOp candidate : bankedFunctions)
            if (candidate != function &&
                (reachable[function.getOperation()].contains(
                     candidate.getOperation()) ||
                 reachable[candidate.getOperation()].contains(
                     function.getOperation())))
              ++degree;
          return degree;
        };
        unsigned leftDegree = conflictDegree(left);
        unsigned rightDegree = conflictDegree(right);
        if (leftDegree != rightDegree)
          return leftDegree > rightDegree;
        return left.getSymName() < right.getSymName();
      });
      for (func::FuncOp function : coloringOrder) {
        llvm::SmallSet<unsigned, 16> used;
        for (func::FuncOp candidate : bankedFunctions) {
          auto color = internalSlotBanks.find(candidate.getOperation());
          if (color == internalSlotBanks.end())
            continue;
          if (reachable[function.getOperation()].contains(
                  candidate.getOperation()) ||
              reachable[candidate.getOperation()].contains(
                  function.getOperation()))
            used.insert(color->second);
        }
        unsigned color = 0;
        while (used.contains(color))
          ++color;
        internalSlotBanks.try_emplace(function.getOperation(), color);
      }
    }

    if (failed(planDeclaredTypedSpillBanks()) ||
        failed(planInterproceduralValueSlots(functions)))
      return failure();
    compactSpillIndices();
    planAddressCaches();
    if (!typedSpillBanks.empty()) {
      addSlot(globalInitializedSlot, "SystemInt32", "0", false);
      addSlot(globalOneSlot, "SystemInt32", "1", false);
      addSlot(globalReadySlot, "SystemBoolean", "null", false);
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
      } else if (function->hasAttr("vrc.compiler_async_completion")) {
        plan.prefix = "__vrc_async_completion_";
        plan.parameterSlots = {asyncCompletionObjectSlot.str(),
                               asyncCompletionStatusSlot.str()};
      } else {
        auto bank = internalSlotBanks.find(function.getOperation());
        plan.prefix =
            bank == internalSlotBanks.end()
                ? "__vrc_internal_" + std::to_string(index) + "_"
                : "__vrc_internal_bank_" + std::to_string(bank->second) + "_";
        for (BlockArgument argument : function.getArguments()) {
          auto slot = interproceduralValueSlots.find(argument);
          if (slot == interproceduralValueSlots.end())
            return function.emitError(
                "internal argument has no interprocedural Udon slot");
          plan.parameterSlots.push_back(slot->second);
        }
        if (function.getNumResults()) {
          auto storageKey = slotStorageKey(function.getResultTypes().front());
          if (failed(storageKey))
            return function.emitError(
                "internal result has no physical Udon type");
          plan.resultSlot = plan.prefix + "result_" + *storageKey;
        }
        plan.returnSlot = plan.prefix + "return";
      }
      functionPlans.try_emplace(function.getOperation(), std::move(plan));
    }

    llvm::SmallSet<std::pair<int64_t, int64_t>, 16> resumeKeys;
    bool invalidResumeRoute = false;
    for (func::FuncOp function : functions) {
      auto frameId = function->getAttrOfType<IntegerAttr>("vrc.async_frame_id");
      function.walk([&](host::AsyncResumePointOp point) {
        std::pair<int64_t, int64_t> key(point.getFrameId(),
                                        point.getResumePc());
        if (!frameId || frameId.getInt() != key.first || key.first <= 0 ||
            key.second <= 0 || !resumeKeys.insert(key).second) {
          point.emitError("requires a unique positive frame/resume route that "
                          "matches its containing coroutine");
          invalidResumeRoute = true;
          return;
        }
        asyncResumeTargets.push_back(
            {key.first, key.second, function, point->getBlock()});
      });
    }
    if (invalidResumeRoute)
      return failure();
    llvm::sort(asyncResumeTargets, [](const AsyncResumeTarget &left,
                                      const AsyncResumeTarget &right) {
      return std::tie(left.frameId, left.resumePc) <
             std::tie(right.frameId, right.resumePc);
    });

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
    for (auto [jump, target] : asyncResumeRelocations) {
      auto found = globalBlockAddresses.find(target);
      if (found == globalBlockAddresses.end())
        return jump->emitError("async resume target was not lowered");
      jump->setAttr("target", builder.getI64IntegerAttr(
                                  static_cast<int64_t>(found->second)));
    }

    StorageSlotClassification storage = classifyStorageSlots();
    if (enforceCapacity && failed(verifyDataSlotCapacity(storage)))
      return failure();

    module->setAttr("udonasm.storage_report", storageReport(storage));

    SmallVector<Attribute> functionAddressMetadata;
    for (auto [index, function] : llvm::enumerate(functions)) {
      const FunctionPlan &plan = functionPlans.at(function.getOperation());
      if (!plan.entryAddress)
        return function.emitError("lowered Udon function has no entry address");
      uint64_t endAddress =
          index + 1 < functions.size()
              ? *functionPlans.at(functions[index + 1].getOperation())
                     .entryAddress
              : address;
      functionAddressMetadata.push_back(DictionaryAttr::get(
          module.getContext(),
          {builder.getNamedAttr("name",
                                builder.getStringAttr(function.getSymName())),
           builder.getNamedAttr("start",
                                builder.getI64IntegerAttr(*plan.entryAddress)),
           builder.getNamedAttr("end",
                                builder.getI64IntegerAttr(endAddress))}));
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
    module->setAttr(
        "udonasm.functions",
        ArrayAttr::get(module.getContext(), functionAddressMetadata));
    module->setAttr("udonasm.stage", builder.getStringAttr("scalar-v1"));
    module->removeAttr("vrc.host_stage");
    return success();
  }

private:
  uint64_t rawHeapWordCount() const {
    return llvm::divideCeil(rawHeapBytes, uint64_t{4});
  }

  unsigned dataSlotCapacity() const {
    if (auto capacity =
            module->getAttrOfType<IntegerAttr>("vrc.data_slot_capacity"))
      return static_cast<unsigned>(capacity.getInt());
    return 512;
  }

  struct StorageSlotClassification {
    llvm::StringSet<> fixed;
    llvm::StringSet<> scalarRegisters;
    llvm::StringSet<> asyncPersistent;
    llvm::StringSet<> arrays;
    llvm::StringSet<> constants;
    llvm::StringSet<> other;
    llvm::StringMap<unsigned> constantTypes;
    llvm::StringMap<unsigned> otherTypes;
  };

  StorageSlotClassification classifyStorageSlots() const {
    StorageSlotClassification result;
    llvm::StringSet<> fixedCandidates;
    llvm::StringSet<> registerCandidates;
    llvm::StringSet<> persistentCandidates;
    llvm::StringSet<> arrayCandidates;
    llvm::StringSet<> constantCandidates;
    for (const Slot &slot : slots)
      if (slot.isPublic)
        fixedCandidates.insert(slot.name);
    for (const auto &method : methodABIs) {
      for (const std::string &parameter : method.second.parameterSlots)
        fixedCandidates.insert(parameter);
      if (method.second.resultSlot)
        fixedCandidates.insert(*method.second.resultSlot);
      auto function = cast<func::FuncOp>(method.first);
      if (auto exportId = function->getAttrOfType<IntegerAttr>("vrc.export_id"))
        fixedCandidates.insert(udonclStatusSlot(exportId.getInt()));
    }
    if (!unityBufferPool.empty())
      fixedCandidates.insert("__udoncl_buffer_pool");
    for (const auto &slot : scalarRegisterSlots)
      registerCandidates.insert(slot.first());
    for (const auto &slot : asyncPersistentSlots)
      persistentCandidates.insert(slot.first());
    for (const GlobalBankPlan &bank : globalBanks)
      arrayCandidates.insert(bank.slot);
    for (const std::string &key : typedSpillBankOrder)
      arrayCandidates.insert(typedSpillBanks.at(key).slot);
    if (rawHeapBytes)
      arrayCandidates.insert(rawHeapSlot);
    for (const auto &slot : immutableSlots)
      constantCandidates.insert(slot.second);
    for (const auto &slot : sharedConstantSlots)
      constantCandidates.insert(slot.second);

    for (const Slot &slot : slots) {
      if (fixedCandidates.contains(slot.name))
        result.fixed.insert(slot.name);
      else if (registerCandidates.contains(slot.name))
        result.scalarRegisters.insert(slot.name);
      else if (persistentCandidates.contains(slot.name))
        result.asyncPersistent.insert(slot.name);
      else if (arrayCandidates.contains(slot.name))
        result.arrays.insert(slot.name);
      else if (constantCandidates.contains(slot.name)) {
        result.constants.insert(slot.name);
        ++result.constantTypes[slot.type];
      } else {
        result.other.insert(slot.name);
        ++result.otherTypes[slot.type];
      }
    }
    return result;
  }

  LogicalResult
  verifyDataSlotCapacity(const StorageSlotClassification &storage) {
    unsigned capacity = dataSlotCapacity();
    if (!capacity || capacity > 512)
      return module.emitError(
          "vrc.data_slot_capacity must be in the range 1..512");
    if (slots.size() <= capacity)
      return success();

    std::string detail;
    llvm::raw_string_ostream detailStream(detail);
    auto printTypes = [&](StringRef label,
                          const llvm::StringMap<unsigned> &types) {
      SmallVector<std::pair<std::string, unsigned>> ordered;
      for (const auto &entry : types)
        ordered.emplace_back(entry.first().str(), entry.second);
      llvm::sort(ordered);
      detailStream << "; " << label << "={";
      for (auto [index, entry] : llvm::enumerate(ordered)) {
        if (index)
          detailStream << ",";
        detailStream << entry.first << ":" << entry.second;
      }
      detailStream << "}";
    };
    printTypes("constant_types", storage.constantTypes);
    printTypes("other_types", storage.otherTypes);
    detailStream.flush();
    return module.emitError()
           << "final Udon data table requires " << slots.size() << " slots, "
           << "exceeding capacity " << capacity
           << "; fixed=" << storage.fixed.size()
           << ", scalar_registers=" << storage.scalarRegisters.size()
           << ", async_persistent=" << storage.asyncPersistent.size()
           << ", typed_arrays=" << storage.arrays.size()
           << ", constants=" << storage.constants.size()
           << ", other=" << storage.other.size() << detail
           << ". Reduce public exports/resources, scalar register pressure, "
              "cached constants, or raw-address scratch state";
  }

  void planMaterializedConstants(ArrayRef<func::FuncOp> functions) {
    auto stage = module->getAttrOfType<StringAttr>("vrc.async_stage");
    if (!stage || stage.getValue() != "udon-static-v1")
      return;
    llvm::DenseMap<uint32_t, ConstantPoolLocation> locations;
    llvm::DenseMap<uint32_t, unsigned> occurrences;
    SmallVector<std::pair<Operation *, uint32_t>> candidates;
    for (func::FuncOp function : functions)
      function.walk([&](arith::ConstantOp constant) {
        Type type = constant.getType();
        if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) {
          llvm::SmallString<32> text;
          floating.getValue().toString(text);
          size_t exponent = text.find_first_of("eE");
          if (exponent != llvm::StringRef::npos && exponent + 1 < text.size() &&
              text[exponent + 1] == '+')
            materializedConstants.insert(constant.getOperation());
          return;
        }
        auto integer = dyn_cast<IntegerType>(type);
        if (materializationLimit && integer && integer.getWidth() == 64) {
          materializedConstants.insert(constant.getOperation());
          return;
        }
        if (integer && integer.getWidth() == 32 &&
            cast<IntegerAttr>(constant.getValue())
                .getValue()
                .isMinSignedValue()) {
          materializedConstants.insert(constant.getOperation());
          return;
        }
        if (!materializationLimit ||
            ((!integer || integer.getWidth() != 32) && !type.isIndex()))
          return;
        uint32_t value = static_cast<uint32_t>(
            cast<IntegerAttr>(constant.getValue()).getValue().getZExtValue());
        ++occurrences[value];
        candidates.emplace_back(constant.getOperation(), value);
      });
    SmallVector<uint32_t> orderedValues;
    orderedValues.reserve(occurrences.size());
    for (const auto &entry : occurrences)
      orderedValues.push_back(entry.first);
    llvm::sort(orderedValues, [&](uint32_t left, uint32_t right) {
      if (occurrences.lookup(left) != occurrences.lookup(right))
        return occurrences.lookup(left) > occurrences.lookup(right);
      return left < right;
    });
    for (uint32_t value : orderedValues) {
      ConstantPoolLocation location;
      location.index = constantPoolValues.size();
      constantPoolValues.push_back(value);
      locations.try_emplace(value, location);
    }
    for (auto [operation, value] : candidates)
      pooledConstants.try_emplace(operation, locations.lookup(value));
    if (!constantPoolValues.empty()) {
      bool hadRawHeap = rawHeapBytes != 0;
      uint64_t base = llvm::alignTo(std::max<uint64_t>(rawHeapBytes, 4), 4);
      constantPoolWordBase = base / 4;
      uint64_t poolBytes = constantPoolValues.size() * uint64_t{4};
      if (poolBytes > INT32_MAX || base > INT32_MAX - poolBytes) {
        module.emitError("constant pool exceeds signed i32 raw heap capacity");
        return;
      }
      rawHeapBytes = base + poolBytes;
      rawObjects.push_back({"__vrc_constant_pool", "constant_pool",
                            "aot_initialized_i32_constants", base, poolBytes,
                            4});
      if (!hadRawHeap) {
        addSlot(rawHeapSlot, "SystemInt32Array", "null",
                externalRawHeapInitialization);
        addSlot(rawHeapLengthSlot, "SystemInt32",
                std::to_string(rawHeapWordCount()), false);
      } else {
        for (Slot &slot : slots)
          if (slot.name == rawHeapLengthSlot) {
            slot.initial = std::to_string(rawHeapWordCount());
            break;
          }
      }
    }
  }

  LogicalResult planDeclaredTypedSpillBanks() {
    auto include = [&](Operation *operation, StringRef storageKey, Type type,
                       int64_t index) -> LogicalResult {
      if (index < 0)
        return operation->emitError("typed spill index must be nonnegative");
      auto scalarType = udonType(type);
      if (failed(scalarType))
        return operation->emitError(
            "declared typed spill has no physical Udon type");
      auto bank = typedSpillBanks.find(storageKey);
      if (bank == typedSpillBanks.end()) {
        std::string slot = "__vrc_spill_bank_" + storageKey.str();
        std::string arrayType = scalarType->str() + "Array";
        addSlot(slot, arrayType, "null", false);
        bank = typedSpillBanks
                   .try_emplace(storageKey,
                                TypedSpillBankPlan{slot, scalarType->str(),
                                                   std::move(arrayType), 0})
                   .first;
        typedSpillBankOrder.push_back(storageKey.str());
      } else if (bank->second.scalarType != *scalarType) {
        return operation->emitError(
            "typed spill storage key has inconsistent physical types");
      }
      bank->second.length =
          std::max(bank->second.length, static_cast<unsigned>(index) + 1);
      return success();
    };

    WalkResult result = module.walk([&](Operation *operation) -> WalkResult {
      if (auto load = dyn_cast<host::SpillLoadOp>(operation))
        return failed(include(operation, load.getStorageKey(),
                              load.getValue().getType(), load.getIndex()))
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      if (auto store = dyn_cast<host::SpillStoreOp>(operation))
        return failed(include(operation, store.getStorageKey(),
                              store.getValue().getType(), store.getIndex()))
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      return WalkResult::advance();
    });
    return result.wasInterrupted() ? failure() : success();
  }

  LogicalResult
  planInterproceduralValueSlots(ArrayRef<func::FuncOp> functions) {
    unsigned registerBudget = 16;
    if (auto budget =
            module->getAttrOfType<IntegerAttr>("vrc.scalar_register_budget")) {
      if (budget.getInt() <= 0 || budget.getInt() > 256)
        return module.emitError(
            "vrc.scalar_register_budget must be in the range 1..256");
      registerBudget = static_cast<unsigned>(budget.getInt());
    }
    requestedScalarRegisterBudget = registerBudget;
    minimumScalarRegisterCount = registerBudget;
    llvm::DenseSet<Operation *> allocatableConstants(materializedConstants);
    for (const auto &entry : pooledConstants)
      allocatableConstants.insert(entry.first);
    auto analysis = analyzeHostStorageInterference(
        module, functions, [&](Type type) { return slotStorageKey(type); },
        allocatableConstants);
    if (failed(analysis))
      return failure();
    auto &functionValues = analysis->functionValues;
    auto &storageKeys = analysis->storageKeys;
    auto &ordinals = analysis->ordinals;
    auto &interference = analysis->interference;
    llvm::StringMap<SmallVector<Value>> valuesByStorage;
    for (const auto &entry : storageKeys)
      valuesByStorage[entry.second].push_back(entry.first);
    llvm::StringMap<unsigned> declaredSpillBases;
    for (const auto &entry : typedSpillBanks)
      declaredSpillBases[entry.first()] = entry.second.length;
    struct SpillCandidate {
      Value value;
      std::string storageKey;
      unsigned index;
    };
    SmallVector<SpillCandidate> spillCandidates;
    for (auto &entry : valuesByStorage) {
      SmallVector<Value> order(entry.getValue().begin(),
                               entry.getValue().end());
      llvm::sort(order, [&](Value left, Value right) {
        size_t leftDegree = interference[left].size();
        size_t rightDegree = interference[right].size();
        if (leftDegree != rightDegree)
          return leftDegree > rightDegree;
        return ordinals.lookup(left) < ordinals.lookup(right);
      });
      llvm::DenseMap<Value, unsigned> colors;
      for (Value value : order) {
        llvm::SmallSet<unsigned, 16> used;
        for (Value neighbor : interference[value]) {
          auto color = colors.find(neighbor);
          if (color != colors.end())
            used.insert(color->second);
        }
        unsigned color = 0;
        while (used.contains(color))
          ++color;
        colors.try_emplace(value, color);
        if (color >= registerBudget) {
          if (typedSpillsInserted) {
            // Reloads immediately before one operation and internal-call
            // argument copies form an unspillable simultaneous-use clique.
            // Grow only to that proven minimum; long-lived values were
            // already moved to typed arrays by the first allocation round.
            minimumScalarRegisterCount =
                std::max(minimumScalarRegisterCount, color + 1);
          } else {
            spillCandidates.push_back(
                {value, entry.getKey().str(),
                 declaredSpillBases.lookup(entry.getKey()) + color -
                     registerBudget});
            continue;
          }
        }
        interproceduralValueSlots.try_emplace(
            value, "__vrc_shared_r_" + entry.getKey().str() + "_" +
                       std::to_string(color));
      }
    }
    if (!spillCandidates.empty()) {
      module.getContext()->getOrLoadDialect<host::UdonHLDialect>();
      llvm::sort(spillCandidates, [&](const SpillCandidate &left,
                                      const SpillCandidate &right) {
        return ordinals.lookup(left.value) < ordinals.lookup(right.value);
      });
      for (const SpillCandidate &candidate : spillCandidates) {
        unsigned spillIndex = candidate.index;
        auto scalarType = udonType(candidate.value.getType());
        if (failed(scalarType))
          return emitError(candidate.value.getLoc())
                 << "typed spill has no physical Udon type";
        auto bank = typedSpillBanks.find(candidate.storageKey);
        if (bank == typedSpillBanks.end()) {
          std::string slot = "__vrc_spill_bank_" + candidate.storageKey;
          std::string arrayType = scalarType->str() + "Array";
          addSlot(slot, arrayType, "null", false);
          bank = typedSpillBanks
                     .try_emplace(candidate.storageKey,
                                  TypedSpillBankPlan{slot, scalarType->str(),
                                                     std::move(arrayType), 0})
                     .first;
          typedSpillBankOrder.push_back(candidate.storageKey);
        }
        bank->second.length = std::max(bank->second.length, spillIndex + 1);

        SmallVector<OpOperand *> uses;
        for (OpOperand &use : candidate.value.getUses())
          uses.push_back(&use);
        llvm::DenseMap<Operation *, Value> reloads;
        for (OpOperand *use : uses) {
          Operation *owner = use->getOwner();
          auto reload = reloads.find(owner);
          Value loaded;
          if (reload == reloads.end()) {
            builder.setInsertionPoint(owner);
            loaded = builder
                         .create<host::SpillLoadOp>(
                             owner->getLoc(), candidate.value.getType(),
                             builder.getStringAttr(candidate.storageKey),
                             builder.getI32IntegerAttr(spillIndex))
                         .getValue();
            reloads.try_emplace(owner, loaded);
          } else {
            loaded = reload->second;
          }
          use->set(loaded);
        }
        if (auto argument = dyn_cast<BlockArgument>(candidate.value)) {
          builder.setInsertionPointToStart(argument.getOwner());
        } else {
          Operation *definition = candidate.value.getDefiningOp();
          builder.setInsertionPointAfter(definition);
        }
        builder.create<host::SpillStoreOp>(
            candidate.value.getLoc(), candidate.value,
            builder.getStringAttr(candidate.storageKey),
            builder.getI32IntegerAttr(spillIndex));
      }
      typedSpillsInserted = true;
      interproceduralValueSlots.clear();
      builder.setInsertionPointToEnd(module.getBody());
      return planInterproceduralValueSlots(functions);
    }
    return success();
  }

  LogicalResult planUnityBufferPool() {
    auto pool = module->getAttrOfType<ArrayAttr>("vrc.unity_buffer_pool");
    if (!pool)
      return success();
    llvm::SmallSet<int64_t, 8> ids;
    llvm::SmallSet<std::string, 8> slots;
    for (Attribute attribute : pool) {
      auto entry = dyn_cast<DictionaryAttr>(attribute);
      auto id = entry ? entry.getAs<IntegerAttr>("id") : IntegerAttr();
      auto logicalBytes =
          entry ? entry.getAs<IntegerAttr>("logical_bytes") : IntegerAttr();
      auto heapSlot =
          entry ? entry.getAs<StringAttr>("heap_slot") : StringAttr();
      if (!id || id.getInt() <= 0 || !logicalBytes ||
          logicalBytes.getInt() < 0 || !heapSlot ||
          heapSlot.getValue().empty() || !ids.insert(id.getInt()).second ||
          !slots.insert(heapSlot.getValue().str()).second)
        return module.emitError("invalid Unity buffer pool metadata");
      unityBufferPool.push_back(
          {id.getInt(), logicalBytes.getInt(), heapSlot.getValue().str(),
           static_cast<unsigned>(unityBufferPool.size())});
    }
    addSlot("__udoncl_buffer_pool", "UnityEngineRenderTextureArray", "null",
            true);
    struct Candidate {
      int64_t id;
      bool commonlyShared;
      unsigned nonzeroBytes;
      unsigned shiftedBytes;
    };
    SmallVector<Candidate> candidates;
    candidates.reserve(unityBufferPool.size());
    for (const UnityBufferPoolEntry &entry : unityBufferPool) {
      unsigned nonzeroBytes = 0;
      unsigned shiftedBytes = 0;
      for (unsigned byte = 0; byte < 4; ++byte)
        if ((static_cast<uint32_t>(entry.id) >> (byte * 8)) & 255) {
          ++nonzeroBytes;
          shiftedBytes += byte != 0;
        }
      candidates.push_back(
          {entry.id, entry.id <= 15, nonzeroBytes, shiftedBytes});
    }
    llvm::sort(candidates, [](const Candidate &left, const Candidate &right) {
      if (left.commonlyShared != right.commonlyShared)
        return !left.commonlyShared;
      if (left.nonzeroBytes != right.nonzeroBytes)
        return left.nonzeroBytes < right.nonzeroBytes;
      if (left.shiftedBytes != right.shiftedBytes)
        return left.shiftedBytes < right.shiftedBytes;
      return left.id > right.id;
    });
    for (const Candidate &candidate :
         llvm::ArrayRef(candidates)
             .take_front(std::min<size_t>(unityBufferIdRematerializationCount,
                                          candidates.size())))
      rematerializedUnityBufferIds.insert(candidate.id);
    return success();
  }

  std::string rawFallbackReason(Value root, bool aggregateLayout) const {
    bool volatileAccess = false;
    bool pointerIntegerRoundTrip = false;
    bool byteOrBulkMemory = false;
    bool dynamicIndex = false;
    bool pointerEscape = false;
    SmallVector<Value> worklist{root};
    llvm::DenseSet<Value> visited;
    while (!worklist.empty()) {
      Value pointer = worklist.pop_back_val();
      if (!visited.insert(pointer).second)
        continue;
      for (OpOperand &use : pointer.getUses()) {
        Operation *owner = use.getOwner();
        if (auto load = dyn_cast<host::LoadOp>(owner)) {
          volatileAccess |= bool(load.getVolatile_());
          continue;
        }
        if (auto store = dyn_cast<host::StoreOp>(owner)) {
          volatileAccess |= bool(store.getVolatile_());
          continue;
        }
        if (isa<host::PtrToIntOp, host::IntToPtrOp>(owner)) {
          pointerIntegerRoundTrip = true;
          continue;
        }
        if (isa<host::MemcpyOp, host::MemmoveOp, host::MemsetOp>(owner)) {
          byteOrBulkMemory = true;
          continue;
        }
        if (auto add = dyn_cast<host::PtrAddOp>(owner)) {
          if (!add.getByteOffset().getDefiningOp<arith::ConstantOp>())
            dynamicIndex = true;
          worklist.push_back(add.getResult());
          continue;
        }
        if (isa<host::LifetimeStartOp, host::LifetimeEndOp>(owner))
          continue;
        pointerEscape = true;
      }
    }
    if (volatileAccess)
      return "volatile_access";
    if (pointerIntegerRoundTrip)
      return "pointer_integer_round_trip";
    if (byteOrBulkMemory)
      return "byte_or_bulk_memory";
    if (dynamicIndex)
      return "dynamic_index";
    if (pointerEscape)
      return "pointer_escape_or_alias";
    if (aggregateLayout)
      return "aggregate_byte_layout";
    return "address_observable_after_promotion";
  }

  DictionaryAttr storageReport(const StorageSlotClassification &storage) {
    auto entry = [&](ArrayRef<NamedAttribute> fields) {
      return DictionaryAttr::get(module.getContext(), fields);
    };
    auto stringField = [&](StringRef name, StringRef value) {
      return builder.getNamedAttr(name, builder.getStringAttr(value));
    };
    auto integerField = [&](StringRef name, uint64_t value) {
      return builder.getNamedAttr(
          name, builder.getI64IntegerAttr(static_cast<int64_t>(value)));
    };

    SmallVector<Attribute> fixed;
    for (const Slot &slot : slots) {
      if (!storage.fixed.contains(slot.name))
        continue;
      SmallVector<NamedAttribute> fields{stringField("name", slot.name),
                                         stringField("type", slot.type)};
      if (slot.isPublic)
        fields.push_back(builder.getNamedAttr("public", builder.getUnitAttr()));
      fixed.push_back(entry(fields));
    }

    SmallVector<std::pair<std::string, std::string>> scalarSlots;
    scalarSlots.reserve(scalarRegisterSlots.size());
    for (const auto &slot : scalarRegisterSlots)
      scalarSlots.emplace_back(slot.first().str(), slot.second);
    llvm::sort(scalarSlots);
    SmallVector<Attribute> scalars;
    llvm::StringMap<uint64_t> scalarCountsByType;
    for (const auto &[name, type] : scalarSlots)
      scalars.push_back(
          entry({stringField("name", name), stringField("type", type)})),
          ++scalarCountsByType[type];
    SmallVector<std::pair<std::string, uint64_t>> orderedScalarCounts;
    for (const auto &count : scalarCountsByType)
      orderedScalarCounts.emplace_back(count.first().str(), count.second);
    llvm::sort(orderedScalarCounts);
    SmallVector<Attribute> registerBanks;
    for (const auto &[type, count] : orderedScalarCounts)
      registerBanks.push_back(
          entry({stringField("type", type), integerField("count", count)}));

    SmallVector<std::pair<std::string, std::string>> persistentSlots;
    persistentSlots.reserve(asyncPersistentSlots.size());
    for (const auto &slot : asyncPersistentSlots)
      persistentSlots.emplace_back(slot.first().str(), slot.second);
    llvm::sort(persistentSlots);
    SmallVector<Attribute> persistent;
    for (const auto &[name, type] : persistentSlots)
      persistent.push_back(
          entry({stringField("name", name), stringField("type", type)}));

    SmallVector<Attribute> spillBanks;
    for (const std::string &key : typedSpillBankOrder) {
      const TypedSpillBankPlan &bank = typedSpillBanks.at(key);
      spillBanks.push_back(entry({stringField("storage_key", key),
                                  stringField("name", bank.slot),
                                  stringField("scalar_type", bank.scalarType),
                                  stringField("array_type", bank.arrayType),
                                  integerField("length", bank.length)}));
    }

    SmallVector<Attribute> globalBankAttributes;
    for (const GlobalBankPlan &bank : globalBanks)
      globalBankAttributes.push_back(
          entry({stringField("name", bank.slot),
                 stringField("element_type", bank.elementType.isF32()
                                                 ? "SystemSingle"
                                                 : "SystemInt32"),
                 integerField("length", bank.length)}));

    SmallVector<RawObjectPlan> raw(rawObjects.begin(), rawObjects.end());
    llvm::sort(raw, [](const RawObjectPlan &left, const RawObjectPlan &right) {
      return std::tie(left.base, left.name) < std::tie(right.base, right.name);
    });
    SmallVector<Attribute> rawAttributes;
    for (const RawObjectPlan &object : raw)
      rawAttributes.push_back(entry(
          {stringField("name", object.name), stringField("kind", object.kind),
           stringField("reason", object.reason),
           integerField("base", object.base),
           integerField("bytes", object.bytes),
           integerField("alignment", object.alignment)}));

    auto classifiedSlots = [&](const llvm::StringSet<> &names) {
      SmallVector<std::pair<std::string, std::string>> ordered;
      for (const Slot &slot : slots)
        if (names.contains(slot.name))
          ordered.emplace_back(slot.name, slot.type);
      llvm::sort(ordered);
      SmallVector<Attribute> attributes;
      attributes.reserve(ordered.size());
      for (const auto &[name, type] : ordered)
        attributes.push_back(
            entry({stringField("name", name), stringField("type", type)}));
      return builder.getArrayAttr(attributes);
    };
    auto typeCounts = [&](const llvm::StringMap<unsigned> &counts) {
      SmallVector<std::pair<std::string, unsigned>> ordered;
      for (const auto &count : counts)
        ordered.emplace_back(count.first().str(), count.second);
      llvm::sort(ordered);
      SmallVector<Attribute> attributes;
      attributes.reserve(ordered.size());
      for (const auto &[type, count] : ordered)
        attributes.push_back(
            entry({stringField("type", type), integerField("count", count)}));
      return builder.getArrayAttr(attributes);
    };
    DictionaryAttr categoryCounts =
        entry({integerField("fixed", storage.fixed.size()),
               integerField("scalar_register", storage.scalarRegisters.size()),
               integerField("async_persistent", storage.asyncPersistent.size()),
               integerField("array", storage.arrays.size()),
               integerField("constant", storage.constants.size()),
               integerField("other", storage.other.size())});
    SmallVector<int64_t> orderedRematerializedIds;
    for (int64_t id : rematerializedUnityBufferIds)
      orderedRematerializedIds.push_back(id);
    llvm::sort(orderedRematerializedIds);
    SmallVector<Attribute> rematerializedIds;
    for (int64_t id : orderedRematerializedIds)
      rematerializedIds.push_back(builder.getI64IntegerAttr(id));
    SmallVector<std::pair<std::string, unsigned>> orderedCachedSpillIndices;
    for (const auto &cached : cachedSpillIndexSlots) {
      StringRef key = cached.first();
      size_t separator = key.rfind('\n');
      uint64_t index = 0;
      if (separator == StringRef::npos ||
          key.drop_front(separator + 1).getAsInteger(10, index))
        continue;
      orderedCachedSpillIndices.emplace_back(key.take_front(separator).str(),
                                             static_cast<unsigned>(index));
    }
    llvm::sort(orderedCachedSpillIndices);
    SmallVector<Attribute> cachedSpillIndexEntries;
    for (const auto &[storageKey, index] : orderedCachedSpillIndices)
      cachedSpillIndexEntries.push_back(
          entry({stringField("storage_key", storageKey),
                 integerField("index", index)}));
    int64_t finalDataSlotReserve = 0;
    if (auto reserve =
            module->getAttrOfType<IntegerAttr>("vrc.final_data_slot_reserve"))
      finalDataSlotReserve = reserve.getInt();

    return entry(
        {integerField("version", 2),
         integerField("data_slot_count", slots.size()),
         integerField("data_slot_capacity", dataSlotCapacity()),
         integerField("final_data_slot_reserve", finalDataSlotReserve),
         integerField("scalar_register_budget", requestedScalarRegisterBudget),
         integerField("minimum_simultaneous_registers",
                      minimumScalarRegisterCount),
         integerField("address_cache_budget", rawAddressCacheLimit),
         integerField("address_cache_loop_weight_base", 8),
         integerField("address_cache_maximum_loop_weight", 4096),
         integerField("cached_spill_indices", cachedSpillIndexSlots.size()),
         builder.getNamedAttr("cached_spill_index_entries",
                              builder.getArrayAttr(cachedSpillIndexEntries)),
         integerField("synthesized_spill_indices",
                      plannedSpillIndexCount - cachedSpillIndexSlots.size()),
         integerField("cached_raw_addresses", cachedRawGlobalSlots.size()),
         integerField("cached_return_addresses",
                      cachedReturnAddressSlots.size()),
         integerField("unity_buffer_id_direct_count",
                      unityBufferPool.size() -
                          rematerializedUnityBufferIds.size()),
         integerField("unity_buffer_id_rematerialized_count",
                      rematerializedUnityBufferIds.size()),
         integerField("unity_buffer_id_rematerialization_address_delta",
                      unityBufferIdRematerializationAddressDelta),
         integerField("unity_buffer_id_rematerialization_attempts",
                      unityBufferIdRematerializationAttempts),
         integerField("unity_buffer_id_requested_rematerialization_count",
                      unityBufferIdRematerializationCount),
         builder.getNamedAttr("unity_buffer_id_rematerialized_ids",
                              builder.getArrayAttr(rematerializedIds)),
         builder.getNamedAttr("fixed_named_slots", builder.getArrayAttr(fixed)),
         builder.getNamedAttr("scalar_registers",
                              builder.getArrayAttr(scalars)),
         builder.getNamedAttr("typed_register_banks",
                              builder.getArrayAttr(registerBanks)),
         builder.getNamedAttr("async_persistent_values",
                              builder.getArrayAttr(persistent)),
         builder.getNamedAttr("typed_spill_arrays",
                              builder.getArrayAttr(spillBanks)),
         builder.getNamedAttr("typed_global_arrays",
                              builder.getArrayAttr(globalBankAttributes)),
         builder.getNamedAttr("slot_category_counts", categoryCounts),
         builder.getNamedAttr("constant_slots",
                              classifiedSlots(storage.constants)),
         builder.getNamedAttr("other_slots", classifiedSlots(storage.other)),
         builder.getNamedAttr("constant_slot_types",
                              typeCounts(storage.constantTypes)),
         builder.getNamedAttr("other_slot_types",
                              typeCounts(storage.otherTypes)),
         builder.getNamedAttr("raw_objects",
                              builder.getArrayAttr(rawAttributes)),
         integerField("raw_heap_bytes", rawHeapBytes),
         integerField("raw_heap_words", rawHeapWordCount())});
  }

  LogicalResult planRawMemory() {
    uint64_t cursor = 4; // Keep the null pointer unmapped.
    llvm::StringMap<host::AddressOfOp> representativeAddresses;
    module.walk([&](host::AddressOfOp address) {
      representativeAddresses.try_emplace(address.getGlobal(), address);
    });
    for (host::GlobalOp global : module.getOps<host::GlobalOp>()) {
      if (global.getAddressSpace() != 0 || global.getLinkage() != "internal")
        return global.emitError(
            "raw Udon globals currently require internal address-space zero");
      Type type = global.getStorageType();
      auto bytes =
          dyn_cast_or_null<DenseI8ArrayAttr>(global.getInitialValueAttr());
      auto integer = dyn_cast<IntegerType>(type);
      bool scalar = (integer &&
                     (integer.getWidth() == 1 ||
                      (integer.getWidth() >= 8 && integer.getWidth() <= 64))) ||
                    type.isF32() || isa<host::PtrType>(type);
      if (!scalar && !bytes)
        return global.emitError() << "raw Udon global '" << global.getSymName()
                                  << "' with storage type " << type
                                  << " requires a serialized byte initializer";
      uint64_t alignment = static_cast<uint64_t>(global.getAlignment());
      cursor = llvm::alignTo(cursor, alignment);
      uint64_t scalarBytes = 4;
      if (integer)
        scalarBytes = llvm::divideCeil(integer.getWidth(), 8u);
      uint64_t objectBytes = bytes ? bytes.size() : scalarBytes;
      rawGlobalBases.try_emplace(global.getSymName(), cursor);
      host::AddressOfOp representative =
          representativeAddresses.lookup(global.getSymName());
      rawObjects.push_back(
          {global.getSymName().str(), "global",
           representative
               ? rawFallbackReason(representative.getResult(), !scalar)
               : (!scalar ? "aggregate_byte_layout"
                          : "address_observable_after_promotion"),
           cursor, objectBytes, alignment});
      cursor += objectBytes;
      Attribute initial = global.getInitialValueAttr();
      SmallVector<int8_t> serialized(objectBytes, 0);
      if (bytes) {
        llvm::copy(bytes.asArrayRef(), serialized.begin());
      } else if (auto value = dyn_cast_or_null<IntegerAttr>(initial)) {
        APInt bits = value.getValue();
        if (isa<host::PtrType>(type)) {
          if (!bits.isZero())
            return global.emitError(
                "raw pointer initializer requires a null value");
        } else if (!integer || bits.getBitWidth() != integer.getWidth()) {
          return global.emitError("raw integer initializer type mismatch");
        } else {
          uint64_t raw = bits.getZExtValue();
          for (uint64_t byte = 0; byte < objectBytes; ++byte)
            serialized[byte] = static_cast<int8_t>((raw >> (byte * 8)) & 0xff);
        }
      } else if (auto value = dyn_cast_or_null<FloatAttr>(initial)) {
        if (!type.isF32() || !value.getType().isF32())
          return global.emitError(
              "raw floating initializer currently requires f32 storage");
        APInt bits = value.getValue().bitcastToAPInt();
        for (uint64_t byte = 0; byte < objectBytes; ++byte)
          serialized[byte] = static_cast<int8_t>(
              bits.extractBitsAsZExtValue(8, static_cast<unsigned>(byte * 8)));
      } else if (initial && !isa<UnitAttr>(initial)) {
        return global.emitError()
               << "raw global initializer is not serializable for @"
               << global.getSymName() << " with storage type " << type;
      }
      if (llvm::any_of(serialized, [](int8_t byte) { return byte != 0; }))
        rawGlobalInitializers.try_emplace(
            global.getSymName(),
            DenseI8ArrayAttr::get(module.getContext(), serialized));
    }
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      unsigned allocaIndex = 0;
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
        rawObjects.push_back(
            {(function.getSymName() + ":alloca" + std::to_string(allocaIndex++))
                 .str(),
             "alloca", rawFallbackReason(allocation.getResult(), false), cursor,
             static_cast<uint64_t>(bytes.getInt()), alignment});
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
      addSlot(rawHeapLengthSlot, "SystemInt32",
              std::to_string(rawHeapWordCount()), false);
    }
    llvm::StringMap<unsigned> addressOccurrences;
    module.walk([&](host::AddressOfOp address) {
      if (rawGlobalBases.contains(address.getGlobal()))
        ++addressOccurrences[address.getGlobal()];
    });
    SmallVector<std::pair<std::string, unsigned>> orderedAddresses;
    orderedAddresses.reserve(addressOccurrences.size());
    for (const auto &entry : addressOccurrences)
      orderedAddresses.emplace_back(entry.first().str(), entry.second);
    llvm::sort(orderedAddresses, [](const auto &left, const auto &right) {
      if (left.second != right.second)
        return left.second > right.second;
      return left.first < right.first;
    });
    rawAddressCandidates = std::move(orderedAddresses);
    return success();
  }

  static std::string spillIndexCacheKey(StringRef storageKey, unsigned index) {
    return (storageKey + "\n" + std::to_string(index)).str();
  }

  void compactSpillIndices() {
    llvm::StringMap<llvm::DenseMap<unsigned, unsigned>> occurrences;
    module.walk([&](Operation *operation) {
      StringRef storageKey;
      int64_t index = -1;
      if (auto load = dyn_cast<host::SpillLoadOp>(operation)) {
        storageKey = load.getStorageKey();
        index = load.getIndex();
      } else if (auto store = dyn_cast<host::SpillStoreOp>(operation)) {
        storageKey = store.getStorageKey();
        index = store.getIndex();
      }
      if (index >= 0)
        ++occurrences[storageKey][static_cast<unsigned>(index)];
    });
    llvm::StringMap<llvm::DenseMap<unsigned, unsigned>> remapped;
    for (auto &storage : occurrences) {
      SmallVector<unsigned> indices;
      indices.reserve(storage.second.size());
      for (const auto &entry : storage.second)
        indices.push_back(entry.first);
      llvm::sort(indices, [&](unsigned left, unsigned right) {
        unsigned leftCount = storage.second.lookup(left);
        unsigned rightCount = storage.second.lookup(right);
        if (leftCount != rightCount)
          return leftCount > rightCount;
        return left < right;
      });
      for (auto [newIndex, oldIndex] : llvm::enumerate(indices))
        remapped[storage.first()][oldIndex] = newIndex;
    }
    module.walk([&](Operation *operation) {
      if (auto load = dyn_cast<host::SpillLoadOp>(operation)) {
        unsigned index = static_cast<unsigned>(load.getIndex());
        load->setAttr("index",
                      builder.getI32IntegerAttr(
                          remapped[load.getStorageKey()].lookup(index)));
      } else if (auto store = dyn_cast<host::SpillStoreOp>(operation)) {
        unsigned index = static_cast<unsigned>(store.getIndex());
        store->setAttr("index",
                       builder.getI32IntegerAttr(
                           remapped[store.getStorageKey()].lookup(index)));
      }
    });
  }

  void planAddressCaches() {
    constexpr uint64_t loopWeightBase = 8;
    constexpr uint64_t maximumLoopWeight = 4096;
    enum class Kind { Spill, RawAddress, ReturnAddress };
    struct Candidate {
      Kind kind;
      std::string id;
      uint64_t value;
      uint64_t score;
      Operation *call = nullptr;
    };
    SmallVector<Candidate> candidates;
    auto literalSavings = [](uint64_t value) {
      unsigned highestNibble = 0;
      for (unsigned nibble = 1; nibble < 8; ++nibble)
        if ((value >> (nibble * 4)) & 15)
          highestNibble = nibble;
      unsigned nonzeroLowerNibbles = 0;
      for (unsigned nibble = 0; nibble < highestNibble; ++nibble)
        nonzeroLowerNibbles += ((value >> (nibble * 4)) & 15) != 0;
      // emitI32Literal copies the highest nibble, then uses one 32-byte shift
      // per lower nibble and one 32-byte OR for each nonzero lower nibble. A
      // cached constant replaces that whole tail with the same 20-byte COPY.
      return (highestNibble + nonzeroLowerNibbles) * 32;
    };
    for (const auto &[symbol, occurrences] : rawAddressCandidates)
      candidates.push_back(
          {Kind::RawAddress, symbol, rawGlobalBases.at(symbol),
           occurrences * literalSavings(rawGlobalBases.at(symbol))});

    llvm::DenseMap<Block *, uint64_t> blockWeights;
    DominanceInfo dominance(module);
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      Region &body = function.getBody();
      if (body.hasOneBlock()) {
        blockWeights.try_emplace(&body.front(), 1);
        continue;
      }
      CFGLoopInfo loops(dominance.getDomTree(&body));
      for (Block &block : body) {
        uint64_t weight = 1;
        for (unsigned depth = loops.getLoopDepth(&block); depth > 0; --depth)
          weight = std::min(maximumLoopWeight, weight * loopWeightBase);
        blockWeights.try_emplace(&block, weight);
      }
    }
    llvm::StringMap<uint64_t> spillOccurrences;
    module.walk([&](Operation *operation) {
      StringRef storageKey;
      int64_t index = -1;
      if (auto load = dyn_cast<host::SpillLoadOp>(operation)) {
        storageKey = load.getStorageKey();
        index = load.getIndex();
      } else if (auto store = dyn_cast<host::SpillStoreOp>(operation)) {
        storageKey = store.getStorageKey();
        index = store.getIndex();
      }
      if (index >= 0) {
        uint64_t weight = blockWeights.lookup(operation->getBlock());
        spillOccurrences[spillIndexCacheKey(storageKey,
                                            static_cast<unsigned>(index))] +=
            std::max<uint64_t>(1, weight);
      }
    });
    for (const auto &entry : spillOccurrences) {
      StringRef key = entry.first();
      size_t separator = key.rfind('\n');
      uint64_t index = 0;
      if (separator == StringRef::npos ||
          key.drop_front(separator + 1).getAsInteger(10, index))
        continue;
      candidates.push_back(
          {Kind::Spill, key.str(), index,
           entry.second * static_cast<uint64_t>(literalSavings(index))});
    }
    plannedSpillIndexCount = spillOccurrences.size();
    unsigned callOrdinal = 0;
    module.walk([&](Operation *operation) {
      if (!isa<func::CallOp, host::CallOp>(operation))
        return;
      candidates.push_back({Kind::ReturnAddress, std::to_string(callOrdinal++),
                            0, 536, operation});
    });
    llvm::sort(candidates, [](const Candidate &left, const Candidate &right) {
      if (left.score != right.score)
        return left.score > right.score;
      if (left.kind != right.kind)
        return left.kind < right.kind;
      return std::tie(left.value, left.id) < std::tie(right.value, right.id);
    });

    unsigned addedSlots = 0;
    for (const Candidate &candidate : candidates) {
      if (candidate.kind == Kind::ReturnAddress) {
        if (addedSlots == rawAddressCacheLimit)
          continue;
        std::string slot = "__vrc_cached_return_address_" + candidate.id;
        cachedReturnAddressSlots.try_emplace(candidate.call, std::move(slot));
        ++addedSlots;
        continue;
      }
      std::string constantKey =
          "SystemInt32\n" + std::to_string(candidate.value);
      // A small value is free only when a previous lowering decision already
      // materialized it. Caching the value can remove the literal path that
      // would otherwise have created its nibble slot, so 0..15 cannot be
      // assumed to exist merely because the generic materializer can use it.
      bool exists = immutableSlots.contains(constantKey);
      if (!exists && addedSlots == rawAddressCacheLimit)
        continue;
      std::string slot =
          immutableSlot("SystemInt32", std::to_string(candidate.value));
      if (!exists)
        ++addedSlots;
      if (candidate.kind == Kind::Spill) {
        cachedSpillIndexSlots.try_emplace(candidate.id, slot);
      } else {
        cachedRawGlobalSlots.try_emplace(candidate.id, slot);
        cachedRawGlobalBases.emplace_back(candidate.value, std::move(slot));
      }
    }
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
    } else if (function->hasAttr("vrc.compiler_async_completion")) {
      auto event = function->getAttrOfType<StringAttr>("vrc.udon_event");
      SmallVector<NamedAttribute> exportFields;
      exportFields.push_back(builder.getNamedAttr("name", event));
      exportFields.push_back(builder.getNamedAttr(
          "address", builder.getI64IntegerAttr(static_cast<int64_t>(address))));
      exportFields.push_back(builder.getNamedAttr(
          "function", builder.getStringAttr(function.getSymName())));
      exports.push_back(DictionaryAttr::get(module.getContext(), exportFields));
    }

    llvm::DenseMap<Value, std::string> valueSlots;
    llvm::DenseMap<Value, uint64_t> globalBaseOffsets;
    llvm::DenseMap<Value, unsigned> valueColors;
    llvm::DenseMap<Value, unsigned> valueOrdinals;
    llvm::DenseMap<Value, llvm::DenseSet<Value>> interference;
    llvm::DenseMap<Value, std::string> valueUdonTypes;
    llvm::DenseMap<Value, unsigned> descriptorColors;
    llvm::DenseMap<Value, unsigned> descriptorOrdinals;
    llvm::DenseMap<Value, llvm::DenseSet<Value>> descriptorInterference;
    SmallVector<Value> colorableValues;
    SmallVector<Value> descriptors;
    auto isColorableResult = [&](Operation &operation) {
      return (!isa<arith::ConstantOp>(operation) ||
              materializedConstants.contains(&operation)) &&
             !isa<memref::GetGlobalOp, ::vrc::vrc::DispatchDescCreateOp,
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
      for (Value operand : operation->getOperands())
        if (valueOrdinals.contains(operand))
          live.push_back(operand);
      for (Value result : operation->getResults())
        if (valueOrdinals.contains(result))
          live.push_back(result);
      llvm::sort(live, [&](Value left, Value right) {
        return valueOrdinals.lookup(left) < valueOrdinals.lookup(right);
      });
      live.erase(std::unique(live.begin(), live.end()), live.end());
      for (auto [leftIndex, left] : llvm::enumerate(live))
        for (Value right : ArrayRef<Value>(live).drop_front(leftIndex + 1))
          if (valueUdonTypes.lookup(left) == valueUdonTypes.lookup(right)) {
            interference[left].insert(right);
            interference[right].insert(left);
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
          descriptorInterference[left].insert(right);
          descriptorInterference[right].insert(left);
        }
    });
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
      auto storageKey = slotStorageKey(value.getType());
      if (failed(type) || failed(storageKey))
        return {};
      auto shared = interproceduralValueSlots.find(value);
      std::string name = shared != interproceduralValueSlots.end()
                             ? shared->second
                             : prefix + "r_" + *storageKey + "_" +
                                   std::to_string(valueColors.lookup(value));
      addSlot(name, *type, "null", false);
      scalarRegisterSlots.try_emplace(name, type->str());
      return name;
    };
    llvm::DenseMap<Operation *, SmallVector<std::string>> frameSlots;
    llvm::DenseMap<Operation *, SmallVector<std::string>> descriptorSlots;
    llvm::DenseMap<Operation *, std::string> allocaLengthSlots;
    for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
      auto type = *udonType(argument.getType());
      std::string name = plan.parameterSlots[index];
      bool publicSlot = function->hasAttr("vrc.compiler_async_completion");
      addSlot(name, type, "null", publicSlot);
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
        emitCopy(function.getLoc(), immutableSlot("SystemInt32", "0"),
                 asyncResumeStateSlot);
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
          globalBaseOffsets.try_emplace(getGlobal.getResult(),
                                        found->second.base);
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
          std::string output = coloredSlot(cast.getOutputs().front());
          if (output.empty())
            return cast.emitError(
                "Udon token conversion has no colorable output slot");
          // A conversion is an SSA definition, even when both sides share the
          // same physical Udon type. Aliasing the result to the input's slot
          // shortens the result's effective live range to that of the input.
          // A later definition may then overwrite the slot before the result
          // is consumed by a multi-operand operation. Keep the result in its
          // own colored value; graph coloring can coalesce it only after the
          // combined live range is represented explicitly.
          valueSlots[cast.getOutputs().front()] = output;
          continue;
        }
        if (auto allocation = dyn_cast<::vrc::vrc::BufferAllocOp>(operation)) {
          std::string logicalId = valueSlots.lookup(allocation.getLogicalId());
          if (logicalId.empty())
            return allocation.emitError(
                "Unity buffer allocation requires a logical-id slot");
          valueSlots[allocation.getBuffer()] =
              coloredSlot(allocation.getBuffer());
          valueSlots[allocation.getStatus()] =
              coloredSlot(allocation.getStatus());
          continue;
        }
        if (auto staticBuffer =
                dyn_cast<::vrc::vrc::StaticBufferOp>(operation)) {
          addSlot("__udoncl_static_tensors", "UnityEngineTexture2DArray",
                  "null", true);
          valueSlots[staticBuffer.getBuffer()] =
              coloredSlot(staticBuffer.getBuffer());
          continue;
        }
        if (auto assignedBuffer =
                dyn_cast<::vrc::vrc::AssignedBufferOp>(operation)) {
          valueSlots[assignedBuffer.getBuffer()] =
              coloredSlot(assignedBuffer.getBuffer());
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
            asyncPersistentSlots.try_emplace(slotName, udon->str());
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
            // The Worlds SDK assembler does not accept scientific notation.
            // Reformat every finite exponent spelling as an equivalent
            // fixed-decimal literal, including small values with a negative
            // exponent.
            size_t exponent = initial.find_first_of("eE");
            if (exponent != std::string::npos) {
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
          if (materializedConstants.contains(&operation) ||
              pooledConstants.contains(&operation)) {
            name = coloredSlot(operation.getResult(0));
            valueSlots[operation.getResult(0)] = name;
            continue;
          }
          Type constantType = operation.getResult(0).getType();
          if (!constantType.isInteger(1) && !constantType.isInteger(64)) {
            name = immutableSlot(*type, initial);
            valueSlots[operation.getResult(0)] = name;
            continue;
          }
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

    if ((exportId || function->hasAttr("vrc.compiler_async_completion")) &&
        (!globalPlans.empty() || !typedSpillBanks.empty() ||
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
        std::string scratch = cycle.type == "SystemInt32"
                                  ? "__vrc_return_address_i32"
                                  : "__vrc_parallel_copy_scratch_" + cycle.type;
        addSlot(scratch, cycle.type, cycle.type == "SystemInt32" ? "0" : "null",
                false);
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
    auto materializeRawGlobalAddress = [&](Value value,
                                           Location location) -> LogicalResult {
      auto addressOf = value.getDefiningOp<host::AddressOfOp>();
      if (!addressOf)
        return success();
      auto found = rawGlobalBases.find(addressOf.getGlobal());
      if (found == rawGlobalBases.end())
        return addressOf.emitError("raw global has no static heap plan");
      std::string result = valueSlots.lookup(value);
      if (result.empty())
        return addressOf.emitError("raw global address has no Udon slot");
      auto cached = cachedRawGlobalSlots.find(addressOf.getGlobal());
      if (cached != cachedRawGlobalSlots.end())
        emitCopy(location, cached->second, result);
      else
        emitRawAddress(location, found->second, result);
      return success();
    };
    auto rawAddressCrossesCall = [&](host::AddressOfOp addressOf,
                                     Operation *use) {
      Operation *cursor = addressOf->getNextNode();
      Operation *end = addressOf->getBlock() == use->getBlock() ? use : nullptr;
      for (; cursor && cursor != end; cursor = cursor->getNextNode())
        if (isa<func::CallOp, host::CallOp>(cursor))
          return true;
      return false;
    };
    auto typedArrayIndex = [&](Location location, Value array,
                               StringRef logicalIndex) -> std::string {
      uint64_t base = globalBaseOffsets.lookup(array);
      if (!base)
        return logicalIndex.str();
      constexpr StringLiteral baseSlot = "__vrc_typed_bank_base";
      constexpr StringLiteral resultSlot = "__vrc_typed_bank_index";
      addSlot(baseSlot, "SystemInt32", "0", false);
      addSlot(resultSlot, "SystemInt32", "0", false);
      emitI32Literal(location, base, baseSlot);
      emitPush(location, logicalIndex);
      emitPush(location, baseSlot);
      emitPush(location, resultSlot);
      emitExtern(location,
                 "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                 "SystemInt32");
      return resultSlot.str();
    };

    for (Block &block : function.getBody()) {
      blockAddresses[&block] = address;
      globalBlockAddresses[&block] = address;
      for (Operation &operation : block) {
        // address_of is a pure constant in the raw-memory model. Re-materialize
        // it at a direct use when the original value crosses an internal call,
        // because interprocedural scalar-slot reuse need not preserve that
        // shared slot.  Avoid doing this for ordinary uses: raw-address
        // construction is intentionally verbose on Udon and belongs outside
        // hot straight-line paths whenever no call can clobber the value.
        llvm::SmallPtrSet<Operation *, 4> materializedAddresses;
        for (Value operand : operation.getOperands()) {
          auto addressOf = operand.getDefiningOp<host::AddressOfOp>();
          if (addressOf && rawAddressCrossesCall(addressOf, &operation) &&
              materializedAddresses.insert(addressOf.getOperation()).second &&
              failed(materializeRawGlobalAddress(operand, operation.getLoc())))
            return failure();
        }
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
          auto cached = cachedRawGlobalSlots.find(addressOf.getGlobal());
          if (cached != cachedRawGlobalSlots.end())
            emitCopy(addressOf.getLoc(), cached->second,
                     valueSlots.lookup(addressOf.getResult()));
          else
            emitRawAddress(addressOf.getLoc(), found->second,
                           valueSlots.lookup(addressOf.getResult()));
          continue;
        }
        if (auto cast = dyn_cast<UnrealizedConversionCastOp>(operation)) {
          std::string source = valueSlots.lookup(cast.getInputs().front());
          std::string output = valueSlots.lookup(cast.getOutputs().front());
          if (source.empty() || output.empty())
            return cast.emitError("Udon token conversion has missing slots");
          emitCopy(cast.getLoc(), source, output);
          continue;
        }
        if (auto cast = dyn_cast<arith::IndexCastOp>(operation)) {
          Type inputType = cast.getOperand().getType();
          Type outputType = cast.getResult().getType();
          if (!((inputType.isInteger(32) && outputType.isIndex()) ||
                (inputType.isIndex() && outputType.isInteger(32))))
            return cast.emitError(
                "vrc-pc index casts support only i32 and index");
          std::string source = valueSlots.lookup(cast.getOperand());
          std::string output = valueSlots.lookup(cast.getResult());
          if (source.empty() || output.empty())
            return cast.emitError("index cast has missing heap slots");
          emitCopy(cast.getLoc(), source, output);
          continue;
        }
        if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
          auto pooled = pooledConstants.find(constant.getOperation());
          if (pooled != pooledConstants.end()) {
            uint64_t poolIndex = constantPoolWordBase + pooled->second.index;
            std::string index;
            if (poolIndex < 16) {
              index = immutableSlot("SystemInt32", std::to_string(poolIndex));
            } else {
              constexpr StringLiteral indexScratch =
                  "__vrc_constant_bank_index";
              addSlot(indexScratch, "SystemInt32", "0", false);
              emitI32Literal(constant.getLoc(), poolIndex, indexScratch);
              index = indexScratch.str();
            }
            emitPush(constant.getLoc(), rawHeapSlot);
            emitPush(constant.getLoc(), index);
            emitPush(constant.getLoc(),
                     valueSlots.lookup(constant.getResult()));
            emitExtern(constant.getLoc(),
                       "SystemInt32Array.__Get__SystemInt32__SystemInt32");
          } else if (materializedConstants.contains(constant.getOperation())) {
            if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) {
              constexpr StringLiteral bitsSlot = "__vrc_float_constant_bits";
              addSlot(bitsSlot, "SystemInt32", "0", false);
              emitI32Literal(
                  constant.getLoc(),
                  floating.getValue().bitcastToAPInt().getZExtValue(),
                  bitsSlot);
              emitPush(constant.getLoc(), bitsSlot);
              emitPush(constant.getLoc(),
                       valueSlots.lookup(constant.getResult()));
              emitExtern(constant.getLoc(),
                         "SystemBitConverter.__Int32BitsToSingle__SystemInt32__"
                         "SystemSingle");
            } else {
              auto integer = cast<IntegerAttr>(constant.getValue());
              if (constant.getType().isInteger(64))
                emitI64Constant(constant.getLoc(), integer.getValue(),
                                valueSlots.lookup(constant.getResult()));
              else
                emitI32Literal(constant.getLoc(),
                               integer.getValue().getZExtValue(),
                               valueSlots.lookup(constant.getResult()));
            }
          } else if (constant.getType().isInteger(1) &&
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
          std::string status = valueSlots.lookup(allocation.getStatus());
          if (logicalId.empty() || token.empty() || status.empty())
            return allocation.emitError(
                "Unity buffer allocation has missing handle slots");
          if (auto id = allocation->getAttrOfType<IntegerAttr>(
                  "vrc.unity_buffer_id")) {
            auto entry = llvm::find_if(unityBufferPool, [&](const auto &item) {
              return item.id == id.getInt();
            });
            if (entry == unityBufferPool.end())
              return allocation.emitError(
                  "Unity buffer allocation references a missing pool id");
            emitCopy(
                allocation.getLoc(),
                immutableSlot("SystemInt32", std::to_string(entry->index + 1)),
                token);
          } else {
            emitUnityBufferToken(allocation.getLoc(), logicalId, token);
          }
          emitCopy(allocation.getLoc(), immutableSlot("SystemInt32", "0"),
                   status);
          continue;
        }
        if (auto staticBuffer =
                dyn_cast<::vrc::vrc::StaticBufferOp>(operation)) {
          std::string token = valueSlots.lookup(staticBuffer.getBuffer());
          if (token.empty())
            return staticBuffer.emitError(
                "static tensor buffer has no allocated token slot");
          // Physical buffer tokens cross the C runtime as plain i32 values.
          // Keep immutable artifact resources disjoint from the positive
          // RenderTexture-pool ids so their origin survives that round trip.
          // Resource zero therefore becomes -1, resource one becomes -2, and
          // so on. selectUnityReadableBuffer decodes the tagged value when a
          // dispatch can no longer recover the defining StaticBufferOp.
          uint32_t tagged = static_cast<uint32_t>(
              -static_cast<int64_t>(staticBuffer.getResourceId()) - 1);
          emitI32Literal(staticBuffer.getLoc(), tagged, token);
          continue;
        }
        if (auto assignedBuffer =
                dyn_cast<::vrc::vrc::AssignedBufferOp>(operation)) {
          std::string token = valueSlots.lookup(assignedBuffer.getBuffer());
          auto heapSlot =
              assignedBuffer->getAttrOfType<StringAttr>("vrc.unity_heap_slot");
          if (token.empty())
            return assignedBuffer.emitError(
                "assigned buffer has no allocated token slot");
          auto entry = llvm::find_if(unityBufferPool, [&](const auto &item) {
            return item.id == assignedBuffer.getTextureId() && heapSlot &&
                   StringRef(item.heapSlot) == heapSlot.getValue();
          });
          if (entry == unityBufferPool.end() ||
              entry->logicalBytes != assignedBuffer.getCapacityBytes())
            return assignedBuffer.emitError(
                "assigned buffer does not match the Unity buffer pool");
          emitCopy(
              assignedBuffer.getLoc(),
              immutableSlot("SystemInt32", std::to_string(entry->index + 1)),
              token);
          continue;
        }
        if (auto release = dyn_cast<::vrc::vrc::BufferReleaseOp>(operation)) {
          // Unity buffers belong to the fixed physical pool. The C runtime
          // has already released its logical cl_mem reference, so retain the
          // RenderTexture backing. Explicitly materialize success because an
          // SSA result may reuse a non-zero slot left by an earlier value.
          std::string status = valueSlots.lookup(release.getStatus());
          if (status.empty())
            return release.emitError(
                "Unity buffer release has no allocated status slot");
          emitCopy(release.getLoc(), immutableSlot("SystemInt32", "0"), status);
          continue;
        }
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
          auto resourceStaticTensorIds =
              dispatch->getAttrOfType<ArrayAttr>("vrc.unity_static_tensor_ids");
          int64_t output = outputIndex ? outputIndex.getInt() : 0;
          std::string destinationToken =
              output >= 0 && size_t(output) < dispatch.getResources().size()
                  ? valueSlots.lookup(dispatch.getResources()[output])
                  : std::string();
          std::string event = valueSlots.lookup(dispatch.getEvent());
          if (!descriptor || !materialSlot || materialSlot.getValue().empty() ||
              destinationToken.empty() || event.empty())
            return dispatch.emitError(
                "Unity dispatch requires a bound output, a copied scalar "
                "descriptor, an event and a kernel material slot");
          llvm::StringMap<std::string> loadedMutableBuffers;
          auto loadBoundMutableBuffer = [&](StringRef heapSlot,
                                            StringRef loadPrefix) {
            auto existing = loadedMutableBuffers.find(heapSlot);
            if (existing != loadedMutableBuffers.end())
              return existing->second;
            std::string loaded = loadUnityBufferFromHeapSlot(
                dispatch.getLoc(), heapSlot, loadPrefix);
            if (!loaded.empty())
              loadedMutableBuffers.try_emplace(heapSlot, loaded);
            return loaded;
          };
          auto physicalBuffer = [&](size_t index, StringRef token,
                                    StringRef dynamicPrefix) {
            if (resourceHeapSlots && index < resourceHeapSlots.size())
              if (auto slot = dyn_cast<StringAttr>(resourceHeapSlots[index]);
                  slot && !slot.getValue().empty())
                return loadBoundMutableBuffer(slot.getValue(), dynamicPrefix);
            return selectUnityBuffer(dispatch.getLoc(), token, dynamicPrefix);
          };
          auto staticResourceId =
              [&](size_t index, Value resource) -> std::optional<int64_t> {
            if (resourceStaticTensorIds &&
                index < resourceStaticTensorIds.size())
              if (auto id =
                      dyn_cast<IntegerAttr>(resourceStaticTensorIds[index]);
                  id && id.getInt() >= 0)
                return id.getInt();
            llvm::SmallPtrSet<Operation *, 4> visited;
            while (Operation *definition = resource.getDefiningOp()) {
              if (!visited.insert(definition).second)
                break;
              if (auto staticBuffer =
                      dyn_cast<::vrc::vrc::StaticBufferOp>(definition))
                return staticBuffer.getResourceId();
              auto cast = dyn_cast<UnrealizedConversionCastOp>(definition);
              if (!cast || cast.getInputs().size() != 1)
                break;
              resource = cast.getInputs().front();
            }
            return std::nullopt;
          };
          if (output >= 0 && size_t(output) < dispatch.getResources().size() &&
              staticResourceId(static_cast<size_t>(output),
                               dispatch.getResources()[output]))
            return dispatch.emitError(
                "immutable static tensor cannot be a dispatch output");
          SmallVector<Operation *> runtimeAliasExitJumps;
          auto runtimeValidation = module->getAttrOfType<BoolAttr>(
              "vrc.runtime_validation.read_write_alias");
          if (runtimeValidation && runtimeValidation.getValue()) {
            for (auto [resourceIndex, accessValue] :
                 llvm::enumerate(dispatch.getResourceAccesses())) {
              auto access = dyn_cast<StringAttr>(accessValue);
              if (!access || access.getValue() != "read")
                continue;
              Value resource = dispatch.getResources()[resourceIndex];
              if (staticResourceId(resourceIndex, resource))
                continue;
              std::string sourceToken = valueSlots.lookup(resource);
              if (sourceToken.empty())
                return dispatch.emitError(
                    "runtime alias validation requires a resource token");
              std::string matches = prefix + "dispatch_alias_match_" +
                                    std::to_string(valueIndex++);
              addSlot(matches, "SystemBoolean", "null", false);
              emitPush(dispatch.getLoc(), sourceToken);
              emitPush(dispatch.getLoc(), destinationToken);
              emitPush(dispatch.getLoc(), matches);
              emitExtern(dispatch.getLoc(),
                         "SystemInt32.__op_Equality__SystemInt32_"
                         "SystemInt32__SystemBoolean");
              emitPush(dispatch.getLoc(), matches);
              Operation *next = create(
                  dispatch.getLoc(), vm::JumpIfFalseOp::getOperationName(),
                  {builder.getNamedAttr("target",
                                        builder.getI64IntegerAttr(0))});
              address += 8;
              emitCopy(dispatch.getLoc(), immutableSlot("SystemInt32", "-59"),
                       event);
              runtimeAliasExitJumps.push_back(emitJump(dispatch.getLoc(), 0));
              next->setAttr("target", builder.getI64IntegerAttr(
                                          static_cast<int64_t>(address)));
            }
          }
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
              size_t resourceIndex = bufferIndex++;
              Value resource = dispatch.getResources()[resourceIndex];
              std::string texture = valueSlots.lookup(resource);
              if (texture.empty())
                return dispatch.emitError(
                    "dispatch buffer has no allocated Udon heap slot");
              if (resourceIndex == size_t(output))
                continue;
              if (auto id = staticResourceId(resourceIndex, resource)) {
                std::string loaded = prefix + "dispatch_" +
                                     std::to_string(dispatch.getKernelId()) +
                                     "_static_" + std::to_string(argumentIndex);
                addSlot(loaded, "UnityEngineTexture2D", "null", false);
                emitPush(dispatch.getLoc(), "__udoncl_static_tensors");
                emitPush(dispatch.getLoc(),
                         immutableSlot("SystemInt32", std::to_string(*id)));
                emitPush(dispatch.getLoc(), loaded);
                emitExtern(dispatch.getLoc(),
                           "UnityEngineTexture2DArray.__Get__SystemInt32__"
                           "UnityEngineTexture2D");
                texture = loaded;
              } else {
                if (resourceHeapSlots &&
                    resourceIndex < resourceHeapSlots.size()) {
                  if (auto slot = dyn_cast<StringAttr>(
                          resourceHeapSlots[resourceIndex]);
                      slot && !slot.getValue().empty())
                    texture = loadBoundMutableBuffer(
                        slot.getValue(),
                        prefix + "dispatch_" +
                            std::to_string(dispatch.getKernelId()) + "_input_" +
                            std::to_string(argumentIndex));
                  else
                    texture = selectUnityReadableBuffer(
                        dispatch.getLoc(), texture,
                        prefix + "dispatch_" +
                            std::to_string(dispatch.getKernelId()) + "_input_" +
                            std::to_string(argumentIndex));
                } else {
                  texture = selectUnityReadableBuffer(
                      dispatch.getLoc(), texture,
                      prefix + "dispatch_" +
                          std::to_string(dispatch.getKernelId()) + "_input_" +
                          std::to_string(argumentIndex));
                }
              }
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
          addSlot("__udoncl_completed_event", "SystemInt32", "0", false);
          emitCopy(dispatch.getLoc(), "__udoncl_completed_event", event);
          for (Operation *jump : runtimeAliasExitJumps)
            jump->setAttr("target", builder.getI64IntegerAttr(
                                        static_cast<int64_t>(address)));
          continue;
        }
        if (auto copy = dyn_cast<::vrc::vrc::BufferCopyOp>(operation)) {
          auto materialSlot =
              copy->getAttrOfType<StringAttr>("vrc.unity_copy_material_slot");
          auto sourceHeapSlot = copy->getAttrOfType<StringAttr>(
              "vrc.unity_copy_source_heap_slot");
          auto destinationHeapSlot = copy->getAttrOfType<StringAttr>(
              "vrc.unity_copy_destination_heap_slot");
          std::string sourceToken = valueSlots.lookup(copy.getSource());
          std::string destinationToken =
              valueSlots.lookup(copy.getDestination());
          bool hasSourceHeapSlot =
              sourceHeapSlot && !sourceHeapSlot.getValue().empty();
          bool hasDestinationHeapSlot =
              destinationHeapSlot && !destinationHeapSlot.getValue().empty();
          if (!materialSlot || materialSlot.getValue().empty() ||
              hasSourceHeapSlot != hasDestinationHeapSlot ||
              (!hasSourceHeapSlot &&
               (sourceToken.empty() || destinationToken.empty())))
            return copy.emitError(
                "Unity buffer copy requires distinct bound buffers and a copy "
                "material slot");
          std::string sourcePrefix =
              prefix + "copy_source_" + std::to_string(valueIndex++);
          std::string destinationPrefix =
              prefix + "copy_destination_" + std::to_string(valueIndex++);
          std::string source =
              hasSourceHeapSlot
                  ? loadUnityBufferFromHeapSlot(
                        copy.getLoc(), sourceHeapSlot.getValue(), sourcePrefix)
                  : selectUnityReadableBuffer(copy.getLoc(), sourceToken,
                                              sourcePrefix);
          std::string destination =
              hasDestinationHeapSlot
                  ? loadUnityBufferFromHeapSlot(copy.getLoc(),
                                                destinationHeapSlot.getValue(),
                                                destinationPrefix)
                  : selectUnityBuffer(copy.getLoc(), destinationToken,
                                      destinationPrefix);
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
        if (auto length = dyn_cast<host::StringLengthOp>(operation)) {
          emitPush(length.getLoc(), valueSlots.lookup(length.getString()));
          emitPush(length.getLoc(), valueSlots.lookup(length.getResult()));
          emitExtern(length.getLoc(), "SystemString.__get_Length__SystemInt32");
          continue;
        }
        if (auto codeUnit = dyn_cast<host::StringCodeUnitOp>(operation)) {
          std::string chars =
              prefix + "string_chars_" + std::to_string(valueIndex++);
          std::string character =
              prefix + "string_char_" + std::to_string(valueIndex++);
          addSlot(chars, "SystemCharArray", "null", false);
          addSlot(character, "SystemChar", "null", false);
          emitPush(codeUnit.getLoc(), valueSlots.lookup(codeUnit.getString()));
          emitPush(codeUnit.getLoc(), valueSlots.lookup(codeUnit.getIndex()));
          emitPush(codeUnit.getLoc(), immutableSlot("SystemInt32", "1"));
          emitPush(codeUnit.getLoc(), chars);
          emitExtern(codeUnit.getLoc(),
                     "SystemString.__ToCharArray__SystemInt32_SystemInt32__"
                     "SystemCharArray");
          emitPush(codeUnit.getLoc(), chars);
          emitPush(codeUnit.getLoc(), immutableSlot("SystemInt32", "0"));
          emitPush(codeUnit.getLoc(), character);
          emitExtern(codeUnit.getLoc(),
                     "SystemCharArray.__Get__SystemInt32__SystemChar");
          emitPush(codeUnit.getLoc(), character);
          emitPush(codeUnit.getLoc(), valueSlots.lookup(codeUnit.getResult()));
          emitExtern(codeUnit.getLoc(),
                     "SystemConvert.__ToInt32__SystemChar__SystemInt32");
          continue;
        }
        if (auto length = dyn_cast<host::Int32ArrayLengthOp>(operation)) {
          emitPush(length.getLoc(), valueSlots.lookup(length.getArray()));
          emitPush(length.getLoc(), valueSlots.lookup(length.getResult()));
          emitExtern(length.getLoc(),
                     "SystemInt32Array.__get_Length__SystemInt32");
          continue;
        }
        if (auto get = dyn_cast<host::Int32ArrayGetOp>(operation)) {
          emitPush(get.getLoc(), valueSlots.lookup(get.getArray()));
          emitPush(get.getLoc(), valueSlots.lookup(get.getIndex()));
          emitPush(get.getLoc(), valueSlots.lookup(get.getResult()));
          emitExtern(get.getLoc(),
                     "SystemInt32Array.__Get__SystemInt32__SystemInt32");
          continue;
        }
        if (auto set = dyn_cast<host::Int32ArraySetOp>(operation)) {
          emitPush(set.getLoc(), valueSlots.lookup(set.getArray()));
          emitPush(set.getLoc(), valueSlots.lookup(set.getIndex()));
          emitPush(set.getLoc(), valueSlots.lookup(set.getValue()));
          emitExtern(set.getLoc(),
                     "SystemInt32Array.__Set__SystemInt32_SystemInt32__"
                     "SystemVoid");
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
                                 load.getResult().getType(),
                                 load.getAlignment())))
            return load.emitError("unsupported raw Udon load");
          continue;
        }
        if (auto store = dyn_cast<host::StoreOp>(operation)) {
          std::string pointer = valueSlots.lookup(store.getPointer());
          std::string value = valueSlots.lookup(store.getValue());
          if (pointer.empty() || value.empty())
            return store.emitError("raw store operand has no Udon heap slot");
          if (failed(emitRawStore(store.getLoc(), pointer, value,
                                  store.getValue().getType(),
                                  store.getAlignment())))
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
        if (auto memset = dyn_cast<host::MemsetOp>(operation)) {
          std::string destination = valueSlots.lookup(memset.getDestination());
          std::string byteValue = valueSlots.lookup(memset.getByteValue());
          std::string byteSize = valueSlots.lookup(memset.getByteSize());
          if (destination.empty() || byteValue.empty() || byteSize.empty())
            return memset.emitError("memset operand has no Udon heap slot");
          bool isZero = false;
          if (auto constant =
                  memset.getByteValue().getDefiningOp<arith::ConstantOp>())
            if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
              isZero = integer.getValue().isZero();
          emitRawMemset(memset.getLoc(), destination, byteValue, byteSize,
                        isZero);
          continue;
        }
        if (isa<host::AsyncResumePointOp>(operation))
          continue;
        if (auto bind = dyn_cast<host::AsyncBindEventOp>(operation)) {
          addSlot(asyncPendingObjectSlot, "SystemInt32", "0", true);
          emitI32Literal(bind.getLoc(), bind.getObjectId(),
                         asyncPendingObjectSlot);
          continue;
        }
        if (auto enter = dyn_cast<host::AsyncEnterOp>(operation)) {
          auto target = llvm::find_if(
              asyncResumeTargets, [&](const AsyncResumeTarget &candidate) {
                return candidate.frameId == enter.getFrameId() &&
                       candidate.resumePc == enter.getResumePc();
              });
          if (target == asyncResumeTargets.end())
            return enter.emitError("static async route has no resume target");
          auto calleePlan = functionPlans.find(target->function.getOperation());
          if (calleePlan == functionPlans.end())
            return enter.emitError("async route target has no function plan");
          if (calleePlan->second.returnSlot) {
            emitReturnAddress(enter.getLoc(), *calleePlan->second.returnSlot);
          }
          Operation *jump = emitJump(enter.getLoc(), 0);
          asyncResumeRelocations.emplace_back(jump, target->block);
          continue;
        }
        if (auto resume = dyn_cast<host::AsyncResumeOp>(operation)) {
          std::string frame = valueSlots.lookup(resume.getFrameId());
          std::string pc = valueSlots.lookup(resume.getResumePc());
          if (frame.empty() || pc.empty())
            return resume.emitError(
                "async resume operands have no allocated Udon heap slots");
          SmallVector<Operation *> done;
          constexpr StringLiteral frameConstant =
              "__udoncl_async_route_frame_constant";
          constexpr StringLiteral pcConstant =
              "__udoncl_async_route_pc_constant";
          addSlot(frameConstant, "SystemInt32", "0", false);
          addSlot(pcConstant, "SystemInt32", "0", false);
          for (auto [routeIndex, target] :
               llvm::enumerate(asyncResumeTargets)) {
            auto calleePlan =
                functionPlans.find(target.function.getOperation());
            if (calleePlan == functionPlans.end())
              return resume.emitError(
                  "async resume target has no physical function plan");
            bool resumesRoot = !calleePlan->second.returnSlot.has_value();
            emitI32Literal(resume.getLoc(), target.frameId, frameConstant);
            emitI32Literal(resume.getLoc(), target.resumePc, pcConstant);
            std::string matchPrefix = prefix + "async_route_" +
                                      std::to_string(callReturnIndex) + "_" +
                                      std::to_string(routeIndex);
            std::string frameMatches = matchPrefix + "_frame";
            std::string pcMatches = matchPrefix + "_pc";
            addSlot(frameMatches, "SystemBoolean", "null", false);
            addSlot(pcMatches, "SystemBoolean", "null", false);
            emitPush(resume.getLoc(), frame);
            emitPush(resume.getLoc(), frameConstant);
            emitPush(resume.getLoc(), frameMatches);
            emitExtern(resume.getLoc(),
                       "SystemInt32.__op_Equality__SystemInt32_SystemInt32__"
                       "SystemBoolean");
            emitPush(resume.getLoc(), frameMatches);
            Operation *nextFrame = create(
                resume.getLoc(), vm::JumpIfFalseOp::getOperationName(),
                {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
            address += 8;
            emitPush(resume.getLoc(), pc);
            emitPush(resume.getLoc(), pcConstant);
            emitPush(resume.getLoc(), pcMatches);
            emitExtern(resume.getLoc(),
                       "SystemInt32.__op_Equality__SystemInt32_SystemInt32__"
                       "SystemBoolean");
            emitPush(resume.getLoc(), pcMatches);
            Operation *nextPc = create(
                resume.getLoc(), vm::JumpIfFalseOp::getOperationName(),
                {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
            address += 8;
            if (resumesRoot) {
              Operation *routeJump = emitJump(resume.getLoc(), 0);
              asyncResumeRelocations.emplace_back(routeJump, target.block);
            } else {
              emitReturnAddress(resume.getLoc(),
                                *calleePlan->second.returnSlot);
              Operation *routeJump = emitJump(resume.getLoc(), 0);
              asyncResumeRelocations.emplace_back(routeJump, target.block);
              done.push_back(emitJump(resume.getLoc(), 0));
            }
            auto nextAddress =
                builder.getI64IntegerAttr(static_cast<int64_t>(address));
            nextFrame->setAttr("target", nextAddress);
            nextPc->setAttr("target", nextAddress);
          }
          for (Operation *jump : done)
            jump->setAttr("target", builder.getI64IntegerAttr(
                                        static_cast<int64_t>(address)));
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
          emitReturnAddress(call.getLoc(), *calleePlan->second.returnSlot,
                            call.getOperation());
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
        if (auto call = dyn_cast<func::CallOp>(operation)) {
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
          emitReturnAddress(call.getLoc(), *calleePlan->second.returnSlot,
                            call.getOperation());
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
        if (auto branch = dyn_cast<cf::SwitchOp>(operation)) {
          if (!branch.getFlag().getType().isInteger(32))
            return branch.emitError(
                "Udon switch lowering currently requires an i32 flag");
          std::string flag = valueSlots.lookup(branch.getFlag());
          if (flag.empty())
            return branch.emitError("switch flag has no Udon heap slot");
          auto caseValues = branch.getCaseValues();
          if (caseValues) {
            constexpr StringLiteral caseConstant = "__vrc_switch_case_constant";
            constexpr StringLiteral condition = "__vrc_switch_condition";
            addSlot(caseConstant, "SystemInt32", "0", false);
            addSlot(condition, "SystemBoolean", "null", false);
            struct SwitchCase {
              llvm::APInt value;
              unsigned originalIndex;
            };
            SmallVector<SwitchCase> cases;
            for (auto [caseIndex, caseValue] :
                 llvm::enumerate(caseValues->getValues<llvm::APInt>()))
              cases.push_back({caseValue, static_cast<unsigned>(caseIndex)});
            llvm::sort(cases,
                       [](const SwitchCase &left, const SwitchCase &right) {
                         return left.value.slt(right.value);
                       });
            for (auto [left, right] : llvm::zip(cases, llvm::drop_begin(cases)))
              if (left.value == right.value)
                return branch.emitError("switch case values must be unique");

            auto emitEquality =
                [&](const SwitchCase &switchCase,
                    SmallVectorImpl<Operation *> &defaults) -> LogicalResult {
              emitI32Literal(branch.getLoc(), switchCase.value.getZExtValue(),
                             caseConstant);
              emitPush(branch.getLoc(), flag);
              emitPush(branch.getLoc(), caseConstant);
              emitPush(branch.getLoc(), condition);
              emitExtern(branch.getLoc(),
                         "SystemInt32.__op_Equality__SystemInt32_SystemInt32__"
                         "SystemBoolean");
              emitPush(branch.getLoc(), condition);
              Operation *nextCase =
                  create(branch.getLoc(), vm::JumpIfFalseOp::getOperationName(),
                         {builder.getNamedAttr("target",
                                               builder.getI64IntegerAttr(0))});
              address += 8;
              defaults.push_back(nextCase);
              if (failed(copies(
                      branch.getLoc(),
                      branch.getCaseOperands(switchCase.originalIndex),
                      branch.getCaseDestinations()[switchCase.originalIndex])))
                return failure();
              jumpTo(branch.getLoc(),
                     branch.getCaseDestinations()[switchCase.originalIndex]);
              return success();
            };

            SmallVector<Operation *> defaultJumps;
            if (cases.size() < 5) {
              for (const SwitchCase &switchCase : cases) {
                if (failed(emitEquality(switchCase, defaultJumps)))
                  return failure();
                defaultJumps.back()->setAttr(
                    "target",
                    builder.getI64IntegerAttr(static_cast<int64_t>(address)));
                defaultJumps.pop_back();
              }
            } else {
              auto emitTree = [&](auto &&self, unsigned begin,
                                  unsigned end) -> LogicalResult {
                if (end - begin == 1)
                  return emitEquality(cases[begin], defaultJumps);
                unsigned middle = begin + (end - begin) / 2;
                emitI32Literal(branch.getLoc(),
                               cases[middle].value.getZExtValue(),
                               caseConstant);
                emitPush(branch.getLoc(), flag);
                emitPush(branch.getLoc(), caseConstant);
                emitPush(branch.getLoc(), condition);
                emitExtern(
                    branch.getLoc(),
                    "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__"
                    "SystemBoolean");
                emitPush(branch.getLoc(), condition);
                Operation *rightTree = create(
                    branch.getLoc(), vm::JumpIfFalseOp::getOperationName(),
                    {builder.getNamedAttr("target",
                                          builder.getI64IntegerAttr(0))});
                address += 8;
                if (failed(self(self, begin, middle)))
                  return failure();
                rightTree->setAttr(
                    "target",
                    builder.getI64IntegerAttr(static_cast<int64_t>(address)));
                return self(self, middle, end);
              };
              if (failed(emitTree(emitTree, 0, cases.size())))
                return failure();
              for (Operation *defaultJump : defaultJumps)
                defaultJump->setAttr(
                    "target",
                    builder.getI64IntegerAttr(static_cast<int64_t>(address)));
            }
          }
          if (failed(copies(branch.getLoc(), branch.getDefaultOperands(),
                            branch.getDefaultDestination())))
            return failure();
          jumpTo(branch.getLoc(), branch.getDefaultDestination());
          continue;
        }
        if (auto store = dyn_cast<host::SpillStoreOp>(operation)) {
          auto bank = typedSpillBanks.find(store.getStorageKey());
          std::string value = valueSlots.lookup(store.getValue());
          if (bank == typedSpillBanks.end() || value.empty() ||
              store.getIndex() < 0 ||
              static_cast<unsigned>(store.getIndex()) >= bank->second.length)
            return store.emitError("invalid typed spill store plan");
          constexpr StringLiteral indexSlot = "__vrc_spill_index";
          addSlot(indexSlot, "SystemInt32", "0", false);
          emitSpillIndex(store.getLoc(), store.getStorageKey(),
                         store.getIndex(), indexSlot);
          emitPush(store.getLoc(), bank->second.slot);
          emitPush(store.getLoc(), indexSlot);
          emitPush(store.getLoc(), value);
          emitExtern(store.getLoc(),
                     bank->second.arrayType + ".__Set__SystemInt32_" +
                         bank->second.scalarType + "__SystemVoid");
          continue;
        }
        if (auto load = dyn_cast<host::SpillLoadOp>(operation)) {
          auto bank = typedSpillBanks.find(load.getStorageKey());
          std::string result = valueSlots.lookup(load.getValue());
          if (bank == typedSpillBanks.end() || result.empty() ||
              load.getIndex() < 0 ||
              static_cast<unsigned>(load.getIndex()) >= bank->second.length)
            return load.emitError("invalid typed spill load plan");
          constexpr StringLiteral indexSlot = "__vrc_spill_index";
          addSlot(indexSlot, "SystemInt32", "0", false);
          emitSpillIndex(load.getLoc(), load.getStorageKey(), load.getIndex(),
                         indexSlot);
          emitPush(load.getLoc(), bank->second.slot);
          emitPush(load.getLoc(), indexSlot);
          emitPush(load.getLoc(), result);
          emitExtern(load.getLoc(), bank->second.arrayType +
                                        ".__Get__SystemInt32__" +
                                        bank->second.scalarType);
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
          index = typedArrayIndex(load.getLoc(), load.getMemRef(), index);
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
          index = typedArrayIndex(store.getLoc(), store.getMemRef(), index);
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
            std::string threshold = immutableSlot("SystemInt32", "128");
            std::string modulus = immutableSlot("SystemInt32", "256");
            constexpr StringLiteral negative = "__vrc_i8_sign_extend_negative";
            constexpr StringLiteral isNegative =
                "__vrc_i8_sign_extend_is_negative";
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
              (!cast.getOut().getType().isInteger(32) &&
               !(cast.getIn().getType().isInteger(32) &&
                 cast.getOut().getType().isInteger(64))))
            return cast.emitError(
                       "unsupported unsigned integer extension from ")
                   << cast.getIn().getType() << " to "
                   << cast.getOut().getType() << " in @"
                   << cast->getParentOfType<func::FuncOp>().getSymName();
          if (cast.getIn().getType().isInteger(8) ||
              cast.getIn().getType().isInteger(16)) {
            // Raw i8/i16 loads are represented by non-negative SystemInt32
            // values in the Udon heap, so zero extension is an exact copy.
            emitCopy(cast.getLoc(), source, result);
            continue;
          }
          if (cast.getIn().getType().isInteger(32) &&
              cast.getOut().getType().isInteger(64)) {
            constexpr StringLiteral isNegative =
                "__vrc_i32_zero_extend_is_negative";
            constexpr StringLiteral adjusted = "__vrc_i32_zero_extend_adjusted";
            constexpr StringLiteral modulus = "__vrc_i32_unsigned_modulus";
            addSlot(isNegative, "SystemBoolean", "null", false);
            addSlot(adjusted, "SystemInt64", "0", false);
            addSlot(modulus, "SystemInt64", "null", false);
            std::string zero = immutableSlot("SystemInt32", "0");
            emitI64Constant(cast.getLoc(), llvm::APInt::getOneBitSet(64, 32),
                            modulus);
            emitPush(cast.getLoc(), source);
            emitPush(cast.getLoc(), result);
            emitExtern(cast.getLoc(),
                       "SystemConvert.__ToInt64__SystemInt32__SystemInt64");
            emitPush(cast.getLoc(), source);
            emitPush(cast.getLoc(), zero);
            emitPush(cast.getLoc(), isNegative);
            emitExtern(cast.getLoc(),
                       "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__"
                       "SystemBoolean");
            emitPush(cast.getLoc(), isNegative);
            Operation *done = create(
                cast.getLoc(), vm::JumpIfFalseOp::getOperationName(),
                {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
            address += 8;
            emitPush(cast.getLoc(), result);
            emitPush(cast.getLoc(), modulus);
            emitPush(cast.getLoc(), adjusted);
            emitExtern(cast.getLoc(),
                       "SystemInt64.__op_Addition__SystemInt64_SystemInt64__"
                       "SystemInt64");
            emitCopy(cast.getLoc(), adjusted, result);
            done->setAttr("target", builder.getI64IntegerAttr(
                                        static_cast<int64_t>(address)));
            continue;
          }
          if (!cast.getIn().getType().isInteger(1))
            return cast.emitError(
                       "unsupported unsigned integer extension from ")
                   << cast.getIn().getType() << " to "
                   << cast.getOut().getType() << " in @"
                   << cast->getParentOfType<func::FuncOp>().getSymName();
          std::string zero = immutableSlot("SystemInt32", "0");
          std::string one = immutableSlot("SystemInt32", "1");
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
          if (source.empty() || result.empty())
            return cast.emitError("unsupported integer truncation");
          if (cast.getIn().getType().isInteger(64) &&
              cast.getOut().getType().isInteger(32)) {
            emitPush(cast.getLoc(), source);
            emitPush(cast.getLoc(), result);
            emitExtern(cast.getLoc(),
                       "SystemConvert.__ToInt32__SystemInt64__SystemInt32");
          } else if (cast.getIn().getType().isInteger(32) &&
                     (cast.getOut().getType().isInteger(8) ||
                      cast.getOut().getType().isInteger(16))) {
            std::string mask = immutableSlot(
                "SystemInt32",
                cast.getOut().getType().isInteger(8) ? "255" : "65535");
            emitPush(cast.getLoc(), source);
            emitPush(cast.getLoc(), mask);
            emitPush(cast.getLoc(), result);
            emitExtern(cast.getLoc(),
                       "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__"
                       "SystemInt32");
          } else {
            return cast.emitError("unsupported integer truncation");
          }
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
        if (auto cast = dyn_cast<arith::UIToFPOp>(operation)) {
          std::string source = valueSlots.lookup(cast.getIn());
          std::string result = valueSlots.lookup(cast.getOut());
          if (source.empty() || result.empty() ||
              !cast.getIn().getType().isInteger(32) ||
              !cast.getOut().getType().isF32())
            return cast.emitError(
                "unsigned integer to float conversion supports only i32 to "
                "f32");
          constexpr StringLiteral widened = "__vrc_uitofp_widened";
          constexpr StringLiteral adjusted = "__vrc_uitofp_adjusted";
          constexpr StringLiteral offset = "__vrc_uitofp_offset";
          constexpr StringLiteral negative = "__vrc_uitofp_negative";
          constexpr StringLiteral unsignedValue = "__vrc_uitofp_unsigned";
          addSlot(widened, "SystemInt64", "null", false);
          addSlot(adjusted, "SystemInt64", "null", false);
          addSlot(offset, "SystemInt64", "null", false);
          addSlot(negative, "SystemBoolean", "null", false);
          addSlot(unsignedValue, "SystemUInt32", "0", false);
          emitI64Constant(cast.getLoc(), llvm::APInt(64, uint64_t{1} << 32),
                          offset);
          emitPush(cast.getLoc(), source);
          emitPush(cast.getLoc(), widened);
          emitExtern(cast.getLoc(),
                     "SystemConvert.__ToInt64__SystemInt32__SystemInt64");
          std::string zero = immutableSlot("SystemInt32", "0");
          emitPush(cast.getLoc(), source);
          emitPush(cast.getLoc(), zero);
          emitPush(cast.getLoc(), negative);
          emitExtern(cast.getLoc(),
                     "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__"
                     "SystemBoolean");
          emitPush(cast.getLoc(), negative);
          Operation *nonnegative = create(
              cast.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          emitPush(cast.getLoc(), widened);
          emitPush(cast.getLoc(), offset);
          emitPush(cast.getLoc(), adjusted);
          emitExtern(cast.getLoc(),
                     "SystemInt64.__op_Addition__SystemInt64_SystemInt64__"
                     "SystemInt64");
          emitCopy(cast.getLoc(), adjusted, widened);
          nonnegative->setAttr("target", builder.getI64IntegerAttr(
                                             static_cast<int64_t>(address)));
          emitPush(cast.getLoc(), widened);
          emitPush(cast.getLoc(), unsignedValue);
          emitExtern(cast.getLoc(),
                     "SystemConvert.__ToUInt32__SystemInt64__SystemUInt32");
          emitPush(cast.getLoc(), unsignedValue);
          emitPush(cast.getLoc(), result);
          emitExtern(cast.getLoc(),
                     "SystemConvert.__ToSingle__SystemUInt32__SystemSingle");
          continue;
        }
        if (auto cast = dyn_cast<arith::FPToSIOp>(operation)) {
          std::string source = valueSlots.lookup(cast.getIn());
          std::string result = valueSlots.lookup(cast.getOut());
          if (source.empty() || result.empty() ||
              !cast.getIn().getType().isF32() ||
              !cast.getOut().getType().isInteger(32))
            return cast.emitError(
                "float to signed integer conversion supports only f32 to "
                "i32");
          constexpr StringLiteral nonnegative = "__vrc_fptosi_i32_nonnegative";
          addSlot(nonnegative, "SystemBoolean", "null", false);
          std::string zero = immutableSlot("SystemSingle", "0");
          emitPush(cast.getLoc(), source);
          emitPush(cast.getLoc(), zero);
          emitPush(cast.getLoc(), nonnegative);
          emitExtern(cast.getLoc(),
                     "SystemSingle.__op_GreaterThanOrEqual__SystemSingle_"
                     "SystemSingle__SystemBoolean");
          emitPush(cast.getLoc(), nonnegative);
          Operation *useCeiling = create(
              cast.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          emitPush(cast.getLoc(), source);
          emitPush(cast.getLoc(), result);
          emitExtern(cast.getLoc(),
                     "UnityEngineMathf.__FloorToInt__SystemSingle__"
                     "SystemInt32");
          Operation *done = emitJump(cast.getLoc(), 0);
          useCeiling->setAttr("target", builder.getI64IntegerAttr(
                                            static_cast<int64_t>(address)));
          emitPush(cast.getLoc(), source);
          emitPush(cast.getLoc(), result);
          emitExtern(cast.getLoc(),
                     "UnityEngineMathf.__CeilToInt__SystemSingle__"
                     "SystemInt32");
          done->setAttr("target", builder.getI64IntegerAttr(
                                      static_cast<int64_t>(address)));
          continue;
        }
        if (auto cast = dyn_cast<arith::BitcastOp>(operation)) {
          std::string source = valueSlots.lookup(cast.getIn());
          std::string result = valueSlots.lookup(cast.getOut());
          if (source.empty() || result.empty())
            return cast.emitError("bitcast has missing Udon heap slots");
          emitPush(cast.getLoc(), source);
          emitPush(cast.getLoc(), result);
          if (cast.getIn().getType().isF32() &&
              cast.getOut().getType().isInteger(32)) {
            emitExtern(cast.getLoc(),
                       "SystemBitConverter.__SingleToInt32Bits__SystemSingle__"
                       "SystemInt32");
          } else if (cast.getIn().getType().isInteger(32) &&
                     cast.getOut().getType().isF32()) {
            emitExtern(cast.getLoc(),
                       "SystemBitConverter.__Int32BitsToSingle__SystemInt32__"
                       "SystemSingle");
          } else {
            return cast.emitError(
                "Udon bitcast supports only bit-exact f32/i32 casts");
          }
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

          std::string zero = immutableSlot("SystemInt32", "0");
          std::string width = immutableSlot("SystemInt32", "32");
          std::string allOnes = immutableSlot("SystemInt32", "-1");
          constexpr StringLiteral isZero = "__vrc_lshr_i32_is_zero";
          constexpr StringLiteral arithmetic = "__vrc_lshr_i32_arithmetic";
          constexpr StringLiteral inverseAmount =
              "__vrc_lshr_i32_inverse_amount";
          constexpr StringLiteral highMask = "__vrc_lshr_i32_high_mask";
          constexpr StringLiteral lowMask = "__vrc_lshr_i32_low_mask";
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
        if (auto bitAnd = dyn_cast<arith::AndIOp>(operation);
            bitAnd && bitAnd.getType().isInteger(1)) {
          std::string lhs = valueSlots.lookup(bitAnd.getLhs());
          std::string rhs = valueSlots.lookup(bitAnd.getRhs());
          std::string result = valueSlots.lookup(bitAnd.getResult());
          if (lhs.empty() || rhs.empty() || result.empty())
            return bitAnd.emitError(
                "boolean conjunction has missing Udon heap slots");

          // Udon has no SystemBoolean logical-and extern. Preserve i1
          // semantics with short-circuit control flow instead of passing
          // Boolean heap slots to the integer bitwise extern.
          emitCopy(bitAnd.getLoc(), lhs, result);
          emitPush(bitAnd.getLoc(), lhs);
          Operation *done = create(
              bitAnd.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          emitCopy(bitAnd.getLoc(), rhs, result);
          done->setAttr("target", builder.getI64IntegerAttr(
                                      static_cast<int64_t>(address)));
          continue;
        }
        if (auto bitOr = dyn_cast<arith::OrIOp>(operation);
            bitOr && bitOr.getType().isInteger(1)) {
          std::string lhs = valueSlots.lookup(bitOr.getLhs());
          std::string rhs = valueSlots.lookup(bitOr.getRhs());
          std::string result = valueSlots.lookup(bitOr.getResult());
          if (lhs.empty() || rhs.empty() || result.empty())
            return bitOr.emitError(
                "boolean disjunction has missing Udon heap slots");
          emitCopy(bitOr.getLoc(), lhs, result);
          emitPush(bitOr.getLoc(), lhs);
          Operation *useRhs = create(
              bitOr.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          Operation *done = emitJump(bitOr.getLoc(), 0);
          useRhs->setAttr("target", builder.getI64IntegerAttr(
                                        static_cast<int64_t>(address)));
          emitCopy(bitOr.getLoc(), rhs, result);
          done->setAttr("target", builder.getI64IntegerAttr(
                                      static_cast<int64_t>(address)));
          continue;
        }
        if (auto bitXor = dyn_cast<arith::XOrIOp>(operation);
            bitXor && bitXor.getType().isInteger(1)) {
          std::string lhs = valueSlots.lookup(bitXor.getLhs());
          std::string rhs = valueSlots.lookup(bitXor.getRhs());
          std::string result = valueSlots.lookup(bitXor.getResult());
          if (lhs.empty() || rhs.empty() || result.empty())
            return bitXor.emitError(
                "boolean exclusive-or has missing Udon heap slots");
          std::string falseValue = immutableSlot("SystemBoolean", "null");
          std::string one = immutableSlot("SystemInt32", "1");
          emitCopy(bitXor.getLoc(), rhs, result);
          emitPush(bitXor.getLoc(), lhs);
          Operation *lhsFalse = create(
              bitXor.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          emitCopy(bitXor.getLoc(), falseValue, result);
          emitPush(bitXor.getLoc(), rhs);
          Operation *rhsFalse = create(
              bitXor.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          Operation *done = emitJump(bitXor.getLoc(), 0);
          rhsFalse->setAttr("target", builder.getI64IntegerAttr(
                                          static_cast<int64_t>(address)));
          emitPush(bitXor.getLoc(), one);
          emitPush(bitXor.getLoc(), one);
          emitPush(bitXor.getLoc(), result);
          emitExtern(bitXor.getLoc(),
                     "SystemInt32.__op_Equality__SystemInt32_SystemInt32__"
                     "SystemBoolean");
          auto doneAddress =
              builder.getI64IntegerAttr(static_cast<int64_t>(address));
          lhsFalse->setAttr("target", doneAddress);
          done->setAttr("target", doneAddress);
          continue;
        }
        if (auto comparison = dyn_cast<arith::CmpIOp>(operation);
            comparison && comparison.getLhs().getType().isInteger(1)) {
          if (comparison.getPredicate() != arith::CmpIPredicate::eq &&
              comparison.getPredicate() != arith::CmpIPredicate::ne)
            return comparison.emitError(
                "boolean Udon comparison supports only equality and "
                "inequality");
          std::string lhs = valueSlots.lookup(comparison.getLhs());
          std::string rhs = valueSlots.lookup(comparison.getRhs());
          std::string result = valueSlots.lookup(comparison.getResult());
          if (lhs.empty() || rhs.empty() || result.empty())
            return comparison.emitError(
                "boolean comparison has missing Udon heap slots");

          // Udon has no guaranteed Boolean equality extern. Compute eq/ne
          // with Boolean control flow, materializing true through the stable
          // Int32 equality extern already used for i1 constants.
          std::string falseValue = immutableSlot("SystemBoolean", "null");
          std::string one = immutableSlot("SystemInt32", "1");
          emitCopy(comparison.getLoc(), rhs, result);
          emitPush(comparison.getLoc(), lhs);
          Operation *lhsFalse = create(
              comparison.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          SmallVector<Operation *> done;
          bool equality = comparison.getPredicate() == arith::CmpIPredicate::eq;
          if (equality) {
            done.push_back(emitJump(comparison.getLoc(), 0));
            lhsFalse->setAttr("target", builder.getI64IntegerAttr(
                                            static_cast<int64_t>(address)));
          }
          emitCopy(comparison.getLoc(), falseValue, result);
          emitPush(comparison.getLoc(), rhs);
          Operation *rhsFalse = create(
              comparison.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          done.push_back(emitJump(comparison.getLoc(), 0));
          rhsFalse->setAttr("target", builder.getI64IntegerAttr(
                                          static_cast<int64_t>(address)));
          emitPush(comparison.getLoc(), one);
          emitPush(comparison.getLoc(), one);
          emitPush(comparison.getLoc(), result);
          emitExtern(comparison.getLoc(),
                     "SystemInt32.__op_Equality__SystemInt32_SystemInt32__"
                     "SystemBoolean");
          auto doneAddress =
              builder.getI64IntegerAttr(static_cast<int64_t>(address));
          if (!equality)
            lhsFalse->setAttr("target", doneAddress);
          for (Operation *jump : done)
            jump->setAttr("target", doneAddress);
          continue;
        }
        if (auto comparison = dyn_cast<arith::CmpFOp>(operation);
            comparison &&
            comparison.getPredicate() == arith::CmpFPredicate::UNO) {
          std::string lhs = valueSlots.lookup(comparison.getLhs());
          std::string rhs = valueSlots.lookup(comparison.getRhs());
          std::string result = valueSlots.lookup(comparison.getResult());
          if (lhs.empty() || rhs.empty() || result.empty() ||
              !comparison.getLhs().getType().isF32())
            return comparison.emitError(
                "unordered float comparison requires allocated f32 slots");
          constexpr StringLiteral lhsNan = "__vrc_cmpf_lhs_nan";
          constexpr StringLiteral rhsNan = "__vrc_cmpf_rhs_nan";
          addSlot(lhsNan, "SystemBoolean", "null", false);
          addSlot(rhsNan, "SystemBoolean", "null", false);
          auto emitIsNan = [&](StringRef value, StringRef destination) {
            emitPush(comparison.getLoc(), value);
            emitPush(comparison.getLoc(), value);
            emitPush(comparison.getLoc(), destination);
            emitExtern(comparison.getLoc(),
                       "SystemSingle.__op_Inequality__SystemSingle_"
                       "SystemSingle__SystemBoolean");
          };
          emitIsNan(lhs, lhsNan);
          emitIsNan(rhs, rhsNan);
          emitCopy(comparison.getLoc(), lhsNan, result);
          emitPush(comparison.getLoc(), lhsNan);
          Operation *useRhs = create(
              comparison.getLoc(), vm::JumpIfFalseOp::getOperationName(),
              {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
          address += 8;
          Operation *done = emitJump(comparison.getLoc(), 0);
          useRhs->setAttr("target", builder.getI64IntegerAttr(
                                        static_cast<int64_t>(address)));
          emitCopy(comparison.getLoc(), rhsNan, result);
          done->setAttr("target", builder.getI64IntegerAttr(
                                      static_cast<int64_t>(address)));
          continue;
        }
        if (auto comparison = dyn_cast<arith::CmpIOp>(operation);
            comparison &&
            (comparison.getPredicate() == arith::CmpIPredicate::ult ||
             comparison.getPredicate() == arith::CmpIPredicate::ule ||
             comparison.getPredicate() == arith::CmpIPredicate::ugt ||
             comparison.getPredicate() == arith::CmpIPredicate::uge)) {
          Type type = comparison.getLhs().getType();
          if (!type.isInteger(32) && !type.isInteger(64))
            return comparison.emitError(
                "unsigned Udon comparison supports only i32/i64");
          std::string lhs = valueSlots.lookup(comparison.getLhs());
          std::string rhs = valueSlots.lookup(comparison.getRhs());
          std::string result = valueSlots.lookup(comparison.getResult());
          if (lhs.empty() || rhs.empty() || result.empty())
            return comparison.emitError(
                "unsigned comparison has missing Udon heap slots");
          StringRef systemType =
              type.isInteger(64) ? "SystemInt64" : "SystemInt32";
          std::string mask;
          if (type.isInteger(64)) {
            mask = "__vrc_unsigned_cmp_sign_i64";
            addSlot(mask, systemType, "null", false);
            emitI64Constant(comparison.getLoc(), llvm::APInt::getSignMask(64),
                            mask);
          } else {
            mask = "__vrc_unsigned_cmp_sign_i32";
            addSlot(mask, systemType, "null", false);
            emitI32Literal(comparison.getLoc(), uint64_t{1} << 31, mask);
          }
          std::string adjustedLhs = type.isInteger(64)
                                        ? "__vrc_unsigned_cmp_lhs_i64"
                                        : "__vrc_unsigned_cmp_lhs_i32";
          std::string adjustedRhs = type.isInteger(64)
                                        ? "__vrc_unsigned_cmp_rhs_i64"
                                        : "__vrc_unsigned_cmp_rhs_i32";
          addSlot(adjustedLhs, systemType, "null", false);
          addSlot(adjustedRhs, systemType, "null", false);
          auto adjust = [&](StringRef source, StringRef destination) {
            emitPush(comparison.getLoc(), source);
            emitPush(comparison.getLoc(), mask);
            emitPush(comparison.getLoc(), destination);
            emitExtern(comparison.getLoc(),
                       (systemType + ".__op_LogicalXor__" + systemType + "_" +
                        systemType + "__" + systemType)
                           .str());
          };
          adjust(lhs, adjustedLhs);
          adjust(rhs, adjustedRhs);
          StringRef comparisonName;
          switch (comparison.getPredicate()) {
          case arith::CmpIPredicate::ult:
            comparisonName = "LessThan";
            break;
          case arith::CmpIPredicate::ule:
            comparisonName = "LessThanOrEqual";
            break;
          case arith::CmpIPredicate::ugt:
            comparisonName = "GreaterThan";
            break;
          case arith::CmpIPredicate::uge:
            comparisonName = "GreaterThanOrEqual";
            break;
          default:
            llvm_unreachable("handled unsigned comparison predicate");
          }
          emitPush(comparison.getLoc(), adjustedLhs);
          emitPush(comparison.getLoc(), adjustedRhs);
          emitPush(comparison.getLoc(), result);
          emitExtern(comparison.getLoc(),
                     (systemType + ".__op_" + comparisonName + "__" +
                      systemType + "_" + systemType + "__SystemBoolean")
                         .str());
          continue;
        }
        if (isa<arith::DivUIOp, arith::RemUIOp>(operation)) {
          if (!operation.getOperand(0).getType().isInteger(32))
            return operation.emitError(
                "unsigned Udon division/remainder currently requires i32");
          std::string lhs = valueSlots.lookup(operation.getOperand(0));
          std::string rhs = valueSlots.lookup(operation.getOperand(1));
          std::string result = valueSlots.lookup(operation.getResult(0));
          if (lhs.empty() || rhs.empty() || result.empty())
            return operation.emitError(
                "unsigned division/remainder has missing Udon heap slots");
          if (failed(emitUnsignedDivRem(operation.getLoc(), lhs, rhs, result,
                                        isa<arith::RemUIOp>(operation))))
            return operation.emitError(
                "failed to lower unsigned division/remainder");
          continue;
        }
        StringRef signature = binaryExtern(&operation);
        if (auto comparison = dyn_cast<host::RefCmpOp>(operation))
          signature = comparison.getPredicate() == "eq"
                          ? "SystemObject.__op_Equality__SystemObject_"
                            "SystemObject__SystemBoolean"
                          : "SystemObject.__op_Inequality__SystemObject_"
                            "SystemObject__SystemBoolean";
        if (auto comparison = dyn_cast<arith::CmpIOp>(operation))
          signature = comparisonExtern(comparison);
        if (auto comparison = dyn_cast<arith::CmpFOp>(operation))
          signature = comparisonExtern(comparison);
        if (signature.empty())
          if (auto comparison = dyn_cast<arith::CmpFOp>(operation))
            return comparison.emitError()
                   << "unsupported floating comparison predicate "
                   << arith::stringifyCmpFPredicate(comparison.getPredicate());
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

  std::string immutableSlot(StringRef type, StringRef initial) {
    std::string key = (type + "\n" + initial).str();
    auto found = immutableSlots.find(key);
    if (found != immutableSlots.end())
      return found->second;
    std::string name =
        "__vrc_constant_" + std::to_string(sharedConstantIndex++);
    addSlot(name, type, initial, false);
    immutableSlots.try_emplace(key, name);
    return name;
  }

  std::string loadUnityBufferAtIndex(Location location, StringRef index,
                                     StringRef prefix) {
    if (unityBufferPool.empty())
      return {};
    std::string selected = (prefix + "_texture").str();
    addSlot(selected, "UnityEngineRenderTexture", "null", false);
    emitPush(location, "__udoncl_buffer_pool");
    emitPush(location, index);
    emitPush(location, selected);
    emitExtern(location, "SystemObjectArray.__Get__SystemInt32__SystemObject");
    return selected;
  }

  std::string loadUnityBufferFromHeapSlot(Location location, StringRef heapSlot,
                                          StringRef prefix) {
    auto entry = llvm::find_if(unityBufferPool, [&](const auto &candidate) {
      return candidate.heapSlot == heapSlot;
    });
    if (entry == unityBufferPool.end())
      return {};
    return loadUnityBufferAtIndex(
        location, immutableSlot("SystemInt32", std::to_string(entry->index)),
        prefix);
  }

  std::string selectUnityBuffer(Location location, StringRef token,
                                StringRef prefix) {
    if (unityBufferPool.empty())
      return {};
    std::string index = (prefix + "_index").str();
    addSlot(index, "SystemInt32", "-1", false);
    emitPush(location, token);
    emitPush(location, immutableSlot("SystemInt32", "1"));
    emitPush(location, index);
    emitExtern(location,
               "SystemInt32.__op_Subtraction__SystemInt32_SystemInt32__"
               "SystemInt32");
    return loadUnityBufferAtIndex(location, index, prefix);
  }

  void emitUnityBufferToken(Location location, StringRef logicalId,
                            StringRef token) {
    emitCopy(location, immutableSlot("SystemInt32", "0"), token);
    SmallVector<Operation *> done;
    for (const auto &entry : unityBufferPool) {
      std::string id;
      if (rematerializedUnityBufferIds.contains(entry.id)) {
        constexpr StringLiteral accumulator = "__vrc_return_address_i32";
        addSlot(accumulator, "SystemInt32", "0", false);
        emitI32Literal(location, static_cast<uint32_t>(entry.id), accumulator);
        id = accumulator.str();
      } else {
        id = immutableSlot("SystemInt32", std::to_string(entry.id));
      }
      constexpr StringLiteral matches = "__vrc_unity_buffer_matches";
      addSlot(matches, "SystemBoolean", "null", false);
      emitPush(location, logicalId);
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
      emitCopy(location,
               immutableSlot("SystemInt32", std::to_string(entry.index + 1)),
               token);
      done.push_back(emitJump(location, 0));
      next->setAttr("target",
                    builder.getI64IntegerAttr(static_cast<int64_t>(address)));
    }
    for (Operation *jump : done)
      jump->setAttr("target",
                    builder.getI64IntegerAttr(static_cast<int64_t>(address)));
  }

  std::string selectUnityReadableBuffer(Location location, StringRef token,
                                        StringRef prefix) {
    if (unityBufferPool.empty())
      return {};
    (void)prefix;
    constexpr StringLiteral selected = "__udoncl_readable_texture";
    constexpr StringLiteral isStatic = "__udoncl_readable_is_static";
    addSlot("__udoncl_static_tensors", "UnityEngineTexture2DArray", "null",
            true);
    addSlot(selected, "UnityEngineTexture", "null", false);
    addSlot(isStatic, "SystemBoolean", "null", false);
    emitPush(location, token);
    emitPush(location, immutableSlot("SystemInt32", "0"));
    emitPush(location, isStatic);
    emitExtern(location, "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__"
                         "SystemBoolean");
    emitPush(location, isStatic);
    Operation *dynamic =
        create(location, vm::JumpIfFalseOp::getOperationName(),
               {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
    address += 8;

    constexpr StringLiteral index = "__udoncl_readable_static_index";
    constexpr StringLiteral loaded = "__udoncl_readable_static_texture";
    addSlot(index, "SystemInt32", "0", false);
    addSlot(loaded, "UnityEngineTexture2D", "null", false);
    emitPush(location, immutableSlot("SystemInt32", "-1"));
    emitPush(location, token);
    emitPush(location, index);
    emitExtern(location,
               "SystemInt32.__op_Subtraction__SystemInt32_SystemInt32__"
               "SystemInt32");
    emitPush(location, "__udoncl_static_tensors");
    emitPush(location, index);
    emitPush(location, loaded);
    emitExtern(location, "UnityEngineTexture2DArray.__Get__SystemInt32__"
                         "UnityEngineTexture2D");
    emitCopy(location, loaded, selected);
    Operation *done = emitJump(location, 0);

    dynamic->setAttr("target",
                     builder.getI64IntegerAttr(static_cast<int64_t>(address)));
    std::string dynamicTexture =
        selectUnityBuffer(location, token, "__udoncl_readable_dynamic");
    if (dynamicTexture.empty())
      return {};
    emitCopy(location, dynamicTexture, selected);
    done->setAttr("target",
                  builder.getI64IntegerAttr(static_cast<int64_t>(address)));
    return selected.str();
  }

  void emitReturnAddress(Location location, StringRef result,
                         Operation *callSite = nullptr) {
    if (callSite) {
      auto cached = cachedReturnAddressSlots.find(callSite);
      if (cached != cachedReturnAddressSlots.end()) {
        constexpr uint64_t copyBytes = 20;
        constexpr uint64_t jumpBytes = 8;
        uint64_t continuation = address + copyBytes + jumpBytes;
        addSlot(cached->second, "SystemUInt32", std::to_string(continuation),
                false);
        emitCopy(location, cached->second, result);
        return;
      }
    }
    // Udon has no immediate PUSH. Materialize the continuation with a fixed
    // four-byte instruction sequence so its value can account for the exact
    // sequence length without reserving one Udon Heap slot per call site.
    // Each byte costs four PUSH/PUSH/PUSH/EXTERN groups (128 bytes), preceded
    // by a 20-byte zero copy and followed by a 24-byte i32 -> u32 conversion.
    constexpr uint64_t materializationBytes = 20 + 4 * 128 + 24;
    constexpr uint64_t jumpBytes = 8;
    uint64_t continuation = address + materializationBytes + jumpBytes;
    assert(continuation <= UINT32_MAX && "Udon return address exceeds 32 bits");

    constexpr StringLiteral accumulator = "__vrc_return_address_i32";
    constexpr StringLiteral highScratch = "__vrc_return_address_high";
    constexpr StringLiteral byteScratch = "__vrc_return_address_byte";
    constexpr StringLiteral shiftedScratch = "__vrc_return_address_shifted";
    addSlot(accumulator, "SystemInt32", "0", false);
    addSlot(highScratch, "SystemInt32", "0", false);
    addSlot(byteScratch, "SystemInt32", "0", false);
    addSlot(shiftedScratch, "SystemInt32", "0", false);
    std::string zero = immutableSlot("SystemInt32", "0");
    std::string nibbleShift = immutableSlot("SystemInt32", "4");
    emitCopy(location, zero, accumulator);
    for (unsigned byteIndex = 0; byteIndex < 4; ++byteIndex) {
      uint8_t byte = static_cast<uint8_t>(continuation >> (byteIndex * 8));
      std::string high =
          immutableSlot("SystemInt32", std::to_string(byte >> 4));
      std::string low = immutableSlot("SystemInt32", std::to_string(byte & 15));
      std::string shift =
          immutableSlot("SystemInt32", std::to_string(byteIndex * 8));
      emitPush(location, high);
      emitPush(location, nibbleShift);
      emitPush(location, highScratch);
      emitExtern(location,
                 "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__"
                 "SystemInt32");
      emitPush(location, highScratch);
      emitPush(location, low);
      emitPush(location, byteScratch);
      emitExtern(location,
                 "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                 "SystemInt32");
      emitPush(location, byteScratch);
      emitPush(location, shift);
      emitPush(location, shiftedScratch);
      emitExtern(location,
                 "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__"
                 "SystemInt32");
      emitPush(location, accumulator);
      emitPush(location, shiftedScratch);
      emitPush(location, accumulator);
      emitExtern(location,
                 "SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__"
                 "SystemInt32");
    }
    emitPush(location, accumulator);
    emitPush(location, result);
    emitExtern(location, "SystemConvert.__ToUInt32__SystemInt32__SystemUInt32");
    assert(address + jumpBytes == continuation &&
           "return address materialization length changed");
  }

  void emitI32Literal(Location location, uint64_t value, StringRef result) {
    assert(value <= UINT32_MAX && "i32 literal exceeds 32 bits");
    std::string nibbleShift = immutableSlot("SystemInt32", "4");
    unsigned highestNibble = 0;
    for (unsigned nibble = 1; nibble < 8; ++nibble)
      if ((value >> (nibble * 4)) & 15)
        highestNibble = nibble;
    emitCopy(location,
             immutableSlot("SystemInt32",
                           std::to_string((value >> (highestNibble * 4)) & 15)),
             result);
    for (unsigned nibble = highestNibble; nibble-- > 0;) {
      emitPush(location, result);
      emitPush(location, nibbleShift);
      emitPush(location, result);
      emitExtern(location,
                 "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__"
                 "SystemInt32");
      uint64_t digit = (value >> (nibble * 4)) & 15;
      if (!digit)
        continue;
      emitPush(location, result);
      emitPush(location, immutableSlot("SystemInt32", std::to_string(digit)));
      emitPush(location, result);
      emitExtern(location,
                 "SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__"
                 "SystemInt32");
    }
  }

  void emitSpillIndex(Location location, StringRef storageKey, unsigned index,
                      StringRef result) {
    auto cached =
        cachedSpillIndexSlots.find(spillIndexCacheKey(storageKey, index));
    if (cached == cachedSpillIndexSlots.end()) {
      emitI32Literal(location, index, result);
      return;
    }
    emitCopy(location, cached->second, result);
  }

  void emitRawAddress(Location location, uint64_t address, StringRef result) {
    auto closest = cachedRawGlobalBases.end();
    uint64_t closestDistance = UINT64_MAX;
    for (auto candidate = cachedRawGlobalBases.begin();
         candidate != cachedRawGlobalBases.end(); ++candidate) {
      uint64_t distance = candidate->first > address
                              ? candidate->first - address
                              : address - candidate->first;
      if (distance < closestDistance) {
        closest = candidate;
        closestDistance = distance;
      }
    }
    // A nearby cached address is cheaper than rebuilding a multi-byte raw
    // address and needs no additional Udon Heap slot.
    if (closest == cachedRawGlobalBases.end() || closestDistance > UINT16_MAX) {
      emitI32Literal(location, address, result);
      return;
    }
    constexpr StringLiteral delta = "__vrc_return_address_i32";
    addSlot(delta, "SystemInt32", "0", false);
    emitCopy(location, closest->second, result);
    emitI32Literal(location, closestDistance, delta);
    emitPush(location, result);
    emitPush(location, delta);
    emitPush(location, result);
    emitExtern(location,
               closest->first <= address
                   ? "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                     "SystemInt32"
                   : "SystemInt32.__op_Subtraction__SystemInt32_SystemInt32__"
                     "SystemInt32");
  }

  void emitI64Constant(Location location, const llvm::APInt &value,
                       StringRef result) {
    constexpr StringLiteral wide = "__vrc_i64_scratch_0";
    constexpr StringLiteral shifted = "__vrc_i64_scratch_1";
    constexpr StringLiteral combined = "__vrc_i64_scratch_2";
    addSlot(wide, "SystemInt64", "null", false);
    addSlot(shifted, "SystemInt64", "null", false);
    addSlot(combined, "SystemInt64", "null", false);
    for (unsigned byteIndex = 0; byteIndex < 8; ++byteIndex) {
      uint64_t byteValue = value.extractBitsAsZExtValue(8, byteIndex * 8);
      std::string seed =
          immutableSlot("SystemInt32", std::to_string(byteValue));
      emitPush(location, seed);
      emitPush(location, wide);
      emitExtern(location, "SystemConvert.__ToInt64__SystemInt32__SystemInt64");
      StringRef component = wide;
      if (byteIndex) {
        std::string shift =
            immutableSlot("SystemInt32", std::to_string(byteIndex * 8));
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
      return immutableSlot("SystemInt32", std::to_string(nibble));
    };
    std::string lowSlot = nibbleSlot(low);
    if (!high) {
      emitCopy(location, lowSlot, result);
      return;
    }
    std::string highSlot = nibbleSlot(high);
    std::string shift = immutableSlot("SystemInt32", "4");
    constexpr StringLiteral shifted = "__vrc_byte_literal_shifted";
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

  LogicalResult emitUnsignedDivRem(Location location, StringRef lhs,
                                   StringRef rhs, StringRef result,
                                   bool remainder) {
    constexpr StringLiteral offset = "__vrc_unsigned_div_offset";
    constexpr StringLiteral widened = "__vrc_unsigned_div_widened";
    constexpr StringLiteral adjusted = "__vrc_unsigned_div_adjusted";
    constexpr StringLiteral negative = "__vrc_unsigned_div_negative";
    constexpr StringLiteral unsignedLhs = "__vrc_unsigned_div_lhs";
    constexpr StringLiteral unsignedRhs = "__vrc_unsigned_div_rhs";
    constexpr StringLiteral unsignedQuotient = "__vrc_unsigned_div_quotient";
    constexpr StringLiteral quotientWide = "__vrc_unsigned_div_quotient_wide";
    constexpr StringLiteral signedQuotient =
        "__vrc_unsigned_div_signed_quotient";
    constexpr StringLiteral aboveSignedMax =
        "__vrc_unsigned_div_above_signed_max";
    constexpr StringLiteral signedMaxWide =
        "__vrc_unsigned_div_signed_max_wide";
    addSlot(offset, "SystemInt64", "null", false);
    addSlot(widened, "SystemInt64", "null", false);
    addSlot(adjusted, "SystemInt64", "null", false);
    addSlot(negative, "SystemBoolean", "null", false);
    addSlot(unsignedLhs, "SystemUInt32", "0", false);
    addSlot(unsignedRhs, "SystemUInt32", "0", false);
    addSlot(unsignedQuotient, "SystemUInt32", "0", false);
    addSlot(quotientWide, "SystemInt64", "null", false);
    addSlot(signedQuotient, "SystemInt64", "null", false);
    addSlot(aboveSignedMax, "SystemBoolean", "null", false);
    addSlot(signedMaxWide, "SystemInt64", "null", false);
    emitI64Constant(location, llvm::APInt(64, uint64_t{1} << 32), offset);

    std::string zero = immutableSlot("SystemInt32", "0");
    auto toUnsigned = [&](StringRef source, StringRef destination) {
      emitPush(location, source);
      emitPush(location, widened);
      emitExtern(location, "SystemConvert.__ToInt64__SystemInt32__SystemInt64");
      emitPush(location, source);
      emitPush(location, zero);
      emitPush(location, negative);
      emitExtern(location,
                 "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__"
                 "SystemBoolean");
      emitPush(location, negative);
      Operation *nonnegative = create(
          location, vm::JumpIfFalseOp::getOperationName(),
          {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
      address += 8;
      emitPush(location, widened);
      emitPush(location, offset);
      emitPush(location, adjusted);
      emitExtern(location,
                 "SystemInt64.__op_Addition__SystemInt64_SystemInt64__"
                 "SystemInt64");
      emitCopy(location, adjusted, widened);
      nonnegative->setAttr(
          "target", builder.getI64IntegerAttr(static_cast<int64_t>(address)));
      emitPush(location, widened);
      emitPush(location, destination);
      emitExtern(location,
                 "SystemConvert.__ToUInt32__SystemInt64__SystemUInt32");
    };
    toUnsigned(lhs, unsignedLhs);
    toUnsigned(rhs, unsignedRhs);
    emitPush(location, unsignedLhs);
    emitPush(location, unsignedRhs);
    emitPush(location, unsignedQuotient);
    emitExtern(location,
               "SystemUInt32.__op_Division__SystemUInt32_SystemUInt32__"
               "SystemUInt32");

    emitPush(location, unsignedQuotient);
    emitPush(location, quotientWide);
    emitExtern(location, "SystemConvert.__ToInt64__SystemUInt32__SystemInt64");
    std::string signedMax = immutableSlot("SystemInt32", "2147483647");
    emitPush(location, signedMax);
    emitPush(location, signedMaxWide);
    emitExtern(location, "SystemConvert.__ToInt64__SystemInt32__SystemInt64");
    emitPush(location, quotientWide);
    emitPush(location, signedMaxWide);
    emitPush(location, aboveSignedMax);
    emitExtern(location,
               "SystemInt64.__op_GreaterThan__SystemInt64_SystemInt64__"
               "SystemBoolean");
    emitCopy(location, quotientWide, signedQuotient);
    emitPush(location, aboveSignedMax);
    Operation *fitsSigned =
        create(location, vm::JumpIfFalseOp::getOperationName(),
               {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
    address += 8;
    emitPush(location, quotientWide);
    emitPush(location, offset);
    emitPush(location, signedQuotient);
    emitExtern(location,
               "SystemInt64.__op_Subtraction__SystemInt64_SystemInt64__"
               "SystemInt64");
    fitsSigned->setAttr(
        "target", builder.getI64IntegerAttr(static_cast<int64_t>(address)));
    constexpr StringLiteral quotient = "__vrc_unsigned_div_quotient_i32";
    addSlot(quotient, "SystemInt32", "0", false);
    emitPush(location, signedQuotient);
    emitPush(location, quotient);
    emitExtern(location, "SystemConvert.__ToInt32__SystemInt64__SystemInt32");
    if (!remainder) {
      emitCopy(location, quotient, result);
      return success();
    }
    constexpr StringLiteral product = "__vrc_unsigned_div_product";
    addSlot(product, "SystemInt32", "0", false);
    emitPush(location, quotient);
    emitPush(location, rhs);
    emitPush(location, product);
    emitExtern(location,
               "SystemInt32.__op_Multiplication__SystemInt32_SystemInt32__"
               "SystemInt32");
    emitPush(location, lhs);
    emitPush(location, product);
    emitPush(location, result);
    emitExtern(location,
               "SystemInt32.__op_Subtraction__SystemInt32_SystemInt32__"
               "SystemInt32");
    return success();
  }

  void emitRawWordIndex(Location location, StringRef byteAddress,
                        StringRef result) {
    std::string shift = immutableSlot("SystemInt32", "2");
    emitPush(location, byteAddress);
    emitPush(location, shift);
    emitPush(location, result);
    emitExtern(location,
               "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__"
               "SystemInt32");
  }

  void emitPackedByteLoad(Location location, StringRef byteAddress,
                          StringRef result) {
    constexpr StringLiteral wordIndex = "__vrc_raw_byte_load_word_index";
    constexpr StringLiteral byteOffset = "__vrc_raw_byte_load_offset";
    constexpr StringLiteral shift = "__vrc_raw_byte_load_shift";
    constexpr StringLiteral word = "__vrc_raw_byte_load_word";
    constexpr StringLiteral shifted = "__vrc_raw_byte_load_shifted";
    addSlot(wordIndex, "SystemInt32", "0", false);
    addSlot(byteOffset, "SystemInt32", "0", false);
    addSlot(shift, "SystemInt32", "0", false);
    addSlot(word, "SystemInt32", "0", false);
    addSlot(shifted, "SystemInt32", "0", false);
    emitRawWordIndex(location, byteAddress, wordIndex);
    emitPush(location, byteAddress);
    emitPush(location, immutableSlot("SystemInt32", "3"));
    emitPush(location, byteOffset);
    emitExtern(location,
               "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__"
               "SystemInt32");
    emitPush(location, byteOffset);
    emitPush(location, immutableSlot("SystemInt32", "3"));
    emitPush(location, shift);
    emitExtern(location, "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__"
                         "SystemInt32");
    emitPush(location, rawHeapSlot);
    emitPush(location, wordIndex);
    emitPush(location, word);
    emitExtern(location, "SystemInt32Array.__Get__SystemInt32__SystemInt32");
    emitPush(location, word);
    emitPush(location, shift);
    emitPush(location, shifted);
    emitExtern(location,
               "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__"
               "SystemInt32");
    emitPush(location, shifted);
    emitPush(location, immutableSlot("SystemInt32", "255"));
    emitPush(location, result);
    emitExtern(location,
               "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__"
               "SystemInt32");
  }

  void emitPackedByteStore(Location location, StringRef byteAddress,
                           StringRef value) {
    constexpr StringLiteral wordIndex = "__vrc_raw_byte_store_word_index";
    constexpr StringLiteral byteOffset = "__vrc_raw_byte_store_offset";
    constexpr StringLiteral shift = "__vrc_raw_byte_store_shift";
    constexpr StringLiteral word = "__vrc_raw_byte_store_word";
    constexpr StringLiteral maskedByte = "__vrc_raw_byte_store_masked_byte";
    constexpr StringLiteral shiftedByte = "__vrc_raw_byte_store_shifted_byte";
    constexpr StringLiteral shiftedMask = "__vrc_raw_byte_store_shifted_mask";
    constexpr StringLiteral inverseMask = "__vrc_raw_byte_store_inverse_mask";
    constexpr StringLiteral clearedWord = "__vrc_raw_byte_store_cleared_word";
    constexpr StringLiteral combined = "__vrc_raw_byte_store_combined";
    for (StringRef slot :
         {wordIndex, byteOffset, shift, word, maskedByte, shiftedByte,
          shiftedMask, inverseMask, clearedWord, combined})
      addSlot(slot, "SystemInt32", "0", false);
    emitRawWordIndex(location, byteAddress, wordIndex);
    emitPush(location, byteAddress);
    emitPush(location, immutableSlot("SystemInt32", "3"));
    emitPush(location, byteOffset);
    emitExtern(location,
               "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__"
               "SystemInt32");
    emitPush(location, byteOffset);
    emitPush(location, immutableSlot("SystemInt32", "3"));
    emitPush(location, shift);
    emitExtern(location, "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__"
                         "SystemInt32");
    emitPush(location, rawHeapSlot);
    emitPush(location, wordIndex);
    emitPush(location, word);
    emitExtern(location, "SystemInt32Array.__Get__SystemInt32__SystemInt32");
    emitPush(location, value);
    emitPush(location, immutableSlot("SystemInt32", "255"));
    emitPush(location, maskedByte);
    emitExtern(location,
               "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__"
               "SystemInt32");
    emitPush(location, maskedByte);
    emitPush(location, shift);
    emitPush(location, shiftedByte);
    emitExtern(location, "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__"
                         "SystemInt32");
    emitPush(location, immutableSlot("SystemInt32", "255"));
    emitPush(location, shift);
    emitPush(location, shiftedMask);
    emitExtern(location, "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__"
                         "SystemInt32");
    emitPush(location, shiftedMask);
    emitPush(location, immutableSlot("SystemInt32", "-1"));
    emitPush(location, inverseMask);
    emitExtern(location,
               "SystemInt32.__op_LogicalXor__SystemInt32_SystemInt32__"
               "SystemInt32");
    emitPush(location, word);
    emitPush(location, inverseMask);
    emitPush(location, clearedWord);
    emitExtern(location,
               "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__"
               "SystemInt32");
    emitPush(location, clearedWord);
    emitPush(location, shiftedByte);
    emitPush(location, combined);
    emitExtern(location, "SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__"
                         "SystemInt32");
    emitPush(location, rawHeapSlot);
    emitPush(location, wordIndex);
    emitPush(location, combined);
    emitExtern(location,
               "SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid");
  }

  LogicalResult emitRawLoad(Location location, StringRef pointer,
                            StringRef result, Type type, uint64_t alignment) {
    if (!type.isInteger(1) && !type.isInteger(8) && !type.isInteger(16) &&
        !type.isInteger(32) && !type.isInteger(64) && !type.isF32() &&
        !isa<host::PtrType, runtime::BufferType, runtime::EventType,
             runtime::DispatchDescType>(type))
      return failure();
    bool wide = type.isInteger(64);
    unsigned byteWidth = type.isInteger(1)    ? 1
                         : type.isInteger(8)  ? 1
                         : type.isInteger(16) ? 2
                         : wide               ? 8
                                              : 4;
    bool wordType = type.isInteger(32) || type.isF32() ||
                    isa<host::PtrType, runtime::BufferType, runtime::EventType,
                        runtime::DispatchDescType>(type);
    if (alignment >= 4 && wordType) {
      constexpr StringLiteral wordIndex = "__vrc_raw_load_word_index";
      constexpr StringLiteral word = "__vrc_raw_load_word";
      addSlot(wordIndex, "SystemInt32", "0", false);
      addSlot(word, "SystemInt32", "0", false);
      emitRawWordIndex(location, pointer, wordIndex);
      emitPush(location, rawHeapSlot);
      emitPush(location, wordIndex);
      emitPush(location, type.isF32() ? word : result);
      emitExtern(location, "SystemInt32Array.__Get__SystemInt32__SystemInt32");
      if (type.isF32()) {
        emitPush(location, word);
        emitPush(location, result);
        emitExtern(location,
                   "SystemBitConverter.__Int32BitsToSingle__SystemInt32__"
                   "SystemSingle");
      }
      return success();
    }
    std::string prefix =
        wide ? "__vrc_raw_load_i64_scratch" : "__vrc_raw_load_i32_scratch";
    std::string bits = wide ? "__vrc_i64_scratch_0" : prefix + "_bits";
    std::string scratchIndex = prefix + "_index";
    std::string scratchByte = prefix + "_byte";
    std::string scratchWide = wide ? "__vrc_i64_scratch_1" : prefix + "_wide";
    std::string scratchComponent =
        wide ? "__vrc_i64_scratch_2" : prefix + "_component";
    std::string scratchCombined =
        wide ? "__vrc_i64_scratch_3" : prefix + "_combined";
    addSlot(bits, wide ? "SystemInt64" : "SystemInt32", "0", false);
    for (unsigned byteIndex = 0; byteIndex < byteWidth; ++byteIndex) {
      std::string index = pointer.str();
      if (byteIndex) {
        std::string offset =
            immutableSlot("SystemInt32", std::to_string(byteIndex));
        index = scratchIndex;
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
      emitPackedByteLoad(location, index, byte);
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
        std::string shift =
            immutableSlot("SystemInt32", std::to_string(byteIndex * 8));
        std::string shifted = scratchComponent;
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
    if (type.isInteger(1)) {
      std::string zero = immutableSlot("SystemInt32", "0");
      emitPush(location, bits);
      emitPush(location, zero);
      emitPush(location, result);
      emitExtern(location,
                 "SystemInt32.__op_Inequality__SystemInt32_SystemInt32__"
                 "SystemBoolean");
    } else if (type.isF32()) {
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
                             StringRef value, Type type, uint64_t alignment) {
    if (!type.isInteger(1) && !type.isInteger(8) && !type.isInteger(16) &&
        !type.isInteger(32) && !type.isInteger(64) && !type.isF32() &&
        !isa<host::PtrType, runtime::BufferType, runtime::EventType,
             runtime::DispatchDescType>(type))
      return failure();
    bool wide = type.isInteger(64);
    unsigned byteWidth = type.isInteger(1)    ? 1
                         : type.isInteger(8)  ? 1
                         : type.isInteger(16) ? 2
                         : wide               ? 8
                                              : 4;
    bool wordType = type.isInteger(32) || type.isF32() ||
                    isa<host::PtrType, runtime::BufferType, runtime::EventType,
                        runtime::DispatchDescType>(type);
    if (alignment >= 4 && wordType) {
      constexpr StringLiteral wordIndex = "__vrc_raw_store_word_index";
      constexpr StringLiteral word = "__vrc_raw_store_word";
      addSlot(wordIndex, "SystemInt32", "0", false);
      addSlot(word, "SystemInt32", "0", false);
      emitRawWordIndex(location, pointer, wordIndex);
      StringRef stored = value;
      if (type.isF32()) {
        emitPush(location, value);
        emitPush(location, word);
        emitExtern(location,
                   "SystemBitConverter.__SingleToInt32Bits__SystemSingle__"
                   "SystemInt32");
        stored = word;
      }
      emitPush(location, rawHeapSlot);
      emitPush(location, wordIndex);
      emitPush(location, stored);
      emitExtern(location,
                 "SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid");
      return success();
    }
    std::string prefix =
        wide ? "__vrc_raw_store_i64_scratch" : "__vrc_raw_store_i32_scratch";
    std::string bits = wide ? "__vrc_i64_scratch_0" : prefix + "_bits";
    std::string mask = wide ? "__vrc_i64_scratch_1" : prefix + "_mask";
    std::string scratchComponent =
        wide ? "__vrc_i64_scratch_2" : prefix + "_component";
    std::string scratchByte = prefix + "_byte";
    std::string scratchWideByte =
        wide ? "__vrc_i64_scratch_3" : prefix + "_byte_wide";
    std::string scratchIndex = prefix + "_index";
    addSlot(bits, wide ? "SystemInt64" : "SystemInt32", "0", false);
    addSlot(mask, wide ? "SystemInt64" : "SystemInt32", wide ? "null" : "255",
            false);
    if (wide) {
      std::string maskSeed = immutableSlot("SystemInt32", "255");
      emitPush(location, maskSeed);
      emitPush(location, mask);
      emitExtern(location, "SystemConvert.__ToInt64__SystemInt32__SystemInt64");
    }
    if (type.isInteger(1)) {
      std::string zero = immutableSlot("SystemInt32", "0");
      std::string one = immutableSlot("SystemInt32", "1");
      emitCopy(location, zero, bits);
      emitPush(location, value);
      Operation *done = create(
          location, vm::JumpIfFalseOp::getOperationName(),
          {builder.getNamedAttr("target", builder.getI64IntegerAttr(0))});
      address += 8;
      emitCopy(location, one, bits);
      done->setAttr("target",
                    builder.getI64IntegerAttr(static_cast<int64_t>(address)));
    } else if (type.isF32()) {
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
        std::string shift =
            immutableSlot("SystemInt32", std::to_string(byteIndex * 8));
        component = scratchComponent;
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
        std::string offset =
            immutableSlot("SystemInt32", std::to_string(byteIndex));
        index = scratchIndex;
        addSlot(index, "SystemInt32", "0", false);
        emitPush(location, pointer);
        emitPush(location, offset);
        emitPush(location, index);
        emitExtern(location,
                   "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                   "SystemInt32");
      }
      emitPackedByteStore(location, index, byte);
    }
    return success();
  }

  void emitRawMemcpy(Location location, StringRef destination, StringRef source,
                     StringRef byteCount) {
    constexpr StringLiteral counter = "__vrc_raw_memcpy_counter";
    constexpr StringLiteral zero = "__vrc_raw_memcpy_zero";
    constexpr StringLiteral one = "__vrc_raw_memcpy_one";
    constexpr StringLiteral condition = "__vrc_raw_memcpy_condition";
    constexpr StringLiteral sourceIndex = "__vrc_raw_memcpy_source_index";
    constexpr StringLiteral destinationIndex =
        "__vrc_raw_memcpy_destination_index";
    constexpr StringLiteral byte = "__vrc_raw_memcpy_byte";
    constexpr StringLiteral next = "__vrc_raw_memcpy_next";
    addSlot(counter, "SystemInt32", "0", false);
    addSlot(zero, "SystemInt32", "0", false);
    addSlot(one, "SystemInt32", "1", false);
    addSlot(condition, "SystemBoolean", "null", false);
    addSlot(sourceIndex, "SystemInt32", "0", false);
    addSlot(destinationIndex, "SystemInt32", "0", false);
    addSlot(byte, "SystemInt32", "0", false);
    addSlot(next, "SystemInt32", "0", false);
    emitCopy(location, zero, counter);
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
    emitPackedByteLoad(location, sourceIndex, byte);
    emitPackedByteStore(location, destinationIndex, byte);
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

  void emitRawMemset(Location location, StringRef destination,
                     StringRef byteValue, StringRef byteCount, bool isZero) {
    (void)isZero;
    constexpr StringLiteral counter = "__vrc_raw_memset_scratch_counter";
    constexpr StringLiteral zero = "__vrc_raw_memset_scratch_zero";
    constexpr StringLiteral one = "__vrc_raw_memset_scratch_one";
    constexpr StringLiteral condition = "__vrc_raw_memset_scratch_condition";
    constexpr StringLiteral index = "__vrc_raw_memset_scratch_index";
    constexpr StringLiteral next = "__vrc_raw_memset_scratch_next";
    addSlot(counter, "SystemInt32", "0", false);
    addSlot(zero, "SystemInt32", "0", false);
    addSlot(one, "SystemInt32", "1", false);
    addSlot(condition, "SystemBoolean", "null", false);
    addSlot(index, "SystemInt32", "0", false);
    addSlot(next, "SystemInt32", "0", false);
    emitCopy(location, zero, counter);
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
    emitPush(location, destination);
    emitPush(location, counter);
    emitPush(location, index);
    emitExtern(location, "SystemInt32.__op_Addition__SystemInt32_SystemInt32__"
                         "SystemInt32");
    emitPackedByteStore(location, index, byteValue);
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
    if (!externalRawHeapInitialization) {
      addSlot(rawIndexSlot, "SystemInt32", "0", false);
      addSlot(rawValueSlot, "SystemInt32", "0", false);
      addSlot(rawNextSlot, "SystemInt32", "0", false);
      addSlot(rawOneSlot, "SystemInt32", "1", false);
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
            emitPackedByteStore(location, rawIndexSlot, rawValueSlot);
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
    for (const std::string &key : typedSpillBankOrder) {
      const TypedSpillBankPlan &bank = typedSpillBanks.at(key);
      emitI32Literal(location, bank.length, globalLengthSlot);
      emitPush(location, globalLengthSlot);
      emitPush(location, bank.slot);
      emitExtern(location,
                 bank.arrayType + ".__ctor__SystemInt32__" + bank.arrayType);
    }
    for (const GlobalBankPlan &bank : globalBanks) {
      emitI32Literal(location, bank.length, globalLengthSlot);
      emitPush(location, globalLengthSlot);
      emitPush(location, bank.slot);
      emitExtern(
          location,
          bank.elementType.isF32()
              ? "SystemSingleArray.__ctor__SystemInt32__SystemSingleArray"
              : "SystemInt32Array.__ctor__SystemInt32__SystemInt32Array");
      for (auto [base, initializer] : bank.initializers) {
        uint64_t index = base;
        for (Attribute value : initializer.getValues<Attribute>()) {
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
            emitPush(location, bank.slot);
            emitPush(location, globalIndexSlot);
            emitPush(location, valueSlot);
            emitExtern(
                location,
                bank.elementType.isF32()
                    ? "SystemSingleArray.__Set__SystemInt32_SystemSingle__"
                      "SystemVoid"
                    : "SystemInt32Array.__Set__SystemInt32_SystemInt32__"
                      "SystemVoid");
          }
          ++index;
        }
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
  llvm::DenseMap<Value, std::string> interproceduralValueSlots;
  llvm::DenseSet<Operation *> materializedConstants;
  llvm::DenseMap<Operation *, ConstantPoolLocation> pooledConstants;
  SmallVector<uint32_t> constantPoolValues;
  uint64_t constantPoolWordBase = 0;
  llvm::StringMap<std::string> scalarRegisterSlots;
  llvm::StringMap<std::string> asyncPersistentSlots;
  llvm::StringMap<GlobalPlan> globalPlans;
  llvm::StringMap<unsigned> globalBankIndices;
  SmallVector<GlobalBankPlan> globalBanks;
  llvm::StringMap<TypedSpillBankPlan> typedSpillBanks;
  SmallVector<std::string> typedSpillBankOrder;
  llvm::StringMap<std::string> sharedConstantSlots;
  llvm::StringMap<std::string> immutableSlots;
  unsigned sharedConstantIndex = 0;
  llvm::StringMap<uint64_t> rawGlobalBases;
  llvm::StringMap<DenseI8ArrayAttr> rawGlobalInitializers;
  SmallVector<std::pair<std::string, unsigned>> rawAddressCandidates;
  llvm::StringMap<std::string> cachedRawGlobalSlots;
  SmallVector<std::pair<uint64_t, std::string>> cachedRawGlobalBases;
  llvm::StringMap<std::string> cachedSpillIndexSlots;
  llvm::DenseMap<Operation *, std::string> cachedReturnAddressSlots;
  llvm::DenseMap<Operation *, uint64_t> rawAllocaBases;
  SmallVector<RawObjectPlan> rawObjects;
  SmallVector<UnityBufferPoolEntry> unityBufferPool;
  llvm::SmallSet<int64_t, 8> rematerializedUnityBufferIds;
  uint64_t rawHeapBytes = 0;
  unsigned materializationLimit = 0;
  unsigned rawAddressCacheLimit = 0;
  unsigned unityBufferIdRematerializationCount = 0;
  unsigned unityBufferIdRematerializationAddressDelta = 0;
  unsigned unityBufferIdRematerializationAttempts = 0;
  bool enforceCapacity = true;
  unsigned requestedScalarRegisterBudget = 16;
  unsigned minimumScalarRegisterCount = 16;
  unsigned plannedSpillIndexCount = 0;
  bool externalRawHeapInitialization = false;
  bool typedSpillsInserted = false;
  SmallVector<std::pair<Operation *, func::FuncOp>> callRelocations;
  SmallVector<AsyncResumeTarget> asyncResumeTargets;
  SmallVector<std::pair<Operation *, Block *>> asyncResumeRelocations;
  llvm::DenseMap<Block *, uint64_t> globalBlockAddresses;
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
  static constexpr StringLiteral asyncCompletionObjectSlot =
      "__udoncl_async_completion_object";
  static constexpr StringLiteral asyncPendingObjectSlot =
      "__udoncl_async_pending_object";
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
};

struct LowerUdonHLToUdonAsmPass
    : impl::LowerUdonHLToUdonAsmPassBase<LowerUdonHLToUdonAsmPass> {
  static unsigned addressCount(ModuleOp module) {
    auto data = module->getAttrOfType<ArrayAttr>("udonasm.data");
    llvm::StringSet<> externs;
    module.walk([&](vm::ExternOp operation) {
      externs.insert(operation.getSignature());
    });
    return (data ? data.size() : 0) + externs.size();
  }

  void runOnOperation() override {
    constexpr unsigned physicalAddressLimit = 512;
    unsigned finalDataSlotReserve = 0;
    if (auto configured = getOperation()->getAttrOfType<IntegerAttr>(
            "vrc.final_data_slot_reserve")) {
      if (configured.getInt() < 0 ||
          configured.getInt() >= physicalAddressLimit) {
        getOperation().emitError(
            "vrc.final_data_slot_reserve must be in the range 0..511");
        signalPassFailure();
        return;
      }
      finalDataSlotReserve = static_cast<unsigned>(configured.getInt());
    }
    unsigned addressLimit = physicalAddressLimit - finalDataSlotReserve;
    struct ProbeResult {
      unsigned addresses;
      unsigned dataSlots;
      bool hasLiteralAccumulator;
    };
    unsigned dataSlotCapacity = addressLimit;
    if (auto configured = getOperation()->getAttrOfType<IntegerAttr>(
            "vrc.data_slot_capacity"))
      if (configured.getInt() > 0 &&
          configured.getInt() <= physicalAddressLimit)
        dataSlotCapacity =
            std::min(addressLimit, static_cast<unsigned>(configured.getInt()));
    getOperation()->setAttr(
        "vrc.data_slot_capacity",
        IntegerAttr::get(IntegerType::get(getOperation().getContext(), 32),
                         dataSlotCapacity));
    unsigned scalarRegisterBudget = 16;
    if (auto configured = getOperation()->getAttrOfType<IntegerAttr>(
            "vrc.scalar_register_budget"))
      scalarRegisterBudget = static_cast<unsigned>(configured.getInt());
    auto probe = [&](unsigned materializationLimit, unsigned registerBudget,
                     unsigned rematerializationCount,
                     unsigned addressCacheLimit = 0) -> FailureOr<ProbeResult> {
      Operation *cloned = getOperation()->clone();
      ModuleOp module = cast<ModuleOp>(cloned);
      module->setAttr(
          "vrc.scalar_register_budget",
          IntegerAttr::get(IntegerType::get(module.getContext(), 32),
                           registerBudget));
      ScalarLoweringOptions options;
      options.materializationLimit = materializationLimit;
      options.rawAddressCacheLimit = addressCacheLimit;
      options.unityBufferIdRematerializationCount = rematerializationCount;
      options.enforceCapacity = false;
      if (failed(ScalarLowering(module, options).run())) {
        cloned->erase();
        return failure();
      }
      unsigned count = addressCount(module);
      unsigned dataSlots = 0;
      bool hasLiteralAccumulator = false;
      if (auto data = module->getAttrOfType<ArrayAttr>("udonasm.data")) {
        dataSlots = data.size();
        for (Attribute attribute : data) {
          auto slot = dyn_cast<DictionaryAttr>(attribute);
          auto name = slot ? slot.getAs<StringAttr>("name") : StringAttr();
          hasLiteralAccumulator |=
              name && name.getValue() == "__vrc_return_address_i32";
        }
      }
      cloned->erase();
      return ProbeResult{count, dataSlots, hasLiteralAccumulator};
    };
    auto pressure = [&](const ProbeResult &result) {
      unsigned totalOverflow =
          result.addresses > addressLimit ? result.addresses - addressLimit : 0;
      unsigned dataOverflow = result.dataSlots > dataSlotCapacity
                                  ? result.dataSlots - dataSlotCapacity
                                  : 0;
      return std::make_tuple(std::max(totalOverflow, dataOverflow),
                             totalOverflow + dataOverflow, result.addresses,
                             result.dataSlots);
    };
    auto requiredReduction = [&](const ProbeResult &result) {
      return std::get<0>(pressure(result));
    };
    auto unmaterialized = probe(0, scalarRegisterBudget, 0);
    if (failed(unmaterialized)) {
      signalPassFailure();
      return;
    }
    unsigned materializationLimit = 0;
    ProbeResult best = *unmaterialized;
    if (requiredReduction(best)) {
      // A fixed cap leaves large, otherwise valid Host programs with one Udon
      // Heap slot per scalar identity. Let the existing interprocedural
      // coloring and typed-spill allocator own every eligible integer
      // constant once direct declarations exceed the target address space.
      materializationLimit = best.addresses;
      auto materialized = probe(materializationLimit, scalarRegisterBudget, 0);
      if (failed(materialized)) {
        signalPassFailure();
        return;
      }
      best = *materialized;
    }
    if (requiredReduction(best) && scalarRegisterBudget > 1) {
      unsigned overflow = requiredReduction(best);
      unsigned reduction = std::min(overflow, scalarRegisterBudget - 1);
      unsigned reducedBudget = scalarRegisterBudget - reduction;
      auto reduced = probe(materializationLimit, reducedBudget, 0);
      if (failed(reduced)) {
        signalPassFailure();
        return;
      }
      if (pressure(*reduced) < pressure(best)) {
        scalarRegisterBudget = reducedBudget;
        best = *reduced;
        getOperation()->setAttr(
            "vrc.scalar_register_budget",
            IntegerAttr::get(IntegerType::get(getOperation().getContext(), 32),
                             reducedBudget));
      }
    }
    unsigned unityBufferIdRematerializationCount = 0;
    unsigned unityBufferIdRematerializationAddressDelta = 0;
    unsigned unityBufferIdRematerializationAttempts = 0;
    unsigned selectorBaselineAddresses = best.addresses;
    auto pool =
        getOperation()->getAttrOfType<ArrayAttr>("vrc.unity_buffer_pool");
    unsigned poolSize = pool ? pool.size() : 0;
    while (requiredReduction(best) &&
           unityBufferIdRematerializationAttempts < 2 &&
           unityBufferIdRematerializationCount < poolSize) {
      unsigned extra = requiredReduction(best);
      if (!unityBufferIdRematerializationCount && !best.hasLiteralAccumulator)
        ++extra;
      unsigned candidateCount =
          std::min(poolSize, unityBufferIdRematerializationCount + extra);
      if (candidateCount == unityBufferIdRematerializationCount)
        break;
      auto rematerialized =
          probe(materializationLimit, scalarRegisterBudget, candidateCount);
      ++unityBufferIdRematerializationAttempts;
      if (failed(rematerialized)) {
        signalPassFailure();
        return;
      }
      if (pressure(*rematerialized) < pressure(best)) {
        unityBufferIdRematerializationCount = candidateCount;
        best = *rematerialized;
      } else
        break;
    }
    unityBufferIdRematerializationAddressDelta =
        selectorBaselineAddresses > best.addresses
            ? selectorBaselineAddresses - best.addresses
            : 0;
    // Optional address caches consume Udon Heap addresses but can also remove
    // literal-synthesis EXTERN signatures. Trial the exact remaining capacity
    // after semantic placement instead of disabling caches for programs that
    // needed constant pooling or relying on a fixed reserve.
    unsigned rawAddressCacheLimit = std::min(
        best.dataSlots < dataSlotCapacity ? dataSlotCapacity - best.dataSlots
                                          : 0,
        best.addresses < addressLimit ? addressLimit - best.addresses : 0);
    if (rawAddressCacheLimit) {
      auto cached =
          probe(materializationLimit, scalarRegisterBudget,
                unityBufferIdRematerializationCount, rawAddressCacheLimit);
      if (failed(cached)) {
        signalPassFailure();
        return;
      }
      // Helper scratch introduced by a selected cache may make the nominal
      // slot headroom optimistic. Retry with the measured overflow removed.
      for (unsigned retry = 0; requiredReduction(*cached) && retry < 2;
           ++retry) {
        unsigned reduction = requiredReduction(*cached);
        rawAddressCacheLimit = reduction >= rawAddressCacheLimit
                                   ? 0
                                   : rawAddressCacheLimit - reduction;
        cached =
            probe(materializationLimit, scalarRegisterBudget,
                  unityBufferIdRematerializationCount, rawAddressCacheLimit);
        if (failed(cached)) {
          signalPassFailure();
          return;
        }
      }
      if (requiredReduction(*cached))
        rawAddressCacheLimit = 0;
      else
        best = *cached;
    }
    ScalarLoweringOptions options;
    options.materializationLimit = materializationLimit;
    options.rawAddressCacheLimit = rawAddressCacheLimit;
    options.unityBufferIdRematerializationCount =
        unityBufferIdRematerializationCount;
    options.unityBufferIdRematerializationAddressDelta =
        unityBufferIdRematerializationAddressDelta;
    options.unityBufferIdRematerializationAttempts =
        unityBufferIdRematerializationAttempts;
    if (failed(ScalarLowering(getOperation(), options).run())) {
      signalPassFailure();
      return;
    }
    unsigned addresses = addressCount(getOperation());
    if (addresses > addressLimit) {
      auto data = getOperation()->getAttrOfType<ArrayAttr>("udonasm.data");
      llvm::StringSet<> externs;
      getOperation().walk([&](vm::ExternOp operation) {
        externs.insert(operation.getSignature());
      });
      auto report = getOperation()->getAttrOfType<DictionaryAttr>(
          "udonasm.storage_report");
      auto arraySize = [&](StringRef name) -> size_t {
        auto value = report ? report.getAs<ArrayAttr>(name) : ArrayAttr();
        return value ? value.size() : 0;
      };
      auto categoryCounts =
          report ? report.getAs<DictionaryAttr>("slot_category_counts")
                 : DictionaryAttr();
      auto categoryCount = [&](StringRef name) -> int64_t {
        auto value = categoryCounts ? categoryCounts.getAs<IntegerAttr>(name)
                                    : IntegerAttr();
        return value ? value.getInt() : 0;
      };
      auto typeCounts = [&](StringRef name) {
        std::string text;
        llvm::raw_string_ostream stream(text);
        stream << "{";
        auto values = report ? report.getAs<ArrayAttr>(name) : ArrayAttr();
        if (values) {
          for (auto [index, value] : llvm::enumerate(values)) {
            auto item = dyn_cast<DictionaryAttr>(value);
            if (!item)
              continue;
            if (index)
              stream << ",";
            stream << item.getAs<StringAttr>("type").getValue() << ":"
                   << item.getAs<IntegerAttr>("count").getInt();
          }
        }
        stream << "}";
        stream.flush();
        return text;
      };
      if (getOperation()->hasAttr("vrc.debug_compiler_phases") && report) {
        auto dumpSlots = [&](StringRef category, StringRef name) {
          auto values = report.getAs<ArrayAttr>(name);
          if (!values)
            return;
          for (Attribute value : values) {
            auto item = dyn_cast<DictionaryAttr>(value);
            if (!item)
              continue;
            llvm::errs() << "udonclc-storage-slot category=" << category
                         << " name="
                         << item.getAs<StringAttr>("name").getValue()
                         << " type="
                         << item.getAs<StringAttr>("type").getValue() << "\n";
          }
        };
        dumpSlots("constant", "constant_slots");
        dumpSlots("other", "other_slots");
      }
      getOperation().emitError()
          << "Udon program requires " << addresses
          << " heap/extern addresses after constant materialization; limit is "
          << addressLimit << " (data=" << (data ? data.size() : 0)
          << ", externs=" << externs.size()
          << ", fixed=" << arraySize("fixed_named_slots")
          << ", scalar_registers=" << arraySize("scalar_registers")
          << ", async_persistent=" << arraySize("async_persistent_values")
          << ", typed_spill_arrays=" << arraySize("typed_spill_arrays")
          << ", typed_global_arrays=" << arraySize("typed_global_arrays")
          << ", constants=" << categoryCount("constant")
          << ", other=" << categoryCount("other")
          << ", constant_types=" << typeCounts("constant_slot_types")
          << ", other_types=" << typeCounts("other_slot_types") << ")";
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace vrc
