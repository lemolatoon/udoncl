#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

struct Slot {
  std::string name;
  std::string type;
  std::string initial;
  bool isPublic = false;
};

static FailureOr<StringRef> udonType(Type type) {
  if (type.isInteger(1))
    return StringRef("SystemBoolean");
  if (type.isInteger(32))
    return StringRef("SystemInt32");
  if (type.isF32())
    return StringRef("SystemSingle");
  return failure();
}

static std::string sanitize(StringRef value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value)
    result.push_back(llvm::isAlnum(c) || c == '_' ? c : '_');
  if (result.empty() || llvm::isDigit(result.front()))
    result.insert(result.begin(), '_');
  return result;
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
    llvm::SmallSet<int64_t, 8> exportIds;
    llvm::SmallSet<std::string, 8> eventNames;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      auto exportId = function->getAttrOfType<IntegerAttr>("vrc.export_id");
      if (!exportId)
        return function.emitError(
            "initial Udon backend requires every remaining function to be an "
            "export; inline or lower internal calls first");
      if (exportId.getInt() <= 0 || !exportIds.insert(exportId.getInt()).second)
        return function.emitError("requires a unique positive vrc.export_id");
      if (function.isExternal() || function.isPrivate())
        return function.emitError(
            "initial scalar Udon lowering requires a public, defined function");
      if (function.getNumResults() > 1)
        return function.emitError(
            "initial scalar Udon ABI supports at most one result");
      for (Type type : function.getArgumentTypes())
        if (failed(udonType(type)))
          return function.emitError("unsupported scalar Udon argument type");
      for (Type type : function.getResultTypes())
        if (failed(udonType(type)))
          return function.emitError("unsupported scalar Udon result type");
      auto event = function->getAttrOfType<StringAttr>("vrc.udon_event");
      std::string eventName =
          event ? event.getValue().str()
                : ("VRC_Call_" + sanitize(function.getSymName()) + "_" +
                   std::to_string(exportId.getInt()));
      if (eventName.empty() || eventName.front() == '_' ||
          sanitize(eventName) != eventName ||
          !eventNames.insert(eventName).second)
        return function.emitError("Udon custom event must be a unique "
                                  "assembler identifier not starting with '_'");
      functions.push_back(function);
    }
    if (functions.empty())
      return module.emitError(
          "scalar Udon lowering requires at least one export");

    for (func::FuncOp function : functions)
      if (failed(lower(function)))
        return failure();

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
    auto explicitEvent = function->getAttrOfType<StringAttr>("vrc.udon_event");
    std::string prefix = "__vrc_f" + std::to_string(exportId.getInt()) + "_";
    std::string event = explicitEvent
                            ? explicitEvent.getValue().str()
                            : "VRC_Call_" + sanitize(function.getSymName()) +
                                  "_" + std::to_string(exportId.getInt());

    SmallVector<NamedAttribute> exportFields;
    exportFields.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(event)));
    exportFields.push_back(builder.getNamedAttr(
        "address", builder.getI64IntegerAttr(static_cast<int64_t>(address))));
    exportFields.push_back(builder.getNamedAttr(
        "function", builder.getStringAttr(function.getSymName())));
    exportFields.push_back(builder.getNamedAttr("export_id", exportId));
    exports.push_back(DictionaryAttr::get(module.getContext(), exportFields));

    llvm::DenseMap<Value, std::string> valueSlots;
    for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
      auto type = *udonType(argument.getType());
      std::string name = "VRC_ARG_" + std::to_string(exportId.getInt()) + "_" +
                         std::to_string(index);
      addSlot(name, type, "null", true);
      valueSlots[argument] = name;
    }
    std::string resultSlot;
    if (function.getNumResults()) {
      auto type = *udonType(function.getResultTypes().front());
      resultSlot = "VRC_RET_" + std::to_string(exportId.getInt());
      addSlot(resultSlot, type, "null", true);
    }
    addSlot("VRC_STATUS_" + std::to_string(exportId.getInt()), "SystemInt32",
            "0", true);

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
        if (operation.getNumResults() != 1)
          return operation.emitError(
              "initial Udon lowering supports one result per operation");
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
        if (auto returned = dyn_cast<func::ReturnOp>(operation)) {
          if (returned.getNumOperands()) {
            auto found = valueSlots.find(returned.getOperand(0));
            if (found == valueSlots.end())
              return returned.emitError(
                  "return value has no allocated Udon heap slot");
            emitCopy(returned.getLoc(), found->second, resultSlot);
          }
          emitJump(returned.getLoc(), 0xfffffffcLL);
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
