#include "HostBufferPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace mlir;
using namespace llvm;

namespace udoncl {

namespace {
using ConstantArguments = SmallVector<std::optional<APInt>>;

std::optional<APInt> evaluateInteger(llvm::Value *value,
                                     ArrayRef<std::optional<APInt>> arguments,
                                     std::set<llvm::Value *> &active) {
  if (auto *constant = dyn_cast<ConstantInt>(value))
    return constant->getValue();
  if (auto *argument = dyn_cast<Argument>(value))
    return argument->getArgNo() < arguments.size()
               ? arguments[argument->getArgNo()]
               : std::nullopt;
  if (!active.insert(value).second)
    return std::nullopt;
  auto done = [&](std::optional<APInt> result) {
    active.erase(value);
    return result;
  };
  if (auto *freeze = dyn_cast<FreezeInst>(value))
    return done(evaluateInteger(freeze->getOperand(0), arguments, active));
  if (auto *cast = dyn_cast<CastInst>(value)) {
    auto operand = evaluateInteger(cast->getOperand(0), arguments, active);
    unsigned width = cast->getType()->getIntegerBitWidth();
    if (!operand)
      return done(std::nullopt);
    if (isa<SExtInst>(cast))
      return done(operand->sextOrTrunc(width));
    return done(operand->zextOrTrunc(width));
  }
  if (auto *binary = dyn_cast<BinaryOperator>(value)) {
    auto left = evaluateInteger(binary->getOperand(0), arguments, active);
    auto right = evaluateInteger(binary->getOperand(1), arguments, active);
    if (!left || !right)
      return done(std::nullopt);
    switch (binary->getOpcode()) {
    case Instruction::Add:
      return done(*left + *right);
    case Instruction::Sub:
      return done(*left - *right);
    case Instruction::Mul:
      return done(*left * *right);
    case Instruction::Shl:
      return done(left->shl(right->getLimitedValue()));
    case Instruction::LShr:
      return done(left->lshr(right->getLimitedValue()));
    case Instruction::AShr:
      return done(left->ashr(right->getLimitedValue()));
    case Instruction::And:
      return done(*left & *right);
    case Instruction::Or:
      return done(*left | *right);
    case Instruction::Xor:
      return done(*left ^ *right);
    default:
      return done(std::nullopt);
    }
  }
  if (auto *select = dyn_cast<SelectInst>(value)) {
    auto condition = evaluateInteger(select->getCondition(), arguments, active);
    if (condition)
      return done(evaluateInteger(
          select->getOperand(condition->isZero() ? 2 : 1), arguments, active));
    auto yes = evaluateInteger(select->getTrueValue(), arguments, active);
    auto no = evaluateInteger(select->getFalseValue(), arguments, active);
    return done(yes && no && *yes == *no ? yes : std::nullopt);
  }
  return done(std::nullopt);
}

LogicalResult scanAllocationCalls(llvm::Function &function,
                                  ArrayRef<std::optional<APInt>> arguments,
                                  HostBufferPlan &plan,
                                  std::set<llvm::Function *> &activeFunctions,
                                  raw_ostream &diagnostics) {
  if (!activeFunctions.insert(&function).second) {
    diagnostics << "recursive Host allocation call graph is unsupported at "
                << function.getName() << "\n";
    return failure();
  }
  DominatorTree dominators(function);
  LoopInfo loops(dominators);
  for (llvm::BasicBlock &block : function) {
    for (llvm::Instruction &instruction : block) {
      auto *call = dyn_cast<CallBase>(&instruction);
      llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
      if (!callee)
        continue;
      if (callee->getName() == "clCreateBuffer") {
        if (loops.getLoopFor(&block)) {
          diagnostics << "clCreateBuffer in a loop has no proven finite "
                         "allocation bound in "
                      << function.getName() << "\n";
          activeFunctions.erase(&function);
          return failure();
        }
        std::set<llvm::Value *> activeValues;
        auto bytes = call->arg_size() == 5
                         ? evaluateInteger(call->getArgOperand(2), arguments,
                                           activeValues)
                         : std::nullopt;
        if (!bytes || bytes->isZero() || bytes->getBitWidth() > 64 ||
            bytes->getActiveBits() > 63) {
          diagnostics << "clCreateBuffer size must be a positive compile-time "
                         "constant in "
                      << function.getName() << "\n";
          activeFunctions.erase(&function);
          return failure();
        }
        plan.extraBufferBytes.push_back(
            static_cast<int64_t>(bytes->getZExtValue()));
        continue;
      }
      if (callee->isDeclaration() || callee->isIntrinsic())
        continue;
      ConstantArguments calleeArguments;
      for (llvm::Value *operand : call->args()) {
        std::set<llvm::Value *> activeValues;
        calleeArguments.push_back(
            evaluateInteger(operand, arguments, activeValues));
      }
      if (failed(scanAllocationCalls(*callee, calleeArguments, plan,
                                     activeFunctions, diagnostics))) {
        activeFunctions.erase(&function);
        return failure();
      }
    }
  }
  activeFunctions.erase(&function);
  return success();
}
} // namespace

FailureOr<HostBufferPlan> planHostBuffers(llvm::Module &module,
                                          int64_t baseCapacity,
                                          raw_ostream &diagnostics) {
  if (baseCapacity < 0) {
    diagnostics << "base mutable handle capacity must be non-negative\n";
    return failure();
  }
  HostBufferPlan plan;
  plan.baseCapacity = baseCapacity;
  SmallVector<llvm::Function *> roots;
  auto *annotations = module.getNamedGlobal("llvm.global.annotations");
  auto *entries =
      annotations
          ? dyn_cast_or_null<ConstantArray>(annotations->getInitializer())
          : nullptr;
  if (annotations && !entries) {
    diagnostics << "llvm.global.annotations has an unsupported initializer\n";
    return failure();
  }
  if (entries)
    for (Use &use : entries->operands()) {
      auto *entry = dyn_cast<ConstantStruct>(use.get());
      if (!entry || entry->getNumOperands() < 2)
        continue;
      auto *function =
          dyn_cast<llvm::Function>(entry->getOperand(0)->stripPointerCasts());
      auto *annotationGlobal = dyn_cast<GlobalVariable>(
          getUnderlyingObject(entry->getOperand(1), /*MaxLookup=*/6));
      auto *annotation = annotationGlobal
                             ? dyn_cast_or_null<ConstantDataArray>(
                                   annotationGlobal->getInitializer())
                             : nullptr;
      if (function && annotation && annotation->isCString() &&
          annotation->getAsCString() == "udoncl.export")
        roots.push_back(function);
    }
  if (roots.empty()) {
    // A generated ONNX Host can be used without a user C export layer. Its
    // complete allocation plan is already represented by baseCapacity.
    return plan;
  }
  for (llvm::Function *root : roots) {
    ConstantArguments arguments(root->arg_size());
    std::set<llvm::Function *> activeFunctions;
    if (failed(scanAllocationCalls(*root, arguments, plan, activeFunctions,
                                   diagnostics)))
      return failure();
  }
  return plan;
}

LogicalResult writeHostBufferPlan(const HostBufferPlan &plan,
                                  StringRef jsonPath, StringRef headerPath,
                                  raw_ostream &diagnostics) {
  json::Array extra;
  for (int64_t bytes : plan.extraBufferBytes)
    extra.push_back(bytes);
  json::Object object{
      {"schema_version", 1},
      {"base_capacity", plan.baseCapacity},
      {"extra_buffer_bytes", std::move(extra)},
      {"total_capacity",
       plan.baseCapacity + static_cast<int64_t>(plan.extraBufferBytes.size())},
  };
  std::error_code error;
  raw_fd_ostream jsonOutput(jsonPath, error);
  if (error) {
    diagnostics << "cannot write Host buffer plan " << jsonPath << ": "
                << error.message() << "\n";
    return failure();
  }
  jsonOutput << formatv("{0:2}\n", json::Value(std::move(object)));
  jsonOutput.close();
  raw_fd_ostream headerOutput(headerPath, error);
  if (error) {
    diagnostics << "cannot write Host buffer capacity header " << headerPath
                << ": " << error.message() << "\n";
    return failure();
  }
  int64_t total =
      plan.baseCapacity + static_cast<int64_t>(plan.extraBufferBytes.size());
  headerOutput << "#define UDONCL_AOT_BUFFER_CAPACITY " << total << "\n"
               << "#define UDONCL_DEVICE_BUFFER_CAPACITY " << total << "\n";
  return success();
}

LogicalResult applyHostBufferPlan(ModuleOp module, StringRef jsonPath) {
  auto input = MemoryBuffer::getFile(jsonPath);
  if (!input)
    return module.emitError() << "cannot read Host buffer plan " << jsonPath;
  auto parsed = json::parse((*input)->getBuffer());
  if (!parsed)
    return module.emitError()
           << "invalid Host buffer plan: " << toString(parsed.takeError());
  auto *object = parsed->getAsObject();
  auto schema = object ? object->getInteger("schema_version") : std::nullopt;
  auto base = object ? object->getInteger("base_capacity") : std::nullopt;
  auto total = object ? object->getInteger("total_capacity") : std::nullopt;
  auto *extra = object ? object->getArray("extra_buffer_bytes") : nullptr;
  auto capacity =
      module->getAttrOfType<IntegerAttr>("udoncl.mutable_handle_capacity");
  auto bytes =
      module->getAttrOfType<DenseI64ArrayAttr>("udoncl.mutable_buffer_bytes");
  if (schema != 1 || !base || !total || !extra || !capacity || !bytes ||
      capacity.getInt() != *base ||
      static_cast<int64_t>(bytes.size()) != *base ||
      *total != *base + static_cast<int64_t>(extra->size()))
    return module.emitError(
        "Host buffer plan is inconsistent with ONNX capacity");
  SmallVector<int64_t> combined(bytes.asArrayRef());
  for (const json::Value &value : *extra) {
    auto item = value.getAsInteger();
    if (!item || *item <= 0)
      return module.emitError("Host buffer plan contains an invalid byte size");
    combined.push_back(*item);
  }
  Builder builder(module.getContext());
  module->setAttr("udoncl.mutable_buffer_bytes",
                  builder.getDenseI64ArrayAttr(combined));
  module->setAttr("udoncl.mutable_handle_capacity",
                  builder.getI64IntegerAttr(*total));
  return success();
}

} // namespace udoncl
