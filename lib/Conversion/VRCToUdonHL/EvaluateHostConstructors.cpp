// Evaluate static Host constructors at build time.
//
// A function marked `vrc.static_constructor` is one the closed-world artifact
// promises to have run exactly once before any other export. Its effect on
// Host memory is then a function of the build: this pass runs it in the Host
// evaluator and commits the memory it leaves as the initializers of the
// globals it wrote, with relocations for the words that hold an address or a
// Unity buffer token. It is the MLIR counterpart of LLVM GlobalOpt's
// evaluation of `llvm.global_ctors`.
//
// Device state is not Host memory. A Device command the constructor issued
// (a fill that primes a buffer, say) stays behind as the residual body of the
// constructor, in issue order and with constant operands; the pattern a fill
// read from Host memory is snapshotted into a constant global.
//
// The pass commits a complete evaluation or fails: a constructor that reads
// something unknown at build time, or that does not return success, is a
// compile error at the blocking operation, never a silent fallback to
// run-time initialization.
#include "VRC/Analysis/HostEvaluator.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringSet.h"

#include <map>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_EVALUATEHOSTCONSTRUCTORSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace host = ::vrc::udonhl;
namespace runtime = ::vrc::vrc;

constexpr StringLiteral kConstructorAttr = "vrc.static_constructor";
constexpr StringLiteral kEvaluatedAttr = "vrc.evaluated_static_constructor";

using Value = HostEvaluator::Value;
using Object = HostEvaluator::Object;

StringRef symbolOf(const Object &object) {
  return cast<SymbolOpInterface>(object.definition).getName();
}

/// Writes the memory the evaluation left in a raw global as its initializer.
LogicalResult commitRawGlobal(func::FuncOp constructor, const Object &object) {
  auto global = cast<host::GlobalOp>(object.definition);
  MLIRContext *context = global.getContext();
  Builder builder(context);
  SmallVector<Attribute> relocations;
  for (const auto &[offset, word] : object.words) {
    host::GlobalRelocation relocation;
    relocation.offset = static_cast<uint32_t>(offset);
    if (offset % 4)
      return constructor.emitError()
             << "static constructor leaves an address or a buffer token at "
                "the unaligned offset "
             << offset << " of @" << global.getSymName();
    if (word.kind == Value::Kind::Pointer) {
      if (word.object->kind != Object::Kind::Global)
        return constructor.emitError()
               << "static constructor leaves the address of automatic or "
                  "array storage in @"
               << global.getSymName() << " at offset " << offset;
      relocation.symbol =
          FlatSymbolRefAttr::get(context, symbolOf(*word.object));
      relocation.addend = static_cast<int32_t>(word.offset);
    } else {
      relocation.assignedBuffer = word.buffer;
    }
    relocations.push_back(relocation.build(context));
  }
  APInt bits(object.bytes.size() * 8, 0);
  bool serialized =
      isa_and_nonnull<DenseI8ArrayAttr>(global.getInitialValueAttr());
  if (!serialized)
    for (auto [index, byte] : llvm::enumerate(object.bytes))
      bits.insertBits(byte, index * 8, 8);
  Type type = global.getStorageType();
  if (serialized) {
    SmallVector<int8_t> bytes(object.bytes.begin(), object.bytes.end());
    global.setInitialValueAttr(DenseI8ArrayAttr::get(context, bytes));
  } else if (isa<host::PtrType>(type)) {
    // A raw pointer initializer is null or relocated; any other word (the
    // odd tag of a static tensor handle, say) is an integer in memory.
    if (!bits.isZero())
      global.setStorageType(builder.getI32Type());
    global.setInitialValueAttr(
        builder.getIntegerAttr(builder.getI32Type(), bits));
  } else if (type.isF32()) {
    global.setInitialValueAttr(
        builder.getFloatAttr(type, APFloat(APFloat::IEEEsingle(), bits)));
  } else {
    unsigned width = cast<IntegerType>(type).getWidth();
    global.setInitialValueAttr(
        builder.getIntegerAttr(type, bits.zextOrTrunc(width)));
  }
  if (relocations.empty()) {
    global.removeRelocationsAttr();
  } else {
    if (global.getAlignment() < 4)
      global.setAlignment(4);
    global.setRelocationsAttr(builder.getArrayAttr(relocations));
  }
  return success();
}

LogicalResult commitMemRefGlobal(func::FuncOp constructor,
                                 const Object &object) {
  auto global = cast<memref::GlobalOp>(object.definition);
  if (!object.words.empty())
    return constructor.emitError()
           << "static constructor leaves an address or a buffer token in the "
              "array @"
           << global.getSymName();
  MemRefType type = global.getType();
  auto tensor = RankedTensorType::get(type.getShape(), type.getElementType());
  SmallVector<APInt> elements;
  for (size_t offset = 0; offset < object.bytes.size(); offset += 4) {
    APInt bits(32, 0);
    for (unsigned byte = 0; byte < 4; ++byte)
      bits.insertBits(object.bytes[offset + byte], byte * 8, 8);
    elements.push_back(std::move(bits));
  }
  if (type.getElementType().isF32()) {
    SmallVector<APFloat> values;
    for (const APInt &bits : elements)
      values.emplace_back(APFloat::IEEEsingle(), bits);
    global.setInitialValueAttr(DenseElementsAttr::get(tensor, values));
  } else {
    global.setInitialValueAttr(DenseElementsAttr::get(tensor, elements));
  }
  return success();
}

/// Rebuilds the Device commands an evaluation recorded as straight-line code
/// with constant operands.
class ResidualTrace {
public:
  ResidualTrace(ModuleOp module, func::FuncOp constructor,
                const HostEvaluator &evaluator, OpBuilder &builder)
      : module(module), constructor(constructor), evaluator(evaluator),
        builder(builder), location(constructor.getLoc()) {}

  /// Emits `command` and returns its event.
  FailureOr<mlir::Value> emit(const HostEvaluator::Command &command) {
    Operation *source = command.operation;
    SmallVector<mlir::Value> operands;
    auto fill = dyn_cast<runtime::BufferFillOp>(source);
    // The descriptor is created last so that it directly precedes its
    // dispatch, as the frontend emits the pair.
    std::optional<unsigned> descriptor;
    for (auto [index, value] : llvm::enumerate(command.operands)) {
      Type type = source->getOperand(index).getType();
      mlir::Value operand;
      if (value.kind == Value::Kind::Descriptor) {
        descriptor = index;
      } else if (fill && index == 3) {
        operand = host::PtrToIntOp::create(builder, location, type,
                                           pattern(command.pattern))
                      .getResult();
      } else if (!(operand = materialize(value, type))) {
        return source->emitError("static constructor issues a Device command "
                                 "whose operand has no constant form");
      }
      operands.push_back(operand);
    }
    if (descriptor) {
      const HostEvaluator::Descriptor &created =
          evaluator.descriptor(command.operands[*descriptor].descriptor);
      SmallVector<mlir::Value> scalars;
      for (auto [index, scalar] : llvm::enumerate(created.scalars))
        scalars.push_back(materialize(
            scalar, created.operation->getOperand(index).getType()));
      Operation *copy = created.operation->clone();
      copy->setOperands(scalars);
      builder.insert(copy);
      operands[*descriptor] = copy->getResult(0);
    }
    Operation *copy = source->clone();
    copy->setOperands(operands);
    builder.insert(copy);
    return copy->getResult(0);
  }

private:
  mlir::Value constant(Type type, const APInt &bits) {
    return arith::ConstantOp::create(builder, location,
                                     builder.getIntegerAttr(type, bits));
  }

  /// A constant global holding `bytes`, shared by equal patterns.
  mlir::Value pattern(const std::vector<uint8_t> &bytes) {
    auto [entry, inserted] = patterns.try_emplace(bytes);
    if (inserted) {
      std::string symbol;
      do
        symbol =
            "__udoncl_static_fill_pattern_" + std::to_string(nextPattern++);
      while (SymbolTable::lookupSymbolIn(module, symbol));
      OpBuilder globals(module.getContext());
      globals.setInsertionPoint(constructor);
      SmallVector<int8_t> initial(bytes.begin(), bytes.end());
      OperationState state(location, host::GlobalOp::getOperationName());
      host::GlobalOp::build(
          globals, state, symbol,
          LLVM::LLVMArrayType::get(globals.getI8Type(), bytes.size()),
          DenseI8ArrayAttr::get(module.getContext(), initial), "internal", 0, 4,
          /*constant=*/true);
      globals.create(state);
      entry->second = symbol;
    }
    return host::AddressOfOp::create(builder, location,
                                     host::PtrType::get(module.getContext(), 0),
                                     entry->second);
  }

  /// The constant form of `value` as an operand of `type`, or null.
  mlir::Value materialize(const Value &value, Type type) {
    Type i32 = builder.getI32Type();
    auto token = [&](mlir::Value word) -> mlir::Value {
      if (type == word.getType())
        return word;
      if (isa<host::PtrType>(type))
        return host::IntToPtrOp::create(builder, location, type, word)
            .getResult();
      return UnrealizedConversionCastOp::create(
                 builder, location, TypeRange{type}, ValueRange{word})
          .getResult(0);
    };
    auto describe = [&](StringRef name) -> mlir::Value {
      OperationState state(location, name);
      for (NamedAttribute field : value.buffer)
        state.addAttribute(field.getName(), field.getValue());
      state.addTypes(runtime::BufferType::get(module.getContext()));
      return token(builder.create(state)->getResult(0));
    };
    switch (value.kind) {
    case Value::Kind::Float:
      return arith::ConstantOp::create(
          builder, location,
          builder.getFloatAttr(
              type,
              APFloat(cast<FloatType>(type).getFloatSemantics(), value.bits)));
    case Value::Kind::Buffer:
      return describe(value.staticBuffer
                          ? runtime::StaticBufferOp::getOperationName()
                          : runtime::AssignedBufferOp::getOperationName());
    case Value::Kind::Pointer: {
      if (value.object->kind != Object::Kind::Global)
        return {};
      mlir::Value address = host::AddressOfOp::create(
          builder, location, host::PtrType::get(module.getContext(), 0),
          symbolOf(*value.object));
      if (value.offset)
        address =
            host::PtrAddOp::create(
                builder, location, address.getType(), address,
                constant(i32, APInt(32, static_cast<uint64_t>(value.offset),
                                    /*isSigned=*/true,
                                    /*implicitTrunc=*/true)))
                .getResult();
      if (isa<host::PtrType>(type))
        return address;
      return token(host::PtrToIntOp::create(builder, location, i32, address)
                       .getResult());
    }
    case Value::Kind::Integer:
    case Value::Kind::CompleteEvent: {
      APInt bits = *HostEvaluator::wordBits(value);
      if (isa<IntegerType>(type) || type.isIndex())
        return constant(type, bits);
      return token(constant(i32, bits));
    }
    case Value::Kind::Descriptor:
      return {};
    }
    return {};
  }

  ModuleOp module;
  func::FuncOp constructor;
  const HostEvaluator &evaluator;
  OpBuilder &builder;
  Location location;
  std::map<std::vector<uint8_t>, std::string> patterns;
  unsigned nextPattern = 0;
};

/// Whether running `function` may suspend the Host, by the rule that fixes
/// the export ABI: it, or a function it transitively calls, is `may_suspend`.
bool maySuspend(ModuleOp module, func::FuncOp function) {
  llvm::StringSet<> visited;
  SmallVector<func::FuncOp> worklist{function};
  while (!worklist.empty()) {
    func::FuncOp current = worklist.pop_back_val();
    if (!visited.insert(current.getSymName()).second)
      continue;
    auto effect = current->getAttrOfType<StringAttr>("vrc.async_effect");
    if (effect && effect.getValue() == "may_suspend")
      return true;
    current.walk([&](func::CallOp call) {
      if (auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee()))
        worklist.push_back(callee);
    });
  }
  return false;
}

LogicalResult evaluateConstructor(ModuleOp module, func::FuncOp constructor,
                                  uint64_t stepBudget) {
  if (constructor.isExternal() || constructor.getNumArguments() != 0 ||
      constructor.getNumResults() > 1 ||
      (constructor.getNumResults() == 1 &&
       !constructor.getResultTypes().front().isInteger(32)))
    return constructor.emitError(
        "static constructor must be a definition without parameters that "
        "returns void or a 32-bit status");
  bool suspends = maySuspend(module, constructor);
  HostEvaluator::Options options;
  options.stepBudget = stepBudget;
  HostEvaluator evaluator(module, options);
  SmallVector<Value> results;
  if (failed(evaluator.evaluate(constructor, results)))
    return failure();
  std::optional<APInt> status;
  if (!results.empty()) {
    status = HostEvaluator::wordBits(results.front());
    if (!status || results.front().kind == Value::Kind::Float)
      return constructor.emitError(
          "static constructor does not return a build-time status");
    // The validation that used to run at start-up now fails the build.
    if (!status->isZero())
      return constructor.emitError()
             << "static constructor fails at build time with status "
             << status->getSExtValue();
  }

  // What is left of the constructor: the Device commands it issued and its
  // result. Built detached, because the commands are cloned from operations
  // the old body may own.
  //
  // Whether an export completes within its Udon event is public ABI, so a
  // constructor that could suspend stays asynchronous. Awaiting each command
  // is where the Device work scheduler places its slice budget checkpoints;
  // with no command left the constructor yields once.
  Location location = constructor.getLoc();
  OpBuilder builder(module.getContext());
  auto residual = std::make_unique<Block>();
  builder.setInsertionPointToEnd(residual.get());
  Type i32 = builder.getI32Type();
  {
    ResidualTrace trace(module, constructor, evaluator, builder);
    for (const HostEvaluator::Command &command : evaluator.commands()) {
      auto event = trace.emit(command);
      if (failed(event))
        return failure();
      if (suspends)
        runtime::AwaitOp::create(builder, location, i32, *event);
    }
  }
  if (suspends && evaluator.commands().empty()) {
    mlir::Value complete = arith::ConstantOp::create(
        builder, location, builder.getI32IntegerAttr(0));
    mlir::Value event =
        UnrealizedConversionCastOp::create(
            builder, location,
            TypeRange{runtime::EventType::get(module.getContext())},
            ValueRange{complete})
            .getResult(0);
    auto checkpoint = runtime::AwaitOp::create(builder, location, i32, event);
    checkpoint->setAttr("vrc.cooperative_checkpoint", builder.getUnitAttr());
  }
  SmallVector<mlir::Value> returned;
  if (status)
    returned.push_back(arith::ConstantOp::create(
        builder, location, builder.getIntegerAttr(i32, *status)));
  func::ReturnOp::create(builder, location, returned);

  for (const std::unique_ptr<Object> &object : evaluator.objects()) {
    if (!object->modified || object->kind == Object::Kind::Alloca)
      continue;
    if (failed(object->kind == Object::Kind::Global
                   ? commitRawGlobal(constructor, *object)
                   : commitMemRefGlobal(constructor, *object)))
      return failure();
  }

  Region &body = constructor.getBody();
  body.dropAllReferences();
  body.getBlocks().clear();
  body.push_back(residual.release());
  if (suspends)
    constructor->setAttr("vrc.async_effect",
                         builder.getStringAttr("may_suspend"));

  SmallVector<NamedAttribute> summary;
  summary.push_back(builder.getNamedAttr(
      "evaluated_operations",
      builder.getI64IntegerAttr(static_cast<int64_t>(evaluator.steps()))));
  summary.push_back(builder.getNamedAttr(
      "residual_commands", builder.getI32IntegerAttr(static_cast<int32_t>(
                               evaluator.commands().size()))));
  constructor->removeAttr(kConstructorAttr);
  constructor->setAttr(kEvaluatedAttr, builder.getDictionaryAttr(summary));
  return success();
}

struct EvaluateHostConstructorsPass
    : impl::EvaluateHostConstructorsPassBase<EvaluateHostConstructorsPass> {
  using EvaluateHostConstructorsPassBase::EvaluateHostConstructorsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> constructors;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (function->hasAttr(kConstructorAttr))
        constructors.push_back(function);
    // Constructors run in module order; each starts from the memory the
    // previous one committed.
    for (func::FuncOp constructor : constructors)
      if (failed(evaluateConstructor(module, constructor, stepBudget))) {
        signalPassFailure();
        return;
      }
  }
};

} // namespace
} // namespace vrc
