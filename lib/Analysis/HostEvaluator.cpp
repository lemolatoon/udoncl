#include "VRC/Analysis/HostEvaluator.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace vrc {
namespace {
namespace host = ::vrc::udonhl;
namespace runtime = ::vrc::vrc;

using Value = HostEvaluator::Value;
using Object = HostEvaluator::Object;

/// The token of a static tensor in a 32-bit Host word. Static tensors stay
/// disjoint from the positive tokens of the RenderTexture pool: resource 0 is
/// -1, resource 1 is -2 and so on. The scalar Udon lowering of
/// `vrc.static_buffer` and the AOT runtime's `udoncl_mem_physical_token`
/// implement the same encoding.
APInt staticBufferToken(DictionaryAttr buffer) {
  int64_t resource = buffer.getAs<IntegerAttr>("resource_id").getInt();
  return APInt(32, static_cast<uint64_t>(-resource - 1), /*isSigned=*/false,
               /*implicitTrunc=*/true);
}

/// Bytes a value of `type` occupies in raw Host memory, or zero.
unsigned memoryBytes(Type type) {
  if (isa<host::PtrType, runtime::BufferType, runtime::EventType>(type))
    return 4;
  if (auto integer = dyn_cast<IntegerType>(type)) {
    switch (integer.getWidth()) {
    case 1:
    case 8:
      return 1;
    case 16:
      return 2;
    case 32:
      return 4;
    case 64:
      return 8;
    default:
      return 0;
    }
  }
  if (type.isF32())
    return 4;
  if (type.isF64())
    return 8;
  return 0;
}

unsigned integerWidth(Type type) {
  if (type.isIndex())
    return 64;
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (isa<host::PtrType, runtime::BufferType, runtime::EventType>(type))
    return 32;
  return 0;
}
} // namespace

std::optional<APInt> HostEvaluator::wordBits(const Value &value) {
  switch (value.kind) {
  case Value::Kind::Integer:
    return value.bits;
  case Value::Kind::CompleteEvent:
    // The scalar lowering stores CL_SUCCESS as the event of a command that
    // is complete at issue.
    return APInt(32, 0);
  case Value::Kind::Buffer:
    if (value.staticBuffer)
      return staticBufferToken(value.buffer);
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

struct HostEvaluator::Frame {
  func::FuncOp function;
  llvm::DenseMap<mlir::Value, Value> values;
};

class HostEvaluator::Implementation {
public:
  explicit Implementation(HostEvaluator &evaluator) : evaluator(evaluator) {}

  LogicalResult call(func::FuncOp function, ArrayRef<Value> arguments,
                     SmallVectorImpl<Value> &results, Operation *site);

private:
  HostEvaluator &evaluator;
  SmallVector<func::FuncOp> stack;

  /// Reports why `operation` stops the evaluation.
  LogicalResult stop(Operation *operation, const Twine &reason) {
    auto diagnostic = operation->emitError()
                      << "cannot be evaluated at build time: " << reason;
    // The call path, innermost first; a recursive run is named once.
    unsigned notes = 0;
    func::FuncOp previous;
    for (func::FuncOp function : llvm::reverse(stack)) {
      if (function == previous)
        continue;
      previous = function;
      if (++notes > 16)
        break;
      diagnostic.attachNote(function.getLoc())
          << "while evaluating @" << function.getSymName();
    }
    return failure();
  }

  Object *objectFor(Operation *definition);
  FailureOr<Object *> globalObject(Operation *user, StringRef symbol);

  /// The object and byte range an access of `bytes` bytes through `pointer`
  /// touches.
  FailureOr<std::pair<Object *, uint64_t>> resolve(Operation *operation,
                                                   const Value &pointer,
                                                   uint64_t bytes, bool write);
  /// Forgets the tracked words inside [offset, offset + bytes); fails when a
  /// tracked word straddles the range boundary.
  LogicalResult clobber(Operation *operation, Object *object, uint64_t offset,
                        uint64_t bytes);
  LogicalResult load(Operation *operation, const Value &pointer, Type type,
                     Value &result);
  LogicalResult store(Operation *operation, const Value &pointer, Type type,
                      const Value &value);
  LogicalResult copy(Operation *operation, const Value &destination,
                     const Value &source, uint64_t bytes);

  FailureOr<Value> lookup(Frame &frame, mlir::Value value, Operation *user);
  /// The integer `value` is, or a diagnostic.
  FailureOr<APInt> integer(Frame &frame, mlir::Value value, Operation *user);

  LogicalResult arithmetic(Frame &frame, Operation *operation);
  LogicalResult compare(Frame &frame, arith::CmpIOp comparison);
  LogicalResult device(Frame &frame, Operation *operation, bool &handled);
  LogicalResult memory(Frame &frame, Operation *operation, bool &handled);
  LogicalResult step(Frame &frame, Operation *operation);
};

Object *HostEvaluator::Implementation::objectFor(Operation *definition) {
  auto found = evaluator.objectOf.find(definition);
  return found == evaluator.objectOf.end() ? nullptr : found->second;
}

FailureOr<Object *>
HostEvaluator::Implementation::globalObject(Operation *user, StringRef symbol) {
  Operation *definition = SymbolTable::lookupSymbolIn(evaluator.module, symbol);
  if (!definition)
    return stop(user, "@" + symbol + " does not resolve");
  if (Object *existing = objectFor(definition))
    return existing;
  auto object = std::make_unique<Object>();
  object->definition = definition;
  if (auto global = dyn_cast<host::GlobalOp>(definition)) {
    auto bytes = global.getObjectByteSize();
    if (global.getAddressSpace() != 0 || global.getLinkage() != "internal" ||
        !bytes)
      return stop(user, "@" + symbol +
                            " is not an internal raw global of known size");
    object->kind = Object::Kind::Global;
    object->alignment = static_cast<uint64_t>(global.getAlignment());
    object->constant = global.getConstant();
    object->bytes.assign(*bytes, 0);
    Attribute initial = global.getInitialValueAttr();
    if (auto serialized = dyn_cast_or_null<DenseI8ArrayAttr>(initial)) {
      for (auto [index, byte] : llvm::enumerate(serialized.asArrayRef()))
        object->bytes[index] = static_cast<uint8_t>(byte);
    } else if (auto integer = dyn_cast_or_null<IntegerAttr>(initial)) {
      APInt bits = integer.getValue().zextOrTrunc(*bytes * 8);
      for (uint64_t byte = 0; byte < *bytes; ++byte)
        object->bytes[byte] =
            static_cast<uint8_t>(bits.extractBitsAsZExtValue(8, byte * 8));
    } else if (auto floating = dyn_cast_or_null<FloatAttr>(initial)) {
      APInt bits = floating.getValue().bitcastToAPInt();
      if (bits.getBitWidth() != *bytes * 8)
        return stop(user, "@" + symbol + " has a mistyped initializer");
      for (uint64_t byte = 0; byte < *bytes; ++byte)
        object->bytes[byte] =
            static_cast<uint8_t>(bits.extractBitsAsZExtValue(8, byte * 8));
    } else if (initial && !isa<UnitAttr>(initial)) {
      return stop(user, "@" + symbol + " has an unsupported initializer");
    }
    Object *created = object.get();
    evaluator.objectOf.try_emplace(definition, created);
    evaluator.storage.push_back(std::move(object));
    if (auto relocations = global.getRelocations())
      for (Attribute attribute : *relocations) {
        auto relocation = host::GlobalRelocation::parse(attribute);
        if (!relocation)
          return stop(user, "@" + symbol + " has a malformed relocation");
        if (relocation->symbol) {
          auto target = globalObject(user, relocation->symbol.getValue());
          if (failed(target))
            return failure();
          created->words[relocation->offset] =
              Value::pointer(*target, relocation->addend);
        } else {
          Value token;
          token.kind = Value::Kind::Buffer;
          token.buffer = relocation->assignedBuffer;
          created->words[relocation->offset] = token;
        }
      }
    return created;
  }
  if (auto global = dyn_cast<memref::GlobalOp>(definition)) {
    MemRefType type = global.getType();
    auto initial = global.getInitialValue();
    auto elements =
        initial ? dyn_cast<DenseElementsAttr>(*initial) : DenseElementsAttr();
    if (type.getRank() != 1 || !type.hasStaticShape() ||
        (!type.getElementType().isInteger(32) &&
         !type.getElementType().isF32()) ||
        !elements)
      return stop(user,
                  "@" + symbol + " is not an initialized fixed i32/f32 array");
    object->kind = Object::Kind::MemRefGlobal;
    object->alignment = 4;
    object->constant = global.getConstant();
    object->bytes.reserve(type.getNumElements() * 4);
    auto append = [&](const APInt &bits) {
      for (unsigned byte = 0; byte < 4; ++byte)
        object->bytes.push_back(
            static_cast<uint8_t>(bits.extractBitsAsZExtValue(8, byte * 8)));
    };
    if (type.getElementType().isF32())
      for (APFloat value : elements.getValues<APFloat>())
        append(value.bitcastToAPInt());
    else
      for (APInt value : elements.getValues<APInt>())
        append(value);
    Object *created = object.get();
    evaluator.objectOf.try_emplace(definition, created);
    evaluator.storage.push_back(std::move(object));
    return created;
  }
  return stop(user, "@" + symbol + " is not Host memory");
}

FailureOr<std::pair<Object *, uint64_t>> HostEvaluator::Implementation::resolve(
    Operation *operation, const Value &pointer, uint64_t bytes, bool write) {
  if (pointer.kind != Value::Kind::Pointer) {
    if (pointer.isInteger())
      return stop(operation,
                  "accesses the absolute address " +
                      llvm::toString(pointer.bits, 10, /*Signed=*/false));
    return stop(operation, "accesses memory through a non-pointer value");
  }
  Object *object = pointer.object;
  if (pointer.offset < 0 ||
      static_cast<uint64_t>(pointer.offset) > object->bytes.size() ||
      bytes > object->bytes.size() - static_cast<uint64_t>(pointer.offset))
    return stop(operation, "accesses memory outside its object");
  if (write && object->constant)
    return stop(operation, "stores to constant memory");
  return std::make_pair(object, static_cast<uint64_t>(pointer.offset));
}

LogicalResult HostEvaluator::Implementation::clobber(Operation *operation,
                                                     Object *object,
                                                     uint64_t offset,
                                                     uint64_t bytes) {
  auto word = object->words.lower_bound(offset >= 3 ? offset - 3 : 0);
  while (word != object->words.end() && word->first < offset + bytes) {
    if (word->first < offset || word->first + 4 > offset + bytes)
      return stop(operation,
                  "partially overwrites a stored pointer or buffer token");
    word = object->words.erase(word);
  }
  return success();
}

LogicalResult HostEvaluator::Implementation::load(Operation *operation,
                                                  const Value &pointer,
                                                  Type type, Value &result) {
  unsigned bytes = memoryBytes(type);
  if (!bytes)
    return stop(operation, "loads an unsupported type");
  auto resolved = resolve(operation, pointer, bytes, /*write=*/false);
  if (failed(resolved))
    return failure();
  auto [object, offset] = *resolved;
  auto word = object->words.lower_bound(offset >= 3 ? offset - 3 : 0);
  if (word != object->words.end() && word->first < offset + bytes) {
    if (word->first != offset || bytes != 4 || type.isF32())
      return stop(operation, "reads part of a stored pointer or buffer token");
    result = word->second;
    return success();
  }
  APInt bits(bytes * 8, 0);
  for (unsigned byte = 0; byte < bytes; ++byte)
    bits.insertBits(object->bytes[offset + byte], byte * 8, 8);
  if (type.isInteger(1)) {
    // The Udon lowering loads a boolean as "byte != 0".
    result = Value::integer(1, !bits.isZero());
    return success();
  }
  result = Value::integer(std::move(bits));
  if (isa<FloatType>(type))
    result.kind = Value::Kind::Float;
  return success();
}

LogicalResult HostEvaluator::Implementation::store(Operation *operation,
                                                   const Value &pointer,
                                                   Type type,
                                                   const Value &value) {
  unsigned bytes = memoryBytes(type);
  if (!bytes)
    return stop(operation, "stores an unsupported type");
  auto resolved = resolve(operation, pointer, bytes, /*write=*/true);
  if (failed(resolved))
    return failure();
  auto [object, offset] = *resolved;
  if (failed(clobber(operation, object, offset, bytes)))
    return failure();
  object->modified = true;
  if (value.kind == Value::Kind::Pointer || value.isAssignedBuffer()) {
    if (bytes != 4)
      return stop(operation, "stores a pointer or buffer token narrowed or "
                             "widened from 32 bits");
    for (unsigned byte = 0; byte < 4; ++byte)
      object->bytes[offset + byte] = 0;
    object->words[offset] = value;
    return success();
  }
  APInt bits;
  if (value.kind == Value::Kind::Float) {
    bits = value.bits;
  } else if (auto word = wordBits(value)) {
    bits = *word;
  } else {
    return stop(operation, "stores a value that has no memory image");
  }
  if (type.isInteger(1))
    bits = APInt(8, bits.isZero() ? 0 : 1);
  if (bits.getBitWidth() != bytes * 8)
    return stop(operation, "stores a value of the wrong width");
  for (unsigned byte = 0; byte < bytes; ++byte)
    object->bytes[offset + byte] =
        static_cast<uint8_t>(bits.extractBitsAsZExtValue(8, byte * 8));
  return success();
}

LogicalResult HostEvaluator::Implementation::copy(Operation *operation,
                                                  const Value &destination,
                                                  const Value &source,
                                                  uint64_t bytes) {
  if (!bytes)
    return success();
  auto from = resolve(operation, source, bytes, /*write=*/false);
  auto to = resolve(operation, destination, bytes, /*write=*/true);
  if (failed(from) || failed(to))
    return failure();
  auto [sourceObject, sourceOffset] = *from;
  auto [destinationObject, destinationOffset] = *to;
  // Snapshot first: the ranges of a memmove may overlap.
  std::vector<uint8_t> payload(sourceObject->bytes.begin() + sourceOffset,
                               sourceObject->bytes.begin() + sourceOffset +
                                   bytes);
  SmallVector<std::pair<uint64_t, Value>> words;
  auto word =
      sourceObject->words.lower_bound(sourceOffset >= 3 ? sourceOffset - 3 : 0);
  for (;
       word != sourceObject->words.end() && word->first < sourceOffset + bytes;
       ++word) {
    if (word->first < sourceOffset || word->first + 4 > sourceOffset + bytes)
      return stop(operation, "copies part of a stored pointer or buffer token");
    words.emplace_back(word->first - sourceOffset, word->second);
  }
  if (failed(clobber(operation, destinationObject, destinationOffset, bytes)))
    return failure();
  destinationObject->modified = true;
  llvm::copy(payload, destinationObject->bytes.begin() + destinationOffset);
  for (auto &[offset, value] : words)
    destinationObject->words[destinationOffset + offset] = value;
  return success();
}

FailureOr<Value> HostEvaluator::Implementation::lookup(Frame &frame,
                                                       mlir::Value value,
                                                       Operation *user) {
  auto found = frame.values.find(value);
  if (found == frame.values.end())
    return stop(user, "uses a value that was not evaluated");
  return found->second;
}

FailureOr<APInt> HostEvaluator::Implementation::integer(Frame &frame,
                                                        mlir::Value value,
                                                        Operation *user) {
  auto evaluated = lookup(frame, value, user);
  if (failed(evaluated))
    return failure();
  auto bits = wordBits(*evaluated);
  if (!bits || evaluated->kind == Value::Kind::Float)
    return stop(user, "needs the integer value of an address, a buffer token "
                      "or a descriptor");
  return *bits;
}

LogicalResult HostEvaluator::Implementation::compare(Frame &frame,
                                                     arith::CmpIOp comparison) {
  Operation *operation = comparison.getOperation();
  auto lhs = lookup(frame, comparison.getLhs(), operation);
  auto rhs = lookup(frame, comparison.getRhs(), operation);
  if (failed(lhs) || failed(rhs))
    return failure();
  arith::CmpIPredicate predicate = comparison.getPredicate();
  auto set = [&](bool value) {
    frame.values[comparison.getResult()] = Value::integer(1, value);
    return success();
  };
  bool equality = predicate == arith::CmpIPredicate::eq ||
                  predicate == arith::CmpIPredicate::ne;
  bool isEqual = predicate == arith::CmpIPredicate::eq;
  auto symbolic = [](const Value &value) {
    return value.kind == Value::Kind::Pointer || value.isAssignedBuffer();
  };
  if (symbolic(*lhs) || symbolic(*rhs)) {
    const Value &left = symbolic(*lhs) ? *lhs : *rhs;
    const Value &right = symbolic(*lhs) ? *rhs : *lhs;
    if (auto bits = wordBits(right)) {
      // An object never sits at address zero (the raw heap keeps the null
      // word unmapped) and the token of an assigned buffer is positive.
      // Static tensors have negative tokens.
      bool distinct =
          bits->isZero() || (left.isAssignedBuffer() && bits->isNegative());
      if (equality && distinct)
        return set(!isEqual);
      return stop(operation, "compares an address or a buffer token with a "
                             "non-null integer");
    }
    if (left.kind != right.kind)
      return stop(operation, "compares an address with a buffer token");
    if (left.kind == Value::Kind::Buffer) {
      if (!equality)
        return stop(operation, "orders buffer tokens");
      // Tokens select textures: two assigned buffers are the same token
      // exactly when they share a texture.
      return set((left.buffer.getAs<IntegerAttr>("texture_id").getInt() ==
                  right.buffer.getAs<IntegerAttr>("texture_id").getInt()) ==
                 isEqual);
    }
    if (left.object == right.object) {
      const Value &first = symbolic(*lhs) ? left : right;
      const Value &second = symbolic(*lhs) ? right : left;
      return set(arith::applyCmpPredicate(
          predicate, APInt(64, static_cast<uint64_t>(first.offset), true),
          APInt(64, static_cast<uint64_t>(second.offset), true)));
    }
    // Distinct objects never overlap, so addresses strictly inside them
    // differ. A one-past-the-end address may equal the next object's base.
    auto inside = [](const Value &pointer) {
      return pointer.offset >= 0 && static_cast<uint64_t>(pointer.offset) <
                                        pointer.object->bytes.size();
    };
    if (equality && inside(left) && inside(right))
      return set(!isEqual);
    return stop(operation, "compares addresses of different objects");
  }
  auto left = integer(frame, comparison.getLhs(), operation);
  auto right = integer(frame, comparison.getRhs(), operation);
  if (failed(left) || failed(right))
    return failure();
  return set(arith::applyCmpPredicate(predicate, *left, *right));
}

LogicalResult HostEvaluator::Implementation::arithmetic(Frame &frame,
                                                        Operation *operation) {
  if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
    Attribute attribute = constant.getValue();
    if (auto value = dyn_cast<IntegerAttr>(attribute)) {
      frame.values[constant.getResult()] = Value::integer(value.getValue());
      return success();
    }
    if (auto value = dyn_cast<FloatAttr>(attribute)) {
      Value result = Value::integer(value.getValue().bitcastToAPInt());
      result.kind = Value::Kind::Float;
      frame.values[constant.getResult()] = result;
      return success();
    }
    return stop(operation, "has an unsupported constant");
  }
  if (auto comparison = dyn_cast<arith::CmpIOp>(operation))
    return compare(frame, comparison);
  if (auto select = dyn_cast<arith::SelectOp>(operation)) {
    auto condition = integer(frame, select.getCondition(), operation);
    if (failed(condition))
      return failure();
    auto chosen = lookup(frame,
                         condition->isZero() ? select.getFalseValue()
                                             : select.getTrueValue(),
                         operation);
    if (failed(chosen))
      return failure();
    frame.values[select.getResult()] = *chosen;
    return success();
  }
  if (operation->getNumResults() != 1)
    return stop(operation, "is not a supported operation");
  mlir::Value result = operation->getResult(0);
  unsigned width = integerWidth(result.getType());

  // Casts.
  if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, arith::IndexCastOp,
          arith::IndexCastUIOp>(operation)) {
    auto input = integer(frame, operation->getOperand(0), operation);
    if (failed(input) || !width)
      return failed(input) ? failure()
                           : stop(operation, "casts to an unsupported type");
    bool isSigned = isa<arith::ExtSIOp, arith::IndexCastOp>(operation);
    frame.values[result] = Value::integer(isSigned ? input->sextOrTrunc(width)
                                                   : input->zextOrTrunc(width));
    return success();
  }
  if (isa<arith::BitcastOp>(operation)) {
    auto input = lookup(frame, operation->getOperand(0), operation);
    if (failed(input))
      return failure();
    if (input->kind != Value::Kind::Integer &&
        input->kind != Value::Kind::Float)
      return stop(operation, "reinterprets an address or a buffer token");
    Value output = Value::integer(input->bits);
    if (isa<FloatType>(result.getType()))
      output.kind = Value::Kind::Float;
    frame.values[result] = output;
    return success();
  }
  if (operation->getNumOperands() != 2 || !width)
    return stop(operation, "is not a supported operation");

  auto lhs = lookup(frame, operation->getOperand(0), operation);
  auto rhs = lookup(frame, operation->getOperand(1), operation);
  if (failed(lhs) || failed(rhs))
    return failure();
  // Address arithmetic keeps the object: pointer +/- integer, the distance
  // of two addresses in one object, and a mask of bits the object's
  // alignment makes zero in its base address.
  if (lhs->kind == Value::Kind::Pointer || rhs->kind == Value::Kind::Pointer) {
    bool add = isa<arith::AddIOp>(operation);
    bool subtract = isa<arith::SubIOp>(operation);
    if (width != 32)
      return stop(operation, "does arithmetic on a widened address");
    if (lhs->kind == Value::Kind::Pointer &&
        rhs->kind == Value::Kind::Pointer) {
      if (subtract && lhs->object == rhs->object) {
        frame.values[result] = Value::integer(
            32, static_cast<uint64_t>(lhs->offset - rhs->offset));
        return success();
      }
      return stop(operation, "combines two addresses");
    }
    const Value &pointer = lhs->kind == Value::Kind::Pointer ? *lhs : *rhs;
    auto other = wordBits(lhs->kind == Value::Kind::Pointer ? *rhs : *lhs);
    if (!other)
      return stop(operation, "combines an address with a buffer token");
    if (add || (subtract && lhs->kind == Value::Kind::Pointer)) {
      int64_t delta = other->getSExtValue();
      frame.values[result] = Value::pointer(
          pointer.object, pointer.offset + (add ? delta : -delta));
      return success();
    }
    if (isa<arith::AndIOp>(operation) &&
        other->ult(pointer.object->alignment)) {
      frame.values[result] = Value::integer(
          32, static_cast<uint64_t>(pointer.offset) & other->getZExtValue());
      return success();
    }
    return stop(operation, "does unsupported arithmetic on an address");
  }
  auto left = integer(frame, operation->getOperand(0), operation);
  auto right = integer(frame, operation->getOperand(1), operation);
  if (failed(left) || failed(right))
    return failure();
  const APInt &a = *left;
  const APInt &b = *right;
  std::optional<APInt> value;
  // Overflow flags only license optimizations; the Udon target wraps.
  if (isa<arith::AddIOp>(operation))
    value = a + b;
  else if (isa<arith::SubIOp>(operation))
    value = a - b;
  else if (isa<arith::MulIOp>(operation))
    value = a * b;
  else if (isa<arith::AndIOp>(operation))
    value = a & b;
  else if (isa<arith::OrIOp>(operation))
    value = a | b;
  else if (isa<arith::XOrIOp>(operation))
    value = a ^ b;
  else if (isa<arith::MaxUIOp>(operation))
    value = llvm::APIntOps::umax(a, b);
  else if (isa<arith::MinUIOp>(operation))
    value = llvm::APIntOps::umin(a, b);
  else if (isa<arith::MaxSIOp>(operation))
    value = llvm::APIntOps::smax(a, b);
  else if (isa<arith::MinSIOp>(operation))
    value = llvm::APIntOps::smin(a, b);
  else if (isa<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp>(operation)) {
    if (b.uge(a.getBitWidth()))
      return stop(operation, "shifts by at least the bit width");
    unsigned amount = static_cast<unsigned>(b.getZExtValue());
    value = isa<arith::ShLIOp>(operation)    ? a.shl(amount)
            : isa<arith::ShRUIOp>(operation) ? a.lshr(amount)
                                             : a.ashr(amount);
  } else if (isa<arith::DivUIOp, arith::RemUIOp, arith::DivSIOp,
                 arith::RemSIOp>(operation)) {
    bool isSigned = isa<arith::DivSIOp, arith::RemSIOp>(operation);
    if (b.isZero() || (isSigned && a.isMinSignedValue() && b.isAllOnes()))
      return stop(operation, "divides by zero or overflows");
    value = isa<arith::DivUIOp>(operation)   ? a.udiv(b)
            : isa<arith::RemUIOp>(operation) ? a.urem(b)
            : isa<arith::DivSIOp>(operation) ? a.sdiv(b)
                                             : a.srem(b);
  }
  if (!value)
    return stop(operation, "is not a supported operation");
  frame.values[result] = Value::integer(*value);
  return success();
}

LogicalResult HostEvaluator::Implementation::memory(Frame &frame,
                                                    Operation *operation,
                                                    bool &handled) {
  handled = true;
  if (auto address = dyn_cast<host::AddressOfOp>(operation)) {
    auto object = globalObject(operation, address.getGlobal());
    if (failed(object))
      return failure();
    frame.values[address.getResult()] = Value::pointer(*object, 0);
    return success();
  }
  if (auto allocation = dyn_cast<host::AllocaOp>(operation)) {
    // The Udon lowering gives every alloca one fixed range of the raw heap,
    // so an alloca is one object for the whole evaluation.
    Object *object = objectFor(operation);
    if (!object) {
      auto bytes = integer(frame, allocation.getByteSize(), operation);
      if (failed(bytes))
        return failure();
      if (bytes->isNonPositive())
        return stop(operation, "allocates a non-positive size");
      auto created = std::make_unique<Object>();
      created->kind = Object::Kind::Alloca;
      created->definition = operation;
      created->alignment = static_cast<uint64_t>(allocation.getAlignment());
      created->bytes.assign(bytes->getZExtValue(), 0);
      object = created.get();
      evaluator.objectOf.try_emplace(operation, object);
      evaluator.storage.push_back(std::move(created));
    }
    frame.values[allocation.getResult()] = Value::pointer(object, 0);
    return success();
  }
  if (auto add = dyn_cast<host::PtrAddOp>(operation)) {
    auto pointer = lookup(frame, add.getPointer(), operation);
    auto offset = integer(frame, add.getByteOffset(), operation);
    if (failed(pointer) || failed(offset))
      return failure();
    if (pointer->kind == Value::Kind::Pointer)
      frame.values[add.getResult()] = Value::pointer(
          pointer->object, pointer->offset + offset->getSExtValue());
    else if (pointer->isInteger())
      frame.values[add.getResult()] = Value::integer(pointer->bits + *offset);
    else
      return stop(operation, "offsets a non-pointer value");
    return success();
  }
  if (isa<host::PtrToIntOp, host::IntToPtrOp>(operation)) {
    // An address keeps its object through an integer round trip.
    auto value = lookup(frame, operation->getOperand(0), operation);
    if (failed(value))
      return failure();
    frame.values[operation->getResult(0)] = *value;
    return success();
  }
  if (auto load = dyn_cast<host::LoadOp>(operation)) {
    auto pointer = lookup(frame, load.getPointer(), operation);
    Value result;
    if (failed(pointer) ||
        failed(this->load(operation, *pointer, load.getResult().getType(),
                          result)))
      return failure();
    frame.values[load.getResult()] = result;
    return success();
  }
  if (auto store = dyn_cast<host::StoreOp>(operation)) {
    auto pointer = lookup(frame, store.getPointer(), operation);
    auto value = lookup(frame, store.getValue(), operation);
    if (failed(pointer) || failed(value))
      return failure();
    return this->store(operation, *pointer, store.getValue().getType(), *value);
  }
  if (isa<host::MemcpyOp, host::MemmoveOp>(operation)) {
    auto destination = lookup(frame, operation->getOperand(0), operation);
    auto source = lookup(frame, operation->getOperand(1), operation);
    auto bytes = integer(frame, operation->getOperand(2), operation);
    if (failed(destination) || failed(source) || failed(bytes))
      return failure();
    return copy(operation, *destination, *source, bytes->getZExtValue());
  }
  if (auto fill = dyn_cast<host::MemsetOp>(operation)) {
    auto destination = lookup(frame, fill.getDestination(), operation);
    auto byte = integer(frame, fill.getByteValue(), operation);
    auto bytes = integer(frame, fill.getByteSize(), operation);
    if (failed(destination) || failed(byte) || failed(bytes))
      return failure();
    if (bytes->isZero())
      return success();
    auto resolved = resolve(operation, *destination, bytes->getZExtValue(),
                            /*write=*/true);
    if (failed(resolved))
      return failure();
    auto [object, offset] = *resolved;
    if (failed(clobber(operation, object, offset, bytes->getZExtValue())))
      return failure();
    object->modified = true;
    std::fill_n(object->bytes.begin() + offset, bytes->getZExtValue(),
                static_cast<uint8_t>(byte->getZExtValue()));
    return success();
  }
  if (isa<host::LifetimeStartOp, host::LifetimeEndOp>(operation))
    return success();
  if (auto global = dyn_cast<memref::GetGlobalOp>(operation)) {
    auto object = globalObject(operation, global.getName());
    if (failed(object))
      return failure();
    frame.values[global.getResult()] = Value::pointer(*object, 0);
    return success();
  }
  if (isa<memref::LoadOp, memref::StoreOp>(operation)) {
    auto load = dyn_cast<memref::LoadOp>(operation);
    auto store = dyn_cast<memref::StoreOp>(operation);
    mlir::Value memref = load ? load.getMemRef() : store.getMemRef();
    ValueRange indices =
        load ? ValueRange(load.getIndices()) : ValueRange(store.getIndices());
    auto base = lookup(frame, memref, operation);
    if (failed(base))
      return failure();
    if (indices.size() != 1 || base->kind != Value::Kind::Pointer ||
        base->object->kind != Object::Kind::MemRefGlobal)
      return stop(operation, "accesses an unsupported memref");
    auto index = integer(frame, indices.front(), operation);
    if (failed(index))
      return failure();
    if (index->isNegative() || index->uge(base->object->bytes.size() / 4))
      return stop(operation, "indexes outside its array");
    Value element = Value::pointer(
        base->object, static_cast<int64_t>(index->getZExtValue() * 4));
    Type type = cast<MemRefType>(memref.getType()).getElementType();
    if (load) {
      Value result;
      if (failed(this->load(operation, element, type, result)))
        return failure();
      frame.values[load.getResult()] = result;
      return success();
    }
    auto value = lookup(frame, store.getValueToStore(), operation);
    if (failed(value))
      return failure();
    return this->store(operation, element, type, *value);
  }
  handled = false;
  return success();
}

LogicalResult HostEvaluator::Implementation::device(Frame &frame,
                                                    Operation *operation,
                                                    bool &handled) {
  handled = true;
  auto describe = [&](std::initializer_list<StringRef> names) {
    SmallVector<NamedAttribute> fields;
    for (StringRef name : names)
      fields.emplace_back(StringAttr::get(operation->getContext(), name),
                          operation->getAttr(name));
    return DictionaryAttr::get(operation->getContext(), fields);
  };
  if (isa<runtime::AssignedBufferOp>(operation)) {
    Value buffer;
    buffer.kind = Value::Kind::Buffer;
    buffer.buffer =
        describe({"resource_id", "texture_id", "capacity_bytes", "format"});
    frame.values[operation->getResult(0)] = buffer;
    return success();
  }
  if (isa<runtime::StaticBufferOp>(operation)) {
    Value buffer;
    buffer.kind = Value::Kind::Buffer;
    buffer.staticBuffer = true;
    buffer.buffer = describe({"resource_id", "byte_size"});
    frame.values[operation->getResult(0)] = buffer;
    return success();
  }
  if (auto cast = dyn_cast<UnrealizedConversionCastOp>(operation)) {
    // A buffer or event token is the same word on both sides of the cast.
    if (cast.getInputs().size() != 1 || cast.getOutputs().size() != 1 ||
        integerWidth(cast.getInputs().front().getType()) != 32 ||
        integerWidth(cast.getOutputs().front().getType()) != 32)
      return stop(operation, "is not a 32-bit token conversion");
    auto value = lookup(frame, cast.getInputs().front(), operation);
    if (failed(value))
      return failure();
    frame.values[cast.getOutputs().front()] = *value;
    return success();
  }
  if (auto create = dyn_cast<runtime::DispatchDescCreateOp>(operation)) {
    Descriptor descriptor{operation, {}};
    for (mlir::Value scalar : create.getScalars()) {
      auto value = lookup(frame, scalar, operation);
      if (failed(value))
        return failure();
      if (value->kind != Value::Kind::Integer &&
          value->kind != Value::Kind::Float)
        return stop(operation, "passes an address or a buffer token as a "
                               "kernel scalar");
      descriptor.scalars.push_back(*value);
    }
    Value result;
    result.kind = Value::Kind::Descriptor;
    result.descriptor = evaluator.descriptors.size();
    evaluator.descriptors.push_back(std::move(descriptor));
    frame.values[create.getDesc()] = result;
    return success();
  }
  // A dispatch, copy or fill is one Blit on Unity's in-order graphics queue,
  // complete when the Host issues it: record it instead of executing it.
  if (isa<runtime::DispatchOp, runtime::BufferCopyOp, runtime::BufferFillOp>(
          operation)) {
    Command command{operation, {}, {}};
    for (mlir::Value operand : operation->getOperands()) {
      auto value = lookup(frame, operand, operation);
      if (failed(value))
        return failure();
      bool buffer = isa<runtime::BufferType>(operand.getType());
      bool descriptor = isa<runtime::DispatchDescType>(operand.getType());
      if (descriptor ? value->kind != Value::Kind::Descriptor
          : buffer ? (value->kind != Value::Kind::Buffer && !value->isInteger())
                   : false)
        return stop(operation, "has an operand that is not a build-time "
                               "buffer or descriptor");
      command.operands.push_back(*value);
    }
    if (auto fill = dyn_cast<runtime::BufferFillOp>(operation)) {
      // The pattern is read when the command is issued.
      auto pointer = lookup(frame, fill.getHostPointer(), operation);
      auto bytes = integer(frame, fill.getPatternSize(), operation);
      if (failed(pointer) || failed(bytes))
        return failure();
      auto resolved = resolve(operation, *pointer, bytes->getZExtValue(),
                              /*write=*/false);
      if (failed(resolved))
        return failure();
      auto [object, offset] = *resolved;
      auto word = object->words.lower_bound(offset >= 3 ? offset - 3 : 0);
      if (word != object->words.end() &&
          word->first < offset + bytes->getZExtValue())
        return stop(operation,
                    "fills with a pattern that holds an address or a buffer "
                    "token");
      command.pattern.assign(object->bytes.begin() + offset,
                             object->bytes.begin() + offset +
                                 bytes->getZExtValue());
      for (unsigned index : {1u, 2u, 4u})
        if (!command.operands[index].isInteger())
          return stop(operation, "has a range that is not a build-time "
                                 "integer");
    } else if (isa<runtime::BufferCopyOp>(operation)) {
      for (unsigned index : {2u, 3u, 4u})
        if (!command.operands[index].isInteger())
          return stop(operation, "has a range that is not a build-time "
                                 "integer");
    } else {
      for (mlir::Value index :
           cast<runtime::DispatchOp>(operation).getStaticResourceIndices())
        if (failed(integer(frame, index, operation)))
          return failure();
    }
    evaluator.trace.push_back(std::move(command));
    Value event;
    event.kind = Value::Kind::CompleteEvent;
    frame.values[operation->getResult(0)] = event;
    return success();
  }
  if (auto await = dyn_cast<runtime::AwaitOp>(operation)) {
    auto event = lookup(frame, await.getEvent(), operation);
    if (failed(event))
      return failure();
    if (event->kind != Value::Kind::CompleteEvent)
      return stop(operation, "waits for a Device event that is not complete "
                             "at issue");
    frame.values[await.getStatus()] = Value::integer(32, 0);
    return success();
  }
  handled = false;
  return success();
}

LogicalResult HostEvaluator::Implementation::step(Frame &frame,
                                                  Operation *operation) {
  bool handled = false;
  if (failed(memory(frame, operation, handled)) ||
      (!handled && failed(device(frame, operation, handled))))
    return failure();
  if (handled)
    return success();
  if (auto call = dyn_cast<func::CallOp>(operation)) {
    auto callee = evaluator.module.lookupSymbol<func::FuncOp>(call.getCallee());
    if (!callee || callee.isExternal())
      return stop(operation,
                  "calls @" + call.getCallee() + ", which has no definition");
    SmallVector<Value> arguments, results;
    for (mlir::Value operand : call.getOperands()) {
      auto value = lookup(frame, operand, operation);
      if (failed(value))
        return failure();
      arguments.push_back(*value);
    }
    if (failed(this->call(callee, arguments, results, operation)))
      return failure();
    for (auto [result, value] : llvm::zip(call.getResults(), results))
      frame.values[result] = value;
    return success();
  }
  if (operation->getDialect() &&
      operation->getDialect()->getNamespace() == "arith")
    return arithmetic(frame, operation);
  return stop(operation, "is not a function of the build");
}

LogicalResult HostEvaluator::Implementation::call(
    func::FuncOp function, ArrayRef<Value> arguments,
    SmallVectorImpl<Value> &results, Operation *site) {
  if (stack.size() >= evaluator.options.depthLimit)
    return stop(site, "exceeds the call depth limit");
  stack.push_back(function);
  Frame frame;
  frame.function = function;
  Block *block = &function.getBody().front();
  for (auto [argument, value] : llvm::zip(block->getArguments(), arguments))
    frame.values[argument] = value;
  while (true) {
    Block *next = nullptr;
    SmallVector<mlir::Value> forwarded;
    for (Operation &operation : *block) {
      if (++evaluator.executed > evaluator.options.stepBudget)
        return stop(&operation, "exceeds the evaluation step budget");
      if (auto branch = dyn_cast<cf::BranchOp>(operation)) {
        next = branch.getDest();
        llvm::append_range(forwarded, branch.getDestOperands());
        break;
      }
      if (auto branch = dyn_cast<cf::CondBranchOp>(operation)) {
        auto condition = integer(frame, branch.getCondition(), &operation);
        if (failed(condition))
          return failure();
        bool taken = !condition->isZero();
        next = taken ? branch.getTrueDest() : branch.getFalseDest();
        llvm::append_range(forwarded, taken ? branch.getTrueDestOperands()
                                            : branch.getFalseDestOperands());
        break;
      }
      if (auto branch = dyn_cast<cf::SwitchOp>(operation)) {
        auto flag = integer(frame, branch.getFlag(), &operation);
        if (failed(flag))
          return failure();
        next = branch.getDefaultDestination();
        llvm::append_range(forwarded, branch.getDefaultOperands());
        if (auto cases = branch.getCaseValues())
          for (auto [index, value] : llvm::enumerate(cases->getValues<APInt>()))
            if (value == *flag) {
              next = branch.getCaseDestinations()[index];
              forwarded.clear();
              llvm::append_range(forwarded, branch.getCaseOperands(index));
              break;
            }
        break;
      }
      if (auto done = dyn_cast<func::ReturnOp>(operation)) {
        for (mlir::Value operand : done.getOperands()) {
          auto value = lookup(frame, operand, &operation);
          if (failed(value))
            return failure();
          results.push_back(*value);
        }
        stack.pop_back();
        return success();
      }
      if (failed(step(frame, &operation)))
        return failure();
    }
    if (!next)
      return stop(&block->back(), "ends in an unsupported terminator");
    // Block arguments are assigned in parallel.
    SmallVector<Value> incoming;
    for (mlir::Value value : forwarded) {
      auto evaluated = lookup(frame, value, &block->back());
      if (failed(evaluated))
        return failure();
      incoming.push_back(*evaluated);
    }
    for (auto [argument, value] : llvm::zip(next->getArguments(), incoming))
      frame.values[argument] = value;
    block = next;
  }
}

HostEvaluator::HostEvaluator(ModuleOp module, Options options)
    : module(module), options(options) {}

HostEvaluator::~HostEvaluator() = default;

LogicalResult HostEvaluator::evaluate(func::FuncOp function,
                                      SmallVectorImpl<Value> &results) {
  if (function.isExternal() || function.getNumArguments() != 0)
    return function.emitError(
        "build-time evaluation requires a definition without parameters");
  Implementation implementation(*this);
  return implementation.call(function, {}, results, function.getOperation());
}

} // namespace vrc
