// Fold loads from Host memory no reachable code can store to.
//
// A raw global of a linked Host program is closed-world state: every store
// that can ever reach it is an operation of this module (or the completion of
// a readback this module issues). A byte range no such store can touch keeps
// its initializer for the life of the Udon instance, under every interleaving
// of export events and continuations, so a load from it is a constant. This
// is the MLIR counterpart of LLVM GlobalOpt's constant-global folding, for
// the byte memory model of `udonhl`.
//
// The analysis is flow-insensitive and field-sensitive. Every integer or
// pointer SSA value gets a set of (object, offset) targets it may be an
// address of, where an offset is a congruence class `base + stride * i`:
// exact for a field, strided for an element of an array of records. Memory
// holds, per object, which congruence classes are stored to and what the
// stored values may point to, so an address loaded back from a table of
// handles still names its record. Addresses only originate at
// `udonhl.address_of` and `udonhl.alloca`; a value the analysis cannot follow
// *exposes* the objects it may address, and an untracked store clobbers
// exactly the exposed objects. An address that only ever sits in tracked
// cells exposes nothing.
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

#include <map>
#include <numeric>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_FOLDIMMUTABLEHOSTMEMORYPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace host = ::vrc::udonhl;
namespace runtime = ::vrc::vrc;

/// The integers `base + stride * i`. A zero stride is the single integer
/// `base`; a stride of one is every integer.
struct Congruence {
  int64_t base = 0;
  uint64_t stride = 0;

  static Congruence exact(int64_t value) { return {value, 0}; }
  static Congruence any() { return {0, 1}; }
  bool isExact() const { return stride == 0; }
  bool operator==(const Congruence &other) const {
    return base == other.base && stride == other.stride;
  }

  static Congruence normalized(int64_t base, uint64_t stride) {
    if (stride == 0)
      return {base, 0};
    int64_t modulus = static_cast<int64_t>(stride);
    int64_t reduced = base % modulus;
    return {reduced < 0 ? reduced + modulus : reduced, stride};
  }
  Congruence join(const Congruence &other) const {
    uint64_t distance =
        base > other.base
            ? static_cast<uint64_t>(base) - static_cast<uint64_t>(other.base)
            : static_cast<uint64_t>(other.base) - static_cast<uint64_t>(base);
    return normalized(base, std::gcd(std::gcd(stride, other.stride), distance));
  }
  Congruence add(const Congruence &other) const {
    return normalized(base + other.base, std::gcd(stride, other.stride));
  }
  Congruence negate() const { return normalized(-base, stride); }
  Congruence multiply(int64_t factor) const {
    uint64_t magnitude = factor < 0 ? static_cast<uint64_t>(-factor)
                                    : static_cast<uint64_t>(factor);
    return normalized(base * factor, stride * magnitude);
  }
};

/// Whether a `width`-byte access at some offset of `left` can share a byte
/// with a `otherWidth`-byte access at some offset of `right`.
bool mayOverlap(Congruence left, uint64_t width, Congruence right,
                uint64_t otherWidth) {
  uint64_t stride = std::gcd(left.stride, right.stride);
  if (stride == 0)
    return left.base < right.base + static_cast<int64_t>(otherWidth) &&
           right.base < left.base + static_cast<int64_t>(width);
  if (width >= stride || otherWidth >= stride)
    return true;
  int64_t modulus = static_cast<int64_t>(stride);
  int64_t distance = (right.base - left.base) % modulus;
  if (distance < 0)
    distance += modulus;
  // `right` starts `distance` bytes after `left`, or `stride - distance`
  // bytes before it.
  return static_cast<uint64_t>(distance) < width ||
         stride - static_cast<uint64_t>(distance) < otherWidth;
}

struct Object;

/// What an SSA value, or the content of memory, may be: an integer, an
/// address inside one of `targets`, or (when `unknown`) something the
/// analysis lost, which can only be an address inside an exposed object.
struct Info {
  /// A null object is the plain integer part of the value.
  SmallVector<std::pair<Object *, Congruence>, 1> targets;
  bool unknown = false;

  static Info integer(Congruence value) {
    Info info;
    info.targets.emplace_back(nullptr, value);
    return info;
  }
  static Info anyInteger() { return integer(Congruence::any()); }
  static Info lost() {
    Info info;
    info.unknown = true;
    return info;
  }
  bool addressesObjects() const {
    return llvm::any_of(targets,
                        [](const auto &target) { return target.first; });
  }
  bool join(Object *object, Congruence offset) {
    for (auto &target : targets)
      if (target.first == object) {
        Congruence joined = target.second.join(offset);
        if (joined == target.second)
          return false;
        target.second = joined;
        return true;
      }
    targets.emplace_back(object, offset);
    return true;
  }
  bool join(const Info &other) {
    bool changed = other.unknown && !unknown;
    unknown |= other.unknown;
    for (const auto &target : other.targets)
      changed |= join(target.first, target.second);
    return changed;
  }
};

/// One store the program may perform into an object.
struct StoreRecord {
  Congruence offset;
  uint64_t width;
  /// What the stored bytes may point to.
  Info content;
};

struct Object {
  /// The `udonhl.global` or `udonhl.alloca`.
  Operation *definition = nullptr;
  host::GlobalOp global;
  /// Byte size when the object can be folded.
  std::optional<uint64_t> bytes;
  std::vector<StoreRecord> stores;
  /// Its address reached something the analysis does not follow.
  bool exposed = false;
  /// An untracked operation was handed its address.
  bool clobbered = false;
};

unsigned accessBytes(Type type) {
  if (isa<host::PtrType, runtime::BufferType, runtime::EventType>(type))
    return 4;
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() == 1 ? 1 : integer.getWidth() / 8;
  if (type.isF32())
    return 4;
  if (type.isF64())
    return 8;
  return 0;
}

bool carriesAddress(Type type) {
  if (isa<host::PtrType>(type))
    return cast<host::PtrType>(type).getAddressSpace() == 0;
  return isa<IntegerType>(type) || type.isIndex();
}

class Analysis {
public:
  explicit Analysis(ModuleOp module) : module(module) {}

  void run() {
    for (host::GlobalOp global : module.getOps<host::GlobalOp>())
      objectFor(global.getOperation());
    // The transfer functions are monotone over a lattice of finite height:
    // an offset only widens to a coarser congruence class.
    do {
      changed = false;
      for (func::FuncOp function : module.getOps<func::FuncOp>())
        if (!function.isExternal())
          visit(function);
      propagateExposure();
    } while (changed);
  }

  /// The single address `value` can be, if it is one inside a global.
  std::optional<std::pair<Object *, int64_t>> exactAddress(Value value) {
    const Info &info = infoOf(value);
    if (info.unknown || info.targets.size() != 1 || !info.targets[0].first ||
        !info.targets[0].second.isExact())
      return std::nullopt;
    return std::make_pair(info.targets[0].first, info.targets[0].second.base);
  }

  /// Whether some store may write a byte of [offset, offset + width).
  bool mayBeStored(const Object &object, int64_t offset, uint64_t width) {
    if (object.clobbered || (object.exposed && unknownStore))
      return true;
    return llvm::any_of(object.stores, [&](const StoreRecord &store) {
      return mayOverlap(Congruence::exact(offset), width, store.offset,
                        store.width);
    });
  }

  /// Why loads of the global `symbol` are not folded, for `explain`.
  std::string describe(StringRef symbol) {
    Operation *definition = SymbolTable::lookupSymbolIn(module, symbol);
    Object *object = definition ? objectOf.lookup(definition) : nullptr;
    if (!object)
      return "";
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << (object->clobbered ? " clobbered" : "")
           << (object->exposed ? " exposed" : "")
           << (object->exposed && unknownStore ? " unknown-store" : "")
           << " stores=" << object->stores.size();
    return text;
  }

  bool isStored(const Object &object) {
    return object.clobbered || (object.exposed && unknownStore) ||
           !object.stores.empty();
  }

  llvm::iterator_range<std::vector<std::unique_ptr<Object>>::iterator>
  objects() {
    return llvm::make_range(storage.begin(), storage.end());
  }

private:
  ModuleOp module;
  std::vector<std::unique_ptr<Object>> storage;
  llvm::DenseMap<Operation *, Object *> objectOf;
  llvm::DenseMap<Value, Info> infos;
  llvm::DenseMap<Operation *, Info> returns;
  /// Some store goes through an address the analysis lost.
  bool unknownStore = false;
  bool changed = false;
  const Info none;

  Object *objectFor(Operation *definition) {
    auto [entry, inserted] = objectOf.try_emplace(definition, nullptr);
    if (!inserted)
      return entry->second;
    auto object = std::make_unique<Object>();
    object->definition = definition;
    if (auto global = dyn_cast<host::GlobalOp>(definition)) {
      object->global = global;
      // Only an internal global is closed-world state.
      if (global.getLinkage() == "internal" && global.getAddressSpace() == 0)
        object->bytes = global.getObjectByteSize();
      else
        object->clobbered = true;
    }
    entry->second = object.get();
    storage.push_back(std::move(object));
    return entry->second;
  }

  const Info &infoOf(Value value) {
    auto found = infos.find(value);
    return found == infos.end() ? none : found->second;
  }

  void join(Value value, const Info &info) {
    // `info` may live in the map the insertion below rehashes.
    Info incoming = info;
    if (infos[value].join(incoming))
      changed = true;
  }

  void expose(const Info &info) {
    for (const auto &target : info.targets)
      if (target.first && !target.first->exposed) {
        target.first->exposed = true;
        changed = true;
      }
  }

  /// An exposed object hands out every address stored in it, and the
  /// addresses its initializer relocates.
  void propagateExposure() {
    for (const std::unique_ptr<Object> &object : storage) {
      if (!object->exposed)
        continue;
      for (const StoreRecord &store : object->stores)
        expose(store.content);
      if (object->global)
        expose(initialContent(*object, Congruence::any(), 1));
    }
  }

  /// What the initializer holds in the bytes an access may read. Automatic
  /// storage has no initial content: reading it before a store is undefined.
  Info initialContent(Object &object, Congruence offset, uint64_t width) {
    Info info;
    if (!object.global)
      return info;
    if (auto relocations = object.global.getRelocations())
      for (Attribute attribute : *relocations) {
        auto relocation = host::GlobalRelocation::parse(attribute);
        if (!relocation ||
            !mayOverlap(offset, width, Congruence::exact(relocation->offset),
                        4))
          continue;
        bool whole =
            offset.isExact() && width == 4 && offset.base == relocation->offset;
        Info word;
        if (relocation->symbol) {
          if (Operation *target = SymbolTable::lookupSymbolIn(
                  module, relocation->symbol.getValue()))
            word.join(objectFor(target), Congruence::exact(relocation->addend));
          else
            word = Info::lost();
        } else {
          word = Info::anyInteger();
        }
        // The access reads exactly this word, or it also reads plain bytes.
        if (whole)
          return word;
        info.join(word);
      }
    info.join(nullptr, Congruence::any());
    return info;
  }

  Info loaded(const Info &pointer, uint64_t width) {
    if (pointer.unknown)
      return Info::lost();
    Info result;
    for (const auto &[object, offset] : pointer.targets) {
      if (!object)
        continue;
      if (object->clobbered || (object->exposed && unknownStore))
        return Info::lost();
      result.join(initialContent(*object, offset, width));
      for (const StoreRecord &store : object->stores)
        if (mayOverlap(offset, width, store.offset, store.width))
          result.join(store.content);
    }
    if (result.targets.empty())
      result = Info::anyInteger();
    return result;
  }

  void stored(const Info &pointer, uint64_t width, const Info &content) {
    if (pointer.unknown) {
      // The store lands in some exposed object, which then holds `content`.
      if (!unknownStore) {
        unknownStore = true;
        changed = true;
      }
      expose(content);
    }
    for (const auto &[object, offset] : pointer.targets) {
      if (!object)
        continue;
      auto record = llvm::find_if(object->stores, [&](const StoreRecord &s) {
        return s.offset == offset && s.width == width;
      });
      if (record == object->stores.end()) {
        object->stores.push_back({offset, width, content});
        changed = true;
      } else if (record->content.join(content)) {
        changed = true;
      }
    }
  }

  /// A store of `bytes` bytes, or of an unknown amount when it is not a
  /// constant.
  void storedRange(const Info &pointer, Value bytes, const Info &content) {
    APInt constant;
    if (matchPattern(bytes, m_ConstantInt(&constant)) &&
        constant.isStrictlyPositive()) {
      stored(pointer, constant.getZExtValue(), content);
      return;
    }
    Info anywhere;
    anywhere.unknown = pointer.unknown;
    for (const auto &target : pointer.targets)
      anywhere.join(target.first, Congruence::any());
    stored(anywhere, 1, content);
  }

  /// Everything the objects `pointer` may address can hold.
  Info contentOf(const Info &pointer) {
    Info anywhere;
    anywhere.unknown = pointer.unknown;
    for (const auto &target : pointer.targets)
      anywhere.join(target.first, Congruence::any());
    return loaded(anywhere, 1);
  }

  /// An operation the analysis does not model was handed `info`: it may
  /// keep the addresses and store through them.
  void escape(const Info &info) {
    if (!info.unknown && !info.addressesObjects())
      return;
    expose(info);
    if (!unknownStore) {
      unknownStore = true;
      changed = true;
    }
    for (const auto &target : info.targets)
      if (target.first && !target.first->clobbered) {
        target.first->clobbered = true;
        changed = true;
      }
  }

  Info sum(const Info &left, const Info &right, bool subtract) {
    if (left.unknown || right.unknown) {
      // Lost plus an offset stays inside an exposed object; an address
      // combined with something lost is itself lost.
      expose(left);
      expose(right);
      return Info::lost();
    }
    Info result;
    for (const auto &[leftObject, leftOffset] : left.targets)
      for (const auto &[rightObject, rightOffset] : right.targets) {
        if (leftObject && rightObject) {
          // The distance of two addresses is an integer; their sum is not
          // an address the analysis follows.
          if (!subtract) {
            expose(left);
            expose(right);
            return Info::lost();
          }
          result.join(nullptr, Congruence::any());
        } else if (rightObject && subtract) {
          expose(right);
          return Info::lost();
        } else {
          result.join(
              leftObject ? leftObject : rightObject,
              leftOffset.add(subtract ? rightOffset.negate() : rightOffset));
        }
      }
    return result;
  }

  Info transfer(Operation *operation) {
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      if (auto value = dyn_cast<IntegerAttr>(constant.getValue()))
        if (value.getValue().getSignificantBits() <= 64)
          return Info::integer(
              Congruence::exact(value.getValue().getSExtValue()));
      return Info::anyInteger();
    }
    if (auto address = dyn_cast<host::AddressOfOp>(operation)) {
      Operation *global =
          SymbolTable::lookupSymbolIn(module, address.getGlobal());
      Info info;
      if (global)
        info.join(objectFor(global), Congruence::exact(0));
      else
        info = Info::lost();
      return info;
    }
    if (isa<host::AllocaOp>(operation)) {
      Info info;
      info.join(objectFor(operation), Congruence::exact(0));
      return info;
    }
    if (isa<host::PtrAddOp, arith::AddIOp>(operation))
      return sum(infoOf(operation->getOperand(0)),
                 infoOf(operation->getOperand(1)), /*subtract=*/false);
    if (isa<arith::SubIOp>(operation))
      return sum(infoOf(operation->getOperand(0)),
                 infoOf(operation->getOperand(1)), /*subtract=*/true);
    if (isa<host::PtrToIntOp, host::IntToPtrOp, UnrealizedConversionCastOp,
            arith::ExtUIOp, arith::ExtSIOp, arith::IndexCastOp,
            arith::IndexCastUIOp>(operation) &&
        operation->getNumOperands() == 1)
      return infoOf(operation->getOperand(0));
    if (auto select = dyn_cast<arith::SelectOp>(operation)) {
      Info info = infoOf(select.getTrueValue());
      info.join(infoOf(select.getFalseValue()));
      return info;
    }
    if (auto load = dyn_cast<host::LoadOp>(operation)) {
      unsigned width = accessBytes(load.getResult().getType());
      Info pointer = infoOf(load.getPointer());
      if (!width) {
        escape(pointer);
        return Info::lost();
      }
      return loaded(pointer, width);
    }
    if (isa<arith::MulIOp, arith::ShLIOp>(operation)) {
      const Info &left = infoOf(operation->getOperand(0));
      const Info &right = infoOf(operation->getOperand(1));
      if (!left.unknown && !right.unknown && !left.addressesObjects() &&
          !right.addressesObjects() && left.targets.size() == 1 &&
          right.targets.size() == 1) {
        Congruence a = left.targets[0].second, b = right.targets[0].second;
        if (isa<arith::ShLIOp>(operation)) {
          if (b.isExact() && b.base >= 0 && b.base < 31)
            return Info::integer(a.multiply(int64_t{1} << b.base));
        } else if (b.isExact() && std::abs(b.base) < (int64_t{1} << 31)) {
          return Info::integer(a.multiply(b.base));
        } else if (a.isExact() && std::abs(a.base) < (int64_t{1} << 31)) {
          return Info::integer(b.multiply(a.base));
        }
        return Info::anyInteger();
      }
    }
    if (isa<arith::AndIOp, arith::TruncIOp>(operation)) {
      // Masking keeps an address inside its object.
      Info info;
      for (Value operand : operation->getOperands()) {
        const Info &operandInfo = infoOf(operand);
        info.unknown |= operandInfo.unknown;
        for (const auto &target : operandInfo.targets)
          info.join(target.first, Congruence::any());
      }
      info.join(nullptr, Congruence::any());
      return info;
    }
    if (isa<arith::CmpIOp, arith::CmpFOp, runtime::AssignedBufferOp,
            runtime::StaticBufferOp, memref::LoadOp>(operation))
      return Info::anyInteger();
    // Any other computation: an address that enters it is no longer
    // followed, and an operation with effects may also store through it.
    bool lost = false;
    bool pure = isMemoryEffectFree(operation);
    for (Value operand : operation->getOperands()) {
      Info info = infoOf(operand);
      if (info.unknown || info.addressesObjects()) {
        if (pure)
          expose(info);
        else
          escape(info);
        lost = true;
      }
    }
    return lost ? Info::lost() : Info::anyInteger();
  }

  void visit(func::FuncOp function) {
    // An export is called from Udon, which holds no Host address: its
    // integer parameters are plain integers.
    if (function.isPublic())
      for (BlockArgument argument : function.getArguments())
        join(argument, isa<host::PtrType>(argument.getType()) &&
                               carriesAddress(argument.getType())
                           ? Info::lost()
                           : Info::anyInteger());
    for (Block &block : function.getBody())
      for (Operation &operation : block)
        visit(function, &operation);
  }

  void forward(Block *destination, OperandRange operands) {
    for (auto [argument, value] :
         llvm::zip(destination->getArguments(), operands))
      join(argument, infoOf(value));
  }

  void visit(func::FuncOp function, Operation *operation) {
    if (auto branch = dyn_cast<cf::BranchOp>(operation))
      return forward(branch.getDest(), branch.getDestOperands());
    if (auto branch = dyn_cast<cf::CondBranchOp>(operation)) {
      forward(branch.getTrueDest(), branch.getTrueDestOperands());
      return forward(branch.getFalseDest(), branch.getFalseDestOperands());
    }
    if (auto branch = dyn_cast<cf::SwitchOp>(operation)) {
      forward(branch.getDefaultDestination(), branch.getDefaultOperands());
      for (unsigned index = 0; index < branch.getCaseDestinations().size();
           ++index)
        forward(branch.getCaseDestinations()[index],
                branch.getCaseOperands(index));
      return;
    }
    if (auto done = dyn_cast<func::ReturnOp>(operation)) {
      Info &result = returns[function.getOperation()];
      for (Value operand : done.getOperands()) {
        // A value returned to Udon leaves the module.
        if (function.isPublic())
          escape(infoOf(operand));
        if (result.join(infoOf(operand)))
          changed = true;
      }
      return;
    }
    if (auto call = dyn_cast<func::CallOp>(operation)) {
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      if (!callee || callee.isExternal()) {
        for (Value operand : call.getOperands())
          escape(infoOf(operand));
        for (Value result : call.getResults())
          join(result, Info::lost());
        return;
      }
      for (auto [argument, operand] :
           llvm::zip(callee.getArguments(), call.getOperands()))
        join(argument, infoOf(operand));
      // Results are joined into one summary: precise for the single-result
      // functions C produces.
      for (Value result : call.getResults())
        join(result, returns[callee.getOperation()]);
      return;
    }
    if (auto store = dyn_cast<host::StoreOp>(operation)) {
      unsigned width = accessBytes(store.getValue().getType());
      if (!width) {
        escape(infoOf(store.getPointer()));
        escape(infoOf(store.getValue()));
        return;
      }
      return stored(infoOf(store.getPointer()), width,
                    infoOf(store.getValue()));
    }
    if (auto fill = dyn_cast<host::MemsetOp>(operation))
      return storedRange(infoOf(fill.getDestination()), fill.getByteSize(),
                         Info::anyInteger());
    if (isa<host::MemcpyOp, host::MemmoveOp>(operation))
      return storedRange(infoOf(operation->getOperand(0)),
                         operation->getOperand(2),
                         contentOf(infoOf(operation->getOperand(1))));
    if (isa<host::LifetimeStartOp, host::LifetimeEndOp>(operation))
      return;
    if (auto resolve =
            dyn_cast<host::CheckedAssignedBufferResolveOp>(operation)) {
      stored(infoOf(resolve.getOutputCell()), 4, Info::anyInteger());
      return join(resolve.getStatus(), Info::anyInteger());
    }
    // The generated binding stores the bytes of a completed readback into
    // the raw heap at the Host pointer of the command.
    if (auto readback = dyn_cast<runtime::BufferReadbackOp>(operation)) {
      storedRange(infoOf(readback.getHostPointer()), readback.getByteSize(),
                  Info::anyInteger());
      return join(readback.getEvent(), Info::anyInteger());
    }
    // An upload or a fill only reads Host memory.
    if (isa<runtime::BufferUploadOp, runtime::BufferFillOp>(operation))
      return join(operation->getResult(0), Info::anyInteger());
    if (auto store = dyn_cast<memref::StoreOp>(operation))
      return escape(infoOf(store.getValueToStore()));
    if (operation->getNumRegions() != 0 || operation->getNumSuccessors() != 0) {
      // Structured or unknown control flow is not expected in Host code.
      for (Value operand : operation->getOperands())
        escape(infoOf(operand));
      operation->walk([&](Operation *nested) {
        for (Value operand : nested->getOperands())
          escape(infoOf(operand));
        for (Value result : nested->getResults())
          join(result, Info::lost());
      });
      return;
    }
    if (operation->getNumResults() == 0) {
      for (Value operand : operation->getOperands())
        escape(infoOf(operand));
      return;
    }
    Info info = transfer(operation);
    for (Value result : operation->getResults())
      join(result, info);
  }
};

/// Replaces `load` by the value its initializer holds, when that value has a
/// constant form. Returns whether it did.
bool foldLoad(host::LoadOp load, Object &object, int64_t offset) {
  host::GlobalOp global = object.global;
  Type type = load.getResult().getType();
  unsigned width = accessBytes(type);
  OpBuilder builder(load);
  Location location = load.getLoc();
  Type i32 = builder.getI32Type();
  auto token = [&](Value word) -> Value {
    if (word.getType() == type)
      return word;
    if (isa<host::PtrType>(type))
      return host::IntToPtrOp::create(builder, location, type, word)
          .getResult();
    return UnrealizedConversionCastOp::create(builder, location,
                                              TypeRange{type}, ValueRange{word})
        .getResult(0);
  };
  Value replacement;
  if (auto relocations = global.getRelocations())
    for (Attribute attribute : *relocations) {
      auto relocation = host::GlobalRelocation::parse(attribute);
      if (!relocation || !mayOverlap(Congruence::exact(offset), width,
                                     Congruence::exact(relocation->offset), 4))
        continue;
      if (relocation->offset != offset || width != 4 || type.isF32())
        return false;
      if (relocation->symbol) {
        Value address = host::AddressOfOp::create(
            builder, location, host::PtrType::get(load.getContext(), 0),
            relocation->symbol.getValue());
        if (relocation->addend)
          address = host::PtrAddOp::create(
                        builder, location, address.getType(), address,
                        arith::ConstantOp::create(
                            builder, location,
                            builder.getI32IntegerAttr(relocation->addend)))
                        .getResult();
        replacement = isa<host::PtrType>(type)
                          ? address
                          : token(host::PtrToIntOp::create(builder, location,
                                                           i32, address)
                                      .getResult());
      } else {
        OperationState state(location,
                             runtime::AssignedBufferOp::getOperationName());
        for (NamedAttribute field : relocation->assignedBuffer)
          state.addAttribute(field.getName(), field.getValue());
        state.addTypes(runtime::BufferType::get(load.getContext()));
        Value buffer = builder.create(state)->getResult(0);
        // A token is the buffer converted to its word.
        replacement = isa<runtime::BufferType>(type)
                          ? buffer
                          : token(UnrealizedConversionCastOp::create(
                                      builder, location, TypeRange{i32},
                                      ValueRange{buffer})
                                      .getResult(0));
      }
    }
  if (!replacement) {
    APInt bits(width * 8, 0);
    Attribute initial = global.getInitialValueAttr();
    if (auto bytes = dyn_cast_or_null<DenseI8ArrayAttr>(initial)) {
      for (unsigned byte = 0; byte < width; ++byte)
        bits.insertBits(static_cast<uint8_t>(bytes.asArrayRef()[offset + byte]),
                        byte * 8, 8);
    } else if (auto integer = dyn_cast_or_null<IntegerAttr>(initial)) {
      bits = integer.getValue()
                 .zextOrTrunc(*object.bytes * 8)
                 .lshr(offset * 8)
                 .zextOrTrunc(width * 8);
    } else if (auto floating = dyn_cast_or_null<FloatAttr>(initial)) {
      bits = floating.getValue()
                 .bitcastToAPInt()
                 .lshr(offset * 8)
                 .zextOrTrunc(width * 8);
    } else if (initial && !isa<UnitAttr>(initial)) {
      return false;
    }
    if (auto floating = dyn_cast<FloatType>(type)) {
      replacement = arith::ConstantOp::create(
          builder, location,
          builder.getFloatAttr(type,
                               APFloat(floating.getFloatSemantics(), bits)));
    } else if (type.isInteger(1)) {
      // The Udon lowering loads a boolean as "byte != 0".
      replacement = arith::ConstantOp::create(
          builder, location, builder.getBoolAttr(!bits.isZero()));
    } else if (isa<IntegerType>(type)) {
      replacement = arith::ConstantOp::create(
          builder, location, builder.getIntegerAttr(type, bits));
    } else {
      replacement = token(arith::ConstantOp::create(
          builder, location, builder.getIntegerAttr(i32, bits)));
    }
  }
  load.getResult().replaceAllUsesWith(replacement);
  load.erase();
  return true;
}

struct FoldImmutableHostMemoryPass
    : impl::FoldImmutableHostMemoryPassBase<FoldImmutableHostMemoryPass> {
  using FoldImmutableHostMemoryPassBase::FoldImmutableHostMemoryPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Analysis analysis(module);
    analysis.run();

    SmallVector<host::LoadOp> loads;
    module.walk([&](host::LoadOp load) {
      if (!load.getVolatile_())
        loads.push_back(load);
    });
    int64_t folded = 0;
    // Why the remaining loads stay, for `explain`.
    int64_t inexact = 0, automatic = 0, stored = 0, unrepresentable = 0;
    llvm::StringMap<int64_t> storedGlobals;
    struct Foldable {
      host::LoadOp load;
      Object *object;
      int64_t offset;
    };
    SmallVector<Foldable> foldable;
    for (host::LoadOp load : loads) {
      auto address = analysis.exactAddress(load.getPointer());
      unsigned width = accessBytes(load.getResult().getType());
      if (!address || !width) {
        ++inexact;
        continue;
      }
      auto [object, offset] = *address;
      if (!object->global || !object->bytes || offset < 0 ||
          static_cast<uint64_t>(offset) + width > *object->bytes) {
        ++automatic;
        continue;
      }
      // A constant global is immutable by declaration.
      if (!object->global.getConstant() &&
          analysis.mayBeStored(*object, offset, width)) {
        ++stored;
        ++storedGlobals[object->global.getSymName()];
        continue;
      }
      foldable.push_back({load, object, offset});
    }
    // Folding replaces values the analysis describes, so decide first.
    for (auto &[load, object, offset] : foldable) {
      if (foldLoad(load, *object, offset))
        ++folded;
      else
        ++unrepresentable;
    }
    if (explain) {
      llvm::errs() << "fold-immutable-host-memory: folded=" << folded
                   << " address-not-exact=" << inexact
                   << " not-a-global=" << automatic
                   << " may-be-stored=" << stored
                   << " no-constant-form=" << unrepresentable << "\n";
      SmallVector<std::pair<StringRef, int64_t>> ranked;
      for (const auto &entry : storedGlobals)
        ranked.emplace_back(entry.getKey(), entry.getValue());
      llvm::sort(ranked, [](const auto &left, const auto &right) {
        return left.second != right.second ? left.second > right.second
                                           : left.first < right.first;
      });
      for (const auto &[symbol, count] : ranked)
        llvm::errs() << "  may-be-stored @" << symbol << " loads=" << count
                     << analysis.describe(symbol) << "\n";
    }
    int64_t constants = 0;
    for (const std::unique_ptr<Object> &object : analysis.objects())
      if (object->global && object->bytes && !object->global.getConstant() &&
          object->global.getInitialValue() && !analysis.isStored(*object)) {
        object->global.setConstant(true);
        ++constants;
      }
    foldedLoads += folded;
    constantGlobals += constants;
    // Callers alternate this pass with scalar simplification until it
    // reports nothing.
    Builder builder(module.getContext());
    module->setAttr(
        "vrc.immutable_host_memory",
        builder.getDictionaryAttr(
            {builder.getNamedAttr("constant_globals",
                                  builder.getI64IntegerAttr(constants)),
             builder.getNamedAttr("folded_loads",
                                  builder.getI64IntegerAttr(folded))}));
  }
};

} // namespace
} // namespace vrc
