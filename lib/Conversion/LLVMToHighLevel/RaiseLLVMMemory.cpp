#include "VRC/Conversion/LLVMToHighLevel/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_RAISELLVMMEMORYPASS
#include "VRC/Conversion/LLVMToHighLevel/Passes.h.inc"

namespace {
LogicalResult reject(Operation *op, StringRef reason) {
  return op->emitError() << "cannot recover memory for " << op->getName()
                         << ": " << reason;
}
std::optional<int64_t> constant(Value value) {
  auto op = value.getDefiningOp<LLVM::ConstantOp>();
  if (!op)
    return std::nullopt;
  auto attr = dyn_cast<IntegerAttr>(op.getValue());
  if (!attr || attr.getValue().getBitWidth() > 64)
    return std::nullopt;
  return attr.getValue().getSExtValue();
}
bool requiresRawMemory(Type type) {
  if (isa<LLVM::LLVMPointerType, LLVM::LLVMStructType>(type))
    return true;
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type))
    return requiresRawMemory(array.getElementType());
  return false;
}

struct Interval {
  int64_t min, max;
};
struct Term {
  Value value;
  int64_t stride;
};
struct PointerInfo {
  unsigned allocation;
  int64_t offset = 0;
  SmallVector<Term> terms;
};
struct Allocation {
  MemRefType type;
  int64_t count;
  uint64_t bytes, alignment;
  std::optional<std::string> global;
  Attribute initialValue;
  bool constant = false;
};
struct CountedLoop {
  BlockArgument iv;
  int64_t start, bound;
  Block *body, *backedge, *exit;
};

// All proofs precede conversion. Offsets are measured in homogeneous leaf
// elements; flattening avoids inventing per-dimension bounds for LLVM GEPs.
class MemoryPlan {
public:
  MemoryPlan(ModuleOp module)
      : module(module), layout(module), dominance(module) {}
  LogicalResult analyze();
  SmallVector<Allocation> allocations;
  llvm::StringMap<unsigned> globalAllocations;
  DenseMap<Value, PointerInfo> pointers;
  DenseMap<Operation *, PointerInfo> accesses;
  DenseMap<Operation *, uint64_t> accessAlignments;

private:
  llvm::Type *sourceType(Type type);
  uint64_t byteStride(Type type);
  uint64_t abiAlignment(Type type);
  LogicalResult addAllocation(LLVM::AllocaOp op);
  LogicalResult addGlobal(LLVM::GlobalOp op);
  FailureOr<PointerInfo> resolve(Value pointer, Operation *at,
                                 unsigned depth = 0);
  FailureOr<int64_t> leafCount(Type type, Type &leaf);
  std::optional<CountedLoop> countedLoop(BlockArgument iv);
  std::optional<Interval> range(Value value, Operation *at, unsigned depth = 0);
  std::optional<Interval> range(const PointerInfo &info, Operation *at);
  LogicalResult checkAccess(Operation *op, Value pointer, Type type,
                            std::optional<uint64_t> alignment, bool isVolatile,
                            LLVM::AtomicOrdering ordering);
  bool initialized(LLVM::LoadOp load, const PointerInfo &info);
  ModuleOp module;
  DataLayout layout;
  DominanceInfo dominance;
  llvm::LLVMContext llvmContext;
  std::optional<llvm::DataLayout> sourceLayout;
};

llvm::Type *MemoryPlan::sourceType(Type type) {
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type))
    return llvm::ArrayType::get(sourceType(array.getElementType()),
                                array.getNumElements());
  if (auto integer = dyn_cast<IntegerType>(type))
    return llvm::IntegerType::get(llvmContext, integer.getWidth());
  if (type.isF32())
    return llvm::Type::getFloatTy(llvmContext);
  assert(type.isF64() && "source type was validated before layout queries");
  return llvm::Type::getDoubleTy(llvmContext);
}
uint64_t MemoryPlan::byteStride(Type type) {
  if (sourceLayout)
    return sourceLayout->getTypeAllocSize(sourceType(type)).getFixedValue();
  return llvm::alignTo(layout.getTypeSize(type).getFixedValue(),
                       layout.getTypeABIAlignment(type));
}
uint64_t MemoryPlan::abiAlignment(Type type) {
  return sourceLayout ? sourceLayout->getABITypeAlign(sourceType(type)).value()
                      : layout.getTypeABIAlignment(type);
}

FailureOr<int64_t> MemoryPlan::leafCount(Type type, Type &leaf) {
  int64_t count = 1;
  while (auto array = dyn_cast<LLVM::LLVMArrayType>(type)) {
    if (!array.getNumElements() || array.getNumElements() > INT32_MAX)
      return failure();
    auto product =
        llvm::checkedMul(count, static_cast<int64_t>(array.getNumElements()));
    if (!product || *product > INT32_MAX)
      return failure();
    count = *product;
    type = array.getElementType();
  }
  bool scalar = type.isF32() || type.isF64();
  if (auto integer = dyn_cast<IntegerType>(type))
    scalar = llvm::is_contained({8u, 16u, 32u, 64u}, integer.getWidth());
  if (!scalar)
    return failure();
  leaf = type;
  return count;
}
LogicalResult MemoryPlan::addAllocation(LLVM::AllocaOp op) {
  if (requiresRawMemory(op.getElemType()))
    return success();
  auto function = op->getParentOfType<LLVM::LLVMFuncOp>();
  if (!function || op->getParentOfType<ModuleOp>() != module ||
      op->getParentOp() != function ||
      op->getBlock() != &function.getBody().front())
    return reject(op,
                  "only entry-block allocas have a proven function lifetime");
  auto stackSpace =
      dyn_cast_or_null<IntegerAttr>(layout.getAllocaMemorySpace());
  if ((sourceLayout && sourceLayout->getAllocaAddrSpace() != 0) ||
      (stackSpace && !stackSpace.getValue().isZero()))
    return reject(op, "nonzero default stack memory space is unsupported");
  if (op.getType().getAddressSpace() != 0 || op.getInalloca())
    return reject(op, "only ordinary address-space-zero allocas are supported");
  for (NamedAttribute attr : op->getAttrs())
    if (!llvm::StringSwitch<bool>(attr.getName())
             .Cases({"alignment", "elem_type"}, true)
             .Default(false))
      return reject(op, "unproven allocation attribute");
  auto size = constant(op.getArraySize());
  if (!size || *size <= 0)
    return reject(op, "allocation count must be a positive static constant");
  Type leaf;
  auto elements = leafCount(op.getElemType(), leaf);
  if (failed(elements))
    return reject(op,
                  "only homogeneous scalar/array storage has a proven layout");
  auto total = llvm::checkedMul(*size, *elements);
  uint64_t bytes = layout.getTypeSize(leaf).getFixedValue();
  if (!total || *total > INT32_MAX ||
      static_cast<uint64_t>(*total) > INT32_MAX / bytes)
    return reject(
        op, "allocation exceeds the supported signed-32-bit byte-offset range");
  uint64_t mlirABI = layout.getTypeABIAlignment(leaf);
  if (!mlirABI || bytes % mlirABI)
    return reject(op, "MLIR scalar allocation stride has unsupported padding");
  uint64_t leafABI = abiAlignment(leaf);
  uint64_t storageABI = abiAlignment(op.getElemType());
  if (!leafABI || bytes % leafABI || !storageABI ||
      (static_cast<uint64_t>(*elements) * bytes) % storageABI)
    return reject(op, "padded scalar/array allocation strides are unsupported");
  auto indexBits = layout.getTypeIndexBitwidth(op.getType());
  if (!indexBits || (*indexBits != 32 && *indexBits != 64) ||
      byteStride(op.getElemType()) != static_cast<uint64_t>(*elements) * bytes)
    return reject(
        op, "pointer index width or padded storage layout is unsupported");
  uint64_t alignment =
      op.getAlignment().value_or(abiAlignment(op.getElemType()));
  if (!alignment || alignment > INT64_MAX)
    return reject(op, "allocation alignment is outside the memref contract");
  if (sourceLayout && (*indexBits != sourceLayout->getIndexSizeInBits(0) ||
                       layout.getTypeSizeInBits(op.getType()).getFixedValue() !=
                           sourceLayout->getPointerSizeInBits(0)))
    return reject(op, "source and MLIR pointer layouts disagree");
  unsigned index = allocations.size();
  allocations.push_back({MemRefType::get({*total}, leaf),
                         *total,
                         bytes,
                         alignment,
                         std::nullopt,
                         {},
                         false});
  pointers.try_emplace(op.getRes(), PointerInfo{index, 0, {}});
  return success();
}

LogicalResult MemoryPlan::addGlobal(LLVM::GlobalOp op) {
  if (op.getAddrSpace() != 0 || (op.getLinkage() != LLVM::Linkage::Internal &&
                                 op.getLinkage() != LLVM::Linkage::Private))
    return success();
  auto array = dyn_cast<LLVM::LLVMArrayType>(op.getGlobalType());
  auto initial = op.getValue();
  auto elements = initial ? dyn_cast<ElementsAttr>(*initial) : ElementsAttr();
  if (!array || !elements)
    return success();
  Type leaf;
  auto count = leafCount(array, leaf);
  if (failed(count) || (!leaf.isInteger(32) && !leaf.isF32()) ||
      elements.getNumElements() != static_cast<uint64_t>(*count) ||
      elements.getElementType() != leaf)
    return success();
  uint64_t bytes = layout.getTypeSize(leaf).getFixedValue();
  uint64_t alignment = op.getAlignment().value_or(abiAlignment(array));
  if (!alignment || alignment > INT64_MAX ||
      byteStride(array) != static_cast<uint64_t>(*count) * bytes)
    return reject(op, "global array layout is outside the typed contract");
  unsigned index = allocations.size();
  allocations.push_back({MemRefType::get({*count}, leaf), *count, bytes,
                         alignment, op.getSymName().str(), elements,
                         op.getConstant()});
  globalAllocations.try_emplace(op.getSymName(), index);
  return success();
}

// Recognize a structural induction proof, not a guessed loop annotation. Two
// incoming unconditional edges: static start and iv+1. The true edge of the
// header's iv < constant comparison must dominate both use and backedge.
std::optional<CountedLoop> MemoryPlan::countedLoop(BlockArgument iv) {
  auto type = dyn_cast<IntegerType>(iv.getType());
  if (!type || type.getWidth() < 2 || type.getWidth() > 64)
    return std::nullopt;
  Block *header = iv.getOwner();
  if (header->isEntryBlock())
    return std::nullopt;
  auto branch = dyn_cast<LLVM::CondBrOp>(header->getTerminator());
  if (!branch || branch.getTrueDest() == branch.getFalseDest() ||
      branch.getTrueDest()->getSinglePredecessor() != header)
    return std::nullopt;
  auto cmp = branch.getCondition().getDefiningOp<LLVM::ICmpOp>();
  if (!cmp || cmp.getLhs() != iv ||
      (cmp.getPredicate() != LLVM::ICmpPredicate::slt &&
       cmp.getPredicate() != LLVM::ICmpPredicate::ult))
    return std::nullopt;
  auto bound = constant(cmp.getRhs());
  if (!bound || *bound <= 0 || *bound >= INT32_MAX ||
      !llvm::APInt(64, *bound).isSignedIntN(type.getWidth()))
    return std::nullopt;
  std::optional<int64_t> start;
  Block *backedge = nullptr;
  unsigned edges = 0;
  for (Block &block : *header->getParent()) {
    auto *term = block.getTerminator();
    for (Block *successor : term->getSuccessors()) {
      if (successor != header)
        continue;
      ++edges;
      auto incoming = dyn_cast<LLVM::BrOp>(term);
      if (!incoming)
        return std::nullopt;
      Value value = incoming.getDestOperands()[iv.getArgNumber()];
      if (auto initial = constant(value)) {
        if (start)
          return std::nullopt;
        start = *initial;
        continue;
      }
      auto increment = value.getDefiningOp<LLVM::AddOp>();
      if (!increment || backedge ||
          !((increment.getLhs() == iv && constant(increment.getRhs()) == 1) ||
            (increment.getRhs() == iv && constant(increment.getLhs()) == 1)) ||
          !dominance.dominates(branch.getTrueDest(), &block))
        return std::nullopt;
      backedge = &block;
    }
  }
  if (edges != 2 || !start || *start < 0 || *start >= *bound || !backedge)
    return std::nullopt;
  return CountedLoop{iv,       *start,
                     *bound,   branch.getTrueDest(),
                     backedge, branch.getFalseDest()};
}
std::optional<Interval> MemoryPlan::range(Value value, Operation *at,
                                          unsigned depth) {
  if (depth > 16)
    return std::nullopt;
  if (auto c = constant(value))
    return Interval{*c, *c};
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto loop = countedLoop(arg))
      if (dominance.dominates(loop->body, at->getBlock()))
        return Interval{loop->start, loop->bound - 1};
  }
  if (auto mask = value.getDefiningOp<LLVM::AndOp>()) {
    auto c = constant(mask.getRhs());
    if (!c)
      c = constant(mask.getLhs());
    if (c && *c >= 0)
      return Interval{0, *c};
  }
  if (auto sext = value.getDefiningOp<LLVM::SExtOp>())
    return range(sext.getArg(), at, depth + 1);
  if (auto zext = value.getDefiningOp<LLVM::ZExtOp>()) {
    auto source = range(zext.getArg(), at, depth + 1);
    if (source && source->min >= 0)
      return source;
    unsigned width = cast<IntegerType>(zext.getArg().getType()).getWidth();
    if (width <= 16)
      return Interval{0, (int64_t(1) << width) - 1};
  }
  return std::nullopt;
}
std::optional<Interval> MemoryPlan::range(const PointerInfo &info,
                                          Operation *at) {
  Interval total{info.offset, info.offset};
  for (const auto &term : info.terms) {
    auto bounds = range(term.value, at);
    if (!bounds)
      return std::nullopt;
    auto low = llvm::checkedMulAdd(bounds->min, term.stride, total.min);
    auto high = llvm::checkedMulAdd(bounds->max, term.stride, total.max);
    if (!low || !high)
      return std::nullopt;
    total = {*low, *high};
  }
  return total;
}
FailureOr<PointerInfo> MemoryPlan::resolve(Value pointer, Operation *at,
                                           unsigned depth) {
  if (auto cached = pointers.find(pointer); cached != pointers.end())
    return cached->second;
  auto gep = pointer.getDefiningOp<LLVM::GEPOp>();
  if (!gep || depth > 64)
    return reject(
        at, "pointer provenance must trace through GEPs to a local allocation");
  if (gep.getNoWrapFlags() != LLVM::GEPNoWrapFlags::none &&
      gep.getNoWrapFlags() != LLVM::GEPNoWrapFlags::inbounds)
    return reject(gep, "GEP no-wrap flags require additional proof");
  for (NamedAttribute attr : gep->getAttrs())
    if (!llvm::StringSwitch<bool>(attr.getName())
             .Cases({"elem_type", "rawConstantIndices", "noWrapFlags"}, true)
             .Default(false))
      return reject(gep, "unproven GEP attribute");
  auto base = resolve(gep.getBase(), gep, depth + 1);
  if (failed(base))
    return failure();
  PointerInfo info = *base;
  Type type = gep.getElemType(), leaf;
  auto size = leafCount(type, leaf);
  auto &allocation = allocations[info.allocation];
  if (failed(size) || leaf != allocation.type.getElementType() ||
      byteStride(type) != static_cast<uint64_t>(*size) * allocation.bytes)
    return reject(gep,
                  "GEP type reinterpretation or padded layout is unsupported");
  uint64_t abi = abiAlignment(type);
  if (!abi || (static_cast<uint64_t>(*size) * allocation.bytes) % abi)
    return reject(gep, "padded GEP allocation stride is unsupported");
  bool first = true;
  for (auto index : gep.getIndices()) {
    if (!first) {
      auto array = dyn_cast<LLVM::LLVMArrayType>(type);
      if (!array)
        return reject(gep, "GEP descent requires an array type");
      type = array.getElementType();
      size = leafCount(type, leaf);
    }
    first = false;
    std::optional<int64_t> c;
    Value dynamic;
    if (auto attr = dyn_cast<IntegerAttr>(index))
      c = attr.getInt();
    else {
      dynamic = cast<Value>(index);
      c = constant(dynamic);
    }
    if (c) {
      auto offset = llvm::checkedMulAdd(*c, *size, info.offset);
      if (!offset)
        return reject(gep, "GEP offset arithmetic overflows");
      info.offset = *offset;
    } else {
      auto bounds = range(dynamic, gep);
      if (!bounds || bounds->min < 0 ||
          bounds->max > INT32_MAX / allocation.bytes / *size)
        return reject(
            gep, "GEP index bounds are not proven without signed overflow");
      info.terms.push_back({dynamic, *size});
    }
    auto bounds = range(info, gep);
    if (!bounds || bounds->min < 0 || bounds->max > allocation.count)
      return reject(
          gep, "GEP offset is outside the allocation or its one-past endpoint");
  }
  pointers.try_emplace(pointer, info);
  return info;
}
LogicalResult MemoryPlan::checkAccess(Operation *op, Value pointer, Type type,
                                      std::optional<uint64_t> alignment,
                                      bool isVolatile,
                                      LLVM::AtomicOrdering ordering) {
  if (isVolatile || ordering != LLVM::AtomicOrdering::not_atomic)
    return reject(
        op, "volatile and atomic accesses require explicit memory semantics");
  for (NamedAttribute attr : op->getAttrs())
    if (!llvm::StringSwitch<bool>(attr.getName())
             .Cases({"alignment", "ordering"}, true)
             .Default(false))
      return reject(op, "unproven memory access attribute");
  auto info = resolve(pointer, op);
  if (failed(info))
    return failure();
  auto &allocation = allocations[info->allocation];
  if (allocation.constant && isa<LLVM::StoreOp>(op))
    return reject(op, "store to a constant global is unsupported");
  if (type != allocation.type.getElementType())
    return reject(op, "mixed access types require raw byte-memory recovery");
  auto bounds = range(*info, op);
  if (!bounds || bounds->min < 0 || bounds->max >= allocation.count)
    return reject(op, "load/store bounds are not proven inside the allocation");
  // Factor the product to avoid signed abs and multiplication overflow.
  uint64_t unit = std::gcd(allocation.alignment, allocation.bytes);
  uint64_t guaranteed = unit * std::gcd(allocation.alignment / unit,
                                        llvm::AbsoluteValue(info->offset));
  for (const auto &term : info->terms)
    guaranteed = std::gcd(guaranteed, static_cast<uint64_t>(term.stride) *
                                          allocation.bytes);
  uint64_t requiredAlignment = alignment.value_or(abiAlignment(type));
  if (requiredAlignment > guaranteed)
    return reject(op,
                  "access alignment is not proven by allocation and offset");
  accesses.try_emplace(op, *info);
  accessAlignments.try_emplace(op, requiredAlignment);
  return success();
}

Value stripExtensions(Value value) {
  while (true) {
    if (auto op = value.getDefiningOp<LLVM::SExtOp>())
      value = op.getArg();
    else if (auto op = value.getDefiningOp<LLVM::ZExtOp>())
      value = op.getArg();
    else
      return value;
  }
}
bool sameOffset(const PointerInfo &a, const PointerInfo &b) {
  if (a.offset != b.offset || a.terms.size() != b.terms.size())
    return false;
  return llvm::all_of(llvm::zip(a.terms, b.terms), [](auto pair) {
    return std::get<0>(pair).value == std::get<1>(pair).value &&
           std::get<0>(pair).stride == std::get<1>(pair).stride;
  });
}
bool MemoryPlan::initialized(LLVM::LoadOp load, const PointerInfo &info) {
  if (allocations[info.allocation].initialValue)
    return true;
  llvm::SmallSet<int64_t, 16> initializedConstants;
  for (const auto &entry : accesses) {
    auto store = dyn_cast<LLVM::StoreOp>(entry.first);
    const auto &stored = entry.second;
    if (!store || stored.allocation != info.allocation)
      continue;
    if (dominance.dominates(store.getOperation(), load.getOperation())) {
      if (sameOffset(stored, info))
        return true;
      if (stored.terms.empty())
        initializedConstants.insert(stored.offset);
    }
    // A complete [0,N) initialization loop must write each element before its
    // backedge. Only its uniquely reached false exit proves completion.
    if (stored.offset != 0 || stored.terms.size() != 1 ||
        stored.terms[0].stride != 1)
      continue;
    auto iv = dyn_cast<BlockArgument>(stripExtensions(stored.terms[0].value));
    auto loop = iv ? countedLoop(iv) : std::nullopt;
    if (loop && loop->start == 0 &&
        loop->bound == allocations[info.allocation].count &&
        loop->exit->getSinglePredecessor() == iv.getOwner() &&
        dominance.dominates(store.getOperation(),
                            loop->backedge->getTerminator()) &&
        dominance.dominates(loop->exit, load->getBlock()))
      return true;
  }
  return static_cast<int64_t>(initializedConstants.size()) ==
         allocations[info.allocation].count;
}
LogicalResult MemoryPlan::analyze() {
  // DLTI import does not represent every LLVM layout rule.
  // Preserve and query the source layout rather than guessing dense strides.
  auto raw = module->getAttrOfType<StringAttr>("vrc.llvm_data_layout");
  if (!raw)
    raw = module->getAttrOfType<StringAttr>(
        LLVM::LLVMDialect::getDataLayoutAttrName());
  if (raw) {
    auto parsed = llvm::DataLayout::parse(raw.getValue());
    if (!parsed)
      return module.emitError() << "invalid source LLVM data layout: "
                                << llvm::toString(parsed.takeError());
    sourceLayout = std::move(*parsed);
    if (auto declared = module->getAttrOfType<StringAttr>(
            LLVM::LLVMDialect::getDataLayoutAttrName())) {
      auto active = llvm::DataLayout::parse(declared.getValue());
      if (!active)
        return module.emitError() << "invalid explicit LLVM data layout: "
                                  << llvm::toString(active.takeError());
      if (*active != *sourceLayout)
        return module.emitError()
               << "conflicting captured and explicit LLVM data layouts";
    }
  }
  if (module
          .walk([&](LLVM::AllocaOp op) {
            return failed(addAllocation(op)) ? WalkResult::interrupt()
                                             : WalkResult::advance();
          })
          .wasInterrupted())
    return failure();
  if (module
          .walk([&](LLVM::GlobalOp op) {
            return failed(addGlobal(op)) ? WalkResult::interrupt()
                                         : WalkResult::advance();
          })
          .wasInterrupted())
    return failure();
  module.walk([&](LLVM::AddressOfOp op) {
    auto found = globalAllocations.find(op.getGlobalName());
    if (found != globalAllocations.end())
      pointers.try_emplace(op.getRes(), PointerInfo{found->second, 0, {}});
  });
  auto hasTrackedRoot = [&](Value pointer) {
    while (auto gep = pointer.getDefiningOp<LLVM::GEPOp>())
      pointer = gep.getBase();
    return pointers.contains(pointer);
  };
  if (module
          .walk([&](Operation *op) {
            if (auto gep = dyn_cast<LLVM::GEPOp>(op)) {
              if (!hasTrackedRoot(gep.getBase()))
                return WalkResult::advance();
              if (failed(resolve(gep.getRes(), gep)))
                return WalkResult::interrupt();
            } else if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
              if (!hasTrackedRoot(load.getAddr()))
                return WalkResult::advance();
              if (failed(checkAccess(op, load.getAddr(), load.getType(),
                                     load.getAlignment(), load.getVolatile_(),
                                     load.getOrdering())))
                return WalkResult::interrupt();
            } else if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
              if (!hasTrackedRoot(store.getAddr()))
                return WalkResult::advance();
              if (failed(checkAccess(op, store.getAddr(),
                                     store.getValue().getType(),
                                     store.getAlignment(), store.getVolatile_(),
                                     store.getOrdering())))
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted())
    return failure();
  // Every use of every derived pointer must have been accounted for. This also
  // rejects calls, phi/select merges, casts, storing a pointer and ptrtoint.
  for (auto &entry : pointers)
    for (OpOperand &use : entry.first.getUses()) {
      auto *user = use.getOwner();
      if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
        if (use.getOperandNumber() == 0)
          continue;
      } else if (isa<LLVM::LoadOp>(user)) {
        continue;
      } else if (isa<LLVM::StoreOp>(user) && use.getOperandNumber() == 1) {
        continue;
      }
      return reject(user, "pointer escapes or has an unsupported alias use");
    }
  for (auto &entry : accesses)
    if (auto load = dyn_cast<LLVM::LoadOp>(entry.first))
      if (!initialized(load, entry.second))
        return reject(
            load,
            "definite initialization is not proven for every loaded element");
  return success();
}

// The type converter is value-aware: two opaque LLVM ptrs may refer to
// different allocations and therefore different static memref types.
struct AllocationPattern : OpConversionPattern<LLVM::AllocaOp> {
  AllocationPattern(TypeConverter &converter, MLIRContext *context,
                    const MemoryPlan &plan)
      : OpConversionPattern(converter, context), plan(plan) {}
  LogicalResult
  matchAndRewrite(LLVM::AllocaOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto &allocation =
        plan.allocations[plan.pointers.lookup(op.getRes()).allocation];
    auto result =
        memref::AllocaOp::create(rewriter, op.getLoc(), allocation.type);
    result.setAlignment(allocation.alignment);
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
  const MemoryPlan &plan;
};
struct GlobalPattern : OpConversionPattern<LLVM::GlobalOp> {
  GlobalPattern(TypeConverter &converter, MLIRContext *context,
                const MemoryPlan &plan)
      : OpConversionPattern(converter, context), plan(plan) {}
  LogicalResult
  matchAndRewrite(LLVM::GlobalOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto found = plan.globalAllocations.find(op.getSymName());
    if (found == plan.globalAllocations.end())
      return failure();
    const auto &allocation = plan.allocations[found->second];
    memref::GlobalOp::create(rewriter, op.getLoc(), op.getSymName(),
                             rewriter.getStringAttr("private"), allocation.type,
                             allocation.initialValue, allocation.constant,
                             rewriter.getI64IntegerAttr(allocation.alignment));
    rewriter.eraseOp(op);
    return success();
  }
  const MemoryPlan &plan;
};
struct AddressPattern : OpConversionPattern<LLVM::AddressOfOp> {
  AddressPattern(TypeConverter &converter, MLIRContext *context,
                 const MemoryPlan &plan)
      : OpConversionPattern(converter, context), plan(plan) {}
  LogicalResult
  matchAndRewrite(LLVM::AddressOfOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto info = plan.pointers.find(op.getRes());
    if (info == plan.pointers.end())
      return failure();
    const auto &allocation = plan.allocations[info->second.allocation];
    if (!allocation.global)
      return failure();
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, allocation.type,
                                                     *allocation.global);
    return success();
  }
  const MemoryPlan &plan;
};
struct GEPattern : OpConversionPattern<LLVM::GEPOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::GEPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The offset is applied by each access; GEP has no observable pointer
    // users.
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};
Value indexFor(const PointerInfo &info, Location loc,
               ConversionPatternRewriter &rewriter) {
  Value index = arith::ConstantIndexOp::create(rewriter, loc, info.offset);
  for (const auto &term : info.terms) {
    Value value = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), term.value);
    if (term.stride != 1) {
      Value stride = arith::ConstantIndexOp::create(rewriter, loc, term.stride);
      value = arith::MulIOp::create(rewriter, loc, value, stride);
    }
    index = arith::AddIOp::create(rewriter, loc, index, value);
  }
  return index;
}
struct LoadPattern : OpConversionPattern<LLVM::LoadOp> {
  LoadPattern(TypeConverter &converter, MLIRContext *context,
              const MemoryPlan &plan)
      : OpConversionPattern(converter, context), plan(plan) {}
  LogicalResult
  matchAndRewrite(LLVM::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto info = plan.accesses.lookup(op);
    Value index = indexFor(info, op.getLoc(), rewriter);
    auto load = memref::LoadOp::create(
        rewriter, op.getLoc(), adaptor.getOperands()[0], ValueRange{index});
    load.setAlignment(plan.accessAlignments.lookup(op));
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
  const MemoryPlan &plan;
};
struct StorePattern : OpConversionPattern<LLVM::StoreOp> {
  StorePattern(TypeConverter &converter, MLIRContext *context,
               const MemoryPlan &plan)
      : OpConversionPattern(converter, context), plan(plan) {}
  LogicalResult
  matchAndRewrite(LLVM::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto info = plan.accesses.lookup(op);
    Value index = indexFor(info, op.getLoc(), rewriter);
    auto store =
        memref::StoreOp::create(rewriter, op.getLoc(), adaptor.getOperands()[0],
                                adaptor.getOperands()[1], ValueRange{index});
    store.setAlignment(plan.accessAlignments.lookup(op));
    rewriter.eraseOp(op);
    return success();
  }
  const MemoryPlan &plan;
};
struct RaiseLLVMMemoryPass
    : impl::RaiseLLVMMemoryPassBase<RaiseLLVMMemoryPass> {
  void runOnOperation() override {
    auto module = getOperation();
    MemoryPlan plan(module);
    if (failed(plan.analyze())) {
      signalPassFailure();
      return;
    }
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([&](Value value) -> std::optional<Type> {
      auto found = plan.pointers.find(value);
      if (found == plan.pointers.end())
        return std::nullopt;
      return plan.allocations[found->second.allocation].type;
    });
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect,
                           memref::MemRefDialect>();
    target.addLegalOp<ModuleOp>();
    target.addDynamicallyLegalOp<LLVM::AllocaOp>([&](LLVM::AllocaOp op) {
      return !plan.pointers.contains(op.getRes());
    });
    target.addDynamicallyLegalOp<LLVM::GEPOp>(
        [&](LLVM::GEPOp op) { return !plan.pointers.contains(op.getRes()); });
    target.addDynamicallyLegalOp<LLVM::LoadOp>(
        [&](LLVM::LoadOp op) { return !plan.accesses.contains(op); });
    target.addDynamicallyLegalOp<LLVM::StoreOp>(
        [&](LLVM::StoreOp op) { return !plan.accesses.contains(op); });
    target.addDynamicallyLegalOp<LLVM::GlobalOp>([&](LLVM::GlobalOp op) {
      return !plan.globalAllocations.contains(op.getSymName());
    });
    target.addDynamicallyLegalOp<LLVM::AddressOfOp>([&](LLVM::AddressOfOp op) {
      return !plan.pointers.contains(op.getRes());
    });
    RewritePatternSet patterns(&getContext());
    patterns.add<AllocationPattern, GlobalPattern, AddressPattern, LoadPattern,
                 StorePattern>(converter, &getContext(), plan);
    patterns.add<GEPattern>(converter, &getContext());
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
