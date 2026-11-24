// Proves gather kernels that the compiler synthesized directly as `gpu.func`.
// The OpenCL C path proves buffer bounds on LLVM-dialect IR before it builds
// the `gpu.func` (see BoundaryFrontend/GatherDevice.cpp); a synthesized kernel
// has no such history, so the same guarantees are established here on the
// structured form that the gather lowering consumes.
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/bit.h"
#include <algorithm>
#include <optional>
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_PROVESYNTHESIZEDGATHERKERNELSPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
// Limits of the gather-f32 profile, shared with the vrcgpu.kernel verifier
// and the native and Unity runtimes.
constexpr int64_t maxWords = vrcgpu::gatherMaxWords;
constexpr int64_t maxExtent = vrcgpu::gatherSurfaceExtent;
constexpr int64_t maxInputElements = maxExtent * maxExtent;
constexpr unsigned maxInputs = 13;
constexpr unsigned maxArguments = 16;
constexpr int64_t maxTripCount = vrcgpu::gatherMaxTripCount;
constexpr int64_t maxValue = INT32_MAX;

/// Closed interval of a 32-bit value. Only values that read the same signed
/// and unsigned are representable (0 <= min <= max <= INT32_MAX), so a range
/// stays valid across index casts and never describes a wrapped result.
struct Range {
  int64_t min, max;
};
using MaybeRange = std::optional<Range>;

/// Returns [min, max], or nothing when it is empty or not representable. A
/// 32-bit operation whose exact result may leave the representable interval
/// may wrap, so its range is unknown.
MaybeRange makeRange(int64_t min, int64_t max) {
  if (min < 0 || max > maxValue || min > max)
    return std::nullopt;
  return Range{min, max};
}

MaybeRange join(MaybeRange a, MaybeRange b) {
  if (!a || !b)
    return std::nullopt;
  return Range{std::min(a->min, b->min), std::max(a->max, b->max)};
}

arith::CmpIPredicate swapOperands(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  default:
    return predicate;
  }
}

Value stripIndexCasts(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<arith::IndexCastOp, arith::IndexCastUIOp>(op))
      break;
    value = op->getOperand(0);
  }
  return value;
}

std::optional<int64_t> constantValue(Value value) {
  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)) ||
      constant.getActiveBits() > 31)
    return std::nullopt;
  return static_cast<int64_t>(constant.getZExtValue());
}

/// Integer range analysis over the structured body of one kernel.
class RangeAnalysis {
public:
  RangeAnalysis(gpu::GPUFuncOp kernel, int64_t width, int64_t height,
                llvm::DenseMap<Value, Range> arguments)
      : kernel(kernel), width(width), height(height),
        arguments(std::move(arguments)) {}

  /// Returns the range of `value` at `at`, an operation that `value`
  /// dominates. Conditions of the scf.if regions enclosing `at` refine it.
  MaybeRange range(Value value, Operation *at) {
    auto key = std::make_pair(value, at->getBlock());
    auto found = cache.find(key);
    if (found != cache.end())
      return found->second;
    if (depth >= maxDepth)
      return std::nullopt;
    ++depth;
    MaybeRange result = definition(value, at);
    for (Operation *inner = at; inner != kernel.getOperation();) {
      Operation *parent = inner->getParentOp();
      if (auto branch = dyn_cast<scf::IfOp>(parent))
        result = refine(value, branch.getCondition(),
                        inner->getParentRegion() == &branch.getThenRegion(),
                        branch, result);
      inner = parent;
    }
    --depth;
    cache[key] = result;
    return result;
  }

  /// A while loop that counts one carried value up by one from a bounded
  /// start while it is below a bounded loop-invariant limit.
  struct Induction {
    Value counter;
    Range header, body;
  };
  std::optional<Induction> induction(scf::WhileOp loop) {
    auto found = inductions.find(loop);
    if (found != inductions.end())
      return found->second;
    std::optional<Induction> result = matchInduction(loop);
    inductions[loop] = result;
    return result;
  }

private:
  std::optional<Induction> matchInduction(scf::WhileOp loop) {
    Block &before = loop.getBefore().front(), &after = loop.getAfter().front();
    auto condition = cast<scf::ConditionOp>(before.getTerminator());
    auto compare = condition.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!compare || (compare.getPredicate() != arith::CmpIPredicate::ult &&
                     compare.getPredicate() != arith::CmpIPredicate::slt))
      return std::nullopt;
    auto counter = dyn_cast<BlockArgument>(compare.getLhs());
    if (!counter || counter.getOwner() != &before ||
        loop->isAncestor(compare.getRhs().getParentRegion()->getParentOp()))
      return std::nullopt;
    MaybeRange limit = range(compare.getRhs(), loop);
    MaybeRange start = range(loop.getInits()[counter.getArgNumber()], loop);
    if (!limit || !start || limit->max > maxTripCount)
      return std::nullopt;
    // The value the next iteration starts from must be the counter of this
    // iteration plus one.
    auto yield = cast<scf::YieldOp>(after.getTerminator());
    auto step =
        yield.getOperand(counter.getArgNumber()).getDefiningOp<arith::AddIOp>();
    if (!step)
      return std::nullopt;
    auto carried = [&](Value value) {
      auto argument = dyn_cast<BlockArgument>(value);
      return argument && argument.getOwner() == &after &&
             condition.getArgs()[argument.getArgNumber()] == counter;
    };
    if (!(carried(step.getLhs()) && constantValue(step.getRhs()) == 1) &&
        !(carried(step.getRhs()) && constantValue(step.getLhs()) == 1))
      return std::nullopt;
    // The counter only advances while it is below the limit, so it never
    // exceeds the limit and never wraps.
    Range header{start->min, std::max(start->max, limit->max)};
    return Induction{counter, header,
                     makeRange(start->min, limit->max - 1).value_or(header)};
  }

  /// Narrows `known` by the fact that `condition` evaluated to `holds`.
  /// `at` is where the comparison bound is evaluated.
  MaybeRange refine(Value value, Value condition, bool holds, Operation *at,
                    MaybeRange known) {
    if (auto conjunction = condition.getDefiningOp<arith::AndIOp>();
        conjunction && holds)
      return refine(value, conjunction.getRhs(), true, at,
                    refine(value, conjunction.getLhs(), true, at, known));
    if (auto disjunction = condition.getDefiningOp<arith::OrIOp>();
        disjunction && !holds)
      return refine(value, disjunction.getRhs(), false, at,
                    refine(value, disjunction.getLhs(), false, at, known));
    auto compare = condition.getDefiningOp<arith::CmpIOp>();
    if (!compare)
      return known;
    arith::CmpIPredicate predicate = compare.getPredicate();
    // An index cast keeps the 32-bit value, so a comparison of either form
    // constrains the other.
    Value bound, compared = stripIndexCasts(value);
    if (stripIndexCasts(compare.getLhs()) == compared)
      bound = compare.getRhs();
    else if (stripIndexCasts(compare.getRhs()) == compared) {
      bound = compare.getLhs();
      predicate = swapOperands(predicate);
    } else
      return known;
    if (!holds)
      predicate = arith::invertPredicate(predicate);
    MaybeRange limit = range(bound, at);
    if (!limit)
      return known;
    int64_t lower = known ? known->min : 0;
    int64_t upper = known ? known->max : maxValue;
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      lower = std::max(lower, limit->min);
      upper = std::min(upper, limit->max);
      break;
    // An unsigned upper bound proves the value on its own: every unsigned
    // value below a representable bound is representable.
    case arith::CmpIPredicate::ult:
      upper = std::min(upper, limit->max - 1);
      break;
    case arith::CmpIPredicate::ule:
      upper = std::min(upper, limit->max);
      break;
    // A signed upper bound admits negative values, and an unsigned lower
    // bound admits values above INT32_MAX: both only narrow a known range.
    case arith::CmpIPredicate::slt:
      if (!known)
        return known;
      upper = std::min(upper, limit->max - 1);
      break;
    case arith::CmpIPredicate::sle:
      if (!known)
        return known;
      upper = std::min(upper, limit->max);
      break;
    case arith::CmpIPredicate::ugt:
      if (!known)
        return known;
      lower = std::max(lower, limit->min + 1);
      break;
    case arith::CmpIPredicate::uge:
      if (!known)
        return known;
      lower = std::max(lower, limit->min);
      break;
    // A value signed-above a non-negative bound is itself non-negative.
    case arith::CmpIPredicate::sgt:
      lower = std::max(lower, limit->min + 1);
      break;
    case arith::CmpIPredicate::sge:
      lower = std::max(lower, limit->min);
      break;
    case arith::CmpIPredicate::ne:
      return known;
    }
    // An empty intersection means the guarded region is unreachable; keep
    // what was known rather than invent a range.
    MaybeRange narrowed = makeRange(lower, upper);
    return narrowed ? narrowed : known;
  }

  /// Returns the range implied by the definition of `value` alone.
  MaybeRange definition(Value value, Operation *at) {
    if (auto known = arguments.find(value); known != arguments.end())
      return known->second;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto loop = dyn_cast<scf::WhileOp>(argument.getOwner()->getParentOp());
      auto counted = loop ? induction(loop) : std::nullopt;
      if (!counted)
        return std::nullopt;
      if (value == counted->counter)
        return counted->header;
      auto condition =
          cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
      if (argument.getOwner() == &loop.getAfter().front() &&
          condition.getArgs()[argument.getArgNumber()] == counted->counter)
        return counted->body;
      return std::nullopt;
    }
    if (auto constant = constantValue(value))
      return Range{*constant, *constant};
    Operation *op = value.getDefiningOp();
    if (auto id = dyn_cast<gpu::GlobalIdOp>(op)) {
      if (id.getDimension() == gpu::Dimension::x)
        return Range{0, width - 1};
      if (id.getDimension() == gpu::Dimension::y)
        return Range{0, height - 1};
      return std::nullopt;
    }
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op))
      return range(op->getOperand(0), at);
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      // Each yielded value is evaluated inside its own region, where the
      // branch condition already constrains it.
      unsigned result = cast<OpResult>(value).getResultNumber();
      Operation *thenYield = branch.getThenRegion().front().getTerminator();
      Operation *elseYield = branch.getElseRegion().front().getTerminator();
      return join(range(thenYield->getOperand(result), thenYield),
                  range(elseYield->getOperand(result), elseYield));
    }
    if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      auto counted = induction(loop);
      auto condition =
          cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
      if (counted &&
          condition.getArgs()[cast<OpResult>(value).getResultNumber()] ==
              counted->counter)
        return counted->header;
      return std::nullopt;
    }
    if (auto select = dyn_cast<arith::SelectOp>(op))
      return join(refine(select.getTrueValue(), select.getCondition(), true, at,
                         range(select.getTrueValue(), at)),
                  refine(select.getFalseValue(), select.getCondition(), false,
                         at, range(select.getFalseValue(), at)));
    if (op->getNumOperands() != 2 || !value.getType().isIntOrIndex() ||
        isa<arith::CmpIOp, arith::CmpFOp>(op))
      return std::nullopt;
    MaybeRange a = range(op->getOperand(0), at);
    MaybeRange b = range(op->getOperand(1), at);
    // These results are bounded by one operand whatever the other holds.
    if (isa<arith::AndIOp>(op)) {
      if (!a && !b)
        return std::nullopt;
      return makeRange(0,
                       std::min(a ? a->max : maxValue, b ? b->max : maxValue));
    }
    if (isa<arith::MinUIOp>(op)) {
      if (a && b)
        return makeRange(std::min(a->min, b->min), std::min(a->max, b->max));
      if (!a && !b)
        return std::nullopt;
      return makeRange(0, a ? a->max : b->max);
    }
    if (isa<arith::RemUIOp>(op)) {
      if (!b || b->min < 1)
        return std::nullopt;
      if (a && a->max < b->min)
        return a;
      return makeRange(0, std::min(a ? a->max : maxValue, b->max - 1));
    }
    if (!a || !b)
      return std::nullopt;
    if (isa<arith::AddIOp>(op))
      return makeRange(a->min + b->min, a->max + b->max);
    if (isa<arith::SubIOp>(op))
      return makeRange(a->min - b->max, a->max - b->min);
    if (isa<arith::MulIOp>(op))
      return makeRange(a->min * b->min, a->max * b->max);
    if (isa<arith::DivUIOp>(op))
      return b->min < 1 ? std::nullopt
                        : makeRange(a->min / b->max, a->max / b->min);
    if (isa<arith::ShLIOp>(op))
      return b->max > 30 ? std::nullopt
                         : makeRange(a->min << b->min, a->max << b->max);
    if (isa<arith::ShRUIOp, arith::ShRSIOp>(op))
      return b->max > 31 ? std::nullopt
                         : makeRange(a->min >> b->max, a->max >> b->min);
    if (isa<arith::OrIOp, arith::XOrIOp>(op)) {
      // Neither can set a bit above the highest bit of both operands.
      auto bits = static_cast<uint64_t>(std::max(a->max, b->max));
      int64_t mask = static_cast<int64_t>(llvm::bit_ceil(bits + 1) - 1);
      return makeRange(
          isa<arith::OrIOp>(op) ? std::max(a->min, b->min) : int64_t{0}, mask);
    }
    return std::nullopt;
  }

  static constexpr unsigned maxDepth = 256;
  gpu::GPUFuncOp kernel;
  int64_t width, height;
  llvm::DenseMap<Value, Range> arguments;
  llvm::DenseMap<std::pair<Value, Block *>, MaybeRange> cache;
  llvm::DenseMap<Operation *, std::optional<Induction>> inductions;
  unsigned depth = 0;
};

bool isBufferElement(Type type) { return type.isF32() || type.isInteger(32); }

bool isFragmentScalar(Type type) {
  return type.isF32() || type.isInteger(32) || type.isInteger(1) ||
         type.isIndex();
}

InFlightDiagnostic reject(Operation *op, const Twine &message) {
  return op->emitError("synthesized gather kernel: ") << message;
}

/// Checks the contract of one synthesized kernel and proves its buffer
/// accesses. Emits a diagnostic on the offending operation on failure.
LogicalResult proveKernel(gpu::GPUFuncOp kernel) {
  Operation *function = kernel.getOperation();
  auto grid = kernel->getAttrOfType<DenseI32ArrayAttr>("known_grid_size");
  auto block = kernel->getAttrOfType<DenseI32ArrayAttr>("known_block_size");
  auto profile = kernel->getAttrOfType<StringAttr>("vrc.profile");
  auto wordsAttr = kernel->getAttrOfType<IntegerAttr>("vrc.words");
  if (!kernel.isKernel() || !profile || profile.getValue() != "gather-f32")
    return reject(function,
                  "requires a gpu.kernel with vrc.profile = \"gather-f32\"");
  if (!grid || grid.size() != 3 || grid[0] < 1 || grid[0] > maxExtent ||
      grid[1] < 1 || grid[1] > maxExtent || grid[2] != 1)
    return reject(function, "known_grid_size must be [width, height, 1] with "
                            "both extents in [1, ")
           << maxExtent << "]";
  if (!block || block.asArrayRef() != ArrayRef<int32_t>{1, 1, 1})
    return reject(function, "known_block_size must be [1, 1, 1]");
  int64_t width = grid[0], height = grid[1];
  if (!wordsAttr || wordsAttr.getInt() != width * height ||
      wordsAttr.getInt() > maxWords)
    return reject(function, "vrc.words must equal width * height and be at "
                            "most ")
           << maxWords;
  int64_t words = wordsAttr.getInt();
  if (Attribute cost = kernel->getAttr("vrc.fragment_cost")) {
    auto entries = dyn_cast<DictionaryAttr>(cost);
    auto loads = entries ? entries.getAs<IntegerAttr>("loads") : IntegerAttr();
    if (!loads || loads.getInt() < 0)
      return reject(function, "vrc.fragment_cost must be a dictionary with "
                              "non-negative integer loads");
  }

  // Arguments: read-only input buffers, the output buffer, then scalars.
  if (kernel.getFunctionType().getNumResults() ||
      kernel.getNumWorkgroupAttributions() ||
      kernel.getNumPrivateAttributions() || !kernel.getBody().hasOneBlock())
    return reject(function, "requires one block, no results and no memory "
                            "attributions");
  unsigned bufferCount = 0;
  bool sawScalar = false;
  for (BlockArgument argument : kernel.getArguments()) {
    Type type = argument.getType();
    if (auto buffer = dyn_cast<MemRefType>(type)) {
      if (sawScalar || buffer.getRank() != 1 || !buffer.hasStaticShape() ||
          buffer.getNumElements() < 1 ||
          buffer.getNumElements() > maxInputElements ||
          !buffer.getLayout().isIdentity() || buffer.getMemorySpace() ||
          !isBufferElement(buffer.getElementType()))
        return reject(function, "argument ")
               << argument.getArgNumber()
               << " must be a static one-dimensional f32/i32 buffer that "
                  "precedes every scalar";
      ++bufferCount;
    } else if (type.isF32() || type.isInteger(32))
      sawScalar = true;
    else
      return reject(function, "argument ")
             << argument.getArgNumber() << " must be a buffer or an f32/i32 "
             << "scalar";
  }
  if (bufferCount < 2 || bufferCount - 1 > maxInputs ||
      kernel.getNumArguments() > maxArguments)
    return reject(function, "requires 1 to ")
           << maxInputs << " input buffers and one output buffer in at most "
           << maxArguments << " arguments";
  unsigned outputIndex = bufferCount - 1;
  Value output = kernel.getArgument(outputIndex);
  auto outputType = cast<MemRefType>(output.getType());
  if (outputType.getNumElements() != words)
    return reject(function, "output argument ")
           << outputIndex << " must hold exactly vrc.words = " << words
           << " elements";

  llvm::DenseMap<Value, Range> argumentRanges;
  if (Attribute bounds = kernel->getAttr("vrc.argument_bounds")) {
    auto entries = dyn_cast<ArrayAttr>(bounds);
    llvm::SmallSet<int64_t, 16> indices;
    for (Attribute attribute :
         entries ? entries.getValue() : ArrayRef<Attribute>{}) {
      auto entry = dyn_cast<DictionaryAttr>(attribute);
      auto index = entry ? entry.getAs<IntegerAttr>("index") : IntegerAttr();
      auto minimum = entry ? entry.getAs<IntegerAttr>("min") : IntegerAttr();
      auto maximum = entry ? entry.getAs<IntegerAttr>("max") : IntegerAttr();
      if (!index || !minimum || !maximum || index.getInt() < 0 ||
          index.getInt() >= int64_t{kernel.getNumArguments()} ||
          !kernel.getArgument(index.getInt()).getType().isInteger(32) ||
          minimum.getInt() > maximum.getInt() ||
          !indices.insert(index.getInt()).second) {
        entries = ArrayAttr();
        break;
      }
      // A bound that admits negative values proves nothing about the
      // unsigned value the fragment computes with.
      if (MaybeRange range = makeRange(minimum.getInt(), maximum.getInt()))
        argumentRanges[kernel.getArgument(index.getInt())] = *range;
    }
    if (!entries)
      return reject(function, "vrc.argument_bounds must uniquely describe i32 "
                              "scalar arguments as {index, min, max}");
  }

  RangeAnalysis analysis(kernel, width, height, std::move(argumentRanges));
  auto globalId = [](Value value, gpu::Dimension dimension) {
    auto id = stripIndexCasts(value).getDefiningOp<gpu::GlobalIdOp>();
    return id && id.getDimension() == dimension;
  };
  auto rowTerm = [&](Value value) {
    auto multiply = stripIndexCasts(value).getDefiningOp<arith::MulIOp>();
    if (!multiply)
      return false;
    auto matches = [&](Value id, Value extent) {
      return globalId(id, gpu::Dimension::y) &&
             constantValue(stripIndexCasts(extent)) == width;
    };
    return matches(multiply.getLhs(), multiply.getRhs()) ||
           matches(multiply.getRhs(), multiply.getLhs());
  };
  // The output word of fragment (x, y) is x + y * width.
  auto linearFragmentIndex = [&](Value value) {
    value = stripIndexCasts(value);
    if (height == 1 && globalId(value, gpu::Dimension::x))
      return true;
    auto add = value.getDefiningOp<arith::AddIOp>();
    return add && ((globalId(add.getLhs(), gpu::Dimension::x) &&
                    rowTerm(add.getRhs())) ||
                   (globalId(add.getRhs(), gpu::Dimension::x) &&
                    rowTerm(add.getLhs())));
  };

  Block &body = kernel.getBody().front();
  // A partial-output kernel stores its word only where a condition holds
  // (the HLSL fragment discards the others, which keep their contents): the
  // store is alone in the then-block of a top-level scf.if without results
  // or else-block. vrc.output_granule promises that the condition is the
  // same for every aligned run of that many words, so a texel of several
  // words is written whole or not at all.
  auto granule = kernel->getAttrOfType<IntegerAttr>("vrc.output_granule");
  if (granule && (granule.getInt() < 1 || words % granule.getInt() != 0))
    return reject(function, "vrc.output_granule must be a positive divisor of "
                            "vrc.words");
  auto partialStore = [&](memref::StoreOp candidate) {
    auto branch = dyn_cast<scf::IfOp>(candidate->getParentOp());
    return granule && branch && branch->getBlock() == &body &&
           branch.getNumResults() == 0 && branch.getElseRegion().empty() &&
           candidate->getBlock() == branch.thenBlock() &&
           candidate->getNextNode() == branch.thenBlock()->getTerminator() &&
           candidate.getOperation() == &branch.thenBlock()->front();
  };
  memref::StoreOp store;
  LogicalResult result = success();
  kernel.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto fail = [&](LogicalResult diagnostic) {
      result = diagnostic;
      return WalkResult::interrupt();
    };
    if (auto flags = op->getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
        flags && flags.getValue() != arith::FastMathFlags::none)
      return fail(reject(op, "fast-math flags are not allowed"));
    if (!llvm::all_of(op->getResultTypes(), isFragmentScalar))
      return fail(reject(op, "results must be f32, i32, i1 or index"));
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      auto argument = dyn_cast<BlockArgument>(load.getMemref());
      if (!argument || argument.getOwner() != &body ||
          argument.getArgNumber() >= outputIndex ||
          load.getIndices().size() != 1)
        return fail(reject(op, "loads must read an input buffer argument"));
      int64_t capacity = load.getMemRefType().getNumElements();
      MaybeRange index = analysis.range(load.getIndices()[0], op);
      if (!index)
        return fail(reject(op, "load index for argument ")
                    << argument.getArgNumber()
                    << " has no proven range; capacity is " << capacity);
      if (index->max >= capacity)
        return fail(reject(op, "load index for argument ")
                    << argument.getArgNumber() << " is not proven in bounds: "
                    << "inferred [" << index->min << ", " << index->max
                    << "] but capacity is " << capacity);
      return WalkResult::advance();
    }
    if (auto candidate = dyn_cast<memref::StoreOp>(op)) {
      if (store)
        return fail(reject(op, "requires exactly one output store"));
      if ((op->getBlock() != &body && !partialStore(candidate)) ||
          candidate.getMemref() != output || candidate.getIndices().size() != 1)
        return fail(reject(op, "the store must write the output buffer at "
                               "the top level of the kernel, or alone in "
                               "a top-level scf.if of a kernel with "
                               "vrc.output_granule"));
      if (!linearFragmentIndex(candidate.getIndices()[0]))
        return fail(reject(op, "the store index must be global_id x + "
                               "global_id y * ")
                    << width);
      store = candidate;
      return WalkResult::advance();
    }
    if (auto id = dyn_cast<gpu::GlobalIdOp>(op)) {
      if (id.getDimension() == gpu::Dimension::z)
        return fail(reject(op, "only x/y global ids are fragment-legal"));
      return WalkResult::advance();
    }
    if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      if (!analysis.induction(loop))
        return fail(reject(op, "loop must count a carried value up by one "
                               "below a proven bound of at most ")
                    << maxTripCount);
      return WalkResult::advance();
    }
    if (isa<gpu::ReturnOp>(op)) {
      if (op->getBlock() != &body || op->getNumOperands())
        return fail(reject(op, "return must end the kernel without values"));
      return WalkResult::advance();
    }
    if (isa<scf::IfOp, scf::YieldOp, scf::ConditionOp>(op) ||
        vrcgpu::isGatherLegalArithmetic(op))
      return WalkResult::advance();
    return fail(reject(op, "operation is not allowed: ") << op->getName());
  });
  if (failed(result))
    return failure();
  if (!store)
    return reject(function, "requires exactly one output store");
  return success();
}

struct ProveSynthesizedGatherKernelsPass
    : impl::ProveSynthesizedGatherKernelsPassBase<
          ProveSynthesizedGatherKernelsPass> {
  void runOnOperation() override {
    SmallVector<gpu::GPUFuncOp> kernels;
    getOperation().walk([&](gpu::GPUFuncOp kernel) {
      if (kernel->hasAttr("vrc.synthesized_kernel") &&
          !kernel->hasAttr("vrc.gather_proven"))
        kernels.push_back(kernel);
    });
    for (gpu::GPUFuncOp kernel : kernels) {
      if (failed(proveKernel(kernel))) {
        signalPassFailure();
        return;
      }
      kernel->setAttr("vrc.gather_proven", UnitAttr::get(&getContext()));
    }
  }
};
} // namespace
} // namespace vrc
