#include "ONNXHostFusionKernel.h"

#include "ONNXHostPlan.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/raw_ostream.h"

#include <deque>
#include <limits>
#include <map>
#include <numeric>

using namespace mlir;

namespace vrc {
namespace {
// The gather surface: a Device buffer is a texture of this width, and the
// fragment at (x, y) writes word x + y * surfaceWidth. A fused kernel launches
// over the rows that hold the elements of its root, so it fits every buffer
// that holds that tensor, also one smaller than the 16 rows of the library
// kernels or taller than them.
constexpr int64_t surfaceWidth = vrcgpu::gatherSurfaceExtent;
constexpr int64_t maximumSurfaceWords = deviceSynthesizedSurfaceWords;

constexpr StringLiteral carrierName = "udononnx_fused_kernels";

/// Words of the launch surface of a kernel that writes `words` words.
int64_t statisticsSurfaceWords(int64_t words) {
  return (words + surfaceWidth - 1) / surfaceWidth * surfaceWidth;
}

int64_t elementCount(ArrayRef<int64_t> shape) {
  int64_t elements = 1;
  for (int64_t extent : shape)
    elements *= extent;
  return elements;
}

ArrayRef<int64_t> shapeOf(const GraphRecord &graph, int64_t tensor) {
  return graph.tensors[tensor].type.getShape();
}

int64_t concatAxis(const GraphRecord &graph, const NodeRecord &node) {
  int64_t axis = integerAttribute(node, "axis", 0);
  return axis < 0 ? axis + graph.tensors[node.output].type.getRank() : axis;
}

/// The axis the ONNX (opset 13) Softmax normalizes, as isDeviceSoftmaxNode
/// and the library dispatch read it.
int64_t softmaxAxis(const GraphRecord &graph, const NodeRecord &node) {
  int64_t axis = integerAttribute(node, "axis", -1);
  return axis < 0 ? axis + graph.tensors[node.output].type.getRank() : axis;
}

/// Writes everything that determines the kernel of the group rooted at
/// `nodeIndex`: operations, shapes, static parameters, immediates and which
/// leaf argument every Load reads. Two groups with the same key have the
/// same kernel.
void structureKey(const DeviceFusion &fusion, ArrayRef<unsigned> leaves,
                  unsigned nodeIndex, raw_ostream &os) {
  const GraphRecord &graph = fusion.graph;
  const NodeRecord &node = graph.nodes[nodeIndex];
  DeviceFusionKind kind = *deviceFusionKind(graph, node);
  // A dynamic axis has its extent and its run-time value: write both.
  auto dynamic = [&](int64_t tensor) {
    if (const auto &axis = graph.tensors[tensor].dynamic)
      os << '~' << axis->axis << ':' << axis->symbol << '+' << axis->offset;
  };
  os << node.operation << '[';
  llvm::interleave(shapeOf(graph, node.output), os, "x");
  os << ']';
  dynamic(node.output);
  if (kind == DeviceFusionKind::Slice) {
    DeviceSliceShape slice =
        *deviceSliceShape(graph, node, DeviceKernelLimits::Synthesized);
    os << '{';
    llvm::interleaveComma(slice.starts, os);
    os << ';';
    llvm::interleaveComma(slice.steps, os);
    os << '}';
  } else if (kind == DeviceFusionKind::Transpose) {
    os << '{';
    llvm::interleaveComma(*deviceTransposePermutation(
                              graph, node, DeviceKernelLimits::Synthesized),
                          os);
    os << '}';
  } else if (kind == DeviceFusionKind::Concat) {
    os << '{' << concatAxis(graph, node) << '}';
  } else if (kind == DeviceFusionKind::Softmax) {
    os << '{' << softmaxAxis(graph, node) << '}';
  }
  os << '(';
  for (unsigned index = 0; index != deviceFusionValueOperands(kind); ++index) {
    if (index)
      os << ',';
    DeviceFusionOperand operand = fusion.operand(nodeIndex, index);
    if (operand.kind == DeviceFusionOperand::Interior) {
      structureKey(fusion, leaves, operand.node, os);
      continue;
    }
    if (operand.kind == DeviceFusionOperand::Immediate) {
      os << '#' << llvm::bit_cast<uint32_t>(operand.immediate);
      continue;
    }
    os << '%' << (llvm::find(leaves, operand.tensor) - leaves.begin()) << ':'
       << graph.tensors[operand.tensor].type.getNumElements() << '[';
    llvm::interleave(shapeOf(graph, node.inputs[index]), os, "x");
    os << ']';
    dynamic(node.inputs[index]);
    dynamic(operand.tensor);
  }
  os << ')';
}

/// An i32 index value with an upper bound that the interval arithmetic of
/// vrc-prove-synthesized-gather-kernels derives from the operations alone:
/// constants, global ids under the fragment guard, and addi, muli, divui,
/// remui and minui with constants. The bound never depends on a branch
/// condition inside the expression, and canonicalization of these operations
/// keeps their intervals, so the proof survives the passes that follow.
struct Bounded {
  Value value;
  int64_t max;
  /// Set for a constant, which is materialized where it is used.
  std::optional<int64_t> constant;
};

/// The element of a tensor of static shape that the current fragment needs,
/// as a linear index, as coordinates, or both. Either form is derived from
/// the other on first use.
struct IndexMap {
  SmallVector<int64_t> shape;
  std::optional<Bounded> linear;
  std::optional<SmallVector<Bounded>> coordinates;
  unsigned id;
};

struct Emitted {
  Value value;
  /// Loads a fragment executes for this value that it had not executed yet.
  unsigned loads;
};

/// The partial statistics of a two-pass Softmax (deviceSoftmaxSchedule):
/// for each row r and chunk c of `chunkLength` elements, word r * chunks + c
/// holds the chunk maximum and word rows * chunks + r * chunks + c the sum
/// of exp(sample - chunk maximum) over the chunk.
struct DeviceRowStatistics {
  int64_t rows;
  int64_t chunks;
  int64_t chunkLength;
  /// The statistics argument of the elementwise kernel.
  Value buffer;
  int64_t partials() const { return rows * chunks; }
};

/// The axis of a bounded dimension of a tensor in a fused kernel: its
/// run-time extent and, for a leaf or the root, the pitch of the axis in its
/// buffer (at least 1) and the words that buffer holds.
struct DynamicTensor {
  int64_t axis;
  Bounded extent;
  Bounded pitch;
  int64_t capacityWords = 0;
};
using DynamicTensors = llvm::DenseMap<unsigned, DynamicTensor>;

/// Emits the expression of one fusion group at the builder's insertion point.
class ExpressionEmitter {
public:
  /// `viewOffset` is the bounded run-time offset of a view whose offset is
  /// dynamic (DeviceFusionLeafView::dynamicOffset).
  ExpressionEmitter(OpBuilder &builder, Location location,
                    const DeviceFusion &fusion, ArrayRef<unsigned> leaves,
                    ArrayRef<BlockArgument> buffers,
                    const DeviceFusionLeafView *view,
                    std::optional<Bounded> viewOffset = std::nullopt,
                    const DynamicTensors *dynamic = nullptr)
      : builder(builder), location(location), fusion(fusion),
        graph(fusion.graph), leaves(leaves), buffers(buffers), view(view),
        viewOffset(viewOffset), dynamic(dynamic) {}

  /// The coordinates of the root element fragment `linear` writes when the
  /// root has an axis of a bounded dimension, laid out with the pitch of its
  /// buffer, and the condition that the element is inside the tensor at its
  /// run-time extent. Emitted where the fragment guard is built.
  std::pair<SmallVector<Bounded>, Value> rootElement(unsigned rootNode,
                                                     Value linear) {
    unsigned tensor = graph.nodes[rootNode].output;
    const DynamicTensor &axisInfo = dynamic->at(tensor);
    ArrayRef<int64_t> shape = shapeOf(graph, tensor);
    int64_t axis = axisInfo.axis;
    int64_t inner = elementCount(shape.drop_front(axis + 1));
    int64_t outer = elementCount(shape.take_front(axis));
    Bounded fragment{linear, axisInfo.capacityWords - 1, std::nullopt};
    Bounded rows = divide(fragment, inner);
    Bounded innerLinear = remainder(fragment, inner);
    Bounded coordinate = remainder(rows, axisInfo.pitch);
    Bounded outerLinear = divide(rows, axisInfo.pitch);
    SmallVector<Bounded> coordinates;
    auto decompose = [&](Bounded value, ArrayRef<int64_t> part) {
      int64_t stride = elementCount(part);
      for (int64_t extent : part) {
        stride /= extent;
        coordinates.push_back(extent == 1
                                  ? constant(0)
                                  : remainder(divide(value, stride), extent));
      }
    };
    decompose(outerLinear, shape.take_front(axis));
    coordinates.push_back(bound(coordinate, shape[axis] - 1));
    decompose(innerLinear, shape.drop_front(axis + 1));
    Value inside =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ult,
                              valueOf(coordinate), valueOf(axisInfo.extent));
    Value inOuter =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ult,
                              valueOf(outerLinear), i32Constant(outer));
    return {coordinates,
            arith::AndIOp::create(builder, location, inside, inOuter)};
  }

  /// The element at `rootCoordinates` of the in-place Concat root
  /// `rootNode`, an element after its first operand's block: the second
  /// operand's element at the coordinate less the first operand's run-time
  /// extent along the axis.
  Emitted emitConcatTail(unsigned rootNode, ArrayRef<Bounded> rootCoordinates) {
    const NodeRecord &node = graph.nodes[rootNode];
    int64_t axis = concatAxis(graph, node);
    int64_t right = shapeOf(graph, node.inputs[1])[axis];
    const DynamicTensor *leftAxis =
        dynamicOf(static_cast<unsigned>(node.inputs[0]));
    return emitConcatSide(
        rootNode, 1, axis, rootCoordinates,
        bound(subtract(rootCoordinates[axis], leftAxis->extent), right - 1));
  }

  Emitted emitRoot(unsigned rootNode, Value linear,
                   const DeviceRowStatistics *statistics,
                   ArrayRef<Bounded> rootCoordinates = {}) {
    ArrayRef<int64_t> shape = shapeOf(graph, graph.nodes[rootNode].output);
    IndexMap &map = newMap(shape);
    if (!rootCoordinates.empty())
      map.coordinates = SmallVector<Bounded>(rootCoordinates);
    else
      map.linear = Bounded{linear, elementCount(shape) - 1, std::nullopt};
    if (*deviceFusionKind(graph, graph.nodes[rootNode]) ==
        DeviceFusionKind::Softmax)
      return statistics ? emitSoftmaxFromStatistics(rootNode, map, *statistics)
                        : emitSoftmax(rootNode, map);
    return emitNode(rootNode, map);
  }

  /// Fragment `linear` of the statistics pass of the Softmax root `rootNode`
  /// (see DeviceRowStatistics): the maximum of one chunk of one row, or the
  /// ascending sum of exp(sample - that maximum) over the chunk.
  Emitted emitSoftmaxStatistics(unsigned rootNode, Value linear,
                                const DeviceRowStatistics &statistics) {
    int64_t axis = softmaxAxis(graph, graph.nodes[rootNode]);
    ArrayRef<int64_t> shape = shapeOf(graph, graph.nodes[rootNode].output);
    int64_t extent = shape[axis];
    int64_t partials = statistics.partials();
    Bounded fragment{linear, 2 * partials - 1, std::nullopt};
    Value isSum =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::uge,
                              linear, i32Constant(partials));
    Bounded partial = remainder(fragment, partials);
    Bounded chunk = remainder(partial, statistics.chunks);
    // The coordinates of the row are those of its linear index in the shape
    // without the reduced axis.
    SmallVector<int64_t> rowShape(shape);
    rowShape[axis] = 1;
    IndexMap &rowMap = newMap(rowShape);
    rowMap.linear = divide(partial, statistics.chunks);
    IndexMap &map = newMap(shape);
    map.coordinates = SmallVector<Bounded>(coordinatesOf(rowMap));
    (*map.coordinates)[axis] = constant(0);
    Value start = valueOf(multiply(chunk, statistics.chunkLength));
    Value end = arith::MinUIOp::create(
        builder, location,
        arith::AddIOp::create(builder, location, start,
                              i32Constant(statistics.chunkLength)),
        axisLimit(graph.nodes[rootNode].output, axis, extent));
    std::pair<Value, unsigned> maximum =
        foldAxis(rootNode, map, axis, f32Constant(-3.402823466e+38f), start,
                 end, [&](Value current, Value sample) -> Value {
                   return maximumOf(current, sample);
                 });
    // A fragment of the maximum plane runs no element of the sum.
    Value sumEnd =
        arith::SelectOp::create(builder, location, isSum, end, start);
    std::pair<Value, unsigned> sum = foldAxis(
        rootNode, map, axis, f32Constant(0.0f), start, sumEnd,
        [&](Value total, Value sample) -> Value {
          return arith::AddFOp::create(builder, location, total,
                                       exponentialAbove(sample, maximum.first));
        });
    Value value = arith::SelectOp::create(builder, location, isSum, sum.first,
                                          maximum.first);
    return {value,
            static_cast<unsigned>(2 * statistics.chunkLength * maximum.second)};
  }

private:
  /// Folds `combine` over the operand of the reduction root `rootNode` at
  /// every element of `axis`, in ascending order, starting from `initial`.
  /// The operand is evaluated inside a counted scf.while whose counter
  /// replaces the axis coordinate of `map`; returns the result and the Loads
  /// of one evaluation.
  template <typename Combine>
  std::pair<Value, unsigned> foldAxis(unsigned rootNode, IndexMap &map,
                                      int64_t axis, Value initial,
                                      Combine &&combine) {
    // The limit is defined outside the loop, so the proof recognizes a
    // counted loop. A dynamic axis ends at its run-time extent.
    Value limit =
        axisLimit(graph.nodes[rootNode].output, axis, map.shape[axis]);
    return foldAxis(rootNode, map, axis, initial, i32Constant(0), limit,
                    std::forward<Combine>(combine));
  }

  /// The fold over the elements [start, limit) of `axis`, with `limit` at
  /// most its extent.
  template <typename Combine>
  std::pair<Value, unsigned> foldAxis(unsigned rootNode, IndexMap &map,
                                      int64_t axis, Value initial, Value start,
                                      Value limit, Combine &&combine) {
    ArrayRef<Bounded> coordinates = coordinatesOf(map);
    int64_t extent = map.shape[axis];
    unsigned loads = 0;
    Value result =
        countedLoop(start, limit, initial, [&](Value counter, Value carried) {
          IndexMap &element = newMap(map.shape);
          element.coordinates = SmallVector<Bounded>(coordinates);
          (*element.coordinates)[axis] =
              Bounded{counter, extent - 1, std::nullopt};
          Emitted sample = emitOperand(rootNode, 0, element);
          loads = sample.loads;
          return combine(carried, sample.value);
        });
    return {result, loads};
  }

  /// An scf.while that counts from `start` up to `limit`, both defined
  /// outside it, carrying one f32 from `initial` through `body`.
  template <typename Body>
  Value countedLoop(Value start, Value limit, Value initial, Body &&body) {
    Type i32 = builder.getI32Type();
    Type f32 = builder.getF32Type();
    Value one = i32Constant(1);
    auto loop = scf::WhileOp::create(builder, location, TypeRange{i32, f32},
                                     ValueRange{start, initial});
    OpBuilder::InsertionGuard guard(builder);
    Block *before = builder.createBlock(&loop.getBefore(), {}, {i32, f32},
                                        {location, location});
    Value inside =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ult,
                              before->getArgument(0), limit);
    scf::ConditionOp::create(builder, location, inside, before->getArguments());
    Block *after = builder.createBlock(&loop.getAfter(), {}, {i32, f32},
                                       {location, location});
    Value carried = body(after->getArgument(0), after->getArgument(1));
    Value next =
        arith::AddIOp::create(builder, location, after->getArgument(0), one);
    scf::YieldOp::create(builder, location, ValueRange{next, carried});
    return loop.getResult(1);
  }

  /// The library kernel's running maximum: `sample` replaces `current` only
  /// when it compares greater, so a NaN sample is skipped.
  Value maximumOf(Value current, Value sample) {
    Value greater = arith::CmpFOp::create(
        builder, location, arith::CmpFPredicate::OGT, sample, current);
    return arith::SelectOp::create(builder, location, greater, sample, current);
  }

  /// exp(sample - maximum).
  Value exponentialAbove(Value sample, Value maximum) {
    Value shifted = arith::SubFOp::create(builder, location, sample, maximum);
    return math::ExpOp::create(builder, location, shifted);
  }

  /// The elementwise pass of a two-pass Softmax. The row's maximum M is the
  /// largest chunk maximum m_c (the same value the one-pass kernel finds),
  /// its sum the ascending sum over chunks of s_c * exp(m_c - M), and the
  /// result exp(own sample - M) / sum.
  Emitted emitSoftmaxFromStatistics(unsigned rootNode, IndexMap &map,
                                    const DeviceRowStatistics &statistics) {
    int64_t axis = softmaxAxis(graph, graph.nodes[rootNode]);
    ArrayRef<Bounded> coordinates = coordinatesOf(map);
    Bounded row = constant(0);
    for (auto [index, extent] : llvm::enumerate(map.shape))
      if (static_cast<int64_t>(index) != axis)
        row = add(multiply(row, extent), coordinates[index]);
    int64_t partials = statistics.partials();
    Bounded base = multiply(row, statistics.chunks);
    // Partial c of the row is word base + c of the maximum plane and of the
    // sum plane after it.
    auto loadPartial = [&](Value chunk, int64_t plane) -> Value {
      Bounded index =
          add(add(base, Bounded{chunk, statistics.chunks - 1, std::nullopt}),
              constant(plane * partials));
      Value address = arith::IndexCastOp::create(
          builder, location, builder.getIndexType(), valueOf(index));
      return memref::LoadOp::create(builder, location, statistics.buffer,
                                    ValueRange{address});
    };
    Value zero = i32Constant(0);
    Value chunks = i32Constant(statistics.chunks);
    Value maximum =
        countedLoop(zero, chunks, f32Constant(-3.402823466e+38f),
                    [&](Value chunk, Value current) {
                      return maximumOf(current, loadPartial(chunk, 0));
                    });
    Value sum = countedLoop(
        zero, chunks, f32Constant(0.0f), [&](Value chunk, Value total) {
          Value scaled = arith::MulFOp::create(
              builder, location, loadPartial(chunk, 1),
              exponentialAbove(loadPartial(chunk, 0), maximum));
          return arith::AddFOp::create(builder, location, total, scaled);
        });
    Emitted own = emitOperand(rootNode, 0, map);
    Value value = arith::DivFOp::create(
        builder, location, exponentialAbove(own.value, maximum), sum);
    return {value, static_cast<unsigned>(3 * statistics.chunks) + own.loads};
  }

  /// udondnn_arena_softmax_forward with algorithm 0 (accurate) over the
  /// root's axis, reading the operand expression where the library kernel
  /// reads its input buffer: the maximum, starting from -FLT_MAX, of the
  /// samples greater than it; the ascending sum of exp(sample - maximum);
  /// and exp(own sample - maximum) / sum.
  Emitted emitSoftmax(unsigned rootNode, IndexMap &map) {
    int64_t axis = softmaxAxis(graph, graph.nodes[rootNode]);
    // Every fragment derives its coordinates before the loops.
    coordinatesOf(map);
    std::pair<Value, unsigned> maximum =
        foldAxis(rootNode, map, axis, f32Constant(-3.402823466e+38f),
                 [&](Value current, Value sample) -> Value {
                   return maximumOf(current, sample);
                 });
    std::pair<Value, unsigned> denominator = foldAxis(
        rootNode, map, axis, f32Constant(0.0f),
        [&](Value sum, Value sample) -> Value {
          return arith::AddFOp::create(builder, location, sum,
                                       exponentialAbove(sample, maximum.first));
        });
    Emitted own = emitOperand(rootNode, 0, map);
    Value value = arith::DivFOp::create(
        builder, location, exponentialAbove(own.value, maximum.first),
        denominator.first);
    int64_t extent = map.shape[axis];
    return {value, static_cast<unsigned>((maximum.second + denominator.second) *
                                         extent) +
                       own.loads};
  }

  Value i32Constant(int64_t value) {
    return arith::ConstantIntOp::create(builder, location, value, 32);
  }
  Value f32Constant(float value) {
    return arith::ConstantFloatOp::create(builder, location,
                                          builder.getF32Type(), APFloat(value));
  }
  Value valueOf(const Bounded &bounded) {
    return bounded.constant ? i32Constant(*bounded.constant) : bounded.value;
  }
  static Bounded constant(int64_t value) { return {Value(), value, value}; }

  Bounded multiply(const Bounded &a, int64_t factor) {
    if (a.constant)
      return constant(*a.constant * factor);
    if (factor == 1)
      return a;
    return {
        arith::MulIOp::create(builder, location, a.value, i32Constant(factor)),
        a.max * factor, std::nullopt};
  }
  Bounded add(const Bounded &a, const Bounded &b) {
    if (a.constant && b.constant)
      return constant(*a.constant + *b.constant);
    if (a.constant == 0)
      return b;
    if (b.constant == 0)
      return a;
    return {arith::AddIOp::create(builder, location, valueOf(a), valueOf(b)),
            a.max + b.max, std::nullopt};
  }
  /// The run-time extent of `axis` of `tensor` when it is the axis of a
  /// bounded dimension, else the constant `extent`.
  Value axisLimit(unsigned tensor, int64_t axis, int64_t extent) {
    if (dynamic)
      if (auto found = dynamic->find(tensor);
          found != dynamic->end() && found->second.axis == axis)
        return valueOf(found->second.extent);
    return i32Constant(extent);
  }
  Bounded multiply(const Bounded &a, const Bounded &b) {
    if (b.constant)
      return multiply(a, *b.constant);
    if (a.constant)
      return multiply(b, *a.constant);
    return {arith::MulIOp::create(builder, location, a.value, b.value),
            a.max * b.max, std::nullopt};
  }
  /// Division by a run-time value of at least 1.
  Bounded divide(const Bounded &a, const Bounded &divisor) {
    if (divisor.constant)
      return divide(a, *divisor.constant);
    if (a.constant && *a.constant == 0)
      return a;
    return {
        arith::DivUIOp::create(builder, location, valueOf(a), divisor.value),
        a.max, std::nullopt};
  }
  Bounded remainder(const Bounded &a, const Bounded &divisor) {
    if (divisor.constant)
      return remainder(a, *divisor.constant);
    if (a.constant && *a.constant == 0)
      return a;
    return {
        arith::RemUIOp::create(builder, location, valueOf(a), divisor.value),
        std::min(a.max, divisor.max - 1), std::nullopt};
  }
  Bounded subtract(const Bounded &a, const Bounded &b) {
    if (a.constant && b.constant)
      return constant(*a.constant - *b.constant);
    if (b.constant == 0)
      return a;
    // The subtraction may wrap; every caller bounds the result.
    return {arith::SubIOp::create(builder, location, valueOf(a), valueOf(b)),
            std::numeric_limits<int32_t>::max(), std::nullopt};
  }

  Bounded divide(const Bounded &a, int64_t divisor) {
    if (a.constant)
      return constant(*a.constant / divisor);
    if (divisor == 1)
      return a;
    return {arith::DivUIOp::create(builder, location, a.value,
                                   i32Constant(divisor)),
            a.max / divisor, std::nullopt};
  }
  Bounded remainder(const Bounded &a, int64_t divisor) {
    if (a.constant)
      return constant(*a.constant % divisor);
    if (a.max < divisor)
      return a;
    return {arith::RemUIOp::create(builder, location, a.value,
                                   i32Constant(divisor)),
            divisor - 1, std::nullopt};
  }
  /// Bounds `a` by `max` for the proof. Every caller passes a value that is
  /// at most `max` already, so the minimum never changes it.
  Bounded bound(const Bounded &a, int64_t max) {
    if (a.constant)
      return constant(std::min(*a.constant, max));
    if (a.max <= max)
      return a;
    if (max == 0)
      return constant(0);
    return {
        arith::MinUIOp::create(builder, location, a.value, i32Constant(max)),
        max, std::nullopt};
  }

  IndexMap &newMap(ArrayRef<int64_t> shape) {
    maps.push_back({SmallVector<int64_t>(shape), std::nullopt, std::nullopt,
                    static_cast<unsigned>(maps.size())});
    return maps.back();
  }

  const Bounded &linearOf(IndexMap &map) {
    if (!map.linear) {
      Bounded linear = constant(0);
      for (auto [extent, coordinate] :
           llvm::zip_equal(map.shape, *map.coordinates))
        linear = add(multiply(linear, extent), coordinate);
      map.linear = linear;
    }
    return *map.linear;
  }

  ArrayRef<Bounded> coordinatesOf(IndexMap &map) {
    if (!map.coordinates) {
      SmallVector<Bounded> coordinates;
      int64_t stride = elementCount(map.shape);
      for (int64_t extent : map.shape) {
        stride /= extent;
        coordinates.push_back(
            extent == 1 ? constant(0)
                        : remainder(divide(*map.linear, stride), extent));
      }
      map.coordinates = std::move(coordinates);
    }
    return *map.coordinates;
  }

  /// Returns the map `derive` builds from `parent`, once per `tag`, so that
  /// operands that read the same element of a tensor share its Load. A map
  /// is only ever derived and completed where its parent is visible: the
  /// maps of the two sides of a Concat are fresh inside their regions.
  template <typename Derive>
  IndexMap &derived(IndexMap &parent, ArrayRef<int64_t> shape, const Twine &tag,
                    Derive &&derive) {
    std::string key;
    llvm::raw_string_ostream os(key);
    os << parent.id << ':' << tag << ':';
    llvm::interleave(shape, os, "x");
    auto [entry, inserted] = derivedMaps.try_emplace(key, nullptr);
    if (inserted) {
      IndexMap &child = newMap(shape);
      derive(child);
      entry->second = &child;
    }
    return *entry->second;
  }

  /// The element of a tensor of `shape` that broadcasts to `parent`.
  IndexMap &broadcast(IndexMap &parent, ArrayRef<int64_t> shape) {
    if (shape == ArrayRef<int64_t>(parent.shape))
      return parent;
    return derived(parent, shape, "broadcast", [&](IndexMap &child) {
      size_t offset = parent.shape.size() - shape.size();
      if (elementCount(shape) == elementCount(parent.shape)) {
        // Only unit dimensions differ: the same linear index.
        child.linear = parent.linear;
        if (parent.coordinates)
          child.coordinates = SmallVector<Bounded>(
              ArrayRef<Bounded>(*parent.coordinates).drop_front(offset));
        return;
      }
      ArrayRef<Bounded> coordinates = coordinatesOf(parent);
      child.coordinates.emplace();
      for (auto [axis, extent] : llvm::enumerate(shape))
        child.coordinates->push_back(extent == 1 ? constant(0)
                                                 : coordinates[axis + offset]);
    });
  }

  /// The element with the same linear index in a tensor of `shape`.
  IndexMap &reshape(IndexMap &parent, ArrayRef<int64_t> shape) {
    if (shape == ArrayRef<int64_t>(parent.shape))
      return parent;
    return derived(parent, shape, "reshape",
                   [&](IndexMap &child) { child.linear = linearOf(parent); });
  }

  Emitted emitLoad(unsigned tensor, IndexMap &map) {
    auto key = std::make_pair(2 * tensor + 1, map.id);
    if (Value known = values.lookup(key))
      return {known, 0};
    unsigned argument = llvm::find(leaves, tensor) - leaves.begin();
    int64_t capacity = graph.tensors[tensor].type.getNumElements();
    Bounded linear;
    if (view && view->tensor == tensor) {
      // The coordinates of the element in the leaf tensor, moved along the
      // view axis, address the larger buffer row-major.
      ArrayRef<Bounded> coordinates =
          coordinatesOf(reshape(map, shapeOf(graph, tensor)));
      linear = constant(0);
      for (auto [axis, extent] : llvm::enumerate(view->shape)) {
        Bounded coordinate = coordinates[axis];
        if (static_cast<int64_t>(axis) == view->axis)
          coordinate = add(coordinate, viewOffsetOf());
        linear = add(multiply(linear, extent), coordinate);
      }
      capacity = elementCount(view->shape);
    } else if (const DynamicTensor *axisInfo = dynamicOf(tensor)) {
      // The leaf's buffer holds its dynamic axis with a run-time pitch:
      // ((outer * pitch) + coordinate) * inner + inner offset.
      ArrayRef<int64_t> shape = shapeOf(graph, tensor);
      ArrayRef<Bounded> coordinates = coordinatesOf(reshape(map, shape));
      Bounded outer = constant(0), inner = constant(0);
      for (int64_t axis = 0; axis < axisInfo->axis; ++axis)
        outer = add(multiply(outer, shape[axis]), coordinates[axis]);
      for (int64_t axis = axisInfo->axis + 1;
           axis < static_cast<int64_t>(shape.size()); ++axis)
        inner = add(multiply(inner, shape[axis]), coordinates[axis]);
      Bounded row =
          add(multiply(outer, axisInfo->pitch), coordinates[axisInfo->axis]);
      linear =
          add(multiply(row, elementCount(shape.drop_front(axisInfo->axis + 1))),
              inner);
      capacity = axisInfo->capacityWords;
    } else {
      linear = linearOf(map);
    }
    linear = bound(linear, capacity - 1);
    Value index =
        linear.constant
            ? Value(arith::ConstantIndexOp::create(builder, location,
                                                   *linear.constant))
            : Value(arith::IndexCastOp::create(
                  builder, location, builder.getIndexType(), linear.value));
    Value loaded = memref::LoadOp::create(builder, location, buffers[argument],
                                          ValueRange{index});
    values[key] = loaded;
    return {loaded, 1};
  }

  Emitted emitOperand(unsigned nodeIndex, unsigned index, IndexMap &map) {
    DeviceFusionOperand operand = fusion.operand(nodeIndex, index);
    switch (operand.kind) {
    case DeviceFusionOperand::Interior:
      return emitNode(operand.node, map);
    case DeviceFusionOperand::Immediate:
      return {f32Constant(operand.immediate), 0};
    case DeviceFusionOperand::Leaf:
      return emitLoad(operand.tensor, map);
    }
    llvm_unreachable("unknown fusion operand");
  }

  /// One side of a Concat, in the region the builder points into.
  Emitted emitConcatSide(unsigned nodeIndex, unsigned side, int64_t axis,
                         ArrayRef<Bounded> coordinates, Bounded coordinate) {
    const NodeRecord &node = graph.nodes[nodeIndex];
    IndexMap &map = newMap(shapeOf(graph, node.inputs[side]));
    map.coordinates = SmallVector<Bounded>(coordinates);
    (*map.coordinates)[axis] = coordinate;
    return emitOperand(nodeIndex, side, map);
  }

  Emitted emitConcat(unsigned nodeIndex, IndexMap &map) {
    const NodeRecord &node = graph.nodes[nodeIndex];
    int64_t axis = concatAxis(graph, node);
    int64_t left = shapeOf(graph, node.inputs[0])[axis];
    int64_t right = shapeOf(graph, node.inputs[1])[axis];
    SmallVector<Bounded> coordinates(coordinatesOf(map));
    Bounded coordinate = coordinates[axis];
    // A left operand with a run-time extent on the axis splits there.
    if (const DynamicTensor *leftAxis =
            dynamicOf(static_cast<unsigned>(node.inputs[0]));
        leftAxis && leftAxis->axis == axis)
      return emitDynamicConcat(nodeIndex, axis, coordinates, coordinate,
                               leftAxis->extent, left, right);
    if (coordinate.constant)
      return *coordinate.constant < left
                 ? emitConcatSide(nodeIndex, 0, axis, coordinates, coordinate)
                 : emitConcatSide(nodeIndex, 1, axis, coordinates,
                                  constant(*coordinate.constant - left));
    Value isLeft =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ult,
                              coordinate.value, i32Constant(left));
    auto branch = scf::IfOp::create(builder, location,
                                    TypeRange{builder.getF32Type()}, isLeft,
                                    /*addThenBlock=*/true,
                                    /*addElseBlock=*/true);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(branch.thenBlock());
    Emitted leftValue = emitConcatSide(nodeIndex, 0, axis, coordinates,
                                       bound(coordinate, left - 1));
    scf::YieldOp::create(builder, location, leftValue.value);
    builder.setInsertionPointToStart(branch.elseBlock());
    // The subtraction wraps on the other side of the branch; the bound makes
    // the coordinate provable without the branch condition.
    Bounded moved{arith::SubIOp::create(builder, location, coordinate.value,
                                        i32Constant(left)),
                  std::numeric_limits<int32_t>::max(), std::nullopt};
    Emitted rightValue = emitConcatSide(nodeIndex, 1, axis, coordinates,
                                        bound(moved, right - 1));
    scf::YieldOp::create(builder, location, rightValue.value);
    return {branch.getResult(0), std::max(leftValue.loads, rightValue.loads)};
  }

  Emitted emitDynamicConcat(unsigned nodeIndex, int64_t axis,
                            ArrayRef<Bounded> coordinates, Bounded coordinate,
                            Bounded leftExtent, int64_t left, int64_t right) {
    Value isLeft =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ult,
                              valueOf(coordinate), valueOf(leftExtent));
    auto branch = scf::IfOp::create(builder, location,
                                    TypeRange{builder.getF32Type()}, isLeft,
                                    /*addThenBlock=*/true,
                                    /*addElseBlock=*/true);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(branch.thenBlock());
    Emitted leftValue = emitConcatSide(nodeIndex, 0, axis, coordinates,
                                       bound(coordinate, left - 1));
    scf::YieldOp::create(builder, location, leftValue.value);
    builder.setInsertionPointToStart(branch.elseBlock());
    Emitted rightValue =
        emitConcatSide(nodeIndex, 1, axis, coordinates,
                       bound(subtract(coordinate, leftExtent), right - 1));
    scf::YieldOp::create(builder, location, rightValue.value);
    return {branch.getResult(0), std::max(leftValue.loads, rightValue.loads)};
  }

  const DynamicTensor *dynamicOf(unsigned tensor) const {
    if (!dynamic)
      return nullptr;
    auto found = dynamic->find(tensor);
    return found == dynamic->end() ? nullptr : &found->second;
  }

  /// The value of the UdonDNN large activation kernel for `mode` with alpha
  /// 1 and beta 0: a NaN sample is returned as it is.
  Value emitActivation(DeviceFusionKind kind, Value sample) {
    Value one = f32Constant(1.0f);
    Value activated;
    if (kind == DeviceFusionKind::Relu) {
      Value zero = f32Constant(0.0f);
      Value positive = arith::CmpFOp::create(
          builder, location, arith::CmpFPredicate::OGT, sample, zero);
      activated =
          arith::SelectOp::create(builder, location, positive, sample, zero);
    } else if (kind == DeviceFusionKind::Sigmoid) {
      // 1.0f / (1.0f + exp(-sample))
      Value negated = arith::NegFOp::create(builder, location, sample);
      Value exponential = math::ExpOp::create(builder, location, negated);
      Value denominator =
          arith::AddFOp::create(builder, location, one, exponential);
      activated = arith::DivFOp::create(builder, location, one, denominator);
    } else {
      // 2.0f / (1.0f + exp(-2.0f * sample)) - 1.0f
      Value scaled =
          arith::MulFOp::create(builder, location, f32Constant(-2.0f), sample);
      Value exponential = math::ExpOp::create(builder, location, scaled);
      Value denominator =
          arith::AddFOp::create(builder, location, one, exponential);
      Value quotient = arith::DivFOp::create(builder, location,
                                             f32Constant(2.0f), denominator);
      activated = arith::SubFOp::create(builder, location, quotient, one);
    }
    Value isNaN = arith::CmpFOp::create(
        builder, location, arith::CmpFPredicate::UNE, sample, sample);
    return arith::SelectOp::create(builder, location, isNaN, sample, activated);
  }

  // The library kernels scale by alpha = 1.0f and add beta * previous only
  // for beta != 0, which the Host never passes for these nodes. 1.0f * x is
  // x for every x that is not a NaN, including signed zeros, denormals and
  // infinities, and for a NaN it is that NaN (the multiplication can at most
  // quiet a signaling one, which every operation that follows does as well),
  // so the multiplications are not emitted. Everything else keeps the
  // operation order of the library kernel: no reassociation and, as the
  // operations carry no fast-math flags and the shader declares its values
  // precise, no contraction.
  Emitted emitNode(unsigned nodeIndex, IndexMap &map) {
    auto key = std::make_pair(2 * nodeIndex, map.id);
    if (Value known = values.lookup(key))
      return {known, 0};
    const NodeRecord &node = graph.nodes[nodeIndex];
    DeviceFusionKind kind = *deviceFusionKind(graph, node);
    ArrayRef<int64_t> input = shapeOf(graph, node.inputs[0]);
    Emitted result;
    switch (kind) {
    case DeviceFusionKind::Add:
    case DeviceFusionKind::Mul: {
      Emitted a = emitOperand(nodeIndex, 0, broadcast(map, input));
      Emitted b = emitOperand(nodeIndex, 1,
                              broadcast(map, shapeOf(graph, node.inputs[1])));
      result.value = kind == DeviceFusionKind::Add
                         ? Value(arith::AddFOp::create(builder, location,
                                                       a.value, b.value))
                         : Value(arith::MulFOp::create(builder, location,
                                                       a.value, b.value));
      result.loads = a.loads + b.loads;
      break;
    }
    case DeviceFusionKind::Relu:
    case DeviceFusionKind::Sigmoid:
    case DeviceFusionKind::Tanh: {
      result = emitOperand(nodeIndex, 0, reshape(map, input));
      result.value = emitActivation(kind, result.value);
      break;
    }
    case DeviceFusionKind::Cos:
    case DeviceFusionKind::Sin:
    case DeviceFusionKind::Neg: {
      result = emitOperand(nodeIndex, 0, reshape(map, input));
      result.value =
          kind == DeviceFusionKind::Cos
              ? Value(math::CosOp::create(builder, location, result.value))
          : kind == DeviceFusionKind::Sin
              ? Value(math::SinOp::create(builder, location, result.value))
              : Value(arith::NegFOp::create(builder, location, result.value));
      break;
    }
    case DeviceFusionKind::Alias:
      result = emitOperand(nodeIndex, 0, reshape(map, input));
      break;
    case DeviceFusionKind::Expand:
      result = emitOperand(nodeIndex, 0, broadcast(map, input));
      break;
    case DeviceFusionKind::Slice: {
      DeviceSliceShape slice =
          *deviceSliceShape(graph, node, DeviceKernelLimits::Synthesized);
      std::string tag;
      llvm::raw_string_ostream os(tag);
      os << "slice";
      for (auto [start, step] : llvm::zip_equal(slice.starts, slice.steps))
        os << ',' << start << '+' << step;
      IndexMap &source = derived(map, input, tag, [&](IndexMap &derivedMap) {
        ArrayRef<Bounded> coordinates = coordinatesOf(map);
        derivedMap.coordinates.emplace();
        for (auto [axis, extent] : llvm::enumerate(input))
          derivedMap.coordinates->push_back(
              bound(add(multiply(coordinates[axis], slice.steps[axis]),
                        constant(slice.starts[axis])),
                    extent - 1));
      });
      result = emitOperand(nodeIndex, 0, source);
      break;
    }
    case DeviceFusionKind::Transpose: {
      SmallVector<int64_t> permutation = *deviceTransposePermutation(
          graph, node, DeviceKernelLimits::Synthesized);
      std::string tag;
      llvm::raw_string_ostream os(tag);
      os << "transpose";
      for (int64_t axis : permutation)
        os << ',' << axis;
      IndexMap &source = derived(map, input, tag, [&](IndexMap &derivedMap) {
        ArrayRef<Bounded> coordinates = coordinatesOf(map);
        derivedMap.coordinates.emplace(input.size(), constant(0));
        for (auto [outputAxis, inputAxis] : llvm::enumerate(permutation))
          (*derivedMap.coordinates)[inputAxis] = coordinates[outputAxis];
      });
      result = emitOperand(nodeIndex, 0, source);
      break;
    }
    case DeviceFusionKind::Concat:
      result = emitConcat(nodeIndex, map);
      break;
    case DeviceFusionKind::Softmax:
      llvm_unreachable("a reduction is only a group root");
    }
    values[key] = result.value;
    return result;
  }

  /// The offset of the view on its axis: the constant, or the bounded
  /// run-time value.
  Bounded viewOffsetOf() {
    return view->dynamicOffset ? *viewOffset : constant(view->offset);
  }

  OpBuilder &builder;
  Location location;
  const DeviceFusion &fusion;
  const GraphRecord &graph;
  ArrayRef<unsigned> leaves;
  ArrayRef<BlockArgument> buffers;
  const DeviceFusionLeafView *view;
  std::optional<Bounded> viewOffset;
  const DynamicTensors *dynamic;
  std::deque<IndexMap> maps;
  std::map<std::string, IndexMap *> derivedMaps;
  /// Value of a node (key 2 * node) or of a leaf Load (key 2 * tensor + 1)
  /// for an index map.
  llvm::DenseMap<std::pair<unsigned, unsigned>, Value> values;
};
} // namespace

FailureOr<DeviceFusionDispatch>
DeviceFusionKernels::dispatch(unsigned rootNode,
                              const DeviceConstantPrefix *prefix) {
  const GraphRecord &graph = fusion.graph;
  // A node outside every group is a group of its own when its library kernel
  // cannot run it: a Softmax that runs in two passes, or a node beyond the
  // limits of its library kernel.
  DeviceRowReduction schedule{0, 0};
  if (!fusion.isInterior(rootNode))
    schedule = deviceSoftmaxSchedule(graph, graph.nodes[rootNode]);
  std::optional<DeviceFusionKind> kind =
      deviceFusionKind(graph, graph.nodes[rootNode]);
  if (fusion.isInterior(rootNode) || !kind ||
      *kind == DeviceFusionKind::Alias ||
      (!fusion.isRoot(rootNode) && !schedule.chunks &&
       !isSynthesizedDeviceNode(graph, graph.nodes[rootNode])))
    return emitError(module.getLoc())
           << "ONNX Device fusion group of node " << rootNode
           << " is outside the fused kernel contract";
  if (!schedule.chunks)
    return synthesize(rootNode, nullptr, nullptr, FusedKernelRole::Whole,
                      prefix);
  FailureOr<DeviceFusionDispatch> statistics = synthesize(
      rootNode, nullptr, &schedule, FusedKernelRole::RowStatistics, nullptr);
  if (failed(statistics))
    return failure();
  FailureOr<DeviceFusionDispatch> elementwise = synthesize(
      rootNode, nullptr, &schedule, FusedKernelRole::FromRowStatistics, prefix);
  if (failed(elementwise))
    return failure();
  elementwise->statisticsCallee = statistics->callee;
  elementwise->statisticsWords = statisticsSurfaceWords(
      2 * deviceSoftmaxRows(graph, graph.nodes[rootNode])->rows() *
      schedule.chunks);
  return elementwise;
}

FailureOr<DeviceFusionDispatch>
DeviceFusionKernels::dispatch(unsigned rootNode,
                              const DeviceFusionLeafView &view,
                              const DeviceConstantPrefix *prefix) {
  const GraphRecord &graph = fusion.graph;
  std::optional<DeviceFusionKind> kind =
      deviceFusionKind(graph, graph.nodes[rootNode]);
  if (fusion.isInterior(rootNode) || !kind ||
      *kind == DeviceFusionKind::Alias || isDeviceFusionReduction(*kind) ||
      view.shape.size() != shapeOf(graph, view.tensor).size() ||
      view.axis < 0 || view.axis >= static_cast<int64_t>(view.shape.size()) ||
      elementCount(view.shape) > maximumSurfaceWords)
    return emitError(module.getLoc())
           << "ONNX Device view of tensor " << view.tensor << " in node "
           << rootNode << " is outside the fused kernel contract";
  for (auto [axis, extent] : llvm::enumerate(shapeOf(graph, view.tensor))) {
    int64_t moved = static_cast<int64_t>(axis) == view.axis ? view.offset : 0;
    if (moved < 0 || extent + moved > view.shape[axis])
      return emitError(module.getLoc())
             << "ONNX Device view of tensor " << view.tensor
             << " does not fit its buffer";
  }
  return synthesize(rootNode, &view, nullptr, FusedKernelRole::Whole, prefix);
}

FailureOr<DeviceFusionDispatch>
DeviceFusionKernels::dispatchInPlace(unsigned rootNode) {
  const GraphRecord &graph = fusion.graph;
  if (fusion.isInterior(rootNode) ||
      (!fusion.isRoot(rootNode) &&
       !isSynthesizedDeviceNode(graph, graph.nodes[rootNode])))
    return emitError(module.getLoc())
           << "ONNX in-place Concat of node " << rootNode
           << " is outside the fused kernel contract";
  return synthesize(rootNode, nullptr, nullptr, FusedKernelRole::Whole, nullptr,
                    /*inPlace=*/true);
}

FailureOr<DeviceFusionDispatch> DeviceFusionKernels::synthesize(
    unsigned rootNode, const DeviceFusionLeafView *view,
    const DeviceRowReduction *schedule, FusedKernelRole role,
    const DeviceConstantPrefix *prefix, bool inPlace) {
  if (role == FusedKernelRole::RowStatistics)
    prefix = nullptr;
  const GraphRecord &graph = fusion.graph;
  Location location = module.getLoc();
  DeviceFusionDispatch result;
  result.leaves = fusion.leaves(rootNode);
  // An in-place Concat leaves its first operand where it is: the kernel does
  // not take it, so the group may read it only as that operand.
  std::optional<unsigned> leadingBlock;
  if (inPlace) {
    const NodeRecord &root = graph.nodes[rootNode];
    DeviceFusionOperand first = fusion.operand(rootNode, 0);
    if (role != FusedKernelRole::Whole || view || prefix ||
        root.operation != "Concat" || root.inputs.size() != 2 ||
        first.kind != DeviceFusionOperand::Leaf ||
        static_cast<int64_t>(first.tensor) != root.inputs[0] ||
        !graph.tensors[first.tensor].dynamic ||
        !graph.tensors[root.output].dynamic)
      return emitError(location) << "ONNX in-place Concat of node " << rootNode
                                 << " is outside the fused kernel contract";
    for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes)) {
      if (nodeIndex != rootNode && fusion.roots[nodeIndex] != rootNode)
        continue;
      for (auto [operand, input] : llvm::enumerate(node.inputs))
        if (input == static_cast<int64_t>(first.tensor) &&
            (nodeIndex != rootNode || operand != 0))
          return emitError(location)
                 << "ONNX in-place Concat of node " << rootNode
                 << " reads its first operand elsewhere in its group";
    }
    leadingBlock = first.tensor;
    llvm::erase(result.leaves, first.tensor);
    if (result.leaves.empty())
      return emitError(location) << "ONNX in-place Concat of node " << rootNode
                                 << " has no operand to write";
  }
  DeviceFusionCost cost = fusion.cost(rootNode);
  int64_t elements =
      graph.tensors[graph.nodes[rootNode].output].type.getNumElements();
  // The words the statistics pass writes and the elementwise pass reads.
  int64_t statisticsWords = 0;
  int64_t rows = 0;
  if (role != FusedKernelRole::Whole) {
    std::optional<DeviceSoftmaxRows> shape =
        deviceSoftmaxRows(graph, graph.nodes[rootNode]);
    if (!shape || !schedule || schedule->chunks < 1 || view)
      return emitError(location)
             << "ONNX Device row reduction of node " << rootNode
             << " is outside the fused kernel contract";
    rows = shape->rows();
    statisticsWords = 2 * rows * schedule->chunks;
    if (role == FusedKernelRole::RowStatistics)
      elements = statisticsWords;
  }
  // The tensors of the group with an axis of a bounded dimension: their
  // run-time extents, and for the root and the leaves the layout of their
  // buffers. The kernel takes the value of every dimension they name and the
  // pitch of every binding among them as i32 scalars after its others.
  unsigned rootTensor = graph.nodes[rootNode].output;
  SmallVector<unsigned> dynamicTensors;
  auto noteDynamic = [&](unsigned tensor) {
    if (graph.tensors[tensor].dynamic &&
        !llvm::is_contained(dynamicTensors, tensor))
      dynamicTensors.push_back(tensor);
  };
  noteDynamic(rootTensor);
  for (unsigned leaf : result.leaves)
    noteDynamic(leaf);
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes))
    if (nodeIndex == rootNode || fusion.roots[nodeIndex] == rootNode) {
      noteDynamic(node.output);
      for (int64_t input : node.inputs)
        if (input >= 0)
          noteDynamic(static_cast<unsigned>(input));
    }
  std::map<unsigned, DeviceFusionLayout> layoutOf;
  auto isStorage = [&](unsigned tensor) {
    return tensor == rootTensor || llvm::is_contained(result.leaves, tensor);
  };
  for (unsigned tensor : dynamicTensors) {
    if (!isStorage(tensor))
      continue;
    if (!layouts || view || prefix)
      return emitError(location)
             << "ONNX Device fusion group of node " << rootNode
             << " has a dynamic extent outside the fused kernel contract";
    layoutOf[tensor] = layouts(tensor);
  }
  for (unsigned tensor : dynamicTensors) {
    const auto &axis = *graph.tensors[tensor].dynamic;
    DeviceFusionScalar scalar{
        DeviceFusionScalar::Dimension,
        static_cast<unsigned>(dimensionIndex(graph, axis.symbol))};
    if (!llvm::is_contained(result.scalars, scalar))
      result.scalars.push_back(scalar);
  }
  // The buffers in kernel order: the root, then the leaves. Groups of the
  // same structure (the layers of a model) share a kernel, so everything a
  // key records about them is by position, never by tensor or binding.
  SmallVector<unsigned> storageOrder{rootTensor};
  llvm::append_range(storageOrder, result.leaves);
  for (unsigned tensor : storageOrder) {
    auto found = layoutOf.find(tensor);
    if (found == layoutOf.end() ||
        found->second.kind != DeviceFusionLayout::Binding)
      continue;
    DeviceFusionScalar scalar{DeviceFusionScalar::BoundPitch,
                              found->second.binding, found->second.output};
    if (!llvm::is_contained(result.scalars, scalar))
      result.scalars.push_back(scalar);
  }
  // The words of the buffer of a dynamic leaf or root: a binding may give
  // its axis a pitch up to the largest extent of its dimension.
  auto bufferWords = [&](unsigned tensor) -> int64_t {
    RankedTensorType type = graph.tensors[tensor].type;
    auto found = layoutOf.find(tensor);
    if (found == layoutOf.end() ||
        found->second.kind != DeviceFusionLayout::Binding)
      return type.getNumElements();
    int64_t axis = graph.tensors[tensor].dynamic->axis;
    return type.getNumElements() / type.getDimSize(axis) *
           found->second.capacity;
  };
  if (role != FusedKernelRole::RowStatistics && layoutOf.count(rootTensor))
    elements = bufferWords(rootTensor);
  if (!cost.withinCaps() || result.leaves.empty() ||
      elements > maximumSurfaceWords ||
      (view && !llvm::is_contained(result.leaves, view->tensor)))
    return emitError(location)
           << "ONNX Device fusion group of node " << rootNode
           << " is outside the fused kernel contract";

  std::string key;
  llvm::raw_string_ostream keyStream(key);
  structureKey(fusion, result.leaves, rootNode, keyStream);
  if (view) {
    keyStream << "|view%"
              << (llvm::find(result.leaves, view->tensor) -
                  result.leaves.begin())
              << '[';
    llvm::interleave(view->shape, keyStream, "x");
    keyStream << "]@" << view->axis << (view->dynamicOffset ? "+<=" : "+")
              << view->offset;
  }
  if (role != FusedKernelRole::Whole)
    keyStream << (role == FusedKernelRole::RowStatistics ? "|statistics"
                                                         : "|from-statistics")
              << schedule->chunks << 'x' << schedule->chunkLength;
  if (leadingBlock)
    keyStream << "|in-place";
  for (auto [position, tensor] : llvm::enumerate(storageOrder))
    if (auto found = layoutOf.find(tensor); found != layoutOf.end())
      keyStream << "|layout@" << position << '=' << found->second.kind << ':'
                << found->second.capacity;
  keyStream << "|scalars";
  for (const DeviceFusionScalar &scalar : result.scalars) {
    if (scalar.kind == DeviceFusionScalar::Dimension) {
      keyStream << ":d" << scalar.index;
      continue;
    }
    // A pitch is that of the first buffer with its binding.
    for (auto [position, tensor] : llvm::enumerate(storageOrder))
      if (auto found = layoutOf.find(tensor);
          found != layoutOf.end() &&
          found->second.kind == DeviceFusionLayout::Binding &&
          found->second.binding == scalar.index &&
          found->second.output == scalar.output) {
        keyStream << ":p" << position;
        break;
      }
  }
  ArrayRef<int64_t> rootShape = shapeOf(graph, graph.nodes[rootNode].output);
  if (prefix) {
    if (prefix->prefix.axis < 0 ||
        prefix->prefix.axis >= static_cast<int64_t>(rootShape.size()))
      return emitError(location) << "ONNX Device constant prefix of node "
                                 << rootNode << " is outside its result";
    keyStream << "|prefix@" << prefix->prefix.axis << '=' << prefix->bits;
  }
  auto [entry, inserted] = callees.try_emplace(key);
  if (!inserted) {
    result.callee = entry->second;
    return result;
  }

  OpBuilder builder(module.getContext());
  if (!carrier) {
    if (SymbolTable::lookupSymbolIn(module, carrierName))
      return emitError(location)
             << "ONNX Host module already defines @" << carrierName;
    builder.setInsertionPointToEnd(module.getBody());
    carrier = gpu::GPUModuleOp::create(builder, location, carrierName);
  }
  unsigned ordinal = callees.size() - 1;
  std::string kernelName = ("udononnx_fused_" + Twine(ordinal)).str();
  result.callee = "__vrc_opencl_dispatch_" + kernelName;
  if (SymbolTable::lookupSymbolIn(module, result.callee))
    return emitError(location)
           << "ONNX Host module already defines @" << result.callee;
  entry->second = result.callee;

  Type f32 = builder.getF32Type();
  Type i32 = builder.getI32Type();
  SmallVector<Type> arguments;
  for (unsigned leaf : result.leaves)
    arguments.push_back(MemRefType::get({view && view->tensor == leaf
                                             ? elementCount(view->shape)
                                             : bufferWords(leaf)},
                                        f32));
  // The statistics buffer is the operator scratch, read as the whole surface
  // the statistics pass writes: a buffer texture is at most surfaceWidth
  // words wide, so this keeps the addressing of the scratch texture.
  if (role == FusedKernelRole::FromRowStatistics)
    arguments.push_back(
        MemRefType::get({statisticsSurfaceWords(statisticsWords)}, f32));
  int64_t surfaceHeight = (elements + surfaceWidth - 1) / surfaceWidth;
  int64_t surfaceWords = surfaceWidth * surfaceHeight;
  arguments.push_back(MemRefType::get({surfaceWords}, f32));
  unsigned bufferArguments = arguments.size();
  // The zero-row extent of the constant prefix follows the buffers, and the
  // run-time offset of a view comes last.
  if (prefix)
    arguments.push_back(i32);
  bool dynamicView = view && view->dynamicOffset;
  if (dynamicView)
    arguments.push_back(i32);
  unsigned firstScalar = arguments.size();
  for (size_t index = 0; index < result.scalars.size(); ++index)
    arguments.push_back(i32);
  builder.setInsertionPointToEnd(cast<gpu::GPUModuleOp>(carrier).getBody());
  auto kernel = gpu::GPUFuncOp::create(
      builder, location, kernelName,
      FunctionType::get(module.getContext(), arguments, {}));
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());
  kernel->setAttr("known_block_size", builder.getDenseI32ArrayAttr({1, 1, 1}));
  kernel->setAttr(
      "known_grid_size",
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(surfaceWidth),
                                    static_cast<int32_t>(surfaceHeight), 1}));
  kernel->setAttr("vrc.profile", builder.getStringAttr("gather-f32"));
  kernel->setAttr("vrc.synthesized_kernel", builder.getUnitAttr());
  kernel->setAttr("vrc.words", builder.getI32IntegerAttr(surfaceWords));
  // The run passes a dimension within its bound and a pitch within the
  // largest extent of its dimension.
  if (!result.scalars.empty()) {
    SmallVector<Attribute> bounds;
    for (auto [index, scalar] : llvm::enumerate(result.scalars)) {
      int64_t maximum =
          scalar.kind == DeviceFusionScalar::Dimension
              ? graph.dimensionBounds[scalar.index].second
              : [&] {
                  for (const auto &[tensor, layout] : layoutOf)
                    if (layout.kind == DeviceFusionLayout::Binding &&
                        layout.binding == scalar.index &&
                        layout.output == scalar.output)
                      return layout.capacity;
                  return int64_t{0};
                }();
      bounds.push_back(builder.getDictionaryAttr(
          {builder.getNamedAttr("index",
                                builder.getI64IntegerAttr(firstScalar + index)),
           builder.getNamedAttr("max", builder.getI64IntegerAttr(maximum)),
           builder.getNamedAttr("min", builder.getI64IntegerAttr(0))}));
    }
    kernel->setAttr("vrc.argument_bounds", builder.getArrayAttr(bounds));
  }

  // Every fragment is defined, also those of a buffer taller than the launch
  // surface: the fragments of the root's elements evaluate the expression,
  // the others write zero like the library kernels do. All Loads are inside
  // the guard, and their bounds follow from it rather than from the surface.
  Block &body = kernel.getBody().front();
  builder.setInsertionPointToEnd(&body);
  // A run-time view offset is bounded once by the largest offset the view
  // allows: a minui with a constant, which the gather prover derives an
  // interval from, so every Load through the view stays proven in bounds.
  std::optional<Bounded> viewOffset;
  if (dynamicView)
    viewOffset = Bounded{
        arith::MinUIOp::create(
            builder, location, body.getArguments().back(),
            arith::ConstantIntOp::create(builder, location, view->offset, 32)),
        view->offset, std::nullopt};
  Value x = gpu::GlobalIdOp::create(builder, location, gpu::Dimension::x);
  Value y = gpu::GlobalIdOp::create(builder, location, gpu::Dimension::y);
  Value row = arith::MulIOp::create(
      builder, location, y,
      arith::ConstantIndexOp::create(builder, location, surfaceWidth));
  Value fragment = arith::AddIOp::create(builder, location, x, row);
  Value linear = arith::IndexCastOp::create(builder, location, i32, fragment);
  Value live = arith::CmpIOp::create(
      builder, location, arith::CmpIPredicate::ult, linear,
      arith::ConstantIntOp::create(builder, location, elements, 32));
  // The run-time extents and pitches, each bounded once (a minui with a
  // constant, which the gather prover derives an interval from). A pitch is
  // at least 1 so that it divides: an empty dynamic axis has no element to
  // address anyway.
  DynamicTensors dynamic;
  SmallVector<Bounded> scalarValues;
  for (auto [index, scalar] : llvm::enumerate(result.scalars)) {
    int64_t maximum = 0;
    if (scalar.kind == DeviceFusionScalar::Dimension)
      maximum = graph.dimensionBounds[scalar.index].second;
    else
      for (const auto &[tensor, layout] : layoutOf)
        if (layout.kind == DeviceFusionLayout::Binding &&
            layout.binding == scalar.index && layout.output == scalar.output)
          maximum = layout.capacity;
    Value value = arith::MinUIOp::create(
        builder, location, body.getArgument(firstScalar + index),
        arith::ConstantIntOp::create(builder, location, maximum, 32));
    scalarValues.push_back(Bounded{value, maximum, std::nullopt});
  }
  auto atLeastOne = [&](const Bounded &value) -> Bounded {
    if (value.constant)
      return Bounded{Value(), std::max<int64_t>(*value.constant, 1),
                     std::max<int64_t>(*value.constant, 1)};
    Value zero = arith::ConstantIntOp::create(builder, location, 0, 32);
    Value positive = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::ugt, value.value, zero);
    Value selected = arith::SelectOp::create(
        builder, location, positive, value.value,
        arith::ConstantIntOp::create(builder, location, 1, 32));
    return Bounded{selected, std::max<int64_t>(value.max, 1), std::nullopt};
  };
  for (unsigned tensor : dynamicTensors) {
    const auto &axis = *graph.tensors[tensor].dynamic;
    unsigned symbol = dimensionIndex(graph, axis.symbol);
    unsigned scalar =
        llvm::find(result.scalars,
                   DeviceFusionScalar{DeviceFusionScalar::Dimension, symbol}) -
        result.scalars.begin();
    const Bounded &value = scalarValues[scalar];
    Bounded extent = axis.offset
                         ? Bounded{arith::AddIOp::create(
                                       builder, location, value.value,
                                       arith::ConstantIntOp::create(
                                           builder, location, axis.offset, 32)),
                                   value.max + axis.offset, std::nullopt}
                         : value;
    DynamicTensor entry{axis.axis, extent, atLeastOne(extent),
                        graph.tensors[tensor].type.getNumElements()};
    if (auto found = layoutOf.find(tensor); found != layoutOf.end()) {
      entry.capacityWords = bufferWords(tensor);
      if (found->second.kind == DeviceFusionLayout::Bound) {
        int64_t pitch = graph.tensors[tensor].type.getDimSize(axis.axis);
        entry.pitch = Bounded{Value(), pitch, pitch};
      } else if (found->second.kind == DeviceFusionLayout::Binding) {
        unsigned pitch =
            llvm::find(result.scalars,
                       DeviceFusionScalar{DeviceFusionScalar::BoundPitch,
                                          found->second.binding,
                                          found->second.output}) -
            result.scalars.begin();
        entry.pitch = atLeastOne(scalarValues[pitch]);
      }
    }
    dynamic[tensor] = entry;
  }
  // A root with a dynamic axis is decoded with the pitch of its buffer, and
  // only the elements inside its run-time extent are evaluated.
  SmallVector<Bounded> rootCoordinates;
  if (role != FusedKernelRole::RowStatistics && dynamic.count(rootTensor)) {
    ExpressionEmitter decoder(builder, location, fusion, result.leaves, {},
                              nullptr, std::nullopt, &dynamic);
    auto [coordinates, inside] = decoder.rootElement(rootNode, linear);
    rootCoordinates = std::move(coordinates);
    live = arith::AndIOp::create(builder, location, live, inside);
  }
  // An in-place Concat writes only the elements after its first operand's
  // block along the axis. `live` decides which fragments store; it depends
  // on the fragment's row (the words after the axis) alone, so it is the
  // same for every word of a row.
  int64_t granule = 0;
  if (leadingBlock) {
    int64_t axis = concatAxis(graph, graph.nodes[rootNode]);
    const DynamicTensor &first = dynamic.at(*leadingBlock);
    auto valueOf = [&](const Bounded &bounded) -> Value {
      if (bounded.constant)
        return arith::ConstantIntOp::create(builder, location,
                                            *bounded.constant, 32);
      return bounded.value;
    };
    Value after = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::uge,
        valueOf(rootCoordinates[axis]), valueOf(first.extent));
    live = arith::AndIOp::create(builder, location, live, after);
    granule =
        std::gcd(elementCount(rootShape.drop_front(axis + 1)), surfaceWords);
  }
  auto guard = scf::IfOp::create(builder, location, TypeRange{f32}, live,
                                 /*addThenBlock=*/true, /*addElseBlock=*/true);
  builder.setInsertionPointToStart(guard.thenBlock());
  SmallVector<BlockArgument> buffers(
      body.getArguments().take_front(bufferArguments));
  // An element of the constant prefix is the prefix's value; its expression
  // is evaluated, and its operands loaded, only outside the prefix. The
  // condition depends on the element's coordinate and the run-time extent,
  // never on a Load, so the Load bounds are the same in both branches.
  scf::IfOp prefixBranch;
  if (prefix) {
    int64_t axis = prefix->prefix.axis;
    int64_t inner = elementCount(rootShape.drop_front(axis + 1));
    Value coordinate = linear;
    if (inner != 1)
      coordinate = arith::DivUIOp::create(
          builder, location, coordinate,
          arith::ConstantIntOp::create(builder, location, inner, 32));
    if (axis != 0)
      coordinate = arith::RemUIOp::create(
          builder, location, coordinate,
          arith::ConstantIntOp::create(builder, location, rootShape[axis], 32));
    Value inPrefix =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ult,
                              coordinate, body.getArgument(bufferArguments));
    prefixBranch = scf::IfOp::create(builder, location, TypeRange{f32},
                                     inPrefix, /*addThenBlock=*/true,
                                     /*addElseBlock=*/true);
    builder.setInsertionPointToStart(prefixBranch.thenBlock());
    scf::YieldOp::create(builder, location,
                         Value(arith::ConstantFloatOp::create(
                             builder, location, builder.getF32Type(),
                             APFloat(llvm::bit_cast<float>(prefix->bits)))));
    builder.setInsertionPointToStart(prefixBranch.elseBlock());
  }
  ExpressionEmitter emitter(
      builder, location, fusion, result.leaves,
      ArrayRef<BlockArgument>(buffers).take_front(result.leaves.size()), view,
      viewOffset, dynamic.empty() ? nullptr : &dynamic);
  Emitted value;
  if (leadingBlock) {
    value = emitter.emitConcatTail(rootNode, rootCoordinates);
  } else if (role == FusedKernelRole::Whole) {
    value = emitter.emitRoot(rootNode, linear, nullptr, rootCoordinates);
  } else {
    DeviceRowStatistics statistics{rows, schedule->chunks,
                                   schedule->chunkLength,
                                   buffers[result.leaves.size()]};
    value =
        role == FusedKernelRole::RowStatistics
            ? emitter.emitSoftmaxStatistics(rootNode, linear, statistics)
            : emitter.emitRoot(rootNode, linear, &statistics, rootCoordinates);
  }
  if (prefixBranch) {
    scf::YieldOp::create(builder, location, value.value);
    builder.setInsertionPointAfter(prefixBranch);
    value.value = prefixBranch.getResult(0);
  }
  scf::YieldOp::create(builder, location, value.value);
  builder.setInsertionPointToStart(guard.elseBlock());
  scf::YieldOp::create(
      builder, location,
      Value(arith::ConstantFloatOp::create(
          builder, location, builder.getF32Type(), APFloat(0.0f))));
  builder.setInsertionPointToEnd(&body);
  if (leadingBlock) {
    // The fragments outside the written elements keep their words.
    auto store = scf::IfOp::create(builder, location, live,
                                   /*withElseRegion=*/false);
    builder.setInsertionPointToStart(store.thenBlock());
    memref::StoreOp::create(builder, location, guard.getResult(0),
                            buffers.back(), ValueRange{fragment});
    builder.setInsertionPointToEnd(&body);
    kernel->setAttr("vrc.output_granule", builder.getI64IntegerAttr(granule));
  } else {
    memref::StoreOp::create(builder, location, guard.getResult(0),
                            buffers.back(), ValueRange{fragment});
  }
  gpu::ReturnOp::create(builder, location);
  kernel->setAttr("vrc.fragment_cost",
                  builder.getDictionaryAttr(builder.getNamedAttr(
                      "loads", builder.getI64IntegerAttr(value.loads))));

  // The dispatch declaration takes one buffer token per kernel argument.
  builder.setInsertionPointToEnd(module.getBody());
  SmallVector<Type> tokens(arguments.size(), i32);
  auto declaration = func::FuncOp::create(
      builder, location, result.callee,
      FunctionType::get(module.getContext(), tokens, {i32}));
  declaration.setPrivate();
  declaration->setAttr(
      "vrc.synthesized_dispatch",
      SymbolRefAttr::get(
          module.getContext(), carrierName,
          {FlatSymbolRefAttr::get(module.getContext(), kernelName)}));
  return result;
}
} // namespace vrc
