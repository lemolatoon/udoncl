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

using namespace mlir;

namespace vrc {
namespace {
// The surface of the UdonDNN large pointwise kernels: a Device buffer is a
// texture of this width, and the fragment at (x, y) writes word
// x + y * surfaceWidth. A fused kernel launches over the rows that hold the
// elements of its root, so it fits every buffer that holds that tensor; a
// bundle whose buffers are smaller than the 16 rows of the library kernels
// still accepts it.
constexpr int64_t surfaceWidth = 8192;
constexpr int64_t maximumSurfaceWords = surfaceWidth * 16;

constexpr StringLiteral carrierName = "udononnx_fused_kernels";

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
  os << node.operation << '[';
  llvm::interleave(shapeOf(graph, node.output), os, "x");
  os << ']';
  if (kind == DeviceFusionKind::Slice) {
    DeviceSliceShape slice = *deviceSliceShape(graph, node);
    os << '{';
    llvm::interleaveComma(slice.starts, os);
    os << ';';
    llvm::interleaveComma(slice.steps, os);
    os << '}';
  } else if (kind == DeviceFusionKind::Transpose) {
    os << '{';
    llvm::interleaveComma(*deviceTransposePermutation(graph, node), os);
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

/// Emits the expression of one fusion group at the builder's insertion point.
class ExpressionEmitter {
public:
  ExpressionEmitter(OpBuilder &builder, Location location,
                    const DeviceFusion &fusion, ArrayRef<unsigned> leaves,
                    ArrayRef<BlockArgument> buffers,
                    const DeviceFusionLeafView *view)
      : builder(builder), location(location), fusion(fusion),
        graph(fusion.graph), leaves(leaves), buffers(buffers), view(view) {}

  Emitted emitRoot(unsigned rootNode, Value linear) {
    ArrayRef<int64_t> shape = shapeOf(graph, graph.nodes[rootNode].output);
    IndexMap &map = newMap(shape);
    map.linear = Bounded{linear, elementCount(shape) - 1, std::nullopt};
    if (*deviceFusionKind(graph, graph.nodes[rootNode]) ==
        DeviceFusionKind::Softmax)
      return emitSoftmax(rootNode, map);
    return emitNode(rootNode, map);
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
    Type i32 = builder.getI32Type();
    Type f32 = builder.getF32Type();
    ArrayRef<Bounded> coordinates = coordinatesOf(map);
    int64_t extent = map.shape[axis];
    // The limit and the step are defined outside the loop, so the proof
    // recognizes a counted loop.
    Value limit = i32Constant(extent);
    Value one = i32Constant(1);
    auto loop = scf::WhileOp::create(builder, location, TypeRange{i32, f32},
                                     ValueRange{i32Constant(0), initial});
    OpBuilder::InsertionGuard guard(builder);
    Block *before = builder.createBlock(&loop.getBefore(), {}, {i32, f32},
                                        {location, location});
    Value inside =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ult,
                              before->getArgument(0), limit);
    scf::ConditionOp::create(builder, location, inside, before->getArguments());
    Block *after = builder.createBlock(&loop.getAfter(), {}, {i32, f32},
                                       {location, location});
    IndexMap &element = newMap(map.shape);
    element.coordinates = SmallVector<Bounded>(coordinates);
    (*element.coordinates)[axis] =
        Bounded{after->getArgument(0), extent - 1, std::nullopt};
    Emitted sample = emitOperand(rootNode, 0, element);
    Value next =
        arith::AddIOp::create(builder, location, after->getArgument(0), one);
    scf::YieldOp::create(
        builder, location,
        ValueRange{next, combine(after->getArgument(1), sample.value)});
    return {loop.getResult(1), sample.loads};
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
    std::pair<Value, unsigned> maximum = foldAxis(
        rootNode, map, axis, f32Constant(-3.402823466e+38f),
        [&](Value current, Value sample) -> Value {
          Value greater = arith::CmpFOp::create(
              builder, location, arith::CmpFPredicate::OGT, sample, current);
          return arith::SelectOp::create(builder, location, greater, sample,
                                         current);
        });
    std::pair<Value, unsigned> denominator = foldAxis(
        rootNode, map, axis, f32Constant(0.0f),
        [&](Value sum, Value sample) -> Value {
          Value shifted =
              arith::SubFOp::create(builder, location, sample, maximum.first);
          Value exponential = math::ExpOp::create(builder, location, shifted);
          return arith::AddFOp::create(builder, location, sum, exponential);
        });
    Emitted own = emitOperand(rootNode, 0, map);
    Value shifted =
        arith::SubFOp::create(builder, location, own.value, maximum.first);
    Value exponential = math::ExpOp::create(builder, location, shifted);
    Value value = arith::DivFOp::create(builder, location, exponential,
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
          coordinate = add(coordinate, constant(view->offset));
        linear = add(multiply(linear, extent), coordinate);
      }
      capacity = elementCount(view->shape);
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
      DeviceSliceShape slice = *deviceSliceShape(graph, node);
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
      SmallVector<int64_t> permutation =
          *deviceTransposePermutation(graph, node);
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

  OpBuilder &builder;
  Location location;
  const DeviceFusion &fusion;
  const GraphRecord &graph;
  ArrayRef<unsigned> leaves;
  ArrayRef<BlockArgument> buffers;
  const DeviceFusionLeafView *view;
  std::deque<IndexMap> maps;
  std::map<std::string, IndexMap *> derivedMaps;
  /// Value of a node (key 2 * node) or of a leaf Load (key 2 * tensor + 1)
  /// for an index map.
  llvm::DenseMap<std::pair<unsigned, unsigned>, Value> values;
};
} // namespace

FailureOr<DeviceFusionDispatch>
DeviceFusionKernels::dispatch(unsigned rootNode) {
  if (!fusion.isRoot(rootNode))
    return emitError(module.getLoc())
           << "ONNX Device fusion group of node " << rootNode
           << " is outside the fused kernel contract";
  return synthesize(rootNode, nullptr);
}

FailureOr<DeviceFusionDispatch>
DeviceFusionKernels::dispatchAlone(unsigned node) {
  const GraphRecord &graph = fusion.graph;
  std::optional<DeviceFusionKind> kind =
      deviceFusionKind(graph, graph.nodes[node]);
  if (fusion.isRoot(node) || fusion.isInterior(node) || !kind ||
      *kind == DeviceFusionKind::Alias || isDeviceFusionReduction(*kind))
    return emitError(module.getLoc())
           << "ONNX Device node " << node
           << " is outside the fused kernel contract";
  return synthesize(node, nullptr);
}

FailureOr<DeviceFusionDispatch>
DeviceFusionKernels::dispatch(unsigned rootNode,
                              const DeviceFusionLeafView &view) {
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
  return synthesize(rootNode, &view);
}

FailureOr<DeviceFusionDispatch>
DeviceFusionKernels::synthesize(unsigned rootNode,
                                const DeviceFusionLeafView *view) {
  const GraphRecord &graph = fusion.graph;
  Location location = module.getLoc();
  DeviceFusionDispatch result;
  result.leaves = fusion.leaves(rootNode);
  DeviceFusionCost cost = fusion.cost(rootNode);
  int64_t elements =
      graph.tensors[graph.nodes[rootNode].output].type.getNumElements();
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
    keyStream << "]@" << view->axis << '+' << view->offset;
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
    arguments.push_back(
        MemRefType::get({view && view->tensor == leaf
                             ? elementCount(view->shape)
                             : graph.tensors[leaf].type.getNumElements()},
                        f32));
  int64_t surfaceHeight = (elements + surfaceWidth - 1) / surfaceWidth;
  int64_t surfaceWords = surfaceWidth * surfaceHeight;
  arguments.push_back(MemRefType::get({surfaceWords}, f32));
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

  // Every fragment is defined, also those of a buffer taller than the launch
  // surface: the fragments of the root's elements evaluate the expression,
  // the others write zero like the library kernels do. All Loads are inside
  // the guard, and their bounds follow from it rather than from the surface.
  Block &body = kernel.getBody().front();
  builder.setInsertionPointToEnd(&body);
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
  auto guard = scf::IfOp::create(builder, location, TypeRange{f32}, live,
                                 /*addThenBlock=*/true, /*addElseBlock=*/true);
  builder.setInsertionPointToStart(guard.thenBlock());
  SmallVector<BlockArgument> buffers(body.getArguments());
  ExpressionEmitter emitter(builder, location, fusion, result.leaves,
                            ArrayRef<BlockArgument>(buffers).drop_back(), view);
  Emitted value = emitter.emitRoot(rootNode, linear);
  scf::YieldOp::create(builder, location, value.value);
  builder.setInsertionPointToStart(guard.elseBlock());
  scf::YieldOp::create(
      builder, location,
      Value(arith::ConstantFloatOp::create(
          builder, location, builder.getF32Type(), APFloat(0.0f))));
  builder.setInsertionPointToEnd(&body);
  memref::StoreOp::create(builder, location, guard.getResult(0), buffers.back(),
                          ValueRange{fragment});
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
