// Merges MatMulNBits nodes that read the same activation into one node over
// the concatenated weights, followed by one Slice per original result.
//
// com.microsoft::MatMulNBits computes every output column n from row n of the
// packed weights B[N, blocks, blob] and the scales of that row alone. Stacking
// the rows of several nodes with equal K, bits and block size therefore
// computes every column with the same operands in the same order, and the
// Slices recover each original result word for word.

#include "VRC/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MERGEONNXSIBLINGMATMULNBITSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
constexpr StringLiteral initializerPrefix = "Initializer_";

bool isMatMulNBits(Operation *operation) {
  if (operation->getName().getStringRef() != "onnx.Custom" ||
      operation->getNumResults() != 1 || operation->getNumOperands() < 3)
    return false;
  auto function = operation->getAttrOfType<StringAttr>("function_name");
  auto domain = operation->getAttrOfType<StringAttr>("domain_name");
  return function && function.getValue() == "MatMulNBits" && domain &&
         domain.getValue() == "com.microsoft";
}

DenseElementsAttr constantElements(Value value) {
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return {};
  StringRef name = operation->getName().getStringRef();
  if (name != "onnx.Constant" && name != "arith.constant")
    return {};
  if (auto dense = operation->getAttrOfType<DenseElementsAttr>("value"))
    return dense;
  if (auto disposable =
          operation->getAttrOfType<DisposableElementsAttr>("value"))
    return disposable.toDenseElementsAttr();
  return {};
}

int64_t integerAttribute(Operation *operation, StringRef name) {
  auto attribute = operation->getAttrOfType<IntegerAttr>(name);
  return attribute ? attribute.getInt() : -1;
}

/// The packed weights and scales of a MatMulNBits node whose output columns
/// can be stacked: static row-major constants with N outermost, no zero
/// points, group index or bias, and a static result whose last dimension is N.
struct Candidate {
  Operation *operation;
  DenseElementsAttr weights;
  DenseElementsAttr scales;
  int64_t columns;
};

std::optional<Candidate> candidate(Operation *operation) {
  if (!isMatMulNBits(operation))
    return std::nullopt;
  for (Value optional : operation->getOperands().drop_front(3))
    if (!isa<NoneType>(optional.getType()))
      return std::nullopt;
  int64_t columns = integerAttribute(operation, "N");
  int64_t depth = integerAttribute(operation, "K");
  auto activation =
      dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  DenseElementsAttr weights = constantElements(operation->getOperand(1));
  DenseElementsAttr scales = constantElements(operation->getOperand(2));
  if (columns < 1 || depth < 1 || !activation || !result ||
      !activation.hasStaticShape() || !result.hasStaticShape() ||
      result.getRank() < 1 || result.getShape().back() != columns ||
      !result.getElementType().isF32() || !weights || !scales)
    return std::nullopt;
  auto weightType = cast<ShapedType>(weights.getType());
  auto scaleType = cast<ShapedType>(scales.getType());
  if (weightType.getRank() < 1 || weightType.getDimSize(0) != columns ||
      scaleType.getRank() < 1 || scaleType.getNumElements() % columns != 0 ||
      (scaleType.getRank() > 1 && scaleType.getDimSize(0) != columns) ||
      !weightType.getElementType().isIntOrFloat() ||
      !scaleType.getElementType().isIntOrFloat() ||
      weightType.getElementType().getIntOrFloatBitWidth() % 8 != 0 ||
      scaleType.getElementType().getIntOrFloatBitWidth() % 8 != 0)
    return std::nullopt;
  return Candidate{operation, weights, scales, columns};
}

/// Every attribute but N and the node name agrees, and so do the activation,
/// the element types and the shapes of a weight row, a scale row and the
/// result apart from N.
bool compatible(const Candidate &first, const Candidate &other) {
  Operation *left = first.operation;
  Operation *right = other.operation;
  // Absent optional operands may be omitted or passed as none.
  if (left->getOperand(0) != right->getOperand(0) ||
      left->getBlock() != right->getBlock())
    return false;
  auto withoutColumns = [](Operation *operation) {
    NamedAttrList attributes(operation->getAttrDictionary());
    attributes.erase("N");
    attributes.erase("onnx_node_name");
    return attributes.getDictionary(operation->getContext());
  };
  if (withoutColumns(left) != withoutColumns(right) ||
      left->getPropertiesAsAttribute() != right->getPropertiesAsAttribute())
    return false;
  auto rowShape = [](ShapedType type, int64_t columns) {
    return std::make_pair(
        type.getElementType(),
        SmallVector<int64_t>(
            type.getRank() > 1
                ? type.getShape().drop_front()
                : ArrayRef<int64_t>{type.getNumElements() / columns}));
  };
  auto leftResult = cast<RankedTensorType>(left->getResult(0).getType());
  auto rightResult = cast<RankedTensorType>(right->getResult(0).getType());
  return rowShape(cast<ShapedType>(first.weights.getType()), first.columns) ==
             rowShape(cast<ShapedType>(other.weights.getType()),
                      other.columns) &&
         rowShape(cast<ShapedType>(first.scales.getType()), first.columns) ==
             rowShape(cast<ShapedType>(other.scales.getType()),
                      other.columns) &&
         cast<ShapedType>(first.scales.getType()).getRank() ==
             cast<ShapedType>(other.scales.getType()).getRank() &&
         leftResult.getShape().drop_back() ==
             rightResult.getShape().drop_back();
}

StringRef initializerName(Location location) {
  if (auto named = dyn_cast<NameLoc>(location)) {
    StringRef name = named.getName().strref();
    if (name.starts_with(initializerPrefix))
      return name.drop_front(initializerPrefix.size());
    return initializerName(named.getChildLoc());
  }
  if (auto fused = dyn_cast<FusedLoc>(location))
    for (Location nested : fused.getLocations())
      if (StringRef name = initializerName(nested); !name.empty())
        return name;
  return {};
}

/// One constant holding the rows of `parts` in order. When every part is an
/// imported initializer the result is one too, named after them, so that it
/// is stored outside the IR like the weights it replaces.
Value concatenateRows(PatternRewriter &rewriter, ArrayRef<Value> sources,
                      ArrayRef<DenseElementsAttr> parts, int64_t rows) {
  auto type = cast<ShapedType>(parts.front().getType());
  SmallVector<int64_t> shape(type.getShape());
  shape[0] = type.getRank() > 1 ? rows : 0;
  if (type.getRank() == 1)
    for (DenseElementsAttr part : parts)
      shape[0] += part.getNumElements();
  auto resultType = RankedTensorType::get(shape, type.getElementType());
  SmallVector<char> bytes;
  for (DenseElementsAttr part : parts) {
    ArrayRef<char> raw = part.getRawData();
    // A splat stores one element.
    for (int64_t copy = 0, copies = part.isSplat() ? part.getNumElements() : 1;
         copy < copies; ++copy)
      bytes.append(raw.begin(), raw.end());
  }
  DenseElementsAttr value =
      DenseElementsAttr::getFromRawBuffer(resultType, bytes);

  SmallVector<Location> locations;
  std::string name;
  bool initializers = true;
  for (Value source : sources) {
    locations.push_back(source.getLoc());
    StringRef part = initializerName(source.getLoc());
    initializers &= !part.empty();
    if (!name.empty())
      name += "+";
    name += part;
  }
  Location location = rewriter.getFusedLoc(locations);
  if (initializers)
    location = NameLoc::get(rewriter.getStringAttr(initializerPrefix + name),
                            location);
  OperationState state(location, "onnx.Constant");
  state.addAttribute("value", value);
  state.addTypes(resultType);
  return rewriter.create(state)->getResult(0);
}

/// Shares one tensor<1xi64> Slice operand per function and value. The greedy
/// driver may erase a constant that became dead, so erased operations leave
/// the cache.
struct SliceOperandCache : RewriterBase::Listener {
  llvm::DenseMap<std::pair<Operation *, int64_t>, Operation *> constants;

  void notifyOperationErased(Operation *operation) override {
    for (auto iterator = constants.begin(); iterator != constants.end();
         ++iterator)
      if (iterator->second == operation) {
        constants.erase(iterator);
        return;
      }
  }

  Value get(PatternRewriter &rewriter, Operation *user, int64_t value) {
    Operation *function = user->getParentOp();
    while (function && !function->hasTrait<OpTrait::IsIsolatedFromAbove>())
      function = function->getParentOp();
    Operation *&cached = constants[{function, value}];
    if (!cached) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&function->getRegion(0).front());
      auto type = RankedTensorType::get({1}, rewriter.getI64Type());
      cached = arith::ConstantOp::create(
                   rewriter, user->getLoc(),
                   DenseIntElementsAttr::get(type, ArrayRef{value}))
                   .getOperation();
    }
    return cached->getResult(0);
  }
};

struct MergeSiblingMatMulNBits : RewritePattern {
  MergeSiblingMatMulNBits(MLIRContext *context, SliceOperandCache &operands,
                          int64_t maxResultWords)
      : RewritePattern("onnx.Custom", /*benefit=*/1, context),
        operands(operands), maxResultWords(maxResultWords) {}

  LogicalResult matchAndRewrite(Operation *operation,
                                PatternRewriter &rewriter) const override {
    std::optional<Candidate> first = candidate(operation);
    if (!first)
      return rewriter.notifyMatchFailure(operation, "not a stackable node");
    llvm::SetVector<Operation *> users;
    for (Operation *user : operation->getOperand(0).getUsers())
      if (user->getBlock() == operation->getBlock())
        users.insert(user);
    SmallVector<Operation *> ordered(users.begin(), users.end());
    llvm::sort(ordered, [](Operation *left, Operation *right) {
      return left->isBeforeInBlock(right);
    });
    // Only the first stackable reader of the activation starts a group, so a
    // group is formed once, in program order.
    auto resultType = cast<RankedTensorType>(operation->getResult(0).getType());
    int64_t rows = resultType.getNumElements() / first->columns;
    SmallVector<Candidate> group;
    int64_t columns = 0;
    for (Operation *user : ordered) {
      std::optional<Candidate> other = candidate(user);
      if (!other || !compatible(*first, *other))
        continue;
      if (group.empty() && user != operation)
        return rewriter.notifyMatchFailure(operation, "not the first sibling");
      if (rows * (columns + other->columns) > maxResultWords)
        continue;
      columns += other->columns;
      group.push_back(*other);
    }
    if (group.size() < 2)
      return rewriter.notifyMatchFailure(operation, "no sibling to merge");

    rewriter.setInsertionPoint(operation);
    SmallVector<Value> weightSources, scaleSources;
    SmallVector<DenseElementsAttr> weights, scales;
    SmallVector<Location> locations;
    std::string nodeName;
    for (const Candidate &member : group) {
      weightSources.push_back(member.operation->getOperand(1));
      scaleSources.push_back(member.operation->getOperand(2));
      weights.push_back(member.weights);
      scales.push_back(member.scales);
      locations.push_back(member.operation->getLoc());
      if (auto name =
              member.operation->getAttrOfType<StringAttr>("onnx_node_name")) {
        if (!nodeName.empty())
          nodeName += "+";
        nodeName += name.getValue();
      }
    }
    Value packed = concatenateRows(rewriter, weightSources, weights, columns);
    Value scale = concatenateRows(rewriter, scaleSources, scales, columns);
    SmallVector<int64_t> shape(resultType.getShape());
    shape.back() = columns;
    Operation *merged = rewriter.clone(*operation);
    merged->setLoc(rewriter.getFusedLoc(locations));
    merged->setOperand(1, packed);
    merged->setOperand(2, scale);
    merged->getResult(0).setType(
        RankedTensorType::get(shape, resultType.getElementType()));
    merged->setAttr(
        "N", rewriter.getIntegerAttr(
                 rewriter.getIntegerType(64, /*isSigned=*/true), columns));
    if (!nodeName.empty())
      merged->setAttr("onnx_node_name", rewriter.getStringAttr(nodeName));

    int64_t offset = 0;
    Value axes = operands.get(rewriter, merged, resultType.getRank() - 1);
    Value steps = operands.get(rewriter, merged, 1);
    for (const Candidate &member : group) {
      Operation *original = member.operation;
      OperationState state(original->getLoc(), "onnx.Slice");
      state.addOperands(
          {merged->getResult(0), operands.get(rewriter, merged, offset),
           operands.get(rewriter, merged, offset + member.columns), axes,
           steps});
      state.addTypes(original->getResult(0).getType());
      if (Attribute name = original->getAttr("onnx_node_name"))
        state.addAttribute("onnx_node_name", name);
      Operation *slice = rewriter.create(state);
      rewriter.replaceOp(original, slice->getResults());
      offset += member.columns;
    }
    // The separate weights are dead now; erase them so they are not stored.
    for (Value source : llvm::concat<Value>(weightSources, scaleSources))
      if (Operation *producer = source.getDefiningOp();
          producer && producer->use_empty())
        rewriter.eraseOp(producer);
    return success();
  }

  SliceOperandCache &operands;
  int64_t maxResultWords;
};

struct MergeONNXSiblingMatMulNBitsPass
    : impl::MergeONNXSiblingMatMulNBitsPassBase<
          MergeONNXSiblingMatMulNBitsPass> {
  using Base::Base;

  void runOnOperation() override {
    SliceOperandCache operands;
    RewritePatternSet patterns(&getContext());
    patterns.add<MergeSiblingMatMulNBits>(&getContext(), operands,
                                          maxResultWords);
    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    config.setListener(&operands);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
