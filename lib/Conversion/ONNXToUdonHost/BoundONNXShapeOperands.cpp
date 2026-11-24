#include "VRC/Conversion/Passes.h"

#include "ONNXDimensionExprs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_BOUNDONNXSHAPEOPERANDSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
constexpr int64_t maximumEvaluatedElements = 1 << 20;
constexpr int64_t maximumCheckedElements = int64_t(1) << 28;
constexpr StringLiteral boundedPrefixFunction = "UdonclBoundedPrefix";
constexpr StringLiteral udonclDomain = "com.lemolatoon.udoncl";

bool isNoValue(Value value) {
  Operation *definition = value.getDefiningOp();
  return definition && definition->getName().getStringRef() == "onnx.NoValue";
}

/// A tensor value evaluated on the Host: integers (and booleans) as int64,
/// floating point as float.
struct Tensor {
  SmallVector<int64_t> shape;
  bool isFloat = false;
  std::vector<int64_t> ints;
  std::vector<float> floats;

  int64_t size() const {
    return std::accumulate(shape.begin(), shape.end(), int64_t(1),
                           std::multiplies<int64_t>());
  }
  double at(int64_t index) const {
    return isFloat ? static_cast<double>(floats[index])
                   : static_cast<double>(ints[index]);
  }
  bool sameBits(int64_t index, const Tensor &other, int64_t otherIndex) const {
    if (isFloat != other.isFloat)
      return false;
    if (!isFloat)
      return ints[index] == other.ints[otherIndex];
    uint32_t left, right;
    std::memcpy(&left, &floats[index], 4);
    std::memcpy(&right, &other.floats[otherIndex], 4);
    return left == right;
  }
};

std::optional<Tensor> readConstant(Operation *op) {
  auto type = op->getNumResults() == 1
                  ? dyn_cast<RankedTensorType>(op->getResult(0).getType())
                  : RankedTensorType();
  if (!type || !type.hasStaticShape() ||
      type.getNumElements() > maximumEvaluatedElements)
    return std::nullopt;
  DenseElementsAttr dense = op->getAttrOfType<DenseElementsAttr>("value");
  if (!dense)
    if (auto disposable = op->getAttrOfType<DisposableElementsAttr>("value"))
      dense = disposable.toDenseElementsAttr();
  if (!dense)
    return std::nullopt;
  Tensor tensor;
  tensor.shape.assign(type.getShape().begin(), type.getShape().end());
  int64_t count = type.getNumElements();
  if (type.getElementType().isF32()) {
    tensor.isFloat = true;
    if (dense.isSplat())
      tensor.floats.assign(count,
                           dense.getSplatValue<APFloat>().convertToFloat());
    else
      for (APFloat value : dense.getValues<APFloat>())
        tensor.floats.push_back(value.convertToFloat());
    return tensor;
  }
  if (!isa<IntegerType>(type.getElementType()))
    return std::nullopt;
  bool boolean = type.getElementType().isInteger(1);
  auto read = [&](const APInt &value) {
    return boolean ? static_cast<int64_t>(value.getBoolValue())
                   : value.getSExtValue();
  };
  if (dense.isSplat())
    tensor.ints.assign(count, read(dense.getSplatValue<APInt>()));
  else
    for (APInt value : dense.getValues<APInt>())
      tensor.ints.push_back(read(value));
  return tensor;
}

SmallVector<int64_t> stridesOf(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t axis = static_cast<int64_t>(shape.size()) - 2; axis >= 0; --axis)
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  return strides;
}

/// Maps a flat index of a broadcast result to the flat index of an operand.
int64_t broadcastIndex(int64_t flat, ArrayRef<int64_t> result,
                       ArrayRef<int64_t> operand) {
  int64_t offset = 0, stride = 1;
  int64_t rankOffset = result.size() - operand.size();
  for (int64_t axis = result.size(); axis-- > 0;) {
    int64_t coordinate = flat % result[axis];
    flat /= result[axis];
    int64_t operandAxis = axis - rankOffset;
    if (operandAxis < 0)
      continue;
    if (operand[operandAxis] != 1)
      offset += coordinate * stride;
    stride *= operand[operandAxis];
  }
  return offset;
}

std::optional<SmallVector<int64_t>> broadcastShapes(ArrayRef<Tensor *> inputs) {
  SmallVector<int64_t> shape;
  for (Tensor *input : inputs) {
    size_t rank = std::max(shape.size(), input->shape.size());
    SmallVector<int64_t> merged(rank, 1);
    for (size_t offset = 0; offset < rank; ++offset) {
      int64_t left =
          offset < shape.size() ? shape[shape.size() - 1 - offset] : 1;
      int64_t right = offset < input->shape.size()
                          ? input->shape[input->shape.size() - 1 - offset]
                          : 1;
      if (left != 1 && right != 1 && left != right)
        return std::nullopt;
      merged[rank - 1 - offset] = left == 1 ? right : left;
    }
    shape = std::move(merged);
  }
  return shape;
}

/// Evaluates the data-independent part of an ONNX graph on the Host for one
/// assignment of the bounded dimensions.
class Evaluator {
public:
  Evaluator(const ONNXDimensionExprs &dimensions,
            const llvm::StringMap<int64_t> &assignment)
      : dimensions(dimensions), assignment(assignment) {}

  std::optional<Tensor> evaluate(Value value) {
    auto found = cache.find(value);
    if (found != cache.end())
      return found->second;
    std::optional<Tensor> result;
    if (Operation *op = value.getDefiningOp())
      result = evaluateOperation(op);
    if (result && result->size() > maximumEvaluatedElements)
      result.reset();
    cache[value] = result;
    return result;
  }

  std::optional<int64_t> extent(const DimExpr &expression) const {
    if (!expression.known)
      return std::nullopt;
    if (!expression.symbol)
      return expression.offset;
    auto found = assignment.find(expression.symbol.getValue());
    if (found == assignment.end())
      return std::nullopt;
    return found->second + expression.offset;
  }

private:
  std::optional<SmallVector<int64_t>> shapeOf(Value value) const {
    const DimShape *shape = dimensions.getShape(value);
    if (!shape)
      return std::nullopt;
    SmallVector<int64_t> result;
    for (const DimExpr &expression : *shape) {
      std::optional<int64_t> value = extent(expression);
      if (!value || *value < 0)
        return std::nullopt;
      result.push_back(*value);
    }
    return result;
  }

  std::optional<Tensor> evaluateOperation(Operation *op);

  const ONNXDimensionExprs &dimensions;
  const llvm::StringMap<int64_t> &assignment;
  llvm::DenseMap<Value, std::optional<Tensor>> cache;
};

std::optional<Tensor> Evaluator::evaluateOperation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name == "onnx.Constant" || name == "arith.constant")
    return readConstant(op);
  if (op->getNumResults() != 1 || !name.starts_with("onnx."))
    return std::nullopt;
  StringRef operation = name.drop_front(5);
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return std::nullopt;
  Type element = resultType.getElementType();
  bool floatResult = element.isF32();
  if (!floatResult && !isa<IntegerType>(element))
    return std::nullopt;
  if (operation == "Shape" || operation == "Dim") {
    std::optional<SmallVector<int64_t>> shape = shapeOf(op->getOperand(0));
    if (!shape)
      return std::nullopt;
    int64_t rank = shape->size();
    Tensor result;
    if (operation == "Dim") {
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      int64_t index = axis ? axis.getInt() : 0;
      if (index < 0)
        index += rank;
      if (index < 0 || index >= rank)
        return std::nullopt;
      result.shape = {1};
      result.ints = {(*shape)[index]};
      return result;
    }
    int64_t start = 0, end = rank;
    if (auto attribute = op->getAttrOfType<IntegerAttr>("start"))
      start = attribute.getInt();
    if (auto attribute = op->getAttrOfType<IntegerAttr>("end"))
      end = attribute.getInt();
    start = std::clamp<int64_t>(start < 0 ? start + rank : start, 0, rank);
    end = std::clamp<int64_t>(end < 0 ? end + rank : end, start, rank);
    result.shape = {end - start};
    result.ints.assign(shape->begin() + start, shape->begin() + end);
    return result;
  }

  SmallVector<std::optional<Tensor>> operands;
  for (Value operand : op->getOperands()) {
    if (isNoValue(operand)) {
      operands.push_back(std::nullopt);
      continue;
    }
    std::optional<Tensor> value = evaluate(operand);
    if (!value)
      return std::nullopt;
    operands.push_back(std::move(value));
  }
  auto integersOf = [&](unsigned index) -> std::optional<std::vector<int64_t>> {
    if (index >= operands.size() || !operands[index] ||
        operands[index]->isFloat)
      return std::nullopt;
    return operands[index]->ints;
  };
  auto scalar = [&](unsigned index) -> std::optional<int64_t> {
    auto values = integersOf(index);
    if (!values || values->size() != 1)
      return std::nullopt;
    return (*values)[0];
  };
  auto store = [&](Tensor &tensor, double value) {
    if (tensor.isFloat)
      tensor.floats.push_back(static_cast<float>(value));
    else
      tensor.ints.push_back(static_cast<int64_t>(value));
  };

  // Element-wise operators with broadcasting.
  bool binary = llvm::StringSwitch<bool>(operation)
                    .Cases({"Add", "Sub", "Mul", "Div", "Max", "Min"}, true)
                    .Cases({"Equal", "Greater", "Less", "GreaterOrEqual"}, true)
                    .Cases({"LessOrEqual", "And", "Or"}, true)
                    .Default(false);
  if (binary || operation == "Where") {
    SmallVector<Tensor *> inputs;
    for (auto &operand : operands) {
      if (!operand)
        return std::nullopt;
      inputs.push_back(&*operand);
    }
    if ((binary && inputs.size() != 2) || (!binary && inputs.size() != 3))
      return std::nullopt;
    std::optional<SmallVector<int64_t>> shape = broadcastShapes(inputs);
    if (!shape)
      return std::nullopt;
    Tensor result;
    result.shape = *shape;
    result.isFloat = floatResult;
    int64_t count = result.size();
    if (count > maximumEvaluatedElements)
      return std::nullopt;
    for (int64_t flat = 0; flat < count; ++flat) {
      if (!binary) {
        Tensor &condition = *inputs[0];
        bool selected =
            condition.at(broadcastIndex(flat, *shape, condition.shape)) != 0;
        Tensor &chosen = selected ? *inputs[1] : *inputs[2];
        int64_t index = broadcastIndex(flat, *shape, chosen.shape);
        if (chosen.isFloat != result.isFloat)
          return std::nullopt;
        if (result.isFloat)
          result.floats.push_back(chosen.floats[index]);
        else
          result.ints.push_back(chosen.ints[index]);
        continue;
      }
      Tensor &left = *inputs[0];
      Tensor &right = *inputs[1];
      int64_t li = broadcastIndex(flat, *shape, left.shape);
      int64_t ri = broadcastIndex(flat, *shape, right.shape);
      if (left.isFloat != right.isFloat)
        return std::nullopt;
      if (left.isFloat) {
        float a = left.floats[li], b = right.floats[ri];
        if (operation == "Add")
          store(result, a + b);
        else if (operation == "Sub")
          store(result, a - b);
        else if (operation == "Mul")
          result.floats.push_back(a * b);
        else if (operation == "Div")
          result.floats.push_back(a / b);
        else if (operation == "Max")
          result.floats.push_back(std::max(a, b));
        else if (operation == "Min")
          result.floats.push_back(std::min(a, b));
        else
          store(result, llvm::StringSwitch<bool>(operation)
                            .Case("Equal", a == b)
                            .Case("Greater", a > b)
                            .Case("Less", a < b)
                            .Case("GreaterOrEqual", a >= b)
                            .Case("LessOrEqual", a <= b)
                            .Default(false));
        continue;
      }
      int64_t a = left.ints[li], b = right.ints[ri];
      int64_t value = 0;
      if (operation == "Add")
        value = a + b;
      else if (operation == "Sub")
        value = a - b;
      else if (operation == "Mul")
        value = a * b;
      else if (operation == "Div") {
        if (!b)
          return std::nullopt;
        value = a / b;
      } else if (operation == "Max")
        value = std::max(a, b);
      else if (operation == "Min")
        value = std::min(a, b);
      else
        value = llvm::StringSwitch<bool>(operation)
                    .Case("Equal", a == b)
                    .Case("Greater", a > b)
                    .Case("Less", a < b)
                    .Case("GreaterOrEqual", a >= b)
                    .Case("LessOrEqual", a <= b)
                    .Case("And", a && b)
                    .Case("Or", a || b)
                    .Default(false);
      if (result.isFloat)
        result.floats.push_back(static_cast<float>(value));
      else
        result.ints.push_back(value);
    }
    return result;
  }
  if (llvm::StringSwitch<bool>(operation)
          .Cases({"Neg", "Not", "Abs", "Cast", "Identity"}, true)
          .Default(false)) {
    if (operands.empty() || !operands[0])
      return std::nullopt;
    Tensor &input = *operands[0];
    Tensor result;
    result.shape = input.shape;
    result.isFloat = floatResult;
    bool boolean = element.isInteger(1);
    for (int64_t index = 0, count = input.size(); index < count; ++index) {
      if (input.isFloat && result.isFloat) {
        float value = input.floats[index];
        if (operation == "Neg")
          value = -value;
        else if (operation == "Abs")
          value = std::fabs(value);
        else if (operation == "Not")
          return std::nullopt;
        result.floats.push_back(value);
        continue;
      }
      if (input.isFloat) {
        // Cast of floating point to an integer truncates toward zero.
        float value = input.floats[index];
        if (operation != "Cast" || std::isnan(value))
          return std::nullopt;
        result.ints.push_back(boolean ? value != 0.0f
                                      : static_cast<int64_t>(value));
        continue;
      }
      int64_t value = input.ints[index];
      if (operation == "Neg")
        value = -value;
      else if (operation == "Abs")
        value = value < 0 ? -value : value;
      else if (operation == "Not")
        value = value == 0;
      if (boolean)
        value = value != 0;
      if (result.isFloat)
        result.floats.push_back(static_cast<float>(value));
      else
        result.ints.push_back(value);
    }
    return result;
  }
  // Views keep the elements and change the shape to the result's extents.
  if (llvm::StringSwitch<bool>(operation)
          .Cases({"Reshape", "Unsqueeze", "Squeeze", "Flatten"}, true)
          .Default(false)) {
    if (operands.empty() || !operands[0])
      return std::nullopt;
    std::optional<SmallVector<int64_t>> shape = shapeOf(op->getResult(0));
    if (!shape)
      return std::nullopt;
    Tensor result = *operands[0];
    result.shape = *shape;
    if (result.size() != operands[0]->size())
      return std::nullopt;
    return result;
  }
  if (operation == "Range") {
    std::optional<int64_t> start = scalar(0), limit = scalar(1),
                           delta = scalar(2);
    if (!start || !limit || !delta || !*delta || floatResult)
      return std::nullopt;
    Tensor result;
    for (int64_t value = *start; *delta > 0 ? value < *limit : value > *limit;
         value += *delta) {
      if (static_cast<int64_t>(result.ints.size()) >= maximumEvaluatedElements)
        return std::nullopt;
      result.ints.push_back(value);
    }
    result.shape = {static_cast<int64_t>(result.ints.size())};
    return result;
  }
  if (operation == "ConstantOfShape") {
    auto shape = integersOf(0);
    if (!shape)
      return std::nullopt;
    Tensor result;
    result.shape.assign(shape->begin(), shape->end());
    if (llvm::any_of(result.shape, [](int64_t extent) { return extent < 0; }) ||
        result.size() > maximumEvaluatedElements)
      return std::nullopt;
    result.isFloat = floatResult;
    // ONNX defaults the fill to a float 0.
    double fill = 0;
    if (Attribute attribute = op->getAttr("value")) {
      DenseElementsAttr value = dyn_cast<DenseElementsAttr>(attribute);
      if (!value)
        if (auto disposable = dyn_cast<DisposableElementsAttr>(attribute))
          value = disposable.toDenseElementsAttr();
      if (!value || value.getNumElements() != 1)
        return std::nullopt;
      if (value.getElementType().isF32())
        fill = (*value.getValues<APFloat>().begin()).convertToFloat();
      else if (isa<IntegerType>(value.getElementType()))
        fill = (*value.getValues<APInt>().begin()).getSExtValue();
      else
        return std::nullopt;
    }
    for (int64_t index = 0, count = result.size(); index < count; ++index)
      store(result, fill);
    return result;
  }
  if (operation == "Trilu") {
    if (operands.empty() || !operands[0] || operands[0]->shape.size() < 2)
      return std::nullopt;
    int64_t k = 0;
    if (operands.size() > 1 && operands[1]) {
      std::optional<int64_t> value = scalar(1);
      if (!value)
        return std::nullopt;
      k = *value;
    }
    auto upperAttr = op->getAttrOfType<IntegerAttr>("upper");
    bool upper = !upperAttr || upperAttr.getInt() != 0;
    Tensor result = *operands[0];
    int64_t rows = result.shape[result.shape.size() - 2];
    int64_t columns = result.shape.back();
    for (int64_t flat = 0, count = result.size(); flat < count; ++flat) {
      int64_t column = flat % columns;
      int64_t row = (flat / columns) % rows;
      bool keep = upper ? column - row >= k : column - row <= k;
      if (keep)
        continue;
      if (result.isFloat)
        result.floats[flat] = 0.0f;
      else
        result.ints[flat] = 0;
    }
    return result;
  }
  if (operation == "Expand") {
    auto shape = integersOf(1);
    if (operands.empty() || !operands[0] || !shape)
      return std::nullopt;
    Tensor target;
    target.shape.assign(shape->begin(), shape->end());
    SmallVector<Tensor *> inputs{&*operands[0], &target};
    std::optional<SmallVector<int64_t>> resultShape = broadcastShapes(inputs);
    if (!resultShape)
      return std::nullopt;
    Tensor result;
    result.shape = *resultShape;
    result.isFloat = operands[0]->isFloat;
    if (result.size() > maximumEvaluatedElements)
      return std::nullopt;
    for (int64_t flat = 0, count = result.size(); flat < count; ++flat) {
      int64_t index = broadcastIndex(flat, *resultShape, operands[0]->shape);
      if (result.isFloat)
        result.floats.push_back(operands[0]->floats[index]);
      else
        result.ints.push_back(operands[0]->ints[index]);
    }
    return result;
  }
  if (operation == "Gather") {
    if (operands.size() != 2 || !operands[0] || !integersOf(1))
      return std::nullopt;
    Tensor &data = *operands[0];
    Tensor &indices = *operands[1];
    int64_t rank = data.shape.size();
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    int64_t axis = axisAttr ? axisAttr.getInt() : 0;
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return std::nullopt;
    int64_t outer = 1, inner = 1;
    for (int64_t index = 0; index < axis; ++index)
      outer *= data.shape[index];
    for (int64_t index = axis + 1; index < rank; ++index)
      inner *= data.shape[index];
    Tensor result;
    result.isFloat = data.isFloat;
    result.shape.assign(data.shape.begin(), data.shape.begin() + axis);
    llvm::append_range(result.shape, indices.shape);
    result.shape.append(data.shape.begin() + axis + 1, data.shape.end());
    for (int64_t o = 0; o < outer; ++o)
      for (int64_t position : indices.ints) {
        if (position < 0)
          position += data.shape[axis];
        if (position < 0 || position >= data.shape[axis])
          return std::nullopt;
        for (int64_t i = 0; i < inner; ++i) {
          int64_t source = (o * data.shape[axis] + position) * inner + i;
          if (result.isFloat)
            result.floats.push_back(data.floats[source]);
          else
            result.ints.push_back(data.ints[source]);
        }
      }
    return result;
  }
  if (operation == "Concat") {
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if (operands.empty() || !operands[0] || !axisAttr)
      return std::nullopt;
    int64_t rank = operands[0]->shape.size();
    int64_t axis = axisAttr.getInt();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return std::nullopt;
    Tensor result;
    result.isFloat = operands[0]->isFloat;
    result.shape = operands[0]->shape;
    result.shape[axis] = 0;
    int64_t outer = 1;
    for (int64_t index = 0; index < axis; ++index)
      outer *= result.shape[index];
    for (auto &operand : operands) {
      if (!operand || operand->shape.size() != static_cast<size_t>(rank) ||
          operand->isFloat != result.isFloat)
        return std::nullopt;
      result.shape[axis] += operand->shape[axis];
    }
    for (int64_t o = 0; o < outer; ++o)
      for (auto &operand : operands) {
        int64_t chunk = outer ? operand->size() / outer : 0;
        for (int64_t i = 0; i < chunk; ++i) {
          if (result.isFloat)
            result.floats.push_back(operand->floats[o * chunk + i]);
          else
            result.ints.push_back(operand->ints[o * chunk + i]);
        }
      }
    return result;
  }
  if (operation == "Slice") {
    if (operands.size() < 3 || !operands[0] || !integersOf(1) || !integersOf(2))
      return std::nullopt;
    Tensor &data = *operands[0];
    int64_t rank = data.shape.size();
    std::vector<int64_t> starts = *integersOf(1), ends = *integersOf(2);
    std::vector<int64_t> axes, steps(starts.size(), 1);
    if (operands.size() > 3 && operands[3])
      axes = *integersOf(3);
    else
      for (size_t index = 0; index < starts.size(); ++index)
        axes.push_back(index);
    if (operands.size() > 4 && operands[4])
      steps = *integersOf(4);
    if (axes.size() != starts.size() || ends.size() != starts.size() ||
        steps.size() != starts.size())
      return std::nullopt;
    SmallVector<int64_t> first(rank, 0), step(rank, 1), extent(data.shape);
    for (size_t index = 0; index < starts.size(); ++index) {
      int64_t axis = axes[index] < 0 ? axes[index] + rank : axes[index];
      if (axis < 0 || axis >= rank || steps[index] <= 0)
        return std::nullopt;
      int64_t size = data.shape[axis];
      auto clamp = [&](int64_t bound) {
        if (bound < 0)
          bound = bound < -size ? 0 : bound + size;
        return std::min(bound, size);
      };
      int64_t begin = clamp(starts[index]), end = clamp(ends[index]);
      first[axis] = begin;
      step[axis] = steps[index];
      extent[axis] =
          end > begin ? (end - begin + steps[index] - 1) / steps[index] : 0;
    }
    Tensor result;
    result.isFloat = data.isFloat;
    result.shape = extent;
    SmallVector<int64_t> strides = stridesOf(data.shape);
    for (int64_t flat = 0, count = result.size(); flat < count; ++flat) {
      int64_t remaining = flat, source = 0;
      for (int64_t axis = rank; axis-- > 0;) {
        int64_t coordinate = remaining % extent[axis];
        remaining /= extent[axis];
        source += (first[axis] + coordinate * step[axis]) * strides[axis];
      }
      if (result.isFloat)
        result.floats.push_back(data.floats[source]);
      else
        result.ints.push_back(data.ints[source]);
    }
    return result;
  }
  if (operation == "Transpose") {
    if (operands.empty() || !operands[0])
      return std::nullopt;
    Tensor &data = *operands[0];
    int64_t rank = data.shape.size();
    SmallVector<int64_t> permutation;
    if (auto attribute = op->getAttrOfType<ArrayAttr>("perm"))
      for (Attribute value : attribute)
        permutation.push_back(cast<IntegerAttr>(value).getInt());
    else
      for (int64_t axis = rank; axis-- > 0;)
        permutation.push_back(axis);
    if (static_cast<int64_t>(permutation.size()) != rank)
      return std::nullopt;
    Tensor result;
    result.isFloat = data.isFloat;
    for (int64_t axis : permutation)
      result.shape.push_back(data.shape[axis]);
    SmallVector<int64_t> strides = stridesOf(data.shape);
    for (int64_t flat = 0, count = result.size(); flat < count; ++flat) {
      int64_t remaining = flat, source = 0;
      for (int64_t axis = rank; axis-- > 0;) {
        int64_t coordinate = remaining % result.shape[axis];
        remaining /= result.shape[axis];
        source += coordinate * strides[permutation[axis]];
      }
      if (result.isFloat)
        result.floats.push_back(data.floats[source]);
      else
        result.ints.push_back(data.ints[source]);
    }
    return result;
  }
  if (operation == "ScatterND") {
    auto reduction = op->getAttrOfType<StringAttr>("reduction");
    if ((reduction && reduction.getValue() != "none") || operands.size() != 3 ||
        !operands[0] || !integersOf(1) || !operands[2])
      return std::nullopt;
    Tensor result = *operands[0];
    Tensor &indices = *operands[1];
    Tensor &updates = *operands[2];
    if (indices.shape.empty() || updates.isFloat != result.isFloat)
      return std::nullopt;
    int64_t depth = indices.shape.back();
    int64_t rank = result.shape.size();
    if (depth > rank)
      return std::nullopt;
    SmallVector<int64_t> strides = stridesOf(result.shape);
    int64_t slice = 1;
    for (int64_t axis = depth; axis < rank; ++axis)
      slice *= result.shape[axis];
    int64_t tuples = indices.size() / std::max<int64_t>(depth, 1);
    if (tuples * slice != updates.size())
      return std::nullopt;
    for (int64_t tuple = 0; tuple < tuples; ++tuple) {
      int64_t offset = 0;
      for (int64_t axis = 0; axis < depth; ++axis) {
        int64_t index = indices.ints[tuple * depth + axis];
        if (index < 0)
          index += result.shape[axis];
        if (index < 0 || index >= result.shape[axis])
          return std::nullopt;
        offset += index * strides[axis];
      }
      for (int64_t i = 0; i < slice; ++i) {
        if (result.isFloat)
          result.floats[offset + i] = updates.floats[tuple * slice + i];
        else
          result.ints[offset + i] = updates.ints[tuple * slice + i];
      }
    }
    return result;
  }
  return std::nullopt;
}

bool isEvaluableOperation(StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Cases({"onnx.Constant", "arith.constant", "onnx.Shape", "onnx.Dim"},
             true)
      .Cases({"onnx.Add", "onnx.Sub", "onnx.Mul", "onnx.Div", "onnx.Max"}, true)
      .Cases({"onnx.Min", "onnx.Equal", "onnx.Greater", "onnx.Less"}, true)
      .Cases({"onnx.GreaterOrEqual", "onnx.LessOrEqual", "onnx.And"}, true)
      .Cases({"onnx.Or", "onnx.Where", "onnx.Neg", "onnx.Not", "onnx.Abs"},
             true)
      .Cases({"onnx.Cast", "onnx.Identity", "onnx.Reshape", "onnx.Unsqueeze"},
             true)
      .Cases({"onnx.Squeeze", "onnx.Flatten", "onnx.Range"}, true)
      .Cases({"onnx.ConstantOfShape", "onnx.Trilu", "onnx.Expand"}, true)
      .Cases({"onnx.Gather", "onnx.Concat", "onnx.Slice", "onnx.Transpose"},
             true)
      .Case("onnx.ScatterND", true)
      .Default(false);
}

/// The operand positions that hold a shape, axes or bounds rather than data.
bool isShapeOperand(Operation *op, unsigned index) {
  StringRef name = op->getName().getStringRef();
  if (name == "onnx.Reshape" || name == "onnx.Expand" ||
      name == "onnx.Unsqueeze" || name == "onnx.Squeeze" || name == "onnx.Tile")
    return index == 1;
  if (name == "onnx.Slice")
    return index >= 1;
  if (name.starts_with("onnx.Reduce"))
    return index == 1;
  return false;
}

struct BoundONNXShapeOperandsPass
    : impl::BoundONNXShapeOperandsPassBase<BoundONNXShapeOperandsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr(onnxDimensionBoundsAttr))
      return;
    for (auto function : module.getOps<func::FuncOp>())
      if (!function.isDeclaration() && failed(runOnFunction(function)))
        return signalPassFailure();
  }

  ONNXConstantReader reader() {
    return [](Operation *op) -> std::optional<SmallVector<int64_t>> {
      auto disposable = op->getAttrOfType<DisposableElementsAttr>("value");
      if (!disposable || !isa<IntegerType>(disposable.getElementType()) ||
          disposable.getNumElements() > 64)
        return std::nullopt;
      SmallVector<int64_t> values;
      for (APInt integer : disposable.toDenseElementsAttr().getValues<APInt>())
        values.push_back(integer.getSExtValue());
      return values;
    };
  }

  /// The bounded dimensions a data-independent value depends on, or
  /// std::nullopt when it depends on graph data.
  using Dependence = std::optional<llvm::SmallSetVector<StringAttr, 1>>;

  LogicalResult runOnFunction(func::FuncOp function);
  LogicalResult checkPrefix(Value value, StringAttr symbol, int64_t axis,
                            DimExpr extent, const ONNXDimensionExprs &analysis,
                            Tensor &atBound);
  LogicalResult rewriteShapeOperands(Operation *op,
                                     const ONNXDimensionExprs &analysis);
};

LogicalResult BoundONNXShapeOperandsPass::runOnFunction(func::FuncOp function) {
  ONNXDimensionExprs analysis(function, reader());
  for (const auto &[op, message] : analysis.getErrors())
    return op->emitError("ONNX operation ") << message;

  // Which values are computed from constants and tensor shapes alone, and
  // which bounded dimensions they depend on.
  llvm::DenseMap<Value, Dependence> dependence;
  auto dependenceOf = [&](Value value) -> Dependence {
    if (isNoValue(value))
      return llvm::SmallSetVector<StringAttr, 1>();
    auto found = dependence.find(value);
    if (found == dependence.end())
      return std::nullopt;
    return found->second;
  };
  function.walk([&](Operation *op) {
    if (op->getNumResults() != 1 ||
        !isEvaluableOperation(op->getName().getStringRef()))
      return;
    llvm::SmallSetVector<StringAttr, 1> symbols;
    StringRef name = op->getName().getStringRef();
    if (name == "onnx.Shape" || name == "onnx.Dim") {
      const DimShape *shape = analysis.getShape(op->getOperand(0));
      if (!shape || llvm::any_of(*shape, [](const DimExpr &extent) {
            return !extent.known;
          }))
        return;
      for (const DimExpr &extent : *shape)
        if (extent.symbol)
          symbols.insert(extent.symbol);
    } else {
      for (Value operand : op->getOperands()) {
        Dependence operandDependence = dependenceOf(operand);
        if (!operandDependence)
          return;
        symbols.insert(operandDependence->begin(), operandDependence->end());
      }
    }
    if (const DimShape *shape = analysis.getShape(op->getResult(0)))
      for (const DimExpr &extent : *shape)
        if (extent.symbol)
          symbols.insert(extent.symbol);
    dependence[op->getResult(0)] = std::move(symbols);
  });
  auto dependsOnBound = [&](Value value) {
    Dependence symbols = dependenceOf(value);
    return symbols && !symbols->empty();
  };

  // A ScatterND that overwrites every element of data-independent data with
  // run-time updates, in row-major order and for every extent, is its
  // updates.
  SmallVector<Operation *> scatters;
  function.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "onnx.ScatterND" &&
        !dependence.count(op->getResult(0)) &&
        dependsOnBound(op->getOperand(1)) && dependenceOf(op->getOperand(0)) &&
        dependenceOf(op->getOperand(1)))
      scatters.push_back(op);
  });
  for (Operation *scatter : scatters) {
    const DimShape *data = analysis.getShape(scatter->getOperand(0));
    const DimShape *updates = analysis.getShape(scatter->getOperand(2));
    auto reduction = scatter->getAttrOfType<StringAttr>("reduction");
    if (!data || !updates || *data != *updates ||
        (reduction && reduction.getValue() != "none"))
      return scatter->emitError(
          "ONNX ScatterND with indices that depend on a bounded dimension "
          "must overwrite its whole data with updates of the same shape");
    Dependence symbols = dependenceOf(scatter->getOperand(1));
    if (symbols->size() != 1)
      return scatter->emitError("ONNX ScatterND indices depend on more than "
                                "one bounded dimension");
    StringAttr symbol = symbols->front();
    int64_t bound = *analysis.getBound(symbol);
    for (int64_t extent = 0; extent <= bound; ++extent) {
      llvm::StringMap<int64_t> assignment{{symbol.getValue(), extent}};
      Evaluator evaluator(analysis, assignment);
      std::optional<Tensor> indices =
          evaluator.evaluate(scatter->getOperand(1));
      std::optional<SmallVector<int64_t>> shape;
      SmallVector<int64_t> dataShape;
      for (const DimExpr &axis : *data) {
        std::optional<int64_t> value = evaluator.extent(axis);
        if (!value)
          return scatter->emitError("ONNX ScatterND data extent is unknown");
        dataShape.push_back(*value);
      }
      int64_t rank = dataShape.size();
      int64_t count = std::accumulate(dataShape.begin(), dataShape.end(),
                                      int64_t(1), std::multiplies<int64_t>());
      bool covers = indices && !indices->shape.empty() &&
                    indices->shape.back() == rank &&
                    indices->size() == count * rank;
      SmallVector<int64_t> strides = stridesOf(dataShape);
      for (int64_t tuple = 0; covers && tuple < count; ++tuple) {
        int64_t linear = 0;
        for (int64_t axis = 0; axis < rank; ++axis) {
          int64_t index = indices->ints[tuple * rank + axis];
          if (index < 0)
            index += dataShape[axis];
          linear += index * strides[axis];
        }
        covers = linear == tuple;
      }
      if (!covers)
        return scatter->emitError("ONNX ScatterND indices do not name every "
                                  "element in order when '")
               << symbol.getValue() << "' is " << extent;
    }
    scatter->getResult(0).replaceAllUsesWith(scatter->getOperand(2));
    scatter->erase();
  }

  // Data-independent values that depend on a bounded dimension and reach
  // run-time data become bounded prefixes of their value at the bound.
  SmallVector<Value> frontier;
  for (auto &[value, symbols] : dependence) {
    if (!symbols || symbols->empty() || value.use_empty())
      continue;
    bool reachesData = llvm::any_of(value.getUses(), [&](OpOperand &use) {
      Operation *user = use.getOwner();
      if (user->getNumResults() == 1 && dependence.count(user->getResult(0)))
        return false;
      return !isShapeOperand(user, use.getOperandNumber());
    });
    if (reachesData)
      frontier.push_back(value);
  }
  llvm::sort(frontier, [](Value left, Value right) {
    return left.getDefiningOp()->isBeforeInBlock(right.getDefiningOp());
  });
  for (Value value : frontier) {
    Operation *op = value.getDefiningOp();
    const DimShape *shape = analysis.getShape(value);
    auto type = dyn_cast<RankedTensorType>(value.getType());
    Dependence symbols = dependenceOf(value);
    auto axis = shape ? llvm::find_if(*shape,
                                      [](const DimExpr &extent) {
                                        return extent.isSymbolic();
                                      })
                      : nullptr;
    if (!shape || !type || symbols->size() != 1 || axis == shape->end())
      return op->emitError("ONNX operation ")
             << op->getName() << " computes data that depends on bounded "
             << "dimension '" << symbols->front().getValue()
             << "'; only a bounded prefix along one dynamic axis is supported";
    int64_t axisIndex = axis - shape->begin();
    Tensor atBound;
    if (failed(checkPrefix(value, symbols->front(), axisIndex, *axis, analysis,
                           atBound)))
      return failure();
    OpBuilder builder(op);
    SmallVector<int64_t> boundShape(atBound.shape);
    auto constantType =
        RankedTensorType::get(boundShape, type.getElementType());
    DenseElementsAttr elements;
    if (atBound.isFloat) {
      SmallVector<APFloat> values;
      for (float element : atBound.floats)
        values.emplace_back(element);
      elements = DenseElementsAttr::get(constantType, values);
    } else {
      unsigned width = type.getElementTypeBitWidth();
      SmallVector<APInt> values;
      for (int64_t element : atBound.ints)
        values.emplace_back(width, element, /*isSigned=*/width > 1);
      elements = DenseElementsAttr::get(constantType, values);
    }
    Value constant = builder.create<arith::ConstantOp>(op->getLoc(), elements);
    OperationState state(op->getLoc(), "onnx.Custom");
    state.addOperands(constant);
    state.addTypes(type);
    state.addAttribute("function_name",
                       builder.getStringAttr(boundedPrefixFunction));
    state.addAttribute("domain_name", builder.getStringAttr(udonclDomain));
    state.addAttribute("axis", builder.getI64IntegerAttr(axisIndex));
    state.addAttribute("symbol", axis->symbol);
    state.addAttribute("offset", builder.getI64IntegerAttr(axis->offset));
    Operation *prefix = builder.create(state);
    value.replaceUsesWithIf(prefix->getResult(0), [&](OpOperand &use) {
      Operation *user = use.getOwner();
      return !(user->getNumResults() == 1 &&
               dependence.count(user->getResult(0))) &&
             !isShapeOperand(user, use.getOperandNumber());
    });
  }

  // Shape operands of run-time operators become extent-free.
  SmallVector<Operation *> users;
  function.walk([&](Operation *op) {
    if (op->getNumResults() == 1 && dependence.count(op->getResult(0)))
      return;
    for (OpOperand &operand : op->getOpOperands())
      if (isShapeOperand(op, operand.getOperandNumber()) &&
          dependsOnBound(operand.get())) {
        users.push_back(op);
        return;
      }
  });
  for (Operation *op : users)
    if (failed(rewriteShapeOperands(op, analysis)))
      return failure();

  // Erase the shape computations nothing uses any more.
  bool erased = true;
  while (erased) {
    erased = false;
    SmallVector<Operation *> dead;
    function.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if ((name.starts_with("onnx.") || name == "arith.constant") &&
          name != "onnx.Return" && name != "onnx.EntryPoint" &&
          op->getNumResults() > 0 && op->use_empty() &&
          (dependence.count(op->getResult(0)) || name == "onnx.Constant" ||
           name == "arith.constant"))
        dead.push_back(op);
    });
    for (Operation *op : llvm::reverse(dead)) {
      op->erase();
      erased = true;
    }
  }

  // Postcondition: nothing that is still used depends on a bounded
  // dimension through a shape computation.
  WalkResult result = function.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (op->getNumResults() != 1 || name == "onnx.Constant" ||
        name == "arith.constant" || op->use_empty())
      return WalkResult::advance();
    auto found = dependence.find(op->getResult(0));
    if (found == dependence.end() || !found->second || found->second->empty())
      return WalkResult::advance();
    op->emitError("ONNX operation ")
        << name << " still depends on bounded dimension '"
        << found->second->front().getValue()
        << "' after its shape operands were bounded";
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

LogicalResult BoundONNXShapeOperandsPass::checkPrefix(
    Value value, StringAttr symbol, int64_t axis, DimExpr extent,
    const ONNXDimensionExprs &analysis, Tensor &atBound) {
  Operation *op = value.getDefiningOp();
  int64_t bound = *analysis.getBound(symbol);
  llvm::StringMap<int64_t> boundAssignment{{symbol.getValue(), bound}};
  Evaluator boundEvaluator(analysis, boundAssignment);
  std::optional<Tensor> full = boundEvaluator.evaluate(value);
  if (!full)
    return op->emitError("ONNX operation ")
           << op->getName() << " depends on bounded dimension '"
           << symbol.getValue() << "' and cannot be evaluated at its bound";
  atBound = *full;
  // The value at every extent must be the leading block of the value at the
  // bound along the dynamic axis: check every extent of the finite range.
  SmallVector<int64_t> boundStrides = stridesOf(full->shape);
  int64_t checked = 0;
  for (int64_t current = 0; current <= bound; ++current) {
    llvm::StringMap<int64_t> assignment{{symbol.getValue(), current}};
    Evaluator evaluator(analysis, assignment);
    std::optional<Tensor> part = evaluator.evaluate(value);
    if (!part || part->shape.size() != full->shape.size())
      return op->emitError("ONNX operation ")
             << op->getName() << " cannot be evaluated when '"
             << symbol.getValue() << "' is " << current;
    for (auto [index, size] : llvm::enumerate(part->shape))
      if (static_cast<int64_t>(index) != axis && size != full->shape[index])
        return op->emitError("ONNX operation ")
               << op->getName() << " changes a static extent with '"
               << symbol.getValue() << "'";
    if (part->shape[axis] != current + extent.offset ||
        part->shape[axis] > full->shape[axis])
      return op->emitError("ONNX operation ")
             << op->getName() << " does not have extent " << extent.str()
             << " on axis " << axis;
    checked += part->size();
    if (checked > maximumCheckedElements)
      return op->emitError("ONNX operation ")
             << op->getName()
             << " is too large to prove a bounded prefix for every extent";
    int64_t rank = part->shape.size();
    for (int64_t flat = 0, count = part->size(); flat < count; ++flat) {
      int64_t remaining = flat, source = 0;
      for (int64_t dimension = rank; dimension-- > 0;) {
        int64_t coordinate = remaining % part->shape[dimension];
        remaining /= part->shape[dimension];
        source += coordinate * boundStrides[dimension];
      }
      if (!part->sameBits(flat, *full, source))
        return op->emitError("ONNX operation ")
               << op->getName() << " computes data that depends on '"
               << symbol.getValue()
               << "' beyond a bounded prefix (it differs from its value at "
                  "the bound when '"
               << symbol.getValue() << "' is " << current << ")";
    }
  }
  return success();
}

LogicalResult BoundONNXShapeOperandsPass::rewriteShapeOperands(
    Operation *op, const ONNXDimensionExprs &analysis) {
  StringRef name = op->getName().getStringRef();
  OpBuilder builder(op);
  auto constant = [&](ArrayRef<int64_t> values) {
    auto type = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                      builder.getI64Type());
    return builder
        .create<arith::ConstantOp>(op->getLoc(),
                                   DenseIntElementsAttr::get(type, values))
        .getResult();
  };
  auto unsupported = [&](const Twine &why) {
    return op->emitError("ONNX operation ")
           << name << " has a shape operand that depends on a bounded "
           << "dimension: " << why;
  };
  const DimShape *input = analysis.getShape(op->getOperand(0));
  if (name == "onnx.Reshape") {
    const DimShape *requested = analysis.getIntegers(op->getOperand(1));
    const DimShape *result = analysis.getShape(op->getResult(0));
    auto allowZero = op->getAttrOfType<IntegerAttr>("allowzero");
    if (!requested || !result || !input || (allowZero && allowZero.getInt()))
      return unsupported("its extents are unknown");
    SmallVector<int64_t> shape;
    bool inferred = false;
    for (auto [index, extent] : llvm::enumerate(*result)) {
      if (extent.isConstant()) {
        shape.push_back(extent.offset);
        continue;
      }
      if (index < input->size() && (*input)[index] == extent) {
        shape.push_back(0);
        continue;
      }
      if (inferred || !extent.known)
        return unsupported("more than one extent is inferred");
      inferred = true;
      shape.push_back(-1);
    }
    op->setOperand(1, constant(shape));
    return success();
  }
  if (name == "onnx.Expand") {
    const DimShape *result = analysis.getShape(op->getResult(0));
    const DimShape *requested = analysis.getIntegers(op->getOperand(1));
    if (!input || !result || !requested)
      return unsupported("its extents are unknown");
    SmallVector<int64_t> shape;
    size_t offset = requested->size() - input->size();
    for (auto [index, extent] : llvm::enumerate(*requested)) {
      if (extent.isConstant()) {
        shape.push_back(extent.offset);
        continue;
      }
      if (index < offset || (*input)[index - offset] != extent)
        return unsupported("it broadcasts to an extent of a bounded "
                           "dimension");
      shape.push_back(1);
    }
    op->setOperand(1, constant(shape));
    return success();
  }
  if (name == "onnx.Slice") {
    const DimShape *starts = analysis.getIntegers(op->getOperand(1));
    const DimShape *ends = analysis.getIntegers(op->getOperand(2));
    if (!input || !starts || !ends || starts->size() != ends->size())
      return unsupported("its bounds are unknown");
    SmallVector<int64_t> axes;
    if (op->getNumOperands() >= 4 && !isNoValue(op->getOperand(3))) {
      const DimShape *values = analysis.getIntegers(op->getOperand(3));
      if (!values)
        return unsupported("its axes are unknown");
      for (const DimExpr &value : *values) {
        if (!value.isConstant())
          return unsupported("its axes depend on it");
        axes.push_back(value.offset < 0 ? value.offset + input->size()
                                        : value.offset);
      }
    } else {
      for (size_t index = 0; index < starts->size(); ++index)
        axes.push_back(index);
    }
    SmallVector<int64_t> steps;
    if (op->getNumOperands() >= 5 && !isNoValue(op->getOperand(4))) {
      const DimShape *values = analysis.getIntegers(op->getOperand(4));
      if (!values)
        return unsupported("its steps are unknown");
      for (const DimExpr &value : *values) {
        if (!value.isConstant())
          return unsupported("its steps depend on it");
        steps.push_back(value.offset);
      }
    }
    SmallVector<int64_t> newStarts, newEnds;
    for (size_t index = 0; index < starts->size(); ++index) {
      const DimExpr &start = (*starts)[index];
      const DimExpr &end = (*ends)[index];
      if (axes[index] < 0 || axes[index] >= static_cast<int64_t>(input->size()))
        return unsupported("an axis is out of range");
      if (!start.isConstant())
        return unsupported("a start depends on it");
      newStarts.push_back(start.offset);
      if (end.isConstant()) {
        newEnds.push_back(end.offset);
        continue;
      }
      std::optional<int> order = analysis.compare(end, (*input)[axes[index]]);
      if (!order || *order < 0)
        return unsupported("an end is not the whole extent of axis " +
                           Twine(axes[index]));
      newEnds.push_back(std::numeric_limits<int64_t>::max());
    }
    op->setOperand(1, constant(newStarts));
    op->setOperand(2, constant(newEnds));
    if (op->getNumOperands() >= 4 && !isNoValue(op->getOperand(3)))
      op->setOperand(3, constant(axes));
    if (!steps.empty())
      op->setOperand(4, constant(steps));
    return success();
  }
  return unsupported("the operator has no extent-free form");
}
} // namespace
} // namespace vrc
