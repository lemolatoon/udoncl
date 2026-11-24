#include "ONNXDimensionExprs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;

namespace {
bool isBetterShapedType(Type current, Type inferred) {
  auto currentTensor = dyn_cast<TensorType>(current);
  auto inferredTensor = dyn_cast<TensorType>(inferred);
  if (!currentTensor || !inferredTensor)
    return false;
  auto currentRanked = dyn_cast<RankedTensorType>(currentTensor);
  auto inferredRanked = dyn_cast<RankedTensorType>(inferredTensor);
  if (!inferredRanked)
    return false;
  if (!currentRanked)
    return true;
  if (currentRanked.getRank() != inferredRanked.getRank())
    return false;
  for (auto [currentDimension, inferredDimension] :
       llvm::zip(currentRanked.getShape(), inferredRanked.getShape()))
    if (ShapedType::isDynamic(currentDimension) &&
        !ShapedType::isDynamic(inferredDimension))
      return true;
  return false;
}

bool inferMatMulNBits(Operation *op) {
  if (op->getName().getStringRef() != "onnx.Custom" ||
      op->getNumOperands() < 1 || op->getNumResults() != 1)
    return false;
  auto domain = op->getAttrOfType<StringAttr>("domain_name");
  auto function = op->getAttrOfType<StringAttr>("function_name");
  auto columns = op->getAttrOfType<IntegerAttr>("N");
  auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!domain || domain.getValue() != "com.microsoft" || !function ||
      function.getValue() != "MatMulNBits" || !columns || !input ||
      input.getRank() == 0 || columns.getInt() <= 0)
    return false;
  SmallVector<int64_t> shape(input.getShape());
  shape.back() = columns.getInt();
  Type inferred = RankedTensorType::get(shape, input.getElementType());
  if (!isBetterShapedType(op->getResult(0).getType(), inferred))
    return false;
  op->getResult(0).setType(inferred);
  return true;
}

bool inferRmsNorm(Operation *op) {
  if (op->getName().getStringRef() != "onnx.Custom" ||
      op->getNumOperands() < 1 || op->getNumResults() != 1)
    return false;
  auto domain = op->getAttrOfType<StringAttr>("domain_name");
  auto function = op->getAttrOfType<StringAttr>("function_name");
  auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!domain || domain.getValue() != "com.lemolatoon.udoncl" || !function ||
      function.getValue() != "RmsNorm" || !input)
    return false;
  Type inferred =
      RankedTensorType::get(input.getShape(), input.getElementType());
  if (!isBetterShapedType(op->getResult(0).getType(), inferred))
    return false;
  op->getResult(0).setType(inferred);
  return true;
}

std::optional<SmallVector<int64_t>> broadcastShape(Operation *op) {
  SmallVector<int64_t> shape;
  for (Value operand : op->getOperands()) {
    auto tensor = dyn_cast<RankedTensorType>(operand.getType());
    if (!tensor)
      return std::nullopt;
    SmallVector<int64_t> next(tensor.getShape());
    size_t rank = std::max(shape.size(), next.size());
    SmallVector<int64_t> merged(rank, 1);
    for (size_t offset = 0; offset < rank; ++offset) {
      int64_t left =
          offset < shape.size() ? shape[shape.size() - 1 - offset] : 1;
      int64_t right = offset < next.size() ? next[next.size() - 1 - offset] : 1;
      int64_t dimension = left;
      if (left == 1)
        dimension = right;
      else if (right == 1 || left == right)
        dimension = left;
      else if (ShapedType::isDynamic(left))
        dimension = right;
      else if (!ShapedType::isDynamic(right))
        return std::nullopt;
      merged[rank - 1 - offset] = dimension;
    }
    shape = std::move(merged);
  }
  return shape;
}

std::optional<SmallVector<int64_t>>
sliceIndices(int64_t dimension, int64_t start, int64_t end, int64_t step) {
  if (dimension < 0 || !step)
    return std::nullopt;
  SmallVector<int64_t> result;
  constexpr size_t maximumElements = 1u << 20;
  if (step > 0) {
    if (start < 0)
      start = start < -dimension ? 0 : start + dimension;
    else
      start = std::min(start, dimension);
    if (end < 0)
      end = end < -dimension ? 0 : end + dimension;
    else
      end = std::min(end, dimension);
    for (int64_t index = start; index < end; index += step) {
      if (result.size() >= maximumElements ||
          index > std::numeric_limits<int64_t>::max() - step)
        return std::nullopt;
      result.push_back(index);
    }
    return result;
  }
  if (start < 0)
    start = start < -dimension ? -1 : start + dimension;
  else
    start = std::min(start, dimension - 1);
  if (end == std::numeric_limits<int64_t>::min())
    end = -1;
  else if (end < 0)
    end = end < -dimension ? -1 : end + dimension;
  else
    end = std::min(end, dimension - 1);
  for (int64_t index = start; index > end; index += step) {
    if (result.size() >= maximumElements ||
        index < std::numeric_limits<int64_t>::min() - step)
      return std::nullopt;
    result.push_back(index);
  }
  return result;
}

bool isONNXNoValue(Value value) {
  Operation *definition = value.getDefiningOp();
  return definition && definition->getName().getStringRef() == "onnx.NoValue";
}

std::optional<SmallVector<int64_t>> evaluateIntegerTensor(Value value,
                                                          unsigned depth = 0) {
  if (depth > 64)
    return std::nullopt;
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  StringRef name = op->getName().getStringRef();
  if (name == "onnx.Constant" || name == "arith.constant") {
    auto dense = op->getAttrOfType<DenseElementsAttr>("value");
    if (!dense)
      if (auto disposable = op->getAttrOfType<DisposableElementsAttr>("value"))
        if (isa<IntegerType>(disposable.getElementType()))
          dense = disposable.toDenseElementsAttr();
    if (!dense || !isa<IntegerType>(dense.getElementType()))
      return std::nullopt;
    SmallVector<int64_t> result;
    if (dense.isSplat()) {
      result.assign(dense.getNumElements(),
                    dense.getSplatValue<APInt>().getSExtValue());
      return result;
    }
    for (APInt integer : dense.getValues<APInt>())
      result.push_back(integer.getSExtValue());
    return result;
  }
  if (name == "onnx.Dim" && op->getNumOperands() == 1) {
    auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    if (!input || !axis)
      return std::nullopt;
    int64_t index = axis.getInt();
    if (index < 0)
      index += input.getRank();
    if (index < 0 || index >= input.getRank() || input.isDynamicDim(index))
      return std::nullopt;
    return SmallVector<int64_t>{input.getDimSize(index)};
  }
  if (name == "onnx.Shape" && op->getNumOperands() == 1) {
    auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!input || !input.hasStaticShape())
      return std::nullopt;
    int64_t start = 0;
    int64_t end = input.getRank();
    if (auto attribute = op->getAttrOfType<IntegerAttr>("start"))
      start = attribute.getInt();
    if (auto attribute = op->getAttrOfType<IntegerAttr>("end"))
      end = attribute.getInt();
    if (start < 0)
      start += input.getRank();
    if (end < 0)
      end += input.getRank();
    start = std::clamp<int64_t>(start, 0, input.getRank());
    end = std::clamp<int64_t>(end, start, input.getRank());
    return SmallVector<int64_t>(input.getShape().slice(start, end - start));
  }
  if (name == "onnx.Gather" && op->getNumOperands() == 2 &&
      (!op->getAttrOfType<IntegerAttr>("axis") ||
       op->getAttrOfType<IntegerAttr>("axis").getInt() == 0)) {
    auto data = evaluateIntegerTensor(op->getOperand(0), depth + 1);
    auto indices = evaluateIntegerTensor(op->getOperand(1), depth + 1);
    if (!data || !indices)
      return std::nullopt;
    SmallVector<int64_t> result;
    for (int64_t index : *indices) {
      if (index < 0)
        index += data->size();
      if (index < 0 || index >= static_cast<int64_t>(data->size()))
        return std::nullopt;
      result.push_back((*data)[index]);
    }
    return result;
  }
  if ((name == "onnx.Equal" || name == "onnx.Greater") &&
      op->getNumOperands() == 2 && op->getNumResults() == 1) {
    auto left = evaluateIntegerTensor(op->getOperand(0), depth + 1);
    auto right = evaluateIntegerTensor(op->getOperand(1), depth + 1);
    auto leftType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto rightType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!left || !right || !leftType || !rightType || !resultType ||
        !leftType.hasStaticShape() || !rightType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        left->size() != static_cast<size_t>(leftType.getNumElements()) ||
        right->size() != static_cast<size_t>(rightType.getNumElements()) ||
        resultType.getNumElements() > (1u << 20))
      return std::nullopt;
    auto inputOffset =
        [&](RankedTensorType type,
            ArrayRef<int64_t> coordinates) -> std::optional<size_t> {
      if (type.getRank() > static_cast<int64_t>(coordinates.size()))
        return std::nullopt;
      size_t offset = 0;
      size_t stride = 1;
      int64_t rankOffset = coordinates.size() - type.getRank();
      for (int64_t axis = type.getRank(); axis-- > 0;) {
        int64_t dimension = type.getDimSize(axis);
        int64_t coordinate = coordinates[rankOffset + axis];
        if (dimension != 1 &&
            dimension != resultType.getDimSize(rankOffset + axis))
          return std::nullopt;
        if (dimension != 1)
          offset += static_cast<size_t>(coordinate) * stride;
        stride *= static_cast<size_t>(dimension);
      }
      return offset;
    };
    SmallVector<int64_t> result;
    result.reserve(resultType.getNumElements());
    for (int64_t flat = 0; flat < resultType.getNumElements(); ++flat) {
      int64_t remaining = flat;
      SmallVector<int64_t> coordinates(resultType.getRank());
      for (int64_t axis = resultType.getRank(); axis-- > 0;) {
        coordinates[axis] = remaining % resultType.getDimSize(axis);
        remaining /= resultType.getDimSize(axis);
      }
      auto leftOffset = inputOffset(leftType, coordinates);
      auto rightOffset = inputOffset(rightType, coordinates);
      if (!leftOffset || !rightOffset)
        return std::nullopt;
      result.push_back(name == "onnx.Equal"
                           ? (*left)[*leftOffset] == (*right)[*rightOffset]
                           : (*left)[*leftOffset] > (*right)[*rightOffset]);
    }
    return result;
  }
  if (name == "onnx.Where" && op->getNumOperands() == 3 &&
      op->getNumResults() == 1) {
    auto condition = evaluateIntegerTensor(op->getOperand(0), depth + 1);
    auto whenTrue = evaluateIntegerTensor(op->getOperand(1), depth + 1);
    auto whenFalse = evaluateIntegerTensor(op->getOperand(2), depth + 1);
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!condition || !whenTrue || !whenFalse || !resultType ||
        !resultType.hasStaticShape() ||
        resultType.getNumElements() > (1u << 20))
      return std::nullopt;
    size_t count = resultType.getNumElements();
    auto compatible = [&](ArrayRef<int64_t> values) {
      return values.size() == 1 || values.size() == count;
    };
    if (!compatible(*condition) || !compatible(*whenTrue) ||
        !compatible(*whenFalse))
      return std::nullopt;
    SmallVector<int64_t> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      bool selected = (*condition)[condition->size() == 1 ? 0 : index] != 0;
      ArrayRef<int64_t> values = selected ? *whenTrue : *whenFalse;
      result.push_back(values[values.size() == 1 ? 0 : index]);
    }
    return result;
  }
  if ((name == "onnx.Add" || name == "onnx.Sub" || name == "onnx.Mul" ||
       name == "onnx.Div") &&
      op->getNumOperands() == 2) {
    auto left = evaluateIntegerTensor(op->getOperand(0), depth + 1);
    auto right = evaluateIntegerTensor(op->getOperand(1), depth + 1);
    if (!left || !right || left->empty() || right->empty())
      return std::nullopt;
    size_t count = std::max(left->size(), right->size());
    if ((left->size() != 1 && left->size() != count) ||
        (right->size() != 1 && right->size() != count))
      return std::nullopt;
    SmallVector<int64_t> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      int64_t lhs = (*left)[left->size() == 1 ? 0 : index];
      int64_t rhs = (*right)[right->size() == 1 ? 0 : index];
      APInt leftValue(64, lhs, true);
      APInt rightValue(64, rhs, true);
      APInt value(64, 0);
      bool overflow = false;
      if (name == "onnx.Add")
        value = leftValue.sadd_ov(rightValue, overflow);
      else if (name == "onnx.Sub")
        value = leftValue.ssub_ov(rightValue, overflow);
      else if (name == "onnx.Mul")
        value = leftValue.smul_ov(rightValue, overflow);
      else {
        if (!rhs || (lhs == std::numeric_limits<int64_t>::min() && rhs == -1))
          return std::nullopt;
        value = leftValue.sdiv(rightValue);
      }
      if (overflow)
        return std::nullopt;
      result.push_back(value.getSExtValue());
    }
    return result;
  }
  if (name == "onnx.Range" && op->getNumOperands() == 3) {
    auto starts = evaluateIntegerTensor(op->getOperand(0), depth + 1);
    auto limits = evaluateIntegerTensor(op->getOperand(1), depth + 1);
    auto deltas = evaluateIntegerTensor(op->getOperand(2), depth + 1);
    if (!starts || starts->size() != 1 || !limits || limits->size() != 1 ||
        !deltas || deltas->size() != 1 || (*deltas)[0] == 0)
      return std::nullopt;
    int64_t current = (*starts)[0];
    int64_t limit = (*limits)[0];
    int64_t delta = (*deltas)[0];
    SmallVector<int64_t> result;
    constexpr size_t maximumElements = 1u << 20;
    while ((delta > 0 && current < limit) || (delta < 0 && current > limit)) {
      if (result.size() >= maximumElements)
        return std::nullopt;
      result.push_back(current);
      if ((delta > 0 &&
           current > std::numeric_limits<int64_t>::max() - delta) ||
          (delta < 0 && current < std::numeric_limits<int64_t>::min() - delta))
        return std::nullopt;
      current += delta;
    }
    return result;
  }
  if (name == "onnx.Slice" && op->getNumOperands() >= 3) {
    auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto data = evaluateIntegerTensor(op->getOperand(0), depth + 1);
    auto starts = evaluateIntegerTensor(op->getOperand(1), depth + 1);
    auto ends = evaluateIntegerTensor(op->getOperand(2), depth + 1);
    if (!type || type.getRank() != 1 || !type.hasStaticShape() || !data ||
        !starts || !ends || starts->size() != 1 || ends->size() != 1)
      return std::nullopt;
    SmallVector<int64_t> axes{0};
    SmallVector<int64_t> steps{1};
    if (op->getNumOperands() >= 4 && !isONNXNoValue(op->getOperand(3))) {
      auto values = evaluateIntegerTensor(op->getOperand(3), depth + 1);
      if (!values)
        return std::nullopt;
      axes = std::move(*values);
    }
    if (op->getNumOperands() >= 5 && !isONNXNoValue(op->getOperand(4))) {
      auto values = evaluateIntegerTensor(op->getOperand(4), depth + 1);
      if (!values)
        return std::nullopt;
      steps = std::move(*values);
    }
    if (axes.size() != 1 || axes[0] != 0 || steps.size() != 1)
      return std::nullopt;
    auto indices =
        sliceIndices(type.getDimSize(0), (*starts)[0], (*ends)[0], steps[0]);
    if (!indices)
      return std::nullopt;
    SmallVector<int64_t> result;
    for (int64_t index : *indices)
      result.push_back((*data)[index]);
    return result;
  }
  if ((name == "onnx.Unsqueeze" || name == "onnx.Squeeze" ||
       name == "onnx.Reshape" || name == "onnx.Identity" ||
       name == "onnx.Cast") &&
      op->getNumOperands() >= 1)
    return evaluateIntegerTensor(op->getOperand(0), depth + 1);
  if (name == "onnx.Concat" && op->getNumResults() == 1) {
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    int64_t axis = 0;
    if (auto attribute = op->getAttrOfType<IntegerAttr>("axis"))
      axis = attribute.getInt();
    if (resultType && axis < 0)
      axis += resultType.getRank();
    if (axis == 0) {
      SmallVector<int64_t> result;
      for (Value operand : op->getOperands()) {
        auto values = evaluateIntegerTensor(operand, depth + 1);
        if (!values)
          return std::nullopt;
        llvm::append_range(result, *values);
      }
      return result;
    }
    // A non-leading axis interleaves one contiguous chunk of every operand
    // per outer index.
    if (!resultType || !resultType.hasStaticShape() || axis < 0 ||
        axis >= resultType.getRank() ||
        resultType.getNumElements() > (1u << 20))
      return std::nullopt;
    int64_t outer = 1;
    for (int64_t extent : resultType.getShape().take_front(axis))
      outer *= extent;
    SmallVector<SmallVector<int64_t>> operands;
    SmallVector<int64_t> chunks;
    for (Value operand : op->getOperands()) {
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      auto values = evaluateIntegerTensor(operand, depth + 1);
      if (!type || !type.hasStaticShape() ||
          type.getRank() != resultType.getRank() || !values ||
          values->size() != static_cast<size_t>(type.getNumElements()))
        return std::nullopt;
      int64_t chunk = 1;
      for (int64_t extent : type.getShape().drop_front(axis))
        chunk *= extent;
      chunks.push_back(chunk);
      operands.push_back(std::move(*values));
    }
    SmallVector<int64_t> result;
    result.reserve(resultType.getNumElements());
    for (int64_t index = 0; index < outer; ++index)
      for (auto [values, chunk] : llvm::zip_equal(operands, chunks))
        llvm::append_range(
            result, ArrayRef<int64_t>(values).slice(index * chunk, chunk));
    if (result.size() != static_cast<size_t>(resultType.getNumElements()))
      return std::nullopt;
    return result;
  }
  if (name == "onnx.Expand" && op->getNumOperands() == 2 &&
      op->getNumResults() == 1) {
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto values = evaluateIntegerTensor(op->getOperand(0), depth + 1);
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        inputType.getRank() > resultType.getRank() ||
        resultType.getNumElements() > (1u << 20) || !values ||
        values->size() != static_cast<size_t>(inputType.getNumElements()))
      return std::nullopt;
    int64_t rankOffset = resultType.getRank() - inputType.getRank();
    for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
      int64_t extent = inputType.getDimSize(axis);
      if (extent != 1 && extent != resultType.getDimSize(rankOffset + axis))
        return std::nullopt;
    }
    SmallVector<int64_t> result;
    result.reserve(resultType.getNumElements());
    for (int64_t flat = 0; flat < resultType.getNumElements(); ++flat) {
      int64_t remaining = flat;
      int64_t offset = 0;
      int64_t stride = 1;
      for (int64_t axis = resultType.getRank(); axis-- > 0;) {
        int64_t coordinate = remaining % resultType.getDimSize(axis);
        remaining /= resultType.getDimSize(axis);
        int64_t inputAxis = axis - rankOffset;
        if (inputAxis < 0)
          continue;
        int64_t extent = inputType.getDimSize(inputAxis);
        if (extent != 1)
          offset += coordinate * stride;
        stride *= extent;
      }
      result.push_back((*values)[offset]);
    }
    return result;
  }
  return std::nullopt;
}

std::optional<Type> inferGenericResultType(Operation *op) {
  if (op->getNumResults() != 1)
    return std::nullopt;
  auto current = dyn_cast<TensorType>(op->getResult(0).getType());
  if (!current)
    return std::nullopt;
  StringRef name = op->getName().getStringRef().drop_front(5);
  auto rankedOperand = [&](unsigned index) -> RankedTensorType {
    if (index >= op->getNumOperands())
      return {};
    return dyn_cast<RankedTensorType>(op->getOperand(index).getType());
  };
  auto dynamicTensor = [&](int64_t rank) -> Type {
    return RankedTensorType::get(
        SmallVector<int64_t>(rank, ShapedType::kDynamic),
        current.getElementType());
  };
  if (llvm::StringSwitch<bool>(name)
          .Cases({"Add", "Div", "Equal", "Mul", "Sub", "Where"}, true)
          .Default(false)) {
    auto shape = broadcastShape(op);
    if (shape)
      return RankedTensorType::get(*shape, current.getElementType());
  }
  if (name == "Shape") {
    if (auto input = rankedOperand(0)) {
      int64_t start = 0;
      int64_t end = input.getRank();
      if (auto attr = op->getAttrOfType<IntegerAttr>("start"))
        start =
            attr.getInt() < 0 ? attr.getInt() + input.getRank() : attr.getInt();
      if (auto attr = op->getAttrOfType<IntegerAttr>("end"))
        end =
            attr.getInt() < 0 ? attr.getInt() + input.getRank() : attr.getInt();
      return RankedTensorType::get({std::max<int64_t>(0, end - start)},
                                   current.getElementType());
    }
  }
  if (name == "ConstantOfShape" && op->getNumOperands() == 1)
    if (auto shape = evaluateIntegerTensor(op->getOperand(0)))
      if (llvm::all_of(*shape,
                       [](int64_t dimension) { return dimension >= 0; }))
        return RankedTensorType::get(*shape, current.getElementType());
  if (name == "Range")
    if (auto values = evaluateIntegerTensor(op->getResult(0)))
      return RankedTensorType::get({static_cast<int64_t>(values->size())},
                                   current.getElementType());
  if (name == "MatMul") {
    auto left = rankedOperand(0);
    auto right = rankedOperand(1);
    if (left && right && left.getRank() >= 2 && right.getRank() >= 2) {
      if (right.getRank() == 2) {
        SmallVector<int64_t> shape(left.getShape());
        shape.back() = right.getDimSize(1);
        return RankedTensorType::get(shape, current.getElementType());
      }
      int64_t rank = std::max(left.getRank(), right.getRank());
      SmallVector<int64_t> shape(rank, ShapedType::kDynamic);
      int64_t leftOffset = rank - left.getRank();
      int64_t rightOffset = rank - right.getRank();
      for (int64_t axis = 0; axis < rank - 2; ++axis) {
        int64_t leftDimension =
            axis < leftOffset ? 1 : left.getDimSize(axis - leftOffset);
        int64_t rightDimension =
            axis < rightOffset ? 1 : right.getDimSize(axis - rightOffset);
        if (leftDimension == 1)
          shape[axis] = rightDimension;
        else if (rightDimension == 1 || leftDimension == rightDimension)
          shape[axis] = leftDimension;
      }
      shape[rank - 2] = left.getDimSize(left.getRank() - 2);
      shape[rank - 1] = right.getDimSize(right.getRank() - 1);
      return RankedTensorType::get(shape, current.getElementType());
    }
  }
  if (name == "Gather") {
    auto data = rankedOperand(0);
    auto indices = rankedOperand(1);
    if (data && indices) {
      int64_t axis = 0;
      if (auto attribute = op->getAttrOfType<IntegerAttr>("axis"))
        axis = attribute.getInt();
      if (axis < 0)
        axis += data.getRank();
      if (axis < 0 || axis >= data.getRank())
        return std::nullopt;
      SmallVector<int64_t> shape;
      llvm::append_range(shape, data.getShape().take_front(axis));
      llvm::append_range(shape, indices.getShape());
      llvm::append_range(shape, data.getShape().drop_front(axis + 1));
      return RankedTensorType::get(shape, current.getElementType());
    }
  }
  if (name == "Concat") {
    int64_t rank = -1;
    SmallVector<int64_t> shape;
    int64_t axis = 0;
    if (auto attribute = op->getAttrOfType<IntegerAttr>("axis"))
      axis = attribute.getInt();
    for (Value operand : op->getOperands()) {
      auto tensor = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensor || (rank >= 0 && tensor.getRank() != rank))
        return std::nullopt;
      if (rank < 0) {
        rank = tensor.getRank();
        if (axis < 0)
          axis += rank;
        if (axis < 0 || axis >= rank)
          return std::nullopt;
        shape.assign(tensor.getShape().begin(), tensor.getShape().end());
        continue;
      }
      for (int64_t dimension = 0; dimension < rank; ++dimension) {
        if (dimension == axis) {
          if (ShapedType::isDynamic(shape[dimension]) ||
              tensor.isDynamicDim(dimension))
            shape[dimension] = ShapedType::kDynamic;
          else
            shape[dimension] += tensor.getDimSize(dimension);
        } else if (shape[dimension] != tensor.getDimSize(dimension)) {
          if (ShapedType::isDynamic(shape[dimension]))
            shape[dimension] = tensor.getDimSize(dimension);
          else if (!tensor.isDynamicDim(dimension))
            return std::nullopt;
        }
      }
    }
    if (rank >= 0)
      return RankedTensorType::get(shape, current.getElementType());
  }
  if (name == "Expand") {
    if (op->getNumOperands() >= 2)
      if (auto dimensions = evaluateIntegerTensor(op->getOperand(1))) {
        auto input = rankedOperand(0);
        if (input) {
          size_t rank = std::max<size_t>(input.getRank(), dimensions->size());
          SmallVector<int64_t> shape(rank, 1);
          for (size_t offset = 0; offset < rank; ++offset) {
            int64_t inputDimension =
                offset < static_cast<size_t>(input.getRank())
                    ? input.getDimSize(input.getRank() - 1 - offset)
                    : 1;
            int64_t requestedDimension =
                offset < dimensions->size()
                    ? (*dimensions)[dimensions->size() - 1 - offset]
                    : 1;
            if (requestedDimension < 0)
              return std::nullopt;
            int64_t dimension;
            if (inputDimension == 1)
              dimension = requestedDimension;
            else if (requestedDimension == 1 ||
                     inputDimension == requestedDimension)
              dimension = inputDimension;
            else if (ShapedType::isDynamic(inputDimension))
              dimension = requestedDimension;
            else
              return std::nullopt;
            shape[rank - 1 - offset] = dimension;
          }
          return RankedTensorType::get(shape, current.getElementType());
        }
      }
    auto shape = rankedOperand(1);
    if (shape && shape.getRank() == 1 &&
        !ShapedType::isDynamic(shape.getDimSize(0)))
      return dynamicTensor(shape.getDimSize(0));
  }
  if (name == "Reshape") {
    if (op->getNumOperands() >= 2)
      if (auto dimensions = evaluateIntegerTensor(op->getOperand(1))) {
        SmallVector<int64_t> shape(*dimensions);
        auto input = rankedOperand(0);
        int64_t inferredIndex = -1;
        int64_t knownElements = 1;
        for (auto [index, dimension] : llvm::enumerate(shape)) {
          auto allowZero = op->getAttrOfType<IntegerAttr>("allowzero");
          if (dimension == 0 && name == "Reshape" && input &&
              index < static_cast<size_t>(input.getRank()) &&
              (!allowZero || allowZero.getInt() == 0))
            dimension = input.getDimSize(index);
          shape[index] = dimension;
          if (dimension == -1) {
            if (inferredIndex >= 0)
              return std::nullopt;
            inferredIndex = index;
          } else if (dimension < 0) {
            return std::nullopt;
          } else {
            knownElements *= dimension;
          }
        }
        if (inferredIndex >= 0 && input && input.hasStaticShape() &&
            knownElements > 0 && input.getNumElements() % knownElements == 0)
          shape[inferredIndex] = input.getNumElements() / knownElements;
        else if (inferredIndex >= 0)
          shape[inferredIndex] = ShapedType::kDynamic;
        if (llvm::none_of(shape, [](int64_t dimension) {
              return ShapedType::isDynamic(dimension);
            }))
          return RankedTensorType::get(shape, current.getElementType());
      }
    auto shape = rankedOperand(1);
    if (shape && shape.getRank() == 1 &&
        !ShapedType::isDynamic(shape.getDimSize(0)))
      return dynamicTensor(shape.getDimSize(0));
  }
  if (name == "Unsqueeze") {
    auto input = rankedOperand(0);
    if (input && op->getNumOperands() >= 2)
      if (auto axes = evaluateIntegerTensor(op->getOperand(1))) {
        int64_t outputRank = input.getRank() + axes->size();
        SmallVector<bool> inserted(outputRank, false);
        for (int64_t axis : *axes) {
          if (axis < 0)
            axis += outputRank;
          if (axis < 0 || axis >= outputRank || inserted[axis])
            return std::nullopt;
          inserted[axis] = true;
        }
        SmallVector<int64_t> shape;
        shape.reserve(outputRank);
        size_t inputAxis = 0;
        for (int64_t axis = 0; axis < outputRank; ++axis)
          shape.push_back(inserted[axis] ? 1 : input.getDimSize(inputAxis++));
        return RankedTensorType::get(shape, current.getElementType());
      }
  }
  if (name == "Squeeze") {
    auto input = rankedOperand(0);
    if (input) {
      SmallVector<bool> removed(input.getRank(), false);
      if (op->getNumOperands() >= 2 && !isONNXNoValue(op->getOperand(1))) {
        auto axes = evaluateIntegerTensor(op->getOperand(1));
        if (!axes)
          return std::nullopt;
        for (int64_t axis : *axes) {
          if (axis < 0)
            axis += input.getRank();
          if (axis < 0 || axis >= input.getRank() || removed[axis] ||
              input.getDimSize(axis) != 1)
            return std::nullopt;
          removed[axis] = true;
        }
      } else {
        for (int64_t axis = 0; axis < input.getRank(); ++axis)
          removed[axis] = input.getDimSize(axis) == 1;
      }
      SmallVector<int64_t> shape;
      for (auto [axis, dimension] : llvm::enumerate(input.getShape()))
        if (!removed[axis])
          shape.push_back(dimension);
      return RankedTensorType::get(shape, current.getElementType());
    }
  }
  if (name == "Slice" && op->getNumOperands() >= 3) {
    auto input = rankedOperand(0);
    auto starts = evaluateIntegerTensor(op->getOperand(1));
    auto ends = evaluateIntegerTensor(op->getOperand(2));
    if (input && input.hasStaticShape() && starts && ends &&
        starts->size() == ends->size()) {
      SmallVector<int64_t> axes;
      SmallVector<int64_t> steps(starts->size(), 1);
      if (op->getNumOperands() >= 4 && !isONNXNoValue(op->getOperand(3))) {
        auto values = evaluateIntegerTensor(op->getOperand(3));
        if (!values)
          return std::nullopt;
        axes = std::move(*values);
      } else {
        for (size_t index = 0; index < starts->size(); ++index)
          axes.push_back(index);
      }
      if (op->getNumOperands() >= 5 && !isONNXNoValue(op->getOperand(4))) {
        auto values = evaluateIntegerTensor(op->getOperand(4));
        if (!values)
          return std::nullopt;
        steps = std::move(*values);
      }
      if (axes.size() == starts->size() && steps.size() == starts->size()) {
        SmallVector<int64_t> shape(input.getShape());
        for (size_t index = 0; index < starts->size(); ++index) {
          int64_t axis = axes[index];
          if (axis < 0)
            axis += input.getRank();
          if (axis < 0 || axis >= input.getRank())
            return std::nullopt;
          auto indices = sliceIndices(input.getDimSize(axis), (*starts)[index],
                                      (*ends)[index], steps[index]);
          if (!indices)
            return std::nullopt;
          shape[axis] = indices->size();
        }
        return RankedTensorType::get(shape, current.getElementType());
      }
    }
  }
  if (name == "Transpose") {
    if (auto input = rankedOperand(0)) {
      SmallVector<int64_t> permutation;
      if (auto attribute = op->getAttrOfType<ArrayAttr>("perm"))
        for (Attribute value : attribute)
          permutation.push_back(cast<IntegerAttr>(value).getInt());
      else
        for (int64_t axis = input.getRank(); axis-- > 0;)
          permutation.push_back(axis);
      if (permutation.size() != static_cast<size_t>(input.getRank()))
        return std::nullopt;
      SmallVector<int64_t> shape;
      SmallVector<bool> seen(input.getRank(), false);
      for (int64_t axis : permutation) {
        if (axis < 0 || axis >= input.getRank() || seen[axis])
          return std::nullopt;
        seen[axis] = true;
        shape.push_back(input.getDimSize(axis));
      }
      return RankedTensorType::get(shape, current.getElementType());
    }
  }
  if (name == "ScatterND") {
    if (auto input = rankedOperand(0))
      return RankedTensorType::get(input.getShape(), current.getElementType());
  }
  if (name == "Flatten")
    return dynamicTensor(2);
  if (llvm::StringSwitch<bool>(name)
          .Cases({"Cast", "Cos", "Identity", "Neg"}, true)
          .Cases({"Relu", "Sigmoid", "Sin", "Softmax", "Sqrt", "Tanh"}, true)
          .Case("Trilu", true)
          .Default(false)) {
    if (op->getNumOperands() >= 1)
      if (auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType()))
        return RankedTensorType::get(input.getShape(),
                                     current.getElementType());
  }
  if (name == "ReduceMeanV13" && op->getNumOperands() >= 1) {
    auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!input)
      return std::nullopt;
    SmallVector<int64_t> axes;
    if (auto attr = op->getAttrOfType<DenseI64ArrayAttr>("axes"))
      llvm::append_range(axes, attr.asArrayRef());
    else if (auto attr = op->getAttrOfType<ArrayAttr>("axes"))
      for (Attribute value : attr)
        axes.push_back(cast<IntegerAttr>(value).getInt());
    if (axes.empty())
      return std::nullopt;
    bool keepDimensions = true;
    if (auto keep = op->getAttrOfType<IntegerAttr>("keepdims"))
      keepDimensions = keep.getInt() != 0;
    SmallVector<bool> reduced(input.getRank(), false);
    for (int64_t axis : axes) {
      if (axis < 0)
        axis += input.getRank();
      if (axis < 0 || axis >= input.getRank())
        return std::nullopt;
      reduced[axis] = true;
    }
    SmallVector<int64_t> shape;
    for (auto [axis, dimension] : llvm::enumerate(input.getShape()))
      if (!reduced[axis])
        shape.push_back(dimension);
      else if (keepDimensions)
        shape.push_back(1);
    return RankedTensorType::get(shape, current.getElementType());
  }
  return std::nullopt;
}

struct InferONNXShapesPass
    : PassWrapper<InferONNXShapesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferONNXShapesPass)
  StringRef getArgument() const final { return "udoncl-infer-onnx-shapes"; }
  StringRef getDescription() const final {
    return "Propagate ONNX shapes across extension operators to a fixed point";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto materializeShapeValues = [&] {
      SmallVector<Operation *> erase;
      module.walk([&](Operation *op) {
        if (!op->getName().getStringRef().starts_with("onnx.") ||
            op->getName().getStringRef() == "onnx.Constant" ||
            op->getNumResults() != 1)
          return;
        if (op->getName().getStringRef() == "onnx.ConstantOfShape" &&
            op->getNumOperands() == 1) {
          auto shape = evaluateIntegerTensor(op->getOperand(0));
          auto current = dyn_cast<TensorType>(op->getResult(0).getType());
          DenseElementsAttr value =
              op->getAttrOfType<DenseElementsAttr>("value");
          if (!value)
            if (auto disposable =
                    op->getAttrOfType<DisposableElementsAttr>("value"))
              value = disposable.toDenseElementsAttr();
          if (!shape || !current || !value || !value.isSplat() ||
              value.getElementType() != current.getElementType() ||
              llvm::any_of(*shape,
                           [](int64_t dimension) { return dimension < 0; }))
            return;
          uint64_t elements = 1;
          for (int64_t dimension : *shape) {
            if (dimension && elements > (1u << 20) / dimension)
              return;
            elements *= dimension;
          }
          auto type = RankedTensorType::get(*shape, current.getElementType());
          DenseElementsAttr dense;
          if (isa<FloatType>(current.getElementType()))
            dense =
                DenseElementsAttr::get(type, value.getSplatValue<APFloat>());
          else if (isa<IntegerType>(current.getElementType()))
            dense = DenseElementsAttr::get(type, value.getSplatValue<APInt>());
          else
            return;
          op->getResult(0).setType(type);
          OpBuilder builder(op);
          Value constant =
              builder.create<arith::ConstantOp>(op->getLoc(), dense);
          op->getResult(0).replaceAllUsesWith(constant);
          erase.push_back(op);
          return;
        }
        auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!type || !type.hasStaticShape())
          return;
        if (auto floatType = dyn_cast<FloatType>(type.getElementType())) {
          // A Cast of a statically known integer tensor to floating point is
          // folded the same way as integer shape values.
          auto inputType =
              op->getName().getStringRef() == "onnx.Cast" &&
                      op->getNumOperands() == 1
                  ? dyn_cast<RankedTensorType>(op->getOperand(0).getType())
                  : RankedTensorType();
          auto inputElement =
              inputType ? dyn_cast<IntegerType>(inputType.getElementType())
                        : IntegerType();
          if (!inputElement || inputElement.isUnsigned() ||
              inputType.getShape() != type.getShape())
            return;
          auto values = evaluateIntegerTensor(op->getOperand(0));
          if (!values ||
              values->size() != static_cast<size_t>(type.getNumElements()))
            return;
          SmallVector<APFloat> floats;
          floats.reserve(values->size());
          for (int64_t value : *values) {
            if (inputElement.getWidth() == 1)
              value = value != 0;
            APFloat converted(floatType.getFloatSemantics());
            converted.convertFromAPInt(APInt(64, value, /*isSigned=*/true),
                                       /*IsSigned=*/true,
                                       APFloat::rmNearestTiesToEven);
            floats.push_back(converted);
          }
          OpBuilder builder(op);
          Value constant = builder.create<arith::ConstantOp>(
              op->getLoc(), DenseElementsAttr::get(type, floats));
          op->getResult(0).replaceAllUsesWith(constant);
          erase.push_back(op);
          return;
        }
        if (!isa<IntegerType>(type.getElementType()))
          return;
        auto values = evaluateIntegerTensor(op->getResult(0));
        if (!values ||
            values->size() != static_cast<size_t>(type.getNumElements()))
          return;
        unsigned width = type.getElementTypeBitWidth();
        SmallVector<APInt> integers;
        for (int64_t value : *values)
          integers.emplace_back(width, value, true);
        OpBuilder builder(op);
        Value constant = builder.create<arith::ConstantOp>(
            op->getLoc(), DenseElementsAttr::get(type, integers));
        op->getResult(0).replaceAllUsesWith(constant);
        erase.push_back(op);
      });
      for (Operation *op : llvm::reverse(erase))
        op->erase();
    };
    auto updateFunctionResults = [&] {
      for (auto function : module.getOps<func::FuncOp>()) {
        if (function.isDeclaration())
          continue;
        Operation *returnOp = nullptr;
        function.walk([&](Operation *candidate) {
          StringRef name = candidate->getName().getStringRef();
          if (!returnOp && (name == func::ReturnOp::getOperationName() ||
                            name == "onnx.Return"))
            returnOp = candidate;
        });
        if (!returnOp)
          continue;
        function.setType(FunctionType::get(module.getContext(),
                                           function.getArgumentTypes(),
                                           returnOp->getOperandTypes()));
      }
    };
    // With bounded dimensions, the symbolic analysis also gives every axis
    // that depends on a bounded dimension its extent expression, keeps it
    // dynamic in the type, and folds the shape values it proves constant.
    bool bounded = module->hasAttr(vrc::onnxDimensionBoundsAttr);
    vrc::ONNXConstantReader reader =
        [](Operation *op) -> std::optional<SmallVector<int64_t>> {
      auto disposable = op->getAttrOfType<DisposableElementsAttr>("value");
      if (!disposable || !isa<IntegerType>(disposable.getElementType()) ||
          disposable.getNumElements() > 64)
        return std::nullopt;
      SmallVector<int64_t> values;
      for (APInt integer : disposable.toDenseElementsAttr().getValues<APInt>())
        values.push_back(integer.getSExtValue());
      return values;
    };
    auto inferSymbolicShapes = [&] {
      bool changed = false;
      for (auto function : module.getOps<func::FuncOp>()) {
        if (function.isDeclaration())
          continue;
        vrc::ONNXDimensionExprs analysis(function, reader);
        function.walk([&](Operation *op) {
          if (!op->getName().getStringRef().starts_with("onnx."))
            return;
          for (OpResult result : op->getResults()) {
            const vrc::DimShape *shape = analysis.getShape(result);
            auto current = dyn_cast<TensorType>(result.getType());
            if (!shape || !current)
              continue;
            Type inferred = vrc::ONNXDimensionExprs::typeOf(
                *shape, current.getElementType());
            if (!isBetterShapedType(result.getType(), inferred))
              continue;
            result.setType(inferred);
            changed = true;
          }
        });
      }
      return changed;
    };
    auto materializeSymbolicValues = [&]() -> LogicalResult {
      for (auto function : module.getOps<func::FuncOp>()) {
        if (function.isDeclaration())
          continue;
        vrc::ONNXDimensionExprs analysis(function, reader);
        for (const auto &[op, message] : analysis.getErrors())
          op->emitError("ONNX operation ") << message;
        if (!analysis.getErrors().empty())
          return failure();
        SmallVector<Operation *> erase;
        function.walk([&](Operation *op) {
          StringRef name = op->getName().getStringRef();
          if (!name.starts_with("onnx.") || name == "onnx.Constant" ||
              op->getNumResults() != 1)
            return;
          auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          const vrc::DimShape *values = analysis.getIntegers(op->getResult(0));
          if (!type || !type.hasStaticShape() ||
              !isa<IntegerType>(type.getElementType()) || !values ||
              values->size() != static_cast<size_t>(type.getNumElements()) ||
              llvm::any_of(*values, [](const vrc::DimExpr &value) {
                return !value.isConstant();
              }))
            return;
          SmallVector<APInt> integers;
          for (const vrc::DimExpr &value : *values)
            integers.emplace_back(type.getElementTypeBitWidth(), value.offset,
                                  true);
          OpBuilder builder(op);
          Value constant = builder.create<arith::ConstantOp>(
              op->getLoc(), DenseElementsAttr::get(type, integers));
          op->getResult(0).replaceAllUsesWith(constant);
          erase.push_back(op);
        });
        for (Operation *op : llvm::reverse(erase))
          if (op->use_empty())
            op->erase();
      }
      return success();
    };
    unsigned operationCount = 0;
    module.walk([&](Operation *) { ++operationCount; });
    for (unsigned iteration = 0; iteration < operationCount; ++iteration) {
      bool changed = false;
      module.walk([&](Operation *op) {
        if (!op->getName().getStringRef().starts_with("onnx."))
          return;
        if (inferRmsNorm(op)) {
          changed = true;
          return;
        }
        if (inferMatMulNBits(op)) {
          changed = true;
          return;
        }
        auto inference = dyn_cast<ResultTypeInferenceOpInterface>(op);
        if (inference) {
          std::vector<Type> inferred = inference.resultTypeInference();
          if (inferred.size() == op->getNumResults())
            for (auto [result, type] : llvm::zip(op->getResults(), inferred)) {
              if (!isBetterShapedType(result.getType(), type))
                continue;
              result.setType(type);
              changed = true;
            }
        }
        if (auto type = inferGenericResultType(op)) {
          OpResult result = op->getResult(0);
          if (isBetterShapedType(result.getType(), *type)) {
            result.setType(*type);
            changed = true;
          }
        }
      });
      if (bounded && inferSymbolicShapes())
        changed = true;
      if (!changed) {
        materializeShapeValues();
        if (bounded && failed(materializeSymbolicValues()))
          return signalPassFailure();
        updateFunctionResults();
        return;
      }
    }
    module.emitError("ONNX shape propagation did not reach a fixed point");
    signalPassFailure();
  }
};

static PassRegistration<InferONNXShapesPass> registration;
} // namespace
