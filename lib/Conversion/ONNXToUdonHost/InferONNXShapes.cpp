#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
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
  if ((name == "onnx.Unsqueeze" || name == "onnx.Squeeze" ||
       name == "onnx.Identity" || name == "onnx.Cast") &&
      op->getNumOperands() >= 1)
    return evaluateIntegerTensor(op->getOperand(0), depth + 1);
  if (name == "onnx.Concat" &&
      (!op->getAttrOfType<IntegerAttr>("axis") ||
       op->getAttrOfType<IntegerAttr>("axis").getInt() == 0)) {
    SmallVector<int64_t> result;
    for (Value operand : op->getOperands()) {
      auto values = evaluateIntegerTensor(operand, depth + 1);
      if (!values)
        return std::nullopt;
      llvm::append_range(result, *values);
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
    if (data && indices)
      return dynamicTensor(data.getRank() + indices.getRank() - 1);
  }
  if (name == "Concat") {
    int64_t rank = -1;
    int64_t oneDimensionalSize = 0;
    bool staticOneDimensionalSize = true;
    for (Value operand : op->getOperands()) {
      auto tensor = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensor || (rank >= 0 && tensor.getRank() != rank))
        return std::nullopt;
      rank = tensor.getRank();
      if (rank == 1 && !ShapedType::isDynamic(tensor.getDimSize(0)))
        oneDimensionalSize += tensor.getDimSize(0);
      else
        staticOneDimensionalSize = false;
    }
    if (rank == 1 && staticOneDimensionalSize)
      return RankedTensorType::get({oneDimensionalSize},
                                   current.getElementType());
    if (rank >= 0)
      return dynamicTensor(rank);
  }
  if (name == "Reshape" || name == "Expand") {
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
    auto axes = rankedOperand(1);
    if (input && input.getRank() == 0 && axes && axes.hasStaticShape() &&
        axes.getNumElements() == 1)
      return RankedTensorType::get({1}, current.getElementType());
    if (input && axes && axes.hasStaticShape())
      return dynamicTensor(input.getRank() + axes.getNumElements());
  }
  if (name == "Squeeze") {
    auto input = rankedOperand(0);
    auto axes = rankedOperand(1);
    if (input && axes && axes.hasStaticShape())
      return dynamicTensor(input.getRank() - axes.getNumElements());
  }
  if (name == "Transpose" || name == "Slice") {
    if (auto input = rankedOperand(0))
      return dynamicTensor(input.getRank());
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
        auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!type || !type.hasStaticShape() ||
            !isa<IntegerType>(type.getElementType()))
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
      if (!changed) {
        materializeShapeValues();
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
