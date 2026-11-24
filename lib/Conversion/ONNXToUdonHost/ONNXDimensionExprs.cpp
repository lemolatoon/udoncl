#include "ONNXDimensionExprs.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace vrc {
namespace {
constexpr size_t maximumIntegerElements = 64;

bool isNoValue(Value value) {
  Operation *definition = value.getDefiningOp();
  return definition && definition->getName().getStringRef() == "onnx.NoValue";
}

std::optional<SmallVector<int64_t>> readDense(Operation *op) {
  auto dense = op->getAttrOfType<DenseElementsAttr>("value");
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

bool overflows(int64_t left, int64_t right, bool subtract) {
  APInt leftValue(64, left, true);
  APInt rightValue(64, right, true);
  bool overflow = false;
  if (subtract)
    (void)leftValue.ssub_ov(rightValue, overflow);
  else
    (void)leftValue.sadd_ov(rightValue, overflow);
  return overflow;
}

int64_t normalizeAxis(int64_t axis, int64_t rank) {
  return axis < 0 ? axis + rank : axis;
}
} // namespace

std::string DimExpr::str() const {
  if (!known)
    return "?";
  if (!symbol)
    return std::to_string(offset);
  if (!offset)
    return symbol.str();
  return symbol.str() + (offset < 0 ? " - " : " + ") +
         std::to_string(offset < 0 ? -offset : offset);
}

ONNXDimensionExprs::ONNXDimensionExprs(func::FuncOp function,
                                       ONNXConstantReader reader)
    : reader(std::move(reader)) {
  if (auto module = function->getParentOfType<ModuleOp>())
    if (auto attribute =
            module->getAttrOfType<DictionaryAttr>(onnxDimensionBoundsAttr))
      for (NamedAttribute entry : attribute)
        if (auto bound = dyn_cast<IntegerAttr>(entry.getValue()))
          bounds[entry.getName().getValue()] = bound.getInt();
  if (function.isExternal())
    return;
  for (BlockArgument argument : function.getArguments()) {
    auto type = dyn_cast<RankedTensorType>(argument.getType());
    if (!type)
      continue;
    DimShape shape;
    for (int64_t extent : type.getShape())
      shape.push_back(ShapedType::isDynamic(extent)
                          ? DimExpr::unknown()
                          : DimExpr::constant(extent));
    if (auto expressions = function.getArgAttrOfType<ArrayAttr>(
            argument.getArgNumber(), onnxDimensionExprsAttr)) {
      for (Attribute attribute : expressions) {
        auto entry = dyn_cast<DictionaryAttr>(attribute);
        auto axis = entry ? entry.getAs<IntegerAttr>("axis") : IntegerAttr();
        auto offset =
            entry ? entry.getAs<IntegerAttr>("offset") : IntegerAttr();
        auto symbol = entry ? entry.getAs<StringAttr>("symbol") : StringAttr();
        if (!axis || !offset || !symbol || axis.getInt() < 0 ||
            axis.getInt() >= type.getRank() ||
            !bounds.contains(symbol.getValue()))
          continue;
        shape[axis.getInt()] = DimExpr::of(symbol, offset.getInt());
      }
    }
    shapes[argument] = std::move(shape);
  }
  function.walk([&](Operation *op) {
    if (op != function.getOperation())
      analyze(op);
  });
}

const DimShape *ONNXDimensionExprs::getShape(Value value) const {
  auto found = shapes.find(value);
  return found == shapes.end() ? nullptr : &found->second;
}

const DimShape *ONNXDimensionExprs::getIntegers(Value value) const {
  auto found = integers.find(value);
  return found == integers.end() ? nullptr : &found->second;
}

std::optional<int64_t> ONNXDimensionExprs::getBound(StringAttr symbol) const {
  auto found = bounds.find(symbol.getValue());
  if (found == bounds.end())
    return std::nullopt;
  return found->second;
}

int64_t ONNXDimensionExprs::minimum(const DimExpr &expression) const {
  return expression.offset;
}

int64_t ONNXDimensionExprs::maximum(const DimExpr &expression) const {
  if (!expression.symbol)
    return expression.offset;
  return expression.offset + getBound(expression.symbol).value_or(0);
}

RankedTensorType ONNXDimensionExprs::typeOf(const DimShape &shape,
                                            Type element) {
  SmallVector<int64_t> extents;
  for (const DimExpr &extent : shape)
    extents.push_back(extent.isConstant() ? extent.offset
                                          : ShapedType::kDynamic);
  return RankedTensorType::get(extents, element);
}

void ONNXDimensionExprs::error(Operation *op, const Twine &message) {
  if (!op)
    return;
  for (const auto &[old, text] : errors)
    if (old == op)
      return;
  errors.emplace_back(op, message.str());
}

DimExpr ONNXDimensionExprs::add(const DimExpr &left,
                                const DimExpr &right) const {
  if (!left.known || !right.known || (left.symbol && right.symbol) ||
      overflows(left.offset, right.offset, false))
    return DimExpr::unknown();
  return DimExpr::of(left.symbol ? left.symbol : right.symbol,
                     left.offset + right.offset);
}

DimExpr ONNXDimensionExprs::subtract(const DimExpr &left,
                                     const DimExpr &right) const {
  if (!left.known || !right.known ||
      (right.symbol && right.symbol != left.symbol) ||
      overflows(left.offset, right.offset, true))
    return DimExpr::unknown();
  if (right.symbol)
    return DimExpr::constant(left.offset - right.offset);
  return DimExpr::of(left.symbol, left.offset - right.offset);
}

DimExpr ONNXDimensionExprs::multiply(const DimExpr &left,
                                     const DimExpr &right) const {
  if (!left.known || !right.known)
    return DimExpr::unknown();
  if (left.symbol && right.symbol) {
    const_cast<ONNXDimensionExprs *>(this)->error(
        current, "computes a product of bounded dimensions (" + left.str() +
                     ") * (" + right.str() + ")");
    return DimExpr::unknown();
  }
  if (left.isConstant(1))
    return right;
  if (right.isConstant(1))
    return left;
  if (left.isConstant(0) || right.isConstant(0))
    return DimExpr::constant(0);
  if (left.symbol || right.symbol)
    return DimExpr::unknown();
  APInt value = APInt(64, left.offset, true);
  bool overflow = false;
  value = value.smul_ov(APInt(64, right.offset, true), overflow);
  if (overflow)
    return DimExpr::unknown();
  return DimExpr::constant(value.getSExtValue());
}

DimExpr ONNXDimensionExprs::divide(const DimExpr &left,
                                   const DimExpr &right) const {
  if (!left.known || !right.isConstant() || right.offset == 0)
    return DimExpr::unknown();
  if (right.offset == 1)
    return left;
  if (left.symbol || (left.offset == std::numeric_limits<int64_t>::min() &&
                      right.offset == -1))
    return DimExpr::unknown();
  return DimExpr::constant(left.offset / right.offset);
}

std::optional<int> ONNXDimensionExprs::compare(const DimExpr &left,
                                               const DimExpr &right) const {
  if (!left.known || !right.known)
    return std::nullopt;
  if (left.symbol == right.symbol)
    return left.offset < right.offset ? -1 : (left.offset > right.offset);
  if (maximum(left) < minimum(right))
    return -1;
  if (minimum(left) > maximum(right))
    return 1;
  return std::nullopt;
}

std::optional<DimShape> ONNXDimensionExprs::shapeOf(Value value) const {
  if (const DimShape *shape = getShape(value))
    return *shape;
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return std::nullopt;
  DimShape shape;
  for (int64_t extent : type.getShape())
    shape.push_back(ShapedType::isDynamic(extent) ? DimExpr::unknown()
                                                  : DimExpr::constant(extent));
  return shape;
}

std::optional<DimShape> ONNXDimensionExprs::integersOf(Value value) const {
  if (const DimShape *values = getIntegers(value))
    return *values;
  return std::nullopt;
}

std::optional<DimShape> ONNXDimensionExprs::broadcast(Operation *op,
                                                      ValueRange operands) {
  DimShape shape;
  for (Value operand : operands) {
    if (isNoValue(operand))
      continue;
    std::optional<DimShape> next = shapeOf(operand);
    if (!next)
      return std::nullopt;
    size_t rank = std::max(shape.size(), next->size());
    DimShape merged(rank, DimExpr::constant(1));
    for (size_t offset = 0; offset < rank; ++offset) {
      DimExpr left = offset < shape.size() ? shape[shape.size() - 1 - offset]
                                           : DimExpr::constant(1);
      DimExpr right = offset < next->size() ? (*next)[next->size() - 1 - offset]
                                            : DimExpr::constant(1);
      DimExpr extent;
      if (left.isConstant(1))
        extent = right;
      else if (right.isConstant(1) || left == right)
        extent = left;
      else if (!left.known)
        extent = right.isSymbolic() ? DimExpr::unknown() : right;
      else if (!right.known)
        extent = left.isSymbolic() ? DimExpr::unknown() : left;
      else
        return std::nullopt;
      merged[rank - 1 - offset] = extent;
    }
    shape = std::move(merged);
  }
  return shape;
}

std::optional<DimExpr> ONNXDimensionExprs::sliceExtent(const DimExpr &dimension,
                                                       const DimExpr &start,
                                                       const DimExpr &end,
                                                       int64_t step) {
  if (!dimension.known || !start.known || !end.known || step == 0)
    return std::nullopt;
  // Resolve a bound that may be negative (counted from the end) and clamp it
  // to [0, dimension]; this needs the sign and the comparisons to hold for
  // every extent.
  auto clamp = [&](DimExpr bound) -> std::optional<DimExpr> {
    if (bound.isConstant() && bound.offset < 0) {
      bound = add(dimension, bound);
      if (!bound.known)
        return std::nullopt;
      if (maximum(bound) <= 0)
        return DimExpr::constant(0);
      if (minimum(bound) < 0)
        return std::nullopt;
    } else if (bound.isSymbolic() && minimum(bound) < 0) {
      return std::nullopt;
    }
    std::optional<int> order = compare(bound, dimension);
    if (!order)
      return std::nullopt;
    return *order >= 0 ? dimension : bound;
  };
  if (step < 0) {
    // A reversed slice keeps the static rule of the ONNX specification.
    if (!dimension.isConstant() || !start.isConstant() || !end.isConstant())
      return std::nullopt;
    int64_t size = dimension.offset;
    int64_t first = start.offset < 0
                        ? (start.offset < -size ? -1 : start.offset + size)
                        : std::min(start.offset, size - 1);
    int64_t last =
        end.offset == std::numeric_limits<int64_t>::min()
            ? -1
            : (end.offset < 0 ? (end.offset < -size ? -1 : end.offset + size)
                              : std::min(end.offset, size - 1));
    if (first <= last)
      return DimExpr::constant(0);
    return DimExpr::constant((first - last + (-step) - 1) / (-step));
  }
  std::optional<DimExpr> first = clamp(start);
  std::optional<DimExpr> last = clamp(end);
  if (!first || !last)
    return std::nullopt;
  std::optional<int> order = compare(*first, *last);
  if (!order)
    return std::nullopt;
  if (*order >= 0)
    return DimExpr::constant(0);
  DimExpr length = subtract(*last, *first);
  if (!length.known)
    return std::nullopt;
  if (step == 1)
    return length;
  if (!length.isConstant())
    return std::nullopt;
  return DimExpr::constant((length.offset + step - 1) / step);
}

void ONNXDimensionExprs::analyze(Operation *op) {
  current = op;
  if (op->getNumResults() != 1) {
    // MatMulNBits-like operators have one result; the others keep their
    // types.
    for (Value result : op->getResults())
      if (std::optional<DimShape> shape = shapeOf(result))
        shapes[result] = *shape;
    return;
  }
  Value result = op->getResult(0);
  std::optional<DimShape> shape = inferShape(op);
  std::optional<DimShape> fromType = shapeOf(result);
  if (shape && fromType && shape->size() == fromType->size()) {
    // A static extent of the type always wins over an inferred one.
    for (auto [inferred, typed] : llvm::zip_equal(*shape, *fromType))
      if (typed.isConstant())
        inferred = typed;
  } else if (!shape || (fromType && shape->size() != fromType->size())) {
    shape = fromType;
  }
  if (shape) {
    unsigned dynamicAxes = llvm::count_if(
        *shape, [](const DimExpr &extent) { return extent.isSymbolic(); });
    if (dynamicAxes > 1) {
      std::string text;
      for (const DimExpr &extent : *shape)
        text += (text.empty() ? "" : ", ") + extent.str();
      error(op, "has more than one bounded dynamic axis: [" + text + "]");
    }
    shapes[result] = *shape;
  }
  if (auto type = dyn_cast<RankedTensorType>(result.getType()))
    if (isa<IntegerType>(type.getElementType()))
      if (std::optional<DimShape> values = inferIntegers(op))
        if (values->size() <= maximumIntegerElements)
          integers[result] = std::move(*values);
  current = nullptr;
}

std::optional<DimShape> ONNXDimensionExprs::inferIntegers(Operation *op) {
  StringRef name = op->getName().getStringRef();
  Value result = op->getResult(0);
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (auto call = dyn_cast<func::CallOp>(op)) {
    // An externalized constant keeps its value in its static tensor record.
    auto module = op->getParentOfType<ModuleOp>();
    auto callee = module ? module.lookupSymbol<func::FuncOp>(call.getCallee())
                         : func::FuncOp();
    auto record =
        callee ? callee->getAttrOfType<DictionaryAttr>("udoncl.static_tensor")
               : DictionaryAttr();
    auto data =
        record ? record.getAs<DenseElementsAttr>("data") : DenseElementsAttr();
    if (!data || !isa<IntegerType>(data.getElementType()) ||
        data.getNumElements() > static_cast<int64_t>(maximumIntegerElements))
      return std::nullopt;
    DimShape result;
    for (APInt value : data.getValues<APInt>())
      result.push_back(DimExpr::constant(value.getSExtValue()));
    return result;
  }
  if (name == "onnx.Constant" || name == "arith.constant") {
    std::optional<SmallVector<int64_t>> values = readDense(op);
    if (!values && reader)
      values = reader(op);
    if (!values || values->size() > maximumIntegerElements)
      return std::nullopt;
    DimShape result;
    for (int64_t value : *values)
      result.push_back(DimExpr::constant(value));
    return result;
  }
  if (name == "onnx.Dim" && op->getNumOperands() == 1) {
    std::optional<DimShape> input = shapeOf(op->getOperand(0));
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    if (!input || !axis)
      return std::nullopt;
    int64_t index = normalizeAxis(axis.getInt(), input->size());
    if (index < 0 || index >= static_cast<int64_t>(input->size()) ||
        !(*input)[index].known)
      return std::nullopt;
    return DimShape{(*input)[index]};
  }
  if (name == "onnx.Shape" && op->getNumOperands() == 1) {
    std::optional<DimShape> input = shapeOf(op->getOperand(0));
    if (!input)
      return std::nullopt;
    int64_t rank = input->size();
    int64_t start = 0;
    int64_t end = rank;
    if (auto attribute = op->getAttrOfType<IntegerAttr>("start"))
      start = attribute.getInt();
    if (auto attribute = op->getAttrOfType<IntegerAttr>("end"))
      end = attribute.getInt();
    start = std::clamp<int64_t>(normalizeAxis(start, rank), 0, rank);
    end = std::clamp<int64_t>(normalizeAxis(end, rank), start, rank);
    DimShape values(input->begin() + start, input->begin() + end);
    if (llvm::any_of(values, [](const DimExpr &value) { return !value.known; }))
      return std::nullopt;
    return values;
  }
  if (name == "onnx.Gather" && op->getNumOperands() == 2) {
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    if (axis && axis.getInt() != 0)
      return std::nullopt;
    std::optional<DimShape> data = integersOf(op->getOperand(0));
    std::optional<DimShape> indices = integersOf(op->getOperand(1));
    if (!data || !indices)
      return std::nullopt;
    DimShape values;
    for (const DimExpr &index : *indices) {
      if (!index.isConstant())
        return std::nullopt;
      int64_t position =
          index.offset < 0 ? index.offset + data->size() : index.offset;
      if (position < 0 || position >= static_cast<int64_t>(data->size()))
        return std::nullopt;
      values.push_back((*data)[position]);
    }
    return values;
  }
  bool arithmetic = llvm::StringSwitch<bool>(name)
                        .Cases({"onnx.Add", "onnx.Sub", "onnx.Mul"}, true)
                        .Cases({"onnx.Div", "onnx.Max", "onnx.Min"}, true)
                        .Cases({"onnx.Equal", "onnx.Greater"}, true)
                        .Cases({"onnx.Less", "onnx.GreaterOrEqual"}, true)
                        .Case("onnx.LessOrEqual", true)
                        .Default(false);
  if (arithmetic && op->getNumOperands() == 2) {
    std::optional<DimShape> left = integersOf(op->getOperand(0));
    std::optional<DimShape> right = integersOf(op->getOperand(1));
    if (!left || !right || left->empty() || right->empty())
      return std::nullopt;
    size_t count = std::max(left->size(), right->size());
    if ((left->size() != 1 && left->size() != count) ||
        (right->size() != 1 && right->size() != count))
      return std::nullopt;
    DimShape values;
    for (size_t index = 0; index < count; ++index) {
      const DimExpr &lhs = (*left)[left->size() == 1 ? 0 : index];
      const DimExpr &rhs = (*right)[right->size() == 1 ? 0 : index];
      DimExpr value;
      if (name == "onnx.Add")
        value = add(lhs, rhs);
      else if (name == "onnx.Sub")
        value = subtract(lhs, rhs);
      else if (name == "onnx.Mul")
        value = multiply(lhs, rhs);
      else if (name == "onnx.Div")
        value = divide(lhs, rhs);
      else {
        std::optional<int> order = compare(lhs, rhs);
        if (!order)
          return std::nullopt;
        bool truth = llvm::StringSwitch<bool>(name)
                         .Case("onnx.Equal", *order == 0)
                         .Case("onnx.Greater", *order > 0)
                         .Case("onnx.Less", *order < 0)
                         .Case("onnx.GreaterOrEqual", *order >= 0)
                         .Case("onnx.LessOrEqual", *order <= 0)
                         .Case("onnx.Max", false)
                         .Default(false);
        if (name == "onnx.Max")
          value = *order >= 0 ? lhs : rhs;
        else if (name == "onnx.Min")
          value = *order <= 0 ? lhs : rhs;
        else
          value = DimExpr::constant(truth);
      }
      if (!value.known)
        return std::nullopt;
      values.push_back(value);
    }
    return values;
  }
  if (name == "onnx.Where" && op->getNumOperands() == 3) {
    std::optional<DimShape> condition = integersOf(op->getOperand(0));
    std::optional<DimShape> whenTrue = integersOf(op->getOperand(1));
    std::optional<DimShape> whenFalse = integersOf(op->getOperand(2));
    if (!condition || !whenTrue || !whenFalse || !resultType ||
        !resultType.hasStaticShape())
      return std::nullopt;
    size_t count = resultType.getNumElements();
    auto compatible = [&](const DimShape &values) {
      return values.size() == 1 || values.size() == count;
    };
    if (!compatible(*condition) || !compatible(*whenTrue) ||
        !compatible(*whenFalse))
      return std::nullopt;
    DimShape values;
    for (size_t index = 0; index < count; ++index) {
      const DimExpr &selector =
          (*condition)[condition->size() == 1 ? 0 : index];
      if (!selector.isConstant())
        return std::nullopt;
      const DimShape &chosen = selector.offset ? *whenTrue : *whenFalse;
      values.push_back(chosen[chosen.size() == 1 ? 0 : index]);
    }
    return values;
  }
  if (name == "onnx.Range" && op->getNumOperands() == 3) {
    std::optional<DimShape> starts = integersOf(op->getOperand(0));
    std::optional<DimShape> limits = integersOf(op->getOperand(1));
    std::optional<DimShape> deltas = integersOf(op->getOperand(2));
    if (!starts || starts->size() != 1 || !limits || limits->size() != 1 ||
        !deltas || deltas->size() != 1 || !(*deltas)[0].isConstant() ||
        (*deltas)[0].offset == 0)
      return std::nullopt;
    DimExpr length = subtract((*limits)[0], (*starts)[0]);
    int64_t delta = (*deltas)[0].offset;
    if (!length.isConstant())
      return std::nullopt;
    int64_t count = delta > 0 ? (length.offset + delta - 1) / delta
                              : (-length.offset - delta - 1) / -delta;
    if (count < 0)
      count = 0;
    if (count > static_cast<int64_t>(maximumIntegerElements))
      return std::nullopt;
    DimShape values;
    for (int64_t index = 0; index < count; ++index) {
      DimExpr value = add((*starts)[0], DimExpr::constant(index * delta));
      if (!value.known)
        return std::nullopt;
      values.push_back(value);
    }
    return values;
  }
  if (name == "onnx.Slice" && op->getNumOperands() >= 3) {
    std::optional<DimShape> data = integersOf(op->getOperand(0));
    std::optional<DimShape> starts = integersOf(op->getOperand(1));
    std::optional<DimShape> ends = integersOf(op->getOperand(2));
    auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!data || !starts || !ends || !type || type.getRank() != 1 ||
        starts->size() != 1 || ends->size() != 1 ||
        !(*starts)[0].isConstant() || !(*ends)[0].isConstant())
      return std::nullopt;
    int64_t step = 1;
    if (op->getNumOperands() >= 4 && !isNoValue(op->getOperand(3))) {
      std::optional<DimShape> axes = integersOf(op->getOperand(3));
      if (!axes || axes->size() != 1 || !(*axes)[0].isConstant() ||
          normalizeAxis((*axes)[0].offset, 1) != 0)
        return std::nullopt;
    }
    if (op->getNumOperands() >= 5 && !isNoValue(op->getOperand(4))) {
      std::optional<DimShape> steps = integersOf(op->getOperand(4));
      if (!steps || steps->size() != 1 || !(*steps)[0].isConstant())
        return std::nullopt;
      step = (*steps)[0].offset;
    }
    if (step <= 0)
      return std::nullopt;
    int64_t size = data->size();
    auto resolve = [&](int64_t bound) {
      if (bound < 0)
        bound = bound < -size ? 0 : bound + size;
      return std::min(bound, size);
    };
    DimShape values;
    for (int64_t index = resolve((*starts)[0].offset),
                 end = resolve((*ends)[0].offset);
         index < end; index += step)
      values.push_back((*data)[index]);
    return values;
  }
  if (llvm::StringSwitch<bool>(name)
          .Cases({"onnx.Unsqueeze", "onnx.Squeeze", "onnx.Reshape"}, true)
          .Cases({"onnx.Identity", "onnx.Cast", "onnx.Flatten"}, true)
          .Default(false) &&
      op->getNumOperands() >= 1) {
    auto input = dyn_cast<IntegerType>(
        getElementTypeOrSelf(op->getOperand(0).getType()));
    if (!input)
      return std::nullopt;
    return integersOf(op->getOperand(0));
  }
  if (name == "onnx.Concat") {
    int64_t axis = 0;
    if (auto attribute = op->getAttrOfType<IntegerAttr>("axis"))
      axis = attribute.getInt();
    if (resultType)
      axis = normalizeAxis(axis, resultType.getRank());
    if (axis != 0 || !resultType || resultType.getRank() != 1)
      return std::nullopt;
    DimShape values;
    for (Value operand : op->getOperands()) {
      std::optional<DimShape> part = integersOf(operand);
      if (!part)
        return std::nullopt;
      llvm::append_range(values, *part);
    }
    return values;
  }
  if (name == "onnx.Expand" && op->getNumOperands() == 2 && resultType &&
      resultType.hasStaticShape()) {
    std::optional<DimShape> values = integersOf(op->getOperand(0));
    if (!values || values->size() != 1 ||
        resultType.getNumElements() >
            static_cast<int64_t>(maximumIntegerElements))
      return std::nullopt;
    return DimShape(resultType.getNumElements(), (*values)[0]);
  }
  if (name == "onnx.ConstantOfShape" && resultType &&
      resultType.hasStaticShape() &&
      resultType.getNumElements() <=
          static_cast<int64_t>(maximumIntegerElements)) {
    auto value = op->getAttrOfType<DenseElementsAttr>("value");
    if (!value || !value.isSplat() || !isa<IntegerType>(value.getElementType()))
      return std::nullopt;
    return DimShape(
        resultType.getNumElements(),
        DimExpr::constant(value.getSplatValue<APInt>().getSExtValue()));
  }
  return std::nullopt;
}

std::optional<DimShape> ONNXDimensionExprs::inferShape(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (!name.starts_with("onnx."))
    return std::nullopt;
  StringRef operation = name.drop_front(5);
  auto operandShape = [&](unsigned index) -> std::optional<DimShape> {
    if (index >= op->getNumOperands() || isNoValue(op->getOperand(index)))
      return std::nullopt;
    return shapeOf(op->getOperand(index));
  };
  if (operation == "Custom") {
    auto function = op->getAttrOfType<StringAttr>("function_name");
    std::optional<DimShape> input = operandShape(0);
    if (!function || !input)
      return std::nullopt;
    if (function.getValue() == "MatMulNBits") {
      auto columns = op->getAttrOfType<IntegerAttr>("N");
      if (!columns || input->empty())
        return std::nullopt;
      input->back() = DimExpr::constant(columns.getInt());
      return input;
    }
    if (function.getValue() == "RmsNorm")
      return input;
    if (function.getValue() == "UdonclBoundedPrefix") {
      // The leading block of a constant at the bound: its extent on `axis`
      // is `symbol + offset`.
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      auto symbol = op->getAttrOfType<StringAttr>("symbol");
      auto offset = op->getAttrOfType<IntegerAttr>("offset");
      if (!axis || !symbol || !offset || axis.getInt() < 0 ||
          axis.getInt() >= static_cast<int64_t>(input->size()) ||
          !getBound(symbol))
        return std::nullopt;
      (*input)[axis.getInt()] = DimExpr::of(symbol, offset.getInt());
      return input;
    }
    return std::nullopt;
  }
  if (llvm::StringSwitch<bool>(operation)
          .Cases({"Add", "Sub", "Mul", "Div", "Pow", "Equal"}, true)
          .Cases({"Greater", "Less", "GreaterOrEqual", "LessOrEqual"}, true)
          .Cases({"Where", "And", "Or", "Max", "Min", "Mod"}, true)
          .Default(false))
    return broadcast(op, op->getOperands());
  if (llvm::StringSwitch<bool>(operation)
          .Cases({"Cast", "Cos", "Identity", "Neg", "Not", "Erf"}, true)
          .Cases({"Relu", "Sigmoid", "Sin", "Softmax", "Sqrt", "Tanh"}, true)
          .Cases({"Exp", "Reciprocal", "Trilu", "ScatterND", "Abs"}, true)
          .Default(false))
    return operandShape(0);
  if (operation == "Shape") {
    std::optional<DimShape> input = operandShape(0);
    if (!input)
      return std::nullopt;
    int64_t rank = input->size();
    int64_t start = 0;
    int64_t end = rank;
    if (auto attribute = op->getAttrOfType<IntegerAttr>("start"))
      start = attribute.getInt();
    if (auto attribute = op->getAttrOfType<IntegerAttr>("end"))
      end = attribute.getInt();
    start = std::clamp<int64_t>(normalizeAxis(start, rank), 0, rank);
    end = std::clamp<int64_t>(normalizeAxis(end, rank), start, rank);
    return DimShape{DimExpr::constant(end - start)};
  }
  if (operation == "Dim")
    return DimShape{DimExpr::constant(1)};
  if (operation == "ConstantOfShape") {
    std::optional<DimShape> values = integersOf(op->getOperand(0));
    if (!values || llvm::any_of(*values, [&](const DimExpr &extent) {
          return !extent.known || minimum(extent) < 0;
        }))
      return std::nullopt;
    return values;
  }
  if (operation == "Range" && op->getNumOperands() == 3) {
    std::optional<DimShape> starts = integersOf(op->getOperand(0));
    std::optional<DimShape> limits = integersOf(op->getOperand(1));
    std::optional<DimShape> deltas = integersOf(op->getOperand(2));
    if (!starts || starts->size() != 1 || !limits || limits->size() != 1 ||
        !deltas || deltas->size() != 1 || !(*deltas)[0].isConstant() ||
        (*deltas)[0].offset == 0)
      return std::nullopt;
    DimExpr length = subtract((*limits)[0], (*starts)[0]);
    int64_t delta = (*deltas)[0].offset;
    if (!length.known)
      return std::nullopt;
    if (length.isConstant()) {
      int64_t count = delta > 0 ? (length.offset + delta - 1) / delta
                                : (-length.offset - delta - 1) / -delta;
      return DimShape{DimExpr::constant(std::max<int64_t>(count, 0))};
    }
    if (delta != 1 || minimum(length) < 0)
      return std::nullopt;
    return DimShape{length};
  }
  if (operation == "MatMul") {
    std::optional<DimShape> left = operandShape(0);
    std::optional<DimShape> right = operandShape(1);
    if (!left || !right || left->size() < 2 || right->size() < 2)
      return std::nullopt;
    if (right->size() == 2) {
      DimShape shape = *left;
      shape.back() = (*right)[1];
      return shape;
    }
    size_t rank = std::max(left->size(), right->size());
    DimShape shape(rank, DimExpr::unknown());
    size_t leftOffset = rank - left->size();
    size_t rightOffset = rank - right->size();
    for (size_t axis = 0; axis + 2 < rank; ++axis) {
      DimExpr lhs =
          axis < leftOffset ? DimExpr::constant(1) : (*left)[axis - leftOffset];
      DimExpr rhs = axis < rightOffset ? DimExpr::constant(1)
                                       : (*right)[axis - rightOffset];
      if (lhs.isConstant(1))
        shape[axis] = rhs;
      else if (rhs.isConstant(1) || lhs == rhs)
        shape[axis] = lhs;
    }
    shape[rank - 2] = (*left)[left->size() - 2];
    shape[rank - 1] = (*right)[right->size() - 1];
    return shape;
  }
  if (operation == "Gemm") {
    std::optional<DimShape> left = operandShape(0);
    std::optional<DimShape> right = operandShape(1);
    if (!left || !right || left->size() != 2 || right->size() != 2)
      return std::nullopt;
    auto transA = op->getAttrOfType<IntegerAttr>("transA");
    auto transB = op->getAttrOfType<IntegerAttr>("transB");
    return DimShape{(*left)[transA && transA.getInt() ? 1 : 0],
                    (*right)[transB && transB.getInt() ? 0 : 1]};
  }
  if (operation == "Gather") {
    std::optional<DimShape> data = operandShape(0);
    std::optional<DimShape> indices = operandShape(1);
    if (!data || !indices)
      return std::nullopt;
    int64_t axis = 0;
    if (auto attribute = op->getAttrOfType<IntegerAttr>("axis"))
      axis = attribute.getInt();
    axis = normalizeAxis(axis, data->size());
    if (axis < 0 || axis >= static_cast<int64_t>(data->size()))
      return std::nullopt;
    DimShape shape(data->begin(), data->begin() + axis);
    llvm::append_range(shape, *indices);
    shape.append(data->begin() + axis + 1, data->end());
    return shape;
  }
  if (operation == "Concat") {
    int64_t axis = 0;
    if (auto attribute = op->getAttrOfType<IntegerAttr>("axis"))
      axis = attribute.getInt();
    std::optional<DimShape> shape;
    for (Value operand : op->getOperands()) {
      std::optional<DimShape> next = shapeOf(operand);
      if (!next)
        return std::nullopt;
      if (!shape) {
        shape = next;
        axis = normalizeAxis(axis, shape->size());
        if (axis < 0 || axis >= static_cast<int64_t>(shape->size()))
          return std::nullopt;
        continue;
      }
      if (next->size() != shape->size())
        return std::nullopt;
      for (size_t dimension = 0; dimension < shape->size(); ++dimension) {
        DimExpr &extent = (*shape)[dimension];
        const DimExpr &other = (*next)[dimension];
        if (static_cast<int64_t>(dimension) == axis)
          extent = add(extent, other);
        else if (!extent.known)
          extent = other;
        else if (other.known && extent != other)
          return std::nullopt;
      }
    }
    return shape;
  }
  if (operation == "Expand" && op->getNumOperands() == 2) {
    std::optional<DimShape> input = operandShape(0);
    std::optional<DimShape> requested = integersOf(op->getOperand(1));
    if (!input || !requested)
      return std::nullopt;
    size_t rank = std::max(input->size(), requested->size());
    DimShape shape(rank, DimExpr::constant(1));
    for (size_t offset = 0; offset < rank; ++offset) {
      DimExpr have = offset < input->size()
                         ? (*input)[input->size() - 1 - offset]
                         : DimExpr::constant(1);
      DimExpr want = offset < requested->size()
                         ? (*requested)[requested->size() - 1 - offset]
                         : DimExpr::constant(1);
      DimExpr extent;
      if (have.isConstant(1))
        extent = want;
      else if (want.isConstant(1) || have == want)
        extent = have;
      else if (!have.known)
        extent = want.isSymbolic() ? DimExpr::unknown() : want;
      else
        return std::nullopt;
      shape[rank - 1 - offset] = extent;
    }
    return shape;
  }
  if (operation == "Reshape" && op->getNumOperands() >= 2) {
    std::optional<DimShape> input = operandShape(0);
    std::optional<DimShape> requested = integersOf(op->getOperand(1));
    if (!input || !requested)
      return std::nullopt;
    auto allowZero = op->getAttrOfType<IntegerAttr>("allowzero");
    bool copyZero = !allowZero || allowZero.getInt() == 0;
    // Element counts are `scale * term`, where term is 1 or one symbolic
    // extent.
    auto count = [&](const DimShape &shape, int64_t skip, int64_t &scale,
                     DimExpr &term) -> bool {
      scale = 1;
      term = DimExpr::constant(1);
      for (auto [index, extent] : llvm::enumerate(shape)) {
        if (static_cast<int64_t>(index) == skip)
          continue;
        if (!extent.known)
          return false;
        if (extent.isConstant()) {
          if (extent.offset && scale > std::numeric_limits<int64_t>::max() /
                                           std::abs(extent.offset))
            return false;
          scale *= extent.offset;
          continue;
        }
        if (!term.isConstant(1)) {
          error(op, "computes a product of bounded dimensions (" + term.str() +
                        ") * (" + extent.str() + ")");
          return false;
        }
        term = extent;
      }
      return true;
    };
    DimShape shape;
    int64_t inferred = -1;
    for (auto [index, extent] : llvm::enumerate(*requested)) {
      DimExpr value = extent;
      if (value.isConstant(0) && copyZero) {
        if (index >= input->size())
          return std::nullopt;
        value = (*input)[index];
      }
      if (value.isConstant(-1)) {
        if (inferred >= 0)
          return std::nullopt;
        inferred = index;
      } else if (!value.known || (value.isConstant() && value.offset < 0)) {
        return std::nullopt;
      }
      shape.push_back(value);
    }
    if (inferred < 0)
      return shape;
    int64_t inputScale, outputScale;
    DimExpr inputTerm, outputTerm;
    if (!count(*input, -1, inputScale, inputTerm) ||
        !count(shape, inferred, outputScale, outputTerm) || outputScale == 0)
      return std::nullopt;
    if (inputTerm == outputTerm) {
      if (inputScale % outputScale)
        return std::nullopt;
      shape[inferred] = DimExpr::constant(inputScale / outputScale);
    } else if (outputTerm.isConstant(1) && inputScale == outputScale) {
      shape[inferred] = inputTerm;
    } else if (outputTerm.isConstant(1) && inputTerm.isConstant(1) &&
               inputScale % outputScale == 0) {
      shape[inferred] = DimExpr::constant(inputScale / outputScale);
    } else {
      shape[inferred] = DimExpr::unknown();
    }
    return shape;
  }
  if (operation == "Unsqueeze" && op->getNumOperands() >= 2) {
    std::optional<DimShape> input = operandShape(0);
    std::optional<DimShape> axes = integersOf(op->getOperand(1));
    if (!input || !axes)
      return std::nullopt;
    int64_t rank = input->size() + axes->size();
    SmallVector<bool> inserted(rank, false);
    for (const DimExpr &axis : *axes) {
      if (!axis.isConstant())
        return std::nullopt;
      int64_t index = normalizeAxis(axis.offset, rank);
      if (index < 0 || index >= rank || inserted[index])
        return std::nullopt;
      inserted[index] = true;
    }
    DimShape shape;
    size_t inputAxis = 0;
    for (int64_t axis = 0; axis < rank; ++axis)
      shape.push_back(inserted[axis] ? DimExpr::constant(1)
                                     : (*input)[inputAxis++]);
    return shape;
  }
  if (operation == "Squeeze") {
    std::optional<DimShape> input = operandShape(0);
    if (!input)
      return std::nullopt;
    SmallVector<bool> removed(input->size(), false);
    if (op->getNumOperands() >= 2 && !isNoValue(op->getOperand(1))) {
      std::optional<DimShape> axes = integersOf(op->getOperand(1));
      if (!axes)
        return std::nullopt;
      for (const DimExpr &axis : *axes) {
        if (!axis.isConstant())
          return std::nullopt;
        int64_t index = normalizeAxis(axis.offset, input->size());
        if (index < 0 || index >= static_cast<int64_t>(input->size()))
          return std::nullopt;
        removed[index] = true;
      }
    } else {
      for (auto [axis, extent] : llvm::enumerate(*input)) {
        if (!extent.known)
          return std::nullopt;
        removed[axis] = extent.isConstant(1);
      }
    }
    DimShape shape;
    for (auto [axis, extent] : llvm::enumerate(*input))
      if (!removed[axis])
        shape.push_back(extent);
    return shape;
  }
  if (operation == "Slice" && op->getNumOperands() >= 3) {
    std::optional<DimShape> input = operandShape(0);
    std::optional<DimShape> starts = integersOf(op->getOperand(1));
    std::optional<DimShape> ends = integersOf(op->getOperand(2));
    if (!input || !starts || !ends || starts->size() != ends->size())
      return std::nullopt;
    SmallVector<int64_t> axes;
    SmallVector<int64_t> steps(starts->size(), 1);
    if (op->getNumOperands() >= 4 && !isNoValue(op->getOperand(3))) {
      std::optional<DimShape> values = integersOf(op->getOperand(3));
      if (!values)
        return std::nullopt;
      for (const DimExpr &value : *values) {
        if (!value.isConstant())
          return std::nullopt;
        axes.push_back(value.offset);
      }
    } else {
      for (size_t index = 0; index < starts->size(); ++index)
        axes.push_back(index);
    }
    if (op->getNumOperands() >= 5 && !isNoValue(op->getOperand(4))) {
      std::optional<DimShape> values = integersOf(op->getOperand(4));
      if (!values)
        return std::nullopt;
      steps.clear();
      for (const DimExpr &value : *values) {
        if (!value.isConstant())
          return std::nullopt;
        steps.push_back(value.offset);
      }
    }
    if (axes.size() != starts->size() || steps.size() != starts->size())
      return std::nullopt;
    DimShape shape = *input;
    for (size_t index = 0; index < starts->size(); ++index) {
      int64_t axis = normalizeAxis(axes[index], input->size());
      if (axis < 0 || axis >= static_cast<int64_t>(input->size()))
        return std::nullopt;
      std::optional<DimExpr> extent = sliceExtent(
          (*input)[axis], (*starts)[index], (*ends)[index], steps[index]);
      shape[axis] = extent ? *extent : DimExpr::unknown();
    }
    return shape;
  }
  if (operation == "Transpose") {
    std::optional<DimShape> input = operandShape(0);
    if (!input)
      return std::nullopt;
    SmallVector<int64_t> permutation;
    if (auto attribute = op->getAttrOfType<ArrayAttr>("perm"))
      for (Attribute value : attribute)
        permutation.push_back(cast<IntegerAttr>(value).getInt());
    else
      for (int64_t axis = input->size(); axis-- > 0;)
        permutation.push_back(axis);
    if (permutation.size() != input->size())
      return std::nullopt;
    DimShape shape;
    for (int64_t axis : permutation) {
      if (axis < 0 || axis >= static_cast<int64_t>(input->size()))
        return std::nullopt;
      shape.push_back((*input)[axis]);
    }
    return shape;
  }
  if (operation == "ReduceMeanV13" || operation == "ReduceMean" ||
      operation == "ReduceSum" || operation == "ReduceMax") {
    std::optional<DimShape> input = operandShape(0);
    if (!input)
      return std::nullopt;
    SmallVector<int64_t> axes;
    if (auto attribute = op->getAttrOfType<DenseI64ArrayAttr>("axes"))
      llvm::append_range(axes, attribute.asArrayRef());
    else if (auto attribute = op->getAttrOfType<ArrayAttr>("axes"))
      for (Attribute value : attribute)
        axes.push_back(cast<IntegerAttr>(value).getInt());
    else if (op->getNumOperands() >= 2 && !isNoValue(op->getOperand(1))) {
      std::optional<DimShape> values = integersOf(op->getOperand(1));
      if (!values)
        return std::nullopt;
      for (const DimExpr &value : *values) {
        if (!value.isConstant())
          return std::nullopt;
        axes.push_back(value.offset);
      }
    }
    if (axes.empty())
      return std::nullopt;
    bool keep = true;
    if (auto attribute = op->getAttrOfType<IntegerAttr>("keepdims"))
      keep = attribute.getInt() != 0;
    SmallVector<bool> reduced(input->size(), false);
    for (int64_t axis : axes) {
      axis = normalizeAxis(axis, input->size());
      if (axis < 0 || axis >= static_cast<int64_t>(input->size()))
        return std::nullopt;
      reduced[axis] = true;
    }
    DimShape shape;
    for (auto [axis, extent] : llvm::enumerate(*input))
      if (!reduced[axis])
        shape.push_back(extent);
      else if (keep)
        shape.push_back(DimExpr::constant(1));
    return shape;
  }
  return std::nullopt;
}
} // namespace vrc
