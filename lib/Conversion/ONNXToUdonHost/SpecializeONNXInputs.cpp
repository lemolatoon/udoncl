#include "VRC/Conversion/Passes.h"

#include "ONNXDimensionExprs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_SPECIALIZEONNXINPUTSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
LogicalResult parseNonnegativeInteger(StringRef text, int64_t &value) {
  if (text.empty() || text.getAsInteger(10, value) || value < 0)
    return failure();
  return success();
}

LogicalResult parseDimensions(StringRef text,
                              llvm::StringMap<int64_t> &dimensions) {
  SmallVector<StringRef> entries;
  text.split(entries, ';', -1, false);
  for (StringRef entry : entries) {
    auto [name, valueText] = entry.split('=');
    int64_t value = 0;
    if (name.empty() || valueText.empty() || valueText.contains('=') ||
        failed(parseNonnegativeInteger(valueText, value)) ||
        !dimensions.try_emplace(name, value).second)
      return failure();
  }
  return success();
}

LogicalResult
parseInputShapes(StringRef text,
                 llvm::StringMap<SmallVector<int64_t>> &inputShapes) {
  SmallVector<StringRef> entries;
  text.split(entries, ';', -1, false);
  for (StringRef entry : entries) {
    auto [name, shapeText] = entry.split('=');
    if (name.empty() || shapeText.empty() || shapeText.contains('=') ||
        inputShapes.contains(name))
      return failure();
    SmallVector<StringRef> dimensions;
    shapeText.split(dimensions, ',', -1, false);
    SmallVector<int64_t> shape;
    for (StringRef dimensionText : dimensions) {
      int64_t dimension = 0;
      if (failed(parseNonnegativeInteger(dimensionText, dimension)))
        return failure();
      shape.push_back(dimension);
    }
    if (shape.empty())
      return failure();
    inputShapes.try_emplace(name, std::move(shape));
  }
  return success();
}

/// A model dimension expression `name` or `name + offset`.
struct DimensionExpression {
  StringRef name;
  int64_t offset = 0;
};

std::optional<DimensionExpression> parseDimensionExpression(StringRef text) {
  text = text.trim();
  auto [name, offsetText] = text.split('+');
  name = name.trim();
  if (name.empty() || name.contains(' '))
    return std::nullopt;
  DimensionExpression expression{name, 0};
  if (!text.contains('+'))
    return expression;
  if (offsetText.contains('+') ||
      failed(parseNonnegativeInteger(offsetText.trim(), expression.offset)))
    return std::nullopt;
  return expression;
}

struct SpecializeONNXInputsPass
    : impl::SpecializeONNXInputsPassBase<SpecializeONNXInputsPass> {
  SpecializeONNXInputsPass() = default;
  explicit SpecializeONNXInputsPass(SpecializeONNXInputsPassOptions options)
      : Base(std::move(options)) {}
  SpecializeONNXInputsPass(const SpecializeONNXInputsPass &other)
      : Base(other) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    Builder builder(context);
    llvm::StringMap<int64_t> dimensions;
    llvm::StringMap<int64_t> bounds;
    llvm::StringMap<SmallVector<int64_t>> inputShapes;
    if (failed(parseDimensions(dimensionBindings, dimensions)) ||
        failed(parseDimensions(dimensionBoundBindings, bounds)) ||
        failed(parseInputShapes(inputShapeBindings, inputShapes))) {
      module.emitError("invalid ONNX input specialization bindings");
      return signalPassFailure();
    }
    for (const auto &bound : bounds) {
      if (dimensions.contains(bound.getKey())) {
        module.emitError("ONNX dimension '")
            << bound.getKey() << "' is both fixed and bounded";
        return signalPassFailure();
      }
      if (bound.getValue() <= 0) {
        module.emitError("bounded ONNX dimension '")
            << bound.getKey() << "' needs a positive bound";
        return signalPassFailure();
      }
    }

    llvm::StringSet<> usedBounds;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      SmallVector<Type> inputs(function.getArgumentTypes());
      bool changed = false;
      for (unsigned index = 0; index < function.getNumArguments(); ++index) {
        auto tensor = dyn_cast<RankedTensorType>(inputs[index]);
        auto publicName =
            function.getArgAttrOfType<StringAttr>(index, "onnx.name");
        auto parameters =
            function.getArgAttrOfType<StringAttr>(index, "onnx.dim_params");
        if (!tensor)
          continue;
        std::string inputName =
            publicName ? publicName.str() : ("input " + Twine(index)).str();
        SmallVector<int64_t> shape(tensor.getShape());
        SmallVector<Attribute> expressions;
        auto override = publicName ? inputShapes.find(publicName.getValue())
                                   : inputShapes.end();
        if (override != inputShapes.end()) {
          if (override->second.size() != shape.size()) {
            function.emitError("ONNX input shape rank mismatch for '")
                << publicName.getValue() << "'";
            return signalPassFailure();
          }
          shape.assign(override->second.begin(), override->second.end());
        } else if (parameters) {
          SmallVector<StringRef> entries;
          parameters.getValue().split(entries, ',');
          for (StringRef entry : entries) {
            auto [axisText, expressionText] = entry.split(':');
            unsigned axis = 0;
            if (axisText.getAsInteger(10, axis) || axis >= shape.size()) {
              function.emitError("invalid onnx.dim_params entry '")
                  << entry << "'";
              return signalPassFailure();
            }
            std::optional<DimensionExpression> expression =
                parseDimensionExpression(expressionText);
            if (!expression) {
              function.emitError("ONNX dimension expression '")
                  << expressionText.trim() << "' of input '" << inputName
                  << "' axis " << axis
                  << " is not supported; only 'name' and 'name + N' are";
              return signalPassFailure();
            }
            if (auto fixed = dimensions.find(expression->name);
                fixed != dimensions.end()) {
              if (fixed->second >
                  std::numeric_limits<int64_t>::max() - expression->offset) {
                function.emitError("ONNX dimension expression '")
                    << expressionText.trim() << "' overflows";
                return signalPassFailure();
              }
              shape[axis] = fixed->second + expression->offset;
              continue;
            }
            if (!bounds.contains(expression->name)) {
              function.emitError("ONNX dimension '")
                  << expression->name << "' of input '" << inputName
                  << "' axis " << axis
                  << " is neither fixed (DIMENSIONS) nor bounded "
                     "(DIMENSION_BOUNDS)";
              return signalPassFailure();
            }
            usedBounds.insert(expression->name);
            shape[axis] = ShapedType::kDynamic;
            expressions.push_back(makeONNXDimensionExprAttr(
                context, axis, expression->name, expression->offset));
          }
        }
        // Every dynamic axis left must be a bounded dimension, and at most
        // one per input, so that a tensor has one run-time extent.
        for (auto [axis, extent] : llvm::enumerate(shape)) {
          if (!ShapedType::isDynamic(extent))
            continue;
          bool named = llvm::any_of(expressions, [&](Attribute attribute) {
            return cast<DictionaryAttr>(attribute)
                       .getAs<IntegerAttr>("axis")
                       .getInt() == static_cast<int64_t>(axis);
          });
          if (!named) {
            function.emitError("ONNX input '")
                << inputName << "' axis " << axis
                << " has a dynamic extent without a dimension name; fix it "
                   "with INPUT_SHAPES";
            return signalPassFailure();
          }
        }
        if (expressions.size() > 1) {
          function.emitError("ONNX input '")
              << inputName
              << "' has more than one bounded dynamic axis; at most one is "
                 "supported";
          return signalPassFailure();
        }
        if (!expressions.empty())
          function.setArgAttr(index, onnxDimensionExprsAttr,
                              ArrayAttr::get(context, expressions));
        Type specialized = RankedTensorType::get(shape, tensor.getElementType(),
                                                 tensor.getEncoding());
        if (specialized == inputs[index])
          continue;
        inputs[index] = specialized;
        function.getArgument(index).setType(specialized);
        changed = true;
      }
      if (changed)
        function.setType(FunctionType::get(function.getContext(), inputs,
                                           function.getResultTypes()));
    }

    if (bounds.empty())
      return;
    SmallVector<NamedAttribute> boundAttributes;
    for (const auto &bound : bounds) {
      if (!usedBounds.contains(bound.getKey())) {
        module.emitError("bounded ONNX dimension '")
            << bound.getKey() << "' is not a dimension of any model input";
        return signalPassFailure();
      }
      boundAttributes.push_back(builder.getNamedAttr(
          bound.getKey(), builder.getI64IntegerAttr(bound.getValue())));
    }
    module->setAttr(onnxDimensionBoundsAttr,
                    builder.getDictionaryAttr(boundAttributes));
  }
};
} // namespace
} // namespace vrc
