#include "VRC/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

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

std::optional<int64_t>
evaluateDimension(StringRef expression,
                  const llvm::StringMap<int64_t> &dimensions) {
  expression = expression.trim();
  auto [name, offsetText] = expression.split('+');
  name = name.trim();
  auto found = dimensions.find(name);
  if (name.empty() || found == dimensions.end())
    return std::nullopt;
  if (offsetText.empty())
    return found->second;
  int64_t offset = 0;
  if (offsetText.contains('+') ||
      failed(parseNonnegativeInteger(offsetText.trim(), offset)) ||
      found->second > std::numeric_limits<int64_t>::max() - offset)
    return std::nullopt;
  return found->second + offset;
}

struct SpecializeONNXInputsPass
    : impl::SpecializeONNXInputsPassBase<SpecializeONNXInputsPass> {
  SpecializeONNXInputsPass() = default;
  explicit SpecializeONNXInputsPass(SpecializeONNXInputsPassOptions options)
      : Base(std::move(options)) {}
  SpecializeONNXInputsPass(const SpecializeONNXInputsPass &other)
      : Base(other) {}

  void runOnOperation() override {
    llvm::StringMap<int64_t> dimensions;
    llvm::StringMap<SmallVector<int64_t>> inputShapes;
    if (failed(parseDimensions(dimensionBindings, dimensions)) ||
        failed(parseInputShapes(inputShapeBindings, inputShapes))) {
      getOperation().emitError("invalid ONNX input specialization bindings");
      return signalPassFailure();
    }

    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
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
        SmallVector<int64_t> shape(tensor.getShape());
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
            auto [axisText, expression] = entry.split(':');
            unsigned axis = 0;
            if (axisText.getAsInteger(10, axis) || axis >= shape.size()) {
              function.emitError("invalid onnx.dim_params entry '")
                  << entry << "'";
              return signalPassFailure();
            }
            if (std::optional<int64_t> value =
                    evaluateDimension(expression, dimensions))
              shape[axis] = *value;
          }
        }
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
  }
};
} // namespace
} // namespace vrc
