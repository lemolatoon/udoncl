#include "ONNXHostGraph.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace vrc {
namespace {
unsigned graphElementByteSize(Type type) {
  if (type.isInteger(1) || type.isInteger(8))
    return 1;
  if (type.isF16() || type.isBF16())
    return 2;
  if (type.isInteger(32) || type.isF32())
    return 4;
  if (type.isInteger(64))
    return 8;
  return 0;
}
} // namespace

FailureOr<GraphRecord> readONNXHostGraph(ModuleOp module) {
  auto graph = module.lookupSymbol<func::FuncOp>("main_graph");
  if (!graph || graph.isExternal() || !llvm::hasSingleElement(graph.getBody()))
    return module.emitError(
        "ONNX Host finalization requires one defined single-block main_graph");

  GraphRecord result;
  llvm::DenseMap<Value, unsigned> values;
  auto addTensor = [&](Value value, StringRef stem,
                       DenseElementsAttr initial = {},
                       int64_t staticResourceId = -1,
                       int64_t staticByteLength = 0) -> FailureOr<unsigned> {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !type.hasStaticShape() || type.getNumElements() < 0)
      return emitError(value.getLoc())
             << "ONNX Host tensor requires a nonnegative static ranked shape, "
                "but got "
             << value.getType()
             << "; shape specialization must complete in the C++ ONNX passes";
    if (!graphElementByteSize(type.getElementType()))
      return emitError(value.getLoc())
             << "ONNX Host ABI does not support tensor element type "
             << type.getElementType();
    unsigned id = result.tensors.size();
    result.tensors.push_back({type,
                              initial,
                              ("__udon_onnx_" + stem + "_" + Twine(id)).str(),
                              {},
                              staticResourceId,
                              staticByteLength});
    values.try_emplace(value, id);
    return id;
  };

  for (auto [index, argument] : llvm::enumerate(graph.getArguments())) {
    auto id = addTensor(argument, "input");
    if (failed(id))
      return failure();
    auto name =
        dyn_cast_or_null<StringAttr>(graph.getArgAttr(index, "onnx.name"));
    result.tensors[*id].name =
        name ? name.str() : ("input_" + Twine(index)).str();
    result.inputs.push_back(*id);
  }

  for (Operation &operation : graph.front()) {
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!dense) {
        constant.emitError("ONNX Host constants must be dense tensors");
        return failure();
      }
      if (failed(addTensor(constant.getResult(), "constant", dense)))
        return failure();
      continue;
    }
    if (auto call = dyn_cast<func::CallOp>(operation)) {
      auto declaration = module.lookupSymbol<func::FuncOp>(call.getCallee());
      auto staticTensor = declaration
                              ? declaration->getAttrOfType<DictionaryAttr>(
                                    "udoncl.static_tensor")
                              : DictionaryAttr();
      if (staticTensor) {
        auto dense = staticTensor.getAs<DenseElementsAttr>("data");
        auto resourceId = staticTensor.getAs<IntegerAttr>("id");
        auto byteLength = staticTensor.getAs<IntegerAttr>("byte_length");
        auto type =
            call.getNumResults() == 1
                ? dyn_cast<RankedTensorType>(call.getResult(0).getType())
                : RankedTensorType();
        int64_t expectedBytes =
            type && type.hasStaticShape()
                ? type.getNumElements() *
                      graphElementByteSize(type.getElementType())
                : 0;
        if (call.getNumOperands() != 0 || call.getNumResults() != 1 || !type ||
            (dense && dense.getType() != call.getResult(0).getType()) ||
            !resourceId || !byteLength || resourceId.getInt() < 0 ||
            byteLength.getInt() <= 0 || byteLength.getInt() != expectedBytes) {
          call.emitError("immutable static tensor symbol has an invalid ABI");
          return failure();
        }
        if (failed(addTensor(call.getResult(0), "static", dense,
                             resourceId.getInt(), byteLength.getInt())))
          return failure();
        continue;
      }
      auto operationName =
          declaration
              ? declaration->getAttrOfType<StringAttr>("udon.onnx.operation")
              : StringAttr();
      auto provider =
          declaration
              ? declaration->getAttrOfType<StringAttr>("udon.onnx.provider")
              : StringAttr();
      if (!operationName || !provider || call.getNumResults() != 1) {
        call.emitError("ONNX Host call lacks one typed provider result");
        return failure();
      }
      NodeRecord node{
          operationName.str(),
          provider.str(),
          {},
          0,
          call->getAttrOfType<DictionaryAttr>("udon.onnx.attributes")};
      for (Value operand : call.getOperands()) {
        auto found = values.find(operand);
        if (found == values.end()) {
          auto absent = operand.getDefiningOp<UnrealizedConversionCastOp>();
          if (absent && absent.getInputs().empty() &&
              isa<NoneType>(operand.getType())) {
            node.inputs.push_back(-1);
            continue;
          }
          call.emitError("ONNX Host provider operand is not a tensor or none");
          return failure();
        }
        node.inputs.push_back(found->second);
      }
      auto id = addTensor(call.getResult(0), "value");
      if (failed(id))
        return failure();
      node.output = *id;
      result.nodes.push_back(std::move(node));
      continue;
    }
    if (auto absent = dyn_cast<UnrealizedConversionCastOp>(operation)) {
      if (absent.getInputs().empty() && absent.getNumResults() == 1 &&
          isa<NoneType>(absent.getResult(0).getType()))
        continue;
    }
    if (auto returnOp = dyn_cast<func::ReturnOp>(operation)) {
      for (Value value : returnOp.getOperands()) {
        auto found = values.find(value);
        if (found == values.end()) {
          returnOp.emitError("ONNX Host output is not a tensor value");
          return failure();
        }
        unsigned outputIndex = result.outputs.size();
        auto name = dyn_cast_or_null<StringAttr>(
            graph.getResultAttr(outputIndex, "onnx.name"));
        result.tensors[found->second].name =
            name ? name.str() : ("output_" + Twine(outputIndex)).str();
        result.outputs.push_back(found->second);
      }
      continue;
    }
    operation.emitError("operation remains before ONNX Host ABI finalization");
    return failure();
  }
  if (result.inputs.empty() || result.outputs.empty())
    return graph.emitError("ONNX Host session requires inputs and outputs");

  auto sequenceStates =
      module->getAttrOfType<ArrayAttr>("udoncl.onnx_sequence_states");
  if (!sequenceStates)
    return result;
  for (Attribute entry : sequenceStates) {
    auto dictionary = dyn_cast<DictionaryAttr>(entry);
    auto inputIndex = dictionary ? dictionary.getAs<IntegerAttr>("input_index")
                                 : IntegerAttr();
    auto outputIndex = dictionary
                           ? dictionary.getAs<IntegerAttr>("output_index")
                           : IntegerAttr();
    auto axis =
        dictionary ? dictionary.getAs<IntegerAttr>("axis") : IntegerAttr();
    auto step =
        dictionary ? dictionary.getAs<IntegerAttr>("step") : IntegerAttr();
    auto capacity =
        dictionary ? dictionary.getAs<IntegerAttr>("capacity") : IntegerAttr();
    if (!inputIndex || !outputIndex || !axis || !step || !capacity)
      return module.emitError("ONNX sequence-state metadata is malformed");
    int64_t input = inputIndex.getInt();
    int64_t output = outputIndex.getInt();
    int64_t sequenceAxis = axis.getInt();
    int64_t sequenceStep = step.getInt();
    int64_t sequenceCapacity = capacity.getInt();
    if (input < 0 || input >= static_cast<int64_t>(result.inputs.size()) ||
        output < 0 || output >= static_cast<int64_t>(result.outputs.size()))
      return module.emitError("ONNX sequence-state index is out of range");
    if (llvm::any_of(result.sequenceStates,
                     [&](const SequenceStateRecord &old) {
                       return old.inputIndex == static_cast<unsigned>(input) ||
                              old.outputIndex == static_cast<unsigned>(output);
                     }))
      return module.emitError("ONNX sequence-state indices must be unique");

    unsigned inputTensorIndex = result.inputs[input];
    unsigned outputTensorIndex = result.outputs[output];
    RankedTensorType inputType = result.tensors[inputTensorIndex].type;
    RankedTensorType outputType = result.tensors[outputTensorIndex].type;
    if (!inputType.getElementType().isF32() ||
        outputType.getElementType() != inputType.getElementType() ||
        inputType.getRank() < 1 || inputType.getRank() > 5 ||
        outputType.getRank() != inputType.getRank() || sequenceAxis < 0 ||
        sequenceAxis >= inputType.getRank() || sequenceStep <= 0 ||
        sequenceCapacity <= 0 ||
        outputType.getDimSize(sequenceAxis) != sequenceCapacity ||
        inputType.getDimSize(sequenceAxis) + sequenceStep != sequenceCapacity ||
        inputType.getNumElements() <= 0)
      return module.emitError("ONNX sequence-state shape is not supported");
    for (int64_t dimension = 0; dimension < inputType.getRank(); ++dimension) {
      if (inputType.getDimSize(dimension) > 65535 ||
          outputType.getDimSize(dimension) > 65535 ||
          (dimension != sequenceAxis &&
           inputType.getDimSize(dimension) != outputType.getDimSize(dimension)))
        return module.emitError("ONNX sequence-state extent is not supported");
    }

    const NodeRecord *concat = nullptr;
    for (const NodeRecord &node : result.nodes)
      if (node.output == outputTensorIndex)
        concat = &node;
    auto concatAxis = concat && concat->attributes
                          ? concat->attributes.getAs<IntegerAttr>("axis")
                          : IntegerAttr();
    // ONNX permits a negative Concat axis counted from the last dimension.
    int64_t normalizedConcatAxis =
        concatAxis && concatAxis.getInt() < 0
            ? concatAxis.getInt() + outputType.getRank()
            : (concatAxis ? concatAxis.getInt() : -1);
    if (!concat || concat->operation != "Concat" || concat->inputs.empty() ||
        concat->inputs[0] != static_cast<int64_t>(inputTensorIndex) ||
        !concatAxis || normalizedConcatAxis != sequenceAxis)
      return module.emitError(
          "ONNX sequence-state metadata does not match a Concat output");
    result.sequenceStates.push_back(
        {static_cast<unsigned>(input), static_cast<unsigned>(output),
         sequenceAxis, sequenceStep, sequenceCapacity});
  }
  return result;
}
} // namespace vrc
