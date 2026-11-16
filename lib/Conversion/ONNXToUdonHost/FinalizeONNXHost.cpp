#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "udoncl/udon_abi_layout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_FINALIZEONNXHOSTPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace host = ::vrc::udonhl;

struct TensorRecord {
  RankedTensorType type;
  DenseElementsAttr initialValue;
  std::string symbol;
  std::string name;
  int64_t staticResourceId = -1;
  int64_t staticByteLength = 0;
};

struct NodeRecord {
  std::string operation;
  std::string provider;
  SmallVector<int64_t> inputs;
  unsigned output;
  DictionaryAttr attributes;
};

struct SequenceStateRecord {
  unsigned inputIndex;
  unsigned outputIndex;
  int64_t axis;
  int64_t step;
  int64_t capacity;
};

struct GraphRecord {
  SmallVector<TensorRecord> tensors;
  SmallVector<unsigned> inputs;
  SmallVector<unsigned> outputs;
  SmallVector<NodeRecord> nodes;
  SmallVector<SequenceStateRecord> sequenceStates;
};

int64_t deviceAllocationBytes(const GraphRecord &graph, unsigned tensorIndex);

unsigned elementByteSize(Type type) {
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

int32_t udonclDataType(Type type) {
  if (type.isInteger(1) || type.isInteger(8))
    return 1;
  if (type.isInteger(32))
    return 2;
  if (type.isInteger(64))
    return 3;
  if (type.isF16())
    return 4;
  if (type.isBF16())
    return 5;
  if (type.isF32())
    return 6;
  return 0;
}

StringRef udonclDataTypeName(Type type) {
  if (type.isInteger(1))
    return "bool";
  if (type.isInteger(8))
    return "uint8";
  if (type.isInteger(32))
    return "int32";
  if (type.isInteger(64))
    return "int64";
  if (type.isF16())
    return "float16";
  if (type.isBF16())
    return "bfloat16";
  if (type.isF32())
    return "float32";
  return "unknown";
}

FailureOr<GraphRecord> readGraph(ModuleOp module) {
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
    if (!elementByteSize(type.getElementType()))
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
                ? type.getNumElements() * elementByteSize(type.getElementType())
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
        inputType.getNumElements() <= 0 || inputType.getNumElements() > 16384)
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
    if (!concat || concat->operation != "Concat" || concat->inputs.empty() ||
        concat->inputs[0] != static_cast<int64_t>(inputTensorIndex) ||
        !concatAxis || concatAxis.getInt() != sequenceAxis)
      return module.emitError(
          "ONNX sequence-state metadata does not match a Concat output");
    result.sequenceStates.push_back(
        {static_cast<unsigned>(input), static_cast<unsigned>(output),
         sequenceAxis, sequenceStep, sequenceCapacity});
  }
  return result;
}

bool isSequenceStateInput(const GraphRecord &graph, size_t index) {
  return llvm::any_of(graph.sequenceStates,
                      [&](const SequenceStateRecord &state) {
                        return state.inputIndex == index;
                      });
}

bool isSequenceStateOutput(const GraphRecord &graph, size_t index) {
  return llvm::any_of(graph.sequenceStates,
                      [&](const SequenceStateRecord &state) {
                        return state.outputIndex == index;
                      });
}

int64_t bindingCapacityBytes(const GraphRecord &graph, unsigned tensorIndex) {
  const TensorRecord &tensor = graph.tensors[tensorIndex];
  return std::max<int64_t>(elementByteSize(tensor.type.getElementType()),
                           deviceAllocationBytes(graph, tensorIndex));
}

void attachBindingMetadata(ModuleOp module, const GraphRecord &graph) {
  Builder builder(module.getContext());
  auto ioMetadata = [&](ArrayRef<unsigned> tensors) {
    SmallVector<Attribute> entries;
    entries.reserve(tensors.size());
    for (auto [index, tensorIndex] : llvm::enumerate(tensors)) {
      const TensorRecord &tensor = graph.tensors[tensorIndex];
      entries.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("index", builder.getI32IntegerAttr(index)),
          builder.getNamedAttr("name", builder.getStringAttr(tensor.name)),
          builder.getNamedAttr("data_type",
                               builder.getStringAttr(udonclDataTypeName(
                                   tensor.type.getElementType()))),
          builder.getNamedAttr(
              "rank", builder.getI32IntegerAttr(tensor.type.getRank())),
          builder.getNamedAttr("dimensions", builder.getDenseI64ArrayAttr(
                                                 tensor.type.getShape())),
          builder.getNamedAttr(
              "logical_bytes",
              builder.getI64IntegerAttr(
                  tensor.type.getNumElements() *
                  elementByteSize(tensor.type.getElementType()))),
          builder.getNamedAttr("capacity_bytes",
                               builder.getI64IntegerAttr(
                                   bindingCapacityBytes(graph, tensorIndex))),
      }));
    }
    return builder.getArrayAttr(entries);
  };
  auto physical = builder.getDictionaryAttr({
      builder.getNamedAttr("create_symbol",
                           builder.getStringAttr("udon_onnx_binding_create")),
      builder.getNamedAttr(
          "bind_tensor_symbol",
          builder.getStringAttr("udon_onnx_binding_bind_tensor")),
      builder.getNamedAttr("bind_all_symbol",
                           builder.getStringAttr("udon_onnx_binding_bind_all")),
      builder.getNamedAttr(
          "resolve_input_symbol",
          builder.getStringAttr("udon_onnx_binding_resolve_input")),
      builder.getNamedAttr(
          "resolve_output_symbol",
          builder.getStringAttr("udon_onnx_binding_resolve_output")),
      builder.getNamedAttr(
          "enqueue_initialize_symbol",
          builder.getStringAttr("udon_onnx_binding_enqueue_initialize")),
      builder.getNamedAttr(
          "enqueue_run_symbol",
          builder.getStringAttr("udon_onnx_binding_enqueue_run")),
      builder.getNamedAttr(
          "enqueue_copy_output_to_input_symbol",
          builder.getStringAttr(
              "udon_onnx_binding_enqueue_copy_output_to_input")),
      builder.getNamedAttr(
          "enqueue_advance_sequence_state_symbol",
          builder.getStringAttr(
              graph.sequenceStates.empty()
                  ? ""
                  : "udon_onnx_binding_enqueue_advance_sequence_state")),
      builder.getNamedAttr(
          "enqueue_reset_sequence_state_symbol",
          builder.getStringAttr(
              graph.sequenceStates.empty()
                  ? ""
                  : "udon_onnx_binding_enqueue_reset_sequence_state")),
      builder.getNamedAttr("dispose_symbol",
                           builder.getStringAttr("udon_onnx_binding_dispose")),
  });
  module->setAttr(
      "udoncl.onnx_binding",
      builder.getDictionaryAttr({
          builder.getNamedAttr("abi",
                               builder.getStringAttr("udononnx-runtime-v2")),
          builder.getNamedAttr("model_index", builder.getI32IntegerAttr(0)),
          builder.getNamedAttr("model_name", builder.getStringAttr("model_0")),
          builder.getNamedAttr("inputs", ioMetadata(graph.inputs)),
          builder.getNamedAttr("outputs", ioMetadata(graph.outputs)),
          builder.getNamedAttr("physical", physical),
      }));
}

DenseI8ArrayAttr rawInitializer(DenseElementsAttr dense) {
  SmallVector<int8_t> bytes;
  Type element = cast<ShapedType>(dense.getType()).getElementType();
  unsigned width = elementByteSize(element);
  bytes.reserve(dense.getNumElements() * width);
  auto append = [&](const APInt &bits) {
    for (unsigned index = 0; index != width; ++index)
      bytes.push_back(
          static_cast<int8_t>(bits.extractBitsAsZExtValue(8, index * 8)));
  };
  if (isa<FloatType>(element))
    for (APFloat value : dense.getValues<APFloat>())
      append(value.bitcastToAPInt());
  else
    for (APInt value : dense.getValues<APInt>())
      append(value);
  return DenseI8ArrayAttr::get(dense.getContext(), bytes);
}

void createGlobal(ModuleOp module, const TensorRecord &tensor) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  OperationState state(module.getLoc(), host::GlobalOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(tensor.symbol));
  state.addAttribute("storage_type",
                     TypeAttr::get(LLVM::LLVMArrayType::get(
                         tensor.type.getElementType(),
                         std::max<int64_t>(tensor.type.getNumElements(), 1))));
  if (tensor.initialValue) {
    state.addAttribute("initial_value", rawInitializer(tensor.initialValue));
    state.addAttribute("constant", builder.getUnitAttr());
  } else
    state.addAttribute(
        "initial_value",
        DenseI8ArrayAttr::get(
            module.getContext(),
            SmallVector<int8_t>(
                std::max<int64_t>(tensor.type.getNumElements(), 1) *
                    elementByteSize(tensor.type.getElementType()),
                0)));
  state.addAttribute("linkage", builder.getStringAttr("internal"));
  state.addAttribute("address_space", builder.getI32IntegerAttr(0));
  state.addAttribute("alignment", builder.getI64IntegerAttr(elementByteSize(
                                      tensor.type.getElementType())));
  builder.create(state);
}

void createI32Global(ModuleOp module, StringRef symbol) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  OperationState state(module.getLoc(), host::GlobalOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(symbol));
  state.addAttribute("storage_type", TypeAttr::get(builder.getI32Type()));
  state.addAttribute("initial_value", builder.getI32IntegerAttr(0));
  state.addAttribute("linkage", builder.getStringAttr("internal"));
  state.addAttribute("address_space", builder.getI32IntegerAttr(0));
  state.addAttribute("alignment", builder.getI64IntegerAttr(4));
  builder.create(state);
}

void createPointerGlobal(ModuleOp module, StringRef symbol) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  OperationState state(module.getLoc(), host::GlobalOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(symbol));
  state.addAttribute("storage_type",
                     TypeAttr::get(host::PtrType::get(module.getContext(), 0)));
  state.addAttribute("initial_value", builder.getI32IntegerAttr(0));
  state.addAttribute("linkage", builder.getStringAttr("internal"));
  state.addAttribute("address_space", builder.getI32IntegerAttr(0));
  state.addAttribute("alignment", builder.getI64IntegerAttr(4));
  builder.create(state);
}

Operation *createByteGlobal(ModuleOp module, StringRef symbol,
                            ArrayRef<int8_t> bytes, int64_t alignment = 1) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  OperationState state(module.getLoc(), host::GlobalOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(symbol));
  state.addAttribute("storage_type", TypeAttr::get(LLVM::LLVMArrayType::get(
                                         builder.getI8Type(),
                                         std::max<size_t>(bytes.size(), 1))));
  SmallVector<int8_t> initial(bytes);
  if (initial.empty())
    initial.push_back(0);
  state.addAttribute("initial_value",
                     DenseI8ArrayAttr::get(module.getContext(), initial));
  state.addAttribute("linkage", builder.getStringAttr("internal"));
  state.addAttribute("address_space", builder.getI32IntegerAttr(0));
  state.addAttribute("alignment", builder.getI64IntegerAttr(alignment));
  return builder.create(state);
}

Operation *createZeroGlobal(ModuleOp module, StringRef symbol, size_t bytes,
                            int64_t alignment = 8) {
  return createByteGlobal(module, symbol, SmallVector<int8_t>(bytes, 0),
                          alignment);
}

void createStringGlobal(ModuleOp module, StringRef symbol, StringRef value) {
  SmallVector<int8_t> bytes;
  bytes.reserve(value.size() + 1);
  for (char character : value)
    bytes.push_back(static_cast<int8_t>(character));
  bytes.push_back(0);
  createByteGlobal(module, symbol, bytes);
}

void createI64ArrayGlobal(ModuleOp module, StringRef symbol,
                          ArrayRef<int64_t> values) {
  SmallVector<int8_t> bytes;
  bytes.reserve(values.size() * 8);
  for (int64_t value : values) {
    uint64_t bits = static_cast<uint64_t>(value);
    for (unsigned byte = 0; byte != 8; ++byte)
      bytes.push_back(static_cast<int8_t>((bits >> (byte * 8)) & 0xffu));
  }
  createByteGlobal(module, symbol, bytes, 8);
}

constexpr StringLiteral modelStorageSymbol = "__udon_onnx_model_storage";
constexpr StringLiteral inputMetadataSymbol = "__udon_onnx_input_metadata";
constexpr StringLiteral outputMetadataSymbol = "__udon_onnx_output_metadata";
constexpr StringLiteral modelNameSymbol = "__udon_onnx_model_name";
constexpr StringLiteral bindingStateSymbol = "__udon_onnx_binding_state";
constexpr StringLiteral bindingInputSymbol = "__udon_onnx_binding_inputs";
constexpr StringLiteral bindingOutputSymbol = "__udon_onnx_binding_outputs";
constexpr StringLiteral bindingEventSymbol = "__udon_onnx_binding_event";
constexpr StringLiteral bindingIoHandlesSymbol = "__udon_onnx_io_handles";
constexpr StringLiteral sequenceStateOwnedSymbol =
    "__udon_onnx_sequence_state_owned";
constexpr StringLiteral sequenceStatePendingSymbol =
    "__udon_onnx_sequence_state_pending";
constexpr StringLiteral sequenceStatePoisonedSymbol =
    "__udon_onnx_sequence_state_poisoned";
constexpr StringLiteral sequenceStateZeroSymbol =
    "__udon_onnx_sequence_state_zero";
constexpr int32_t metadataStorageBytes = 64;
constexpr int32_t modelStorageBytes = 64;
constexpr int32_t bindingStateStorageBytes = 256;
constexpr int32_t bindingTensorStorageBytes = 128;
constexpr int32_t bindingBufferHandleBytes = 8;
constexpr int32_t maximumTensorRank = 8;

std::string ioNameSymbol(bool output, unsigned index) {
  return (Twine("__udon_onnx_") + (output ? "output" : "input") + "_name_" +
          Twine(index))
      .str();
}

std::string ioDimensionsSymbol(bool output, unsigned index) {
  return (Twine("__udon_onnx_") + (output ? "output" : "input") +
          "_dimensions_" + Twine(index))
      .str();
}

void createModelMetadataGlobals(ModuleOp module, const GraphRecord &graph) {
  Builder builder(module.getContext());
  auto setBufferViewLayout = [&](Operation *global, int32_t count,
                                 int32_t stride, ArrayRef<int32_t> offsets) {
    global->setAttr(
        "vrc.buffer_view_layout",
        builder.getDictionaryAttr({
            builder.getNamedAttr("count", builder.getI32IntegerAttr(count)),
            builder.getNamedAttr("offsets",
                                 builder.getDenseI32ArrayAttr(offsets)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("stride_bytes",
                                 builder.getI32IntegerAttr(stride)),
        }));
  };
  setBufferViewLayout(
      createZeroGlobal(module, modelStorageSymbol, modelStorageBytes), 0, 0,
      {});
  setBufferViewLayout(
      createZeroGlobal(module, inputMetadataSymbol,
                       std::max<size_t>(graph.inputs.size(), 1) *
                           metadataStorageBytes),
      0, 0, {});
  setBufferViewLayout(
      createZeroGlobal(module, outputMetadataSymbol,
                       std::max<size_t>(graph.outputs.size(), 1) *
                           metadataStorageBytes),
      0, 0, {});
  createStringGlobal(module, modelNameSymbol, "model");
  setBufferViewLayout(
      createZeroGlobal(module, bindingStateSymbol, bindingStateStorageBytes), 0,
      0, {});
  setBufferViewLayout(
      createZeroGlobal(module, bindingInputSymbol,
                       std::max<size_t>(graph.inputs.size(), 1) *
                           bindingTensorStorageBytes),
      static_cast<int32_t>(graph.inputs.size()),
      UDONCL_UDON_ABI_ONNX_TENSOR_BYTES, {0});
  setBufferViewLayout(
      createZeroGlobal(module, bindingOutputSymbol,
                       std::max<size_t>(graph.outputs.size(), 1) *
                           bindingTensorStorageBytes),
      static_cast<int32_t>(graph.outputs.size()),
      UDONCL_UDON_ABI_ONNX_TENSOR_BYTES, {0});
  createZeroGlobal(
      module, bindingIoHandlesSymbol,
      std::max<size_t>(graph.inputs.size() + graph.outputs.size(), 1) *
          bindingBufferHandleBytes,
      4);
  createI32Global(module, bindingEventSymbol);
  if (!graph.sequenceStates.empty()) {
    createI32Global(module, sequenceStateOwnedSymbol);
    createI32Global(module, sequenceStatePendingSymbol);
    createI32Global(module, sequenceStatePoisonedSymbol);
    size_t zeroBytes = 1;
    for (const SequenceStateRecord &sequence : graph.sequenceStates) {
      const TensorRecord &input =
          graph.tensors[graph.inputs[sequence.inputIndex]];
      zeroBytes = std::max<size_t>(
          zeroBytes, input.type.getNumElements() *
                         elementByteSize(input.type.getElementType()));
    }
    createZeroGlobal(module, sequenceStateZeroSymbol, zeroBytes);
  }
  for (auto [index, tensorIndex] : llvm::enumerate(graph.inputs)) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    createStringGlobal(module, ioNameSymbol(false, index), tensor.name);
    createI64ArrayGlobal(module, ioDimensionsSymbol(false, index),
                         tensor.type.getShape());
  }
  for (auto [index, tensorIndex] : llvm::enumerate(graph.outputs)) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    createStringGlobal(module, ioNameSymbol(true, index), tensor.name);
    createI64ArrayGlobal(module, ioDimensionsSymbol(true, index),
                         tensor.type.getShape());
  }
}

std::string inputBindingSymbol(unsigned index) {
  return ("__udon_onnx_input_binding_" + Twine(index)).str();
}

std::string deviceBufferSymbol(unsigned index) {
  return ("__udon_onnx_device_buffer_" + Twine(index)).str();
}

std::string deviceBufferOwnedSymbol(unsigned index) {
  return ("__udon_onnx_device_buffer_owned_" + Twine(index)).str();
}

constexpr StringLiteral hasRunSymbol = "__udon_onnx_has_run";

Value addressOf(OpBuilder &builder, Location location, StringRef symbol) {
  OperationState state(location, host::AddressOfOp::getOperationName());
  state.addTypes(host::PtrType::get(builder.getContext(), 0));
  state.addAttribute("global",
                     FlatSymbolRefAttr::get(builder.getContext(), symbol));
  return builder.create(state)->getResult(0);
}

void store(OpBuilder &builder, Location location, Value value, Value pointer);
Value loadI32(OpBuilder &builder, Location location, Value pointer);

Value allocaI32(OpBuilder &builder, Location location) {
  Value bytes = arith::ConstantIntOp::create(builder, location, 4, 32);
  OperationState state(location, host::AllocaOp::getOperationName());
  state.addOperands(bytes);
  state.addTypes(host::PtrType::get(builder.getContext(), 0));
  state.addAttribute("alignment", builder.getI64IntegerAttr(4));
  return builder.create(state)->getResult(0);
}

Value allocaF32(OpBuilder &builder, Location location, float value) {
  Value bytes = arith::ConstantIntOp::create(builder, location, 4, 32);
  OperationState state(location, host::AllocaOp::getOperationName());
  state.addOperands(bytes);
  state.addTypes(host::PtrType::get(builder.getContext(), 0));
  state.addAttribute("alignment", builder.getI64IntegerAttr(4));
  Value pointer = builder.create(state)->getResult(0);
  Value constant = arith::ConstantFloatOp::create(
      builder, location, builder.getF32Type(), APFloat(value));
  store(builder, location, constant, pointer);
  return pointer;
}

Value allocaI64Array(OpBuilder &builder, Location location,
                     ArrayRef<int64_t> values) {
  Value bytes = arith::ConstantIntOp::create(
      builder, location, static_cast<int64_t>(values.size()) * 8, 32);
  OperationState state(location, host::AllocaOp::getOperationName());
  state.addOperands(bytes);
  state.addTypes(host::PtrType::get(builder.getContext(), 0));
  state.addAttribute("alignment", builder.getI64IntegerAttr(8));
  Value pointer = builder.create(state)->getResult(0);
  for (auto [index, integer] : llvm::enumerate(values)) {
    Value offset = arith::ConstantIntOp::create(
        builder, location, static_cast<int64_t>(index) * 8, 32);
    OperationState add(location, host::PtrAddOp::getOperationName());
    add.addOperands({pointer, offset});
    add.addTypes(pointer.getType());
    Value element = builder.create(add)->getResult(0);
    Value value = arith::ConstantIntOp::create(builder, location, integer, 64);
    OperationState write(location, host::StoreOp::getOperationName());
    write.addOperands({value, element});
    write.addAttribute("alignment", builder.getI64IntegerAttr(8));
    builder.create(write);
  }
  return pointer;
}

Value nullPointer(OpBuilder &builder, Location location) {
  Value zero = arith::ConstantIntOp::create(builder, location, 0, 32);
  OperationState state(location, host::IntToPtrOp::getOperationName());
  state.addOperands(zero);
  state.addTypes(host::PtrType::get(builder.getContext(), 0));
  return builder.create(state)->getResult(0);
}

Value pointerToInt(OpBuilder &builder, Location location, Value pointer) {
  OperationState state(location, host::PtrToIntOp::getOperationName());
  state.addOperands(pointer);
  state.addTypes(builder.getI32Type());
  return builder.create(state)->getResult(0);
}

Value intToPointer(OpBuilder &builder, Location location, Value integer) {
  OperationState state(location, host::IntToPtrOp::getOperationName());
  state.addOperands(integer);
  state.addTypes(host::PtrType::get(builder.getContext(), 0));
  return builder.create(state)->getResult(0);
}

Value boundInputPointer(OpBuilder &builder, Location location, unsigned index,
                        StringRef fallbackSymbol) {
  Value bindingAddress =
      addressOf(builder, location, inputBindingSymbol(index));
  Value binding = loadI32(builder, location, bindingAddress);
  Value zero = arith::ConstantIntOp::create(builder, location, 0, 32);
  Value isBound = arith::CmpIOp::create(
      builder, location, arith::CmpIPredicate::ne, binding, zero);
  OperationState cast(location, host::IntToPtrOp::getOperationName());
  cast.addOperands(binding);
  cast.addTypes(host::PtrType::get(builder.getContext(), 0));
  Value selected = builder.create(cast)->getResult(0);
  return arith::SelectOp::create(builder, location, isBound, selected,
                                 addressOf(builder, location, fallbackSymbol));
}

void store(OpBuilder &builder, Location location, Value value, Value pointer) {
  OperationState state(location, host::StoreOp::getOperationName());
  state.addOperands({value, pointer});
  state.addAttribute("alignment", builder.getI64IntegerAttr(4));
  builder.create(state);
}

Value loadI32(OpBuilder &builder, Location location, Value pointer) {
  OperationState state(location, host::LoadOp::getOperationName());
  state.addOperands(pointer);
  state.addTypes(builder.getI32Type());
  state.addAttribute("alignment", builder.getI64IntegerAttr(4));
  return builder.create(state)->getResult(0);
}

Value loadPointer(OpBuilder &builder, Location location, Value pointer) {
  OperationState state(location, host::LoadOp::getOperationName());
  state.addOperands(pointer);
  state.addTypes(host::PtrType::get(builder.getContext(), 0));
  state.addAttribute("alignment", builder.getI64IntegerAttr(4));
  return builder.create(state)->getResult(0);
}

Value pointerAdd(OpBuilder &builder, Location location, Value pointer,
                 Value offset) {
  OperationState state(location, host::PtrAddOp::getOperationName());
  state.addOperands({pointer, offset});
  state.addTypes(pointer.getType());
  return builder.create(state)->getResult(0);
}

func::FuncOp declaration(ModuleOp module, StringRef name, TypeRange arguments,
                         TypeRange results) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  auto function = func::FuncOp::create(
      builder, module.getLoc(), name,
      FunctionType::get(module.getContext(), arguments, results));
  function.setPrivate();
  return function;
}

SmallVector<int64_t, 4> tensor4d(RankedTensorType type) {
  SmallVector<int64_t, 4> shape(4, 1);
  ArrayRef<int64_t> source = type.getShape();
  if (source.size() == 1) {
    shape[1] = source[0];
  } else if (source.size() > 4) {
    shape[0] = 1;
    for (int64_t dimension : source.drop_back(3))
      shape[0] *= dimension;
    llvm::copy(source.take_back(3), shape.begin() + 1);
  } else {
    for (auto [index, dimension] : llvm::enumerate(source))
      shape[index] = dimension;
  }
  return shape;
}

SmallVector<int64_t, 4> broadcastTensor4d(RankedTensorType type) {
  SmallVector<int64_t, 4> shape(4, 1);
  ArrayRef<int64_t> source = type.getShape();
  if (source.size() > 4) {
    shape[0] = 1;
    for (int64_t dimension : source.drop_back(3))
      shape[0] *= dimension;
    llvm::copy(source.take_back(3), shape.begin() + 1);
  } else {
    llvm::copy(source, shape.end() - source.size());
  }
  return shape;
}

int64_t integerAttribute(const NodeRecord &node, StringRef name,
                         int64_t fallback) {
  if (!node.attributes)
    return fallback;
  auto value = node.attributes.getAs<IntegerAttr>(name);
  return value ? value.getInt() : fallback;
}

SmallVector<int64_t> integerArrayAttribute(const NodeRecord &node,
                                           StringRef name,
                                           ArrayRef<int64_t> fallback) {
  if (!node.attributes)
    return SmallVector<int64_t>(fallback);
  Attribute attribute = node.attributes.get(name);
  if (auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(attribute))
    return SmallVector<int64_t>(dense.asArrayRef());
  SmallVector<int64_t> result;
  if (auto array = dyn_cast_or_null<ArrayAttr>(attribute))
    for (Attribute value : array)
      result.push_back(cast<IntegerAttr>(value).getInt());
  return result.empty() ? SmallVector<int64_t>(fallback) : result;
}

bool isMetadataOnlyAlias(const NodeRecord &node) {
  return node.operation == "Flatten" || node.operation == "Reshape" ||
         node.operation == "Identity" || node.operation == "Squeeze" ||
         node.operation == "Unsqueeze";
}

bool isDeviceAlias(const GraphRecord &graph, const NodeRecord &node) {
  if (!isMetadataOnlyAlias(node) || node.inputs.empty() || node.inputs[0] < 0)
    return false;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  return input.type.getElementType() == output.type.getElementType() &&
         input.type.getNumElements() == output.type.getNumElements();
}

constexpr StringLiteral operatorScratchSymbol = "__udon_onnx_operator_scratch";
constexpr StringLiteral operatorScratchOwnedSymbol =
    "__udon_onnx_operator_scratch_owned";
constexpr int64_t q4SurfaceWords = 131072;
constexpr int64_t q4SurfaceBytes = q4SurfaceWords * 4;
constexpr int64_t rmsScratchBytes = 16384 * 4;

std::optional<float> scalarF32(const TensorRecord &tensor) {
  if (!tensor.initialValue || !tensor.type.getElementType().isF32() ||
      tensor.type.getNumElements() != 1)
    return std::nullopt;
  auto values = tensor.initialValue.getValues<APFloat>();
  APFloat value = *values.begin();
  return value.convertToFloat();
}

std::optional<SmallVector<int64_t>>
staticI64Values(const TensorRecord &tensor) {
  if (!tensor.initialValue || !tensor.type.getElementType().isInteger(64))
    return std::nullopt;
  SmallVector<int64_t> result;
  result.reserve(tensor.type.getNumElements());
  for (APInt value : tensor.initialValue.getValues<APInt>())
    result.push_back(value.getSExtValue());
  return result;
}

struct DeviceSliceShape {
  SmallVector<int64_t> starts;
  SmallVector<int64_t> steps;
};

std::optional<DeviceSliceShape> deviceSliceShape(const GraphRecord &graph,
                                                 const NodeRecord &node) {
  if (node.operation != "Slice" || node.provider != "udononnxSlice" ||
      node.inputs.size() < 3 || node.inputs[0] < 0 || node.inputs[1] < 0 ||
      node.inputs[2] < 0)
    return std::nullopt;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t rank = input.type.getRank();
  if (!input.type.getElementType().isF32() ||
      !output.type.getElementType().isF32() || rank < 1 || rank > 5 ||
      output.type.getRank() != rank || input.type.getNumElements() < 1 ||
      input.type.getNumElements() > 16384 || output.type.getNumElements() < 1 ||
      output.type.getNumElements() > 16384)
    return std::nullopt;
  auto starts = staticI64Values(graph.tensors[node.inputs[1]]);
  auto ends = staticI64Values(graph.tensors[node.inputs[2]]);
  if (!starts || !ends || starts->size() != ends->size() || starts->empty() ||
      starts->size() > static_cast<size_t>(rank))
    return std::nullopt;
  SmallVector<int64_t> axes;
  if (node.inputs.size() >= 4 && node.inputs[3] >= 0) {
    auto values = staticI64Values(graph.tensors[node.inputs[3]]);
    if (!values || values->size() != starts->size())
      return std::nullopt;
    axes = std::move(*values);
  } else {
    for (int64_t axis = 0; axis < static_cast<int64_t>(starts->size()); ++axis)
      axes.push_back(axis);
  }
  SmallVector<int64_t> specifiedSteps(starts->size(), 1);
  if (node.inputs.size() >= 5 && node.inputs[4] >= 0) {
    auto values = staticI64Values(graph.tensors[node.inputs[4]]);
    if (!values || values->size() != starts->size())
      return std::nullopt;
    specifiedSteps = std::move(*values);
  }
  DeviceSliceShape result{SmallVector<int64_t>(rank, 0),
                          SmallVector<int64_t>(rank, 1)};
  SmallVector<bool> sliced(rank, false);
  for (auto [startValue, endValue, axisValue, stepValue] :
       llvm::zip_equal(*starts, *ends, axes, specifiedSteps)) {
    int64_t axis = axisValue < 0 ? axisValue + rank : axisValue;
    if (axis < 0 || axis >= rank || sliced[axis] || !stepValue)
      return std::nullopt;
    sliced[axis] = true;
    int64_t extent = input.type.getDimSize(axis);
    int64_t start = startValue;
    int64_t end = endValue;
    int64_t expected;
    if (stepValue > 0) {
      start = start < 0 ? (start < -extent ? 0 : start + extent)
                        : std::min(start, extent);
      end =
          end < 0 ? (end < -extent ? 0 : end + extent) : std::min(end, extent);
      expected = end > start ? (end - start + stepValue - 1) / stepValue : 0;
    } else {
      int64_t magnitude = -stepValue;
      start = start < 0 ? (start < -extent ? -1 : start + extent)
                        : std::min(start, extent - 1);
      if (end == std::numeric_limits<int64_t>::min())
        end = -1;
      else
        end = end < 0 ? (end < -extent ? -1 : end + extent)
                      : std::min(end, extent - 1);
      expected = start > end ? (start - end + magnitude - 1) / magnitude : 0;
    }
    if (expected != output.type.getDimSize(axis) || start < 0 ||
        start > 65535 || stepValue < 1 || stepValue > 65535)
      return std::nullopt;
    result.starts[axis] = start;
    result.steps[axis] = stepValue;
  }
  for (int64_t axis = 0; axis < rank; ++axis)
    if ((!sliced[axis] &&
         input.type.getDimSize(axis) != output.type.getDimSize(axis)) ||
        input.type.getDimSize(axis) < 1 ||
        input.type.getDimSize(axis) > 65535 ||
        output.type.getDimSize(axis) < 1 ||
        output.type.getDimSize(axis) > 65535)
      return std::nullopt;
  return result;
}

bool isDeviceExpandNode(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Expand" || node.provider != "udononnxExpand" ||
      node.inputs.empty() || node.inputs[0] < 0)
    return false;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t inputRank = input.type.getRank();
  int64_t outputRank = output.type.getRank();
  if (!input.type.getElementType().isF32() ||
      !output.type.getElementType().isF32() || inputRank < 0 || inputRank > 5 ||
      outputRank < std::max<int64_t>(inputRank, 1) || outputRank > 5 ||
      input.type.getNumElements() < 1 || input.type.getNumElements() > 16384 ||
      output.type.getNumElements() < 1 || output.type.getNumElements() > 131072)
    return false;
  int64_t rankOffset = outputRank - inputRank;
  for (int64_t outputAxis = 0; outputAxis < outputRank; ++outputAxis) {
    int64_t outputExtent = output.type.getDimSize(outputAxis);
    if (outputExtent < 1 || outputExtent > 65535)
      return false;
    int64_t inputAxis = outputAxis - rankOffset;
    if (inputAxis < 0)
      continue;
    int64_t inputExtent = input.type.getDimSize(inputAxis);
    if (inputExtent < 1 || inputExtent > 65535 ||
        (inputExtent != 1 && inputExtent != outputExtent))
      return false;
  }
  return true;
}

struct DeviceConcatShape {
  int64_t innerWords;
  int64_t leftExtent;
  int64_t rightExtent;
};

std::optional<DeviceConcatShape> deviceConcatShape(const GraphRecord &graph,
                                                   const NodeRecord &node) {
  if (node.operation != "Concat" || node.provider != "udononnxConcat" ||
      node.inputs.size() != 2 || node.inputs[0] < 0 || node.inputs[1] < 0)
    return std::nullopt;
  const TensorRecord &left = graph.tensors[node.inputs[0]];
  const TensorRecord &right = graph.tensors[node.inputs[1]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t rank = output.type.getRank();
  int64_t axis = integerAttribute(node, "axis", 0);
  if (axis < 0)
    axis += rank;
  if (!left.type.getElementType().isF32() ||
      !right.type.getElementType().isF32() ||
      !output.type.getElementType().isF32() || rank < 1 || rank > 5 ||
      left.type.getRank() != rank || right.type.getRank() != rank || axis < 0 ||
      axis >= rank || left.type.getNumElements() < 1 ||
      left.type.getNumElements() > 16384 || right.type.getNumElements() < 1 ||
      right.type.getNumElements() > 16384 || output.type.getNumElements() < 1 ||
      output.type.getNumElements() > 16384)
    return std::nullopt;
  int64_t innerWords = 1;
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    int64_t leftExtent = left.type.getDimSize(dimension);
    int64_t rightExtent = right.type.getDimSize(dimension);
    int64_t outputExtent = output.type.getDimSize(dimension);
    if (leftExtent < 1 || leftExtent > 65535 || rightExtent < 1 ||
        rightExtent > 65535 || outputExtent < 1 || outputExtent > 65535 ||
        (dimension == axis
             ? outputExtent != leftExtent + rightExtent
             : leftExtent != outputExtent || rightExtent != outputExtent))
      return std::nullopt;
    if (dimension > axis)
      innerWords *= outputExtent;
  }
  return DeviceConcatShape{innerWords, left.type.getDimSize(axis),
                           right.type.getDimSize(axis)};
}

struct DeviceGatherShape {
  int64_t innerWords;
  int64_t axisExtent;
  int64_t indexWords;
};

std::optional<DeviceGatherShape> deviceGatherShape(const GraphRecord &graph,
                                                   const NodeRecord &node) {
  if (node.operation != "Gather" || node.provider != "udononnxGather" ||
      node.inputs.size() != 2 || node.inputs[0] < 0 || node.inputs[1] < 0)
    return std::nullopt;
  const TensorRecord &data = graph.tensors[node.inputs[0]];
  const TensorRecord &indices = graph.tensors[node.inputs[1]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t axis = integerAttribute(node, "axis", 0);
  if (axis < 0)
    axis += data.type.getRank();
  if (!data.type.getElementType().isF32() ||
      !output.type.getElementType().isF32() || data.staticResourceId < 0 ||
      axis != 0 || data.type.getRank() < 1 || data.type.getRank() > 5 ||
      (!indices.type.getElementType().isInteger(32) &&
       !indices.type.getElementType().isInteger(64)) ||
      output.type.getNumElements() < 1 ||
      output.type.getNumElements() > 16384 || data.type.getDimSize(0) < 1 ||
      data.type.getDimSize(0) > std::numeric_limits<int32_t>::max())
    return std::nullopt;
  int64_t innerWords = 1;
  for (int64_t extent : data.type.getShape().drop_front()) {
    if (extent < 1 || innerWords > 16384 / extent)
      return std::nullopt;
    innerWords *= extent;
  }
  if (indices.type.getNumElements() < 1 ||
      indices.type.getNumElements() > 16384 / innerWords ||
      output.type.getNumElements() !=
          indices.type.getNumElements() * innerWords)
    return std::nullopt;
  SmallVector<int64_t> expectedShape(indices.type.getShape());
  llvm::append_range(expectedShape, data.type.getShape().drop_front());
  if (output.type.getShape() != ArrayRef<int64_t>(expectedShape))
    return std::nullopt;
  return DeviceGatherShape{innerWords, data.type.getDimSize(0),
                           indices.type.getElementType().isInteger(64) ? 2 : 1};
}

bool isDeviceCastI64F32Node(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Cast" || node.provider != "udononnxCast" ||
      node.inputs.size() != 1 || node.inputs[0] < 0)
    return false;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  return input.type.getElementType().isInteger(64) &&
         output.type.getElementType().isF32() &&
         input.type.getShape() == output.type.getShape() &&
         output.type.getNumElements() >= 1 &&
         output.type.getNumElements() <= 16384;
}

bool isDeviceBroadcast(ArrayRef<const TensorRecord *> inputs,
                       const TensorRecord &output) {
  int64_t outputRank = output.type.getRank();
  if (outputRank < 0 || outputRank > 4 || output.type.getNumElements() < 1 ||
      output.type.getNumElements() > 16384)
    return false;
  for (int64_t extent : output.type.getShape())
    if (extent < 1 || extent > 65535)
      return false;
  for (const TensorRecord *input : inputs) {
    int64_t inputRank = input->type.getRank();
    if (inputRank < 0 || inputRank > outputRank ||
        input->type.getNumElements() < 1 ||
        input->type.getNumElements() > 16384)
      return false;
    int64_t offset = outputRank - inputRank;
    for (int64_t axis = 0; axis < inputRank; ++axis) {
      int64_t inputExtent = input->type.getDimSize(axis);
      int64_t outputExtent = output.type.getDimSize(axis + offset);
      if (inputExtent < 1 || inputExtent > 65535 ||
          (inputExtent != 1 && inputExtent != outputExtent))
        return false;
    }
  }
  return true;
}

bool isDeviceEqualNode(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Equal" || node.provider != "udononnxEqual" ||
      node.inputs.size() != 2 || node.inputs[0] < 0 || node.inputs[1] < 0)
    return false;
  const TensorRecord &left = graph.tensors[node.inputs[0]];
  const TensorRecord &right = graph.tensors[node.inputs[1]];
  const TensorRecord &output = graph.tensors[node.output];
  return left.type.getElementType().isF32() &&
         right.type.getElementType().isF32() &&
         output.type.getElementType().isInteger(1) &&
         isDeviceBroadcast({&left, &right}, output);
}

bool isDeviceWhereNode(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Where" || node.provider != "udononnxWhere" ||
      node.inputs.size() != 3 || node.inputs[0] < 0 || node.inputs[1] < 0 ||
      node.inputs[2] < 0)
    return false;
  const TensorRecord &condition = graph.tensors[node.inputs[0]];
  const TensorRecord &whenTrue = graph.tensors[node.inputs[1]];
  const TensorRecord &whenFalse = graph.tensors[node.inputs[2]];
  const TensorRecord &output = graph.tensors[node.output];
  return condition.type.getElementType().isInteger(1) &&
         whenTrue.type.getElementType().isF32() &&
         whenFalse.type.getElementType().isF32() &&
         output.type.getElementType().isF32() &&
         isDeviceBroadcast({&condition, &whenTrue, &whenFalse}, output);
}

struct DeviceScatterNDShape {
  int64_t tupleRank;
  int64_t updateCount;
  int64_t sliceWords;
  int64_t indexWords;
};

std::optional<DeviceScatterNDShape>
deviceScatterNDShape(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "ScatterND" || node.provider != "udononnxScatterND" ||
      node.inputs.size() != 3 || node.inputs[0] < 0 || node.inputs[1] < 0 ||
      node.inputs[2] < 0)
    return std::nullopt;
  const TensorRecord &data = graph.tensors[node.inputs[0]];
  const TensorRecord &indices = graph.tensors[node.inputs[1]];
  const TensorRecord &updates = graph.tensors[node.inputs[2]];
  const TensorRecord &output = graph.tensors[node.output];
  auto reduction = node.attributes
                       ? node.attributes.getAs<StringAttr>("reduction")
                       : StringAttr();
  int64_t rank = data.type.getRank();
  int64_t indicesRank = indices.type.getRank();
  int64_t tupleRank = indicesRank ? indices.type.getShape().back() : 0;
  if (!data.type.getElementType().isF32() || data.type != output.type ||
      !updates.type.getElementType().isF32() || rank < 1 || rank > 4 ||
      indicesRank < 1 || tupleRank < 1 || tupleRank > rank ||
      (!indices.type.getElementType().isInteger(32) &&
       !indices.type.getElementType().isInteger(64)) ||
      data.type.getNumElements() < 1 || data.type.getNumElements() > 16384 ||
      (reduction && reduction.getValue() != "none"))
    return std::nullopt;
  int64_t updateCount = indices.type.getNumElements() / tupleRank;
  int64_t sliceWords = 1;
  for (int64_t extent : data.type.getShape().drop_front(tupleRank))
    sliceWords *= extent;
  if (updateCount < 1 || updateCount > 256 || sliceWords < 1 ||
      updateCount > 16384 / sliceWords ||
      updates.type.getNumElements() != updateCount * sliceWords)
    return std::nullopt;
  SmallVector<int64_t> expectedUpdates(indices.type.getShape().drop_back());
  llvm::append_range(expectedUpdates,
                     data.type.getShape().drop_front(tupleRank));
  if (updates.type.getShape() != ArrayRef<int64_t>(expectedUpdates))
    return std::nullopt;
  for (int64_t extent : data.type.getShape())
    if (extent < 1 || extent > 65535)
      return std::nullopt;
  return DeviceScatterNDShape{tupleRank, updateCount, sliceWords,
                              indices.type.getElementType().isInteger(64) ? 2
                                                                          : 1};
}

bool isDeviceRmsNormNode(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "RmsNorm" || node.provider != "udonatenRmsNorm" ||
      node.inputs.size() != 3 || node.inputs[0] < 0 || node.inputs[1] < 0 ||
      node.inputs[2] < 0)
    return false;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &weight = graph.tensors[node.inputs[1]];
  const TensorRecord &epsilon = graph.tensors[node.inputs[2]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t words = input.type.getNumElements();
  int64_t normalizedWords = weight.type.getNumElements();
  std::optional<float> epsilonValue = scalarF32(epsilon);
  if (!input.type.getElementType().isF32() ||
      !weight.type.getElementType().isF32() || input.type != output.type ||
      weight.type.getRank() != 1 || weight.staticResourceId < 0 ||
      !epsilonValue || *epsilonValue < 0.0f || words < 1 || words > 16384 ||
      normalizedWords < 1 || normalizedWords > 4096 ||
      words % normalizedWords != 0 ||
      input.type.getShape().back() != normalizedWords)
    return false;
  int64_t rows = words / normalizedWords;
  int64_t partialsPerRow = (normalizedWords + 63) / 64;
  return rows * partialsPerRow <= 16384;
}

bool hasDeviceRmsNorm(const GraphRecord &graph) {
  return llvm::any_of(graph.nodes, [&](const NodeRecord &node) {
    return isDeviceRmsNormNode(graph, node);
  });
}

bool isDeviceSoftmaxNode(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Softmax" || node.provider != "udondnnSoftmaxForward" ||
      node.inputs.size() != 1 || node.inputs[0] < 0)
    return false;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t rank = input.type.getRank();
  int64_t axis = integerAttribute(node, "axis", -1);
  if (axis < 0)
    axis += rank;
  return input.type.getElementType().isF32() && input.type == output.type &&
         axis >= 0 && axis < rank && input.type.getNumElements() >= 1 &&
         input.type.getNumElements() <= 16384 &&
         input.type.getDimSize(axis) >= 1 && input.type.getDimSize(axis) <= 512;
}

std::optional<SmallVector<int64_t>>
deviceTransposePermutation(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Transpose" || node.provider != "udononnxTranspose" ||
      node.inputs.size() != 1 || node.inputs[0] < 0)
    return std::nullopt;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t rank = input.type.getRank();
  if (!input.type.getElementType().isF32() || rank < 1 || rank > 5 ||
      output.type.getRank() != rank || input.type.getNumElements() < 1 ||
      input.type.getNumElements() > 131072 ||
      input.type.getNumElements() != output.type.getNumElements())
    return std::nullopt;
  SmallVector<int64_t> permutation = integerArrayAttribute(node, "perm", {});
  if (permutation.empty())
    for (int64_t axis = rank; axis-- > 0;)
      permutation.push_back(axis);
  if (permutation.size() != static_cast<size_t>(rank))
    return std::nullopt;
  SmallVector<bool> seen(rank, false);
  for (auto [outputAxis, inputAxis] : llvm::enumerate(permutation)) {
    if (inputAxis < 0 || inputAxis >= rank || seen[inputAxis] ||
        input.type.getDimSize(inputAxis) < 1 ||
        input.type.getDimSize(inputAxis) > 65535 ||
        output.type.getDimSize(outputAxis) != input.type.getDimSize(inputAxis))
      return std::nullopt;
    seen[inputAxis] = true;
  }
  return permutation;
}

struct DeviceMatMulShape {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t leftBatches;
  int64_t rightBatches;
  int64_t outputBatches;
};

std::optional<DeviceMatMulShape> deviceMatMulShape(const GraphRecord &graph,
                                                   const NodeRecord &node) {
  if (node.operation != "MatMul" || node.provider != "udonblasSgemm" ||
      node.inputs.size() != 2 || node.inputs[0] < 0 || node.inputs[1] < 0)
    return std::nullopt;
  const TensorRecord &leftTensor = graph.tensors[node.inputs[0]];
  const TensorRecord &rightTensor = graph.tensors[node.inputs[1]];
  const TensorRecord &outputTensor = graph.tensors[node.output];
  ArrayRef<int64_t> left = leftTensor.type.getShape();
  ArrayRef<int64_t> right = rightTensor.type.getShape();
  ArrayRef<int64_t> output = outputTensor.type.getShape();
  if (!leftTensor.type.getElementType().isF32() ||
      !rightTensor.type.getElementType().isF32() ||
      !outputTensor.type.getElementType().isF32() || left.size() < 2 ||
      right.size() < 2 || output.size() < 2 ||
      left.back() != right[right.size() - 2])
    return std::nullopt;
  int64_t m = left[left.size() - 2];
  int64_t k = left.back();
  int64_t n = right.back();
  if (m < 1 || n < 1 || k < 1 || k > 256 || output[output.size() - 2] != m ||
      output.back() != n || leftTensor.type.getNumElements() > 131072 ||
      rightTensor.type.getNumElements() > 131072 ||
      outputTensor.type.getNumElements() > 16384)
    return std::nullopt;
  int64_t leftBatches = leftTensor.type.getNumElements() / (m * k);
  int64_t rightBatches = rightTensor.type.getNumElements() / (k * n);
  int64_t outputBatches = outputTensor.type.getNumElements() / (m * n);
  if (outputBatches < 1 || (leftBatches != 1 && leftBatches != outputBatches) ||
      (rightBatches != 1 && rightBatches != outputBatches))
    return std::nullopt;
  return DeviceMatMulShape{m, n, k, leftBatches, rightBatches, outputBatches};
}

bool isDeviceQ4Node(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "MatMulNBits" ||
      node.provider != "udononnxMatMulNBits" || node.inputs.size() < 3 ||
      node.inputs[0] < 0 || node.inputs[1] < 0 || node.inputs[2] < 0 ||
      llvm::any_of(
          ArrayRef<int64_t>(node.inputs).drop_front(3),
          [](int64_t input) { return input >= 0; }))
    return false;
  const TensorRecord &activation = graph.tensors[node.inputs[0]];
  const TensorRecord &packed = graph.tensors[node.inputs[1]];
  const TensorRecord &scales = graph.tensors[node.inputs[2]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t k = integerAttribute(node, "K", 0);
  int64_t n = integerAttribute(node, "N", 0);
  int64_t blockSize = integerAttribute(node, "block_size", 0);
  int64_t bits = integerAttribute(node, "bits", 0);
  if (!activation.type.getElementType().isF32() ||
      !packed.type.getElementType().isInteger(8) ||
      !scales.type.getElementType().isF32() ||
      !output.type.getElementType().isF32() || packed.staticResourceId < 0 ||
      scales.staticResourceId < 0 || k < 1 || k > 5632 || n < 1 || n > 32000 ||
      blockSize != 32 || bits != 4 || activation.type.getNumElements() % k)
    return false;
  int64_t m = activation.type.getNumElements() / k;
  int64_t blocks = (k + 31) / 32;
  return m >= 1 && m <= 4 && m * n <= q4SurfaceWords &&
         output.type.getNumElements() == m * n &&
         packed.type.getNumElements() == n * blocks * 16 &&
         scales.type.getNumElements() == n * blocks;
}

bool hasDeviceQ4(const GraphRecord &graph) {
  return llvm::any_of(graph.nodes, [&](const NodeRecord &node) {
    return isDeviceQ4Node(graph, node);
  });
}

int64_t operatorScratchBytes(const GraphRecord &graph) {
  int64_t bytes = hasDeviceQ4(graph) ? q4SurfaceBytes : 0;
  if (hasDeviceRmsNorm(graph))
    bytes = std::max(bytes, rmsScratchBytes);
  for (const NodeRecord &node : graph.nodes)
    if (node.operation == "Gemm" && node.inputs.size() >= 3 &&
        node.inputs[2] >= 0)
      bytes = std::max(bytes, deviceAllocationBytes(graph, node.output));
  return bytes;
}

bool needsQ4Surface(const GraphRecord &graph, unsigned tensorIndex) {
  return llvm::any_of(graph.nodes, [&](const NodeRecord &node) {
    return isDeviceQ4Node(graph, node) && node.output == tensorIndex;
  });
}

int64_t deviceAllocationBytes(const GraphRecord &graph, unsigned tensorIndex) {
  const TensorRecord &tensor = graph.tensors[tensorIndex];
  int64_t elementBytes = tensor.type.getElementType().isInteger(1)
                             ? 4
                             : elementByteSize(tensor.type.getElementType());
  int64_t bytes = tensor.type.getNumElements() * elementBytes;
  return needsQ4Surface(graph, tensorIndex) ? std::max(bytes, q4SurfaceBytes)
                                            : bytes;
}

bool supportsDeviceNode(const GraphRecord &graph, const NodeRecord &node) {
  auto isF32 = [&](int64_t index) {
    return index >= 0 && static_cast<size_t>(index) < graph.tensors.size() &&
           graph.tensors[static_cast<unsigned>(index)]
               .type.getElementType()
               .isF32();
  };
  auto hasF32InputsAndOutput = [&](unsigned inputCount) {
    return node.inputs.size() >= inputCount &&
           llvm::all_of(ArrayRef<int64_t>(node.inputs).take_front(inputCount),
                        isF32) &&
           isF32(node.output);
  };
  auto supportsLargePointwise = [&]() {
    if (!hasF32InputsAndOutput(2))
      return false;
    SmallVector<int64_t, 4> output =
        broadcastTensor4d(graph.tensors[node.output].type);
    if (graph.tensors[node.output].type.getNumElements() > 131072)
      return false;
    for (int64_t inputIndex : ArrayRef<int64_t>(node.inputs).take_front(2)) {
      const TensorRecord &input = graph.tensors[inputIndex];
      if (input.type.getNumElements() > 131072)
        return false;
      SmallVector<int64_t, 4> inputShape = broadcastTensor4d(input.type);
      for (auto [inputDimension, outputDimension] :
           llvm::zip_equal(inputShape, output))
        if (inputDimension != 1 && inputDimension != outputDimension)
          return false;
    }
    return llvm::all_of(output, [](int64_t dimension) {
      return dimension > 0 && dimension <= 65535;
    });
  };
  auto supportsLargeActivation = [&]() {
    if (!hasF32InputsAndOutput(1))
      return false;
    const TensorRecord &input = graph.tensors[node.inputs[0]];
    const TensorRecord &output = graph.tensors[node.output];
    return input.type.getNumElements() == output.type.getNumElements() &&
           output.type.getNumElements() > 0 &&
           output.type.getNumElements() <= 32768;
  };
  if (isMetadataOnlyAlias(node))
    return isDeviceAlias(graph, node);
  if (node.operation == "Conv")
    return node.provider == "udondnnConvolutionForward" &&
           hasF32InputsAndOutput(2);
  if (node.operation == "Add")
    return node.provider == "udonatenAdd" && supportsLargePointwise();
  if (node.operation == "Mul")
    return node.provider == "udonatenMul" && supportsLargePointwise();
  if (node.operation == "Relu")
    return node.provider == "udonatenRelu" && supportsLargeActivation();
  if (node.operation == "Sigmoid")
    return node.provider == "udonatenSigmoid" && supportsLargeActivation();
  if (node.operation == "Tanh")
    return node.provider == "udonatenTanh" && supportsLargeActivation();
  if (node.operation == "AveragePool")
    return node.provider == "udondnnPoolingForward" && hasF32InputsAndOutput(1);
  if (node.operation == "Cos" || node.operation == "Sin" ||
      node.operation == "Neg")
    return node.provider == "udononnx" + node.operation &&
           supportsLargeActivation() &&
           graph.tensors[node.output].type.getNumElements() <= 16384;
  if (node.operation == "MatMulNBits")
    return isDeviceQ4Node(graph, node);
  if (node.operation == "RmsNorm")
    return isDeviceRmsNormNode(graph, node);
  if (node.operation == "Softmax")
    return isDeviceSoftmaxNode(graph, node);
  if (node.operation == "Transpose")
    return deviceTransposePermutation(graph, node).has_value();
  if (node.operation == "Slice")
    return deviceSliceShape(graph, node).has_value();
  if (node.operation == "Expand")
    return isDeviceExpandNode(graph, node);
  if (node.operation == "Concat")
    return deviceConcatShape(graph, node).has_value();
  if (node.operation == "Gather")
    return deviceGatherShape(graph, node).has_value();
  if (node.operation == "Cast")
    return isDeviceCastI64F32Node(graph, node);
  if (node.operation == "Equal")
    return isDeviceEqualNode(graph, node);
  if (node.operation == "Where")
    return isDeviceWhereNode(graph, node);
  if (node.operation == "ScatterND")
    return deviceScatterNDShape(graph, node).has_value();
  if (node.operation == "MatMul")
    return deviceMatMulShape(graph, node).has_value();
  if (node.operation == "Gemm") {
    if (node.provider != "udonblasSgemm" || !hasF32InputsAndOutput(2))
      return false;
    const TensorRecord &left = graph.tensors[node.inputs[0]];
    const TensorRecord &right = graph.tensors[node.inputs[1]];
    return left.type.getRank() == 2 && right.type.getRank() == 2;
  }
  return false;
}

bool supportsDeviceGraph(const GraphRecord &graph) {
  return llvm::all_of(graph.nodes, [&](const NodeRecord &node) {
    return supportsDeviceNode(graph, node);
  });
}

void attachDeviceGraphMetadata(ModuleOp module, const GraphRecord &graph,
                               bool deviceGraph) {
  Builder builder(module.getContext());
  llvm::StringMap<int64_t> counts;
  llvm::StringMap<int64_t> unsupported;
  for (const NodeRecord &node : graph.nodes) {
    ++counts[node.operation];
    if (!supportsDeviceNode(graph, node))
      ++unsupported[node.operation + "\n" + node.provider];
  }
  SmallVector<NamedAttribute> countAttributes;
  for (const auto &entry : counts)
    countAttributes.push_back(builder.getNamedAttr(
        entry.first(), builder.getI64IntegerAttr(entry.second)));
  llvm::sort(countAttributes, [](NamedAttribute left, NamedAttribute right) {
    return left.getName().strref() < right.getName().strref();
  });
  SmallVector<Attribute> unsupportedAttributes;
  for (const auto &entry : unsupported) {
    StringRef key = entry.first();
    size_t separator = key.find('\n');
    unsupportedAttributes.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("operation",
                             builder.getStringAttr(key.take_front(separator))),
        builder.getNamedAttr(
            "provider", builder.getStringAttr(key.drop_front(separator + 1))),
        builder.getNamedAttr("count", builder.getI64IntegerAttr(entry.second)),
    }));
  }
  llvm::sort(unsupportedAttributes, [](Attribute left, Attribute right) {
    auto lhs = cast<DictionaryAttr>(left);
    auto rhs = cast<DictionaryAttr>(right);
    return std::make_pair(lhs.getAs<StringAttr>("operation").getValue(),
                          lhs.getAs<StringAttr>("provider").getValue()) <
           std::make_pair(rhs.getAs<StringAttr>("operation").getValue(),
                          rhs.getAs<StringAttr>("provider").getValue());
  });
  module->setAttr("udoncl.onnx_device_graph", builder.getBoolAttr(deviceGraph));
  module->setAttr("udoncl.onnx_operator_counts",
                  builder.getDictionaryAttr(countAttributes));
  module->setAttr("udoncl.onnx_device_unsupported",
                  builder.getArrayAttr(unsupportedAttributes));
}

SmallVector<unsigned> deviceStorage(const GraphRecord &graph) {
  SmallVector<unsigned> storage(graph.tensors.size());
  std::iota(storage.begin(), storage.end(), 0u);
  for (const NodeRecord &node : graph.nodes)
    if (isDeviceAlias(graph, node))
      storage[node.output] = storage[static_cast<unsigned>(node.inputs[0])];

  // Assign non-overlapping intermediate lifetimes to the same physical
  // allocation. A dispatch may read every input while writing its output, so
  // a slot becomes reusable only after the last input use, never at that same
  // node. Graph outputs remain live through the final public-buffer copy.
  SmallVector<int64_t> lastUse(graph.tensors.size(), -1);
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes))
    for (int64_t input : node.inputs)
      if (input >= 0) {
        unsigned root = storage[static_cast<unsigned>(input)];
        lastUse[root] =
            std::max(lastUse[root], static_cast<int64_t>(nodeIndex));
      }
  for (unsigned output : graph.outputs)
    lastUse[storage[output]] = static_cast<int64_t>(graph.nodes.size());

  struct Slot {
    unsigned root;
    int64_t bytes;
    int64_t availableAfter;
  };
  SmallVector<Slot> slots;
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes)) {
    if (isDeviceAlias(graph, node))
      continue;
    unsigned output = node.output;
    const TensorRecord &tensor = graph.tensors[output];
    if (tensor.staticResourceId >= 0 ||
        llvm::is_contained(graph.inputs, output) ||
        llvm::is_contained(graph.outputs, output))
      continue;
    int64_t bytes = deviceAllocationBytes(graph, output);
    Slot *best = nullptr;
    for (Slot &slot : slots)
      if (slot.availableAfter < static_cast<int64_t>(nodeIndex) &&
          slot.bytes >= bytes && (!best || slot.bytes < best->bytes))
        best = &slot;
    if (best) {
      storage[output] = best->root;
      best->availableAfter = lastUse[output];
    } else {
      slots.push_back({output, bytes, lastUse[output]});
    }
  }
  for (unsigned &root : storage)
    while (storage[root] != root)
      root = storage[root];
  return storage;
}

constexpr int64_t maximumMutableHandleCapacity = 512;

SmallVector<int64_t> mutableBufferBytes(const GraphRecord &graph,
                                        bool deviceGraph) {
  SmallVector<int64_t> bytes;
  bytes.reserve(graph.inputs.size() + graph.outputs.size());
  for (unsigned tensorIndex : graph.inputs)
    bytes.push_back(bindingCapacityBytes(graph, tensorIndex));
  for (unsigned tensorIndex : graph.outputs)
    bytes.push_back(bindingCapacityBytes(graph, tensorIndex));
  if (!deviceGraph)
    return bytes;

  SmallVector<unsigned> storage = deviceStorage(graph);
  for (auto [tensorIndex, tensor] : llvm::enumerate(graph.tensors)) {
    if (storage[tensorIndex] != tensorIndex || tensor.staticResourceId >= 0 ||
        llvm::is_contained(graph.inputs, tensorIndex) ||
        (llvm::is_contained(graph.outputs, tensorIndex) &&
         !needsQ4Surface(graph, tensorIndex)))
      continue;
    bytes.push_back(deviceAllocationBytes(graph, tensorIndex));
  }
  if (int64_t scratchBytes = operatorScratchBytes(graph))
    bytes.push_back(scratchBytes);
  return bytes;
}

void createWriteExport(ModuleOp module, const GraphRecord &graph,
                       Type elementType, StringRef suffix) {
  SmallVector<std::pair<unsigned, unsigned>> inputs;
  for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph.inputs))
    if (graph.tensors[tensorIndex].type.getElementType() == elementType)
      inputs.emplace_back(static_cast<unsigned>(inputIndex), tensorIndex);
  if (inputs.empty())
    return;
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  auto i32 = builder.getI32Type();
  auto function = func::FuncOp::create(
      builder, module.getLoc(),
      ("udon_onnx_session_write_input_" + suffix).str(),
      FunctionType::get(module.getContext(), {i32, i32, elementType}, {i32}));
  Block *entry = function.addEntryBlock();
  Block *invalid = function.addBlock();
  Block *select = entry;
  for (auto [matchIndex, selectedInput] : llvm::enumerate(inputs)) {
    auto [inputIndex, tensorIndex] = selectedInput;
    const TensorRecord &inputTensor = graph.tensors[tensorIndex];
    Block *range = function.addBlock();
    Block *valid = function.addBlock();
    Block *next =
        matchIndex + 1 == inputs.size() ? invalid : function.addBlock();
    builder.setInsertionPointToEnd(select);
    Value selected =
        arith::ConstantIntOp::create(builder, module.getLoc(), inputIndex, 32);
    Value indexMatches = arith::CmpIOp::create(builder, module.getLoc(),
                                               arith::CmpIPredicate::eq,
                                               entry->getArgument(0), selected);
    cf::CondBranchOp::create(builder, module.getLoc(), indexMatches, range,
                             ValueRange{}, next, ValueRange{});
    builder.setInsertionPointToEnd(range);
    Value zero = arith::ConstantIntOp::create(builder, module.getLoc(), 0, 32);
    Value count = arith::ConstantIntOp::create(
        builder, module.getLoc(), inputTensor.type.getNumElements(), 32);
    Value nonnegative = arith::CmpIOp::create(builder, module.getLoc(),
                                              arith::CmpIPredicate::sge,
                                              entry->getArgument(1), zero);
    Value below = arith::CmpIOp::create(builder, module.getLoc(),
                                        arith::CmpIPredicate::slt,
                                        entry->getArgument(1), count);
    Value inRange =
        arith::AndIOp::create(builder, module.getLoc(), nonnegative, below);
    cf::CondBranchOp::create(builder, module.getLoc(), inRange, valid,
                             ValueRange{}, invalid, ValueRange{});
    builder.setInsertionPointToEnd(valid);
    Value base = boundInputPointer(builder, module.getLoc(), inputIndex,
                                   inputTensor.symbol);
    unsigned width = elementByteSize(elementType);
    Value elementBytes =
        arith::ConstantIntOp::create(builder, module.getLoc(), width, 32);
    Value offset = arith::MulIOp::create(builder, module.getLoc(),
                                         entry->getArgument(1), elementBytes);
    OperationState write(module.getLoc(), host::StoreOp::getOperationName());
    write.addOperands({entry->getArgument(2),
                       pointerAdd(builder, module.getLoc(), base, offset)});
    write.addAttribute("alignment", builder.getI64IntegerAttr(width));
    builder.create(write);
    func::ReturnOp::create(builder, module.getLoc(), zero);
    select = next;
  }
  builder.setInsertionPointToEnd(invalid);
  Value failure =
      arith::ConstantIntOp::create(builder, module.getLoc(), -1, 32);
  func::ReturnOp::create(builder, module.getLoc(), failure);
}

void createReadExport(ModuleOp module, const GraphRecord &graph,
                      Type elementType, StringRef suffix) {
  SmallVector<std::pair<unsigned, unsigned>> outputs;
  for (auto [outputIndex, tensorIndex] : llvm::enumerate(graph.outputs))
    if (graph.tensors[tensorIndex].type.getElementType() == elementType)
      outputs.emplace_back(static_cast<unsigned>(outputIndex), tensorIndex);
  if (outputs.empty())
    return;
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  auto i32 = builder.getI32Type();
  auto function = func::FuncOp::create(
      builder, module.getLoc(),
      ("udon_onnx_session_read_output_" + suffix).str(),
      FunctionType::get(module.getContext(), {i32, i32}, {elementType}));
  Block *entry = function.addEntryBlock();
  Block *invalid = function.addBlock();
  Block *select = entry;
  for (auto [matchIndex, selectedOutput] : llvm::enumerate(outputs)) {
    auto [outputIndex, tensorIndex] = selectedOutput;
    const TensorRecord &outputTensor = graph.tensors[tensorIndex];
    Block *range = function.addBlock();
    Block *valid = function.addBlock();
    Block *next =
        matchIndex + 1 == outputs.size() ? invalid : function.addBlock();
    builder.setInsertionPointToEnd(select);
    Value selected =
        arith::ConstantIntOp::create(builder, module.getLoc(), outputIndex, 32);
    Value indexMatches = arith::CmpIOp::create(builder, module.getLoc(),
                                               arith::CmpIPredicate::eq,
                                               entry->getArgument(0), selected);
    cf::CondBranchOp::create(builder, module.getLoc(), indexMatches, range,
                             ValueRange{}, next, ValueRange{});
    builder.setInsertionPointToEnd(range);
    Value zero = arith::ConstantIntOp::create(builder, module.getLoc(), 0, 32);
    Value count = arith::ConstantIntOp::create(
        builder, module.getLoc(), outputTensor.type.getNumElements(), 32);
    Value nonnegative = arith::CmpIOp::create(builder, module.getLoc(),
                                              arith::CmpIPredicate::sge,
                                              entry->getArgument(1), zero);
    Value below = arith::CmpIOp::create(builder, module.getLoc(),
                                        arith::CmpIPredicate::slt,
                                        entry->getArgument(1), count);
    Value inRange =
        arith::AndIOp::create(builder, module.getLoc(), nonnegative, below);
    cf::CondBranchOp::create(builder, module.getLoc(), inRange, valid,
                             ValueRange{}, invalid, ValueRange{});
    builder.setInsertionPointToEnd(valid);
    Value base = addressOf(builder, module.getLoc(), outputTensor.symbol);
    unsigned width = elementByteSize(elementType);
    Value elementBytes =
        arith::ConstantIntOp::create(builder, module.getLoc(), width, 32);
    Value offset = arith::MulIOp::create(builder, module.getLoc(),
                                         entry->getArgument(1), elementBytes);
    OperationState load(module.getLoc(), host::LoadOp::getOperationName());
    load.addOperands(pointerAdd(builder, module.getLoc(), base, offset));
    load.addTypes(elementType);
    load.addAttribute("alignment",
                      builder.getI64IntegerAttr(elementByteSize(elementType)));
    Value value = builder.create(load)->getResult(0);
    func::ReturnOp::create(builder, module.getLoc(), value);
    select = next;
  }
  builder.setInsertionPointToEnd(invalid);
  Value fallback;
  if (auto floatType = dyn_cast<FloatType>(elementType))
    fallback = arith::ConstantFloatOp::create(
        builder, module.getLoc(), floatType,
        APFloat::getZero(floatType.getFloatSemantics()));
  else
    fallback =
        arith::ConstantIntOp::create(builder, module.getLoc(), elementType, 0);
  func::ReturnOp::create(builder, module.getLoc(), fallback);
}

void createBindingExports(ModuleOp module, const GraphRecord &graph) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  Type i32 = builder.getI32Type();
  auto bind = func::FuncOp::create(
      builder, module.getLoc(), "udon_onnx_session_bind_output_to_input",
      FunctionType::get(module.getContext(), {i32, i32}, {i32}));
  Block *entry = bind.addEntryBlock();
  Block *invalid = bind.addBlock();
  builder.setInsertionPointToEnd(entry);
  Value zero = arith::ConstantIntOp::create(builder, module.getLoc(), 0, 32);
  Value hasRun = loadI32(builder, module.getLoc(),
                         addressOf(builder, module.getLoc(), hasRunSymbol));
  Value ready = arith::CmpIOp::create(builder, module.getLoc(),
                                      arith::CmpIPredicate::ne, hasRun, zero);
  Block *select = bind.addBlock();
  cf::CondBranchOp::create(builder, module.getLoc(), ready, select,
                           ValueRange{}, invalid, ValueRange{});
  SmallVector<std::pair<unsigned, unsigned>> compatible;
  for (auto [outputIndex, outputTensor] : llvm::enumerate(graph.outputs))
    for (auto [inputIndex, inputTensor] : llvm::enumerate(graph.inputs)) {
      const TensorRecord &output = graph.tensors[outputTensor];
      const TensorRecord &input = graph.tensors[inputTensor];
      if (output.type.getElementType() == input.type.getElementType() &&
          output.type.getNumElements() == input.type.getNumElements())
        compatible.emplace_back(static_cast<unsigned>(outputIndex),
                                static_cast<unsigned>(inputIndex));
    }
  for (auto [matchIndex, indices] : llvm::enumerate(compatible)) {
    auto [outputIndex, inputIndex] = indices;
    Block *matchedOutput = bind.addBlock();
    Block *valid = bind.addBlock();
    Block *next =
        matchIndex + 1 == compatible.size() ? invalid : bind.addBlock();
    builder.setInsertionPointToEnd(select);
    Value expectedOutput =
        arith::ConstantIntOp::create(builder, module.getLoc(), outputIndex, 32);
    Value outputMatches = arith::CmpIOp::create(
        builder, module.getLoc(), arith::CmpIPredicate::eq,
        entry->getArgument(0), expectedOutput);
    cf::CondBranchOp::create(builder, module.getLoc(), outputMatches,
                             matchedOutput, ValueRange{}, next, ValueRange{});
    builder.setInsertionPointToEnd(matchedOutput);
    Value expectedInput =
        arith::ConstantIntOp::create(builder, module.getLoc(), inputIndex, 32);
    Value inputMatches = arith::CmpIOp::create(
        builder, module.getLoc(), arith::CmpIPredicate::eq,
        entry->getArgument(1), expectedInput);
    cf::CondBranchOp::create(builder, module.getLoc(), inputMatches, valid,
                             ValueRange{}, next, ValueRange{});
    builder.setInsertionPointToEnd(valid);
    Value output = addressOf(builder, module.getLoc(),
                             graph.tensors[graph.outputs[outputIndex]].symbol);
    Value binding =
        addressOf(builder, module.getLoc(), inputBindingSymbol(inputIndex));
    store(builder, module.getLoc(),
          pointerToInt(builder, module.getLoc(), output), binding);
    func::ReturnOp::create(builder, module.getLoc(), zero);
    select = next;
  }
  if (compatible.empty()) {
    builder.setInsertionPointToEnd(select);
    cf::BranchOp::create(builder, module.getLoc(), invalid);
  }
  builder.setInsertionPointToEnd(invalid);
  Value failure =
      arith::ConstantIntOp::create(builder, module.getLoc(), -1, 32);
  func::ReturnOp::create(builder, module.getLoc(), failure);

  builder.setInsertionPointToEnd(module.getBody());
  auto unbind = func::FuncOp::create(
      builder, module.getLoc(), "udon_onnx_session_unbind_input",
      FunctionType::get(module.getContext(), {i32}, {i32}));
  entry = unbind.addEntryBlock();
  invalid = unbind.addBlock();
  select = entry;
  for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph.inputs)) {
    (void)tensorIndex;
    Block *valid = unbind.addBlock();
    Block *next =
        inputIndex + 1 == graph.inputs.size() ? invalid : unbind.addBlock();
    builder.setInsertionPointToEnd(select);
    Value expected =
        arith::ConstantIntOp::create(builder, module.getLoc(), inputIndex, 32);
    Value matches = arith::CmpIOp::create(builder, module.getLoc(),
                                          arith::CmpIPredicate::eq,
                                          entry->getArgument(0), expected);
    cf::CondBranchOp::create(builder, module.getLoc(), matches, valid,
                             ValueRange{}, next, ValueRange{});
    builder.setInsertionPointToEnd(valid);
    Value binding =
        addressOf(builder, module.getLoc(), inputBindingSymbol(inputIndex));
    Value unbound =
        arith::ConstantIntOp::create(builder, module.getLoc(), 0, 32);
    store(builder, module.getLoc(), unbound, binding);
    func::ReturnOp::create(builder, module.getLoc(), unbound);
    select = next;
  }
  builder.setInsertionPointToEnd(invalid);
  failure = arith::ConstantIntOp::create(builder, module.getLoc(), -1, 32);
  func::ReturnOp::create(builder, module.getLoc(), failure);
}

LogicalResult createRunExport(ModuleOp module, const GraphRecord &graph) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  auto i32 = builder.getI32Type();
  auto pointer = host::PtrType::get(module.getContext(), 0);
  auto run =
      func::FuncOp::create(builder, module.getLoc(), "udon_onnx_session_run",
                           FunctionType::get(module.getContext(), {}, {i32}));
  run->setAttr("vrc.async_effect", builder.getStringAttr("may_suspend"));
  Block *entry = run.addEntryBlock();
  Block *cleanup = run.addBlock();
  cleanup->addArgument(i32, module.getLoc());
  builder.setInsertionPointToEnd(entry);

  SmallVector<Value> tensorPointers;
  for (const TensorRecord &tensor : graph.tensors)
    tensorPointers.push_back(
        addressOf(builder, module.getLoc(), tensor.symbol));
  for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph.inputs))
    tensorPointers[tensorIndex] = boundInputPointer(
        builder, module.getLoc(), static_cast<unsigned>(inputIndex),
        graph.tensors[tensorIndex].symbol);
  Value zero = arith::ConstantIntOp::create(builder, module.getLoc(), 0, 32);
  Value one = arith::ConstantFloatOp::create(
      builder, module.getLoc(), builder.getF32Type(), APFloat(1.0f));
  Value dnnSlot = allocaI32(builder, module.getLoc());
  Value blasSlot = allocaI32(builder, module.getLoc());
  Value onnxSlot = allocaI32(builder, module.getLoc());
  store(builder, module.getLoc(), zero, dnnSlot);
  store(builder, module.getLoc(), zero, blasSlot);
  store(builder, module.getLoc(), zero, onnxSlot);
  Value onePointer = allocaF32(builder, module.getLoc(), 1.0f);
  Value zeroPointer = allocaF32(builder, module.getLoc(), 0.0f);
  SmallVector<Value> descriptorSlots(graph.tensors.size());
  SmallVector<unsigned> descriptorRepresentatives;
  llvm::DenseMap<Type, Value> descriptorSlotsByType;
  for (size_t index = 0; index != graph.tensors.size(); ++index) {
    if (!graph.tensors[index].type.getElementType().isF32())
      continue;
    auto [slot, inserted] =
        descriptorSlotsByType.try_emplace(graph.tensors[index].type, Value());
    if (inserted) {
      slot->second = allocaI32(builder, module.getLoc());
      store(builder, module.getLoc(), zero, slot->second);
      descriptorRepresentatives.push_back(index);
    }
    descriptorSlots[index] = slot->second;
  }
  llvm::DenseMap<unsigned, Value> filterSlots;
  llvm::DenseMap<unsigned, Value> convolutionSlots;
  llvm::DenseMap<unsigned, Value> poolingSlots;
  llvm::DenseMap<unsigned, Value> softmaxSlots;
  bool usesBlas = false;
  bool usesONNXOperators = false;
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    usesBlas |= node.operation == "Gemm" || node.operation == "MatMul";
    usesONNXOperators |=
        node.operation == "MatMulNBits" || node.operation == "Gather" ||
        node.operation == "Cast" || node.operation == "Expand" ||
        node.operation == "Transpose" || node.operation == "Concat";
    usesONNXOperators |= node.operation == "Cos" || node.operation == "Sin" ||
                         node.operation == "Neg";
    usesONNXOperators |= node.operation == "Slice";
    usesONNXOperators |= node.operation == "Equal" || node.operation == "Where";
    usesONNXOperators |= node.operation == "ScatterND";
    if (node.operation == "Add" && node.inputs.size() == 2 &&
        node.inputs[0] >= 0 && node.inputs[1] >= 0 &&
        !graph.tensors[node.output].type.getElementType().isF32())
      usesONNXOperators = true;
    if (node.operation == "Conv") {
      filterSlots[index] = allocaI32(builder, module.getLoc());
      convolutionSlots[index] = allocaI32(builder, module.getLoc());
      store(builder, module.getLoc(), zero, filterSlots[index]);
      store(builder, module.getLoc(), zero, convolutionSlots[index]);
    } else if (node.operation == "AveragePool") {
      poolingSlots[index] = allocaI32(builder, module.getLoc());
      store(builder, module.getLoc(), zero, poolingSlots[index]);
    } else if (node.operation == "Softmax") {
      softmaxSlots[index] = allocaI32(builder, module.getLoc());
      store(builder, module.getLoc(), zero, softmaxSlots[index]);
    }
  }

  auto checked = [&](StringRef callee, ValueRange arguments) {
    auto call = func::CallOp::create(builder, module.getLoc(), callee,
                                     TypeRange{i32}, arguments);
    Block *next = run.addBlock();
    Value succeeded = arith::CmpIOp::create(builder, module.getLoc(),
                                            arith::CmpIPredicate::eq,
                                            call.getResult(0), zero);
    cf::CondBranchOp::create(builder, module.getLoc(), succeeded, next,
                             ValueRange{}, cleanup, call.getResults());
    builder.setInsertionPointToEnd(next);
  };

  auto constant = [&](int64_t value) -> Value {
    return arith::ConstantIntOp::create(builder, module.getLoc(), value, 32);
  };
  checked("udondnnCreate", dnnSlot);
  if (usesBlas)
    checked("udonblasCreate", blasSlot);
  if (usesONNXOperators)
    checked("udononnxCreateOperatorHandle", onnxSlot);
  for (unsigned index : descriptorRepresentatives) {
    const TensorRecord &tensor = graph.tensors[index];
    checked("udondnnCreateTensorDescriptor", descriptorSlots[index]);
    SmallVector<int64_t, 4> shape = tensor4d(tensor.type);
    SmallVector<Value> arguments = {
        loadI32(builder, module.getLoc(), descriptorSlots[index]),
        arith::ConstantIntOp::create(builder, module.getLoc(), 0, 32),
        arith::ConstantIntOp::create(builder, module.getLoc(), 0, 32)};
    for (int64_t dimension : shape)
      arguments.push_back(arith::ConstantIntOp::create(builder, module.getLoc(),
                                                       dimension, 32));
    checked("udondnnSetTensor4dDescriptor", arguments);
  }
  Value dnn = loadI32(builder, module.getLoc(), dnnSlot);
  Value blas = loadI32(builder, module.getLoc(), blasSlot);
  Value onnx = loadI32(builder, module.getLoc(), onnxSlot);
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes)) {
    auto tensor = [&](int64_t index) -> const TensorRecord & {
      return graph.tensors[static_cast<unsigned>(index)];
    };
    auto descriptor = [&](int64_t index) -> Value {
      Value slot = descriptorSlots[static_cast<unsigned>(index)];
      assert(slot && "operator requires an FP32 tensor descriptor");
      return loadI32(builder, module.getLoc(), slot);
    };
    auto tensorPointer = [&](int64_t index) -> Value {
      return tensorPointers[static_cast<unsigned>(index)];
    };
    if (isMetadataOnlyAlias(node) && node.inputs.size() >= 1 &&
        node.inputs[0] >= 0) {
      tensorPointers[node.output] = tensorPointer(node.inputs[0]);
      continue;
    }
    if (node.operation == "Add" && node.provider == "udonatenAdd" &&
        node.inputs.size() == 2 && node.inputs[0] >= 0 && node.inputs[1] >= 0 &&
        tensor(node.inputs[0]).type.getElementType().isF32() &&
        tensor(node.inputs[1]).type.getElementType().isF32() &&
        tensor(node.output).type.getElementType().isF32()) {
      checked("udonatenAdd",
              ValueRange{dnn, descriptor(node.inputs[0]),
                         tensorPointer(node.inputs[0]),
                         descriptor(node.inputs[1]),
                         tensorPointer(node.inputs[1]), one,
                         descriptor(node.output), tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "ScatterND" && node.provider == "udononnxScatterND" &&
        node.inputs.size() == 3 && node.inputs[0] >= 0 && node.inputs[1] >= 0 &&
        node.inputs[2] >= 0) {
      const TensorRecord &data = tensor(node.inputs[0]);
      const TensorRecord &indices = tensor(node.inputs[1]);
      const TensorRecord &updates = tensor(node.inputs[2]);
      const TensorRecord &output = tensor(node.output);
      int32_t dataType = udonclDataType(data.type.getElementType());
      int32_t indicesType = udonclDataType(indices.type.getElementType());
      auto reduction = node.attributes
                           ? node.attributes.getAs<StringAttr>("reduction")
                           : StringAttr();
      int64_t depth =
          indices.type.getRank() ? indices.type.getShape().back() : 0;
      int64_t expectedUpdates = 1;
      if (depth > 0 && depth <= data.type.getRank()) {
        expectedUpdates = indices.type.getNumElements() / depth;
        for (int64_t axis = depth; axis < data.type.getRank(); ++axis)
          expectedUpdates *= data.type.getDimSize(axis);
      }
      if (!dataType || (indicesType != 2 && indicesType != 3) || depth <= 0 ||
          depth > data.type.getRank() ||
          data.type.getElementType() != updates.type.getElementType() ||
          data.type != output.type ||
          updates.type.getNumElements() != expectedUpdates ||
          (reduction && reduction.getValue() != "none"))
        return emitError(module.getLoc())
               << "ONNX ScatterND requires reduction=none, INT32/INT64 "
                  "indices, and compatible static data/updates/output shapes";
      Value dataShape =
          allocaI64Array(builder, module.getLoc(), data.type.getShape());
      Value indicesShape =
          allocaI64Array(builder, module.getLoc(), indices.type.getShape());
      checked("udononnxScatterND",
              ValueRange{
                  onnx, constant(dataType), tensorPointer(node.inputs[0]),
                  dataShape, constant(data.type.getRank()),
                  constant(indicesType), tensorPointer(node.inputs[1]),
                  indicesShape, constant(indices.type.getRank()),
                  tensorPointer(node.inputs[2]), tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "Add" && node.provider == "udonatenAdd" &&
        node.inputs.size() == 2 && node.inputs[0] >= 0 && node.inputs[1] >= 0) {
      const TensorRecord &left = tensor(node.inputs[0]);
      const TensorRecord &right = tensor(node.inputs[1]);
      const TensorRecord &output = tensor(node.output);
      int32_t dataType = udonclDataType(left.type.getElementType());
      if (!dataType ||
          left.type.getElementType() != right.type.getElementType() ||
          left.type.getElementType() != output.type.getElementType())
        return emitError(module.getLoc())
               << "ONNX Add requires matching supported scalar types";
      Value leftShape =
          allocaI64Array(builder, module.getLoc(), left.type.getShape());
      Value rightShape =
          allocaI64Array(builder, module.getLoc(), right.type.getShape());
      Value outputShape =
          allocaI64Array(builder, module.getLoc(), output.type.getShape());
      checked("udononnxAdd",
              ValueRange{
                  onnx, constant(dataType), tensorPointer(node.inputs[0]),
                  leftShape, constant(left.type.getRank()),
                  tensorPointer(node.inputs[1]), rightShape,
                  constant(right.type.getRank()), tensorPointers[node.output],
                  outputShape, constant(output.type.getRank())});
      continue;
    }
    if (node.operation == "Mul" && node.provider == "udonatenMul" &&
        node.inputs.size() == 2 && node.inputs[0] >= 0 && node.inputs[1] >= 0) {
      checked("udonatenMul",
              ValueRange{dnn, descriptor(node.inputs[0]),
                         tensorPointer(node.inputs[0]),
                         descriptor(node.inputs[1]),
                         tensorPointer(node.inputs[1]), descriptor(node.output),
                         tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "Relu" && node.provider == "udonatenRelu" &&
        node.inputs.size() == 1 && node.inputs[0] >= 0) {
      checked("udonatenRelu",
              ValueRange{dnn, descriptor(node.inputs[0]),
                         tensorPointer(node.inputs[0]), descriptor(node.output),
                         tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "Tanh" && node.provider == "udonatenTanh" &&
        node.inputs.size() == 1 && node.inputs[0] >= 0) {
      checked("udonatenTanh",
              ValueRange{dnn, descriptor(node.inputs[0]),
                         tensorPointer(node.inputs[0]), descriptor(node.output),
                         tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "Sigmoid" && node.provider == "udonatenSigmoid" &&
        node.inputs.size() == 1 && node.inputs[0] >= 0) {
      checked("udonatenSigmoid",
              ValueRange{dnn, descriptor(node.inputs[0]),
                         tensorPointer(node.inputs[0]), descriptor(node.output),
                         tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "RmsNorm" && node.provider == "udonatenRmsNorm" &&
        node.inputs.size() == 3 && node.inputs[0] >= 0 && node.inputs[1] >= 0 &&
        node.inputs[2] >= 0) {
      const TensorRecord &input = tensor(node.inputs[0]);
      const TensorRecord &weight = tensor(node.inputs[1]);
      const TensorRecord &epsilon = tensor(node.inputs[2]);
      const TensorRecord &output = tensor(node.output);
      if (!input.type.getElementType().isF32() ||
          !weight.type.getElementType().isF32() ||
          !epsilon.type.getElementType().isF32() ||
          !output.type.getElementType().isF32() || input.type != output.type ||
          weight.type.getRank() == 0 || weight.type.getRank() > 4 ||
          epsilon.type.getNumElements() != 1 ||
          input.type.getRank() < weight.type.getRank() ||
          input.type.getShape().take_back(weight.type.getRank()) !=
              weight.type.getShape())
        return emitError(module.getLoc())
               << "RmsNorm requires matching FP32 input/output, a trailing "
                  "FP32 weight shape of rank one through four, and one FP32 "
                  "epsilon";
      Value normalizedShape =
          allocaI64Array(builder, module.getLoc(), weight.type.getShape());
      checked("udonatenRmsNorm",
              ValueRange{
                  dnn, descriptor(node.inputs[0]),
                  tensorPointer(node.inputs[0]), normalizedShape,
                  constant(weight.type.getRank()), descriptor(node.inputs[1]),
                  tensorPointer(node.inputs[1]), tensorPointer(node.inputs[2]),
                  descriptor(node.output), tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "Gather" && node.provider == "udononnxGather" &&
        node.inputs.size() == 2 && node.inputs[0] >= 0 && node.inputs[1] >= 0) {
      const TensorRecord &data = tensor(node.inputs[0]);
      const TensorRecord &indices = tensor(node.inputs[1]);
      const TensorRecord &output = tensor(node.output);
      int64_t axis = integerAttribute(node, "axis", 0);
      if (axis < 0)
        axis += data.type.getRank();
      if (axis < 0 || axis >= data.type.getRank() ||
          !udonclDataType(data.type.getElementType()) ||
          (!indices.type.getElementType().isInteger(32) &&
           !indices.type.getElementType().isInteger(64)) ||
          output.type.getElementType() != data.type.getElementType())
        return emitError(module.getLoc())
               << "ONNX Gather requires a supported data type, INT32 or "
                  "INT64 indices, a valid axis, and matching output type";
      SmallVector<int64_t> expectedShape;
      llvm::append_range(expectedShape, data.type.getShape().take_front(axis));
      llvm::append_range(expectedShape, indices.type.getShape());
      llvm::append_range(expectedShape,
                         data.type.getShape().drop_front(axis + 1));
      if (output.type.getShape() != ArrayRef<int64_t>(expectedShape))
        return emitError(module.getLoc())
               << "ONNX Gather output shape does not match data, indices, "
                  "and axis";
      Value dataShape =
          allocaI64Array(builder, module.getLoc(), data.type.getShape());
      checked(
          "udononnxGather",
          ValueRange{
              onnx, constant(udonclDataType(data.type.getElementType())),
              tensorPointer(node.inputs[0]), dataShape,
              constant(data.type.getRank()),
              constant(udonclDataType(indices.type.getElementType())),
              tensorPointer(node.inputs[1]),
              constant(indices.type.getNumElements()),
              arith::ConstantIntOp::create(builder, module.getLoc(), axis, 64),
              tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "Cast" && node.provider == "udononnxCast" &&
        node.inputs.size() == 1 && node.inputs[0] >= 0) {
      const TensorRecord &input = tensor(node.inputs[0]);
      const TensorRecord &output = tensor(node.output);
      int32_t inputType = udonclDataType(input.type.getElementType());
      int32_t outputType = udonclDataType(output.type.getElementType());
      if (!inputType || !outputType ||
          input.type.getNumElements() != output.type.getNumElements())
        return emitError(module.getLoc())
               << "ONNX Cast requires supported scalar types and matching "
                  "input/output element counts";
      checked("udononnxCast",
              ValueRange{onnx, constant(inputType),
                         tensorPointer(node.inputs[0]), constant(outputType),
                         tensorPointers[node.output],
                         constant(output.type.getNumElements())});
      continue;
    }
    if (node.operation == "Expand" && node.provider == "udononnxExpand" &&
        node.inputs.size() >= 1 && node.inputs[0] >= 0) {
      const TensorRecord &input = tensor(node.inputs[0]);
      const TensorRecord &output = tensor(node.output);
      int32_t dataType = udonclDataType(input.type.getElementType());
      if (!dataType ||
          input.type.getElementType() != output.type.getElementType() ||
          output.type.getRank() < input.type.getRank())
        return emitError(module.getLoc())
               << "ONNX Expand requires a supported matching scalar type "
                  "and output rank at least input rank";
      int64_t rankOffset = output.type.getRank() - input.type.getRank();
      for (int64_t axis = 0; axis < input.type.getRank(); ++axis) {
        int64_t inputExtent = input.type.getDimSize(axis);
        int64_t outputExtent = output.type.getDimSize(axis + rankOffset);
        if (inputExtent != 1 && inputExtent != outputExtent)
          return emitError(module.getLoc())
                 << "ONNX Expand input shape is not broadcast-compatible "
                    "with its output shape";
      }
      Value inputShape =
          allocaI64Array(builder, module.getLoc(), input.type.getShape());
      Value outputShape =
          allocaI64Array(builder, module.getLoc(), output.type.getShape());
      checked("udononnxExpand",
              ValueRange{onnx, constant(dataType),
                         tensorPointer(node.inputs[0]), inputShape,
                         constant(input.type.getRank()),
                         tensorPointers[node.output], outputShape,
                         constant(output.type.getRank())});
      continue;
    }
    if (node.operation == "Transpose" && node.provider == "udononnxTranspose" &&
        node.inputs.size() == 1 && node.inputs[0] >= 0) {
      const TensorRecord &input = tensor(node.inputs[0]);
      const TensorRecord &output = tensor(node.output);
      int32_t dataType = udonclDataType(input.type.getElementType());
      SmallVector<int64_t> permutation =
          integerArrayAttribute(node, "perm", {});
      if (permutation.empty())
        for (int64_t axis = input.type.getRank(); axis-- > 0;)
          permutation.push_back(axis);
      SmallVector<bool> seen(input.type.getRank(), false);
      bool valid =
          dataType &&
          input.type.getElementType() == output.type.getElementType() &&
          input.type.getRank() == output.type.getRank() &&
          permutation.size() == static_cast<size_t>(input.type.getRank());
      for (auto [outputAxis, inputAxis] : llvm::enumerate(permutation)) {
        valid &= inputAxis >= 0 && inputAxis < input.type.getRank() &&
                 !seen[inputAxis];
        if (!valid)
          break;
        seen[inputAxis] = true;
        valid &= output.type.getDimSize(outputAxis) ==
                 input.type.getDimSize(inputAxis);
      }
      if (!valid)
        return emitError(module.getLoc())
               << "ONNX Transpose requires a valid static permutation and "
                  "matching input/output scalar types";
      Value inputShape =
          allocaI64Array(builder, module.getLoc(), input.type.getShape());
      Value permutationValue =
          allocaI64Array(builder, module.getLoc(), permutation);
      checked("udononnxTranspose",
              ValueRange{onnx, constant(dataType),
                         tensorPointer(node.inputs[0]), inputShape,
                         constant(input.type.getRank()), permutationValue,
                         tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "Concat" && node.provider == "udononnxConcat" &&
        !node.inputs.empty()) {
      const TensorRecord &output = tensor(node.output);
      int32_t dataType = udonclDataType(output.type.getElementType());
      int64_t axis = integerAttribute(node, "axis", 0);
      if (axis < 0)
        axis += output.type.getRank();
      if (!dataType || axis < 0 || axis >= output.type.getRank())
        return emitError(module.getLoc())
               << "ONNX Concat requires a supported scalar type and valid "
                  "axis";
      Value outputShape =
          allocaI64Array(builder, module.getLoc(), output.type.getShape());
      int64_t axisOffset = 0;
      for (int64_t inputIndex : node.inputs) {
        if (inputIndex < 0)
          return emitError(module.getLoc())
                 << "ONNX Concat does not accept absent inputs";
        const TensorRecord &input = tensor(inputIndex);
        bool compatible =
            input.type.getElementType() == output.type.getElementType() &&
            input.type.getRank() == output.type.getRank();
        for (int64_t dimension = 0;
             compatible && dimension < output.type.getRank(); ++dimension)
          compatible =
              dimension == axis || input.type.getDimSize(dimension) ==
                                       output.type.getDimSize(dimension);
        if (!compatible)
          return emitError(module.getLoc())
                 << "ONNX Concat input shape or scalar type does not match "
                    "its output";
        Value inputShape =
            allocaI64Array(builder, module.getLoc(), input.type.getShape());
        checked("udononnxConcat",
                ValueRange{onnx, constant(dataType), tensorPointer(inputIndex),
                           inputShape, constant(input.type.getRank()),
                           constant(axis),
                           arith::ConstantIntOp::create(
                               builder, module.getLoc(), axisOffset, 64),
                           tensorPointers[node.output], outputShape});
        axisOffset += input.type.getDimSize(axis);
      }
      if (axisOffset != output.type.getDimSize(axis))
        return emitError(module.getLoc())
               << "ONNX Concat input extents do not fill the output axis";
      continue;
    }
    if ((node.operation == "Cos" || node.operation == "Sin" ||
         node.operation == "Neg") &&
        node.provider == "udononnx" + node.operation &&
        node.inputs.size() == 1 && node.inputs[0] >= 0) {
      const TensorRecord &input = tensor(node.inputs[0]);
      const TensorRecord &output = tensor(node.output);
      if (!input.type.getElementType().isF32() || input.type != output.type)
        return emitError(module.getLoc())
               << "ONNX " << node.operation
               << " requires matching static FP32 input/output tensors";
      checked(node.provider, ValueRange{onnx, tensorPointer(node.inputs[0]),
                                        constant(input.type.getNumElements()),
                                        tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "Slice" && node.provider == "udononnxSlice" &&
        node.inputs.size() >= 3 && node.inputs[0] >= 0 && node.inputs[1] >= 0 &&
        node.inputs[2] >= 0) {
      const TensorRecord &input = tensor(node.inputs[0]);
      const TensorRecord &starts = tensor(node.inputs[1]);
      const TensorRecord &ends = tensor(node.inputs[2]);
      const TensorRecord &output = tensor(node.output);
      int32_t dataType = udonclDataType(input.type.getElementType());
      int64_t sliceRank = starts.type.getNumElements();
      if (!dataType ||
          input.type.getElementType() != output.type.getElementType() ||
          input.type.getRank() != output.type.getRank() ||
          !starts.type.getElementType().isInteger(64) ||
          !ends.type.getElementType().isInteger(64) ||
          ends.type.getNumElements() != sliceRank || sliceRank <= 0 ||
          sliceRank > input.type.getRank())
        return emitError(module.getLoc())
               << "ONNX Slice requires static-rank tensors and equally sized "
                  "INT64 starts/ends";
      Value axes;
      if (node.inputs.size() >= 4 && node.inputs[3] >= 0) {
        const TensorRecord &axesTensor = tensor(node.inputs[3]);
        if (!axesTensor.type.getElementType().isInteger(64) ||
            axesTensor.type.getNumElements() != sliceRank)
          return emitError(module.getLoc())
                 << "ONNX Slice axes must be INT64 and match starts";
        axes = tensorPointer(node.inputs[3]);
      } else {
        SmallVector<int64_t> defaultAxes;
        for (int64_t axis = 0; axis < sliceRank; ++axis)
          defaultAxes.push_back(axis);
        axes = allocaI64Array(builder, module.getLoc(), defaultAxes);
      }
      Value steps;
      if (node.inputs.size() >= 5 && node.inputs[4] >= 0) {
        const TensorRecord &stepsTensor = tensor(node.inputs[4]);
        if (!stepsTensor.type.getElementType().isInteger(64) ||
            stepsTensor.type.getNumElements() != sliceRank)
          return emitError(module.getLoc())
                 << "ONNX Slice steps must be INT64 and match starts";
        steps = tensorPointer(node.inputs[4]);
      } else {
        steps = allocaI64Array(builder, module.getLoc(),
                               SmallVector<int64_t>(sliceRank, 1));
      }
      Value inputShape =
          allocaI64Array(builder, module.getLoc(), input.type.getShape());
      Value outputShape =
          allocaI64Array(builder, module.getLoc(), output.type.getShape());
      checked("udononnxSlice",
              ValueRange{
                  onnx, constant(dataType), tensorPointer(node.inputs[0]),
                  inputShape, constant(input.type.getRank()),
                  tensorPointer(node.inputs[1]), tensorPointer(node.inputs[2]),
                  axes, steps, constant(sliceRank), tensorPointers[node.output],
                  outputShape});
      continue;
    }
    if (node.operation == "Equal" && node.provider == "udononnxEqual" &&
        node.inputs.size() == 2 && node.inputs[0] >= 0 && node.inputs[1] >= 0) {
      const TensorRecord &left = tensor(node.inputs[0]);
      const TensorRecord &right = tensor(node.inputs[1]);
      const TensorRecord &output = tensor(node.output);
      int32_t dataType = udonclDataType(left.type.getElementType());
      if (!dataType ||
          left.type.getElementType() != right.type.getElementType() ||
          !output.type.getElementType().isInteger(1))
        return emitError(module.getLoc())
               << "ONNX Equal requires matching supported input scalar types "
                  "and a boolean output";
      Value leftShape =
          allocaI64Array(builder, module.getLoc(), left.type.getShape());
      Value rightShape =
          allocaI64Array(builder, module.getLoc(), right.type.getShape());
      Value outputShape =
          allocaI64Array(builder, module.getLoc(), output.type.getShape());
      checked("udononnxEqual",
              ValueRange{
                  onnx, constant(dataType), tensorPointer(node.inputs[0]),
                  leftShape, constant(left.type.getRank()),
                  tensorPointer(node.inputs[1]), rightShape,
                  constant(right.type.getRank()), tensorPointers[node.output],
                  outputShape, constant(output.type.getRank())});
      continue;
    }
    if (node.operation == "Where" && node.provider == "udononnxWhere" &&
        node.inputs.size() == 3 && node.inputs[0] >= 0 && node.inputs[1] >= 0 &&
        node.inputs[2] >= 0) {
      const TensorRecord &condition = tensor(node.inputs[0]);
      const TensorRecord &whenTrue = tensor(node.inputs[1]);
      const TensorRecord &whenFalse = tensor(node.inputs[2]);
      const TensorRecord &output = tensor(node.output);
      int32_t dataType = udonclDataType(output.type.getElementType());
      if (!condition.type.getElementType().isInteger(1) || !dataType ||
          whenTrue.type.getElementType() != output.type.getElementType() ||
          whenFalse.type.getElementType() != output.type.getElementType())
        return emitError(module.getLoc())
               << "ONNX Where requires a boolean condition and matching "
                  "supported value/output scalar types";
      Value conditionShape =
          allocaI64Array(builder, module.getLoc(), condition.type.getShape());
      Value trueShape =
          allocaI64Array(builder, module.getLoc(), whenTrue.type.getShape());
      Value falseShape =
          allocaI64Array(builder, module.getLoc(), whenFalse.type.getShape());
      Value outputShape =
          allocaI64Array(builder, module.getLoc(), output.type.getShape());
      checked("udononnxWhere",
              ValueRange{onnx, constant(dataType),
                         tensorPointer(node.inputs[0]), conditionShape,
                         constant(condition.type.getRank()),
                         tensorPointer(node.inputs[1]), trueShape,
                         constant(whenTrue.type.getRank()),
                         tensorPointer(node.inputs[2]), falseShape,
                         constant(whenFalse.type.getRank()),
                         tensorPointers[node.output], outputShape,
                         constant(output.type.getRank())});
      continue;
    }
    if (node.operation == "Conv" &&
        node.provider == "udondnnConvolutionForward" &&
        node.inputs.size() >= 2 && node.inputs[0] >= 0 && node.inputs[1] >= 0) {
      ArrayRef<int64_t> filterShape = tensor(node.inputs[1]).type.getShape();
      if (filterShape.size() != 4)
        return emitError(module.getLoc())
               << "ONNX Conv filter must have rank four";
      SmallVector<int64_t> pads =
          integerArrayAttribute(node, "pads", {0, 0, 0, 0});
      SmallVector<int64_t> strides =
          integerArrayAttribute(node, "strides", {1, 1});
      SmallVector<int64_t> dilations =
          integerArrayAttribute(node, "dilations", {1, 1});
      if (pads.size() != 4 || pads[0] != pads[2] || pads[1] != pads[3] ||
          strides.size() != 2 || dilations.size() != 2)
        return emitError(module.getLoc())
               << "ONNX Conv requires symmetric 2D padding/stride/dilation";
      Value filterSlot = filterSlots.lookup(nodeIndex);
      Value convolutionSlot = convolutionSlots.lookup(nodeIndex);
      checked("udondnnCreateFilterDescriptor", filterSlot);
      checked("udondnnSetFilter4dDescriptor",
              ValueRange{loadI32(builder, module.getLoc(), filterSlot),
                         constant(0), constant(0), constant(filterShape[0]),
                         constant(filterShape[1]), constant(filterShape[2]),
                         constant(filterShape[3])});
      checked("udondnnCreateConvolutionDescriptor", convolutionSlot);
      checked("udondnnSetConvolution2dDescriptor",
              ValueRange{loadI32(builder, module.getLoc(), convolutionSlot),
                         constant(pads[0]), constant(pads[1]),
                         constant(strides[0]), constant(strides[1]),
                         constant(dilations[0]), constant(dilations[1]),
                         constant(1), constant(0)});
      checked("udondnnConvolutionForward",
              ValueRange{dnn, onePointer, descriptor(node.inputs[0]),
                         tensorPointer(node.inputs[0]),
                         loadI32(builder, module.getLoc(), filterSlot),
                         tensorPointer(node.inputs[1]),
                         loadI32(builder, module.getLoc(), convolutionSlot),
                         constant(0), nullPointer(builder, module.getLoc()),
                         constant(0), zeroPointer, descriptor(node.output),
                         tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "AveragePool" &&
        node.provider == "udondnnPoolingForward" && node.inputs.size() == 1 &&
        node.inputs[0] >= 0) {
      SmallVector<int64_t> kernel =
          integerArrayAttribute(node, "kernel_shape", {});
      SmallVector<int64_t> pads =
          integerArrayAttribute(node, "pads", {0, 0, 0, 0});
      SmallVector<int64_t> strides =
          integerArrayAttribute(node, "strides", {1, 1});
      if (kernel.size() != 2 || pads.size() != 4 || pads[0] != pads[2] ||
          pads[1] != pads[3] || strides.size() != 2)
        return emitError(module.getLoc())
               << "ONNX AveragePool requires symmetric 2D attributes";
      int64_t mode = integerAttribute(node, "count_include_pad", 0) ? 1 : 2;
      Value poolingSlot = poolingSlots.lookup(nodeIndex);
      checked("udondnnCreatePoolingDescriptor", poolingSlot);
      checked("udondnnSetPooling2dDescriptor",
              ValueRange{loadI32(builder, module.getLoc(), poolingSlot),
                         constant(mode), constant(1), constant(kernel[0]),
                         constant(kernel[1]), constant(pads[0]),
                         constant(pads[1]), constant(strides[0]),
                         constant(strides[1])});
      checked("udondnnPoolingForward",
              ValueRange{dnn, loadI32(builder, module.getLoc(), poolingSlot),
                         onePointer, descriptor(node.inputs[0]),
                         tensorPointer(node.inputs[0]), zeroPointer,
                         descriptor(node.output), tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "Softmax" &&
        node.provider == "udondnnSoftmaxForward" && node.inputs.size() == 1 &&
        node.inputs[0] >= 0) {
      const TensorRecord &input = tensor(node.inputs[0]);
      const TensorRecord &output = tensor(node.output);
      int64_t axis = integerAttribute(node, "axis", -1);
      if (axis < 0)
        axis += input.type.getRank();
      if (!input.type.getElementType().isF32() || input.type != output.type ||
          axis < 0 || axis >= input.type.getRank())
        return emitError(module.getLoc())
               << "ONNX Softmax requires matching static FP32 tensors and a "
                  "valid axis";
      int64_t outer = 1;
      int64_t inner = 1;
      for (int64_t dimension : input.type.getShape().take_front(axis))
        outer *= dimension;
      for (int64_t dimension : input.type.getShape().drop_front(axis + 1))
        inner *= dimension;
      Value slot = softmaxSlots.lookup(nodeIndex);
      checked("udondnnCreateTensorDescriptor", slot);
      checked("udondnnSetTensor4dDescriptor",
              ValueRange{loadI32(builder, module.getLoc(), slot), constant(0),
                         constant(0), constant(outer),
                         constant(input.type.getDimSize(axis)), constant(inner),
                         constant(1)});
      checked("udondnnSoftmaxForward",
              ValueRange{dnn, constant(1), constant(1), onePointer,
                         loadI32(builder, module.getLoc(), slot),
                         tensorPointer(node.inputs[0]), zeroPointer,
                         loadI32(builder, module.getLoc(), slot),
                         tensorPointers[node.output]});
      continue;
    }
    if (node.operation == "Gemm" && node.provider == "udonblasSgemm" &&
        node.inputs.size() >= 2 && node.inputs[0] >= 0 && node.inputs[1] >= 0) {
      ArrayRef<int64_t> left = tensor(node.inputs[0]).type.getShape();
      ArrayRef<int64_t> right = tensor(node.inputs[1]).type.getShape();
      if (left.size() != 2 || right.size() != 2 || left[1] != right[0])
        return emitError(module.getLoc())
               << "initial ONNX Gemm requires compatible rank-two tensors";
      int64_t transA = integerAttribute(node, "transA", 0);
      int64_t transB = integerAttribute(node, "transB", 0);
      Value alphaPointer = allocaF32(
          builder, module.getLoc(),
          node.attributes && node.attributes.getAs<FloatAttr>("alpha")
              ? static_cast<float>(node.attributes.getAs<FloatAttr>("alpha")
                                       .getValueAsDouble())
              : 1.0f);
      checked("udonblasSgemm",
              ValueRange{blas, constant(transB ? 1 : 0),
                         constant(transA ? 1 : 0), constant(right[1]),
                         constant(left[0]), constant(left[1]), alphaPointer,
                         tensorPointer(node.inputs[1]), constant(right[1]),
                         tensorPointer(node.inputs[0]), constant(left[1]),
                         zeroPointer, tensorPointers[node.output],
                         constant(right[1])});
      if (node.inputs.size() >= 3 && node.inputs[2] >= 0) {
        float beta =
            node.attributes && node.attributes.getAs<FloatAttr>("beta")
                ? static_cast<float>(node.attributes.getAs<FloatAttr>("beta")
                                         .getValueAsDouble())
                : 1.0f;
        checked("udonatenAdd",
                ValueRange{
                    dnn, descriptor(node.output), tensorPointers[node.output],
                    descriptor(node.inputs[2]), tensorPointer(node.inputs[2]),
                    arith::ConstantFloatOp::create(builder, module.getLoc(),
                                                   builder.getF32Type(),
                                                   APFloat(beta)),
                    descriptor(node.output), tensorPointers[node.output]});
      }
      continue;
    }
    if (node.operation == "MatMul" && node.provider == "udonblasSgemm" &&
        node.inputs.size() == 2 && node.inputs[0] >= 0 && node.inputs[1] >= 0) {
      const TensorRecord &leftTensor = tensor(node.inputs[0]);
      const TensorRecord &rightTensor = tensor(node.inputs[1]);
      const TensorRecord &outputTensor = tensor(node.output);
      ArrayRef<int64_t> left = leftTensor.type.getShape();
      ArrayRef<int64_t> right = rightTensor.type.getShape();
      ArrayRef<int64_t> output = outputTensor.type.getShape();
      if (left.size() < 2 || right.size() < 2 ||
          left.back() != right[right.size() - 2] ||
          !leftTensor.type.getElementType().isF32() ||
          !rightTensor.type.getElementType().isF32() ||
          !outputTensor.type.getElementType().isF32())
        return emitError(module.getLoc())
               << "ONNX MatMul requires FP32 [..., M, K] x [..., K, N]";
      int64_t k = left.back();
      int64_t m = left[left.size() - 2];
      int64_t n = right.back();
      int64_t leftBatches = leftTensor.type.getNumElements() / (m * k);
      int64_t rightBatches = rightTensor.type.getNumElements() / (k * n);
      int64_t outputBatches = outputTensor.type.getNumElements() / (m * n);
      if (output.size() < 2 || output[output.size() - 2] != m ||
          output.back() != n ||
          (leftBatches != 1 && leftBatches != outputBatches) ||
          (rightBatches != 1 && rightBatches != outputBatches))
        return emitError(module.getLoc())
               << "ONNX MatMul batch dimensions require contiguous equal or "
                  "whole-tensor broadcast batches";
      if (outputBatches > 1) {
        func::CallOp call = func::CallOp::create(
            builder, module.getLoc(), "udonblasSgemmStridedBatched",
            TypeRange{i32},
            ValueRange{
                blas, constant(0), constant(0), constant(n), constant(m),
                constant(k), onePointer, tensorPointer(node.inputs[1]),
                constant(n),
                arith::ConstantIntOp::create(builder, module.getLoc(),
                                             rightBatches == 1 ? 0 : k * n, 64),
                tensorPointer(node.inputs[0]), constant(k),
                arith::ConstantIntOp::create(builder, module.getLoc(),
                                             leftBatches == 1 ? 0 : m * k, 64),
                zeroPointer, tensorPointers[node.output], constant(n),
                arith::ConstantIntOp::create(builder, module.getLoc(), m * n,
                                             64),
                constant(outputBatches)});
        Block *next = run.addBlock();
        Value succeeded = arith::CmpIOp::create(builder, module.getLoc(),
                                                arith::CmpIPredicate::eq,
                                                call.getResult(0), zero);
        cf::CondBranchOp::create(builder, module.getLoc(), succeeded, next,
                                 ValueRange{}, cleanup, call.getResults());
        builder.setInsertionPointToEnd(next);
        continue;
      }
      checked(
          "udonblasSgemm",
          ValueRange{blas, constant(0), constant(0), constant(n), constant(m),
                     constant(k), onePointer, tensorPointer(node.inputs[1]),
                     constant(n), tensorPointer(node.inputs[0]), constant(k),
                     zeroPointer, tensorPointers[node.output], constant(n)});
      continue;
    }
    if (node.operation == "MatMulNBits" &&
        node.provider == "udononnxMatMulNBits" && node.inputs.size() >= 3 &&
        node.inputs[0] >= 0 && node.inputs[1] >= 0 && node.inputs[2] >= 0) {
      const TensorRecord &activation = tensor(node.inputs[0]);
      const TensorRecord &packedWeight = tensor(node.inputs[1]);
      const TensorRecord &scales = tensor(node.inputs[2]);
      int64_t k = integerAttribute(node, "K", 0);
      int64_t n = integerAttribute(node, "N", 0);
      int64_t blockSize = integerAttribute(node, "block_size", 0);
      int64_t bits = integerAttribute(node, "bits", 0);
      if (!activation.type.getElementType().isF32() || k <= 0 || n <= 0 ||
          activation.type.getNumElements() % k != 0 || blockSize <= 0 ||
          bits < 2 || bits > 8)
        return emitError(module.getLoc())
               << "invalid FP32 MatMulNBits specialization";
      auto optionalPointer = [&](unsigned index) -> Value {
        return index < node.inputs.size() && node.inputs[index] >= 0
                   ? tensorPointer(node.inputs[index])
                   : nullPointer(builder, module.getLoc());
      };
      auto optionalElements = [&](unsigned index) -> int64_t {
        return index < node.inputs.size() && node.inputs[index] >= 0
                   ? tensor(node.inputs[index]).type.getNumElements()
                   : 0;
      };
      checked("udononnxMatMulNBits",
              ValueRange{
                  onnx,
                  constant(6),
                  tensorPointer(node.inputs[0]),
                  arith::ConstantIntOp::create(
                      builder, module.getLoc(),
                      activation.type.getNumElements() / k, 64),
                  arith::ConstantIntOp::create(builder, module.getLoc(), k, 64),
                  tensorPointer(node.inputs[1]),
                  constant(packedWeight.type.getNumElements() *
                           elementByteSize(packedWeight.type.getElementType())),
                  tensorPointer(node.inputs[2]),
                  constant(scales.type.getNumElements()),
                  optionalPointer(3),
                  constant(optionalElements(3)),
                  constant(optionalElements(3) ? 1 : 0),
                  optionalPointer(4),
                  constant(optionalElements(4)),
                  optionalPointer(5),
                  constant(optionalElements(5)),
                  arith::ConstantIntOp::create(builder, module.getLoc(), n, 64),
                  constant(blockSize),
                  constant(bits),
                  tensorPointers[node.output]});
      continue;
    }
    {
      return emitError(module.getLoc())
             << "ONNX Host ABI finalizer does not implement " << node.operation
             << " through " << node.provider;
    }
  }
  cf::BranchOp::create(builder, module.getLoc(), cleanup, zero);

  builder.setInsertionPointToEnd(cleanup);
  for (unsigned index : llvm::reverse(descriptorRepresentatives)) {
    Value slot = descriptorSlots[index];
    func::CallOp::create(builder, module.getLoc(),
                         "udondnnDestroyTensorDescriptor", TypeRange{i32},
                         loadI32(builder, module.getLoc(), slot));
  }
  for (Value slot : poolingSlots.values())
    func::CallOp::create(builder, module.getLoc(),
                         "udondnnDestroyPoolingDescriptor", TypeRange{i32},
                         loadI32(builder, module.getLoc(), slot));
  for (Value slot : softmaxSlots.values())
    func::CallOp::create(builder, module.getLoc(),
                         "udondnnDestroyTensorDescriptor", TypeRange{i32},
                         loadI32(builder, module.getLoc(), slot));
  for (Value slot : convolutionSlots.values())
    func::CallOp::create(builder, module.getLoc(),
                         "udondnnDestroyConvolutionDescriptor", TypeRange{i32},
                         loadI32(builder, module.getLoc(), slot));
  for (Value slot : filterSlots.values())
    func::CallOp::create(builder, module.getLoc(),
                         "udondnnDestroyFilterDescriptor", TypeRange{i32},
                         loadI32(builder, module.getLoc(), slot));
  func::CallOp::create(builder, module.getLoc(), "udondnnDestroy",
                       TypeRange{i32},
                       loadI32(builder, module.getLoc(), dnnSlot));
  if (usesBlas)
    func::CallOp::create(builder, module.getLoc(), "udonblasDestroy",
                         TypeRange{i32},
                         loadI32(builder, module.getLoc(), blasSlot));
  if (usesONNXOperators)
    func::CallOp::create(builder, module.getLoc(),
                         "udononnxDestroyOperatorHandle", TypeRange{i32},
                         loadI32(builder, module.getLoc(), onnxSlot));
  Value completed =
      arith::CmpIOp::create(builder, module.getLoc(), arith::CmpIPredicate::eq,
                            cleanup->getArgument(0), zero);
  Value previous = loadI32(builder, module.getLoc(),
                           addressOf(builder, module.getLoc(), hasRunSymbol));
  Value hasRun = arith::SelectOp::create(builder, module.getLoc(), completed,
                                         constant(1), previous);
  store(builder, module.getLoc(), hasRun,
        addressOf(builder, module.getLoc(), hasRunSymbol));
  func::ReturnOp::create(builder, module.getLoc(), cleanup->getArgument(0));

  declaration(module, "udondnnCreate", {pointer}, {i32});
  declaration(module, "udondnnDestroy", {i32}, {i32});
  declaration(module, "udondnnCreateTensorDescriptor", {pointer}, {i32});
  declaration(module, "udondnnSetTensor4dDescriptor",
              {i32, i32, i32, i32, i32, i32, i32}, {i32});
  declaration(module, "udondnnDestroyTensorDescriptor", {i32}, {i32});
  declaration(module, "udondnnCreateFilterDescriptor", {pointer}, {i32});
  declaration(module, "udondnnSetFilter4dDescriptor",
              {i32, i32, i32, i32, i32, i32, i32}, {i32});
  declaration(module, "udondnnDestroyFilterDescriptor", {i32}, {i32});
  declaration(module, "udondnnCreateConvolutionDescriptor", {pointer}, {i32});
  declaration(module, "udondnnSetConvolution2dDescriptor",
              {i32, i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
  declaration(module, "udondnnDestroyConvolutionDescriptor", {i32}, {i32});
  declaration(module, "udondnnConvolutionForward",
              {i32, pointer, i32, pointer, i32, pointer, i32, i32, pointer, i32,
               pointer, i32, pointer},
              {i32});
  declaration(module, "udondnnCreatePoolingDescriptor", {pointer}, {i32});
  declaration(module, "udondnnSetPooling2dDescriptor",
              {i32, i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
  declaration(module, "udondnnDestroyPoolingDescriptor", {i32}, {i32});
  declaration(module, "udondnnPoolingForward",
              {i32, i32, pointer, i32, pointer, pointer, i32, pointer}, {i32});
  declaration(
      module, "udonatenAdd",
      {i32, i32, pointer, i32, pointer, builder.getF32Type(), i32, pointer},
      {i32});
  declaration(module, "udonatenMul",
              {i32, i32, pointer, i32, pointer, i32, pointer}, {i32});
  declaration(module, "udonatenRelu", {i32, i32, pointer, i32, pointer}, {i32});
  declaration(module, "udonatenTanh", {i32, i32, pointer, i32, pointer}, {i32});
  declaration(module, "udonatenSigmoid", {i32, i32, pointer, i32, pointer},
              {i32});
  declaration(
      module, "udonatenRmsNorm",
      {i32, i32, pointer, pointer, i32, i32, pointer, pointer, i32, pointer},
      {i32});
  declaration(module, "udononnxGather",
              {i32, i32, pointer, pointer, i32, i32, pointer, i32,
               builder.getI64Type(), pointer},
              {i32});
  declaration(module, "udondnnSoftmaxForward",
              {i32, i32, i32, pointer, i32, pointer, pointer, i32, pointer},
              {i32});
  declaration(module, "udononnxCast", {i32, i32, pointer, i32, pointer, i32},
              {i32});
  declaration(module, "udononnxExpand",
              {i32, i32, pointer, pointer, i32, pointer, pointer, i32}, {i32});
  declaration(module, "udononnxTranspose",
              {i32, i32, pointer, pointer, i32, pointer, pointer}, {i32});
  declaration(module, "udononnxConcat",
              {i32, i32, pointer, pointer, i32, i32, builder.getI64Type(),
               pointer, pointer},
              {i32});
  declaration(module, "udononnxCos", {i32, pointer, i32, pointer}, {i32});
  declaration(module, "udononnxSin", {i32, pointer, i32, pointer}, {i32});
  declaration(module, "udononnxNeg", {i32, pointer, i32, pointer}, {i32});
  declaration(module, "udononnxSlice",
              {i32, i32, pointer, pointer, i32, pointer, pointer, pointer,
               pointer, i32, pointer, pointer},
              {i32});
  declaration(module, "udononnxEqual",
              {i32, i32, pointer, pointer, i32, pointer, pointer, i32, pointer,
               pointer, i32},
              {i32});
  declaration(module, "udononnxWhere",
              {i32, i32, pointer, pointer, i32, pointer, pointer, i32, pointer,
               pointer, i32, pointer, pointer, i32},
              {i32});
  declaration(module, "udononnxAdd",
              {i32, i32, pointer, pointer, i32, pointer, pointer, i32, pointer,
               pointer, i32},
              {i32});
  declaration(module, "udononnxScatterND",
              {i32, i32, pointer, pointer, i32, i32, pointer, pointer, i32,
               pointer, pointer},
              {i32});
  declaration(module, "udonblasCreate", {pointer}, {i32});
  declaration(module, "udonblasDestroy", {i32}, {i32});
  declaration(module, "udonblasSgemm",
              {i32, i32, i32, i32, i32, i32, pointer, pointer, i32, pointer,
               i32, pointer, pointer, i32},
              {i32});
  declaration(module, "udonblasSgemmStridedBatched",
              {i32, i32, i32, i32, i32, i32, pointer, pointer, i32,
               builder.getI64Type(), pointer, i32, builder.getI64Type(),
               pointer, pointer, i32, builder.getI64Type(), i32},
              {i32});
  declaration(module, "udononnxCreateOperatorHandle", {pointer}, {i32});
  declaration(module, "udononnxDestroyOperatorHandle", {i32}, {i32});
  declaration(module, "udononnxMatMulNBits",
              {i32,
               i32,
               pointer,
               builder.getI64Type(),
               builder.getI64Type(),
               pointer,
               i32,
               pointer,
               i32,
               pointer,
               i32,
               i32,
               pointer,
               i32,
               pointer,
               i32,
               builder.getI64Type(),
               i32,
               i32,
               pointer},
              {i32});
  return success();
}

void createGeneratedModelExports(ModuleOp module, const GraphRecord &graph) {
  OpBuilder builder(module.getBodyRegion());
  Location location = module.getLoc();
  Type i32 = builder.getI32Type();
  Type pointer = host::PtrType::get(module.getContext(), 0);
  auto constant = [&](int64_t value) -> Value {
    return arith::ConstantIntOp::create(builder, location, value, 32);
  };

  builder.setInsertionPointToEnd(module.getBody());
  auto count =
      func::FuncOp::create(builder, location, "udononnxGeneratedGetModelCount",
                           FunctionType::get(module.getContext(), {}, {i32}));
  Block *countEntry = count.addEntryBlock();
  builder.setInsertionPointToEnd(countEntry);
  func::ReturnOp::create(builder, location, constant(1));

  builder.setInsertionPointToEnd(module.getBody());
  auto get = func::FuncOp::create(
      builder, location, "udononnxGeneratedGetModel",
      FunctionType::get(module.getContext(), {i32, pointer}, {i32}));
  Block *entry = get.addEntryBlock();
  Block *body = get.addBlock();
  Block *invalid = get.addBlock();
  builder.setInsertionPointToEnd(entry);
  Value isModel =
      arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                            entry->getArgument(0), constant(0));
  cf::CondBranchOp::create(builder, location, isModel, body, ValueRange{},
                           invalid, ValueRange{});
  builder.setInsertionPointToEnd(invalid);
  func::ReturnOp::create(builder, location, constant(1));
  builder.setInsertionPointToEnd(body);

  auto returnIfNonzero = [&](Value status) {
    Block *next = get.addBlock();
    Block *failure = get.addBlock();
    Value succeeded = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, status, constant(0));
    cf::CondBranchOp::create(builder, location, succeeded, next, ValueRange{},
                             failure, ValueRange{});
    builder.setInsertionPointToEnd(failure);
    func::ReturnOp::create(builder, location, status);
    builder.setInsertionPointToEnd(next);
  };
  auto initializeMetadata = [&](bool output, unsigned index,
                                const TensorRecord &tensor) {
    StringRef storage = output ? outputMetadataSymbol : inputMetadataSymbol;
    int64_t storageBytes =
        std::max<size_t>(output ? graph.outputs.size() : graph.inputs.size(),
                         1) *
        metadataStorageBytes;
    Value dimensions =
        tensor.type.getRank()
            ? addressOf(builder, location, ioDimensionsSymbol(output, index))
            : nullPointer(builder, location);
    auto call = func::CallOp::create(
        builder, location, "udononnxBindingInitializeTensorMetadata",
        TypeRange{i32},
        ValueRange{addressOf(builder, location, storage),
                   constant(storageBytes), constant(index),
                   addressOf(builder, location, ioNameSymbol(output, index)),
                   constant(udonclDataType(tensor.type.getElementType())),
                   constant(tensor.type.getRank()), dimensions,
                   nullPointer(builder, location)});
    returnIfNonzero(call.getResult(0));
  };
  for (auto [index, tensorIndex] : llvm::enumerate(graph.inputs))
    initializeMetadata(false, index, graph.tensors[tensorIndex]);
  for (auto [index, tensorIndex] : llvm::enumerate(graph.outputs))
    initializeMetadata(true, index, graph.tensors[tensorIndex]);
  auto initializeModel = func::CallOp::create(
      builder, location, "udononnxBindingInitializeModel", TypeRange{i32},
      ValueRange{addressOf(builder, location, modelStorageSymbol),
                 constant(modelStorageBytes),
                 addressOf(builder, location, modelNameSymbol), constant(0),
                 addressOf(builder, location, inputMetadataSymbol),
                 constant(graph.inputs.size()),
                 addressOf(builder, location, outputMetadataSymbol),
                 constant(graph.outputs.size()), entry->getArgument(1)});
  func::ReturnOp::create(builder, location, initializeModel.getResult(0));

  declaration(module, "udononnxBindingInitializeTensorMetadata",
              {pointer, i32, i32, pointer, i32, i32, pointer, pointer}, {i32});
  declaration(module, "udononnxBindingInitializeModel",
              {pointer, i32, pointer, i32, pointer, i32, pointer, i32, pointer},
              {i32});
}

void createPhysicalBindingExports(ModuleOp module, const GraphRecord &graph) {
  OpBuilder builder(module.getBodyRegion());
  Location location = module.getLoc();
  Type i32 = builder.getI32Type();
  Type pointer = host::PtrType::get(module.getContext(), 0);
  auto constant = [&](int64_t value) -> Value {
    return arith::ConstantIntOp::create(builder, location, value, 32);
  };
  auto state = [&]() -> Value {
    return addressOf(builder, location, bindingStateSymbol);
  };
  auto markExport = [&](func::FuncOp function) {
    function->setAttr("udoncl.export", builder.getUnitAttr());
  };
  auto publicResourceBinding = [&](bool output, size_t index,
                                   size_t tensorIndex) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    return builder.getDictionaryAttr({
        builder.getNamedAttr("capacity_bytes",
                             builder.getI64IntegerAttr(
                                 bindingCapacityBytes(graph, tensorIndex))),
        builder.getNamedAttr(
            "data_type", builder.getStringAttr(
                             udonclDataTypeName(tensor.type.getElementType()))),
        builder.getNamedAttr(
            "logical_bytes",
            builder.getI64IntegerAttr(
                tensor.type.getNumElements() *
                elementByteSize(tensor.type.getElementType()))),
        builder.getNamedAttr("name", builder.getStringAttr(tensor.name)),
        builder.getNamedAttr("public_access", builder.getStringAttr(
                                                  output ? "output" : "input")),
        builder.getNamedAttr("public_index", builder.getI32IntegerAttr(
                                                 static_cast<int32_t>(index))),
        builder.getNamedAttr("resource_source",
                             builder.getStringAttr("call_allocation")),
        builder.getNamedAttr("schema_version", builder.getI32IntegerAttr(1)),
    });
  };

  builder.setInsertionPointToEnd(module.getBody());
  auto create = func::FuncOp::create(
      builder, location, "udon_onnx_binding_create",
      FunctionType::get(module.getContext(), {pointer, pointer}, {i32}));
  markExport(create);
  Block *entry = create.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  auto createCall = func::CallOp::create(
      builder, location, "udononnxBindingStateCreate", TypeRange{i32},
      ValueRange{state(), constant(bindingStateStorageBytes),
                 entry->getArgument(0), entry->getArgument(1), constant(0),
                 addressOf(builder, location, bindingInputSymbol),
                 constant(std::max<size_t>(graph.inputs.size(), 1) *
                          bindingTensorStorageBytes),
                 addressOf(builder, location, bindingOutputSymbol),
                 constant(std::max<size_t>(graph.outputs.size(), 1) *
                          bindingTensorStorageBytes)});
  auto returnIfNonzero = [&](func::FuncOp function, Value status) {
    Block *next = function.addBlock();
    Block *failure = function.addBlock();
    Value succeeded = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, status, constant(0));
    cf::CondBranchOp::create(builder, location, succeeded, next, ValueRange{},
                             failure, ValueRange{});
    builder.setInsertionPointToEnd(failure);
    func::ReturnOp::create(builder, location, status);
    builder.setInsertionPointToEnd(next);
  };
  returnIfNonzero(create, createCall.getResult(0));
  auto handleAddress = [&](size_t index) {
    return pointerAdd(builder, location,
                      addressOf(builder, location, bindingIoHandlesSymbol),
                      constant(index * bindingBufferHandleBytes));
  };
  auto allocate = [&](uint32_t access, bool output, size_t publicIndex,
                      size_t tensorIndex, size_t handleIndex) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    auto allocated = func::CallOp::create(
        builder, location, "udononnxBindingStateAllocateBuffer", TypeRange{i32},
        ValueRange{state(), constant(access),
                   constant(std::max<int64_t>(
                       elementByteSize(tensor.type.getElementType()),
                       deviceAllocationBytes(graph, tensorIndex))),
                   handleAddress(handleIndex)});
    allocated->setAttr("vrc.static_public_resource_binding",
                       publicResourceBinding(output, publicIndex, tensorIndex));
    returnIfNonzero(create, allocated.getResult(0));
  };
  size_t handleIndex = 0;
  for (auto [index, tensorIndex] : llvm::enumerate(graph.inputs))
    allocate(isSequenceStateInput(graph, index) ? 2 : 0, false, index,
             tensorIndex, handleIndex++);
  for (auto [index, tensorIndex] : llvm::enumerate(graph.outputs))
    allocate(isSequenceStateOutput(graph, index) ? 2 : 1, true, index,
             tensorIndex, handleIndex++);
  if (!graph.sequenceStates.empty()) {
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStateOwnedSymbol));
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStatePendingSymbol));
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStatePoisonedSymbol));
  }
  func::ReturnOp::create(builder, location, constant(0));

  builder.setInsertionPointToEnd(module.getBody());
  SmallVector<Type> bindArguments(16, i32);
  auto bind = func::FuncOp::create(
      builder, location, "udon_onnx_binding_bind_tensor",
      FunctionType::get(module.getContext(), bindArguments, {i32}));
  markExport(bind);
  Block *bindEntry = bind.addEntryBlock();
  builder.setInsertionPointToEnd(bindEntry);
  Value dimensionBytes = constant(8 * 4);
  OperationState allocation(location, host::AllocaOp::getOperationName());
  allocation.addOperands(dimensionBytes);
  allocation.addTypes(pointer);
  allocation.addAttribute("alignment", builder.getI64IntegerAttr(4));
  Value dimensions = builder.create(allocation)->getResult(0);
  for (int64_t index = 0; index != 8; ++index)
    store(builder, location, bindEntry->getArgument(8 + index),
          pointerAdd(builder, location, dimensions, constant(index * 4)));
  SmallVector<Value> bindValues{state()};
  llvm::append_range(bindValues, bindEntry->getArguments().take_front(8));
  bindValues.push_back(dimensions);
  auto bindCall =
      func::CallOp::create(builder, location, "udononnxBindingStateBindTensor",
                           TypeRange{i32}, bindValues);
  if (!graph.sequenceStates.empty()) {
    Value affected = arith::ConstantIntOp::create(builder, location, 0, 1);
    for (const SequenceStateRecord &sequence : graph.sequenceStates) {
      Value input = arith::AndIOp::create(
          builder, location,
          arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                bindEntry->getArgument(0), constant(0)),
          arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                bindEntry->getArgument(1),
                                constant(sequence.inputIndex)));
      Value output = arith::AndIOp::create(
          builder, location,
          arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                bindEntry->getArgument(0), constant(1)),
          arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                bindEntry->getArgument(1),
                                constant(sequence.outputIndex)));
      affected = arith::OrIOp::create(
          builder, location, affected,
          arith::OrIOp::create(builder, location, input, output));
    }
    Value succeeded =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              bindCall.getResult(0), constant(0));
    Value invalidate =
        arith::AndIOp::create(builder, location, succeeded, affected);
    Value owned = addressOf(builder, location, sequenceStateOwnedSymbol);
    store(builder, location,
          arith::SelectOp::create(builder, location, invalidate, constant(0),
                                  loadI32(builder, location, owned)),
          owned);
  }
  func::ReturnOp::create(builder, location, bindCall.getResult(0));

  builder.setInsertionPointToEnd(module.getBody());
  auto bindAll =
      func::FuncOp::create(builder, location, "udon_onnx_binding_bind_all",
                           FunctionType::get(module.getContext(), {}, {i32}));
  markExport(bindAll);
  entry = bindAll.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  auto bindOne = [&](bool output, size_t publicIndex, unsigned tensorIndex,
                     size_t ioHandleIndex) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    Value dimensionBytes = constant(maximumTensorRank * 4);
    OperationState allocation(location, host::AllocaOp::getOperationName());
    allocation.addOperands(dimensionBytes);
    allocation.addTypes(pointer);
    allocation.addAttribute("alignment", builder.getI64IntegerAttr(4));
    Value dimensions = builder.create(allocation)->getResult(0);
    for (int64_t axis = 0; axis != maximumTensorRank; ++axis) {
      int64_t dimension =
          axis < tensor.type.getRank() ? tensor.type.getShape()[axis] : 0;
      store(builder, location, constant(dimension),
            pointerAdd(builder, location, dimensions, constant(axis * 4)));
    }
    Value handle = handleAddress(ioHandleIndex);
    auto bound = func::CallOp::create(
        builder, location, "udononnxBindingStateBindTensor", TypeRange{i32},
        ValueRange{state(), constant(output), constant(publicIndex),
                   loadI32(builder, location, handle),
                   loadI32(builder, location,
                           pointerAdd(builder, location, handle, constant(4))),
                   constant(0),
                   constant(tensor.type.getNumElements() *
                            elementByteSize(tensor.type.getElementType())),
                   constant(udonclDataType(tensor.type.getElementType())),
                   constant(tensor.type.getRank()), dimensions});
    returnIfNonzero(bindAll, bound.getResult(0));
  };
  handleIndex = 0;
  for (auto [index, tensorIndex] : llvm::enumerate(graph.inputs))
    bindOne(false, index, tensorIndex, handleIndex++);
  for (auto [index, tensorIndex] : llvm::enumerate(graph.outputs))
    bindOne(true, index, tensorIndex, handleIndex++);
  if (!graph.sequenceStates.empty()) {
    store(builder, location, constant(1),
          addressOf(builder, location, sequenceStateOwnedSymbol));
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStatePendingSymbol));
  }
  func::ReturnOp::create(builder, location, constant(0));

  auto createResolve = [&](StringRef name, StringRef callee) {
    builder.setInsertionPointToEnd(module.getBody());
    auto resolve = func::FuncOp::create(
        builder, location, name,
        FunctionType::get(module.getContext(), {i32, pointer, pointer, pointer},
                          {i32}));
    markExport(resolve);
    Block *resolveEntry = resolve.addEntryBlock();
    builder.setInsertionPointToEnd(resolveEntry);
    auto resolved = func::CallOp::create(
        builder, location, callee, TypeRange{i32},
        ValueRange{state(), resolveEntry->getArgument(0),
                   resolveEntry->getArgument(1), resolveEntry->getArgument(2),
                   resolveEntry->getArgument(3)});
    func::ReturnOp::create(builder, location, resolved.getResult(0));
  };
  createResolve("udon_onnx_binding_resolve_input",
                "udononnxBindingStateResolveInput");
  createResolve("udon_onnx_binding_resolve_output",
                "udononnxBindingStateResolveOutput");

  auto createEnqueue = [&](StringRef name, StringRef callee, bool run) {
    builder.setInsertionPointToEnd(module.getBody());
    auto function =
        func::FuncOp::create(builder, location, name,
                             FunctionType::get(module.getContext(), {}, {i32}));
    markExport(function);
    function->setAttr("vrc.async_effect", builder.getStringAttr("may_suspend"));
    Block *enqueueEntry = function.addEntryBlock();
    Block *enqueue = function.addBlock();
    Block *busy = function.addBlock();
    Block *waitEvent = function.addBlock();
    Block *enqueueFailed = function.addBlock();
    builder.setInsertionPointToEnd(enqueueEntry);
    if (run && !graph.sequenceStates.empty()) {
      Block *stateReady = function.addBlock();
      Block *invalidState = function.addBlock();
      Value owned =
          loadI32(builder, location,
                  addressOf(builder, location, sequenceStateOwnedSymbol));
      Value pending =
          loadI32(builder, location,
                  addressOf(builder, location, sequenceStatePendingSymbol));
      Value poisoned =
          loadI32(builder, location,
                  addressOf(builder, location, sequenceStatePoisonedSymbol));
      Value isOwned = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::eq, owned, constant(1));
      Value hasNoPending = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::eq, pending, constant(0));
      Value isHealthy = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::eq, poisoned, constant(0));
      Value valid = arith::AndIOp::create(
          builder, location, isOwned,
          arith::AndIOp::create(builder, location, hasNoPending, isHealthy));
      cf::CondBranchOp::create(builder, location, valid, stateReady,
                               ValueRange{}, invalidState, ValueRange{});
      builder.setInsertionPointToEnd(invalidState);
      func::ReturnOp::create(builder, location, constant(12));
      builder.setInsertionPointToEnd(stateReady);
    }
    Value active = loadI32(builder, location,
                           addressOf(builder, location, bindingEventSymbol));
    Value available = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, active, constant(0));
    cf::CondBranchOp::create(builder, location, available, enqueue,
                             ValueRange{}, busy, ValueRange{});
    builder.setInsertionPointToEnd(busy);
    func::ReturnOp::create(builder, location, constant(7));
    builder.setInsertionPointToEnd(enqueue);
    auto call = func::CallOp::create(
        builder, location, callee, TypeRange{i32},
        ValueRange{state(), constant(0), nullPointer(builder, location),
                   addressOf(builder, location, bindingEventSymbol)});
    Value accepted =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              call.getResult(0), constant(0));
    cf::CondBranchOp::create(builder, location, accepted, waitEvent,
                             ValueRange{}, enqueueFailed, ValueRange{});
    builder.setInsertionPointToEnd(enqueueFailed);
    func::ReturnOp::create(builder, location, call.getResult(0));
    builder.setInsertionPointToEnd(waitEvent);
    Value eventInteger = loadI32(
        builder, location, addressOf(builder, location, bindingEventSymbol));
    auto waited = func::CallOp::create(
        builder, location, "clWaitForEvents", TypeRange{i32},
        ValueRange{constant(1),
                   addressOf(builder, location, bindingEventSymbol)});
    Value waitSucceeded =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              waited.getResult(0), constant(0));
    Value completionStatus = arith::SelectOp::create(
        builder, location, waitSucceeded, constant(0), constant(9));
    auto completed = func::CallOp::create(
        builder, location, "udononnxBindingStateCompleteEvent", TypeRange{i32},
        ValueRange{state(), intToPointer(builder, location, eventInteger),
                   completionStatus});
    auto released = func::CallOp::create(
        builder, location, "clReleaseEvent", TypeRange{i32},
        intToPointer(builder, location, eventInteger));
    store(builder, location, constant(0),
          addressOf(builder, location, bindingEventSymbol));
    Value releaseSucceeded =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              released.getResult(0), constant(0));
    Value finalStatus =
        arith::SelectOp::create(builder, location, releaseSucceeded,
                                completed.getResult(0), constant(5));
    if (!graph.sequenceStates.empty()) {
      Value succeeded =
          arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                finalStatus, constant(0));
      if (run) {
        Value pending =
            addressOf(builder, location, sequenceStatePendingSymbol);
        store(builder, location,
              arith::SelectOp::create(builder, location, succeeded, constant(1),
                                      loadI32(builder, location, pending)),
              pending);
      } else {
        Value pending =
            addressOf(builder, location, sequenceStatePendingSymbol);
        store(builder, location,
              arith::SelectOp::create(builder, location, succeeded, constant(0),
                                      loadI32(builder, location, pending)),
              pending);
        Value poisoned =
            addressOf(builder, location, sequenceStatePoisonedSymbol);
        store(builder, location,
              arith::SelectOp::create(builder, location, succeeded, constant(0),
                                      loadI32(builder, location, poisoned)),
              poisoned);
      }
    }
    func::ReturnOp::create(builder, location, finalStatus);
  };
  createEnqueue("udon_onnx_binding_enqueue_initialize",
                "udononnxBindingStateEnqueueInitialize", false);
  createEnqueue("udon_onnx_binding_enqueue_run",
                "udononnxBindingStateEnqueueRun", true);

  builder.setInsertionPointToEnd(module.getBody());
  auto copyOutputToInput = func::FuncOp::create(
      builder, location, "udon_onnx_binding_enqueue_copy_output_to_input",
      FunctionType::get(module.getContext(), {i32, i32}, {i32}));
  markExport(copyOutputToInput);
  Block *copyEntry = copyOutputToInput.addEntryBlock();
  builder.setInsertionPointToEnd(copyEntry);
  auto copyReturnIfNonzero = [&](Value status, int64_t failureStatus = -1) {
    Block *next = copyOutputToInput.addBlock();
    Block *failure = copyOutputToInput.addBlock();
    Value succeeded = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, status, constant(0));
    cf::CondBranchOp::create(builder, location, succeeded, next, ValueRange{},
                             failure, ValueRange{});
    builder.setInsertionPointToEnd(failure);
    func::ReturnOp::create(builder, location,
                           failureStatus < 0 ? status
                                             : constant(failureStatus));
    builder.setInsertionPointToEnd(next);
  };
  Value sessionSlot = allocaI32(builder, location);
  auto gotSession =
      func::CallOp::create(builder, location, "udononnxBindingStateGetSession",
                           TypeRange{i32}, ValueRange{state(), sessionSlot});
  copyReturnIfNonzero(gotSession.getResult(0));
  auto physicalBuffer = [&](size_t logicalId, int64_t bytes) -> Value {
    return func::CallOp::create(
               builder, location, "__vrc_opencl_buffer_alloc", TypeRange{i32},
               ValueRange{constant(logicalId), constant(bytes)})
        .getResult(0);
  };
  for (auto [outputIndex, outputTensorIndex] : llvm::enumerate(graph.outputs)) {
    const TensorRecord &output = graph.tensors[outputTensorIndex];
    int64_t outputBytes =
        std::max<int64_t>(elementByteSize(output.type.getElementType()),
                          deviceAllocationBytes(graph, outputTensorIndex));
    for (auto [inputIndex, inputTensorIndex] : llvm::enumerate(graph.inputs)) {
      const TensorRecord &input = graph.tensors[inputTensorIndex];
      int64_t inputBytes =
          std::max<int64_t>(elementByteSize(input.type.getElementType()),
                            deviceAllocationBytes(graph, inputTensorIndex));
      if (output.type != input.type || outputBytes != inputBytes)
        continue;
      Block *copy = copyOutputToInput.addBlock();
      Block *next = copyOutputToInput.addBlock();
      Value outputMatches = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::eq,
          copyEntry->getArgument(0), constant(outputIndex));
      Value inputMatches = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::eq,
          copyEntry->getArgument(1), constant(inputIndex));
      Value pairMatches =
          arith::AndIOp::create(builder, location, outputMatches, inputMatches);
      cf::CondBranchOp::create(builder, location, pairMatches, copy,
                               ValueRange{}, next, ValueRange{});
      builder.setInsertionPointToEnd(copy);
      Value source =
          physicalBuffer(graph.inputs.size() + outputIndex + 1, outputBytes);
      Value destination = physicalBuffer(inputIndex + 1, inputBytes);
      auto copied = func::CallOp::create(
          builder, location, "__vrc_opencl_buffer_copy", TypeRange{i32},
          ValueRange{source, destination, constant(0), constant(0),
                     constant(outputBytes)});
      auto awaited =
          func::CallOp::create(builder, location, "__vrc_opencl_await",
                               TypeRange{i32}, copied.getResult(0));
      copyReturnIfNonzero(awaited.getResult(0), 5);
      func::ReturnOp::create(builder, location, constant(0));
      builder.setInsertionPointToEnd(next);
    }
  }
  func::ReturnOp::create(builder, location, constant(1));

  if (!graph.sequenceStates.empty()) {
    builder.setInsertionPointToEnd(module.getBody());
    auto reset = func::FuncOp::create(
        builder, location, "udon_onnx_binding_enqueue_reset_sequence_state",
        FunctionType::get(module.getContext(), {}, {i32}));
    markExport(reset);
    reset->setAttr("vrc.async_effect", builder.getStringAttr("may_suspend"));
    Block *resetEntry = reset.addEntryBlock();
    Block *resetReady = reset.addBlock();
    Block *resetBusy = reset.addBlock();
    Block *resetInvalid = reset.addBlock();
    Block *resetCheckState = reset.addBlock();
    builder.setInsertionPointToEnd(resetEntry);
    Value resetActive = loadI32(
        builder, location, addressOf(builder, location, bindingEventSymbol));
    Value resetIdle = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, resetActive, constant(0));
    cf::CondBranchOp::create(builder, location, resetIdle, resetCheckState,
                             ValueRange{}, resetBusy, ValueRange{});
    builder.setInsertionPointToEnd(resetBusy);
    func::ReturnOp::create(builder, location, constant(7));
    builder.setInsertionPointToEnd(resetCheckState);
    Value resetOwned =
        loadI32(builder, location,
                addressOf(builder, location, sequenceStateOwnedSymbol));
    Value resetValid = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, resetOwned, constant(1));
    cf::CondBranchOp::create(builder, location, resetValid, resetReady,
                             ValueRange{}, resetInvalid, ValueRange{});
    builder.setInsertionPointToEnd(resetInvalid);
    func::ReturnOp::create(builder, location, constant(12));
    builder.setInsertionPointToEnd(resetReady);
    auto resetReturnIfFailed = [&](Value status) {
      Block *next = reset.addBlock();
      Block *failure = reset.addBlock();
      Value succeeded = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::eq, status, constant(0));
      cf::CondBranchOp::create(builder, location, succeeded, next, ValueRange{},
                               failure, ValueRange{});
      builder.setInsertionPointToEnd(failure);
      store(builder, location, constant(1),
            addressOf(builder, location, sequenceStatePoisonedSymbol));
      func::ReturnOp::create(builder, location, constant(5));
      builder.setInsertionPointToEnd(next);
    };
    auto resetPhysicalBuffer = [&](size_t logicalId, int64_t bytes) -> Value {
      return func::CallOp::create(
                 builder, location, "__vrc_opencl_buffer_alloc", TypeRange{i32},
                 ValueRange{constant(logicalId), constant(bytes)})
          .getResult(0);
    };
    for (const SequenceStateRecord &sequence : graph.sequenceStates) {
      unsigned inputTensorIndex = graph.inputs[sequence.inputIndex];
      const TensorRecord &input = graph.tensors[inputTensorIndex];
      int64_t inputBytes = input.type.getNumElements() *
                           elementByteSize(input.type.getElementType());
      Value destination =
          resetPhysicalBuffer(sequence.inputIndex + 1, inputBytes);
      auto uploaded = func::CallOp::create(
          builder, location, "__vrc_opencl_buffer_upload", TypeRange{i32},
          ValueRange{destination, constant(0), constant(inputBytes),
                     addressOf(builder, location, sequenceStateZeroSymbol)});
      auto awaited =
          func::CallOp::create(builder, location, "__vrc_opencl_await",
                               TypeRange{i32}, uploaded.getResult(0));
      resetReturnIfFailed(awaited.getResult(0));
    }
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStatePendingSymbol));
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStatePoisonedSymbol));
    func::ReturnOp::create(builder, location, constant(0));

    builder.setInsertionPointToEnd(module.getBody());
    auto advance = func::FuncOp::create(
        builder, location, "udon_onnx_binding_enqueue_advance_sequence_state",
        FunctionType::get(module.getContext(), {}, {i32}));
    markExport(advance);
    advance->setAttr("vrc.async_effect", builder.getStringAttr("may_suspend"));
    Block *advanceEntry = advance.addEntryBlock();
    builder.setInsertionPointToEnd(advanceEntry);
    Block *advanceReady = advance.addBlock();
    Block *advanceBusy = advance.addBlock();
    Block *advanceInvalid = advance.addBlock();
    Value active = loadI32(builder, location,
                           addressOf(builder, location, bindingEventSymbol));
    Value idle = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, active, constant(0));
    Value owned =
        loadI32(builder, location,
                addressOf(builder, location, sequenceStateOwnedSymbol));
    Value pending =
        loadI32(builder, location,
                addressOf(builder, location, sequenceStatePendingSymbol));
    Value poisoned =
        loadI32(builder, location,
                addressOf(builder, location, sequenceStatePoisonedSymbol));
    Value valid = arith::AndIOp::create(
        builder, location,
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              owned, constant(1)),
        arith::AndIOp::create(
            builder, location,
            arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                  pending, constant(1)),
            arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                  poisoned, constant(0))));
    Block *checkState = advance.addBlock();
    cf::CondBranchOp::create(builder, location, idle, checkState, ValueRange{},
                             advanceBusy, ValueRange{});
    builder.setInsertionPointToEnd(advanceBusy);
    func::ReturnOp::create(builder, location, constant(7));
    builder.setInsertionPointToEnd(checkState);
    cf::CondBranchOp::create(builder, location, valid, advanceReady,
                             ValueRange{}, advanceInvalid, ValueRange{});
    builder.setInsertionPointToEnd(advanceInvalid);
    func::ReturnOp::create(builder, location, constant(12));
    builder.setInsertionPointToEnd(advanceReady);
    auto advanceReturnIfNonzero = [&](Value status,
                                      int64_t failureStatus = -1) {
      Block *next = advance.addBlock();
      Block *failure = advance.addBlock();
      Value succeeded = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::eq, status, constant(0));
      cf::CondBranchOp::create(builder, location, succeeded, next, ValueRange{},
                               failure, ValueRange{});
      builder.setInsertionPointToEnd(failure);
      if (failureStatus == 5)
        store(builder, location, constant(1),
              addressOf(builder, location, sequenceStatePoisonedSymbol));
      func::ReturnOp::create(builder, location,
                             failureStatus < 0 ? status
                                               : constant(failureStatus));
      builder.setInsertionPointToEnd(next);
    };
    Value sessionSlot = allocaI32(builder, location);
    auto gotSession = func::CallOp::create(
        builder, location, "udononnxBindingStateGetSession", TypeRange{i32},
        ValueRange{state(), sessionSlot});
    advanceReturnIfNonzero(gotSession.getResult(0));
    auto physicalBuffer = [&](size_t logicalId, int64_t bytes) -> Value {
      return func::CallOp::create(
                 builder, location, "__vrc_opencl_buffer_alloc", TypeRange{i32},
                 ValueRange{constant(logicalId), constant(bytes)})
          .getResult(0);
    };
    auto packUnsigned = [](ArrayRef<int64_t> values, int64_t first,
                           int64_t fallback) {
      int64_t low = first < static_cast<int64_t>(values.size()) ? values[first]
                                                                : fallback;
      int64_t high = first + 1 < static_cast<int64_t>(values.size())
                         ? values[first + 1]
                         : fallback;
      return (low & 65535) | ((high & 65535) << 16);
    };
    for (const SequenceStateRecord &sequence : graph.sequenceStates) {
      unsigned inputTensorIndex = graph.inputs[sequence.inputIndex];
      unsigned outputTensorIndex = graph.outputs[sequence.outputIndex];
      const TensorRecord &input = graph.tensors[inputTensorIndex];
      const TensorRecord &output = graph.tensors[outputTensorIndex];
      int64_t inputBytes = input.type.getNumElements() * sizeof(float);
      int64_t outputBytes = output.type.getNumElements() * sizeof(float);
      Value source = physicalBuffer(
          graph.inputs.size() + sequence.outputIndex + 1, outputBytes);
      Value destination = physicalBuffer(sequence.inputIndex + 1, inputBytes);
      SmallVector<int64_t> starts(input.type.getRank(), 0);
      SmallVector<int64_t> steps(input.type.getRank(), 1);
      starts[sequence.axis] = sequence.step;
      auto dispatched = func::CallOp::create(
          builder, location, "__vrc_opencl_dispatch_udononnx_slice_f32",
          TypeRange{i32},
          ValueRange{source, destination, constant(input.type.getNumElements()),
                     constant(input.type.getRank()),
                     constant(packUnsigned(output.type.getShape(), 0, 1)),
                     constant(packUnsigned(output.type.getShape(), 2, 1)),
                     constant(packUnsigned(output.type.getShape(), 4, 1)),
                     constant(packUnsigned(input.type.getShape(), 0, 1)),
                     constant(packUnsigned(input.type.getShape(), 2, 1)),
                     constant(packUnsigned(input.type.getShape(), 4, 1)),
                     constant(packUnsigned(starts, 0, 0)),
                     constant(packUnsigned(starts, 2, 0)),
                     constant(packUnsigned(starts, 4, 0)),
                     constant(packUnsigned(steps, 0, 1)),
                     constant(packUnsigned(steps, 2, 1)),
                     constant(packUnsigned(steps, 4, 1))});
      auto awaited =
          func::CallOp::create(builder, location, "__vrc_opencl_await",
                               TypeRange{i32}, dispatched.getResult(0));
      advanceReturnIfNonzero(awaited.getResult(0), 5);
    }
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStatePendingSymbol));
    func::ReturnOp::create(builder, location, constant(0));
  }

  builder.setInsertionPointToEnd(module.getBody());
  auto dispose =
      func::FuncOp::create(builder, location, "udon_onnx_binding_dispose",
                           FunctionType::get(module.getContext(), {}, {i32}));
  markExport(dispose);
  entry = dispose.addEntryBlock();
  Block *disposeState = dispose.addBlock();
  Block *disposeBusy = dispose.addBlock();
  builder.setInsertionPointToEnd(entry);
  Value eventInteger = loadI32(
      builder, location, addressOf(builder, location, bindingEventSymbol));
  Value idle = arith::CmpIOp::create(
      builder, location, arith::CmpIPredicate::eq, eventInteger, constant(0));
  cf::CondBranchOp::create(builder, location, idle, disposeState, ValueRange{},
                           disposeBusy, ValueRange{});
  builder.setInsertionPointToEnd(disposeBusy);
  func::ReturnOp::create(builder, location, constant(7));
  builder.setInsertionPointToEnd(disposeState);
  auto disposed =
      func::CallOp::create(builder, location, "udononnxBindingStateDispose",
                           TypeRange{i32}, state());
  returnIfNonzero(dispose, disposed.getResult(0));
  if (!graph.sequenceStates.empty()) {
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStateOwnedSymbol));
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStatePendingSymbol));
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStatePoisonedSymbol));
  }
  for (size_t index = 0; index != graph.inputs.size() + graph.outputs.size();
       ++index) {
    Value handle = handleAddress(index);
    Value allocated =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ne,
                              loadI32(builder, location, handle), constant(0));
    Block *release = dispose.addBlock();
    Block *next = dispose.addBlock();
    cf::CondBranchOp::create(builder, location, allocated, release,
                             ValueRange{}, next, ValueRange{});
    builder.setInsertionPointToEnd(release);
    auto released = func::CallOp::create(
        builder, location, "udonclDeviceBufferDestroy", TypeRange{i32}, handle);
    Value releaseSucceeded =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              released.getResult(0), constant(0));
    Block *releaseFailure = dispose.addBlock();
    cf::CondBranchOp::create(builder, location, releaseSucceeded, next,
                             ValueRange{}, releaseFailure, ValueRange{});
    builder.setInsertionPointToEnd(releaseFailure);
    func::ReturnOp::create(builder, location, constant(5));
    builder.setInsertionPointToEnd(next);
  }
  func::ReturnOp::create(builder, location, constant(0));

  declaration(module, "udononnxBindingStateCreate",
              {pointer, i32, pointer, pointer, i32, pointer, i32, pointer, i32},
              {i32});
  declaration(module, "udononnxBindingStateAllocateBuffer",
              {pointer, i32, i32, pointer}, {i32});
  declaration(module, "udononnxBindingStateBindTensor",
              {pointer, i32, i32, i32, i32, i32, i32, i32, i32, pointer},
              {i32});
  declaration(module, "udononnxBindingStateEnqueueInitialize",
              {pointer, i32, pointer, pointer}, {i32});
  declaration(module, "udononnxBindingStateEnqueueRun",
              {pointer, i32, pointer, pointer}, {i32});
  declaration(module, "udononnxBindingStateGetSession", {pointer, pointer},
              {i32});
  declaration(module, "udononnxBindingStateResolveInput",
              {pointer, i32, pointer, pointer, pointer}, {i32});
  declaration(module, "udononnxBindingStateResolveOutput",
              {pointer, i32, pointer, pointer, pointer}, {i32});
  declaration(module, "udononnxSessionResolveInput",
              {pointer, i32, pointer, pointer, pointer}, {i32});
  declaration(module, "udononnxSessionResolveOutput",
              {pointer, i32, pointer, pointer, pointer}, {i32});
  declaration(module, "udonclResolvePhysicalBuffer", {pointer, pointer}, {i32});
  declaration(module, "__vrc_opencl_buffer_copy", {i32, i32, i32, i32, i32},
              {i32});
  declaration(module, "__vrc_opencl_buffer_upload", {i32, i32, i32, pointer},
              {i32});
  declaration(module, "__vrc_opencl_buffer_alloc", {i32, i32}, {i32});
  declaration(module, "__vrc_opencl_await", {i32}, {i32});
  declaration(module, "udononnxBindingStateCompleteEvent",
              {pointer, pointer, i32}, {i32});
  declaration(module, "udononnxBindingStateDispose", {pointer}, {i32});
  declaration(module, "udonclDeviceBufferDestroy", {pointer}, {i32});
  declaration(module, "clWaitForEvents", {i32, pointer}, {i32});
  declaration(module, "clReleaseEvent", {pointer}, {i32});
}

LogicalResult createRuntimeDispatchers(ModuleOp module,
                                       const GraphRecord &graph) {
  OpBuilder builder(module.getBodyRegion());
  Location location = module.getLoc();
  Type i32 = builder.getI32Type();
  Type pointer = host::PtrType::get(module.getContext(), 0);
  auto signature =
      FunctionType::get(module.getContext(),
                        {i32, pointer, pointer, i32, pointer, pointer}, {i32});
  auto onnxStatus = [&](int32_t value) -> Value {
    return arith::ConstantIntOp::create(builder, location, value, 32);
  };
  auto graphMatches = [&](Block *entry, Block *valid, Block *invalid) {
    Value expected = arith::ConstantIntOp::create(builder, location, 0, 32);
    Value matches =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              entry->getArgument(0), expected);
    cf::CondBranchOp::create(builder, location, matches, valid, ValueRange{},
                             invalid, ValueRange{});
  };

  builder.setInsertionPointToEnd(module.getBody());
  auto initialize = func::FuncOp::create(
      builder, location, "udononnxGeneratedEnqueueInitialize", signature);
  Block *initializeEntry = initialize.addEntryBlock();
  Block *initializeValid = initialize.addBlock();
  Block *initializeInvalid = initialize.addBlock();
  builder.setInsertionPointToEnd(initializeEntry);
  graphMatches(initializeEntry, initializeValid, initializeInvalid);
  builder.setInsertionPointToEnd(initializeInvalid);
  func::ReturnOp::create(builder, location, onnxStatus(1));
  builder.setInsertionPointToEnd(initializeValid);
  if (supportsDeviceGraph(graph)) {
    Value bufferError = allocaI32(builder, location);
    SmallVector<unsigned> storage = deviceStorage(graph);
    auto storeDeviceBuffer = [&](StringRef symbol, StringRef ownedSymbol,
                                 Value created) {
      Value errorSucceeded = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::eq,
          loadI32(builder, location, bufferError), onnxStatus(0));
      Block *next = initialize.addBlock();
      Block *failure = initialize.addBlock();
      cf::CondBranchOp::create(builder, location, errorSucceeded, next,
                               ValueRange{}, failure, ValueRange{});
      builder.setInsertionPointToEnd(failure);
      func::ReturnOp::create(builder, location, onnxStatus(5));
      builder.setInsertionPointToEnd(next);
      store(builder, location, created, addressOf(builder, location, symbol));
      store(builder, location, onnxStatus(1),
            addressOf(builder, location, ownedSymbol));
    };
    auto allocateDeviceBuffer = [&](StringRef symbol, StringRef ownedSymbol,
                                    int64_t bytes) {
      auto created = func::CallOp::create(
          builder, location, "udonclCreateBufferForQueue", TypeRange{pointer},
          ValueRange{initializeEntry->getArgument(2),
                     arith::ConstantIntOp::create(builder, location, 1, 64),
                     arith::ConstantIntOp::create(builder, location, bytes, 32),
                     bufferError});
      storeDeviceBuffer(symbol, ownedSymbol, created.getResult(0));
    };
    for (auto [tensorIndex, tensor] : llvm::enumerate(graph.tensors)) {
      if (storage[tensorIndex] != tensorIndex)
        continue;
      if (tensor.staticResourceId >= 0)
        continue;
      if (llvm::is_contained(graph.inputs, tensorIndex) ||
          (llvm::is_contained(graph.outputs, tensorIndex) &&
           !needsQ4Surface(graph, tensorIndex)))
        continue;
      allocateDeviceBuffer(deviceBufferSymbol(tensorIndex),
                           deviceBufferOwnedSymbol(tensorIndex),
                           deviceAllocationBytes(graph, tensorIndex));
    }
    if (int64_t bytes = operatorScratchBytes(graph))
      allocateDeviceBuffer(operatorScratchSymbol, operatorScratchOwnedSymbol,
                           bytes);
  }
  auto marker = func::CallOp::create(
      builder, location, "clEnqueueMarkerWithWaitList", TypeRange{i32},
      ValueRange{
          initializeEntry->getArgument(2), initializeEntry->getArgument(3),
          initializeEntry->getArgument(4), initializeEntry->getArgument(5)});
  Value markerSucceeded =
      arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                            marker.getResult(0), onnxStatus(0));
  Value markerStatus = arith::SelectOp::create(
      builder, location, markerSucceeded, onnxStatus(0), onnxStatus(5));
  func::ReturnOp::create(builder, location, markerStatus);

  builder.setInsertionPointToEnd(module.getBody());
  auto disposeSignature =
      FunctionType::get(module.getContext(), {i32, pointer}, {i32});
  auto dispose = func::FuncOp::create(
      builder, location, "udononnxGeneratedDispose", disposeSignature);
  Block *disposeEntry = dispose.addEntryBlock();
  Block *disposeBody = dispose.addBlock();
  Block *disposeInvalid = dispose.addBlock();
  builder.setInsertionPointToEnd(disposeEntry);
  Value expectedGraph = arith::ConstantIntOp::create(builder, location, 0, 32);
  Value validGraph =
      arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                            disposeEntry->getArgument(0), expectedGraph);
  cf::CondBranchOp::create(builder, location, validGraph, disposeBody,
                           ValueRange{}, disposeInvalid, ValueRange{});
  builder.setInsertionPointToEnd(disposeInvalid);
  func::ReturnOp::create(builder, location, onnxStatus(1));
  builder.setInsertionPointToEnd(disposeBody);
  Value disposeStatus = allocaI32(builder, location);
  store(builder, location, onnxStatus(0), disposeStatus);
  SmallVector<unsigned> disposeStorage = deviceStorage(graph);
  auto releaseDeviceBuffer = [&](StringRef symbol, StringRef ownedSymbol) {
    Value address = addressOf(builder, location, symbol);
    Value active = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::ne,
        loadI32(builder, location, addressOf(builder, location, ownedSymbol)),
        onnxStatus(0));
    Block *release = dispose.addBlock();
    Block *next = dispose.addBlock();
    cf::CondBranchOp::create(builder, location, active, release, ValueRange{},
                             next, ValueRange{});
    builder.setInsertionPointToEnd(release);
    Value handle = loadPointer(builder, location, address);
    auto released = func::CallOp::create(
        builder, location, "clReleaseMemObject", TypeRange{i32}, handle);
    Value releaseSucceeded =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              released.getResult(0), onnxStatus(0));
    Value updatedStatus = arith::SelectOp::create(
        builder, location, releaseSucceeded,
        loadI32(builder, location, disposeStatus), onnxStatus(5));
    store(builder, location, updatedStatus, disposeStatus);
    store(builder, location, nullPointer(builder, location), address);
    store(builder, location, onnxStatus(0),
          addressOf(builder, location, ownedSymbol));
    cf::BranchOp::create(builder, location, next);
    builder.setInsertionPointToEnd(next);
  };
  for (auto [tensorIndex, tensor] : llvm::enumerate(graph.tensors)) {
    bool ownsDeviceBuffer = supportsDeviceGraph(graph) &&
                            disposeStorage[tensorIndex] == tensorIndex &&
                            tensor.staticResourceId < 0 &&
                            !llvm::is_contained(graph.inputs, tensorIndex) &&
                            (!llvm::is_contained(graph.outputs, tensorIndex) ||
                             needsQ4Surface(graph, tensorIndex));
    if (!ownsDeviceBuffer)
      continue;
    releaseDeviceBuffer(deviceBufferSymbol(tensorIndex),
                        deviceBufferOwnedSymbol(tensorIndex));
  }
  if (supportsDeviceGraph(graph) && operatorScratchBytes(graph))
    releaseDeviceBuffer(operatorScratchSymbol, operatorScratchOwnedSymbol);
  func::ReturnOp::create(builder, location,
                         loadI32(builder, location, disposeStatus));
  declaration(module, "clReleaseMemObject", {pointer}, {i32});

  builder.setInsertionPointToEnd(module.getBody());
  auto run = func::FuncOp::create(builder, location,
                                  "udononnxGeneratedEnqueueRun", signature);
  run->setAttr("vrc.async_effect", builder.getStringAttr("may_suspend"));
  Block *entry = run.addEntryBlock();
  Block *body = run.addBlock();
  Block *invalidGraph = run.addBlock();
  builder.setInsertionPointToEnd(entry);
  graphMatches(entry, body, invalidGraph);
  builder.setInsertionPointToEnd(invalidGraph);
  func::ReturnOp::create(builder, location, onnxStatus(1));
  builder.setInsertionPointToEnd(body);

  auto returnIfNonzero = [&](Value status) {
    Block *next = run.addBlock();
    Block *failure = run.addBlock();
    Value succeeded = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, status, onnxStatus(0));
    cf::CondBranchOp::create(builder, location, succeeded, next, ValueRange{},
                             failure, ValueRange{});
    builder.setInsertionPointToEnd(failure);
    func::ReturnOp::create(builder, location, status);
    builder.setInsertionPointToEnd(next);
  };
  auto returnOpenCLErrorIfNonzero = [&](Value status) {
    Block *next = run.addBlock();
    Block *failure = run.addBlock();
    Value succeeded = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, status, onnxStatus(0));
    cf::CondBranchOp::create(builder, location, succeeded, next, ValueRange{},
                             failure, ValueRange{});
    builder.setInsertionPointToEnd(failure);
    func::ReturnOp::create(builder, location, onnxStatus(5));
    builder.setInsertionPointToEnd(next);
  };
  auto requireBytes = [&](Value actual, int64_t expected) -> LogicalResult {
    if (expected < 0 || expected > std::numeric_limits<int32_t>::max())
      return emitError(location) << "ONNX tensor byte size exceeds Host ABI";
    Block *next = run.addBlock();
    Block *failure = run.addBlock();
    Value expectedValue = arith::ConstantIntOp::create(
        builder, location, static_cast<int32_t>(expected), 32);
    Value matches = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, actual, expectedValue);
    cf::CondBranchOp::create(builder, location, matches, next, ValueRange{},
                             failure, ValueRange{});
    builder.setInsertionPointToEnd(failure);
    func::ReturnOp::create(builder, location, onnxStatus(2));
    builder.setInsertionPointToEnd(next);
    return success();
  };

  if (supportsDeviceGraph(graph)) {
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIntOp::create(builder, location, value, 32);
    };
    auto floating = [&](float value) -> Value {
      return arith::ConstantFloatOp::create(
          builder, location, builder.getF32Type(), APFloat(value));
    };
    auto requireZero = [&](Value actual) {
      Block *next = run.addBlock();
      Block *failure = run.addBlock();
      Value matches = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::eq, actual, onnxStatus(0));
      cf::CondBranchOp::create(builder, location, matches, next, ValueRange{},
                               failure, ValueRange{});
      builder.setInsertionPointToEnd(failure);
      func::ReturnOp::create(builder, location, onnxStatus(2));
      builder.setInsertionPointToEnd(next);
    };
    SmallVector<unsigned> storage = deviceStorage(graph);
    SmallVector<Value> buffers(graph.tensors.size());
    SmallVector<Value> publicOutputBuffers(graph.tensors.size());
    auto resolvePhysicalBuffer = [&](Value handle) {
      Value tokenSlot = allocaI32(builder, location);
      auto resolved =
          func::CallOp::create(builder, location, "udonclResolvePhysicalBuffer",
                               TypeRange{i32}, ValueRange{handle, tokenSlot});
      returnOpenCLErrorIfNonzero(resolved.getResult(0));
      return loadI32(builder, location, tokenSlot);
    };
    auto resolve = [&](bool output, unsigned bindingIndex,
                       unsigned tensorIndex) -> LogicalResult {
      const TensorRecord &tensor = graph.tensors[tensorIndex];
      int64_t bytes = tensor.type.getNumElements() *
                      elementByteSize(tensor.type.getElementType());
      Value memorySlot = allocaI32(builder, location);
      Value offsetSlot = allocaI32(builder, location);
      Value lengthSlot = allocaI32(builder, location);
      auto resolved = func::CallOp::create(
          builder, location,
          output ? "udononnxSessionResolveOutput"
                 : "udononnxSessionResolveInput",
          TypeRange{i32},
          ValueRange{entry->getArgument(1), constant(bindingIndex), memorySlot,
                     offsetSlot, lengthSlot});
      returnIfNonzero(resolved.getResult(0));
      if (failed(requireBytes(loadI32(builder, location, lengthSlot), bytes)))
        return failure();
      requireZero(loadI32(builder, location, offsetSlot));
      Value resolvedBuffer = intToPointer(
          builder, location, loadI32(builder, location, memorySlot));
      if (output) {
        publicOutputBuffers[tensorIndex] =
            resolvePhysicalBuffer(resolvedBuffer);
        buffers[tensorIndex] = publicOutputBuffers[tensorIndex];
      } else {
        buffers[tensorIndex] = resolvePhysicalBuffer(resolvedBuffer);
      }
      return success();
    };
    for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph.inputs))
      if (failed(resolve(false, inputIndex, tensorIndex)))
        return failure();
    for (auto [outputIndex, tensorIndex] : llvm::enumerate(graph.outputs))
      if (failed(resolve(true, outputIndex, tensorIndex)))
        return failure();
    for (auto [tensorIndex, tensor] : llvm::enumerate(graph.tensors)) {
      if (buffers[tensorIndex] || storage[tensorIndex] != tensorIndex)
        continue;
      if (tensor.staticResourceId >= 0) {
        Value resourceId = arith::ConstantIntOp::create(
            builder, location, tensor.staticResourceId, 32);
        Value bytes = arith::ConstantIntOp::create(builder, location,
                                                   tensor.staticByteLength, 32);
        buffers[tensorIndex] =
            func::CallOp::create(builder, location,
                                 "__vrc_opencl_static_tensor", TypeRange{i32},
                                 ValueRange{resourceId, bytes})
                .getResult(0);
      } else if (!llvm::is_contained(graph.inputs, tensorIndex) &&
                 (!llvm::is_contained(graph.outputs, tensorIndex) ||
                  needsQ4Surface(graph, tensorIndex))) {
        Value handle = loadPointer(
            builder, location,
            addressOf(builder, location, deviceBufferSymbol(tensorIndex)));
        buffers[tensorIndex] = resolvePhysicalBuffer(handle);
      }
    }

    Value dispatchEvent = allocaI32(builder, location);
    auto initialWait = func::CallOp::create(
        builder, location, "clEnqueueMarkerWithWaitList", TypeRange{i32},
        ValueRange{entry->getArgument(2), entry->getArgument(3),
                   entry->getArgument(4), dispatchEvent});
    returnOpenCLErrorIfNonzero(initialWait.getResult(0));
    auto waitAndRelease = [&]() {
      auto waited = func::CallOp::create(
          builder, location, "clWaitForEvents", TypeRange{i32},
          ValueRange{constant(1), dispatchEvent});
      returnOpenCLErrorIfNonzero(waited.getResult(0));
      auto released = func::CallOp::create(
          builder, location, "clReleaseEvent", TypeRange{i32},
          intToPointer(builder, location,
                       loadI32(builder, location, dispatchEvent)));
      returnOpenCLErrorIfNonzero(released.getResult(0));
      store(builder, location, constant(0), dispatchEvent);
    };
    waitAndRelease();
    auto dispatch = [&](StringRef callee, ValueRange arguments) {
      auto enqueued = func::CallOp::create(builder, location, callee,
                                           TypeRange{i32}, arguments);
      auto waited =
          func::CallOp::create(builder, location, "__vrc_opencl_await",
                               TypeRange{i32}, enqueued.getResult(0));
      returnOpenCLErrorIfNonzero(waited.getResult(0));
    };
    auto buffer = [&](int64_t tensorIndex) -> Value {
      unsigned storageIndex = storage[static_cast<unsigned>(tensorIndex)];
      if (buffers[storageIndex])
        return buffers[storageIndex];
      return {};
    };
    Value operatorScratch;
    if (operatorScratchBytes(graph)) {
      Value handle =
          loadPointer(builder, location,
                      addressOf(builder, location, operatorScratchSymbol));
      operatorScratch = resolvePhysicalBuffer(handle);
    }
    for (const NodeRecord &node : graph.nodes) {
      if (isDeviceAlias(graph, node)) {
        continue;
      }
      Value output = buffer(node.output);
      if (!output)
        return emitError(location)
               << "Device ONNX graph output lacks physical storage";
      if (node.operation == "Conv") {
        if (node.inputs.size() < 2 || node.inputs[0] < 0 || node.inputs[1] < 0)
          return emitError(location) << "Device Conv requires input and filter";
        SmallVector<int64_t, 4> inputShape =
            tensor4d(graph.tensors[node.inputs[0]].type);
        SmallVector<int64_t, 4> outputShape =
            tensor4d(graph.tensors[node.output].type);
        ArrayRef<int64_t> filter =
            graph.tensors[node.inputs[1]].type.getShape();
        SmallVector<int64_t> pads =
            integerArrayAttribute(node, "pads", {0, 0, 0, 0});
        SmallVector<int64_t> strides =
            integerArrayAttribute(node, "strides", {1, 1});
        SmallVector<int64_t> dilations =
            integerArrayAttribute(node, "dilations", {1, 1});
        if (filter.size() != 4 || pads.size() != 4 || strides.size() != 2 ||
            dilations.size() != 2 || pads[0] != pads[2] || pads[1] != pads[3])
          return emitError(location)
                 << "Device Conv requires symmetric 2D attributes";
        dispatch("__vrc_opencl_dispatch_udondnn_large_convolution",
                 ValueRange{
                     buffer(node.inputs[0]), buffer(node.inputs[1]),
                     buffer(node.inputs[0]), output,
                     constant(graph.tensors[node.output].type.getNumElements()),
                     constant(inputShape[1] | (inputShape[2] << 16)),
                     constant(inputShape[3]),
                     constant(outputShape[1] | (outputShape[2] << 16)),
                     constant(outputShape[3]),
                     constant(filter[2] | (filter[3] << 16)),
                     constant(pads[0] | (pads[1] << 16)),
                     constant(strides[0] | (strides[1] << 16)),
                     constant(dilations[0] | (dilations[1] << 16)), constant(1),
                     floating(1.0f), floating(0.0f)});
        continue;
      }
      if (node.operation == "Add" || node.operation == "Mul") {
        if (node.inputs.size() != 2 || node.inputs[0] < 0 || node.inputs[1] < 0)
          return emitError(location) << "Device Add requires two inputs";
        SmallVector<int64_t, 4> a =
            broadcastTensor4d(graph.tensors[node.inputs[0]].type);
        SmallVector<int64_t, 4> b =
            broadcastTensor4d(graph.tensors[node.inputs[1]].type);
        SmallVector<int64_t, 4> c =
            broadcastTensor4d(graph.tensors[node.output].type);
        int64_t operation = node.operation == "Mul" ? 1 : 0;
        dispatch(
            "__vrc_opencl_dispatch_udondnn_large_op_tensor",
            ValueRange{
                buffer(node.inputs[0]), buffer(node.inputs[1]),
                buffer(node.inputs[0]), output,
                constant(graph.tensors[node.output].type.getNumElements()),
                constant(a[0] | (a[1] << 16)), constant(a[2] | (a[3] << 16)),
                constant(b[0] | (b[1] << 16)), constant(b[2] | (b[3] << 16)),
                constant(c[0] | (c[1] << 16)), constant(c[2] | (c[3] << 16)),
                constant(operation | (1 << 8)), floating(1.0f), floating(1.0f),
                floating(0.0f)});
        continue;
      }
      if (node.operation == "Relu" || node.operation == "Sigmoid" ||
          node.operation == "Tanh") {
        int32_t mode = node.operation == "Sigmoid" ? 0
                       : node.operation == "Relu"  ? 1
                                                   : 2;
        dispatch("__vrc_opencl_dispatch_udondnn_large_activation",
                 ValueRange{
                     buffer(node.inputs[0]), buffer(node.inputs[0]), output,
                     constant(graph.tensors[node.output].type.getNumElements()),
                     constant(mode), constant(1), floating(0.0f),
                     floating(1.0f), floating(0.0f)});
        continue;
      }
      if (node.operation == "AveragePool") {
        SmallVector<int64_t, 4> inputShape =
            tensor4d(graph.tensors[node.inputs[0]].type);
        SmallVector<int64_t, 4> outputShape =
            tensor4d(graph.tensors[node.output].type);
        SmallVector<int64_t> kernel =
            integerArrayAttribute(node, "kernel_shape", {});
        SmallVector<int64_t> pads =
            integerArrayAttribute(node, "pads", {0, 0, 0, 0});
        SmallVector<int64_t> strides =
            integerArrayAttribute(node, "strides", {1, 1});
        if (kernel.size() != 2 || pads.size() != 4 || strides.size() != 2 ||
            pads[0] != pads[2] || pads[1] != pads[3])
          return emitError(location)
                 << "Device AveragePool requires symmetric 2D attributes";
        int64_t mode = integerAttribute(node, "count_include_pad", 0) ? 1 : 2;
        dispatch("__vrc_opencl_dispatch_udondnn_large_pooling",
                 ValueRange{
                     buffer(node.inputs[0]), buffer(node.inputs[0]), output,
                     constant(graph.tensors[node.output].type.getNumElements()),
                     constant(inputShape[1]), constant(inputShape[2]),
                     constant(inputShape[3]), constant(outputShape[2]),
                     constant(outputShape[3]),
                     constant(kernel[0] | (kernel[1] << 16)),
                     constant(pads[0] | (pads[1] << 16)),
                     constant(strides[0] | (strides[1] << 16)),
                     constant(mode | (1 << 8)), floating(1.0f),
                     floating(0.0f)});
        continue;
      }
      if (node.operation == "Cos" || node.operation == "Sin" ||
          node.operation == "Neg") {
        int32_t operation = node.operation == "Cos"   ? 1
                            : node.operation == "Sin" ? 2
                                                      : 3;
        dispatch(
            "__vrc_opencl_dispatch_udononnx_unary_f32",
            ValueRange{
                buffer(node.inputs[0]), output, constant(operation),
                constant(graph.tensors[node.output].type.getNumElements())});
        continue;
      }
      if (node.operation == "Transpose") {
        std::optional<SmallVector<int64_t>> permutation =
            deviceTransposePermutation(graph, node);
        if (!permutation)
          return emitError(location)
                 << "Device Transpose requires a static FP32 tensor of rank "
                    "one through five and at most 16384 elements";
        ArrayRef<int64_t> shape = graph.tensors[node.inputs[0]].type.getShape();
        auto packDimensions = [&](int64_t first) {
          int64_t low =
              first < static_cast<int64_t>(shape.size()) ? shape[first] : 1;
          int64_t high = first + 1 < static_cast<int64_t>(shape.size())
                             ? shape[first + 1]
                             : 1;
          return low | (high << 16);
        };
        int64_t packedPermutation = 0;
        for (auto [outputAxis, inputAxis] : llvm::enumerate(*permutation))
          packedPermutation |= inputAxis << (outputAxis * 3);
        dispatch("__vrc_opencl_dispatch_udononnx_transpose_f32",
                 ValueRange{
                     buffer(node.inputs[0]), output,
                     constant(graph.tensors[node.output].type.getNumElements()),
                     constant(shape.size()), constant(packDimensions(0)),
                     constant(packDimensions(2)), constant(packDimensions(4)),
                     constant(packedPermutation)});
        continue;
      }
      if (node.operation == "Slice") {
        std::optional<DeviceSliceShape> slice = deviceSliceShape(graph, node);
        if (!slice)
          return emitError(location)
                 << "Device Slice requires static FP32 tensors of rank one "
                    "through five, static slice parameters, and at most "
                    "16384 elements";
        ArrayRef<int64_t> inputShape =
            graph.tensors[node.inputs[0]].type.getShape();
        ArrayRef<int64_t> outputShape =
            graph.tensors[node.output].type.getShape();
        auto packUnsigned = [](ArrayRef<int64_t> values, int64_t first,
                               int64_t fallback) {
          int64_t low = first < static_cast<int64_t>(values.size())
                            ? values[first]
                            : fallback;
          int64_t high = first + 1 < static_cast<int64_t>(values.size())
                             ? values[first + 1]
                             : fallback;
          return (low & 65535) | ((high & 65535) << 16);
        };
        dispatch("__vrc_opencl_dispatch_udononnx_slice_f32",
                 ValueRange{
                     buffer(node.inputs[0]), output,
                     constant(graph.tensors[node.output].type.getNumElements()),
                     constant(inputShape.size()),
                     constant(packUnsigned(inputShape, 0, 1)),
                     constant(packUnsigned(inputShape, 2, 1)),
                     constant(packUnsigned(inputShape, 4, 1)),
                     constant(packUnsigned(outputShape, 0, 1)),
                     constant(packUnsigned(outputShape, 2, 1)),
                     constant(packUnsigned(outputShape, 4, 1)),
                     constant(packUnsigned(slice->starts, 0, 0)),
                     constant(packUnsigned(slice->starts, 2, 0)),
                     constant(packUnsigned(slice->starts, 4, 0)),
                     constant(packUnsigned(slice->steps, 0, 1)),
                     constant(packUnsigned(slice->steps, 2, 1)),
                     constant(packUnsigned(slice->steps, 4, 1))});
        continue;
      }
      if (node.operation == "Expand") {
        if (!isDeviceExpandNode(graph, node))
          return emitError(location)
                 << "Device Expand requires broadcast-compatible static FP32 "
                    "tensors of rank zero through five and at most 16384 "
                    "elements";
        ArrayRef<int64_t> inputShape =
            graph.tensors[node.inputs[0]].type.getShape();
        ArrayRef<int64_t> outputShape =
            graph.tensors[node.output].type.getShape();
        auto packDimensions = [](ArrayRef<int64_t> shape, int64_t first) {
          int64_t low =
              first < static_cast<int64_t>(shape.size()) ? shape[first] : 1;
          int64_t high = first + 1 < static_cast<int64_t>(shape.size())
                             ? shape[first + 1]
                             : 1;
          return (low & 65535) | ((high & 65535) << 16);
        };
        dispatch("__vrc_opencl_dispatch_udononnx_expand_f32",
                 ValueRange{
                     buffer(node.inputs[0]), output,
                     constant(graph.tensors[node.output].type.getNumElements()),
                     constant(inputShape.size()),
                     constant(packDimensions(inputShape, 0)),
                     constant(packDimensions(inputShape, 2)),
                     constant(packDimensions(inputShape, 4)),
                     constant(outputShape.size()),
                     constant(packDimensions(outputShape, 0)),
                     constant(packDimensions(outputShape, 2)),
                     constant(packDimensions(outputShape, 4))});
        continue;
      }
      if (node.operation == "Concat") {
        std::optional<DeviceConcatShape> shape = deviceConcatShape(graph, node);
        if (!shape)
          return emitError(location)
                 << "Device Concat requires two compatible contiguous static "
                    "FP32 tensors of rank one through five and at most 16384 "
                    "output elements";
        dispatch("__vrc_opencl_dispatch_udononnx_concat2_f32",
                 ValueRange{
                     buffer(node.inputs[0]), buffer(node.inputs[1]), output,
                     constant(graph.tensors[node.output].type.getNumElements()),
                     constant(shape->innerWords), constant(shape->leftExtent),
                     constant(shape->rightExtent)});
        continue;
      }
      if (node.operation == "Gather") {
        std::optional<DeviceGatherShape> shape = deviceGatherShape(graph, node);
        if (!shape)
          return emitError(location)
                 << "Device Gather requires immutable FP32 axis-zero data, "
                    "INT32 or INT64 indices, and at most 16384 output words";
        dispatch("__vrc_opencl_dispatch_udononnx_gather_axis0_f32",
                 ValueRange{
                     buffer(node.inputs[0]), buffer(node.inputs[1]), output,
                     constant(graph.tensors[node.output].type.getNumElements()),
                     constant(shape->innerWords), constant(shape->axisExtent),
                     constant(shape->indexWords)});
        continue;
      }
      if (node.operation == "Cast") {
        if (!isDeviceCastI64F32Node(graph, node))
          return emitError(location)
                 << "Device Cast requires matching INT64 input and FP32 "
                    "output shapes of at most 16384 elements";
        dispatch(
            "__vrc_opencl_dispatch_udononnx_cast_i64_f32",
            ValueRange{
                buffer(node.inputs[0]), output,
                constant(graph.tensors[node.output].type.getNumElements())});
        continue;
      }
      if (node.operation == "Equal") {
        if (!isDeviceEqualNode(graph, node))
          return emitError(location)
                 << "Device Equal requires broadcast-compatible static FP32 "
                    "inputs and at most 16384 boolean outputs";
        auto packShape = [](ArrayRef<int64_t> shape, int64_t first) {
          int64_t low =
              first < static_cast<int64_t>(shape.size()) ? shape[first] : 1;
          int64_t high = first + 1 < static_cast<int64_t>(shape.size())
                             ? shape[first + 1]
                             : 1;
          return (low & 65535) | ((high & 65535) << 16);
        };
        const RankedTensorType left = graph.tensors[node.inputs[0]].type;
        const RankedTensorType right = graph.tensors[node.inputs[1]].type;
        const RankedTensorType result = graph.tensors[node.output].type;
        dispatch("__vrc_opencl_dispatch_udononnx_equal_f32",
                 ValueRange{buffer(node.inputs[0]), buffer(node.inputs[1]),
                            output, constant(result.getNumElements()),
                            constant(left.getRank() | (right.getRank() << 3) |
                                     (result.getRank() << 6)),
                            constant(packShape(left.getShape(), 0)),
                            constant(packShape(left.getShape(), 2)),
                            constant(packShape(right.getShape(), 0)),
                            constant(packShape(right.getShape(), 2)),
                            constant(packShape(result.getShape(), 0)),
                            constant(packShape(result.getShape(), 2))});
        continue;
      }
      if (node.operation == "Where") {
        if (!isDeviceWhereNode(graph, node))
          return emitError(location)
                 << "Device Where requires a broadcast-compatible boolean "
                    "condition, static FP32 values, and at most 16384 outputs";
        auto packShape = [](ArrayRef<int64_t> shape, int64_t first) {
          int64_t low =
              first < static_cast<int64_t>(shape.size()) ? shape[first] : 1;
          int64_t high = first + 1 < static_cast<int64_t>(shape.size())
                             ? shape[first + 1]
                             : 1;
          return (low & 65535) | ((high & 65535) << 16);
        };
        const RankedTensorType condition = graph.tensors[node.inputs[0]].type;
        const RankedTensorType whenTrue = graph.tensors[node.inputs[1]].type;
        const RankedTensorType whenFalse = graph.tensors[node.inputs[2]].type;
        const RankedTensorType result = graph.tensors[node.output].type;
        dispatch(
            "__vrc_opencl_dispatch_udononnx_where_f32",
            ValueRange{
                buffer(node.inputs[0]), buffer(node.inputs[1]),
                buffer(node.inputs[2]), output,
                constant(result.getNumElements()),
                constant(condition.getRank() | (whenTrue.getRank() << 3) |
                         (whenFalse.getRank() << 6) | (result.getRank() << 9)),
                constant(packShape(condition.getShape(), 0)),
                constant(packShape(condition.getShape(), 2)),
                constant(packShape(whenTrue.getShape(), 0)),
                constant(packShape(whenTrue.getShape(), 2)),
                constant(packShape(whenFalse.getShape(), 0)),
                constant(packShape(whenFalse.getShape(), 2)),
                constant(packShape(result.getShape(), 0)),
                constant(packShape(result.getShape(), 2))});
        continue;
      }
      if (node.operation == "ScatterND") {
        std::optional<DeviceScatterNDShape> shape =
            deviceScatterNDShape(graph, node);
        if (!shape)
          return emitError(location)
                 << "Device ScatterND requires reduction=none, static FP32 "
                    "data/updates of rank one through four, INT32/INT64 "
                    "indices, and at most 256 update tuples";
        auto packShape = [](ArrayRef<int64_t> values, int64_t first) {
          int64_t low =
              first < static_cast<int64_t>(values.size()) ? values[first] : 1;
          int64_t high = first + 1 < static_cast<int64_t>(values.size())
                             ? values[first + 1]
                             : 1;
          return (low & 65535) | ((high & 65535) << 16);
        };
        const RankedTensorType data = graph.tensors[node.inputs[0]].type;
        dispatch("__vrc_opencl_dispatch_udononnx_scatter_nd_f32",
                 ValueRange{
                     buffer(node.inputs[0]), buffer(node.inputs[1]),
                     buffer(node.inputs[2]), output,
                     constant(data.getNumElements()), constant(data.getRank()),
                     constant(packShape(data.getShape(), 0)),
                     constant(packShape(data.getShape(), 2)),
                     constant(shape->tupleRank), constant(shape->updateCount),
                     constant(shape->sliceWords), constant(shape->indexWords)});
        continue;
      }
      if (isDeviceQ4Node(graph, node)) {
        const TensorRecord &activation = graph.tensors[node.inputs[0]];
        int64_t k = integerAttribute(node, "K", 0);
        int64_t n = integerAttribute(node, "N", 0);
        int64_t m = activation.type.getNumElements() / k;
        int64_t kPasses = (k + 127) / 128;
        int64_t loopStart = 0;
        if (kPasses & 1) {
          int64_t tileK = std::min<int64_t>(128, k);
          dispatch("__vrc_opencl_dispatch_udononnx_matmul_nbits_q4",
                   ValueRange{buffer(node.inputs[0]), buffer(node.inputs[1]),
                              buffer(node.inputs[2]), operatorScratch, output,
                              constant(m), constant(n), constant(k),
                              constant(32), constant(0), constant(0),
                              constant(n), constant(tileK), constant(0)});
          loopStart = 128;
        }
        if (loopStart < k) {
          Block *loop = run.addBlock();
          Block *body = run.addBlock();
          Block *done = run.addBlock();
          BlockArgument kBase = loop->addArgument(i32, location);
          cf::BranchOp::create(builder, location, loop,
                               ValueRange{constant(loopStart)});

          builder.setInsertionPointToEnd(loop);
          Value keepGoing = arith::CmpIOp::create(
              builder, location, arith::CmpIPredicate::slt, kBase, constant(k));
          cf::CondBranchOp::create(builder, location, keepGoing, body,
                                   ValueRange{}, done, ValueRange{});

          builder.setInsertionPointToEnd(body);
          Value hasPrevious = arith::CmpIOp::create(
              builder, location, arith::CmpIPredicate::ne, kBase, constant(0));
          Value accumulate = arith::SelectOp::create(
              builder, location, hasPrevious, constant(1), constant(0));
          dispatch("__vrc_opencl_dispatch_udononnx_matmul_nbits_q4",
                   ValueRange{buffer(node.inputs[0]), buffer(node.inputs[1]),
                              buffer(node.inputs[2]), output, operatorScratch,
                              constant(m), constant(n), constant(k),
                              constant(32), constant(0), kBase, constant(n),
                              constant(128), accumulate});

          Value secondBase =
              arith::AddIOp::create(builder, location, kBase, constant(128));
          dispatch("__vrc_opencl_dispatch_udononnx_matmul_nbits_q4",
                   ValueRange{buffer(node.inputs[0]), buffer(node.inputs[1]),
                              buffer(node.inputs[2]), operatorScratch, output,
                              constant(m), constant(n), constant(k),
                              constant(32), constant(0), secondBase,
                              constant(n), constant(128), constant(1)});
          Value nextBase =
              arith::AddIOp::create(builder, location, kBase, constant(256));
          cf::BranchOp::create(builder, location, loop, ValueRange{nextBase});
          builder.setInsertionPointToEnd(done);
        }
        continue;
      }
      if (isDeviceRmsNormNode(graph, node)) {
        int64_t words = graph.tensors[node.inputs[0]].type.getNumElements();
        int64_t normalizedWords =
            graph.tensors[node.inputs[1]].type.getNumElements();
        float epsilon = *scalarF32(graph.tensors[node.inputs[2]]);
        dispatch("__vrc_opencl_dispatch_udonaten_arena_rms_norm_reduce",
                 ValueRange{buffer(node.inputs[0]), operatorScratch,
                            constant(words), constant(normalizedWords)});
        dispatch("__vrc_opencl_dispatch_udonaten_arena_rms_norm_finalize",
                 ValueRange{buffer(node.inputs[0]), operatorScratch,
                            buffer(node.inputs[1]), output, constant(words),
                            constant(normalizedWords), floating(epsilon),
                            constant(1)});
        continue;
      }
      if (isDeviceSoftmaxNode(graph, node)) {
        const TensorRecord &input = graph.tensors[node.inputs[0]];
        int64_t axis = integerAttribute(node, "axis", -1);
        if (axis < 0)
          axis += input.type.getRank();
        int64_t outer = 1;
        int64_t inner = 1;
        for (int64_t extent : input.type.getShape().take_front(axis))
          outer *= extent;
        for (int64_t extent : input.type.getShape().drop_front(axis + 1))
          inner *= extent;
        dispatch("__vrc_opencl_dispatch_udondnn_arena_softmax",
                 ValueRange{buffer(node.inputs[0]), output, constant(outer),
                            constant(input.type.getDimSize(axis)), constant(1),
                            constant(inner), constant(0), constant(1)});
        continue;
      }
      if (node.operation == "MatMul") {
        std::optional<DeviceMatMulShape> shape = deviceMatMulShape(graph, node);
        if (!shape)
          return emitError(location)
                 << "Device MatMul requires a supported contiguous static "
                    "batch shape";
        if (shape->outputBatches == 1) {
          dispatch("__vrc_opencl_dispatch_udonblas_arena_sgemm",
                   ValueRange{buffer(node.inputs[0]), buffer(node.inputs[1]),
                              output, constant(shape->m), constant(shape->n),
                              constant(shape->k), constant(0), constant(0),
                              floating(1.0f)});
        } else {
          dispatch(
              "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched",
              ValueRange{
                  buffer(node.inputs[0]), buffer(node.inputs[1]), output,
                  constant(shape->m), constant(shape->n), constant(shape->k),
                  constant(shape->leftBatches == 1 ? 0 : shape->m * shape->k),
                  constant(shape->rightBatches == 1 ? 0 : shape->k * shape->n),
                  constant(shape->m * shape->n), constant(shape->outputBatches),
                  constant(0), constant(0), floating(1.0f)});
        }
        continue;
      }
      if (node.operation == "Gemm") {
        ArrayRef<int64_t> left = graph.tensors[node.inputs[0]].type.getShape();
        ArrayRef<int64_t> right = graph.tensors[node.inputs[1]].type.getShape();
        if (left.size() != 2 || right.size() != 2 || left[1] != right[0])
          return emitError(location) << "Device matrix product requires "
                                        "compatible rank-two tensors";
        int64_t transA = integerAttribute(node, "transA", 0);
        int64_t transB = integerAttribute(node, "transB", 0);
        float alpha =
            node.attributes && node.attributes.getAs<FloatAttr>("alpha")
                ? static_cast<float>(node.attributes.getAs<FloatAttr>("alpha")
                                         .getValueAsDouble())
                : 1.0f;
        bool hasBias = node.inputs.size() >= 3 && node.inputs[2] >= 0;
        Value product = hasBias ? operatorScratch : output;
        if (!product)
          return emitError(location)
                 << "Device Gemm with bias lacks operator scratch storage";
        dispatch("__vrc_opencl_dispatch_udonblas_arena_sgemm",
                 ValueRange{buffer(node.inputs[0]), buffer(node.inputs[1]),
                            product, constant(left[0]), constant(right[1]),
                            constant(left[1]), constant(transA),
                            constant(transB), floating(alpha)});
        if (hasBias) {
          SmallVector<int64_t, 4> a = tensor4d(graph.tensors[node.output].type);
          SmallVector<int64_t, 4> b =
              tensor4d(graph.tensors[node.inputs[2]].type);
          float beta =
              node.attributes && node.attributes.getAs<FloatAttr>("beta")
                  ? static_cast<float>(node.attributes.getAs<FloatAttr>("beta")
                                           .getValueAsDouble())
                  : 1.0f;
          dispatch(
              "__vrc_opencl_dispatch_udondnn_large_op_tensor",
              ValueRange{
                  product, buffer(node.inputs[2]), product, output,
                  constant(graph.tensors[node.output].type.getNumElements()),
                  constant(a[0] | (a[1] << 16)), constant(a[2] | (a[3] << 16)),
                  constant(b[0] | (b[1] << 16)), constant(b[2] | (b[3] << 16)),
                  constant(a[0] | (a[1] << 16)), constant(a[2] | (a[3] << 16)),
                  constant(1 << 8), floating(1.0f), floating(beta),
                  floating(0.0f)});
        }
        continue;
      }
    }
    for (unsigned tensorIndex : graph.outputs) {
      Value source = buffer(tensorIndex);
      Value destination = publicOutputBuffers[tensorIndex];
      if (!source || !destination)
        return emitError(location)
               << "Device ONNX graph output lacks a public buffer";
      Value sameBuffer = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::eq, source, destination);
      Block *next = run.addBlock();
      Block *copy = run.addBlock();
      cf::CondBranchOp::create(builder, location, sameBuffer, next,
                               ValueRange{}, copy, ValueRange{});
      builder.setInsertionPointToEnd(copy);
      int64_t bytes =
          graph.tensors[tensorIndex].type.getNumElements() *
          elementByteSize(graph.tensors[tensorIndex].type.getElementType());
      auto copied = func::CallOp::create(
          builder, location, "__vrc_opencl_buffer_copy", TypeRange{i32},
          ValueRange{source, destination, constant(0), constant(0),
                     constant(bytes)});
      auto copyWaited =
          func::CallOp::create(builder, location, "__vrc_opencl_await",
                               TypeRange{i32}, copied.getResult(0));
      returnOpenCLErrorIfNonzero(copyWaited.getResult(0));
      cf::BranchOp::create(builder, location, next);
      builder.setInsertionPointToEnd(next);
    }
    auto completed = func::CallOp::create(
        builder, location, "clEnqueueMarkerWithWaitList", TypeRange{i32},
        ValueRange{entry->getArgument(2), constant(0),
                   nullPointer(builder, location), entry->getArgument(5)});
    returnOpenCLErrorIfNonzero(completed.getResult(0));
    func::ReturnOp::create(builder, location, onnxStatus(0));

    Type f32 = builder.getF32Type();
    declaration(module, "__vrc_opencl_dispatch_udondnn_large_convolution",
                {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
                 i32, i32, f32, f32},
                {i32});
    declaration(module, "__vrc_opencl_dispatch_udondnn_large_activation",
                {i32, i32, i32, i32, i32, i32, f32, f32, f32}, {i32});
    declaration(module, "__vrc_opencl_dispatch_udondnn_large_pooling",
                {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
                 i32, f32, f32},
                {i32});
    declaration(module, "__vrc_opencl_dispatch_udondnn_large_op_tensor",
                {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
                 f32, f32, f32},
                {i32});
    declaration(module, "__vrc_opencl_dispatch_udonblas_arena_sgemm",
                {i32, i32, i32, i32, i32, i32, i32, i32, f32}, {i32});
    declaration(
        module, "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched",
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32},
        {i32});
    declaration(module, "__vrc_opencl_dispatch_udononnx_unary_f32",
                {i32, i32, i32, i32}, {i32});
    declaration(module, "__vrc_opencl_dispatch_udononnx_transpose_f32",
                {i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
    declaration(module, "__vrc_opencl_dispatch_udononnx_slice_f32",
                {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
                 i32, i32, i32, i32},
                {i32});
    declaration(module, "__vrc_opencl_dispatch_udononnx_expand_f32",
                {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
    declaration(module, "__vrc_opencl_dispatch_udononnx_concat2_f32",
                {i32, i32, i32, i32, i32, i32, i32}, {i32});
    declaration(module, "__vrc_opencl_dispatch_udononnx_gather_axis0_f32",
                {i32, i32, i32, i32, i32, i32, i32}, {i32});
    declaration(module, "__vrc_opencl_dispatch_udononnx_cast_i64_f32",
                {i32, i32, i32}, {i32});
    declaration(module, "__vrc_opencl_dispatch_udononnx_equal_f32",
                {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
    declaration(
        module, "__vrc_opencl_dispatch_udononnx_where_f32",
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32},
        {i32});
    declaration(module, "__vrc_opencl_dispatch_udononnx_scatter_nd_f32",
                {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32},
                {i32});
    declaration(
        module, "__vrc_opencl_dispatch_udononnx_matmul_nbits_q4",
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32},
        {i32});
    declaration(module, "__vrc_opencl_dispatch_udonaten_arena_rms_norm_reduce",
                {i32, i32, i32, i32}, {i32});
    declaration(module,
                "__vrc_opencl_dispatch_udonaten_arena_rms_norm_finalize",
                {i32, i32, i32, i32, i32, i32, f32, i32}, {i32});
    declaration(module, "__vrc_opencl_dispatch_udondnn_arena_softmax",
                {i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
    declaration(module, "clEnqueueMarkerWithWaitList",
                {pointer, i32, pointer, pointer}, {i32});
    declaration(module, "__vrc_opencl_static_tensor", {i32, i32}, {i32});
    declaration(module, "udonclCreateBufferForQueue",
                {pointer, builder.getI64Type(), i32, pointer}, {pointer});
    return success();
  }

  for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph.inputs)) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    int64_t bytes = tensor.type.getNumElements() *
                    elementByteSize(tensor.type.getElementType());
    if (!bytes)
      continue;
    Value memorySlot = allocaI32(builder, location);
    Value offsetSlot = allocaI32(builder, location);
    Value lengthSlot = allocaI32(builder, location);
    auto resolved = func::CallOp::create(
        builder, location, "udononnxSessionResolveInput", TypeRange{i32},
        ValueRange{
            entry->getArgument(1),
            arith::ConstantIntOp::create(builder, location, inputIndex, 32),
            memorySlot, offsetSlot, lengthSlot});
    returnIfNonzero(resolved.getResult(0));
    if (failed(requireBytes(loadI32(builder, location, lengthSlot), bytes)))
      return failure();
    auto read = func::CallOp::create(
        builder, location, "clEnqueueReadBuffer", TypeRange{i32},
        ValueRange{entry->getArgument(2),
                   intToPointer(builder, location,
                                loadI32(builder, location, memorySlot)),
                   onnxStatus(1), loadI32(builder, location, offsetSlot),
                   arith::ConstantIntOp::create(builder, location, bytes, 32),
                   addressOf(builder, location, tensor.symbol),
                   entry->getArgument(3), entry->getArgument(4),
                   nullPointer(builder, location)});
    returnOpenCLErrorIfNonzero(read.getResult(0));
  }

  auto execute = func::CallOp::create(
      builder, location, "udon_onnx_session_run", TypeRange{i32}, ValueRange{});
  Block *graphSucceeded = run.addBlock();
  Block *graphFailed = run.addBlock();
  Value graphStatus =
      arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                            execute.getResult(0), onnxStatus(0));
  cf::CondBranchOp::create(builder, location, graphStatus, graphSucceeded,
                           ValueRange{}, graphFailed, ValueRange{});
  builder.setInsertionPointToEnd(graphFailed);
  func::ReturnOp::create(builder, location, onnxStatus(9));
  builder.setInsertionPointToEnd(graphSucceeded);

  SmallVector<unsigned> storage(graph.tensors.size());
  std::iota(storage.begin(), storage.end(), 0u);
  for (const NodeRecord &node : graph.nodes)
    if (isMetadataOnlyAlias(node) && !node.inputs.empty() &&
        node.inputs[0] >= 0)
      storage[node.output] = storage[static_cast<unsigned>(node.inputs[0])];

  for (auto [outputIndex, tensorIndex] : llvm::enumerate(graph.outputs)) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    int64_t bytes = tensor.type.getNumElements() *
                    elementByteSize(tensor.type.getElementType());
    if (!bytes)
      continue;
    Value memorySlot = allocaI32(builder, location);
    Value offsetSlot = allocaI32(builder, location);
    Value lengthSlot = allocaI32(builder, location);
    auto resolved = func::CallOp::create(
        builder, location, "udononnxSessionResolveOutput", TypeRange{i32},
        ValueRange{
            entry->getArgument(1),
            arith::ConstantIntOp::create(builder, location, outputIndex, 32),
            memorySlot, offsetSlot, lengthSlot});
    Block *bound = run.addBlock();
    Block *maybeMissing = run.addBlock();
    Block *nextOutput = run.addBlock();
    Value succeeded =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              resolved.getResult(0), onnxStatus(0));
    cf::CondBranchOp::create(builder, location, succeeded, bound, ValueRange{},
                             maybeMissing, ValueRange{});
    builder.setInsertionPointToEnd(maybeMissing);
    Value missing =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              resolved.getResult(0), onnxStatus(11));
    Block *resolveFailure = run.addBlock();
    cf::CondBranchOp::create(builder, location, missing, nextOutput,
                             ValueRange{}, resolveFailure, ValueRange{});
    builder.setInsertionPointToEnd(resolveFailure);
    func::ReturnOp::create(builder, location, resolved.getResult(0));
    builder.setInsertionPointToEnd(bound);
    if (failed(requireBytes(loadI32(builder, location, lengthSlot), bytes)))
      return failure();
    auto write = func::CallOp::create(
        builder, location, "clEnqueueWriteBuffer", TypeRange{i32},
        ValueRange{entry->getArgument(2),
                   intToPointer(builder, location,
                                loadI32(builder, location, memorySlot)),
                   onnxStatus(1), loadI32(builder, location, offsetSlot),
                   arith::ConstantIntOp::create(builder, location, bytes, 32),
                   addressOf(builder, location,
                             graph.tensors[storage[tensorIndex]].symbol),
                   onnxStatus(0), nullPointer(builder, location),
                   nullPointer(builder, location)});
    returnOpenCLErrorIfNonzero(write.getResult(0));
    cf::BranchOp::create(builder, location, nextOutput);
    builder.setInsertionPointToEnd(nextOutput);
  }

  marker = func::CallOp::create(
      builder, location, "clEnqueueMarkerWithWaitList", TypeRange{i32},
      ValueRange{entry->getArgument(2), onnxStatus(0),
                 nullPointer(builder, location), entry->getArgument(5)});
  returnOpenCLErrorIfNonzero(marker.getResult(0));
  func::ReturnOp::create(builder, location, onnxStatus(0));

  declaration(module, "clEnqueueReadBuffer",
              {pointer, pointer, i32, i32, i32, pointer, i32, pointer, pointer},
              {i32});
  declaration(module, "clEnqueueWriteBuffer",
              {pointer, pointer, i32, i32, i32, pointer, i32, pointer, pointer},
              {i32});
  declaration(module, "clEnqueueMarkerWithWaitList",
              {pointer, i32, pointer, pointer}, {i32});
  declaration(module, "__vrc_opencl_static_tensor", {i32, i32}, {i32});
  declaration(module, "udonclCreateBufferForQueue",
              {pointer, builder.getI64Type(), i32, pointer}, {pointer});
  return success();
}

struct FinalizeONNXHostPass
    : impl::FinalizeONNXHostPassBase<FinalizeONNXHostPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto graph = readGraph(module);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    module.getBody()->getOperations().clear();
    bool deviceGraph = supportsDeviceGraph(*graph);
    if (!deviceGraph) {
      for (const NodeRecord &node : graph->nodes) {
        if (!StringRef(node.provider).starts_with("udonblas"))
          continue;
        module.emitError()
            << "ONNX Host fallback cannot call Device-pointer provider "
            << node.provider << " for " << node.operation;
        signalPassFailure();
        return;
      }
    }
    if (!deviceGraph)
      for (const TensorRecord &tensor : graph->tensors)
        createGlobal(module, tensor);
    createModelMetadataGlobals(module, *graph);
    if (!deviceGraph) {
      for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph->inputs)) {
        (void)tensorIndex;
        createI32Global(module,
                        inputBindingSymbol(static_cast<unsigned>(inputIndex)));
      }
      createI32Global(module, hasRunSymbol);
    }
    SmallVector<unsigned> deviceStorageMap = deviceStorage(*graph);
    for (auto [tensorIndex, tensor] : llvm::enumerate(graph->tensors)) {
      if (deviceGraph && deviceStorageMap[tensorIndex] == tensorIndex &&
          tensor.staticResourceId < 0 &&
          !llvm::is_contained(graph->inputs, tensorIndex) &&
          (!llvm::is_contained(graph->outputs, tensorIndex) ||
           needsQ4Surface(*graph, tensorIndex))) {
        createPointerGlobal(module, deviceBufferSymbol(tensorIndex));
        createI32Global(module, deviceBufferOwnedSymbol(tensorIndex));
      }
    }
    if (deviceGraph && operatorScratchBytes(*graph)) {
      createPointerGlobal(module, operatorScratchSymbol);
      createI32Global(module, operatorScratchOwnedSymbol);
    }
    Builder builder(module.getContext());
    if (!deviceGraph) {
      createWriteExport(module, *graph, builder.getF32Type(), "float");
      createWriteExport(module, *graph, builder.getI32Type(), "int32");
      createWriteExport(module, *graph, builder.getI64Type(), "int64");
      createReadExport(module, *graph, builder.getF32Type(), "float");
      createReadExport(module, *graph, builder.getI32Type(), "int32");
      createReadExport(module, *graph, builder.getI64Type(), "int64");
      createBindingExports(module, *graph);
    }
    createGeneratedModelExports(module, *graph);
    createPhysicalBindingExports(module, *graph);
    if (!deviceGraph && failed(createRunExport(module, *graph))) {
      signalPassFailure();
      return;
    }
    if (failed(createRuntimeDispatchers(module, *graph))) {
      signalPassFailure();
      return;
    }
    SmallVector<int64_t> bufferBytes = mutableBufferBytes(*graph, deviceGraph);
    int64_t handleCapacity = bufferBytes.size();
    if (handleCapacity <= 0 || handleCapacity > maximumMutableHandleCapacity) {
      module.emitError()
          << "ONNX graph requires " << handleCapacity
          << " simultaneously live mutable handles; backend limit is "
          << maximumMutableHandleCapacity;
      signalPassFailure();
      return;
    }
    attachBindingMetadata(module, *graph);
    attachDeviceGraphMetadata(module, *graph, deviceGraph);
    module->setAttr("udoncl.mutable_handle_capacity",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     handleCapacity));
    module->setAttr("udoncl.mutable_buffer_bytes",
                    DenseI64ArrayAttr::get(module.getContext(), bufferBytes));
    module->setAttr("udoncl.onnx_lowering",
                    StringAttr::get(module.getContext(), "library-calls-v2"));
  }
};
} // namespace
} // namespace vrc
