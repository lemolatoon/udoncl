#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"

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

struct GraphRecord {
  SmallVector<TensorRecord> tensors;
  SmallVector<unsigned> inputs;
  SmallVector<unsigned> outputs;
  SmallVector<NodeRecord> nodes;
};

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
        if (!dense || call.getNumOperands() != 0 || call.getNumResults() != 1 ||
            dense.getType() != call.getResult(0).getType() || !resourceId ||
            !byteLength || resourceId.getInt() < 0 ||
            byteLength.getInt() <= 0) {
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
  return result;
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
      builder.getNamedAttr(
          "enqueue_initialize_symbol",
          builder.getStringAttr("udon_onnx_binding_enqueue_initialize")),
      builder.getNamedAttr(
          "enqueue_run_symbol",
          builder.getStringAttr("udon_onnx_binding_enqueue_run")),
      builder.getNamedAttr("dispose_symbol",
                           builder.getStringAttr("udon_onnx_binding_dispose")),
  });
  module->setAttr(
      "udoncl.onnx_binding",
      builder.getDictionaryAttr({
          builder.getNamedAttr("abi",
                               builder.getStringAttr("udononnx-runtime-v1")),
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

void createByteGlobal(ModuleOp module, StringRef symbol, ArrayRef<int8_t> bytes,
                      int64_t alignment = 1) {
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
  builder.create(state);
}

void createZeroGlobal(ModuleOp module, StringRef symbol, size_t bytes,
                      int64_t alignment = 8) {
  createByteGlobal(module, symbol, SmallVector<int8_t>(bytes, 0), alignment);
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
constexpr int32_t metadataStorageBytes = 64;
constexpr int32_t modelStorageBytes = 64;
constexpr int32_t bindingStateStorageBytes = 256;
constexpr int32_t bindingTensorStorageBytes = 128;

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
  createZeroGlobal(module, modelStorageSymbol, modelStorageBytes);
  createZeroGlobal(module, inputMetadataSymbol,
                   std::max<size_t>(graph.inputs.size(), 1) *
                       metadataStorageBytes);
  createZeroGlobal(module, outputMetadataSymbol,
                   std::max<size_t>(graph.outputs.size(), 1) *
                       metadataStorageBytes);
  createStringGlobal(module, modelNameSymbol, "model");
  createZeroGlobal(module, bindingStateSymbol, bindingStateStorageBytes);
  createZeroGlobal(module, bindingInputSymbol,
                   std::max<size_t>(graph.inputs.size(), 1) *
                       bindingTensorStorageBytes);
  createZeroGlobal(module, bindingOutputSymbol,
                   std::max<size_t>(graph.outputs.size(), 1) *
                       bindingTensorStorageBytes);
  createI32Global(module, bindingEventSymbol);
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

constexpr StringLiteral q4ScratchSymbol = "__udon_onnx_q4_scratch";
constexpr int64_t q4SurfaceWords = 131072;
constexpr int64_t q4SurfaceBytes = q4SurfaceWords * 4;

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

bool needsQ4Surface(const GraphRecord &graph, unsigned tensorIndex) {
  return llvm::any_of(graph.nodes, [&](const NodeRecord &node) {
    return isDeviceQ4Node(graph, node) && node.output == tensorIndex;
  });
}

int64_t deviceAllocationBytes(const GraphRecord &graph, unsigned tensorIndex) {
  const TensorRecord &tensor = graph.tensors[tensorIndex];
  int64_t bytes = tensor.type.getNumElements() *
                  elementByteSize(tensor.type.getElementType());
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
    if (graph.tensors[node.output].type.getNumElements() > 32768)
      return false;
    for (int64_t inputIndex : ArrayRef<int64_t>(node.inputs).take_front(2)) {
      const TensorRecord &input = graph.tensors[inputIndex];
      if (input.type.getNumElements() > 32768)
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
  if (node.operation == "Gemm" || node.operation == "MatMul") {
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

  builder.setInsertionPointToEnd(module.getBody());
  auto create =
      func::FuncOp::create(builder, location, "udon_onnx_binding_create",
                           FunctionType::get(module.getContext(), {}, {i32}));
  markExport(create);
  Block *entry = create.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  auto createCall = func::CallOp::create(
      builder, location, "udononnxBindingStateCreate", TypeRange{i32},
      ValueRange{state(), constant(bindingStateStorageBytes), constant(0),
                 addressOf(builder, location, bindingInputSymbol),
                 constant(std::max<size_t>(graph.inputs.size(), 1) *
                          bindingTensorStorageBytes),
                 addressOf(builder, location, bindingOutputSymbol),
                 constant(std::max<size_t>(graph.outputs.size(), 1) *
                          bindingTensorStorageBytes)});
  func::ReturnOp::create(builder, location, createCall.getResult(0));

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
  func::ReturnOp::create(builder, location, bindCall.getResult(0));

  auto createEnqueue = [&](StringRef name, StringRef callee) {
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
    func::ReturnOp::create(builder, location, finalStatus);
  };
  createEnqueue("udon_onnx_binding_enqueue_initialize",
                "udononnxBindingStateEnqueueInitialize");
  createEnqueue("udon_onnx_binding_enqueue_run",
                "udononnxBindingStateEnqueueRun");

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
  func::ReturnOp::create(builder, location, disposed.getResult(0));

  declaration(module, "udononnxBindingStateCreate",
              {pointer, i32, i32, pointer, i32, pointer, i32}, {i32});
  declaration(module, "udononnxBindingStateBindTensor",
              {pointer, i32, i32, i32, i32, i32, i32, i32, i32, pointer},
              {i32});
  declaration(module, "udononnxBindingStateEnqueueInitialize",
              {pointer, i32, pointer, pointer}, {i32});
  declaration(module, "udononnxBindingStateEnqueueRun",
              {pointer, i32, pointer, pointer}, {i32});
  declaration(module, "udononnxBindingStateCompleteEvent",
              {pointer, pointer, i32}, {i32});
  declaration(module, "udononnxBindingStateDispose", {pointer}, {i32});
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
  Value staticError = allocaI32(builder, location);
  for (auto [tensorIndex, tensor] : llvm::enumerate(graph.tensors)) {
    if (tensor.staticResourceId < 0)
      continue;
    Value resourceId = arith::ConstantIntOp::create(
        builder, location, tensor.staticResourceId, 32);
    Value byteLength = arith::ConstantIntOp::create(
        builder, location, tensor.staticByteLength, 32);
    auto physical = func::CallOp::create(
        builder, location, "__vrc_opencl_static_tensor", TypeRange{i32},
        ValueRange{resourceId, byteLength});
    auto created = func::CallOp::create(
        builder, location, "udonclAdoptStaticBuffer", TypeRange{pointer},
        ValueRange{initializeEntry->getArgument(2), resourceId, byteLength,
                   physical.getResult(0), staticError});
    Value handle = pointerToInt(builder, location, created.getResult(0));
    Value errorSucceeded = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq,
        loadI32(builder, location, staticError), onnxStatus(0));
    Value handleValid = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::ne, handle, onnxStatus(0));
    Value succeeded =
        arith::AndIOp::create(builder, location, errorSucceeded, handleValid);
    Block *next = initialize.addBlock();
    Block *failure = initialize.addBlock();
    cf::CondBranchOp::create(builder, location, succeeded, next, ValueRange{},
                             failure, ValueRange{});
    builder.setInsertionPointToEnd(failure);
    func::ReturnOp::create(builder, location, onnxStatus(5));
    builder.setInsertionPointToEnd(next);
    store(builder, location, handle,
          addressOf(builder, location, deviceBufferSymbol(tensorIndex)));
  }
  if (supportsDeviceGraph(graph)) {
    SmallVector<unsigned> storage = deviceStorage(graph);
    auto allocateDeviceBuffer = [&](StringRef symbol, int64_t bytes) {
      auto created = func::CallOp::create(
          builder, location, "udonclCreateBufferForQueue", TypeRange{pointer},
          ValueRange{initializeEntry->getArgument(2),
                     arith::ConstantIntOp::create(builder, location, 1, 64),
                     arith::ConstantIntOp::create(builder, location, bytes, 32),
                     staticError});
      Value handle = pointerToInt(builder, location, created.getResult(0));
      Value errorSucceeded = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::eq,
          loadI32(builder, location, staticError), onnxStatus(0));
      Value handleValid = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::ne, handle, onnxStatus(0));
      Value succeeded =
          arith::AndIOp::create(builder, location, errorSucceeded, handleValid);
      Block *next = initialize.addBlock();
      Block *failure = initialize.addBlock();
      cf::CondBranchOp::create(builder, location, succeeded, next, ValueRange{},
                               failure, ValueRange{});
      builder.setInsertionPointToEnd(failure);
      func::ReturnOp::create(builder, location, onnxStatus(5));
      builder.setInsertionPointToEnd(next);
      store(builder, location, handle, addressOf(builder, location, symbol));
    };
    for (auto [tensorIndex, tensor] : llvm::enumerate(graph.tensors)) {
      if (tensor.staticResourceId >= 0 ||
          llvm::is_contained(graph.inputs, tensorIndex) ||
          (llvm::is_contained(graph.outputs, tensorIndex) &&
           !needsQ4Surface(graph, tensorIndex)) ||
          storage[tensorIndex] != tensorIndex)
        continue;
      allocateDeviceBuffer(deviceBufferSymbol(tensorIndex),
                           deviceAllocationBytes(graph, tensorIndex));
    }
    if (hasDeviceQ4(graph))
      allocateDeviceBuffer(q4ScratchSymbol, q4SurfaceBytes);
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
  auto releaseDeviceBuffer = [&](StringRef symbol) {
    Value address = addressOf(builder, location, symbol);
    Value handle = loadI32(builder, location, address);
    Value active = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::ne, handle, onnxStatus(0));
    Block *release = dispose.addBlock();
    Block *next = dispose.addBlock();
    cf::CondBranchOp::create(builder, location, active, release, ValueRange{},
                             next, ValueRange{});
    builder.setInsertionPointToEnd(release);
    auto released = func::CallOp::create(
        builder, location, "clReleaseMemObject", TypeRange{i32},
        intToPointer(builder, location, handle));
    Value releaseSucceeded =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              released.getResult(0), onnxStatus(0));
    Value updatedStatus = arith::SelectOp::create(
        builder, location, releaseSucceeded,
        loadI32(builder, location, disposeStatus), onnxStatus(5));
    store(builder, location, updatedStatus, disposeStatus);
    store(builder, location, onnxStatus(0), address);
    cf::BranchOp::create(builder, location, next);
    builder.setInsertionPointToEnd(next);
  };
  for (auto [tensorIndex, tensor] : llvm::enumerate(graph.tensors)) {
    bool ownsDeviceBuffer = tensor.staticResourceId >= 0 ||
                            (supportsDeviceGraph(graph) &&
                             !llvm::is_contained(graph.inputs, tensorIndex) &&
                             (!llvm::is_contained(graph.outputs, tensorIndex) ||
                              needsQ4Surface(graph, tensorIndex)) &&
                             disposeStorage[tensorIndex] == tensorIndex);
    if (!ownsDeviceBuffer)
      continue;
    releaseDeviceBuffer(deviceBufferSymbol(tensorIndex));
  }
  if (supportsDeviceGraph(graph) && hasDeviceQ4(graph))
    releaseDeviceBuffer(q4ScratchSymbol);
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
    if (expected <= 0 || expected > std::numeric_limits<int32_t>::max())
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
      if (output)
        publicOutputBuffers[tensorIndex] = resolvedBuffer;
      if (!output || !needsQ4Surface(graph, tensorIndex))
        buffers[tensorIndex] = resolvedBuffer;
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
      if (tensor.staticResourceId >= 0 ||
          (!llvm::is_contained(graph.inputs, tensorIndex) &&
           (!llvm::is_contained(graph.outputs, tensorIndex) ||
            needsQ4Surface(graph, tensorIndex))))
        buffers[tensorIndex] =
            intToPointer(builder, location,
                         loadI32(builder, location,
                                 addressOf(builder, location,
                                           deviceBufferSymbol(tensorIndex))));
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
      SmallVector<Value> values(arguments);
      values.push_back(constant(0));
      values.push_back(nullPointer(builder, location));
      values.push_back(dispatchEvent);
      auto enqueued = func::CallOp::create(builder, location, callee,
                                           TypeRange{i32}, values);
      returnOpenCLErrorIfNonzero(enqueued.getResult(0));
      waitAndRelease();
    };
    auto buffer = [&](int64_t tensorIndex) -> Value {
      return buffers[storage[static_cast<unsigned>(tensorIndex)]];
    };
    Value q4Scratch;
    if (hasDeviceQ4(graph))
      q4Scratch =
          intToPointer(builder, location,
                       loadI32(builder, location,
                               addressOf(builder, location, q4ScratchSymbol)));
    for (const NodeRecord &node : graph.nodes) {
      if (isDeviceAlias(graph, node)) {
        buffers[node.output] = buffer(node.inputs[0]);
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
        dispatch("__vrc_enqueue_udondnn_large_convolution_device_args",
                 ValueRange{
                     entry->getArgument(2), buffer(node.inputs[0]),
                     buffer(node.inputs[1]), output, output,
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
            "__vrc_enqueue_udondnn_large_op_tensor_device_args",
            ValueRange{
                entry->getArgument(2), buffer(node.inputs[0]),
                buffer(node.inputs[1]), output, output,
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
        dispatch("__vrc_enqueue_udondnn_large_activation_device_args",
                 ValueRange{
                     entry->getArgument(2), buffer(node.inputs[0]), output,
                     output,
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
        dispatch(
            "__vrc_enqueue_udondnn_large_pooling_device_args",
            ValueRange{
                entry->getArgument(2), buffer(node.inputs[0]), output, output,
                constant(graph.tensors[node.output].type.getNumElements()),
                constant(inputShape[1]), constant(inputShape[2]),
                constant(inputShape[3]), constant(outputShape[2]),
                constant(outputShape[3]),
                constant(kernel[0] | (kernel[1] << 16)),
                constant(pads[0] | (pads[1] << 16)),
                constant(strides[0] | (strides[1] << 16)),
                constant(mode | (1 << 8)), floating(1.0f), floating(0.0f)});
        continue;
      }
      if (node.operation == "Cos" || node.operation == "Sin" ||
          node.operation == "Neg") {
        int32_t operation = node.operation == "Cos"   ? 1
                            : node.operation == "Sin" ? 2
                                                      : 3;
        dispatch(
            "__vrc_enqueue_udononnx_unary_device_args",
            ValueRange{
                entry->getArgument(2), buffer(node.inputs[0]), output,
                constant(operation),
                constant(graph.tensors[node.output].type.getNumElements())});
        continue;
      }
      if (isDeviceQ4Node(graph, node)) {
        const TensorRecord &activation = graph.tensors[node.inputs[0]];
        int64_t k = integerAttribute(node, "K", 0);
        int64_t n = integerAttribute(node, "N", 0);
        int64_t m = activation.type.getNumElements() / k;
        int64_t kPasses = (k + 127) / 128;
        int64_t passCount = kPasses;
        Value previous = passCount & 1 ? q4Scratch : output;
        Value destination = passCount & 1 ? output : q4Scratch;
        for (int64_t kBase = 0; kBase < k; kBase += 128) {
          int64_t tileK = std::min<int64_t>(128, k - kBase);
          dispatch("__vrc_enqueue_udononnx_matmul_nbits_q4_device_args",
                   ValueRange{entry->getArgument(2), buffer(node.inputs[0]),
                              buffer(node.inputs[1]), buffer(node.inputs[2]),
                              previous, destination, constant(m), constant(n),
                              constant(k), constant(32), constant(0),
                              constant(kBase), constant(n), constant(tileK),
                              constant(kBase ? 1 : 0)});
          std::swap(previous, destination);
        }
        if (previous != output)
          return emitError(location)
                 << "Q4 ping-pong schedule did not finish in output storage";
        continue;
      }
      if (node.operation == "Gemm" || node.operation == "MatMul") {
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
        dispatch("__vrc_enqueue_udonblas_arena_sgemm_device_args",
                 ValueRange{entry->getArgument(2), buffer(node.inputs[0]),
                            buffer(node.inputs[1]), output, constant(left[0]),
                            constant(right[1]), constant(left[1]),
                            constant(transA), constant(transB),
                            floating(alpha)});
        if (node.operation == "Gemm" && node.inputs.size() >= 3 &&
            node.inputs[2] >= 0) {
          SmallVector<int64_t, 4> a = tensor4d(graph.tensors[node.output].type);
          SmallVector<int64_t, 4> b =
              tensor4d(graph.tensors[node.inputs[2]].type);
          float beta =
              node.attributes && node.attributes.getAs<FloatAttr>("beta")
                  ? static_cast<float>(node.attributes.getAs<FloatAttr>("beta")
                                           .getValueAsDouble())
                  : 1.0f;
          dispatch(
              "__vrc_enqueue_udondnn_large_op_tensor_device_args",
              ValueRange{
                  entry->getArgument(2), output, buffer(node.inputs[2]), output,
                  output,
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
      Value sameBuffer =
          arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                pointerToInt(builder, location, source),
                                pointerToInt(builder, location, destination));
      Block *next = run.addBlock();
      Block *copy = run.addBlock();
      cf::CondBranchOp::create(builder, location, sameBuffer, next,
                               ValueRange{}, copy, ValueRange{});
      builder.setInsertionPointToEnd(copy);
      int64_t bytes =
          graph.tensors[tensorIndex].type.getNumElements() *
          elementByteSize(graph.tensors[tensorIndex].type.getElementType());
      auto copied = func::CallOp::create(
          builder, location, "clEnqueueCopyBuffer", TypeRange{i32},
          ValueRange{entry->getArgument(2), source, destination, constant(0),
                     constant(0), constant(bytes), constant(0),
                     nullPointer(builder, location), dispatchEvent});
      returnOpenCLErrorIfNonzero(copied.getResult(0));
      waitAndRelease();
      cf::BranchOp::create(builder, location, next);
      builder.setInsertionPointToEnd(next);
    }
    auto completed = func::CallOp::create(
        builder, location, "clEnqueueMarkerWithWaitList", TypeRange{i32},
        ValueRange{entry->getArgument(2), constant(0),
                   nullPointer(builder, location), entry->getArgument(5)});
    returnOpenCLErrorIfNonzero(completed.getResult(0));
    func::ReturnOp::create(builder, location, onnxStatus(0));

    declaration(module, "__vrc_enqueue_udondnn_large_convolution_device_args",
                {pointer,
                 pointer,
                 pointer,
                 pointer,
                 pointer,
                 i32,
                 i32,
                 i32,
                 i32,
                 i32,
                 i32,
                 i32,
                 i32,
                 i32,
                 i32,
                 builder.getF32Type(),
                 builder.getF32Type(),
                 i32,
                 pointer,
                 pointer},
                {i32});
    declaration(module, "__vrc_enqueue_udondnn_large_activation_device_args",
                {pointer, pointer, pointer, pointer, i32, i32, i32,
                 builder.getF32Type(), builder.getF32Type(),
                 builder.getF32Type(), i32, pointer, pointer},
                {i32});
    declaration(module, "__vrc_enqueue_udondnn_large_pooling_device_args",
                {pointer, pointer, pointer, pointer, i32, i32, i32, i32, i32,
                 i32, i32, i32, i32, i32, builder.getF32Type(),
                 builder.getF32Type(), i32, pointer, pointer},
                {i32});
    declaration(module, "__vrc_enqueue_udondnn_large_op_tensor_device_args",
                {pointer, pointer, pointer, pointer, pointer, i32, i32, i32,
                 i32, i32, i32, i32, i32, builder.getF32Type(),
                 builder.getF32Type(), builder.getF32Type(), i32, pointer,
                 pointer},
                {i32});
    declaration(module, "__vrc_enqueue_udonblas_arena_sgemm_device_args",
                {pointer, pointer, pointer, pointer, i32, i32, i32, i32, i32,
                 builder.getF32Type(), i32, pointer, pointer},
                {i32});
    declaration(module, "__vrc_enqueue_udononnx_unary_device_args",
                {pointer, pointer, pointer, i32, i32, i32, pointer, pointer},
                {i32});
    declaration(module, "__vrc_enqueue_udononnx_matmul_nbits_q4_device_args",
                {pointer, pointer, pointer, pointer, pointer, pointer, i32, i32,
                 i32, i32, i32, i32, i32, i32, i32, i32, pointer, pointer},
                {i32});
    declaration(module, "udononnxSessionResolveInput",
                {pointer, i32, pointer, pointer, pointer}, {i32});
    declaration(module, "udononnxSessionResolveOutput",
                {pointer, i32, pointer, pointer, pointer}, {i32});
    declaration(module, "clEnqueueMarkerWithWaitList",
                {pointer, i32, pointer, pointer}, {i32});
    declaration(
        module, "clEnqueueCopyBuffer",
        {pointer, pointer, pointer, i32, i32, i32, i32, pointer, pointer},
        {i32});
    declaration(module, "__vrc_opencl_static_tensor", {i32, i32}, {i32});
    declaration(module, "udonclAdoptStaticBuffer",
                {pointer, i32, i32, i32, pointer}, {pointer});
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

  declaration(module, "udononnxSessionResolveInput",
              {pointer, i32, pointer, pointer, pointer}, {i32});
  declaration(module, "udononnxSessionResolveOutput",
              {pointer, i32, pointer, pointer, pointer}, {i32});
  declaration(module, "clEnqueueReadBuffer",
              {pointer, pointer, i32, i32, i32, pointer, i32, pointer, pointer},
              {i32});
  declaration(module, "clEnqueueWriteBuffer",
              {pointer, pointer, i32, i32, i32, pointer, i32, pointer, pointer},
              {i32});
  declaration(module, "clEnqueueMarkerWithWaitList",
              {pointer, i32, pointer, pointer}, {i32});
  declaration(module, "__vrc_opencl_static_tensor", {i32, i32}, {i32});
  declaration(module, "udonclAdoptStaticBuffer",
              {pointer, i32, i32, i32, pointer}, {pointer});
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
    for (auto [tensorIndex, tensor] : llvm::enumerate(graph->tensors))
      if (tensor.staticResourceId >= 0 ||
          (deviceGraph && !llvm::is_contained(graph->inputs, tensorIndex) &&
           (!llvm::is_contained(graph->outputs, tensorIndex) ||
            needsQ4Surface(*graph, tensorIndex)) &&
           deviceStorageMap[tensorIndex] == tensorIndex))
        createI32Global(module, deviceBufferSymbol(tensorIndex));
    if (deviceGraph && hasDeviceQ4(*graph))
      createI32Global(module, q4ScratchSymbol);
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
    attachBindingMetadata(module, *graph);
    attachDeviceGraphMetadata(module, *graph, deviceGraph);
    module->setAttr("udoncl.onnx_lowering",
                    StringAttr::get(module.getContext(), "library-calls-v2"));
  }
};
} // namespace
} // namespace vrc
