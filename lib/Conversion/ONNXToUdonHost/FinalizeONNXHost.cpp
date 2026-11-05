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
  if (type.isInteger(8))
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
  if (type.isInteger(8))
    return "int8";
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
                       DenseElementsAttr initial = {}) -> FailureOr<unsigned> {
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
    result.tensors.push_back(
        {type, initial, ("__udon_onnx_" + stem + "_" + Twine(id)).str(), {}});
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
  bool usesBlas = false;
  bool usesONNXOperators = false;
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    usesBlas |= node.operation == "Gemm" || node.operation == "MatMul";
    usesONNXOperators |=
        node.operation == "MatMulNBits" || node.operation == "Gather";
    if (node.operation == "Conv") {
      filterSlots[index] = allocaI32(builder, module.getLoc());
      convolutionSlots[index] = allocaI32(builder, module.getLoc());
      store(builder, module.getLoc(), zero, filterSlots[index]);
      store(builder, module.getLoc(), zero, convolutionSlots[index]);
    } else if (node.operation == "AveragePool") {
      poolingSlots[index] = allocaI32(builder, module.getLoc());
      store(builder, module.getLoc(), zero, poolingSlots[index]);
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
    if ((node.operation == "Flatten" || node.operation == "Reshape" ||
         node.operation == "Identity") &&
        node.inputs.size() >= 1 && node.inputs[0] >= 0) {
      tensorPointers[node.output] = tensorPointer(node.inputs[0]);
      continue;
    }
    if (node.operation == "Add" && node.provider == "udonatenAdd" &&
        node.inputs.size() == 2 && node.inputs[0] >= 0 && node.inputs[1] >= 0) {
      checked("udonatenAdd",
              ValueRange{dnn, descriptor(node.inputs[0]),
                         tensorPointer(node.inputs[0]),
                         descriptor(node.inputs[1]),
                         tensorPointer(node.inputs[1]), one,
                         descriptor(node.output), tensorPointers[node.output]});
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
      ArrayRef<int64_t> left = leftTensor.type.getShape();
      ArrayRef<int64_t> right = rightTensor.type.getShape();
      if (left.empty() || right.size() != 2 || left.back() != right.front() ||
          !leftTensor.type.getElementType().isF32() ||
          !rightTensor.type.getElementType().isF32())
        return emitError(module.getLoc())
               << "ONNX MatMul currently requires FP32 [..., K] x [K, N]";
      int64_t k = left.back();
      int64_t m = leftTensor.type.getNumElements() / k;
      int64_t n = right.back();
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
  declaration(module, "udonatenRelu", {i32, i32, pointer, i32, pointer}, {i32});
  declaration(module, "udonatenTanh", {i32, i32, pointer, i32, pointer}, {i32});
  declaration(
      module, "udonatenRmsNorm",
      {i32, i32, pointer, pointer, i32, i32, pointer, pointer, i32, pointer},
      {i32});
  declaration(module, "udononnxGather",
              {i32, i32, pointer, pointer, i32, i32, pointer, i32,
               builder.getI64Type(), pointer},
              {i32});
  declaration(module, "udonblasCreate", {pointer}, {i32});
  declaration(module, "udonblasDestroy", {i32}, {i32});
  declaration(module, "udonblasSgemm",
              {i32, i32, i32, i32, i32, i32, pointer, pointer, i32, pointer,
               i32, pointer, pointer, i32},
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

  for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph.inputs)) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    int64_t bytes = tensor.type.getNumElements() *
                    elementByteSize(tensor.type.getElementType());
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
    if ((node.operation == "Flatten" || node.operation == "Reshape" ||
         node.operation == "Identity") &&
        !node.inputs.empty() && node.inputs[0] >= 0)
      storage[node.output] = storage[static_cast<unsigned>(node.inputs[0])];

  for (auto [outputIndex, tensorIndex] : llvm::enumerate(graph.outputs)) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    int64_t bytes = tensor.type.getNumElements() *
                    elementByteSize(tensor.type.getElementType());
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
    for (const TensorRecord &tensor : graph->tensors)
      createGlobal(module, tensor);
    createModelMetadataGlobals(module, *graph);
    for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph->inputs)) {
      (void)tensorIndex;
      createI32Global(module,
                      inputBindingSymbol(static_cast<unsigned>(inputIndex)));
    }
    createI32Global(module, hasRunSymbol);
    Builder builder(module.getContext());
    createWriteExport(module, *graph, builder.getF32Type(), "float");
    createWriteExport(module, *graph, builder.getI32Type(), "int32");
    createWriteExport(module, *graph, builder.getI64Type(), "int64");
    createReadExport(module, *graph, builder.getF32Type(), "float");
    createReadExport(module, *graph, builder.getI32Type(), "int32");
    createReadExport(module, *graph, builder.getI64Type(), "int64");
    createBindingExports(module, *graph);
    createGeneratedModelExports(module, *graph);
    createPhysicalBindingExports(module, *graph);
    if (failed(createRunExport(module, *graph))) {
      signalPassFailure();
      return;
    }
    if (failed(createRuntimeDispatchers(module, *graph))) {
      signalPassFailure();
      return;
    }
    attachBindingMetadata(module, *graph);
    module->setAttr("udoncl.onnx_lowering",
                    StringAttr::get(module.getContext(), "library-calls-v2"));
  }
};
} // namespace
} // namespace vrc
