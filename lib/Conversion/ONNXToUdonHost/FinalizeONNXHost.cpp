#include "ONNXHostABI.h"
#include "ONNXHostFusion.h"
#include "ONNXHostFusionKernel.h"
#include "ONNXHostGraph.h"
#include "ONNXHostIRBuilder.h"
#include "ONNXHostMaterialization.h"
#include "ONNXHostPlan.h"
#include "ONNXHostSymbols.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "udoncl/udon_abi_layout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <optional>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_FINALIZEONNXHOSTPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace host = ::vrc::udonhl;

constexpr StringRef operatorScratchSymbol = onnxHostOperatorScratchSymbol;
constexpr StringRef operatorScratchOwnedSymbol =
    onnxHostOperatorScratchOwnedSymbol;
constexpr StringRef hasRunSymbol = onnxHostHasRunSymbol;

std::string inputBindingSymbol(unsigned index) {
  return onnxHostInputBindingSymbol(index);
}

std::string deviceBufferSymbol(unsigned index) {
  return onnxHostDeviceBufferSymbol(index);
}

std::string deviceBufferOwnedSymbol(unsigned index) {
  return onnxHostDeviceBufferOwnedSymbol(index);
}

unsigned elementByteSize(Type type) { return onnxHostElementByteSize(type); }

int32_t udonclDataType(Type type) { return onnxHostDataType(type); }

StringRef udonclDataTypeName(Type type) { return onnxHostDataTypeName(type); }

Value addressOf(OpBuilder &builder, Location location, StringRef symbol) {
  return onnxHostAddressOf(builder, location, symbol);
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
  return onnxHostNullPointer(builder, location);
}

Value pointerToInt(OpBuilder &builder, Location location, Value pointer) {
  return onnxHostPointerToInt(builder, location, pointer);
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
  onnxHostStore(builder, location, value, pointer);
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
  return onnxHostPointerAdd(builder, location, pointer, offset);
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
  ONNXHostDeclarations declarations(module);
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

  declarations.require("udondnnCreate", {pointer}, {i32});
  declarations.require("udondnnDestroy", {i32}, {i32});
  declarations.require("udondnnCreateTensorDescriptor", {pointer}, {i32});
  declarations.require("udondnnSetTensor4dDescriptor",
                       {i32, i32, i32, i32, i32, i32, i32}, {i32});
  declarations.require("udondnnDestroyTensorDescriptor", {i32}, {i32});
  declarations.require("udondnnCreateFilterDescriptor", {pointer}, {i32});
  declarations.require("udondnnSetFilter4dDescriptor",
                       {i32, i32, i32, i32, i32, i32, i32}, {i32});
  declarations.require("udondnnDestroyFilterDescriptor", {i32}, {i32});
  declarations.require("udondnnCreateConvolutionDescriptor", {pointer}, {i32});
  declarations.require("udondnnSetConvolution2dDescriptor",
                       {i32, i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
  declarations.require("udondnnDestroyConvolutionDescriptor", {i32}, {i32});
  declarations.require("udondnnConvolutionForward",
                       {i32, pointer, i32, pointer, i32, pointer, i32, i32,
                        pointer, i32, pointer, i32, pointer},
                       {i32});
  declarations.require("udondnnCreatePoolingDescriptor", {pointer}, {i32});
  declarations.require("udondnnSetPooling2dDescriptor",
                       {i32, i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
  declarations.require("udondnnDestroyPoolingDescriptor", {i32}, {i32});
  declarations.require("udondnnPoolingForward",
                       {i32, i32, pointer, i32, pointer, pointer, i32, pointer},
                       {i32});
  declarations.require(
      "udonatenAdd",
      {i32, i32, pointer, i32, pointer, builder.getF32Type(), i32, pointer},
      {i32});
  declarations.require("udonatenMul",
                       {i32, i32, pointer, i32, pointer, i32, pointer}, {i32});
  declarations.require("udonatenRelu", {i32, i32, pointer, i32, pointer},
                       {i32});
  declarations.require("udonatenTanh", {i32, i32, pointer, i32, pointer},
                       {i32});
  declarations.require("udonatenSigmoid", {i32, i32, pointer, i32, pointer},
                       {i32});
  declarations.require(
      "udonatenRmsNorm",
      {i32, i32, pointer, pointer, i32, i32, pointer, pointer, i32, pointer},
      {i32});
  declarations.require("udononnxGather",
                       {i32, i32, pointer, pointer, i32, i32, pointer, i32,
                        builder.getI64Type(), pointer},
                       {i32});
  declarations.require(
      "udondnnSoftmaxForward",
      {i32, i32, i32, pointer, i32, pointer, pointer, i32, pointer}, {i32});
  declarations.require("udononnxCast", {i32, i32, pointer, i32, pointer, i32},
                       {i32});
  declarations.require("udononnxExpand",
                       {i32, i32, pointer, pointer, i32, pointer, pointer, i32},
                       {i32});
  declarations.require("udononnxTranspose",
                       {i32, i32, pointer, pointer, i32, pointer, pointer},
                       {i32});
  declarations.require("udononnxConcat",
                       {i32, i32, pointer, pointer, i32, i32,
                        builder.getI64Type(), pointer, pointer},
                       {i32});
  declarations.require("udononnxCos", {i32, pointer, i32, pointer}, {i32});
  declarations.require("udononnxSin", {i32, pointer, i32, pointer}, {i32});
  declarations.require("udononnxNeg", {i32, pointer, i32, pointer}, {i32});
  declarations.require("udononnxSlice",
                       {i32, i32, pointer, pointer, i32, pointer, pointer,
                        pointer, pointer, i32, pointer, pointer},
                       {i32});
  declarations.require("udononnxEqual",
                       {i32, i32, pointer, pointer, i32, pointer, pointer, i32,
                        pointer, pointer, i32},
                       {i32});
  declarations.require("udononnxWhere",
                       {i32, i32, pointer, pointer, i32, pointer, pointer, i32,
                        pointer, pointer, i32, pointer, pointer, i32},
                       {i32});
  declarations.require("udononnxAdd",
                       {i32, i32, pointer, pointer, i32, pointer, pointer, i32,
                        pointer, pointer, i32},
                       {i32});
  declarations.require("udononnxScatterND",
                       {i32, i32, pointer, pointer, i32, i32, pointer, pointer,
                        i32, pointer, pointer},
                       {i32});
  declarations.require("udonblasCreate", {pointer}, {i32});
  declarations.require("udonblasDestroy", {i32}, {i32});
  declarations.require("udonblasSgemm",
                       {i32, i32, i32, i32, i32, i32, pointer, pointer, i32,
                        pointer, i32, pointer, pointer, i32},
                       {i32});
  declarations.require("udonblasSgemmStridedBatched",
                       {i32, i32, i32, i32, i32, i32, pointer, pointer, i32,
                        builder.getI64Type(), pointer, i32,
                        builder.getI64Type(), pointer, pointer, i32,
                        builder.getI64Type(), i32},
                       {i32});
  declarations.require("udononnxCreateOperatorHandle", {pointer}, {i32});
  declarations.require("udononnxDestroyOperatorHandle", {i32}, {i32});
  declarations.require("udononnxMatMulNBits",
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
  return declarations.result();
}

/// Declares the runtime functions the generated dispatchers call.
LogicalResult declareBindingRuntime(ModuleOp module) {
  ONNXHostDeclarations declarations(module);
  Builder builder(module.getContext());
  Type i32 = builder.getI32Type();
  Type pointer = host::PtrType::get(module.getContext(), 0);
  declarations.require("udonclDeviceBufferView",
                       {pointer, i32, i32, i32, pointer}, {i32});
  declarations.require("udononnxSessionResolveInput",
                       {pointer, i32, pointer, pointer, pointer}, {i32});
  declarations.require("udononnxSessionResolveOutput",
                       {pointer, i32, pointer, pointer, pointer}, {i32});
  declarations.require("udonclResolvePhysicalBuffer", {pointer, pointer},
                       {i32});
  declarations.require("__vrc_opencl_buffer_copy", {i32, i32, i32, i32, i32},
                       {i32});
  declarations.require("__vrc_opencl_buffer_upload", {i32, i32, i32, pointer},
                       {i32});
  declarations.require("__vrc_opencl_buffer_fill",
                       {i32, i32, i32, pointer, i32}, {i32});
  declarations.require("clFinish", {pointer}, {i32});
  declarations.require("__vrc_opencl_buffer_alloc", {i32, i32}, {i32});
  declarations.require("__vrc_opencl_await", {i32}, {i32});
  declarations.require("udonclDeviceBufferDestroy", {pointer}, {i32});
  declarations.require("clWaitForEvents", {i32, pointer}, {i32});
  declarations.require("clReleaseEvent", {pointer}, {i32});
  return declarations.result();
}

LogicalResult createRuntimeDispatchers(ModuleOp module,
                                       const GraphRecord &graph,
                                       const HostPlanRecord &plan) {
  ONNXHostDeclarations declarations(module);
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

  // Create allocates the Device buffers the session owns on the session's
  // queue and enqueues nothing; under the static Unity lowering they are
  // assigned Textures, so it does no Device work at all.
  builder.setInsertionPointToEnd(module.getBody());
  auto createSignature =
      FunctionType::get(module.getContext(), {i32, pointer, pointer}, {i32});
  auto initialize = func::FuncOp::create(
      builder, location, "udononnxGeneratedCreate", createSignature);
  Block *initializeEntry = initialize.addEntryBlock();
  Block *initializeValid = initialize.addBlock();
  Block *initializeInvalid = initialize.addBlock();
  builder.setInsertionPointToEnd(initializeEntry);
  graphMatches(initializeEntry, initializeValid, initializeInvalid);
  builder.setInsertionPointToEnd(initializeInvalid);
  func::ReturnOp::create(builder, location, onnxStatus(1));
  builder.setInsertionPointToEnd(initializeValid);
  if (supportsDeviceGraph(graph)) {
    // The session's Device buffers are created again: tokens a run resolved
    // before are stale.
    store(builder, location, onnxStatus(0),
          addressOf(builder, location, onnxHostResolvedBindingVersionSymbol));
    Value bufferError = allocaI32(builder, location);
    ArrayRef<unsigned> storage = plan.deviceStorage;
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
  func::ReturnOp::create(builder, location, onnxStatus(0));

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
  ArrayRef<unsigned> disposeStorage = plan.deviceStorage;
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
  // The buffers a UdonONNX session bound stay owned by the caller; the
  // session forgets them.
  for (bool output : {false, true}) {
    unsigned count = output ? graph.outputs.size() : graph.inputs.size();
    for (unsigned index = 0; index < count; ++index)
      store(builder, location, nullPointer(builder, location),
            addressOf(builder, location,
                      onnxHostBoundMemorySymbol(output, index)));
  }
  func::ReturnOp::create(builder, location,
                         loadI32(builder, location, disposeStatus));
  declarations.require("clReleaseMemObject", {pointer}, {i32});

  // BindMemory(graph_id, output, index, memory, extent, pitch) keeps the
  // cl_mem a UdonONNX session binds to graph input (output = 0) or output
  // (output = 1) `index` in the generated memory cell udononnxGeneratedRun
  // reads; a null memory unbinds it. For a tensor with an axis of a bounded
  // dimension it also keeps that axis's extent and its pitch in the buffer;
  // a pitch beyond the largest extent of the dimension is not supported.
  builder.setInsertionPointToEnd(module.getBody());
  auto bindMemory = func::FuncOp::create(
      builder, location, "udononnxGeneratedBindMemory",
      FunctionType::get(module.getContext(), {i32, i32, i32, pointer, i32, i32},
                        {i32}));
  {
    Block *bindEntry = bindMemory.addEntryBlock();
    Block *bindValid = bindMemory.addBlock();
    Block *bindInvalid = bindMemory.addBlock();
    builder.setInsertionPointToEnd(bindEntry);
    graphMatches(bindEntry, bindValid, bindInvalid);
    builder.setInsertionPointToEnd(bindInvalid);
    func::ReturnOp::create(builder, location, onnxStatus(1));
    Block *inputs = bindMemory.addBlock();
    Block *outputs = bindMemory.addBlock();
    builder.setInsertionPointToEnd(bindValid);
    Value isOutput =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ne,
                              bindEntry->getArgument(1), onnxStatus(0));
    cf::CondBranchOp::create(builder, location, isOutput, outputs, ValueRange{},
                             inputs, ValueRange{});
    for (bool output : {false, true}) {
      unsigned count = output ? graph.outputs.size() : graph.inputs.size();
      SmallVector<int32_t> caseValues;
      SmallVector<Block *> caseBlocks;
      for (unsigned index = 0; index < count; ++index) {
        Block *bind = bindMemory.addBlock();
        builder.setInsertionPointToEnd(bind);
        const TensorRecord &tensor =
            graph.tensors[output ? graph.outputs[index] : graph.inputs[index]];
        if (tensor.dynamic) {
          Value pitch = bindEntry->getArgument(5);
          Value capacity = arith::ConstantIntOp::create(
              builder, location,
              dimensionCapacity(graph, tensor.dynamic->symbol), 32);
          Value fits = arith::CmpIOp::create(
              builder, location, arith::CmpIPredicate::ule, pitch, capacity);
          Block *store_ = bindMemory.addBlock();
          Block *tooWide = bindMemory.addBlock();
          cf::CondBranchOp::create(builder, location, fits, store_,
                                   ValueRange{}, tooWide, ValueRange{});
          builder.setInsertionPointToEnd(tooWide);
          func::ReturnOp::create(builder, location, onnxStatus(4));
          builder.setInsertionPointToEnd(store_);
          store(builder, location, bindEntry->getArgument(4),
                addressOf(builder, location,
                          onnxHostBoundExtentSymbol(output, index)));
          store(builder, location, pitch,
                addressOf(builder, location,
                          onnxHostBoundPitchSymbol(output, index)));
        }
        store(builder, location, bindEntry->getArgument(3),
              addressOf(builder, location,
                        onnxHostBoundMemorySymbol(output, index)));
        func::ReturnOp::create(builder, location, onnxStatus(0));
        caseValues.push_back(static_cast<int32_t>(index));
        caseBlocks.push_back(bind);
      }
      builder.setInsertionPointToEnd(output ? outputs : inputs);
      SmallVector<ValueRange> caseOperands(caseBlocks.size(), ValueRange{});
      cf::SwitchOp::create(builder, location, bindEntry->getArgument(2),
                           bindInvalid, ValueRange{}, caseValues, caseBlocks,
                           caseOperands);
    }
  }

  // udononnxGeneratedRun(graph_id, queue, completion) runs the graph on the
  // buffers a UdonONNX session bound (udononnxGeneratedBindMemory); a graph
  // the Device cannot run is not supported there.
  builder.setInsertionPointToEnd(module.getBody());
  auto sessionRun = func::FuncOp::create(
      builder, location, "udononnxGeneratedRun",
      FunctionType::get(module.getContext(), {i32, pointer, pointer}, {i32}));
  sessionRun->setAttr("vrc.async_effect", builder.getStringAttr("may_suspend"));
  Block *sessionRunEntry = sessionRun.addEntryBlock();
  if (!supportsDeviceGraph(graph)) {
    builder.setInsertionPointToEnd(sessionRunEntry);
    func::ReturnOp::create(builder, location, onnxStatus(4));
  }

  builder.setInsertionPointToEnd(module.getBody());
  func::FuncOp run = func::FuncOp::create(
      builder, location, "udononnxGeneratedEnqueueRun", signature);
  run->setAttr("vrc.async_effect", builder.getStringAttr("may_suspend"));
  Block *entry = run.addEntryBlock();
  Block *body = run.addBlock();
  Block *invalidGraph = run.addBlock();
  builder.setInsertionPointToEnd(entry);
  graphMatches(entry, body, invalidGraph);
  builder.setInsertionPointToEnd(invalidGraph);
  func::ReturnOp::create(builder, location, onnxStatus(1));
  builder.setInsertionPointToEnd(body);
  // The legacy run checks the byte length of every binding, which a tensor
  // with an axis of a bounded dimension does not have; such a graph runs
  // through the session API only.
  if (llvm::any_of(graph.tensors,
                   [](const TensorRecord &tensor) { return tensor.dynamic; })) {
    func::ReturnOp::create(builder, location, onnxStatus(4));
    body = run.addBlock();
    builder.setInsertionPointToEnd(body);
  }

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
    ArrayRef<unsigned> storage = plan.deviceStorage;
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
    // Resolving the bound inputs and outputs and the Device buffers the
    // session owns to physical tokens depends only on the binding table and
    // on the buffers Initialize created, not on the run. The tokens are kept
    // in globals under the session binding version they were resolved for;
    // a run of an unchanged table (the session has checked that its buffers
    // are alive) reads them instead of resolving every buffer again.
    Value bindingVersion =
        func::CallOp::create(builder, location, "udononnxSessionBindingVersion",
                             TypeRange{i32}, ValueRange{entry->getArgument(1)})
            .getResult(0);
    Value resolvedVersion = loadI32(
        builder, location,
        addressOf(builder, location, onnxHostResolvedBindingVersionSymbol));
    Value reuse = arith::AndIOp::create(
        builder, location,
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                              bindingVersion, resolvedVersion),
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ne,
                              resolvedVersion, constant(0)));
    Block *resolveTokens = run.addBlock();
    Block *tokensReady = run.addBlock();
    cf::CondBranchOp::create(builder, location, reuse, tokensReady,
                             ValueRange{}, resolveTokens, ValueRange{});
    builder.setInsertionPointToEnd(resolveTokens);
    for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph.inputs))
      if (failed(resolve(false, inputIndex, tensorIndex)))
        return failure();
    for (auto [outputIndex, tensorIndex] : llvm::enumerate(graph.outputs))
      if (failed(resolve(true, outputIndex, tensorIndex)))
        return failure();
    for (unsigned tensorIndex : plan.ownedDeviceBuffers) {
      if (buffers[tensorIndex])
        continue;
      Value handle = loadPointer(
          builder, location,
          addressOf(builder, location, deviceBufferSymbol(tensorIndex)));
      buffers[tensorIndex] = resolvePhysicalBuffer(handle);
    }
    SmallVector<unsigned> resolvedTensors = resolvedRunTensors(graph, plan);
    for (unsigned tensorIndex : resolvedTensors) {
      if (!buffers[tensorIndex])
        return emitError(location)
               << "ONNX Device run did not resolve tensor " << tensorIndex;
      store(builder, location, buffers[tensorIndex],
            addressOf(builder, location,
                      onnxHostResolvedTokenSymbol(tensorIndex)));
    }
    store(builder, location, bindingVersion,
          addressOf(builder, location, onnxHostResolvedBindingVersionSymbol));
    cf::BranchOp::create(builder, location, tokensReady);

    // Everything after the tokens are resolved is the resolved run, which
    // the legacy run and the UdonONNX session run share. It reads the
    // tokens from the cache, so it takes the arguments of the legacy run.
    builder.setInsertionPointToEnd(module.getBody());
    auto resolvedRun = func::FuncOp::create(
        builder, location, "__udononnx_enqueue_resolved_run", signature);
    resolvedRun->setAttr("vrc.async_effect",
                         builder.getStringAttr("may_suspend"));
    resolvedRun.setPrivate();
    Block *resolvedEntry = resolvedRun.addEntryBlock();
    builder.setInsertionPointToEnd(tokensReady);
    func::ReturnOp::create(builder, location,
                           func::CallOp::create(builder, location, resolvedRun,
                                                entry->getArguments())
                               .getResult(0));

    // The UdonONNX session run resolves the bound memory cells (an empty
    // cell is a missing binding) and the owned buffers to tokens. It keeps
    // no binding version: a legacy run after it resolves again.
    run = sessionRun;
    entry = sessionRunEntry;
    {
      Block *valid = run.addBlock();
      Block *invalid = run.addBlock();
      builder.setInsertionPointToEnd(entry);
      graphMatches(entry, valid, invalid);
      builder.setInsertionPointToEnd(invalid);
      func::ReturnOp::create(builder, location, onnxStatus(1));
      builder.setInsertionPointToEnd(valid);
      SmallVector<Value> sessionTokens(graph.tensors.size());
      auto resolveCell = [&](bool output, unsigned index,
                             unsigned tensorIndex) {
        Value memory =
            loadPointer(builder, location,
                        addressOf(builder, location,
                                  onnxHostBoundMemorySymbol(output, index)));
        Value isBound = arith::CmpIOp::create(
            builder, location, arith::CmpIPredicate::ne,
            pointerToInt(builder, location, memory), constant(0));
        Block *bound = run.addBlock();
        Block *missing = run.addBlock();
        cf::CondBranchOp::create(builder, location, isBound, bound,
                                 ValueRange{}, missing, ValueRange{});
        builder.setInsertionPointToEnd(missing);
        func::ReturnOp::create(builder, location, onnxStatus(11));
        builder.setInsertionPointToEnd(bound);
        sessionTokens[tensorIndex] = resolvePhysicalBuffer(memory);
      };
      for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph.inputs))
        resolveCell(false, inputIndex, tensorIndex);
      for (auto [outputIndex, tensorIndex] : llvm::enumerate(graph.outputs))
        resolveCell(true, outputIndex, tensorIndex);
      // Every bounded dimension takes its value from the first binding of a
      // tensor that has it, inputs first: the extent minus the tensor's
      // offset. Every other such binding, inputs and outputs, must have the
      // extent that value gives it (a present of past + 1); a disagreement
      // is INVALID_VALUE, as a preallocated output of the wrong shape is for
      // ONNX Runtime's RunWithBinding.
      for (auto [symbolIndex, dimension] :
           llvm::enumerate(graph.dimensionBounds)) {
        Value value;
        for (bool output : {false, true}) {
          ArrayRef<unsigned> tensors = output ? graph.outputs : graph.inputs;
          for (auto [index, tensorIndex] : llvm::enumerate(tensors)) {
            const TensorRecord &tensor = graph.tensors[tensorIndex];
            if (!tensor.dynamic || tensor.dynamic->symbol != dimension.first)
              continue;
            Value extent =
                loadI32(builder, location,
                        addressOf(builder, location,
                                  onnxHostBoundExtentSymbol(output, index)));
            Value offset = constant(tensor.dynamic->offset);
            if (!value) {
              // The bind kept the extent within [0, bound + offset].
              Value fits = arith::CmpIOp::create(
                  builder, location, arith::CmpIPredicate::sge, extent, offset);
              Block *next = run.addBlock();
              Block *invalid = run.addBlock();
              cf::CondBranchOp::create(builder, location, fits, next,
                                       ValueRange{}, invalid, ValueRange{});
              builder.setInsertionPointToEnd(invalid);
              func::ReturnOp::create(builder, location, onnxStatus(1));
              builder.setInsertionPointToEnd(next);
              value = arith::SubIOp::create(builder, location, extent, offset);
              continue;
            }
            Value expected =
                arith::AddIOp::create(builder, location, value, offset);
            Value agrees = arith::CmpIOp::create(
                builder, location, arith::CmpIPredicate::eq, extent, expected);
            Block *next = run.addBlock();
            Block *invalid = run.addBlock();
            cf::CondBranchOp::create(builder, location, agrees, next,
                                     ValueRange{}, invalid, ValueRange{});
            builder.setInsertionPointToEnd(invalid);
            func::ReturnOp::create(builder, location, onnxStatus(1));
            builder.setInsertionPointToEnd(next);
          }
        }
        if (!value)
          return emitError(location)
                 << "bounded ONNX dimension '" << dimension.first
                 << "' is not a dimension of any graph input or output";
        store(
            builder, location, value,
            addressOf(builder, location, onnxHostDimensionSymbol(symbolIndex)));
      }
      // The input of an in-place Concat pair is the leading block of its
      // output when both are bound to one buffer, which needs the same pitch
      // (the static axes are the model's): one buffer with two layouts is
      // INVALID_VALUE.
      for (const InPlaceConcat &pair : inPlaceConcats(graph)) {
        Value shared =
            arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                  sessionTokens[graph.inputs[pair.input]],
                                  sessionTokens[graph.outputs[pair.output]]);
        Value differ = arith::CmpIOp::create(
            builder, location, arith::CmpIPredicate::ne,
            loadI32(builder, location,
                    addressOf(builder, location,
                              onnxHostBoundPitchSymbol(false, pair.input))),
            loadI32(builder, location,
                    addressOf(builder, location,
                              onnxHostBoundPitchSymbol(true, pair.output))));
        Block *next = run.addBlock();
        Block *invalid = run.addBlock();
        cf::CondBranchOp::create(
            builder, location,
            arith::AndIOp::create(builder, location, shared, differ), invalid,
            ValueRange{}, next, ValueRange{});
        builder.setInsertionPointToEnd(invalid);
        func::ReturnOp::create(builder, location, onnxStatus(1));
        builder.setInsertionPointToEnd(next);
      }
      for (unsigned tensorIndex : plan.ownedDeviceBuffers) {
        if (sessionTokens[tensorIndex])
          continue;
        Value handle = loadPointer(
            builder, location,
            addressOf(builder, location, deviceBufferSymbol(tensorIndex)));
        sessionTokens[tensorIndex] = resolvePhysicalBuffer(handle);
      }
      for (unsigned tensorIndex : resolvedTensors) {
        if (!sessionTokens[tensorIndex])
          return emitError(location)
                 << "ONNX Device session run did not resolve tensor "
                 << tensorIndex;
        store(builder, location, sessionTokens[tensorIndex],
              addressOf(builder, location,
                        onnxHostResolvedTokenSymbol(tensorIndex)));
      }
      store(builder, location, constant(0),
            addressOf(builder, location, onnxHostResolvedBindingVersionSymbol));
      func::ReturnOp::create(
          builder, location,
          func::CallOp::create(
              builder, location, resolvedRun,
              ValueRange{entry->getArgument(0), nullPointer(builder, location),
                         entry->getArgument(1), constant(0),
                         nullPointer(builder, location), entry->getArgument(2)})
              .getResult(0));
    }

    run = resolvedRun;
    entry = resolvedEntry;
    builder.setInsertionPointToEnd(entry);
    for (unsigned tensorIndex : resolvedTensors) {
      Value token =
          loadI32(builder, location,
                  addressOf(builder, location,
                            onnxHostResolvedTokenSymbol(tensorIndex)));
      buffers[tensorIndex] = token;
      if (publicOutputBuffers[tensorIndex])
        publicOutputBuffers[tensorIndex] = token;
    }
    for (auto [tensorIndex, tensor] : llvm::enumerate(graph.tensors)) {
      if (buffers[tensorIndex] || storage[tensorIndex] != tensorIndex ||
          tensor.staticResourceId < 0)
        continue;
      Value resourceId = arith::ConstantIntOp::create(
          builder, location, tensor.staticResourceId, 32);
      Value bytes = arith::ConstantIntOp::create(builder, location,
                                                 tensor.staticByteLength, 32);
      buffers[tensorIndex] =
          func::CallOp::create(builder, location, "__vrc_opencl_static_tensor",
                               TypeRange{i32}, ValueRange{resourceId, bytes})
              .getResult(0);
    }

    // The run is recorded behind everything the caller already enqueued on
    // the in-order queue. With a wait list it also waits for those events.
    // Without one nothing has to complete first: earlier commands only have
    // to be issued (clFlush), so a non-blocking read the caller has not
    // consumed yet stays in flight while the run is enqueued.
    Value dispatchEvent = allocaI32(builder, location);
    Block *waitForList = run.addBlock();
    Block *flushOnly = run.addBlock();
    Block *enqueueRun = run.addBlock();
    Value hasWaitList =
        arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ne,
                              entry->getArgument(3), constant(0));
    cf::CondBranchOp::create(builder, location, hasWaitList, waitForList,
                             ValueRange{}, flushOnly, ValueRange{});
    builder.setInsertionPointToEnd(flushOnly);
    auto flushed =
        func::CallOp::create(builder, location, "clFlush", TypeRange{i32},
                             ValueRange{entry->getArgument(2)});
    returnOpenCLErrorIfNonzero(flushed.getResult(0));
    cf::BranchOp::create(builder, location, enqueueRun);
    builder.setInsertionPointToEnd(waitForList);
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
    cf::BranchOp::create(builder, location, enqueueRun);
    builder.setInsertionPointToEnd(enqueueRun);
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
    DeviceFusion fusion(graph, plan.deviceFusionRoots);
    // The layout of the buffer of every fused leaf or root with an axis of a
    // bounded dimension (dynamicLayout).
    DeviceFusionKernels fusedKernels(module, fusion, [&](unsigned tensorIndex) {
      const TensorRecord &tensor = graph.tensors[tensorIndex];
      DynamicLayout layout = *dynamicLayout(graph, tensorIndex);
      DeviceFusionLayout result;
      result.kind =
          layout.kind == DynamicLayout::Dense   ? DeviceFusionLayout::Dense
          : layout.kind == DynamicLayout::Bound ? DeviceFusionLayout::Bound
                                                : DeviceFusionLayout::Binding;
      result.output = layout.output;
      result.binding = layout.binding;
      result.capacity = layout.kind == DynamicLayout::Binding
                            ? dimensionCapacity(graph, tensor.dynamic->symbol)
                            : tensor.type.getDimSize(tensor.dynamic->axis);
      return result;
    });
    // The value of every bounded dimension in this run (set by the session
    // run), and the run-time extent and pitch of a tensor's dynamic axis.
    SmallVector<Value> dimensionValues;
    for (unsigned index = 0; index < graph.dimensionBounds.size(); ++index)
      dimensionValues.push_back(loadI32(
          builder, location,
          addressOf(builder, location, onnxHostDimensionSymbol(index))));
    auto extentOf = [&](unsigned tensorIndex, int64_t axis) -> Value {
      const TensorRecord &tensor = graph.tensors[tensorIndex];
      if (!tensor.dynamic || tensor.dynamic->axis != axis)
        return constant(tensor.type.getDimSize(axis));
      Value symbol =
          dimensionValues[dimensionIndex(graph, tensor.dynamic->symbol)];
      if (!tensor.dynamic->offset)
        return symbol;
      return arith::AddIOp::create(builder, location, symbol,
                                   constant(tensor.dynamic->offset));
    };
    auto pitchOf = [&](unsigned tensorIndex) -> Value {
      const TensorRecord &tensor = graph.tensors[tensorIndex];
      int64_t axis = tensor.dynamic->axis;
      DynamicLayout layout = *dynamicLayout(graph, tensorIndex);
      if (layout.kind == DynamicLayout::Dense)
        return extentOf(tensorIndex, axis);
      if (layout.kind == DynamicLayout::Bound)
        return constant(tensor.type.getDimSize(axis));
      return loadI32(
          builder, location,
          addressOf(builder, location,
                    onnxHostBoundPitchSymbol(layout.output, layout.binding)));
    };
    auto multiplyValues = [&](Value left, Value right) -> Value {
      return arith::MulIOp::create(builder, location, left, right);
    };
    // The run-time element count of a tensor, and two of its extents packed
    // into the low and high halves of a word, as the per-element library
    // kernels take them.
    auto elementCountOf = [&](unsigned tensorIndex) -> Value {
      const TensorRecord &tensor = graph.tensors[tensorIndex];
      if (!tensor.dynamic)
        return constant(tensor.type.getNumElements());
      int64_t axis = tensor.dynamic->axis;
      return multiplyValues(extentOf(tensorIndex, axis),
                            constant(tensor.type.getNumElements() /
                                     tensor.type.getDimSize(axis)));
    };
    auto packedShapeOf = [&](unsigned tensorIndex, int64_t first) -> Value {
      const TensorRecord &tensor = graph.tensors[tensorIndex];
      int64_t rank = tensor.type.getRank();
      auto extent = [&](int64_t axis) -> Value {
        return axis < rank ? extentOf(tensorIndex, axis) : constant(1);
      };
      bool dynamicPart = tensor.dynamic && (tensor.dynamic->axis == first ||
                                            tensor.dynamic->axis == first + 1);
      if (!dynamicPart) {
        int64_t low = first < rank ? tensor.type.getDimSize(first) : 1;
        int64_t high = first + 1 < rank ? tensor.type.getDimSize(first + 1) : 1;
        return constant((low & 65535) | ((high & 65535) << 16));
      }
      Value high = arith::ShLIOp::create(builder, location, extent(first + 1),
                                         constant(16));
      return arith::OrIOp::create(builder, location, extent(first), high);
    };
    // The in-place Concat pairs whose input and output buffers this run
    // binds directly; their Concat writes only the new rows when a session
    // bound both to one buffer.
    SmallVector<InPlaceConcat> inPlace;
    for (const InPlaceConcat &pair : inPlaceConcats(graph)) {
      unsigned input = graph.inputs[pair.input];
      unsigned output = graph.outputs[pair.output];
      if (storage[input] == input && storage[output] == output &&
          buffers[input] && buffers[output])
        inPlace.push_back(pair);
    }
    auto scalarValuesOf = [&](const DeviceFusionDispatch &fused) {
      // The run-time scalars of a group over a bounded dimension: the
      // dimension's value in this run and the pitch of a bound buffer.
      SmallVector<Value> scalars;
      for (const DeviceFusionScalar &scalar : fused.scalars)
        scalars.push_back(
            scalar.kind == DeviceFusionScalar::Dimension
                ? dimensionValues[scalar.index]
                : loadI32(builder, location,
                          addressOf(builder, location,
                                    onnxHostBoundPitchSymbol(scalar.output,
                                                             scalar.index))));
      return scalars;
    };
    auto emitNodeDispatch = [&](const NodeRecord &node,
                                unsigned nodeIndex) -> LogicalResult {
      Value output = buffer(node.output);
      if (!output)
        return emitError(location)
               << "Device ONNX graph output lacks physical storage";
      // A node runs a fused kernel also outside every group when its library
      // kernel cannot run it: a Softmax whose schedule has two passes
      // (deviceSoftmaxSchedule), or a node beyond the limits of its library
      // kernel (isSynthesizedDeviceNode).
      if (fusion.isRoot(nodeIndex) ||
          deviceSoftmaxSchedule(graph, node).chunks ||
          isSynthesizedDeviceNode(graph, node)) {
        FailureOr<DeviceFusionDispatch> fused =
            fusedKernels.dispatch(nodeIndex);
        if (failed(fused))
          return failure();
        SmallVector<Value> arguments;
        for (unsigned leaf : fused->leaves) {
          arguments.push_back(buffer(leaf));
          if (!arguments.back())
            return emitError(location)
                   << "Device ONNX fusion operand lacks physical storage";
        }
        SmallVector<Value> scalars = scalarValuesOf(*fused);
        auto withScalars = [&](ArrayRef<Value> values) {
          SmallVector<Value> all(values);
          llvm::append_range(all, scalars);
          return all;
        };
        // A two-pass row reduction writes its partial statistics to the
        // operator scratch, and the elementwise pass reads them there.
        if (!fused->statisticsCallee.empty()) {
          if (!operatorScratch)
            return emitError(location)
                   << "Device row reduction lacks operator scratch storage";
          arguments.push_back(operatorScratch);
          dispatch(fused->statisticsCallee, withScalars(arguments));
        }
        arguments.push_back(output);
        // An in-place Concat whose input and output a session bound to one
        // buffer (the run checked their pitches) writes only the rows after
        // the input; with two buffers it runs as any Concat.
        const InPlaceConcat *pair =
            llvm::find_if(inPlace, [&](const InPlaceConcat &candidate) {
              return candidate.node == nodeIndex;
            });
        if (pair != inPlace.end() && fused->statisticsCallee.empty()) {
          FailureOr<DeviceFusionDispatch> partial =
              fusedKernels.dispatchInPlace(nodeIndex);
          if (failed(partial))
            return failure();
          SmallVector<Value> partialArguments;
          for (unsigned leaf : partial->leaves)
            partialArguments.push_back(buffer(leaf));
          partialArguments.push_back(output);
          llvm::append_range(partialArguments, scalarValuesOf(*partial));
          Value shared =
              arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                    buffer(graph.inputs[pair->input]), output);
          Block *inPlaceBlock = run.addBlock();
          Block *separate = run.addBlock();
          Block *joined = run.addBlock();
          cf::CondBranchOp::create(builder, location, shared, inPlaceBlock,
                                   ValueRange{}, separate, ValueRange{});
          builder.setInsertionPointToEnd(inPlaceBlock);
          dispatch(partial->callee, partialArguments);
          cf::BranchOp::create(builder, location, joined);
          builder.setInsertionPointToEnd(separate);
          dispatch(fused->callee, withScalars(arguments));
          cf::BranchOp::create(builder, location, joined);
          builder.setInsertionPointToEnd(joined);
          return success();
        }
        dispatch(fused->callee, withScalars(arguments));
        return success();
      }
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
        return success();
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
        return success();
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
        return success();
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
        return success();
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
        return success();
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
        return success();
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
        return success();
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
        return success();
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
        return success();
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
        return success();
      }
      if (node.operation == "Cast") {
        if (!isDeviceCastI64F32Node(graph, node))
          return emitError(location)
                 << "Device Cast requires matching INT64 input and FP32 "
                    "output shapes of at most 16384 elements";
        dispatch("__vrc_opencl_dispatch_udononnx_cast_i64_f32",
                 ValueRange{buffer(node.inputs[0]), output,
                            elementCountOf(node.output)});
        return success();
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
        (void)packShape;
        const RankedTensorType left = graph.tensors[node.inputs[0]].type;
        const RankedTensorType right = graph.tensors[node.inputs[1]].type;
        const RankedTensorType result = graph.tensors[node.output].type;
        unsigned leftIndex = static_cast<unsigned>(node.inputs[0]);
        unsigned rightIndex = static_cast<unsigned>(node.inputs[1]);
        dispatch("__vrc_opencl_dispatch_udononnx_equal_f32",
                 ValueRange{
                     buffer(node.inputs[0]), buffer(node.inputs[1]), output,
                     elementCountOf(node.output),
                     constant(left.getRank() | (right.getRank() << 3) |
                              (result.getRank() << 6)),
                     packedShapeOf(leftIndex, 0), packedShapeOf(leftIndex, 2),
                     packedShapeOf(rightIndex, 0), packedShapeOf(rightIndex, 2),
                     packedShapeOf(node.output, 0),
                     packedShapeOf(node.output, 2)});
        return success();
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
        (void)packShape;
        const RankedTensorType condition = graph.tensors[node.inputs[0]].type;
        const RankedTensorType whenTrue = graph.tensors[node.inputs[1]].type;
        const RankedTensorType whenFalse = graph.tensors[node.inputs[2]].type;
        const RankedTensorType result = graph.tensors[node.output].type;
        unsigned conditionIndex = static_cast<unsigned>(node.inputs[0]);
        unsigned trueIndex = static_cast<unsigned>(node.inputs[1]);
        unsigned falseIndex = static_cast<unsigned>(node.inputs[2]);
        dispatch(
            "__vrc_opencl_dispatch_udononnx_where_f32",
            ValueRange{
                buffer(node.inputs[0]), buffer(node.inputs[1]),
                buffer(node.inputs[2]), output, elementCountOf(node.output),
                constant(condition.getRank() | (whenTrue.getRank() << 3) |
                         (whenFalse.getRank() << 6) | (result.getRank() << 9)),
                packedShapeOf(conditionIndex, 0),
                packedShapeOf(conditionIndex, 2), packedShapeOf(trueIndex, 0),
                packedShapeOf(trueIndex, 2), packedShapeOf(falseIndex, 0),
                packedShapeOf(falseIndex, 2), packedShapeOf(node.output, 0),
                packedShapeOf(node.output, 2)});
        return success();
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
        return success();
      }
      if (isDeviceQ4Node(graph, node)) {
        const TensorRecord &activation = graph.tensors[node.inputs[0]];
        int64_t k = integerAttribute(node, "K", 0);
        int64_t n = integerAttribute(node, "N", 0);
        int64_t m = activation.type.getNumElements() / k;
        std::optional<DeviceSplitK> split = deviceQ4SplitK(graph, node);
        if (!split)
          return emitError(location)
                 << "Device Q4 MatMulNBits has no split-K schedule";
        auto q4 = [&](Value previous, Value destination, int64_t blockCount,
                      int64_t splits, int64_t previousPartials) {
          dispatch("__vrc_opencl_dispatch_udononnx_matmul_nbits_q4",
                   ValueRange{buffer(node.inputs[0]), buffer(node.inputs[1]),
                              buffer(node.inputs[2]), previous, destination,
                              constant(m), constant(n), constant(k),
                              constant(blockCount), constant(splits),
                              constant(previousPartials)});
        };
        // Split-K (see chooseDeviceSplitK): the partial pass writes S planes
        // of m x n partial sums to the operator scratch, and the sum pass
        // adds them in ascending plane order into the output, which is the
        // only buffer readers (fused or not) see. One split is the serial
        // whole-K kernel; its `previous` is never read.
        if (split->splits == 1) {
          q4(operatorScratch, output, split->unitsPerSplit, 1, 0);
        } else {
          q4(output, operatorScratch, split->unitsPerSplit, split->splits, 0);
          q4(operatorScratch, output, 0, 1, split->splits);
        }
        return success();
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
        return success();
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
        return success();
      }
      if (node.operation == "MatMul") {
        std::optional<DeviceMatMulShape> shape = deviceMatMulShape(graph, node);
        if (!shape)
          return emitError(location)
                 << "Device MatMul requires a supported contiguous static "
                    "batch shape";
        // The non-batched kernel addresses at most deviceSgemmOperandWords
        // per operand; a larger single product uses the strided-batched
        // kernel with one batch instead of wrapping its indices.
        bool smallOperands = shape->m * shape->k <= deviceSgemmOperandWords &&
                             shape->k * shape->n <= deviceSgemmOperandWords;
        // With an axis of a bounded dimension, m, n or k is that axis's
        // run-time extent; the kernel is chosen for the largest extent. An
        // operand whose dynamic axis is its rows has its buffer's pitch
        // there, so its batch stride is pitch x columns; the validator has
        // checked that every other dynamic operand and the result are dense.
        auto tensorOf = [&](unsigned operand) {
          return static_cast<unsigned>(operand < 2 ? node.inputs[operand]
                                                   : node.output);
        };
        if (llvm::any_of(std::array<unsigned, 3>{0, 1, 2}, [&](unsigned op) {
              return graph.tensors[tensorOf(op)].dynamic.has_value();
            })) {
          unsigned left = tensorOf(0), right = tensorOf(1);
          int64_t leftRank = graph.tensors[left].type.getRank();
          int64_t rightRank = graph.tensors[right].type.getRank();
          Value m = extentOf(left, leftRank - 2);
          Value k = extentOf(left, leftRank - 1);
          Value n = extentOf(right, rightRank - 1);
          auto batchStride = [&](unsigned tensor, Value rows, Value columns,
                                 int64_t batches) -> Value {
            if (batches == 1)
              return constant(0);
            const TensorRecord &record = graph.tensors[tensor];
            if (record.dynamic &&
                record.dynamic->axis == record.type.getRank() - 2)
              return multiplyValues(pitchOf(tensor), columns);
            return multiplyValues(rows, columns);
          };
          dispatch(
              needsLargeBatchedSgemm(*shape)
                  ? "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_"
                    "batched_large"
                  : "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_"
                    "batched",
              ValueRange{buffer(node.inputs[0]), buffer(node.inputs[1]), output,
                         m, n, k, batchStride(left, m, k, shape->leftBatches),
                         batchStride(right, k, n, shape->rightBatches),
                         multiplyValues(m, n), constant(shape->outputBatches),
                         constant(0), constant(0), floating(1.0f)});
          return success();
        }
        if (shape->outputBatches == 1 && smallOperands) {
          dispatch("__vrc_opencl_dispatch_udonblas_arena_sgemm",
                   ValueRange{buffer(node.inputs[0]), buffer(node.inputs[1]),
                              output, constant(shape->m), constant(shape->n),
                              constant(shape->k), constant(0), constant(0),
                              floating(1.0f)});
        } else {
          // A product beyond the operand or output words of the
          // strided-batched kernel runs its taller variant.
          dispatch(
              needsLargeBatchedSgemm(*shape)
                  ? "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_"
                    "batched_large"
                  : "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_"
                    "batched",
              ValueRange{
                  buffer(node.inputs[0]), buffer(node.inputs[1]), output,
                  constant(shape->m), constant(shape->n), constant(shape->k),
                  constant(shape->leftBatches == 1 ? 0 : shape->m * shape->k),
                  constant(shape->rightBatches == 1 ? 0 : shape->k * shape->n),
                  constant(shape->m * shape->n), constant(shape->outputBatches),
                  constant(0), constant(0), floating(1.0f)});
        }
        return success();
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
        if (left[0] * left[1] <= deviceSgemmOperandWords &&
            right[0] * right[1] <= deviceSgemmOperandWords)
          dispatch("__vrc_opencl_dispatch_udonblas_arena_sgemm",
                   ValueRange{buffer(node.inputs[0]), buffer(node.inputs[1]),
                              product, constant(left[0]), constant(right[1]),
                              constant(left[1]), constant(transA),
                              constant(transB), floating(alpha)});
        else
          dispatch("__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched",
                   ValueRange{buffer(node.inputs[0]), buffer(node.inputs[1]),
                              product, constant(left[0]), constant(right[1]),
                              constant(left[1]), constant(0), constant(0),
                              constant(left[0] * right[1]), constant(1),
                              constant(transA), constant(transB),
                              floating(alpha)});
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
        return success();
      }
      return success();
    };
    for (const NodeRecord &node : graph.nodes) {
      unsigned nodeIndex = static_cast<unsigned>(&node - graph.nodes.data());
      // The root of a fusion group computes its interior nodes.
      if (isDeviceAlias(graph, node) || fusion.isInterior(nodeIndex))
        continue;
      if (failed(emitNodeDispatch(node, nodeIndex)))
        return failure();
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
    declarations.require("__vrc_opencl_dispatch_udondnn_large_convolution",
                         {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
                          i32, i32, i32, f32, f32},
                         {i32});
    declarations.require("__vrc_opencl_dispatch_udondnn_large_activation",
                         {i32, i32, i32, i32, i32, i32, f32, f32, f32}, {i32});
    declarations.require("__vrc_opencl_dispatch_udondnn_large_pooling",
                         {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
                          i32, i32, f32, f32},
                         {i32});
    declarations.require("__vrc_opencl_dispatch_udondnn_large_op_tensor",
                         {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
                          i32, f32, f32, f32},
                         {i32});
    declarations.require("__vrc_opencl_dispatch_udonblas_arena_sgemm",
                         {i32, i32, i32, i32, i32, i32, i32, i32, f32}, {i32});
    declarations.require(
        "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched",
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32},
        {i32});
    declarations.require(
        "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched_large",
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32},
        {i32});
    declarations.require("__vrc_opencl_dispatch_udononnx_unary_f32",
                         {i32, i32, i32, i32}, {i32});
    declarations.require("__vrc_opencl_dispatch_udononnx_transpose_f32",
                         {i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
    declarations.require("__vrc_opencl_dispatch_udononnx_slice_f32",
                         {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
                          i32, i32, i32, i32, i32},
                         {i32});
    declarations.require(
        "__vrc_opencl_dispatch_udononnx_expand_f32",
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
    declarations.require("__vrc_opencl_dispatch_udononnx_concat2_f32",
                         {i32, i32, i32, i32, i32, i32, i32}, {i32});
    declarations.require("__vrc_opencl_dispatch_udononnx_gather_axis0_f32",
                         {i32, i32, i32, i32, i32, i32, i32}, {i32});
    declarations.require("__vrc_opencl_dispatch_udononnx_cast_i64_f32",
                         {i32, i32, i32}, {i32});
    declarations.require(
        "__vrc_opencl_dispatch_udononnx_equal_f32",
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
    declarations.require(
        "__vrc_opencl_dispatch_udononnx_where_f32",
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32},
        {i32});
    declarations.require(
        "__vrc_opencl_dispatch_udononnx_scatter_nd_f32",
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
    declarations.require(
        "__vrc_opencl_dispatch_udononnx_matmul_nbits_q4",
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
    declarations.require("__vrc_opencl_dispatch_udonaten_arena_rms_norm_reduce",
                         {i32, i32, i32, i32}, {i32});
    declarations.require(
        "__vrc_opencl_dispatch_udonaten_arena_rms_norm_finalize",
        {i32, i32, i32, i32, i32, i32, f32, i32}, {i32});
    declarations.require("__vrc_opencl_dispatch_udondnn_arena_softmax",
                         {i32, i32, i32, i32, i32, i32, i32, i32}, {i32});
    declarations.require("clEnqueueMarkerWithWaitList",
                         {pointer, i32, pointer, pointer}, {i32});
    declarations.require("clFlush", {pointer}, {i32});
    declarations.require("__vrc_opencl_static_tensor", {i32, i32}, {i32});
    declarations.require("udonclCreateBufferForQueue",
                         {pointer, builder.getI64Type(), i32, pointer},
                         {pointer});
    declarations.require("udononnxSessionBindingVersion", {pointer}, {i32});
    return declarations.result();
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

  auto marker = func::CallOp::create(
      builder, location, "clEnqueueMarkerWithWaitList", TypeRange{i32},
      ValueRange{entry->getArgument(2), onnxStatus(0),
                 nullPointer(builder, location), entry->getArgument(5)});
  returnOpenCLErrorIfNonzero(marker.getResult(0));
  func::ReturnOp::create(builder, location, onnxStatus(0));

  declarations.require(
      "clEnqueueReadBuffer",
      {pointer, pointer, i32, i32, i32, pointer, i32, pointer, pointer}, {i32});
  declarations.require(
      "clEnqueueWriteBuffer",
      {pointer, pointer, i32, i32, i32, pointer, i32, pointer, pointer}, {i32});
  declarations.require("clEnqueueMarkerWithWaitList",
                       {pointer, i32, pointer, pointer}, {i32});
  declarations.require("__vrc_opencl_static_tensor", {i32, i32}, {i32});
  declarations.require("udonclCreateBufferForQueue",
                       {pointer, builder.getI64Type(), i32, pointer},
                       {pointer});
  return declarations.result();
}

} // namespace

namespace {

struct FinalizeONNXHostPass
    : impl::FinalizeONNXHostPassBase<FinalizeONNXHostPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(onnxHostFinalizedAttr)) {
      module.emitError("ONNX Host has already been finalized");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(onnxHostGlobalsMaterializedAttr)) {
      module.emitError(
          "ONNX Host finalization requires global materialization");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(onnxHostModelRegistryMaterializedAttr)) {
      module.emitError(
          "ONNX Host finalization requires model registry materialization");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(onnxHostMetadataAttachedAttr)) {
      module.emitError("ONNX Host finalization requires metadata attachment");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(hostGraphValidatedAttr)) {
      module.emitError(
          "ONNX Host finalization requires the graph validation pass");
      signalPassFailure();
      return;
    }
    auto graph = readONNXHostGraph(module);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(hostGraphPlanAttr)) {
      module.emitError(
          "ONNX Host finalization requires the graph planning pass");
      signalPassFailure();
      return;
    }
    auto plan = readHostPlan(module, *graph);
    if (failed(plan)) {
      signalPassFailure();
      return;
    }
    bool deviceGraph = plan->deviceGraph;
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
    // The Host binds its buffers through the UdonONNX session API.
    if (failed(declareBindingRuntime(module))) {
      signalPassFailure();
      return;
    }
    if (!deviceGraph && failed(createRunExport(module, *graph))) {
      signalPassFailure();
      return;
    }
    if (failed(createRuntimeDispatchers(module, *graph, *plan))) {
      signalPassFailure();
      return;
    }
    module->setAttr(onnxHostFinalizedAttr, UnitAttr::get(module.getContext()));
  }
};
} // namespace
} // namespace vrc
