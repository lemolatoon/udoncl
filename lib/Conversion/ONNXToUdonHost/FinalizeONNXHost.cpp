#include "ONNXHostABI.h"
#include "ONNXHostFusion.h"
#include "ONNXHostFusionKernel.h"
#include "ONNXHostGraph.h"
#include "ONNXHostIRBuilder.h"
#include "ONNXHostMaterialization.h"
#include "ONNXHostPlan.h"
#include "ONNXHostSequenceExtents.h"
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
constexpr StringRef bindingStateSymbol = onnxHostBindingStateSymbol;
constexpr StringRef bindingInputSymbol = onnxHostBindingInputSymbol;
constexpr StringRef bindingOutputSymbol = onnxHostBindingOutputSymbol;
constexpr StringRef bindingEventSymbol = onnxHostBindingEventSymbol;
constexpr StringRef bindingIoHandlesSymbol = onnxHostBindingIoHandlesSymbol;
constexpr StringRef sequenceStateOwnedSymbol = onnxHostSequenceStateOwnedSymbol;
constexpr StringRef sequenceStatePendingSymbol =
    onnxHostSequenceStatePendingSymbol;
constexpr StringRef sequenceStatePoisonedSymbol =
    onnxHostSequenceStatePoisonedSymbol;
constexpr StringRef sequenceStateZeroSymbol = onnxHostSequenceStateZeroSymbol;
constexpr StringRef sequenceStateBankSymbol = onnxHostSequenceStateBankSymbol;
constexpr StringRef sequenceStateAdvancedRowsSymbol =
    onnxHostSequenceStateAdvancedRowsSymbol;
constexpr StringRef sequenceStateFillAgeSymbol =
    onnxHostSequenceStateFillAgeSymbol;
constexpr StringRef sequenceStateExposedSymbol =
    onnxHostSequenceStateExposedSymbol;
constexpr StringRef hasRunSymbol = onnxHostHasRunSymbol;
constexpr int32_t bindingStateStorageBytes = onnxHostBindingStateStorageBytes;
constexpr int32_t bindingTensorStorageBytes = onnxHostBindingTensorStorageBytes;
constexpr int32_t bindingBufferHandleBytes = onnxHostBindingBufferHandleBytes;
constexpr int32_t maximumTensorRank = onnxHostMaximumTensorRank;

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

bool isBankedSequenceState(const HostPlanRecord &plan, size_t stateIndex) {
  return llvm::is_contained(plan.sequenceStateBanks, stateIndex);
}

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

LogicalResult createPhysicalBindingExports(ModuleOp module,
                                           const GraphRecord &graph,
                                           const HostPlanRecord &plan) {
  ONNXHostDeclarations declarations(module);
  OpBuilder builder(module.getBodyRegion());
  Location location = module.getLoc();
  Type i32 = builder.getI32Type();
  Type pointer = host::PtrType::get(module.getContext(), 0);
  auto constant = [&](int64_t value) -> Value {
    return arith::ConstantIntOp::create(builder, location, value, 32);
  };
  bool banked = !plan.sequenceStateBanks.empty();
  // create, bind-all and reset leave the carried rows of a banked state in
  // the past input, laid out as the ONNX past tensor.
  auto resetBank = [&]() {
    if (banked)
      store(builder, location, constant(0),
            addressOf(builder, location, sequenceStateBankSymbol));
  };
  // The fill age (ONNXHostSequenceExtents.h): advances since the binding
  // zero-filled the past inputs, plus one, or 0 when it cannot vouch for
  // their contents. Only a graph whose runs skip zero rows keeps it.
  bool fillAge = deviceSequenceExtents(graph).used();
  auto storeFillAge = [&](Value value) {
    if (fillAge)
      store(builder, location, value,
            addressOf(builder, location, sequenceStateFillAgeSymbol));
  };
  auto loadFillAge = [&]() {
    return loadI32(builder, location,
                   addressOf(builder, location, sequenceStateFillAgeSymbol));
  };
  auto state = [&]() -> Value {
    return addressOf(builder, location, bindingStateSymbol);
  };
  // The binding functions are roots of the Host the user C calls into, never
  // U# exports of their own: udoncl.internal_export keeps them out of the
  // generated binding.
  auto markExport = [&](func::FuncOp function) {
    function->setAttr("udoncl.export", builder.getUnitAttr());
    function->setAttr("udoncl.internal_export", builder.getUnitAttr());
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
    resetBank();
    // New buffers: nothing is filled, and no caller holds a past input.
    storeFillAge(constant(0));
    if (fillAge)
      store(builder, location, constant(0),
            addressOf(builder, location, sequenceStateExposedSymbol));
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
    if (banked) {
      Value bank = addressOf(builder, location, sequenceStateBankSymbol);
      store(builder, location,
            arith::SelectOp::create(builder, location, invalidate, constant(0),
                                    loadI32(builder, location, bank)),
            bank);
    }
    if (fillAge)
      storeFillAge(arith::SelectOp::create(builder, location, invalidate,
                                           constant(0), loadFillAge()));
  }
  func::ReturnOp::create(builder, location, bindCall.getResult(0));

  builder.setInsertionPointToEnd(module.getBody());
  auto bindAll =
      func::FuncOp::create(builder, location, "udon_onnx_binding_bind_all",
                           FunctionType::get(module.getContext(), {}, {i32}));
  markExport(bindAll);
  entry = bindAll.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  auto beginStaticBind = func::CallOp::create(
      builder, location, "udononnxBindingStateBeginStaticBind", TypeRange{i32},
      state());
  returnIfNonzero(bindAll, beginStaticBind.getResult(0));
  auto returnIfViewFailed = [&](Value status) {
    Block *next = bindAll.addBlock();
    Block *failure = bindAll.addBlock();
    Value succeeded = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, status, constant(0));
    cf::CondBranchOp::create(builder, location, succeeded, next, ValueRange{},
                             failure, ValueRange{});
    builder.setInsertionPointToEnd(failure);
    Value stale = arith::CmpIOp::create(
        builder, location, arith::CmpIPredicate::eq, status, constant(3));
    Value mapped = arith::SelectOp::create(builder, location, stale,
                                           constant(3), constant(1));
    auto completed = func::CallOp::create(
        builder, location, "udononnxBindingStateCompleteStaticBind",
        TypeRange{i32}, ValueRange{state(), mapped});
    func::ReturnOp::create(builder, location, completed.getResult(0));
    builder.setInsertionPointToEnd(next);
  };
  auto bindOne = [&](bool output, size_t publicIndex, unsigned tensorIndex,
                     size_t ioHandleIndex) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    Value bindings = addressOf(
        builder, location, output ? bindingOutputSymbol : bindingInputSymbol);
    Value target =
        pointerAdd(builder, location, bindings,
                   constant(publicIndex * UDONCL_UDON_ABI_ONNX_TENSOR_BYTES));
    Value handle = handleAddress(ioHandleIndex);
    auto viewed = func::CallOp::create(
        builder, location, "udonclDeviceBufferView", TypeRange{i32},
        ValueRange{handle, constant(0), constant(tensor.type.getNumElements()),
                   constant(udonclDataType(tensor.type.getElementType())),
                   target});
    returnIfViewFailed(viewed.getResult(0));
    store(builder, location, constant(tensor.type.getRank()),
          pointerAdd(builder, location, target,
                     constant(UDONCL_UDON_ABI_ONNX_TENSOR_RANK_OFFSET)));
    for (int64_t axis = 0; axis != maximumTensorRank; ++axis) {
      int64_t dimension =
          axis < tensor.type.getRank() ? tensor.type.getShape()[axis] : 0;
      Value field = pointerAdd(
          builder, location, target,
          constant(UDONCL_UDON_ABI_ONNX_TENSOR_DIMENSIONS_OFFSET +
                   axis * UDONCL_UDON_ABI_ONNX_TENSOR_DIMENSION_STRIDE));
      store(builder, location,
            arith::ConstantIntOp::create(builder, location, dimension, 64),
            field);
    }
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
    resetBank();
    storeFillAge(constant(0));
  }
  auto completedStaticBind = func::CallOp::create(
      builder, location, "udononnxBindingStateCompleteStaticBind",
      TypeRange{i32}, ValueRange{state(), constant(0)});
  func::ReturnOp::create(builder, location, completedStaticBind.getResult(0));

  auto createResolve = [&](StringRef name, StringRef callee, bool input) {
    builder.setInsertionPointToEnd(module.getBody());
    auto resolve = func::FuncOp::create(
        builder, location, name,
        FunctionType::get(module.getContext(), {i32, pointer, pointer, pointer},
                          {i32}));
    markExport(resolve);
    Block *resolveEntry = resolve.addEntryBlock();
    builder.setInsertionPointToEnd(resolveEntry);
    if (input && fillAge) {
      // A caller holding a past input may write it whenever it likes, so the
      // binding stops vouching for any fill from now on.
      Value exposed = addressOf(builder, location, sequenceStateExposedSymbol);
      Value pastInput = arith::ConstantIntOp::create(builder, location, 0, 1);
      for (const SequenceStateRecord &sequence : graph.sequenceStates)
        pastInput = arith::OrIOp::create(
            builder, location, pastInput,
            arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                  resolveEntry->getArgument(0),
                                  constant(sequence.inputIndex)));
      store(builder, location,
            arith::SelectOp::create(builder, location, pastInput, constant(1),
                                    loadI32(builder, location, exposed)),
            exposed);
      storeFillAge(arith::SelectOp::create(builder, location, pastInput,
                                           constant(0), loadFillAge()));
    }
    auto resolved = func::CallOp::create(
        builder, location, callee, TypeRange{i32},
        ValueRange{state(), resolveEntry->getArgument(0),
                   resolveEntry->getArgument(1), resolveEntry->getArgument(2),
                   resolveEntry->getArgument(3)});
    func::ReturnOp::create(builder, location, resolved.getResult(0));
  };
  createResolve("udon_onnx_binding_resolve_input",
                "udononnxBindingStateResolveInput", true);
  createResolve("udon_onnx_binding_resolve_output",
                "udononnxBindingStateResolveOutput", false);

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
  if (fillAge) {
    // A copy into a past input replaces its fill.
    Value pastInput = arith::ConstantIntOp::create(builder, location, 0, 1);
    for (const SequenceStateRecord &sequence : graph.sequenceStates)
      pastInput = arith::OrIOp::create(
          builder, location, pastInput,
          arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                copyEntry->getArgument(1),
                                constant(sequence.inputIndex)));
    storeFillAge(arith::SelectOp::create(builder, location, pastInput,
                                         constant(0), loadFillAge()));
  }
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
    // A transfer still waiting in the binding's command slot reaches the
    // Device before the state is cleared: drain the session queue, then zero
    // every past state with one word-fill command each. A fill is recorded
    // on the Device queue at issue and needs no staged zero array.
    Value queueSlot = allocaI32(builder, location);
    auto gotQueue = func::CallOp::create(
        builder, location, "udononnxBindingStateGetCommandQueue",
        TypeRange{i32}, ValueRange{state(), queueSlot});
    resetReturnIfFailed(gotQueue.getResult(0));
    auto finished = func::CallOp::create(
        builder, location, "clFinish", TypeRange{i32},
        ValueRange{loadPointer(builder, location, queueSlot)});
    resetReturnIfFailed(finished.getResult(0));
    for (const SequenceStateRecord &sequence : graph.sequenceStates) {
      unsigned inputTensorIndex = graph.inputs[sequence.inputIndex];
      const TensorRecord &input = graph.tensors[inputTensorIndex];
      int64_t inputBytes = input.type.getNumElements() *
                           elementByteSize(input.type.getElementType());
      // The buffer holds a whole present; only the past tensor is cleared.
      Value destination =
          resetPhysicalBuffer(sequence.inputIndex + 1,
                              deviceAllocationBytes(graph, inputTensorIndex));
      auto filled = func::CallOp::create(
          builder, location, "__vrc_opencl_buffer_fill", TypeRange{i32},
          ValueRange{destination, constant(0), constant(inputBytes),
                     addressOf(builder, location, sequenceStateZeroSymbol),
                     constant(4)});
      auto awaited =
          func::CallOp::create(builder, location, "__vrc_opencl_await",
                               TypeRange{i32}, filled.getResult(0));
      resetReturnIfFailed(awaited.getResult(0));
    }
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStatePendingSymbol));
    store(builder, location, constant(0),
          addressOf(builder, location, sequenceStatePoisonedSymbol));
    resetBank();
    // Every past input now holds zeros, unless a caller holds one.
    if (fillAge)
      storeFillAge(arith::SelectOp::create(
          builder, location,
          arith::CmpIOp::create(
              builder, location, arith::CmpIPredicate::eq,
              loadI32(builder, location,
                      addressOf(builder, location, sequenceStateExposedSymbol)),
              constant(0)),
          constant(1), constant(0)));
    func::ReturnOp::create(builder, location, constant(0));

    // The advance commits the rows the pending run appended. The plain
    // advance commits all of them; with a `rows` argument
    // (sequenceStatePartialAdvanceStep) only the first `rows`, and the rows
    // after them, which the run computed from padding, are never carried:
    // the next run's carried rows start `rows` rows into the previous
    // present instead of a whole step.
    int64_t partialStep = sequenceStatePartialAdvanceStep(graph);
    auto emitAdvance = [&](StringRef name, bool rowsArgument) -> LogicalResult {
      builder.setInsertionPointToEnd(module.getBody());
      SmallVector<Type> parameters;
      if (rowsArgument)
        parameters.push_back(i32);
      auto advance = func::FuncOp::create(
          builder, location, name,
          FunctionType::get(module.getContext(), parameters, {i32}));
      // Only a C Host that runs part of a sequence calls the partial
      // advance, so like materialization it is not a Udon export.
      if (!rowsArgument)
        markExport(advance);
      // An advance that only flips banks issues no Device command and never
      // suspends; the Udon async lowering supports no coroutine without a
      // suspension point.
      if (plan.sequenceStateBanks.size() != graph.sequenceStates.size())
        advance->setAttr("vrc.async_effect",
                         builder.getStringAttr("may_suspend"));
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
      cf::CondBranchOp::create(builder, location, idle, checkState,
                               ValueRange{}, advanceBusy, ValueRange{});
      builder.setInsertionPointToEnd(advanceBusy);
      func::ReturnOp::create(builder, location, constant(7));
      builder.setInsertionPointToEnd(checkState);
      cf::CondBranchOp::create(builder, location, valid, advanceReady,
                               ValueRange{}, advanceInvalid, ValueRange{});
      builder.setInsertionPointToEnd(advanceInvalid);
      func::ReturnOp::create(builder, location, constant(12));
      builder.setInsertionPointToEnd(advanceReady);
      // The rows this advance commits of every state: a whole step unless
      // the caller passes fewer.
      Value rows;
      if (rowsArgument) {
        rows = advanceEntry->getArgument(0);
        Block *rowsValid = advance.addBlock();
        Block *rowsInvalid = advance.addBlock();
        cf::CondBranchOp::create(
            builder, location,
            arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ule,
                                  rows, constant(partialStep)),
            rowsValid, ValueRange{}, rowsInvalid, ValueRange{});
        builder.setInsertionPointToEnd(rowsInvalid);
        func::ReturnOp::create(builder, location, constant(1));
        builder.setInsertionPointToEnd(rowsValid);
      } else if (partialStep) {
        rows = constant(partialStep);
      }
      auto advanceReturnIfNonzero = [&](Value status,
                                        int64_t failureStatus = -1) {
        Block *next = advance.addBlock();
        Block *failure = advance.addBlock();
        Value succeeded = arith::CmpIOp::create(
            builder, location, arith::CmpIPredicate::eq, status, constant(0));
        cf::CondBranchOp::create(builder, location, succeeded, next,
                                 ValueRange{}, failure, ValueRange{});
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
                   builder, location, "__vrc_opencl_buffer_alloc",
                   TypeRange{i32},
                   ValueRange{constant(logicalId), constant(bytes)})
            .getResult(0);
      };
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
      for (auto [stateIndex, sequence] :
           llvm::enumerate(graph.sequenceStates)) {
        // A banked state moves nothing: the bank flip below makes the rows
        // the last run wrote the carried rows of the next run.
        if (isBankedSequenceState(plan, stateIndex))
          continue;
        unsigned inputTensorIndex = graph.inputs[sequence.inputIndex];
        unsigned outputTensorIndex = graph.outputs[sequence.outputIndex];
        const TensorRecord &input = graph.tensors[inputTensorIndex];
        const TensorRecord &output = graph.tensors[outputTensorIndex];
        // The copying advance is one udononnx_slice_f32 dispatch, which
        // addresses at most 16384 words; a banked state has no such limit.
        if (input.type.getNumElements() > 16384 ||
            output.type.getNumElements() > 16384)
          return emitError(location)
                 << "ONNX sequence state that is not banked exceeds the "
                    "16384 elements of its copying advance";
        Value source =
            physicalBuffer(graph.inputs.size() + sequence.outputIndex + 1,
                           deviceAllocationBytes(graph, outputTensorIndex));
        Value destination =
            physicalBuffer(sequence.inputIndex + 1,
                           deviceAllocationBytes(graph, inputTensorIndex));
        SmallVector<int64_t> starts(input.type.getRank(), 0);
        SmallVector<int64_t> steps(input.type.getRank(), 1);
        starts[sequence.axis] = rowsArgument ? 0 : sequence.step;
        SmallVector<Value> packedStarts{constant(packUnsigned(starts, 0, 0)),
                                        constant(packUnsigned(starts, 2, 0)),
                                        constant(packUnsigned(starts, 4, 0))};
        // The slice starts `rows` rows into the present: the axis's half of
        // its packed start word is filled in at run time.
        if (rowsArgument) {
          Value &word = packedStarts[sequence.axis / 2];
          Value shifted = sequence.axis % 2
                              ? Value(arith::ShLIOp::create(builder, location,
                                                            rows, constant(16)))
                              : rows;
          word = arith::OrIOp::create(builder, location, word, shifted);
        }
        auto dispatched = func::CallOp::create(
            builder, location, "__vrc_opencl_dispatch_udononnx_slice_f32",
            TypeRange{i32},
            ValueRange{source, destination,
                       constant(input.type.getNumElements()),
                       constant(input.type.getRank()),
                       constant(packUnsigned(output.type.getShape(), 0, 1)),
                       constant(packUnsigned(output.type.getShape(), 2, 1)),
                       constant(packUnsigned(output.type.getShape(), 4, 1)),
                       constant(packUnsigned(input.type.getShape(), 0, 1)),
                       constant(packUnsigned(input.type.getShape(), 2, 1)),
                       constant(packUnsigned(input.type.getShape(), 4, 1)),
                       packedStarts[0], packedStarts[1], packedStarts[2],
                       constant(packUnsigned(steps, 0, 1)),
                       constant(packUnsigned(steps, 2, 1)),
                       constant(packUnsigned(steps, 4, 1))});
        auto awaited =
            func::CallOp::create(builder, location, "__vrc_opencl_await",
                                 TypeRange{i32}, dispatched.getResult(0));
        advanceReturnIfNonzero(awaited.getResult(0), 5);
      }
      if (banked) {
        // The pending run wrote the present into the past input from bank 1
        // and into the present output otherwise.
        Value bank = addressOf(builder, location, sequenceStateBankSymbol);
        Value fromInput = arith::CmpIOp::create(
            builder, location, arith::CmpIPredicate::eq,
            loadI32(builder, location, bank), constant(1));
        store(builder, location,
              arith::SelectOp::create(builder, location, fromInput, constant(2),
                                      constant(1)),
              bank);
        // The next run reads its carried rows this many rows into that
        // present.
        if (partialStep)
          store(builder, location, rows,
                addressOf(builder, location, sequenceStateAdvancedRowsSymbol));
      }
      if (fillAge && partialStep) {
        // The age counts committed rows (sequenceStateFillAgeRows). Past
        // the limit no state has zero rows left, so it stays there and
        // never overflows.
        Value age = loadFillAge();
        Value known = arith::CmpIOp::create(
            builder, location, arith::CmpIPredicate::ne, age, constant(0));
        Value counted = arith::MinUIOp::create(
            builder, location,
            arith::AddIOp::create(builder, location, age, rows),
            constant(sequenceStateFillAgeLimit(graph)));
        storeFillAge(
            arith::SelectOp::create(builder, location, known, counted, age));
      } else if (fillAge) {
        // One more advance since the fill. Past the limit no state has zero
        // rows left, so the count stops there and never overflows.
        Value age = loadFillAge();
        Value counting = arith::AndIOp::create(
            builder, location,
            arith::CmpIOp::create(builder, location, arith::CmpIPredicate::ne,
                                  age, constant(0)),
            arith::CmpIOp::create(builder, location, arith::CmpIPredicate::slt,
                                  age,
                                  constant(sequenceStateFillAgeLimit(graph))));
        storeFillAge(arith::SelectOp::create(
            builder, location, counting,
            arith::AddIOp::create(builder, location, age, constant(1)), age));
      }
      store(builder, location, constant(0),
            addressOf(builder, location, sequenceStatePendingSymbol));
      func::ReturnOp::create(builder, location, constant(0));
      return success();
    };
    if (failed(emitAdvance("udon_onnx_binding_enqueue_advance_sequence_state",
                           /*rowsArgument=*/false)))
      return failure();
    if (partialStep &&
        failed(
            emitAdvance("udon_onnx_binding_enqueue_advance_sequence_state_rows",
                        /*rowsArgument=*/true)))
      return failure();
  }

  if (banked) {
    // The present of a banked state is in its past input after a run from
    // bank 1: while that run is pending, and once it was advanced to bank 2.
    // Materializing copies it into the present output, where the presents of
    // all other runs already are, so that a caller reads every present from
    // its output. The bank and the carried rows stay as they are.
    builder.setInsertionPointToEnd(module.getBody());
    auto materialize = func::FuncOp::create(
        builder, location,
        "udon_onnx_binding_enqueue_materialize_sequence_state",
        FunctionType::get(module.getContext(), {}, {i32}));
    // Only a C Host that reads presents calls it, so it is not a Udon export
    // and an artifact that never calls it does not keep it.
    materialize->setAttr("vrc.async_effect",
                         builder.getStringAttr("may_suspend"));
    Block *materializeEntry = materialize.addEntryBlock();
    Block *materializeCheck = materialize.addBlock();
    Block *materializeBusy = materialize.addBlock();
    Block *materializeInvalid = materialize.addBlock();
    Block *materializeCopy = materialize.addBlock();
    Block *materializeDone = materialize.addBlock();
    builder.setInsertionPointToEnd(materializeEntry);
    auto loadState = [&](StringRef symbol) {
      return loadI32(builder, location, addressOf(builder, location, symbol));
    };
    auto equals = [&](Value value, int64_t expected) -> Value {
      return arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                   value, constant(expected));
    };
    cf::CondBranchOp::create(
        builder, location, equals(loadState(bindingEventSymbol), 0),
        materializeCheck, ValueRange{}, materializeBusy, ValueRange{});
    builder.setInsertionPointToEnd(materializeBusy);
    func::ReturnOp::create(builder, location, constant(7));
    builder.setInsertionPointToEnd(materializeCheck);
    Value healthy = arith::AndIOp::create(
        builder, location, equals(loadState(sequenceStateOwnedSymbol), 1),
        equals(loadState(sequenceStatePoisonedSymbol), 0));
    Block *materializeBank = materialize.addBlock();
    cf::CondBranchOp::create(builder, location, healthy, materializeBank,
                             ValueRange{}, materializeInvalid, ValueRange{});
    builder.setInsertionPointToEnd(materializeInvalid);
    func::ReturnOp::create(builder, location, constant(12));
    builder.setInsertionPointToEnd(materializeBank);
    Value bank = loadState(sequenceStateBankSymbol);
    Value inInput = arith::SelectOp::create(
        builder, location, equals(loadState(sequenceStatePendingSymbol), 1),
        equals(bank, 1), equals(bank, 2));
    cf::CondBranchOp::create(builder, location, inInput, materializeCopy,
                             ValueRange{}, materializeDone, ValueRange{});
    builder.setInsertionPointToEnd(materializeDone);
    func::ReturnOp::create(builder, location, constant(0));
    builder.setInsertionPointToEnd(materializeCopy);
    for (unsigned stateIndex : plan.sequenceStateBanks) {
      const SequenceStateRecord &sequence = graph.sequenceStates[stateIndex];
      unsigned inputTensorIndex = graph.inputs[sequence.inputIndex];
      unsigned outputTensorIndex = graph.outputs[sequence.outputIndex];
      const TensorRecord &present = graph.tensors[outputTensorIndex];
      int64_t presentBytes = present.type.getNumElements() *
                             elementByteSize(present.type.getElementType());
      Value source =
          func::CallOp::create(
              builder, location, "__vrc_opencl_buffer_alloc", TypeRange{i32},
              ValueRange{
                  constant(sequence.inputIndex + 1),
                  constant(deviceAllocationBytes(graph, inputTensorIndex))})
              .getResult(0);
      Value destination =
          func::CallOp::create(
              builder, location, "__vrc_opencl_buffer_alloc", TypeRange{i32},
              ValueRange{
                  constant(graph.inputs.size() + sequence.outputIndex + 1),
                  constant(deviceAllocationBytes(graph, outputTensorIndex))})
              .getResult(0);
      auto copied = func::CallOp::create(
          builder, location, "__vrc_opencl_buffer_copy", TypeRange{i32},
          ValueRange{source, destination, constant(0), constant(0),
                     constant(presentBytes)});
      auto awaited =
          func::CallOp::create(builder, location, "__vrc_opencl_await",
                               TypeRange{i32}, copied.getResult(0));
      Block *next = materialize.addBlock();
      Block *failure = materialize.addBlock();
      cf::CondBranchOp::create(builder, location,
                               equals(awaited.getResult(0), 0), next,
                               ValueRange{}, failure, ValueRange{});
      builder.setInsertionPointToEnd(failure);
      func::ReturnOp::create(builder, location, constant(5));
      builder.setInsertionPointToEnd(next);
    }
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
    resetBank();
    storeFillAge(constant(0));
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

  declarations.require(
      "udononnxBindingStateCreate",
      {pointer, i32, pointer, pointer, i32, pointer, i32, pointer, i32}, {i32});
  declarations.require("udononnxBindingStateAllocateBuffer",
                       {pointer, i32, i32, pointer}, {i32});
  declarations.require(
      "udononnxBindingStateBindTensor",
      {pointer, i32, i32, i32, i32, i32, i32, i32, i32, pointer}, {i32});
  declarations.require("udononnxBindingStateBeginStaticBind", {pointer}, {i32});
  declarations.require("udononnxBindingStateCompleteStaticBind", {pointer, i32},
                       {i32});
  declarations.require("udonclDeviceBufferView",
                       {pointer, i32, i32, i32, pointer}, {i32});
  declarations.require("udononnxBindingStateEnqueueInitialize",
                       {pointer, i32, pointer, pointer}, {i32});
  declarations.require("udononnxBindingStateEnqueueRun",
                       {pointer, i32, pointer, pointer}, {i32});
  declarations.require("udononnxBindingStateGetSession", {pointer, pointer},
                       {i32});
  declarations.require("udononnxBindingStateResolveInput",
                       {pointer, i32, pointer, pointer, pointer}, {i32});
  declarations.require("udononnxBindingStateResolveOutput",
                       {pointer, i32, pointer, pointer, pointer}, {i32});
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
  declarations.require("udononnxBindingStateGetCommandQueue",
                       {pointer, pointer}, {i32});
  declarations.require("clFinish", {pointer}, {i32});
  declarations.require("__vrc_opencl_buffer_alloc", {i32, i32}, {i32});
  declarations.require("__vrc_opencl_await", {i32}, {i32});
  declarations.require("udononnxBindingStateCompleteEvent",
                       {pointer, pointer, i32}, {i32});
  declarations.require("udononnxBindingStateDispose", {pointer}, {i32});
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
  func::ReturnOp::create(builder, location,
                         loadI32(builder, location, disposeStatus));
  declarations.require("clReleaseMemObject", {pointer}, {i32});

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
    // a run of an unchanged table (udononnxEnqueueRun has checked that its
    // buffers are alive) reads them instead of resolving every buffer again.
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
    builder.setInsertionPointToEnd(tokensReady);
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
    // The buffer of a banked present while a dispatch for bank 1 is emitted.
    llvm::DenseMap<unsigned, Value> bankOverrides;
    int64_t activeBank = 0;
    auto buffer = [&](int64_t tensorIndex) -> Value {
      unsigned storageIndex = storage[static_cast<unsigned>(tensorIndex)];
      if (Value overridden = bankOverrides.lookup(storageIndex))
        return overridden;
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
    DeviceFusionKernels fusedKernels(module, fusion);
    // Banked sequence states (sequenceStateBanks in ONNXHostPlan.h). The
    // bank, read once per run, decides where the carried rows are and where
    // the present goes; every dispatch that touches either buffer is
    // emitted once per bank it distinguishes.
    struct BankedState {
      unsigned past;
      unsigned present;
      unsigned concat;
      DeviceFusionLeafView view;
    };
    SmallVector<BankedState> bankedStates;
    // A binding that may commit part of a run reads the carried rows as many
    // rows into the previous present as the last advance committed
    // (sequenceStatePartialAdvanceStep), at most a whole step.
    bool partialAdvance = sequenceStatePartialAdvanceStep(graph) != 0;
    for (unsigned stateIndex : plan.sequenceStateBanks) {
      const SequenceStateRecord &sequence = graph.sequenceStates[stateIndex];
      unsigned past = graph.inputs[sequence.inputIndex];
      unsigned present = graph.outputs[sequence.outputIndex];
      const NodeRecord *concat =
          llvm::find_if(graph.nodes, [&](const NodeRecord &node) {
            return node.output == present;
          });
      if (concat == graph.nodes.end() || storage[past] != past ||
          storage[present] != present || !buffers[past] || !buffers[present])
        return emitError(location)
               << "banked ONNX sequence state lacks its public buffers";
      ArrayRef<int64_t> presentShape = graph.tensors[present].type.getShape();
      bankedStates.push_back(
          {past,
           present,
           static_cast<unsigned>(concat - graph.nodes.begin()),
           {past, SmallVector<int64_t>(presentShape), sequence.axis,
            sequence.step, partialAdvance}});
    }
    auto bankedConcat = [&](unsigned nodeIndex) -> const BankedState * {
      for (const BankedState &bank : bankedStates)
        if (bank.concat == nodeIndex)
          return &bank;
      return nullptr;
    };
    auto touchesBank = [&](unsigned nodeIndex) {
      const NodeRecord &node = graph.nodes[nodeIndex];
      SmallVector<unsigned> tensors;
      if (fusion.isRoot(nodeIndex))
        tensors = fusion.leaves(nodeIndex);
      for (int64_t input : node.inputs)
        if (input >= 0)
          tensors.push_back(static_cast<unsigned>(input));
      tensors.push_back(node.output);
      return llvm::any_of(tensors, [&](unsigned tensor) {
        return llvm::any_of(bankedStates, [&](const BankedState &bank) {
          return storage[tensor] == bank.past ||
                 storage[tensor] == bank.present;
        });
      });
    };
    Value bankValue, advancedRows;
    if (!bankedStates.empty())
      bankValue =
          loadI32(builder, location,
                  addressOf(builder, location, sequenceStateBankSymbol));
    if (!bankedStates.empty() && partialAdvance)
      advancedRows = loadI32(
          builder, location,
          addressOf(builder, location, sequenceStateAdvancedRowsSymbol));
    // The zero rows of every state this run skips
    // (ONNXHostSequenceExtents.h), read once from the fill age:
    // max(0, carried - (age - 1) * rows per unit) when the age is known,
    // else 0.
    DeviceSequenceExtents extents = deviceSequenceExtents(graph);
    SmallVector<Value> zeroRows(graph.sequenceStates.size());
    if (extents.used()) {
      Value age =
          loadI32(builder, location,
                  addressOf(builder, location, sequenceStateFillAgeSymbol));
      Value advances =
          arith::SubIOp::create(builder, location, age, constant(1));
      Value known = arith::CmpIOp::create(
          builder, location, arith::CmpIPredicate::ne, age, constant(0));
      SmallVector<bool> used(graph.sequenceStates.size(), false);
      for (const auto &state : extents.skippedRows)
        if (state)
          used[*state] = true;
      for (const auto &prefix : extents.constantPrefix)
        if (prefix)
          used[prefix->prefix.state] = true;
      // States of one carried length and step share the value.
      std::map<std::pair<int64_t, int64_t>, Value> shared;
      for (auto [stateIndex, sequence] :
           llvm::enumerate(graph.sequenceStates)) {
        if (!used[stateIndex])
          continue;
        Value &sharedRows = shared[{sequence.capacity, sequence.step}];
        if (sharedRows) {
          zeroRows[stateIndex] = sharedRows;
          continue;
        }
        Value rows = arith::SubIOp::create(
            builder, location, constant(sequence.capacity - sequence.step),
            arith::MulIOp::create(
                builder, location, advances,
                constant(sequenceStateFillAgeRows(graph, stateIndex))));
        Value positive = arith::CmpIOp::create(
            builder, location, arith::CmpIPredicate::sgt, rows, constant(0));
        zeroRows[stateIndex] = arith::SelectOp::create(
            builder, location,
            arith::AndIOp::create(builder, location, known, positive), rows,
            constant(0));
        sharedRows = zeroRows[stateIndex];
      }
    }
    auto constantPrefix =
        [&](unsigned nodeIndex) -> const DeviceConstantPrefix * {
      const auto &prefix = extents.constantPrefix[nodeIndex];
      return prefix ? &*prefix : nullptr;
    };
    auto emitNodeDispatch = [&](const NodeRecord &node,
                                unsigned nodeIndex) -> LogicalResult {
      Value output = buffer(node.output);
      if (!output)
        return emitError(location)
               << "Device ONNX graph output lacks physical storage";
      if (activeBank != 0) {
        if (const BankedState *bank = bankedConcat(nodeIndex)) {
          // Bank 1 reads the previous present in the present output and
          // writes the past input; bank 2 the other way around.
          Value previous =
              activeBank == 1 ? buffers[bank->present] : buffers[bank->past];
          Value next =
              activeBank == 1 ? buffers[bank->past] : buffers[bank->present];
          const DeviceConstantPrefix *prefix = constantPrefix(nodeIndex);
          FailureOr<DeviceFusionDispatch> viewed =
              fusedKernels.dispatch(nodeIndex, bank->view, prefix);
          if (failed(viewed))
            return failure();
          SmallVector<Value> arguments;
          for (unsigned leaf : viewed->leaves) {
            arguments.push_back(leaf == bank->past ? previous : buffer(leaf));
            if (!arguments.back())
              return emitError(location)
                     << "Device ONNX fusion operand lacks physical storage";
          }
          arguments.push_back(next);
          if (prefix)
            arguments.push_back(zeroRows[prefix->prefix.state]);
          if (bank->view.dynamicOffset)
            arguments.push_back(advancedRows);
          dispatch(viewed->callee, arguments);
          return success();
        }
      }
      // A node runs a fused kernel also outside every group when its library
      // kernel cannot run it: a Softmax whose schedule has two passes
      // (deviceSoftmaxSchedule), or a node beyond the limits of its library
      // kernel (isSynthesizedDeviceNode).
      if (fusion.isRoot(nodeIndex) ||
          deviceSoftmaxSchedule(graph, node).chunks ||
          isSynthesizedDeviceNode(graph, node)) {
        const DeviceConstantPrefix *prefix = constantPrefix(nodeIndex);
        FailureOr<DeviceFusionDispatch> fused =
            fusedKernels.dispatch(nodeIndex, prefix);
        if (failed(fused))
          return failure();
        SmallVector<Value> arguments;
        for (unsigned leaf : fused->leaves) {
          arguments.push_back(buffer(leaf));
          if (!arguments.back())
            return emitError(location)
                   << "Device ONNX fusion operand lacks physical storage";
        }
        // A two-pass row reduction writes its partial statistics to the
        // operator scratch, and the elementwise pass reads them there.
        if (!fused->statisticsCallee.empty()) {
          if (!operatorScratch)
            return emitError(location)
                   << "Device row reduction lacks operator scratch storage";
          arguments.push_back(operatorScratch);
          dispatch(fused->statisticsCallee, arguments);
        }
        arguments.push_back(output);
        if (prefix)
          arguments.push_back(zeroRows[prefix->prefix.state]);
        dispatch(fused->callee, arguments);
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
        dispatch(
            "__vrc_opencl_dispatch_udononnx_cast_i64_f32",
            ValueRange{
                buffer(node.inputs[0]), output,
                constant(graph.tensors[node.output].type.getNumElements())});
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
        // The first K terms the product skips (ONNXHostSequenceExtents.h);
        // only the strided-batched kernels take them.
        std::optional<unsigned> skipped = extents.skippedRows[nodeIndex];
        if (shape->outputBatches == 1 && smallOperands && !skipped) {
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
                  constant(0), constant(0), floating(1.0f),
                  skipped ? zeroRows[*skipped] : constant(0)});
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
                              floating(alpha), constant(0)});
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
      if (bankedStates.empty() || !touchesBank(nodeIndex)) {
        if (failed(emitNodeDispatch(node, nodeIndex)))
          return failure();
        continue;
      }
      // Bank 1 moves every present into its past input. The Concat of a
      // banked state also reads its carried rows differently in banks 1 and
      // 2; everything else is the same in banks 0 and 2.
      SmallVector<int64_t, 3> variants{1};
      if (bankedConcat(nodeIndex))
        variants.push_back(2);
      variants.push_back(0);
      Block *joined = run.addBlock();
      for (auto [position, variant] : llvm::enumerate(variants)) {
        if (position + 1 != variants.size()) {
          Block *selected = run.addBlock();
          Block *other = run.addBlock();
          Value matches =
              arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                    bankValue, constant(variant));
          cf::CondBranchOp::create(builder, location, matches, selected,
                                   ValueRange{}, other, ValueRange{});
          builder.setInsertionPointToEnd(selected);
          activeBank = variant;
          bankOverrides.clear();
          if (variant == 1)
            for (const BankedState &bank : bankedStates)
              bankOverrides[bank.present] = buffers[bank.past];
          if (failed(emitNodeDispatch(node, nodeIndex)))
            return failure();
          cf::BranchOp::create(builder, location, joined);
          builder.setInsertionPointToEnd(other);
          continue;
        }
        activeBank = 0;
        bankOverrides.clear();
        if (failed(emitNodeDispatch(node, nodeIndex)))
          return failure();
        cf::BranchOp::create(builder, location, joined);
      }
      activeBank = 0;
      bankOverrides.clear();
      builder.setInsertionPointToEnd(joined);
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
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, i32},
        {i32});
    declarations.require(
        "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched_large",
        {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, i32},
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

  marker = func::CallOp::create(
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
    if (failed(createPhysicalBindingExports(module, *graph, *plan))) {
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
