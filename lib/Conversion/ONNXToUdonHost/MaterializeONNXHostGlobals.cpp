#include "ONNXHostABI.h"
#include "ONNXHostGraph.h"
#include "ONNXHostMaterialization.h"
#include "ONNXHostPlan.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "udoncl/udon_abi_layout.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MATERIALIZEONNXHOSTGLOBALSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace host = ::vrc::udonhl;

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

Operation *createGlobal(ModuleOp module, StringRef symbol, Type storageType,
                        Attribute initialValue, int64_t alignment,
                        bool constant = false) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  OperationState state(module.getLoc(), host::GlobalOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(symbol));
  state.addAttribute("storage_type", TypeAttr::get(storageType));
  state.addAttribute("initial_value", initialValue);
  state.addAttribute("linkage", builder.getStringAttr("internal"));
  state.addAttribute("address_space", builder.getI32IntegerAttr(0));
  state.addAttribute("alignment", builder.getI64IntegerAttr(alignment));
  if (constant)
    state.addAttribute("constant", builder.getUnitAttr());
  return builder.create(state);
}

void createTensorGlobal(ModuleOp module, const TensorRecord &tensor) {
  int64_t elements = std::max<int64_t>(tensor.type.getNumElements(), 1);
  unsigned width = elementByteSize(tensor.type.getElementType());
  Attribute initial =
      tensor.initialValue
          ? Attribute(rawInitializer(tensor.initialValue))
          : Attribute(DenseI8ArrayAttr::get(
                module.getContext(), SmallVector<int8_t>(elements * width, 0)));
  createGlobal(module, tensor.symbol,
               LLVM::LLVMArrayType::get(tensor.type.getElementType(), elements),
               initial, width, static_cast<bool>(tensor.initialValue));
}

void createI32Global(ModuleOp module, StringRef symbol) {
  Builder builder(module.getContext());
  createGlobal(module, symbol, builder.getI32Type(),
               builder.getI32IntegerAttr(0), 4);
}

void createPointerGlobal(ModuleOp module, StringRef symbol) {
  Builder builder(module.getContext());
  createGlobal(module, symbol, host::PtrType::get(module.getContext(), 0),
               builder.getI32IntegerAttr(0), 4);
}

Operation *createByteGlobal(ModuleOp module, StringRef symbol,
                            ArrayRef<int8_t> bytes, int64_t alignment = 1) {
  SmallVector<int8_t> initial(bytes);
  if (initial.empty())
    initial.push_back(0);
  Builder builder(module.getContext());
  return createGlobal(
      module, symbol,
      LLVM::LLVMArrayType::get(builder.getI8Type(), initial.size()),
      DenseI8ArrayAttr::get(module.getContext(), initial), alignment);
}

Operation *createZeroGlobal(ModuleOp module, StringRef symbol, size_t bytes,
                            int64_t alignment = 8) {
  return createByteGlobal(module, symbol, SmallVector<int8_t>(bytes, 0),
                          alignment);
}

void createStringGlobal(ModuleOp module, StringRef symbol, StringRef value) {
  SmallVector<int8_t> bytes(value.bytes_begin(), value.bytes_end());
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
  setBufferViewLayout(createZeroGlobal(module, onnxHostModelStorageSymbol,
                                       onnxHostModelStorageBytes),
                      0, 0, {});
  setBufferViewLayout(
      createZeroGlobal(module, onnxHostInputMetadataSymbol,
                       std::max<size_t>(graph.inputs.size(), 1) *
                           onnxHostMetadataStorageBytes),
      0, 0, {});
  setBufferViewLayout(
      createZeroGlobal(module, onnxHostOutputMetadataSymbol,
                       std::max<size_t>(graph.outputs.size(), 1) *
                           onnxHostMetadataStorageBytes),
      0, 0, {});
  createStringGlobal(module, onnxHostModelNameSymbol, "model");
  setBufferViewLayout(createZeroGlobal(module, onnxHostBindingStateSymbol,
                                       onnxHostBindingStateStorageBytes),
                      0, 0, {});
  setBufferViewLayout(
      createZeroGlobal(module, onnxHostBindingInputSymbol,
                       std::max<size_t>(graph.inputs.size(), 1) *
                           onnxHostBindingTensorStorageBytes),
      static_cast<int32_t>(graph.inputs.size()),
      UDONCL_UDON_ABI_ONNX_TENSOR_BYTES, {0});
  setBufferViewLayout(
      createZeroGlobal(module, onnxHostBindingOutputSymbol,
                       std::max<size_t>(graph.outputs.size(), 1) *
                           onnxHostBindingTensorStorageBytes),
      static_cast<int32_t>(graph.outputs.size()),
      UDONCL_UDON_ABI_ONNX_TENSOR_BYTES, {0});
  createZeroGlobal(
      module, onnxHostBindingIoHandlesSymbol,
      std::max<size_t>(graph.inputs.size() + graph.outputs.size(), 1) *
          onnxHostBindingBufferHandleBytes,
      4);
  createI32Global(module, onnxHostBindingEventSymbol);
  if (!graph.sequenceStates.empty()) {
    createI32Global(module, onnxHostSequenceStateOwnedSymbol);
    createI32Global(module, onnxHostSequenceStatePendingSymbol);
    createI32Global(module, onnxHostSequenceStatePoisonedSymbol);
    // The fill pattern that clears every sequence state: one zero word.
    createZeroGlobal(module, onnxHostSequenceStateZeroSymbol, 4, 4);
  }
  for (auto [index, tensorIndex] : llvm::enumerate(graph.inputs)) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    createStringGlobal(module, onnxHostIoNameSymbol(false, index), tensor.name);
    createI64ArrayGlobal(module, onnxHostIoDimensionsSymbol(false, index),
                         tensor.type.getShape());
  }
  for (auto [index, tensorIndex] : llvm::enumerate(graph.outputs)) {
    const TensorRecord &tensor = graph.tensors[tensorIndex];
    createStringGlobal(module, onnxHostIoNameSymbol(true, index), tensor.name);
    createI64ArrayGlobal(module, onnxHostIoDimensionsSymbol(true, index),
                         tensor.type.getShape());
  }
}

LogicalResult preflightGlobalSymbols(ModuleOp module, const GraphRecord &graph,
                                     const HostPlanRecord &plan) {
  llvm::StringSet<> symbols;
  auto reserve = [&](StringRef symbol) -> LogicalResult {
    if (!symbols.insert(symbol).second) {
      module.emitError() << "duplicate planned ONNX Host global symbol "
                         << symbol;
      return failure();
    }
    return success();
  };
  for (StringRef symbol :
       {onnxHostModelStorageSymbol, onnxHostInputMetadataSymbol,
        onnxHostOutputMetadataSymbol, onnxHostModelNameSymbol,
        onnxHostBindingStateSymbol, onnxHostBindingInputSymbol,
        onnxHostBindingOutputSymbol, onnxHostBindingEventSymbol,
        onnxHostBindingIoHandlesSymbol})
    if (failed(reserve(symbol)))
      return failure();
  if (!graph.sequenceStates.empty())
    for (StringRef symbol :
         {onnxHostSequenceStateOwnedSymbol, onnxHostSequenceStatePendingSymbol,
          onnxHostSequenceStatePoisonedSymbol, onnxHostSequenceStateZeroSymbol})
      if (failed(reserve(symbol)))
        return failure();
  for (auto [index, tensorIndex] : llvm::enumerate(graph.inputs)) {
    (void)tensorIndex;
    if (failed(reserve(onnxHostIoNameSymbol(false, index))) ||
        failed(reserve(onnxHostIoDimensionsSymbol(false, index))))
      return failure();
  }
  for (auto [index, tensorIndex] : llvm::enumerate(graph.outputs)) {
    (void)tensorIndex;
    if (failed(reserve(onnxHostIoNameSymbol(true, index))) ||
        failed(reserve(onnxHostIoDimensionsSymbol(true, index))))
      return failure();
  }
  if (!plan.deviceGraph) {
    for (const TensorRecord &tensor : graph.tensors)
      if (failed(reserve(tensor.symbol)))
        return failure();
    for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph.inputs)) {
      (void)tensorIndex;
      if (failed(reserve(onnxHostInputBindingSymbol(inputIndex))))
        return failure();
    }
    if (failed(reserve(onnxHostHasRunSymbol)))
      return failure();
  }
  for (unsigned tensorIndex : plan.ownedDeviceBuffers)
    if (failed(reserve(onnxHostDeviceBufferSymbol(tensorIndex))) ||
        failed(reserve(onnxHostDeviceBufferOwnedSymbol(tensorIndex))))
      return failure();
  if (plan.deviceGraph && plan.operatorScratchBytes &&
      (failed(reserve(onnxHostOperatorScratchSymbol)) ||
       failed(reserve(onnxHostOperatorScratchOwnedSymbol))))
    return failure();

  for (const auto &entry : symbols)
    if (Operation *existing =
            SymbolTable::lookupSymbolIn(module, entry.getKey()))
      return existing->emitError() << "ONNX Host global symbol "
                                   << entry.getKey() << " is already defined";
  return success();
}

struct MaterializeONNXHostGlobalsPass
    : impl::MaterializeONNXHostGlobalsPassBase<MaterializeONNXHostGlobalsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<host::UdonHLDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(onnxHostGlobalsMaterializedAttr)) {
      module.emitError("ONNX Host globals have already been materialized");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(hostGraphValidatedAttr)) {
      module.emitError(
          "ONNX Host global materialization requires graph validation");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(hostGraphPlanAttr)) {
      module.emitError(
          "ONNX Host global materialization requires graph planning");
      signalPassFailure();
      return;
    }
    auto graph = readONNXHostGraph(module);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    auto plan = readHostPlan(module, *graph);
    if (failed(plan)) {
      signalPassFailure();
      return;
    }
    if (failed(preflightGlobalSymbols(module, *graph, *plan))) {
      signalPassFailure();
      return;
    }

    if (!plan->deviceGraph)
      for (const TensorRecord &tensor : graph->tensors)
        createTensorGlobal(module, tensor);
    createModelMetadataGlobals(module, *graph);
    if (!plan->deviceGraph) {
      for (auto [inputIndex, tensorIndex] : llvm::enumerate(graph->inputs)) {
        (void)tensorIndex;
        createI32Global(module, onnxHostInputBindingSymbol(inputIndex));
      }
      createI32Global(module, onnxHostHasRunSymbol);
    }
    for (unsigned tensorIndex : plan->ownedDeviceBuffers) {
      createPointerGlobal(module, onnxHostDeviceBufferSymbol(tensorIndex));
      createI32Global(module, onnxHostDeviceBufferOwnedSymbol(tensorIndex));
    }
    if (plan->deviceGraph && plan->operatorScratchBytes) {
      createPointerGlobal(module, onnxHostOperatorScratchSymbol);
      createI32Global(module, onnxHostOperatorScratchOwnedSymbol);
    }
    module->setAttr(onnxHostGlobalsMaterializedAttr,
                    UnitAttr::get(module.getContext()));
  }
};
} // namespace
} // namespace vrc
