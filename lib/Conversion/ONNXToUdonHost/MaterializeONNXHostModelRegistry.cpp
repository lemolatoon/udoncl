#include "ONNXHostABI.h"
#include "ONNXHostGraph.h"
#include "ONNXHostIRBuilder.h"
#include "ONNXHostMaterialization.h"
#include "ONNXHostPlan.h"
#include "ONNXHostSymbols.h"
#include "VRC/Conversion/Passes.h"
#include "udoncl/udon_abi_layout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MATERIALIZEONNXHOSTMODELREGISTRYPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace host = ::vrc::udonhl;

LogicalResult validateModelMetadata(ModuleOp module, const GraphRecord &graph) {
  auto validate = [&](ArrayRef<unsigned> indices) {
    for (unsigned tensorIndex : indices) {
      const TensorRecord &tensor = graph.tensors[tensorIndex];
      if (tensor.name.empty()) {
        module.emitError("ONNX model input/output name must not be empty");
        return failure();
      }
      if (tensor.type.getRank() > UDONCL_UDON_ABI_ONNX_MAX_TENSOR_RANK) {
        module.emitError("ONNX model input/output rank exceeds Udon ABI limit");
        return failure();
      }
    }
    return success();
  };
  if (failed(validate(graph.inputs)) || failed(validate(graph.outputs)))
    return failure();
  return success();
}

LogicalResult materializeModelRegistry(ModuleOp module,
                                       const GraphRecord &graph) {
  if (failed(validateModelMetadata(module, graph)))
    return failure();

  OpBuilder builder(module.getBodyRegion());
  Location location = module.getLoc();
  Type i32 = builder.getI32Type();
  Type pointer = host::PtrType::get(module.getContext(), 0);
  ONNXHostDeclarations symbols(module);
  auto count = symbols.reservePublicDefinition("udononnxGeneratedGetModelCount",
                                               {}, {i32});
  auto get = symbols.reservePublicDefinition("udononnxGeneratedGetModel",
                                             {i32, pointer}, {i32});
  if (failed(count) || failed(get) || failed(symbols.result()))
    return failure();
  count->setPublic();
  get->setPublic();

  auto constant = [&](int64_t value) -> Value {
    return arith::ConstantIntOp::create(builder, location, value, 32);
  };

  Block *countEntry = count->addEntryBlock();
  builder.setInsertionPointToEnd(countEntry);
  func::ReturnOp::create(builder, location, constant(1));

  Block *entry = get->addEntryBlock();
  Block *body = get->addBlock();
  Block *invalid = get->addBlock();
  builder.setInsertionPointToEnd(entry);
  Value isModel =
      arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                            entry->getArgument(0), constant(0));
  Value modelOutput =
      onnxHostPointerToInt(builder, location, entry->getArgument(1));
  Value hasOutput = arith::CmpIOp::create(
      builder, location, arith::CmpIPredicate::ne, modelOutput, constant(0));
  Value valid = arith::AndIOp::create(builder, location, isModel, hasOutput);
  cf::CondBranchOp::create(builder, location, valid, body, ValueRange{},
                           invalid, ValueRange{});
  builder.setInsertionPointToEnd(invalid);
  func::ReturnOp::create(builder, location, constant(1));
  builder.setInsertionPointToEnd(body);

  auto storeAt = [&](Value base, uint32_t offset, Value value) {
    Value field =
        offset ? onnxHostPointerAdd(builder, location, base, constant(offset))
               : base;
    onnxHostStore(builder, location, value, field);
  };
  auto initializeMetadata = [&](bool output, unsigned index,
                                const TensorRecord &tensor) {
    StringRef storage =
        output ? onnxHostOutputMetadataSymbol : onnxHostInputMetadataSymbol;
    Value base = onnxHostAddressOf(builder, location, storage);
    Value metadata = onnxHostPointerAdd(
        builder, location, base,
        constant(static_cast<int64_t>(index) *
                 UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_BYTES));
    Value dimensions =
        tensor.type.getRank()
            ? onnxHostAddressOf(builder, location,
                                onnxHostIoDimensionsSymbol(output, index))
            : onnxHostNullPointer(builder, location);
    storeAt(metadata, UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_NAME_OFFSET,
            onnxHostAddressOf(builder, location,
                              onnxHostIoNameSymbol(output, index)));
    storeAt(metadata, UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_DATA_TYPE_OFFSET,
            constant(onnxHostDataType(tensor.type.getElementType())));
    storeAt(metadata, UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_RANK_OFFSET,
            constant(tensor.type.getRank()));
    storeAt(metadata, UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_DIMENSIONS_OFFSET,
            dimensions);
    storeAt(metadata,
            UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_DIMENSION_BOUNDS_OFFSET,
            onnxHostNullPointer(builder, location));
  };
  for (auto [index, tensorIndex] : llvm::enumerate(graph.inputs))
    initializeMetadata(false, index, graph.tensors[tensorIndex]);
  for (auto [index, tensorIndex] : llvm::enumerate(graph.outputs))
    initializeMetadata(true, index, graph.tensors[tensorIndex]);

  Value model =
      onnxHostAddressOf(builder, location, onnxHostModelStorageSymbol);
  storeAt(model, UDONCL_UDON_ABI_ONNX_MODEL_NAME_OFFSET,
          onnxHostAddressOf(builder, location, onnxHostModelNameSymbol));
  storeAt(model, UDONCL_UDON_ABI_ONNX_MODEL_GRAPH_ID_OFFSET, constant(0));
  storeAt(model, UDONCL_UDON_ABI_ONNX_MODEL_INPUTS_OFFSET,
          onnxHostAddressOf(builder, location, onnxHostInputMetadataSymbol));
  storeAt(model, UDONCL_UDON_ABI_ONNX_MODEL_INPUT_COUNT_OFFSET,
          constant(graph.inputs.size()));
  storeAt(model, UDONCL_UDON_ABI_ONNX_MODEL_OUTPUTS_OFFSET,
          onnxHostAddressOf(builder, location, onnxHostOutputMetadataSymbol));
  storeAt(model, UDONCL_UDON_ABI_ONNX_MODEL_OUTPUT_COUNT_OFFSET,
          constant(graph.outputs.size()));
  onnxHostStore(builder, location, model, entry->getArgument(1));
  func::ReturnOp::create(builder, location, constant(0));
  return success();
}

struct MaterializeONNXHostModelRegistryPass
    : impl::MaterializeONNXHostModelRegistryPassBase<
          MaterializeONNXHostModelRegistryPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<host::UdonHLDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(onnxHostModelRegistryMaterializedAttr)) {
      module.emitError(
          "ONNX Host model registry has already been materialized");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(onnxHostGlobalsMaterializedAttr)) {
      module.emitError(
          "ONNX Host model registry materialization requires globals");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(hostGraphValidatedAttr)) {
      module.emitError(
          "ONNX Host model registry materialization requires graph validation");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(hostGraphPlanAttr)) {
      module.emitError(
          "ONNX Host model registry materialization requires graph planning");
      signalPassFailure();
      return;
    }
    auto graph = readONNXHostGraph(module);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    if (failed(readHostPlan(module, *graph))) {
      signalPassFailure();
      return;
    }
    if (failed(materializeModelRegistry(module, *graph))) {
      signalPassFailure();
      return;
    }
    module->setAttr(onnxHostModelRegistryMaterializedAttr,
                    UnitAttr::get(module.getContext()));
  }
};
} // namespace
} // namespace vrc
