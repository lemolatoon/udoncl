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

  // The registry is a pure function of the graph: the model record and the
  // metadata tables are initializers whose pointer fields are relocations, so
  // looking the model up stores nothing but the result.
  struct RegistryImage {
    host::GlobalOp global;
    SmallVector<int8_t> bytes;
    SmallVector<Attribute> relocations;
  };
  auto openImage = [&](StringRef symbol) -> FailureOr<RegistryImage> {
    auto global = module.lookupSymbol<host::GlobalOp>(symbol);
    auto bytes =
        global
            ? dyn_cast_or_null<DenseI8ArrayAttr>(global.getInitialValueAttr())
            : DenseI8ArrayAttr();
    if (!bytes || global.getRelocations() || global.getAlignment() < 4) {
      module.emitError() << "ONNX Host model registry storage @" << symbol
                         << " is not an unrelocated aligned byte global";
      return failure();
    }
    return RegistryImage{global, SmallVector<int8_t>(bytes.asArrayRef()), {}};
  };
  auto writeWord = [&](RegistryImage &image, uint32_t offset,
                       uint32_t value) -> LogicalResult {
    if (offset + uint64_t{4} > image.bytes.size())
      return image.global.emitError(
          "ONNX Host model registry field is outside its storage");
    for (unsigned byte = 0; byte != 4; ++byte)
      image.bytes[offset + byte] =
          static_cast<int8_t>((value >> (byte * 8)) & 0xffu);
    return success();
  };
  auto writeAddress = [&](RegistryImage &image, uint32_t offset,
                          StringRef symbol) -> LogicalResult {
    if (failed(writeWord(image, offset, 0)))
      return failure();
    host::GlobalRelocation relocation;
    relocation.offset = offset;
    relocation.symbol = FlatSymbolRefAttr::get(module.getContext(), symbol);
    image.relocations.push_back(relocation.build(module.getContext()));
    return success();
  };
  auto commitImage = [&](RegistryImage &image) {
    image.global.setInitialValueAttr(
        DenseI8ArrayAttr::get(module.getContext(), image.bytes));
    if (!image.relocations.empty())
      image.global.setRelocationsAttr(builder.getArrayAttr(image.relocations));
  };
  auto initializeMetadata = [&](bool output,
                                ArrayRef<unsigned> tensors) -> LogicalResult {
    auto image = openImage(output ? onnxHostOutputMetadataSymbol
                                  : onnxHostInputMetadataSymbol);
    if (failed(image))
      return failure();
    for (auto [index, tensorIndex] : llvm::enumerate(tensors)) {
      const TensorRecord &tensor = graph.tensors[tensorIndex];
      uint32_t base = static_cast<uint32_t>(index) *
                      UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_BYTES;
      if (failed(writeAddress(
              *image, base + UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_NAME_OFFSET,
              onnxHostIoNameSymbol(output, index))) ||
          failed(writeWord(
              *image,
              base + UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_DATA_TYPE_OFFSET,
              static_cast<uint32_t>(
                  onnxHostDataType(tensor.type.getElementType())))) ||
          failed(writeWord(
              *image, base + UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_RANK_OFFSET,
              static_cast<uint32_t>(tensor.type.getRank()))) ||
          failed(
              tensor.type.getRank()
                  ? writeAddress(
                        *image,
                        base +
                            UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_DIMENSIONS_OFFSET,
                        onnxHostIoDimensionsSymbol(output, index))
                  : writeWord(
                        *image,
                        base +
                            UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_DIMENSIONS_OFFSET,
                        0)) ||
          failed(
              tensor.dynamic
                  ? writeAddress(
                        *image,
                        base +
                            UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_DIMENSION_BOUNDS_OFFSET,
                        onnxHostIoDimensionBoundsSymbol(output, index))
                  : writeWord(
                        *image,
                        base +
                            UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_DIMENSION_BOUNDS_OFFSET,
                        0)))
        return failure();
    }
    commitImage(*image);
    return success();
  };
  if (failed(initializeMetadata(false, graph.inputs)) ||
      failed(initializeMetadata(true, graph.outputs)))
    return failure();
  auto modelImage = openImage(onnxHostModelStorageSymbol);
  if (failed(modelImage) ||
      failed(writeAddress(*modelImage, UDONCL_UDON_ABI_ONNX_MODEL_NAME_OFFSET,
                          onnxHostModelNameSymbol)) ||
      failed(writeWord(*modelImage, UDONCL_UDON_ABI_ONNX_MODEL_GRAPH_ID_OFFSET,
                       0)) ||
      failed(writeAddress(*modelImage, UDONCL_UDON_ABI_ONNX_MODEL_INPUTS_OFFSET,
                          onnxHostInputMetadataSymbol)) ||
      failed(writeWord(*modelImage,
                       UDONCL_UDON_ABI_ONNX_MODEL_INPUT_COUNT_OFFSET,
                       static_cast<uint32_t>(graph.inputs.size()))) ||
      failed(writeAddress(*modelImage,
                          UDONCL_UDON_ABI_ONNX_MODEL_OUTPUTS_OFFSET,
                          onnxHostOutputMetadataSymbol)) ||
      failed(writeWord(*modelImage,
                       UDONCL_UDON_ABI_ONNX_MODEL_OUTPUT_COUNT_OFFSET,
                       static_cast<uint32_t>(graph.outputs.size()))))
    return failure();
  commitImage(*modelImage);

  Value model =
      onnxHostAddressOf(builder, location, onnxHostModelStorageSymbol);
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
