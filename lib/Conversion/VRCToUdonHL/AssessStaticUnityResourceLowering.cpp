#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ASSESSSTATICUNITYRESOURCELOWERINGPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral layoutAttribute = "vrc.static_unity_buffer_layout_plan";
constexpr StringLiteral kernelAttribute = "vrc.static_kernel_bindings";
constexpr StringLiteral materializationAttribute =
    "vrc.static_dispatch_resource_materialization";
constexpr StringLiteral transferAttribute = "vrc.static_transfer_bindings";
constexpr StringLiteral readinessAttribute =
    "vrc.static_unity_resource_lowering_readiness";
constexpr StringLiteral snapshotAttribute =
    "vrc.static_kernel_binding_snapshot";

bool schemaV1(DictionaryAttr value) {
  auto version = value ? value.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

IntegerAttr integer(DictionaryAttr value, StringRef name) {
  return value ? value.getAs<IntegerAttr>(name) : nullptr;
}

bool assignedOrStatic(Value value) {
  while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1)
      return false;
    value = cast.getInputs().front();
  }
  return value.getDefiningOp<::vrc::vrc::AssignedBufferOp>() ||
         value.getDefiningOp<::vrc::vrc::StaticBufferOp>();
}

bool transferCall(func::CallOp call) {
  return call.getCallee() == "clEnqueueWriteBuffer" ||
         call.getCallee() == "clEnqueueReadBuffer" ||
         call.getCallee() == "clEnqueueCopyBuffer";
}

struct AssessStaticUnityResourceLoweringPass
    : impl::AssessStaticUnityResourceLoweringPassBase<
          AssessStaticUnityResourceLoweringPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(readinessAttribute);

    auto layout = module->getAttrOfType<DictionaryAttr>(layoutAttribute);
    auto layoutReady = layout ? layout.getAs<BoolAttr>("ready") : nullptr;
    auto layoutBlockers = integer(layout, "blocker_count");
    if (!schemaV1(layout) || !layoutReady || !layoutBlockers ||
        layoutBlockers.getInt() < 0) {
      module.emitError(
          "static Unity resource readiness requires layout schema v1");
      signalPassFailure();
      return;
    }

    int32_t dispatchCount = 0;
    int32_t snapshotCount = 0;
    int32_t assignedCount = 0;
    int32_t unresolvedDispatchOperands = 0;
    module.walk([&](::vrc::vrc::DispatchOp dispatch) {
      ++dispatchCount;
      snapshotCount += dispatch->hasAttr(snapshotAttribute);
      for (Value resource : dispatch.getResources()) {
        if (resource.getDefiningOp<::vrc::vrc::AssignedBufferOp>())
          ++assignedCount;
        if (!assignedOrStatic(resource))
          ++unresolvedDispatchOperands;
      }
    });

    auto kernels = module->getAttrOfType<DictionaryAttr>(kernelAttribute);
    auto materialization =
        module->getAttrOfType<DictionaryAttr>(materializationAttribute);
    auto kernelCandidates = integer(kernels, "candidate_count");
    auto kernelSnapshots = integer(kernels, "snapshot_count");
    auto kernelUnresolved = integer(kernels, "unresolved_count");
    auto materializedCandidates =
        integer(materialization, "candidate_dispatch_count");
    auto materializedSnapshots =
        integer(materialization, "snapshotted_dispatch_count");
    auto materializedOperands =
        integer(materialization, "materialized_operand_count");
    auto materializedUnresolvedDispatches =
        integer(materialization, "unresolved_dispatch_count");
    auto materializedUnresolvedOperands =
        integer(materialization, "unresolved_operand_count");
    if (dispatchCount > 0 &&
        (!schemaV1(kernels) || !schemaV1(materialization) ||
         !kernelCandidates || !kernelSnapshots || !kernelUnresolved ||
         !materializedCandidates || !materializedSnapshots ||
         !materializedOperands || !materializedUnresolvedDispatches ||
         !materializedUnresolvedOperands ||
         kernelCandidates.getInt() != dispatchCount ||
         kernelSnapshots.getInt() != snapshotCount ||
         materializedCandidates.getInt() != dispatchCount ||
         materializedSnapshots.getInt() != snapshotCount ||
         materializedOperands.getInt() != assignedCount)) {
      module.emitError(
          "static dispatch readiness metadata does not match the IR");
      signalPassFailure();
      return;
    }
    bool dispatchReady =
        dispatchCount == 0 || (kernelUnresolved.getInt() == 0 &&
                               materializedUnresolvedDispatches.getInt() == 0 &&
                               materializedUnresolvedOperands.getInt() == 0 &&
                               unresolvedDispatchOperands == 0);

    int32_t transferCalls = 0;
    int32_t dynamicResolveCount = 0;
    module.walk([&](func::CallOp call) {
      transferCalls += transferCall(call);
      dynamicResolveCount += call.getCallee() == "udonclResolvePhysicalBuffer";
    });
    auto transfers = module->getAttrOfType<DictionaryAttr>(transferAttribute);
    auto transferCandidates = integer(transfers, "candidate_count");
    auto transferResolved = integer(transfers, "resolved_count");
    auto transferUnresolved = integer(transfers, "unresolved_count");
    auto transferSpecialization =
        integer(transfers, "requires_context_specialization_count");
    if (transferCalls > 0 &&
        (!schemaV1(transfers) || !transferCandidates || !transferResolved ||
         !transferUnresolved || !transferSpecialization ||
         transferCandidates.getInt() != transferCalls)) {
      module.emitError(
          "static transfer readiness metadata does not match the IR");
      signalPassFailure();
      return;
    }

    int32_t uploadCount = 0;
    int32_t readbackCount = 0;
    int32_t copyCount = 0;
    int32_t unresolvedTypedTransferOperands = 0;
    module.walk([&](::vrc::vrc::BufferUploadOp operation) {
      ++uploadCount;
      unresolvedTypedTransferOperands +=
          !assignedOrStatic(operation.getDestination());
    });
    module.walk([&](::vrc::vrc::BufferReadbackOp operation) {
      ++readbackCount;
      unresolvedTypedTransferOperands +=
          !assignedOrStatic(operation.getSource());
    });
    module.walk([&](::vrc::vrc::BufferCopyOp operation) {
      ++copyCount;
      unresolvedTypedTransferOperands +=
          !assignedOrStatic(operation.getSource());
      unresolvedTypedTransferOperands +=
          !assignedOrStatic(operation.getDestination());
    });
    bool transferReady =
        transferCalls == 0 && unresolvedTypedTransferOperands == 0;

    int32_t dynamicAllocationCount = 0;
    int32_t dynamicReleaseCount = 0;
    module.walk([&](::vrc::vrc::BufferAllocOp) { ++dynamicAllocationCount; });
    module.walk([&](::vrc::vrc::BufferReleaseOp) { ++dynamicReleaseCount; });

    bool resourceReady =
        layoutReady.getValue() && dispatchReady && transferReady;
    bool poolReady = resourceReady && dynamicAllocationCount == 0 &&
                     dynamicReleaseCount == 0 && dynamicResolveCount == 0;
    module->setAttr(
        readinessAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("assigned_dispatch_operand_count",
                                 builder.getI32IntegerAttr(assignedCount)),
            builder.getNamedAttr("dispatch_candidate_count",
                                 builder.getI32IntegerAttr(dispatchCount)),
            builder.getNamedAttr("dispatch_ready",
                                 builder.getBoolAttr(dispatchReady)),
            builder.getNamedAttr("dispatch_snapshot_count",
                                 builder.getI32IntegerAttr(snapshotCount)),
            builder.getNamedAttr(
                "dynamic_allocation_count",
                builder.getI32IntegerAttr(dynamicAllocationCount)),
            builder.getNamedAttr(
                "dynamic_release_count",
                builder.getI32IntegerAttr(dynamicReleaseCount)),
            builder.getNamedAttr(
                "dynamic_resolve_count",
                builder.getI32IntegerAttr(dynamicResolveCount)),
            builder.getNamedAttr("layout_blocker_count", layoutBlockers),
            builder.getNamedAttr("layout_ready", layoutReady),
            builder.getNamedAttr("pool_replacement_ready",
                                 builder.getBoolAttr(poolReady)),
            builder.getNamedAttr("resource_operands_ready",
                                 builder.getBoolAttr(resourceReady)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("transfer_call_count",
                                 builder.getI32IntegerAttr(transferCalls)),
            builder.getNamedAttr("transfer_ready",
                                 builder.getBoolAttr(transferReady)),
            builder.getNamedAttr("typed_copy_count",
                                 builder.getI32IntegerAttr(copyCount)),
            builder.getNamedAttr("typed_readback_count",
                                 builder.getI32IntegerAttr(readbackCount)),
            builder.getNamedAttr("typed_upload_count",
                                 builder.getI32IntegerAttr(uploadCount)),
            builder.getNamedAttr(
                "unresolved_dispatch_operand_count",
                builder.getI32IntegerAttr(unresolvedDispatchOperands)),
            builder.getNamedAttr(
                "unresolved_typed_transfer_operand_count",
                builder.getI32IntegerAttr(unresolvedTypedTransferOperands)),
        }));
  }
};

} // namespace
} // namespace vrc
