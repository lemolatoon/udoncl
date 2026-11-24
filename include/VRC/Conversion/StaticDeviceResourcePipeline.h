#ifndef VRC_CONVERSION_STATICDEVICERESOURCEPIPELINE_H
#define VRC_CONVERSION_STATICDEVICERESOURCEPIPELINE_H

namespace mlir {
class OpPassManager;
}

namespace vrc {

/// The allocation contexts of a recovered Unity Host and their inferred
/// requirements: the discovery pipeline through
/// vrc-infer-static-device-resource-requirements.
void buildStaticDeviceResourceRequirementsPipeline(
    mlir::OpPassManager &manager);

/// The requirements pipeline, then vrc-plan-host-device-handles: one OpenCL
/// buffer handle per allocation context (vrc.host_device_handle_plan).
void buildHostDeviceHandlePlanPipeline(mlir::OpPassManager &manager);
void registerHostDeviceHandlePlanPipeline();

/// The static Device resource analysis of a recovered Unity Host, from the
/// Device resource operand contracts of the runtime entry points to the
/// public resource aliases of the assigned Unity Textures.
void buildStaticDeviceResourceDiscoveryPipeline(mlir::OpPassManager &manager);
void registerStaticDeviceResourceDiscoveryPipeline();

/// The static Unity resource lowering of a Host whose Unity buffers are
/// reconciled and laid out: materializes the assigned Unity Textures of its
/// logical buffers, dispatches and physical allocations, and assesses whether
/// every Device resource then resolves to one.
void buildStaticUnityResourceReadinessPipeline(mlir::OpPassManager &manager);
void registerStaticUnityResourceReadinessPipeline();

} // namespace vrc

#endif
