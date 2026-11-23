#ifndef VRC_CONVERSION_STATICDEVICERESOURCEPIPELINE_H
#define VRC_CONVERSION_STATICDEVICERESOURCEPIPELINE_H

namespace mlir {
class OpPassManager;
}

namespace vrc {

/// The static Device resource analysis of a recovered Unity Host, from the
/// Device resource operand contracts of the runtime entry points to the
/// public resource aliases of the assigned Unity Textures.
void buildStaticDeviceResourceDiscoveryPipeline(mlir::OpPassManager &manager);
void registerStaticDeviceResourceDiscoveryPipeline();

} // namespace vrc

#endif
