#include "VRC/Conversion/StaticDeviceResourcePipeline.h"
#include "VRC/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace vrc {

void buildStaticDeviceResourceDiscoveryPipeline(OpPassManager &manager) {
  manager.addPass(createAttachDeviceResourceOperandContractsPass());
  manager.addPass(createDiscoverStaticDeviceResourcesPass());
  manager.addPass(createPlanStaticDeviceResourceContextSpecializationPass());
  manager.addPass(createSpecializeStaticAllocationContextsPass());
  // Specialization clones allocation paths; discover their contexts again.
  manager.addPass(createDiscoverStaticDeviceResourcesPass());
  manager.addPass(createPlanStaticDeviceResourceContextSpecializationPass());
  manager.addPass(createInferStaticDeviceResourceRequirementsPass());
  manager.addPass(createPropagateDeviceResourceProvenancePass());
  manager.addPass(createResolveStaticTransferBindingsPass());
  manager.addPass(createClassifyStaticPublicDeviceResourcesPass());
  manager.addPass(createAnalyzeStaticDeviceResourcePersistencePass());
  manager.addPass(createBuildStaticDeviceResourceInventoryPass());
  manager.addPass(createAllocateUnityTexturesPass());
  manager.addPass(createResolveStaticPublicResourceAliasesPass());
}

void registerStaticDeviceResourceDiscoveryPipeline() {
  PassPipelineRegistration<>(
      "vrc-discover-static-device-resource-pipeline",
      "Discover the static Device resources of a recovered Unity Host, "
      "resolve their provenance and public bindings, and assign Unity "
      "Textures",
      buildStaticDeviceResourceDiscoveryPipeline);
}

} // namespace vrc
