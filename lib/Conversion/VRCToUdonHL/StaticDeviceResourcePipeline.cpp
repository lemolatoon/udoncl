#include "VRC/Conversion/StaticDeviceResourcePipeline.h"
#include "VRC/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace vrc {

void buildStaticDeviceResourceDiscoveryPipeline(OpPassManager &manager) {
  manager.addPass(createAttachDeviceResourceOperandContractsPass());
  // A constant-trip loop of allocations has one context per iteration.
  manager.addPass(createUnrollStaticResourceLoopsPass());
  // An unrolled loop over a constant table reads one entry per iteration.
  manager.addPass(createFoldConstantHostTablesPass());
  manager.addPass(createDiscoverStaticDeviceResourcesPass());
  manager.addPass(createPlanStaticDeviceResourceContextSpecializationPass());
  manager.addPass(createSpecializeStaticAllocationContextsPass());
  // Specialization clones allocation paths; discover their contexts again.
  manager.addPass(createDiscoverStaticDeviceResourcesPass());
  manager.addPass(createPlanStaticDeviceResourceContextSpecializationPass());
  manager.addPass(createInferStaticDeviceResourceRequirementsPass());
  manager.addPass(createPropagateDeviceResourceProvenancePass());
  // UDONCL_PUBLIC variables resolve to the allocation their global cell
  // holds.
  manager.addPass(createClassifyAnnotatedPublicDeviceResourcesPass());
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

void buildStaticUnityResourceReadinessPipeline(OpPassManager &manager) {
  manager.addPass(createMaterializeStaticLogicalBuffersPass());
  manager.addPass(createMaterializeStaticDispatchResourcesPass());
  manager.addPass(createMaterializeStaticTransferResourcesPass());
  manager.addPass(createPlanStaticPhysicalAllocationSpecializationPass());
  manager.addPass(createSpecializeStaticPhysicalAllocationPathsPass());
  manager.addPass(createMaterializeStaticPhysicalAllocationsPass());
  manager.addPass(createMaterializeStaticPhysicalResolvesPass());
  manager.addPass(createLowerCheckedAssignedBufferResolvesPass());
  manager.addPass(createMaterializeFixedPhysicalReleasesPass());
  manager.addPass(createEliminateDeadPhysicalAllocationFallbacksPass());
  manager.addPass(createAssessStaticUnityResourceLoweringPass());
}

void registerStaticUnityResourceReadinessPipeline() {
  PassPipelineRegistration<>(
      "vrc-static-unity-resource-readiness-pipeline",
      "Materialize the assigned Unity Textures of a reconciled Unity Host's "
      "Device resources and assess whether the static Unity resource "
      "lowering applies",
      buildStaticUnityResourceReadinessPipeline);
}

} // namespace vrc
