#include "VRC/Conversion/HostStoragePipeline.h"
#include "VRC/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace vrc {

void buildUdonHostStorageFinalizationPipeline(OpPassManager &manager) {
  manager.addPass(createMaterializeFixedPhysicalReleasesPass());
  // Exposing backend literals inserts operations next to their users.
  // Persistent async storage records positional value identities, so the
  // literals must be exposed before those identities are taken.
  manager.addPass(createExposeHostBackendLiteralUsesPass());
  manager.addPass(createDiscoverAsyncPersistentStoragePass());
  manager.addPass(createMaterializeAsyncPersistentStoragePass());
  manager.addPass(createDiscoverHostRematerializationPass());
  manager.addPass(createFitHostStorageCapacityPass());
  manager.addPass(createVerifyHostStorageCapacityPass());
  manager.addPass(createLowerUdonHLToUdonAsmPass());
}

void registerUdonHostStorageFinalizationPipeline() {
  PassPipelineRegistration<>(
      "vrc-finalize-udon-host-storage",
      "Finalize fixed resources, fit Host storage, verify capacity, and lower "
      "to Udon Assembly",
      buildUdonHostStorageFinalizationPipeline);
}

} // namespace vrc
