#ifndef VRC_CONVERSION_HOSTSTORAGEPIPELINE_H
#define VRC_CONVERSION_HOSTSTORAGEPIPELINE_H

namespace mlir {
class OpPassManager;
}

namespace vrc {

void buildUdonHostStorageFinalizationPipeline(mlir::OpPassManager &manager);
void registerUdonHostStorageFinalizationPipeline();

} // namespace vrc

#endif
