#ifndef VRC_CONVERSION_BOUNDARYFRONTEND_FRONTEND_H
#define VRC_CONVERSION_BOUNDARYFRONTEND_FRONTEND_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
namespace mlir {
class OpPassManager;
}
namespace vrc {
// Imports both real LLVM modules and an explicit, versioned AOT ABI contract.
mlir::OwningOpRef<mlir::ModuleOp>
importCBoundary(llvm::StringRef hostFile, llvm::StringRef deviceFile,
                const llvm::json::Object &config, mlir::MLIRContext &context);
void buildCBoundaryRecoveryPipeline(mlir::OpPassManager &manager);
void registerCBoundaryRecoveryPipeline();
} // namespace vrc
#endif
