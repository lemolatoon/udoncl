#ifndef VRC_ABI_MANIFEST_BOUNDARY_H
#define VRC_ABI_MANIFEST_BOUNDARY_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
namespace vrc {
// Validates both lowered sides, including descriptor and launch contracts.
mlir::LogicalResult verifyHostDeviceBoundary(mlir::ModuleOp module);
llvm::json::Object buildBoundaryManifest(mlir::ModuleOp module);
// Recomputes the fingerprint after artifact-specific public binding metadata,
// such as whole-buffer U# accessors, has been added to the manifest.
void refreshPublicABIFingerprint(llvm::json::Object &manifest);
// Checks the copied scalar payload and ordered GPU resources without issuing
// GPU work.
bool validateBoundaryRequest(const llvm::json::Object &manifest,
                             const llvm::json::Object &request,
                             llvm::raw_ostream &error);
} // namespace vrc
#endif
