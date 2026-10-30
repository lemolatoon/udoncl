#ifndef VRC_CONVERSION_LLVMTOHIGHLEVEL_IMPORT_H
#define VRC_CONVERSION_LLVMTOHIGHLEVEL_IMPORT_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/Module.h"
namespace vrc {
// Imports LLVM dialect and retains pre-import metadata as vrc.frontend.
// Optional JSON sidecar includes original LLVM IR for lossless debug recovery.
mlir::OwningOpRef<mlir::ModuleOp>
importLLVM(std::unique_ptr<llvm::Module> module, mlir::MLIRContext &context,
           std::string *sidecar = nullptr);
} // namespace vrc

#endif // VRC_CONVERSION_LLVMTOHIGHLEVEL_IMPORT_H
