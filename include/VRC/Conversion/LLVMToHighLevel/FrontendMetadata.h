#ifndef VRC_CONVERSION_LLVMTOHIGHLEVEL_FRONTENDMETADATA_H
#define VRC_CONVERSION_LLVMTOHIGHLEVEL_FRONTENDMETADATA_H
#include "llvm/IR/Module.h"
#include "llvm/Support/JSON.h"

namespace vrc {
// Schema v1 captures semantic metadata before MLIR's LLVM importer drops it.
// The sidecar additionally stores the complete input IR, including debug info.
llvm::json::Object collectFrontendMetadata(const llvm::Module &module,
                                           llvm::StringRef sourceIR);
} // namespace vrc

#endif // VRC_CONVERSION_LLVMTOHIGHLEVEL_FRONTENDMETADATA_H
