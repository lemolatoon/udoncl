#ifndef VRC_CONVERSION_LLVMTOHIGHLEVEL_PASSES_H
#define VRC_CONVERSION_LLVMTOHIGHLEVEL_PASSES_H
#include "mlir/Pass/Pass.h"

namespace vrc {
#define GEN_PASS_DECL
#include "VRC/Conversion/LLVMToHighLevel/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "VRC/Conversion/LLVMToHighLevel/Passes.h.inc"

// Promotes locals, recovers proven memory/scalars, then structures CFG as SCF.
void registerHighLevelRecoveryPipeline();
} // namespace vrc

#endif // VRC_CONVERSION_LLVMTOHIGHLEVEL_PASSES_H
