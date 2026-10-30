#ifndef VRC_CONVERSION_PASSES_H
#define VRC_CONVERSION_PASSES_H
#include "mlir/Pass/Pass.h"
namespace vrc {
#define GEN_PASS_DECL
#include "VRC/Conversion/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "VRC/Conversion/Passes.h.inc"
} // namespace vrc
#endif
