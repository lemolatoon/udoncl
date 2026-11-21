#ifndef VRC_TARGET_HLSL_EMITTER_H
#define VRC_TARGET_HLSL_EMITTER_H
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "llvm/Support/raw_ostream.h"
namespace vrc {
mlir::LogicalResult emitGatherFragmentHLSL(::vrc::vrcgpu::KernelOp kernel,
                                           llvm::raw_ostream &out);
/// A gather kernel whose buffers hold several words per texel: each fragment
/// computes the words of one output texel together.
mlir::LogicalResult emitWordTexelFragmentHLSL(::vrc::vrcgpu::KernelOp kernel,
                                              llvm::raw_ostream &out);
mlir::LogicalResult emitFragmentHLSL(::vrc::vrcgpu::KernelOp kernel,
                                     llvm::raw_ostream &out);
mlir::LogicalResult emitShaderLab(::vrc::vrcgpu::KernelOp kernel,
                                  llvm::raw_ostream &out);
} // namespace vrc
#endif
