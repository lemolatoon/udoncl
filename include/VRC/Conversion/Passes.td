#ifndef VRC_BOUNDARY_PASSES
#define VRC_BOUNDARY_PASSES
include "mlir/Pass/PassBase.td"
def LowerHostToUdonHLPass : Pass<"vrc-lower-host-to-udonhl", "mlir::ModuleOp"> {
  let summary = "Lower recovered structured Host control flow and direct C calls to UdonHL";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "::vrc::vrc::VRCDialect", "mlir::cf::ControlFlowDialect", "mlir::arith::ArithDialect"];
}
def LowerONNXToUdonHostPass : Pass<"udoncl-lower-onnx-to-host", "mlir::ModuleOp"> {
  let summary = "Lower typed ONNX dialect operations directly to Udon library calls";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
def SubstituteRuntimeIntrinsicsPass : Pass<"vrc-substitute-runtime-intrinsics", "mlir::ModuleOp"> {
  let summary = "Replace typed __vrc_* calls after UdonHL construction; preserve C runtime ownership";
  let dependentDialects = ["::vrc::vrc::VRCDialect"];
}
def LowerOpenCLAOTIntrinsicsPass : Pass<"vrc-lower-opencl-aot-intrinsics", "mlir::ModuleOp"> {
  let summary = "Recover the linked OpenCL AOT runtime backend ABI as typed VRC operations";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect", "mlir::arith::ArithDialect"];
}
def LowerAwaitToContinuationsPass : Pass<"vrc-lower-await-to-continuations", "mlir::ModuleOp"> {
  let summary = "Split a single typed await into explicit UdonHL frame, suspend and resume blocks";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "::vrc::vrc::VRCDialect"];
}
def LowerDeviceSingleInvocationPass : Pass<"vrc-lower-device-single-invocation", "mlir::ModuleOp"> {
  let summary = "Lower a proven 1-invocation scalar-output GPU kernel to fragment-safe vrcgpu";
  let dependentDialects = ["::vrc::vrcgpu::VRCGPUDialect", "mlir::arith::ArithDialect"];
}
def VerifyHostDeviceBoundaryPass : Pass<"vrc-verify-host-device", "mlir::ModuleOp"> {
  let summary = "Verify kernel ids, descriptors, resource order and launch ABI across host/device";
}
def RaiseLLVMBoundaryDevicePass : Pass<"vrc-raise-llvm-boundary-device", "mlir::ModuleOp"> {
  let summary = "Recover the selected single-invocation OpenCL output pointer as a bounded memref";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::arith::ArithDialect", "mlir::memref::MemRefDialect"];
}
def LinkCBoundaryPass : Pass<"vrc-link-c-boundary", "mlir::ModuleOp"> {
  let summary = "Recover explicit uint32 Host handle roles and link recovered C/OpenCL modules";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::gpu::GPUDialect"];
}
def LowerHostToNativeTestPass : Pass<"vrc-lower-host-to-native-test", "mlir::ModuleOp"> {
  let summary = "Lower the verified UdonHL Host boundary to native func calls for testing only";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::arith::ArithDialect"];
}
def LowerUdonHLToUdonAsmPass : Pass<"vrc-lower-udonhl-to-udonasm", "mlir::ModuleOp"> {
  let summary = "Allocate typed Udon heap slots and lower scalar Host SSA to the nine real VM opcodes";
  let dependentDialects = ["::vrc::udonasm::UdonAsmDialect", "mlir::memref::MemRefDialect"];
}
def RaiseNativeHostCallsPass : Pass<"vrc-raise-native-host-calls", "mlir::ModuleOp"> {
  let summary = "Expose native C calls as UdonHL while retaining native LLVM memory and ABI semantics";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect"];
}
def LowerNativeHostCallsPass : Pass<"vrc-lower-native-host-calls", "mlir::ModuleOp"> {
  let summary = "Restore native LLVM function/call ABI from the native-only UdonHL adapter";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}
def PrepareNativeDevicePass : Pass<"vrc-prepare-native-device", "mlir::ModuleOp"> {
  let summary = "Wrap the single recovered Device kernel for a native-only shader bundle";
  let dependentDialects = ["mlir::gpu::GPUDialect"];
}
def RaiseGatherDevicePass : Pass<"vrc-raise-gather-device", "mlir::ModuleOp"> {
  let summary = "Prove fixed global-buffer gather bounds, loop bounds and exclusive scalar output ownership";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::memref::MemRefDialect", "mlir::arith::ArithDialect", "mlir::math::MathDialect", "mlir::gpu::GPUDialect"];
}
def LowerGatherDevicePass : Pass<"vrc-lower-gather-device", "mlir::ModuleOp"> {
  let summary = "Lower proven gather kernels with scalar loops to fragment-safe resources and passes";
  let dependentDialects = ["::vrc::vrcgpu::VRCGPUDialect", "mlir::arith::ArithDialect"];
}
#endif
