#ifndef VRC_BOUNDARY_PASSES
#define VRC_BOUNDARY_PASSES
include "mlir/Pass/PassBase.td"
def LowerHostToUdonHLPass : Pass<"vrc-lower-host-to-udonhl", "mlir::ModuleOp"> {
  let summary = "Validate a small host ABI and lower direct C calls to udonhl.call";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "::vrc::vrc::VRCDialect"];
}
def SubstituteRuntimeIntrinsicsPass : Pass<"vrc-substitute-runtime-intrinsics", "mlir::ModuleOp"> {
  let summary = "Replace typed __vrc_* calls after UdonHL construction; preserve C runtime ownership";
  let dependentDialects = ["::vrc::vrc::VRCDialect"];
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
#endif
