#ifndef VRC_VRCGPU_DIALECT
#define VRC_VRCGPU_DIALECT
include "mlir/IR/OpBase.td"
def VRCGPU_Dialect : Dialect {
  let name = "vrcgpu";
  let useDefaultTypePrinterParser = 1;
  let cppNamespace = "::vrc::vrcgpu";
  let summary = "VRCGPU dialect; see PLAN.md for the full contract";
}
#endif
