#ifndef VRC_GPU_TYPES
#define VRC_GPU_TYPES
include "VRC/Dialect/VRCGPU/VRCGPUDialect.td"
include "mlir/IR/AttrTypeBase.td"
def VRCGPU_VersionType : TypeDef<VRCGPU_Dialect, "Version"> {
  let mnemonic = "version";
  let parameters = (ins "uint32_t":$bufferId);
  let assemblyFormat = "`<` $bufferId `>`";
}
#endif
