#ifndef VRC_UDONHL_TYPES
#define VRC_UDONHL_TYPES
include "VRC/Dialect/UdonHL/UdonHLDialect.td"
include "mlir/IR/AttrTypeBase.td"
def UdonHL_FrameType : TypeDef<UdonHL_Dialect, "Frame"> {
  let mnemonic = "frame";
}
def UdonHL_PtrType : TypeDef<UdonHL_Dialect, "Ptr"> {
  let mnemonic = "ptr";
  let summary = "32-bit logical byte address in one UdonCL address space";
  let parameters = (ins "uint32_t":$addressSpace);
  let assemblyFormat = "`<` $addressSpace `>`";
}
#endif
