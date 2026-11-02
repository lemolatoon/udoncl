#ifndef VRC_UDONHL_TYPES
#define VRC_UDONHL_TYPES
include "VRC/Dialect/UdonHL/UdonHLDialect.td"
include "mlir/IR/AttrTypeBase.td"
def UdonHL_FrameType : TypeDef<UdonHL_Dialect, "Frame"> {
  let mnemonic = "frame";
}
#endif
