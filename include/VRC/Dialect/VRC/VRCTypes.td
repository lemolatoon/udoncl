#ifndef VRC_TYPES
#define VRC_TYPES
include "VRC/Dialect/VRC/VRCDialect.td"
include "mlir/IR/AttrTypeBase.td"
def VRC_BufferType : TypeDef<VRC_Dialect, "Buffer"> { let mnemonic = "buffer"; }
def VRC_EventType : TypeDef<VRC_Dialect, "Event"> { let mnemonic = "event"; }
def VRC_DispatchDescType : TypeDef<VRC_Dialect, "DispatchDesc"> { let mnemonic = "dispatch_desc"; }
#endif
