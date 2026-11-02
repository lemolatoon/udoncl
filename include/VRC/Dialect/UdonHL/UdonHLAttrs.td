#ifndef VRC_UDONHL_ATTRS
#define VRC_UDONHL_ATTRS
include "VRC/Dialect/UdonHL/UdonHLDialect.td"
include "mlir/IR/EnumAttr.td"
def UdonHL_CallPolicy : I32EnumAttr<"CallPolicy", "C call/continuation policy", [
  I32EnumAttrCase<"Normal", 0, "normal">,
  I32EnumAttrCase<"Recursive", 1, "recursive">,
  I32EnumAttrCase<"MaySuspend", 2, "may_suspend">
]> {
  let cppNamespace = "::vrc::udonhl";
  let genSpecializedAttr = 0;
}
def UdonHL_CallPolicyAttr : EnumAttr<UdonHL_Dialect, UdonHL_CallPolicy, "call_policy">;
def UdonHL_SuspendReason : I32EnumAttr<"SuspendReason", "continuation suspension reason", [
  I32EnumAttrCase<"Gpu", 0, "gpu">,
  I32EnumAttrCase<"Budget", 1, "budget">,
  I32EnumAttrCase<"Service", 2, "service">
]> {
  let cppNamespace = "::vrc::udonhl";
  let genSpecializedAttr = 0;
}
def UdonHL_SuspendReasonAttr : EnumAttr<UdonHL_Dialect, UdonHL_SuspendReason, "suspend_reason">;
#endif
