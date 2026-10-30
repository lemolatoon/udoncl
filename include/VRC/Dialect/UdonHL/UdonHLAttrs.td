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
#endif
