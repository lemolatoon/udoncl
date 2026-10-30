#ifndef VRC_UDONASM_DIALECT
#define VRC_UDONASM_DIALECT
include "mlir/IR/OpBase.td"
def UdonAsm_Dialect : Dialect {
  let name = "udonasm";
  let cppNamespace = "::vrc::udonasm";
  let summary = "UdonAsm dialect; see PLAN.md for the full contract";
}
#endif
