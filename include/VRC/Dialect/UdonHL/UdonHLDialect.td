#ifndef VRC_UDONHL_DIALECT
#define VRC_UDONHL_DIALECT
include "mlir/IR/OpBase.td"
def UdonHL_Dialect : Dialect {
  let name = "udonhl";
  let cppNamespace = "::vrc::udonhl";
  let summary = "UdonHL dialect; see PLAN.md for the full contract";
}
#endif
