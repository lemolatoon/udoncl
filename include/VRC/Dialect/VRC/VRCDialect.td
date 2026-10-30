#ifndef VRC_VRC_DIALECT
#define VRC_VRC_DIALECT
include "mlir/IR/OpBase.td"
def VRC_Dialect : Dialect {
  let name = "vrc";
  let useDefaultTypePrinterParser = 1;
  let cppNamespace = "::vrc::vrc";
  let summary = "VRC dialect; see PLAN.md for the full contract";
}
#endif
