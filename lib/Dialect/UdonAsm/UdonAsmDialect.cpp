#include "llvm/ADT/StringExtras.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.cpp.inc"
void vrc::udonasm::UdonAsmDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "VRC/Dialect/UdonAsm/UdonAsmOps.cpp.inc"
  >();
}
#define GET_OP_CLASSES
#include "VRC/Dialect/UdonAsm/UdonAsmOps.cpp.inc"

namespace vrc::udonasm {
static mlir::LogicalResult symbol(mlir::Operation *op, llvm::StringRef value) {
  if (value.empty() || !(llvm::isAlpha(value.front()) || value.front() == '_') ||
      !llvm::all_of(value, [](char c) { return llvm::isAlnum(c) || c == '_'; }))
    return op->emitOpError("requires a nonempty assembler identifier");
  return mlir::success();
}
mlir::LogicalResult PushOp::verify() { return symbol(*this, getSlot()); }
mlir::LogicalResult JumpIndirectOp::verify() { return symbol(*this, getSlot()); }
mlir::LogicalResult ExternOp::verify() {
  if (getSignature().empty() || getSignature().contains('\n') ||
      getSignature().contains('\r'))
    return emitOpError("requires a nonempty, single-line extern signature");
  return mlir::success();
}
static mlir::LogicalResult address(mlir::Operation *op, int64_t value, bool sentinel) {
  if (value < 0 || value > 0xffffffffLL || (value & 3) ||
      (!sentinel && value == 0xfffffffcLL))
    return op->emitOpError("requires an aligned u32 code address; sentinel only for jump");
  return mlir::success();
}
mlir::LogicalResult JumpOp::verify() { return address(*this, getTarget(), true); }
mlir::LogicalResult JumpIfFalseOp::verify() { return address(*this, getTarget(), false); }
mlir::LogicalResult AnnotationOp::verify() {
  if (getParameter() < 0 || getParameter() > 0xffffffffLL)
    return emitOpError("requires a u32 annotation parameter");
  return mlir::success();
}
} // namespace vrc::udonasm
