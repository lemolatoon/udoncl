#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "VRC/Dialect/UdonHL/UdonHLEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "VRC/Dialect/UdonHL/UdonHLAttrs.cpp.inc"
#include "VRC/Dialect/UdonHL/UdonHLDialect.cpp.inc"
void vrc::udonhl::UdonHLDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "VRC/Dialect/UdonHL/UdonHLAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "VRC/Dialect/UdonHL/UdonHLOps.cpp.inc"
      >();
}
#define GET_OP_CLASSES
#include "VRC/Dialect/UdonHL/UdonHLOps.cpp.inc"

mlir::LogicalResult vrc::udonhl::CallOp::verify() {
  if (!(*this)->getParentOfType<mlir::func::FuncOp>() ||
      !mlir::isa<mlir::ModuleOp>(
          (*this)->getParentOfType<mlir::func::FuncOp>()->getParentOp()))
    return emitOpError("C call must belong to a Host func.func");
  if (!getCallee().size())
    return emitOpError("requires a nonempty callee");
  return mlir::success();
}
mlir::LogicalResult
vrc::udonhl::CallOp::verifySymbolUses(mlir::SymbolTableCollection &symbols) {
  auto callee = symbols.lookupNearestSymbolFrom<mlir::func::FuncOp>(
      *this, getCalleeAttr());
  if (!callee)
    return emitOpError("callee must resolve to func.func");
  auto type = callee.getFunctionType();
  if (getOperandTypes() != type.getInputs() ||
      getResultTypes() != type.getResults())
    return emitOpError("callee argument/result types do not match");
  auto effect = callee->getAttrOfType<mlir::StringAttr>("vrc.async_effect");
  if (effect && effect.getValue() == "may_suspend" &&
      getCallPolicy() != CallPolicy::MaySuspend)
    return emitOpError("may_suspend callee requires may_suspend call policy");
  if (getCallPolicy() == CallPolicy::MaySuspend) {
    auto caller = (*this)->getParentOfType<mlir::func::FuncOp>();
    auto callerEffect =
        caller ? caller->getAttrOfType<mlir::StringAttr>("vrc.async_effect")
               : mlir::StringAttr();
    if (!callerEffect || callerEffect.getValue() != "may_suspend")
      return emitOpError("may_suspend must propagate to the caller");
  }
  return mlir::success();
}
