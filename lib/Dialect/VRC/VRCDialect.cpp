#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#define GET_TYPEDEF_CLASSES
#include "VRC/Dialect/VRC/VRCDialect.cpp.inc"
#include "VRC/Dialect/VRC/VRCTypes.cpp.inc"
void vrc::vrc::VRCDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "VRC/Dialect/VRC/VRCTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "VRC/Dialect/VRC/VRCOps.cpp.inc"
      >();
}
#define GET_OP_CLASSES
#include "VRC/Dialect/VRC/VRCOps.cpp.inc"

mlir::LogicalResult vrc::vrc::DispatchOp::verify() {
  if (!(*this)->getParentOfType<mlir::func::FuncOp>() ||
      !mlir::isa<mlir::ModuleOp>(
          (*this)->getParentOfType<mlir::func::FuncOp>()->getParentOp()))
    return emitOpError("dispatch must belong to the Host runtime path");
  if (getKernelIdAttr().getInt() <= 0)
    return emitOpError("requires a positive kernel id");
  unsigned buffers = 0;
  for (auto value : getArgumentKinds()) {
    auto kind = mlir::dyn_cast<mlir::StringAttr>(value);
    if (!kind || (kind != "buffer" && kind != "f32" && kind != "i32"))
      return emitOpError("unsupported descriptor argument kind");
    buffers += kind == "buffer";
  }
  if (buffers != getResources().size())
    return emitOpError("ordered resource count must match buffer arguments");
  auto global = getGlobalSize(), local = getLocalSize(),
       offset = getGlobalOffset();
  if (global.size() != 3 || local.size() != 3 || offset.size() != 3)
    return emitOpError("NDRange must contain three dimensions");
  for (unsigned i = 0; i < 3; ++i)
    if (global[i] <= 0 || local[i] <= 0 || offset[i] < 0 ||
        global[i] % local[i] || int64_t(global[i]) + offset[i] > INT32_MAX)
      return emitOpError("invalid size, offset or local-size divisibility");
  return mlir::success();
}
