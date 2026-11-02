#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

namespace {
mlir::LogicalResult verifyHostRuntimeOperation(mlir::Operation *operation) {
  auto function = operation->getParentOfType<mlir::func::FuncOp>();
  if (!function || !mlir::isa<mlir::ModuleOp>(function->getParentOp()))
    return operation->emitOpError("must belong to the Host runtime path");
  return mlir::success();
}

std::optional<int64_t> constantI32(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>();
  if (!constant || !constant.getType().isInteger(32))
    return std::nullopt;
  return constant.value();
}
} // namespace

mlir::LogicalResult vrc::vrc::BufferAllocOp::verify() {
  if (mlir::failed(verifyHostRuntimeOperation(*this)))
    return mlir::failure();
  if (getFormat() != "rgba8_unorm")
    return emitOpError("initial Device buffers require rgba8_unorm format");
  if (getUsage() != "storage")
    return emitOpError("initial Device buffers require storage usage");
  if (auto size = constantI32(getByteSize()); size && *size <= 0)
    return emitOpError("constant byte size must be positive");
  return mlir::success();
}

mlir::LogicalResult vrc::vrc::BufferCopyOp::verify() {
  if (mlir::failed(verifyHostRuntimeOperation(*this)))
    return mlir::failure();
  for (auto [name, value] :
       {std::pair{"source offset", getSourceOffset()},
        std::pair{"destination offset", getDestinationOffset()}})
    if (auto constant = constantI32(value); constant && *constant < 0)
      return emitOpError() << "constant " << name << " must be non-negative";
  if (auto size = constantI32(getByteSize()); size && *size <= 0)
    return emitOpError("constant byte size must be positive");
  return mlir::success();
}

mlir::LogicalResult vrc::vrc::DispatchOp::verify() {
  if (mlir::failed(verifyHostRuntimeOperation(*this)))
    return mlir::failure();
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
