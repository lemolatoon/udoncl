#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTIRBUILDER_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTIRBUILDER_H

#include "VRC/Dialect/UdonHL/UdonHLDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

namespace vrc {
inline mlir::Value onnxHostAddressOf(mlir::OpBuilder &builder,
                                     mlir::Location location,
                                     llvm::StringRef symbol) {
  mlir::OperationState state(location,
                             ::vrc::udonhl::AddressOfOp::getOperationName());
  state.addTypes(::vrc::udonhl::PtrType::get(builder.getContext(), 0));
  state.addAttribute(
      "global", mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol));
  return builder.create(state)->getResult(0);
}

inline mlir::Value onnxHostNullPointer(mlir::OpBuilder &builder,
                                       mlir::Location location) {
  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, location, 0, 32);
  mlir::OperationState state(location,
                             ::vrc::udonhl::IntToPtrOp::getOperationName());
  state.addOperands(zero);
  state.addTypes(::vrc::udonhl::PtrType::get(builder.getContext(), 0));
  return builder.create(state)->getResult(0);
}

inline mlir::Value onnxHostPointerToInt(mlir::OpBuilder &builder,
                                        mlir::Location location,
                                        mlir::Value pointer) {
  mlir::OperationState state(location,
                             ::vrc::udonhl::PtrToIntOp::getOperationName());
  state.addOperands(pointer);
  state.addTypes(builder.getI32Type());
  return builder.create(state)->getResult(0);
}

inline mlir::Value onnxHostPointerAdd(mlir::OpBuilder &builder,
                                      mlir::Location location,
                                      mlir::Value pointer, mlir::Value offset) {
  mlir::OperationState state(location,
                             ::vrc::udonhl::PtrAddOp::getOperationName());
  state.addOperands({pointer, offset});
  state.addTypes(pointer.getType());
  return builder.create(state)->getResult(0);
}

inline void onnxHostStore(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Value value, mlir::Value pointer) {
  mlir::OperationState state(location,
                             ::vrc::udonhl::StoreOp::getOperationName());
  state.addOperands({value, pointer});
  state.addAttribute("alignment", builder.getI64IntegerAttr(4));
  builder.create(state);
}
} // namespace vrc

#endif
