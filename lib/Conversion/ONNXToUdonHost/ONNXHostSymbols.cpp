#include "ONNXHostSymbols.h"

#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace vrc {
void ONNXHostDeclarations::require(StringRef name, TypeRange arguments,
                                   TypeRange results) {
  FunctionType expected =
      FunctionType::get(module.getContext(), arguments, results);
  if (Operation *symbol = SymbolTable::lookupSymbolIn(module, name)) {
    auto function = dyn_cast<func::FuncOp>(symbol);
    if (!function) {
      symbol->emitError() << "ONNX Host runtime symbol " << name
                          << " is not a function";
      valid = false;
      return;
    }
    if (!function.isExternal()) {
      function.emitError() << "ONNX Host runtime declaration " << name
                           << " conflicts with a definition";
      valid = false;
      return;
    }
    if (!function.isPrivate()) {
      function.emitError() << "ONNX Host runtime declaration " << name
                           << " must have private visibility";
      valid = false;
      return;
    }
    if (function.getFunctionType() != expected) {
      function.emitError() << "ONNX Host runtime declaration " << name
                           << " has type " << function.getFunctionType()
                           << "; expected " << expected;
      valid = false;
    }
    return;
  }

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  auto function =
      func::FuncOp::create(builder, module.getLoc(), name, expected);
  function.setPrivate();
}

FailureOr<func::FuncOp> ONNXHostDeclarations::reservePublicDefinition(
    StringRef name, TypeRange arguments, TypeRange results) {
  FunctionType expected =
      FunctionType::get(module.getContext(), arguments, results);
  if (Operation *symbol = SymbolTable::lookupSymbolIn(module, name)) {
    auto function = dyn_cast<func::FuncOp>(symbol);
    if (!function) {
      symbol->emitError() << "ONNX Host generated symbol " << name
                          << " is not a function";
      valid = false;
      return failure();
    }
    if (!function.isExternal()) {
      function.emitError() << "ONNX Host generated function " << name
                           << " conflicts with a definition";
      valid = false;
      return failure();
    }
    if (!function.isPrivate()) {
      function.emitError() << "ONNX Host generated function " << name
                           << " declaration must have private visibility";
      valid = false;
      return failure();
    }
    if (function.getFunctionType() != expected) {
      function.emitError() << "ONNX Host generated function " << name
                           << " has type " << function.getFunctionType()
                           << "; expected " << expected;
      valid = false;
      return failure();
    }
    return function;
  }

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  auto function =
      func::FuncOp::create(builder, module.getLoc(), name, expected);
  function.setPrivate();
  return function;
}

LogicalResult ONNXHostDeclarations::result() const { return success(valid); }
} // namespace vrc
