#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTSYMBOLS_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTSYMBOLS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace vrc {
class ONNXHostDeclarations {
public:
  explicit ONNXHostDeclarations(mlir::ModuleOp module) : module(module) {}

  void require(llvm::StringRef name, mlir::TypeRange arguments,
               mlir::TypeRange results);
  mlir::LogicalResult result() const;

private:
  mlir::ModuleOp module;
  bool valid = true;
};
} // namespace vrc

#endif
