#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/Module.h"
namespace vrc {
// Preserves the LLVM dialect. High-level recovery is not implemented yet.
mlir::OwningOpRef<mlir::ModuleOp> importLLVM(std::unique_ptr<llvm::Module> module,
                                           mlir::MLIRContext &context);
}
