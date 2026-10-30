#include "VRC/Conversion/LLVMToHighLevel/Import.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/DLTI/DLTI.h"
mlir::OwningOpRef<mlir::ModuleOp> vrc::importLLVM(
    std::unique_ptr<llvm::Module> module, mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::DLTIDialect>();
  mlir::registerLLVMDialectImport(registry);
  context.appendDialectRegistry(registry);
  return mlir::translateLLVMIRToModule(std::move(module), &context);
}
