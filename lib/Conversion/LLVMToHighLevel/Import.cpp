#include "VRC/Conversion/LLVMToHighLevel/Import.h"
#include "VRC/Conversion/LLVMToHighLevel/FrontendMetadata.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/Support/FormatVariadic.h"
mlir::OwningOpRef<mlir::ModuleOp>
vrc::importLLVM(std::unique_ptr<llvm::Module> module,
                mlir::MLIRContext &context, std::string *sidecar) {
  std::string source;
  llvm::raw_string_ostream stream(source);
  module->print(stream, nullptr);
  auto metadata = vrc::collectFrontendMetadata(*module, source);
  std::string summary =
      llvm::formatv("{0}", llvm::json::Value(std::move(metadata))).str();
  if (sidecar) {
    auto document = llvm::json::parse(summary);
    (*document->getAsObject())["source_llvm_ir"] = source;
    *sidecar = llvm::formatv("{0:2}", *document).str();
  }
  mlir::DialectRegistry registry;
  registry.insert<mlir::DLTIDialect>();
  mlir::registerLLVMDialectImport(registry);
  context.appendDialectRegistry(registry);
  auto imported = mlir::translateLLVMIRToModule(std::move(module), &context);
  if (imported)
    (*imported)->setAttr("vrc.frontend",
                         mlir::StringAttr::get(&context, summary));
  return imported;
}
