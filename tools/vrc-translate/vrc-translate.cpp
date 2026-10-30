#include "VRC/Conversion/LLVMToHighLevel/Import.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"
int main(int argc, char **argv) {
  if (argc != 3 || llvm::StringRef(argv[1]) != "--import-llvm") {
    llvm::errs() << "usage: vrc-translate --import-llvm input.ll\n"
                    "Imports LLVM dialect only; high-level recovery is pending.\n";
    return 1;
  }
  llvm::LLVMContext llvmContext;
  llvm::SMDiagnostic diagnostic;
  auto input = llvm::parseIRFile(argv[2], diagnostic, llvmContext);
  if (!input) { diagnostic.print(argv[0], llvm::errs()); return 1; }
  mlir::MLIRContext context;
  auto module = vrc::importLLVM(std::move(input), context);
  if (!module || mlir::failed(mlir::verify(*module))) return 1;
  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
