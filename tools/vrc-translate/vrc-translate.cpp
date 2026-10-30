#include "VRC/Conversion/LLVMToHighLevel/Import.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.h"
#include "VRC/Target/UdonAssembly/Emitter.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
namespace {
cl::opt<bool>
    importLLVM("import-llvm",
               cl::desc("Import LLVM IR and retain frontend metadata"));
cl::opt<bool>
    emitUdon("emit-udon-assembly",
             cl::desc("Emit textual Udon Assembly from lowered MLIR"));
cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input LLVM IR>"),
                                   cl::Required);
cl::opt<std::string> outputFilename("o", cl::desc("Output MLIR file"),
                                    cl::init("-"));
cl::opt<std::string>
    metadataFilename("metadata-output",
                     cl::desc("Output schema v1 frontend sidecar"));
} // namespace

int main(int argc, char **argv) {
  InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "udoncl IR translator\n");
  if (int(importLLVM) + int(emitUdon) != 1) {
    errs() << "specify exactly one of --import-llvm or --emit-udon-assembly\n";
    return 1;
  }
  if (emitUdon) {
    if (!metadataFilename.empty()) {
      errs() << "--metadata-output is only valid with --import-llvm\n";
      return 1;
    }
    mlir::DialectRegistry registry;
    registry.insert<vrc::udonasm::UdonAsmDialect>();
    mlir::MLIRContext context(registry);
    auto module =
        mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
    if (!module || mlir::failed(mlir::verify(*module)))
      return 1;
    std::error_code error;
    ToolOutputFile output(outputFilename, error, sys::fs::OF_Text);
    if (error) {
      errs() << error.message() << "\n";
      return 1;
    }
    if (mlir::failed(vrc::emitUdonAssembly(*module, output.os())))
      return 1;
    output.os().flush();
    if (output.os().has_error()) {
      errs() << "failed to write Udon Assembly\n";
      output.os().clear_error();
      return 1;
    }
    output.keep();
    return 0;
  }
  LLVMContext llvmContext;
  SMDiagnostic diagnostic;
  auto input = parseIRFile(inputFilename, diagnostic, llvmContext);
  if (!input) {
    diagnostic.print(argv[0], errs());
    return 1;
  }
  mlir::MLIRContext context;
  std::string sidecar;
  auto module = vrc::importLLVM(std::move(input), context,
                                metadataFilename.empty() ? nullptr : &sidecar);
  if (!module || mlir::failed(mlir::verify(*module)))
    return 1;
  std::error_code error;
  ToolOutputFile output(outputFilename, error, sys::fs::OF_Text);
  if (error) {
    errs() << error.message() << "\n";
    return 1;
  }
  std::unique_ptr<ToolOutputFile> metadata;
  if (!metadataFilename.empty()) {
    metadata = std::make_unique<ToolOutputFile>(metadataFilename, error,
                                                sys::fs::OF_Text);
    if (error) {
      errs() << error.message() << "\n";
      return 1;
    }
    metadata->os() << sidecar << "\n";
    metadata->os().flush();
    if (metadata->os().has_error()) {
      errs() << "failed to write frontend metadata\n";
      metadata->os().clear_error();
      return 1;
    }
  }
  module->print(output.os());
  output.os() << "\n";
  output.os().flush();
  if (output.os().has_error()) {
    errs() << "failed to write MLIR\n";
    output.os().clear_error();
    return 1;
  }
  if (metadata)
    metadata->keep();
  output.keep();
  return 0;
}
