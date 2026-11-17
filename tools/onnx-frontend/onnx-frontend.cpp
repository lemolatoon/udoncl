#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerDialects.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>

static llvm::cl::opt<bool> emitBytecode("emit-bytecode",
                                        llvm::cl::desc("Emit MLIR bytecode"),
                                        llvm::cl::init(false));

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  mlir::registerAllPasses();
  vrc::registerSpecializeONNXInputsPass();
  vrc::registerLowerONNXToUdonHostPass();
  vrc::registerValidateONNXHostGraphPass();
  vrc::registerPlanONNXHostGraphPass();
  vrc::registerMaterializeONNXHostGlobalsPass();
  vrc::registerMaterializeONNXHostModelRegistryPass();
  vrc::registerFinalizeONNXHostPass();
  vrc::registerCleanupONNXHostSourcePass();
  vrc::registerEmitONNXHostMetadataPass();
  mlir::registerPass([] { return onnx_mlir::createShapeInferencePass(); });
  mlir::registerPass([] { return onnx_mlir::createConstPropONNXToONNXPass(); });
  mlir::registerPass(
      [] { return onnx_mlir::createSimplifyShapeRelatedOpsPass(); });
  mlir::registerPass([] { return onnx_mlir::createStandardFuncReturnPass(); });

  mlir::DialectRegistry registry = onnx_mlir::registerDialects({});
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::ONNXDialect, vrc::udonhl::UdonHLDialect>();

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::PassPipelineCLParser pipeline("", "Compiler passes to run");
  if (!llvm::cl::ParseCommandLineOptions(argc, argv,
                                         "internal UdonCL ONNX frontend\n"))
    return 1;
  if (!llvm::StringRef(onnx_mlir::inputFilename)
           .ends_with_insensitive(".onnx")) {
    llvm::errs() << "internal ONNX frontend requires an .onnx input\n";
    return 1;
  }

  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::ONNXDialect, mlir::func::FuncDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string error;
  onnx_mlir::ImportOptions importOptions;
  importOptions.externalDataDir =
      std::filesystem::absolute(onnx_mlir::inputFilename)
          .parent_path()
          .string();
  int importResult = onnx_mlir::ImportFrontendModelFile(
      onnx_mlir::inputFilename, context, module, &error, importOptions);
  if (importResult != 0 || !module) {
    llvm::errs() << "failed to import ONNX model " << onnx_mlir::inputFilename
                 << ": " << error << "\n";
    return importResult == 0 ? 1 : importResult;
  }

  auto setup = [&](mlir::PassManager &passManager) {
    return pipeline.addToPipeline(passManager, [&](const llvm::Twine &message) {
      mlir::emitError(mlir::UnknownLoc::get(&context)) << message;
      return mlir::failure();
    });
  };
  mlir::MlirOptMainConfig config;
  config.setPassPipelineSetupFn(setup)
      .verifyPasses(onnx_mlir::verify_passes)
      .allowUnregisteredDialects(onnx_mlir::allowUnregisteredDialects)
      .emitBytecode(emitBytecode);
  mlir::PassManager manager = mlir::PassManager::on<mlir::ModuleOp>(
      &context, mlir::OpPassManager::Nesting::Implicit);
  if (mlir::failed(config.setupPassPipeline(manager)) ||
      mlir::failed(mlir::applyPassManagerCLOptions(manager)) ||
      mlir::failed(manager.run(*module)))
    return 1;

  auto output = mlir::openOutputFile(onnx_mlir::outputBaseName, &error);
  if (!output) {
    llvm::errs() << error << "\n";
    return 1;
  }
  if (config.shouldEmitBytecode()) {
    if (mlir::failed(mlir::writeBytecodeToFile(*module, output->os())))
      return 1;
  } else {
    mlir::OpPrintingFlags flags;
    module->print(output->os(), flags);
    output->os() << "\n";
  }
  output->keep();
  return 0;
}
