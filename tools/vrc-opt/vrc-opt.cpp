#include "VRC/Conversion/BoundaryFrontend/Frontend.h"
#include "VRC/Conversion/HostStoragePipeline.h"
#include "VRC/Conversion/LLVMToHighLevel/Passes.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Conversion/StaticDeviceResourcePipeline.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Threading.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

static llvm::cl::opt<bool> emitBytecode("emit-bytecode",
                                        llvm::cl::desc("Emit MLIR bytecode"),
                                        llvm::cl::init(false));

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  mlir::registerAllPasses();
  vrc::registerCBoundaryRecoveryPipeline();
  vrc::registerVRCRecoveryPasses();
  vrc::registerHighLevelRecoveryPipeline();
  vrc::registerUdonHostStorageFinalizationPipeline();
  vrc::registerStaticDeviceResourceDiscoveryPipeline();
  vrc::registerHostDeviceHandlePlanPipeline();
  vrc::registerStaticUnityResourceReadinessPipeline();
  vrc::registerVRCBoundaryPasses();
  mlir::registerPass([] { return onnx_mlir::createShapeInferencePass(); });
  mlir::registerPass([] { return onnx_mlir::createConstPropONNXToONNXPass(); });
  mlir::registerPass(
      [] { return onnx_mlir::createSimplifyShapeRelatedOpsPass(); });
  mlir::registerPass([] { return onnx_mlir::createStandardFuncReturnPass(); });
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::func::registerInlinerExtension(registry);
  registry.insert<vrc::udonhl::UdonHLDialect, vrc::udonasm::UdonAsmDialect,
                  vrc::vrc::VRCDialect, vrc::vrcgpu::VRCGPUDialect,
                  mlir::gpu::GPUDialect, mlir::ub::UBDialect, mlir::DLTIDialect,
                  mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::ONNXDialect>();

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::PassPipelineCLParser pipeline("", "Compiler passes to run");
  if (!llvm::cl::ParseCommandLineOptions(argc, argv, "udoncl pass driver\n"))
    return 1;

  std::string error;
  auto input = mlir::openInputFile(onnx_mlir::inputFilename, &error);
  if (!input) {
    llvm::errs() << error << "\n";
    return 1;
  }
  auto output = mlir::openOutputFile(onnx_mlir::outputBaseName, &error);
  if (!output) {
    llvm::errs() << error << "\n";
    return 1;
  }
  auto setup = [&](mlir::PassManager &manager) {
    mlir::MLIRContext *context = manager.getContext();
    return pipeline.addToPipeline(manager, [&](const llvm::Twine &message) {
      mlir::emitError(mlir::UnknownLoc::get(context)) << message;
      return mlir::failure();
    });
  };
  mlir::MlirOptMainConfig config;
  config.setPassPipelineSetupFn(setup)
      .splitInputFile(onnx_mlir::split_input_file ? mlir::kDefaultSplitMarker
                                                  : "")
      .outputSplitMarker(onnx_mlir::split_input_file ? mlir::kDefaultSplitMarker
                                                     : "")
      .verifyDiagnostics(
          onnx_mlir::verify_diagnostics
              ? mlir::SourceMgrDiagnosticVerifierHandler::Level::All
              : mlir::SourceMgrDiagnosticVerifierHandler::Level::None)
      .verifyPasses(onnx_mlir::verify_passes)
      .allowUnregisteredDialects(onnx_mlir::allowUnregisteredDialects)
      .emitBytecode(emitBytecode)
      .useExplicitModule(false);
  if (mlir::failed(
          mlir::MlirOptMain(output->os(), std::move(input), registry, config)))
    return 1;
  output->keep();
  return 0;
}
