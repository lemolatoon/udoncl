#include "VRC/Conversion/BoundaryFrontend/Frontend.h"
#include "VRC/Conversion/LLVMToHighLevel/Passes.h"
#include "VRC/Conversion/Passes.h"
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
int main(int argc, char **argv) {
  mlir::registerTransformsPasses();
  mlir::registerAsyncPasses();
  vrc::registerCBoundaryRecoveryPipeline();
  vrc::registerVRCRecoveryPasses();
  vrc::registerHighLevelRecoveryPipeline();
  vrc::registerVRCBoundaryPasses();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  registry.insert<vrc::udonhl::UdonHLDialect, vrc::udonasm::UdonAsmDialect,
                  vrc::vrc::VRCDialect, vrc::vrcgpu::VRCGPUDialect,
                  mlir::gpu::GPUDialect, mlir::ub::UBDialect, mlir::DLTIDialect,
                  mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "udoncl pass driver", registry));
}
