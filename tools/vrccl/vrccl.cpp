#include "VRC/Conversion/BoundaryFrontend/Frontend.h"
#include "VRC/ABI/Manifest/Boundary.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "VRC/Target/HLSL/Emitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include <map>
using namespace mlir;
using namespace llvm;
namespace {
cl::opt<bool> emitBundle(
    "emit-boundary-bundle",
    cl::desc("Compile the initial Host/Device boundary MLIR pattern"));
cl::opt<bool> emitLLVMBundle(
    "emit-llvm-boundary-bundle",
    cl::desc(
        "Recover paired C/OpenCL LLVM IR using the input JSON ABI contract"));
cl::opt<std::string> hostLLVM("host-llvm", cl::desc("Host LLVM IR from C"));
cl::opt<std::string> deviceLLVM("device-llvm",
                                cl::desc("Device LLVM IR from OpenCL C"));
cl::opt<bool> checkRequest(
    "check-boundary-request",
    cl::desc(
        "Validate a request against a prototype manifest; no GPU dispatch"));
cl::opt<std::string>
    requestFilename("request", cl::desc("JSON dispatch descriptor request"));
cl::opt<std::string>
    inputFilename(cl::Positional,
                  cl::desc("<boundary MLIR, config.json or manifest.json>"),
                  cl::Required);
cl::opt<std::string> outputDirectory("o",
                                     cl::desc("New output bundle directory"));

// Validate and generate all content before touching existing assets. A fresh
// directory is published with one rename; identical existing bundles are
// reused.
bool publish(const std::map<std::string, std::string> &files) {
  if (sys::fs::exists(outputDirectory)) {
    for (const auto &file : files) {
      SmallString<256> path(outputDirectory);
      sys::path::append(path, file.first);
      auto existing = MemoryBuffer::getFile(path);
      if (!existing || (*existing)->getBuffer() != file.second) {
        errs() << "output directory already exists with different content; "
                  "choose a new -o directory\n";
        return false;
      }
    }
    return true;
  }
  SmallString<256> parent(outputDirectory);
  sys::path::remove_filename(parent);
  if (!parent.empty())
    if (auto error = sys::fs::create_directories(parent)) {
      errs() << error.message() << "\n";
      return false;
    }
  SmallString<256> temporary, absolute(outputDirectory);
  if (auto error = sys::fs::make_absolute(absolute)) {
    errs() << error.message() << "\n";
    return false;
  }
  if (auto error =
          sys::fs::createUniqueDirectory(Twine(absolute) + ".tmp", temporary)) {
    errs() << error.message() << "\n";
    return false;
  }
  bool ok = true;
  for (const auto &file : files) {
    SmallString<256> path(temporary);
    sys::path::append(path, file.first);
    SmallString<256> directory(path);
    sys::path::remove_filename(directory);
    if (auto error = sys::fs::create_directories(directory)) {
      errs() << error.message() << "\n";
      ok = false;
      break;
    }
    std::error_code error;
    ToolOutputFile out(path, error, sys::fs::OF_None);
    if (error) {
      errs() << error.message() << "\n";
      ok = false;
      break;
    }
    out.os() << file.second;
    out.os().flush();
    if (out.os().has_error()) {
      out.os().clear_error();
      ok = false;
      break;
    }
    out.keep();
  }
  if (ok) {
    if (auto error = sys::fs::rename(temporary, outputDirectory)) {
      errs() << error.message() << "\n";
      ok = false;
    }
  }
  if (!ok)
    (void)sys::fs::remove_directories(temporary);
  return ok;
}
} // namespace
int main(int argc, char **argv) {
  InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(
      argc, argv,
      "udoncl boundary prototype compiler (no Udon program emission yet)\n");
  if (int(emitBundle) + int(emitLLVMBundle) + int(checkRequest) != 1) {
    errs() << "select exactly one boundary mode\n";
    return 1;
  }
  if (checkRequest) {
    auto manifest = MemoryBuffer::getFile(inputFilename),
         request = MemoryBuffer::getFile(requestFilename);
    if (!manifest || !request) {
      errs() << "cannot read manifest or request JSON\n";
      return 1;
    }
    auto manifestJSON = json::parse((*manifest)->getBuffer()),
         requestJSON = json::parse((*request)->getBuffer());
    if (!manifestJSON) {
      errs() << toString(manifestJSON.takeError()) << "\n";
      return 1;
    }
    if (!requestJSON) {
      errs() << toString(requestJSON.takeError()) << "\n";
      return 1;
    }
    if (!manifestJSON->getAsObject() || !requestJSON->getAsObject()) {
      errs() << "JSON root must be an object\n";
      return 1;
    }
    if (!::vrc::validateBoundaryRequest(*manifestJSON->getAsObject(),
                                        *requestJSON->getAsObject(), errs()))
      return 1;
    outs() << "Descriptor contract valid; no GPU dispatch executed\n";
    return 0;
  }
  if (outputDirectory.empty()) {
    errs() << "bundle mode requires -o output-directory\n";
    return 1;
  }
  DialectRegistry registry;
  registry.insert<::vrc::udonhl::UdonHLDialect, ::vrc::vrc::VRCDialect,
                  ::vrc::vrcgpu::VRCGPUDialect, arith::ArithDialect,
                  func::FuncDialect, gpu::GPUDialect, memref::MemRefDialect>();
  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module;
  std::map<std::string, std::string> files;
  if (emitLLVMBundle) {
    auto input = MemoryBuffer::getFile(inputFilename);
    if (!input || hostLLVM.empty() || deviceLLVM.empty()) {
      errs() << "LLVM bundle mode requires a JSON config, --host-llvm and "
                "--device-llvm\n";
      return 1;
    }
    auto config = json::parse((*input)->getBuffer());
    if (!config) {
      errs() << toString(config.takeError()) << "\n";
      return 1;
    }
    if (!config->getAsObject()) {
      errs() << "config must be a JSON object\n";
      return 1;
    }
    module = ::vrc::importCBoundary(hostLLVM, deviceLLVM,
                                    *config->getAsObject(), context);
    if (!module || failed(verify(*module)))
      return 1;
    raw_string_ostream imported(files["frontend.imported.mlir"]);
    module->print(imported);
    imported << "\n";
    PassManager recovery(&context);
    ::vrc::buildCBoundaryRecoveryPipeline(recovery);
    if (failed(recovery.run(*module)))
      return 1;
    raw_string_ostream recovered(files["frontend.recovered.mlir"]);
    module->print(recovered);
    recovered << "\n";
  } else
    module = parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module || failed(verify(*module)))
    return 1;
  PassManager manager(&context);
  manager.addPass(::vrc::createLowerHostToUdonHLPass());
  manager.addPass(::vrc::createSubstituteRuntimeIntrinsicsPass());
  manager.addPass(::vrc::createLowerDeviceSingleInvocationPass());
  manager.addPass(::vrc::createVerifyHostDeviceBoundaryPass());
  if (failed(manager.run(*module)) || failed(verify(*module)))
    return 1;
  for (auto kernel : module->getOps<::vrc::vrcgpu::KernelOp>()) {
    std::string shader, hlsl;
    raw_string_ostream shaderOut(shader), hlslOut(hlsl);
    if (failed(::vrc::emitShaderLab(kernel, shaderOut)) ||
        failed(::vrc::emitFragmentHLSL(kernel, hlslOut)))
      return 1;
    std::string prefix =
        "kernels/kernel_" + std::to_string(kernel.getKernelId());
    files[prefix + ".shader"] = shader;
    files[prefix + ".hlsl"] = hlsl;
  }
  std::string ir;
  raw_string_ostream irOut(ir);
  module->print(irOut);
  irOut << "\n";
  files["host-device.mlir"] = ir;
  files["manifest.json"] =
      formatv("{0:2}\n", json::Value(::vrc::buildBoundaryManifest(*module)))
          .str();
  if (!publish(files))
    return 1;
  outs() << "Boundary prototype emitted to " << outputDirectory
         << "; runtime/SDK/client validation pending\n";
  return 0;
}
