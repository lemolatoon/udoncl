#include "VRC/ABI/Manifest/Boundary.h"
#include "VRC/Conversion/BoundaryFrontend/Frontend.h"
#include "VRC/Conversion/LLVMToHighLevel/Import.h"
#include "VRC/Conversion/LLVMToHighLevel/Passes.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "VRC/Target/HLSL/Emitter.h"
#include "VRC/Target/UdonAssembly/Emitter.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IRReader/IRReader.h"
#include <set>
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
cl::opt<bool> emitNativeDevice(
    "emit-native-device-bundle",
    cl::desc(
        "Compile a standalone Device bundle for the native OpenCL runtime"));
cl::opt<bool> emitUnityLibrary(
    "emit-unity-library-bundle",
    cl::desc("Compile scalar Host C exports into an importable Unity bundle"));
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

bool identifier(StringRef value) {
  if (value.empty() || !(isAlpha(value.front()) || value.front() == '_'))
    return false;
  return llvm::all_of(value.drop_front(),
                      [](char c) { return isAlnum(c) || c == '_'; });
}

bool configureUnityExports(ModuleOp module, const json::Object &config) {
  auto program = config.getString("program");
  auto *exports = config.getArray("exports");
  if (config.getInteger("schema_version") != 1 || !program ||
      !identifier(*program) || !exports || exports->empty()) {
    errs() << "Unity library config requires schema_version 1, a safe program "
              "name and non-empty exports\n";
    return false;
  }
  std::set<std::string> selected, publicNames;
  std::set<int64_t> ids;
  OpBuilder builder(module.getContext());
  for (const json::Value &value : *exports) {
    auto *entry = value.getAsObject();
    auto symbol = entry ? entry->getString("symbol") : std::nullopt;
    auto publicName = entry ? entry->getString("public_name") : std::nullopt;
    auto id = entry ? entry->getInteger("id") : std::nullopt;
    auto *parameters = entry ? entry->getArray("parameters") : nullptr;
    if (!symbol || !identifier(*symbol) || !publicName ||
        !identifier(*publicName) || !id || *id <= 0 || *id > INT32_MAX ||
        !parameters || !selected.insert(symbol->str()).second ||
        !publicNames.insert(publicName->str()).second ||
        !ids.insert(*id).second) {
      errs() << "invalid or duplicate Unity library export\n";
      return false;
    }
    auto function = module.lookupSymbol<func::FuncOp>(*symbol);
    if (!function || function.isExternal() || function.isPrivate() ||
        parameters->size() != function.getNumArguments()) {
      errs() << "Unity library export does not resolve to a public definition: "
             << *symbol << "\n";
      return false;
    }
    SmallVector<mlir::Attribute> names;
    for (const json::Value &parameter : *parameters) {
      auto name = parameter.getAsString();
      if (!name || !identifier(*name)) {
        errs() << "invalid Unity library parameter name\n";
        return false;
      }
      names.push_back(builder.getStringAttr(*name));
    }
    function->setAttr("vrc.export_id", builder.getI32IntegerAttr(*id));
    function->setAttr("vrc.public_name", builder.getStringAttr(*publicName));
    function->setAttr("vrc.parameter_names", builder.getArrayAttr(names));
  }
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (selected.count(function.getSymName().str()))
      for (func::CallOp call : function.getOps<func::CallOp>())
        if (!selected.count(call.getCallee().str())) {
          call.emitError("Unity library helper calls must be inlined before "
                         "Udon lowering");
          return false;
        }
  for (func::FuncOp function :
       llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
    if (!selected.count(function.getSymName().str()))
      function.erase();
  module->setAttr("vrc.target", builder.getStringAttr("vrc-pc"));
  return true;
}

std::string unityLibraryDisplayShader(StringRef program) {
  return (Twine("Shader \"Hidden/UdonCL/") + program +
          "/Display\" {\n"
          "  SubShader { Pass { HLSLPROGRAM\n"
          "    #pragma vertex vert\n"
          "    #pragma fragment frag\n"
          "    #include \"UnityCG.cginc\"\n"
          "    float4 vert(float4 vertex : POSITION) : SV_POSITION {\n"
          "      return UnityObjectToClipPos(vertex);\n"
          "    }\n"
          "    float4 frag() : SV_Target { return float4(0, 0, 0, 1); }\n"
          "  ENDHLSL } }\n"
          "}\n")
      .str();
}
} // namespace
int main(int argc, char **argv) {
  InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                              "udoncl C/OpenCL C artifact compiler\n");
  if (int(emitBundle) + int(emitLLVMBundle) + int(checkRequest) +
          int(emitNativeDevice) + int(emitUnityLibrary) !=
      1) {
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
  registry.insert<::vrc::udonasm::UdonAsmDialect, ::vrc::udonhl::UdonHLDialect,
                  ::vrc::vrc::VRCDialect, ::vrc::vrcgpu::VRCGPUDialect,
                  arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect, gpu::GPUDialect, memref::MemRefDialect>();
  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module;
  std::map<std::string, std::string> files;
  bool gatherProfile = false;
  std::optional<json::Object> unityConfig;
  if (emitUnityLibrary) {
    auto input = MemoryBuffer::getFile(inputFilename);
    if (!input || hostLLVM.empty() || !deviceLLVM.empty()) {
      errs() << "Unity library mode requires a JSON config and --host-llvm "
                "without --device-llvm\n";
      return 1;
    }
    auto parsed = json::parse((*input)->getBuffer());
    if (!parsed) {
      errs() << toString(parsed.takeError()) << "\n";
      return 1;
    }
    auto *object = parsed->getAsObject();
    if (!object) {
      errs() << "config must be a JSON object\n";
      return 1;
    }
    unityConfig = *object;
    LLVMContext llvmContext;
    SMDiagnostic diagnostic;
    auto inputModule = parseIRFile(hostLLVM, diagnostic, llvmContext);
    if (!inputModule) {
      diagnostic.print(argv[0], errs());
      return 1;
    }
    if (inputModule->getTargetTriple().str() != "spir-unknown-unknown" ||
        !inputModule->getDataLayout().isLittleEndian() ||
        inputModule->getDataLayout().getPointerSizeInBits() != 32) {
      errs() << "Unity Host library requires 32-bit little-endian SPIR input\n";
      return 1;
    }
    module = ::vrc::importLLVM(std::move(inputModule), context);
    if (!module || failed(verify(*module)))
      return 1;
    PassManager recovery(&context);
    recovery.addPass(createMem2Reg());
    recovery.addPass(::vrc::createRaiseLLVMToScalarPass());
    recovery.addPass(createCanonicalizerPass());
    if (failed(recovery.run(*module)) ||
        !configureUnityExports(*module, *unityConfig))
      return 1;
    raw_string_ostream recovered(files["frontend.recovered.mlir"]);
    module->print(recovered);
    recovered << "\n";
  } else if (emitLLVMBundle || emitNativeDevice) {
    auto input = MemoryBuffer::getFile(inputFilename);
    if (!input || (!emitNativeDevice && hostLLVM.empty()) ||
        deviceLLVM.empty()) {
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
    gatherProfile = config->getAsObject()->getString("profile") == "gather-f32";
    module = ::vrc::importCBoundary(
        emitNativeDevice ? StringRef() : StringRef(hostLLVM), deviceLLVM,
        *config->getAsObject(), context);
    if (!module || failed(verify(*module)))
      return 1;
    raw_string_ostream imported(files["frontend.imported.mlir"]);
    module->print(imported);
    imported << "\n";
    PassManager recovery(&context);
    if (emitNativeDevice) {
      recovery.addNestedPass<ModuleOp>(createMem2Reg());
      if (gatherProfile) {
        recovery.addNestedPass<ModuleOp>(createCanonicalizerPass());
        recovery.addNestedPass<ModuleOp>(::vrc::createRaiseGatherDevicePass());
      } else
        recovery.addNestedPass<ModuleOp>(
            ::vrc::createRaiseLLVMBoundaryDevicePass());
      recovery.addNestedPass<ModuleOp>(::vrc::createRaiseLLVMToScalarPass());
      recovery.addNestedPass<ModuleOp>(createCanonicalizerPass());
      if (gatherProfile) {
        recovery.addNestedPass<ModuleOp>(createLiftControlFlowToSCFPass());
        recovery.addNestedPass<ModuleOp>(createCanonicalizerPass());
      }
      recovery.addPass(::vrc::createPrepareNativeDevicePass());
    } else
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
  if (emitUnityLibrary) {
    manager.addPass(::vrc::createLowerHostToUdonHLPass());
    if (failed(manager.run(*module)) || failed(verify(*module)))
      return 1;
    auto manifest = ::vrc::buildBoundaryManifest(*module);
    auto program = unityConfig->getString("program");
    manifest["artifact_stage"] = "unity-udoncl-library";
    manifest["program"] = json::Object{{"name", *program},
                                       {"assembly", "host.uasm"},
                                       {"display_shader", "display.shader"},
                                       {"ir", "host.udonasm.mlir"}};
    manifest["ready_for_unity"] = true;
    PassManager udon(&context);
    udon.addPass(::vrc::createLowerUdonHLToUdonAsmPass());
    if (failed(udon.run(*module)) || failed(verify(*module)))
      return 1;
    std::string assembly;
    raw_string_ostream assemblyOut(assembly);
    if (failed(::vrc::emitUdonAssembly(*module, assemblyOut)))
      return 1;
    files["host.uasm"] = std::move(assembly);
    files["display.shader"] = unityLibraryDisplayShader(*program);
    raw_string_ostream lowered(files["host.udonasm.mlir"]);
    module->print(lowered);
    lowered << "\n";
    files["manifest.json"] =
        formatv("{0:2}\n", json::Value(std::move(manifest))).str();
    if (!publish(files))
      return 1;
    outs() << "Unity Host library emitted to " << outputDirectory << "\n";
    return 0;
  }
  if (!emitNativeDevice) {
    manager.addPass(::vrc::createLowerHostToUdonHLPass());
    manager.addPass(::vrc::createSubstituteRuntimeIntrinsicsPass());
  }
  if (gatherProfile)
    manager.addPass(::vrc::createLowerGatherDevicePass());
  else
    manager.addPass(::vrc::createLowerDeviceSingleInvocationPass());
  if (!emitNativeDevice)
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
  auto manifest = ::vrc::buildBoundaryManifest(*module);
  if (emitNativeDevice)
    manifest["artifact_stage"] = "native-device-prototype";
  files["manifest.json"] =
      formatv("{0:2}\n", json::Value(std::move(manifest))).str();
  if (!publish(files))
    return 1;
  outs() << "Boundary prototype emitted to " << outputDirectory
         << "; runtime/SDK/client validation pending\n";
  return 0;
}
