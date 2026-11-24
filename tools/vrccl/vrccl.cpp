#include "VRC/ABI/Manifest/Boundary.h"
#include "VRC/ABI/Manifest/BindingSource.h"
#include "VRC/ABI/Manifest/UdonSharpABI.h"
#include "VRC/Conversion/BoundaryFrontend/Frontend.h"
#include "VRC/Conversion/HostStoragePipeline.h"
#include "VRC/Conversion/LLVMToHighLevel/Import.h"
#include "VRC/Conversion/LLVMToHighLevel/Passes.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Conversion/StaticDeviceResourcePipeline.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "VRC/Target/HLSL/Emitter.h"
#include "VRC/Target/UdonAssembly/Emitter.h"
#include "HostAnnotations.h"
#include "HostBufferPlan.h"
#include "OpenCLUserKernels.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IRReader/IRReader.h"
#include <chrono>
#include <set>
#include <map>
#include <limits>
using namespace mlir;
using namespace llvm;
namespace {
FailureOr<std::string> denseTensorBytes(DenseElementsAttr value) {
  auto type = cast<ShapedType>(value.getType());
  mlir::Type element = type.getElementType();
  unsigned bits = element.getIntOrFloatBitWidth();
  if (!bits || bits % 8)
    return failure();
  size_t elementBytes = bits / 8;
  if (!value.isSplat()) {
    ArrayRef<char> raw = value.getRawData();
    return std::string(raw.begin(), raw.end());
  }
  std::string result;
  result.reserve(value.getNumElements() * elementBytes);
  APInt splat = element.isIntOrIndex()
                    ? value.getSplatValue<APInt>()
                    : value.getSplatValue<APFloat>().bitcastToAPInt();
  for (int64_t index = 0; index < value.getNumElements(); ++index)
    for (size_t byte = 0; byte < elementBytes; ++byte)
      result.push_back(
          static_cast<char>(splat.extractBitsAsZExtValue(8, byte * 8)));
  return result;
}

void normalizeUnityHostCallingConventions(llvm::Module &module) {
  SmallVector<llvm::CallInst *> unsignedMultiplyOverflowCalls;
  SmallVector<llvm::CallInst *> unsignedMaximumCalls;
  SmallVector<llvm::CallInst *> unsignedMinimumCalls;
  SmallVector<llvm::CallInst *> signedMaximumCalls;
  SmallVector<llvm::CallInst *> signedMinimumCalls;
  SmallVector<llvm::CallInst *> absoluteFloatCalls;
  SmallVector<llvm::CallInst *> populationCountCalls;
  SmallVector<llvm::FreezeInst *> freezes;
  for (llvm::Function &function : module) {
    if (function.getCallingConv() == llvm::CallingConv::Fast)
      function.setCallingConv(llvm::CallingConv::C);
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &instruction : block) {
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction))
          call->setTailCallKind(llvm::CallInst::TCK_None);
        if (llvm::isa<llvm::FPMathOperator>(instruction))
          instruction.copyFastMathFlags(llvm::FastMathFlags());
        if (auto *disjoint =
                llvm::dyn_cast<llvm::PossiblyDisjointInst>(&instruction))
          disjoint->setIsDisjoint(false);
        if (llvm::isa<llvm::ZExtInst, llvm::UIToFPInst>(instruction) &&
            instruction.hasNonNeg())
          instruction.setNonNeg(false);
        if (auto *freeze = llvm::dyn_cast<llvm::FreezeInst>(&instruction))
          freezes.push_back(freeze);
        if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction))
          if (call->getCallingConv() == llvm::CallingConv::Fast)
            call->setCallingConv(llvm::CallingConv::C);
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction))
          call->setTailCallKind(llvm::CallInst::TCK_None);
        auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
        llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
        if (callee && callee->getName() == "llvm.umul.with.overflow.i32")
          unsignedMultiplyOverflowCalls.push_back(call);
        if (callee && callee->getName() == "llvm.umax.i32")
          unsignedMaximumCalls.push_back(call);
        if (callee && callee->getName() == "llvm.umin.i32")
          unsignedMinimumCalls.push_back(call);
        if (callee && callee->getName() == "llvm.smax.i32")
          signedMaximumCalls.push_back(call);
        if (callee && callee->getName() == "llvm.smin.i32")
          signedMinimumCalls.push_back(call);
        if (callee && callee->getName() == "llvm.fabs.f32")
          absoluteFloatCalls.push_back(call);
        if (callee && callee->getName() == "llvm.ctpop.i32")
          populationCountCalls.push_back(call);
      }
    }
  }

  // Freeze differs from its input only for poison/undef. The supported Host C
  // subset already refines those undefined executions to deterministic zero,
  // so forwarding the operand preserves every defined C execution.
  for (llvm::FreezeInst *freeze : freezes) {
    freeze->replaceAllUsesWith(freeze->getOperand(0));
    freeze->eraseFromParent();
  }

  // Clang uses this intrinsic for checked allocation-size arithmetic at O1.
  // MLIR imports its aggregate result as an LLVM struct, which cannot cross
  // the scalar Host recovery boundary. Preserve the exact unsigned semantics
  // using a widened product before importing the module.
  SmallVector<llvm::Function *> deadIntrinsics;
  for (llvm::CallInst *call : unsignedMultiplyOverflowCalls) {
    llvm::IRBuilder<> builder(call);
    llvm::Type *wideType = builder.getInt64Ty();
    llvm::Value *lhs = builder.CreateZExt(call->getArgOperand(0), wideType);
    llvm::Value *rhs = builder.CreateZExt(call->getArgOperand(1), wideType);
    llvm::Value *wideProduct = builder.CreateMul(lhs, rhs);
    llvm::Value *product =
        builder.CreateTrunc(wideProduct, builder.getInt32Ty());
    llvm::Value *overflow = builder.CreateICmpUGT(
        wideProduct, llvm::ConstantInt::get(wideType, UINT32_MAX));

    SmallVector<llvm::ExtractValueInst *> extracts;
    for (llvm::User *user : call->users())
      if (auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(user))
        extracts.push_back(extract);
    for (llvm::ExtractValueInst *extract : extracts) {
      ArrayRef<unsigned> indices = extract->getIndices();
      if (indices.size() != 1 || indices.front() > 1)
        continue;
      extract->replaceAllUsesWith(indices.front() == 0 ? product : overflow);
      extract->eraseFromParent();
    }
    if (!call->use_empty())
      continue;
    llvm::Function *callee = call->getCalledFunction();
    call->eraseFromParent();
    if (callee && callee->use_empty())
      deadIntrinsics.push_back(callee);
  }
  for (llvm::Function *function : deadIntrinsics)
    if (function->use_empty())
      function->eraseFromParent();

  deadIntrinsics.clear();
  for (llvm::CallInst *call : unsignedMaximumCalls) {
    llvm::IRBuilder<> builder(call);
    llvm::Value *lhs = call->getArgOperand(0);
    llvm::Value *rhs = call->getArgOperand(1);
    llvm::Value *maximum =
        builder.CreateSelect(builder.CreateICmpUGT(lhs, rhs), lhs, rhs);
    call->replaceAllUsesWith(maximum);
    llvm::Function *callee = call->getCalledFunction();
    call->eraseFromParent();
    if (callee && callee->use_empty())
      deadIntrinsics.push_back(callee);
  }
  for (llvm::Function *function : deadIntrinsics)
    if (function->use_empty())
      function->eraseFromParent();

  deadIntrinsics.clear();
  for (llvm::CallInst *call : unsignedMinimumCalls) {
    llvm::IRBuilder<> builder(call);
    llvm::Value *lhs = call->getArgOperand(0);
    llvm::Value *rhs = call->getArgOperand(1);
    llvm::Value *minimum =
        builder.CreateSelect(builder.CreateICmpULT(lhs, rhs), lhs, rhs);
    call->replaceAllUsesWith(minimum);
    llvm::Function *callee = call->getCalledFunction();
    call->eraseFromParent();
    if (callee && callee->use_empty())
      deadIntrinsics.push_back(callee);
  }
  for (llvm::Function *function : deadIntrinsics)
    if (function->use_empty())
      function->eraseFromParent();

  deadIntrinsics.clear();
  for (llvm::CallInst *call : signedMaximumCalls) {
    llvm::IRBuilder<> builder(call);
    llvm::Value *lhs = call->getArgOperand(0);
    llvm::Value *rhs = call->getArgOperand(1);
    llvm::Value *maximum =
        builder.CreateSelect(builder.CreateICmpSGT(lhs, rhs), lhs, rhs);
    call->replaceAllUsesWith(maximum);
    llvm::Function *callee = call->getCalledFunction();
    call->eraseFromParent();
    if (callee && callee->use_empty())
      deadIntrinsics.push_back(callee);
  }
  for (llvm::Function *function : deadIntrinsics)
    if (function->use_empty())
      function->eraseFromParent();

  deadIntrinsics.clear();
  for (llvm::CallInst *call : signedMinimumCalls) {
    llvm::IRBuilder<> builder(call);
    llvm::Value *lhs = call->getArgOperand(0);
    llvm::Value *rhs = call->getArgOperand(1);
    llvm::Value *minimum =
        builder.CreateSelect(builder.CreateICmpSLT(lhs, rhs), lhs, rhs);
    call->replaceAllUsesWith(minimum);
    llvm::Function *callee = call->getCalledFunction();
    call->eraseFromParent();
    if (callee && callee->use_empty())
      deadIntrinsics.push_back(callee);
  }
  for (llvm::Function *function : deadIntrinsics)
    if (function->use_empty())
      function->eraseFromParent();

  deadIntrinsics.clear();
  for (llvm::CallInst *call : absoluteFloatCalls) {
    llvm::IRBuilder<> builder(call);
    llvm::Value *bits =
        builder.CreateBitCast(call->getArgOperand(0), builder.getInt32Ty());
    llvm::Value *magnitude = builder.CreateAnd(
        bits, llvm::ConstantInt::get(builder.getInt32Ty(), 0x7fffffffU));
    llvm::Value *absolute =
        builder.CreateBitCast(magnitude, builder.getFloatTy());
    call->replaceAllUsesWith(absolute);
    llvm::Function *callee = call->getCalledFunction();
    call->eraseFromParent();
    if (callee && callee->use_empty())
      deadIntrinsics.push_back(callee);
  }
  for (llvm::Function *function : deadIntrinsics)
    if (function->use_empty())
      function->eraseFromParent();

  deadIntrinsics.clear();
  for (llvm::CallInst *call : populationCountCalls) {
    llvm::IRBuilder<> builder(call);
    auto constant = [&](uint32_t value) {
      return llvm::ConstantInt::get(builder.getInt32Ty(), value);
    };
    llvm::Value *value = call->getArgOperand(0);
    value = builder.CreateSub(
        value, builder.CreateAnd(builder.CreateLShr(value, constant(1)),
                                 constant(0x55555555U)));
    value = builder.CreateAdd(
        builder.CreateAnd(value, constant(0x33333333U)),
        builder.CreateAnd(builder.CreateLShr(value, constant(2)),
                          constant(0x33333333U)));
    value = builder.CreateAnd(
        builder.CreateAdd(value, builder.CreateLShr(value, constant(4))),
        constant(0x0f0f0f0fU));
    value = builder.CreateAdd(value, builder.CreateLShr(value, constant(8)));
    value = builder.CreateAdd(value, builder.CreateLShr(value, constant(16)));
    value = builder.CreateAnd(value, constant(0x3fU));
    call->replaceAllUsesWith(value);
    llvm::Function *callee = call->getCalledFunction();
    call->eraseFromParent();
    if (callee && callee->use_empty())
      deadIntrinsics.push_back(callee);
  }
  for (llvm::Function *function : deadIntrinsics)
    if (function->use_empty())
      function->eraseFromParent();
}

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
cl::opt<bool> emitOpenCLUserKernels(
    "emit-opencl-user-kernels",
    cl::desc("Write the OpenCL AOT runtime declarations of the user OpenCL C "
             "kernels the Host LLVM input creates (--device-llvm, -o "
             "directory)"));
cl::opt<bool> planHostBufferCapacity(
    "plan-host-buffer-capacity",
    cl::desc("Plan constant C Host cl_mem allocations before runtime linking"));
cl::opt<bool> omitIntermediateIR(
    "omit-intermediate-ir",
    cl::desc("Omit diagnostic MLIR snapshots from a Unity library bundle"));
cl::opt<bool> debugCompilerPhases(
    "debug-compiler-phases",
    cl::desc("Print coarse compiler phase boundaries for diagnostics"));
cl::opt<bool> reportStaticResources(
    "report-static-resources",
    cl::desc("Print whether static Unity resource lowering is applied to a "
             "Unity library and what blocks it"));
cl::opt<bool> runtimeValidation(
    "runtime-validation",
    cl::desc("Record that read/write physical-resource alias validation is "
             "compiled into the Host runtime"));
cl::opt<bool> noRuntimeValidation(
    "no-runtime-validation",
    cl::desc("Record that read/write physical-resource alias validation is "
             "compiled out of the Host runtime"));
cl::opt<std::string>
    hostLLVM("host-llvm",
             cl::desc("Host LLVM IR from C; runtime-free when paired with "
                      "--host-library-llvm"));
cl::opt<std::string>
    hostMLIR("host-mlir", cl::desc("Typed Host MLIR for compiler validation"));
cl::opt<std::string> hostLibraryLLVM(
    "host-library-llvm",
    cl::desc("OpenCL AOT runtime LLVM IR linked after Host analysis"));
cl::opt<std::string>
    hostBufferPlan("host-buffer-plan",
                   cl::desc("Final linked Host mutable-buffer plan JSON"));
cl::opt<std::string>
    capacityHeader("capacity-header",
                   cl::desc("Header generated by --plan-host-buffer-capacity"));
cl::opt<int64_t> baseMutableHandleCapacity(
    "base-mutable-handle-capacity",
    cl::desc("ONNX-generated mutable handle capacity"), cl::init(0));
cl::opt<std::string> staticTensorDirectory(
    "static-tensor-directory",
    cl::desc("Directory containing external ONNX static tensor payloads"));
cl::opt<std::string> deviceLLVM("device-llvm",
                                cl::desc("Device LLVM IR from OpenCL C"));
cl::opt<std::string> deviceMLIR(
    "device-mlir",
    cl::desc("Device MLIR holding compiler-synthesized gather kernels"));
cl::opt<bool> checkRequest(
    "check-boundary-request",
    cl::desc(
        "Validate a request against a prototype manifest; no GPU dispatch"));
cl::opt<std::string>
    requestFilename("request", cl::desc("JSON dispatch descriptor request"));
cl::opt<std::string>
    inputFilename(cl::Positional,
                  cl::desc("<boundary MLIR, config.json or manifest.json>; "
                           "none for a Unity library"));
cl::opt<std::string> outputDirectory("o",
                                     cl::desc("New output bundle directory"));

// Unity library settings. The Host source names the program, exports and
// public buffers; these options set the rest.
cl::opt<std::string> bindingNamespace(
    "binding-namespace",
    cl::desc("Namespace of the generated U# binding of a Unity library "
             "(default Lemolatoon.Udoncl.Generated)"));
cl::opt<std::string> bindingClass(
    "binding-class",
    cl::desc("Class of the generated U# binding of a Unity library "
             "(default Udoncl<PascalCase program>Binding)"));
cl::opt<std::string> deviceTexelFormat(
    "device-texel-format",
    cl::desc("Texel format of the Device buffers and immutable tensor "
             "textures of a Unity library: rgba8_unorm (default) or "
             "rgba32_float"));
cl::opt<std::string> deviceWorkBudget(
    "device-work-budget",
    cl::desc("Device work slice and frame budgets of a Unity library as "
             "comma-separated key=value pairs (device_cost, host_cost, "
             "fragment_weight, frame_microseconds, frame_continuations)"));

/// Folds the Unity library command-line settings into `config`, the
/// internal record of the library settings the compiler passes on.
LogicalResult applyUnityLibraryOptions(json::Object &config) {
  if (!bindingNamespace.empty() || !bindingClass.empty()) {
    json::Object binding;
    if (!bindingNamespace.empty())
      binding["namespace"] = bindingNamespace;
    if (!bindingClass.empty())
      binding["class_name"] = bindingClass;
    config["binding"] = std::move(binding);
  }
  if (!deviceTexelFormat.empty())
    config["device_texel_format"] = deviceTexelFormat;
  if (!deviceWorkBudget.empty()) {
    json::Object budget;
    SmallVector<StringRef> entries;
    StringRef(deviceWorkBudget).split(entries, ',', -1, false);
    for (StringRef entry : entries) {
      auto [key, value] = entry.split('=');
      int64_t number;
      key = key.trim();
      if (key.empty() || value.trim().getAsInteger(10, number)) {
        errs() << "--device-work-budget entry must be key=integer: " << entry
               << "\n";
        return failure();
      }
      if (!budget.try_emplace(key, number).second) {
        errs() << "--device-work-budget sets " << key << " twice\n";
        return failure();
      }
    }
    config["device_work_budget"] = std::move(budget);
  }
  return success();
}

// Validate and generate all content before touching existing assets. A fresh
// directory is published with one rename; identical existing bundles are
// reused.
bool publish(const std::map<std::string, std::string> &files,
             const std::map<std::string, std::string> &externalFiles = {}) {
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
    for (const auto &file : externalFiles) {
      SmallString<256> path(outputDirectory);
      sys::path::append(path, file.first);
      auto existing = MemoryBuffer::getFile(path, false, false);
      auto source = MemoryBuffer::getFile(file.second, false, false);
      if (!existing || !source ||
          (*existing)->getBuffer() != (*source)->getBuffer()) {
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
      errs() << "cannot create Unity bundle parent '" << parent
             << "': " << error.message() << "\n";
      return false;
    }
  SmallString<256> temporary, absolute(outputDirectory);
  if (auto error = sys::fs::make_absolute(absolute)) {
    errs() << "cannot resolve Unity bundle output '" << outputDirectory
           << "': " << error.message() << "\n";
    return false;
  }
  if (auto error =
          sys::fs::createUniqueDirectory(Twine(absolute) + ".tmp", temporary)) {
    errs() << "cannot create temporary Unity bundle directory for '" << absolute
           << "': " << error.message() << "\n";
    return false;
  }
  bool ok = true;
  for (const auto &file : files) {
    SmallString<256> path(temporary);
    sys::path::append(path, file.first);
    SmallString<256> directory(path);
    sys::path::remove_filename(directory);
    if (auto error = sys::fs::create_directories(directory)) {
      errs() << "cannot create Unity bundle resource directory '" << directory
             << "': " << error.message() << "\n";
      ok = false;
      break;
    }
    std::error_code error;
    ToolOutputFile out(path, error, sys::fs::OF_None);
    if (error) {
      errs() << "cannot open Unity bundle resource '" << path
             << "': " << error.message() << "\n";
      ok = false;
      break;
    }
    out.os() << file.second;
    out.os().flush();
    if (out.os().has_error()) {
      errs() << "cannot write Unity bundle resource '" << path << "'\n";
      out.os().clear_error();
      ok = false;
      break;
    }
    out.keep();
  }
  for (const auto &file : externalFiles) {
    if (!ok)
      break;
    SmallString<256> path(temporary);
    sys::path::append(path, file.first);
    SmallString<256> directory(path);
    sys::path::remove_filename(directory);
    if (auto error = sys::fs::create_directories(directory)) {
      errs() << "cannot create Unity bundle resource directory '" << directory
             << "': " << error.message() << "\n";
      ok = false;
      break;
    }
    if (auto error = sys::fs::copy_file(file.second, path)) {
      errs() << "cannot copy Unity bundle resource '" << file.second << "' to '"
             << path << "': " << error.message() << "\n";
      ok = false;
      break;
    }
  }
  if (ok) {
    if (auto error = sys::fs::rename(temporary, outputDirectory)) {
      errs() << "cannot publish Unity bundle '" << temporary << "' to '"
             << outputDirectory << "': " << error.message() << "\n";
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

bool dottedIdentifier(StringRef value) {
  SmallVector<StringRef> components;
  value.split(components, '.');
  return !components.empty() &&
         llvm::all_of(components, [](StringRef component) {
           return identifier(component);
         });
}

/// The texel format a config requests for its Device buffers (key `key`):
/// `rgba8_unorm` (the default, one word per texel) or `rgba32_float`. A
/// diagnostic names the setting as the user wrote it, `setting`.
FailureOr<std::string> configTexelFormat(const json::Object &config,
                                         StringRef key, StringRef setting) {
  const json::Value *value = config.get(key);
  if (!value)
    return std::string("rgba8_unorm");
  auto format = value->getAsString();
  if (!format || !::vrc::vrcgpu::isGatherTexelFormat(*format)) {
    errs() << setting << " must be \"rgba8_unorm\" or \"rgba32_float\"\n";
    return failure();
  }
  return format->str();
}

void setModuleTexelFormat(ModuleOp module, StringRef format) {
  if (format != "rgba8_unorm")
    module->setAttr("vrc.texel_format",
                    StringAttr::get(module.getContext(), format));
}

struct UnityBuffer {
  int64_t id;
  int64_t allocationBytes;
  int64_t logicalBytes;
  int64_t width;
  int64_t height;
  std::string name;
  std::string heapSlot;
  std::string elementType;
  std::string publicAccess;
  std::optional<int64_t> publicIndex;
  /// Where a public buffer comes from (vrc-describe-canonical-unity-buffers);
  /// copied into the manifest as is.
  std::optional<json::Value> source;
};

FailureOr<SmallVector<UnityBuffer>>
parseUnityBuffers(const json::Object &config) {
  SmallVector<UnityBuffer> result;
  auto *buffers = config.getArray("buffers");
  if (!buffers)
    return result;
  std::optional<std::pair<int64_t, int64_t>> dispatchShape;
  const json::Object *shapeKernel = nullptr;
  if (auto *kernels = config.getArray("device_kernels");
      kernels && !kernels->empty())
    shapeKernel = (*kernels)[0].getAsObject();
  if (auto *kernel = shapeKernel)
    if (auto *global = kernel->getArray("global_size");
        global && global->size() == 3) {
      auto width = (*global)[0].getAsInteger();
      auto height = (*global)[1].getAsInteger();
      auto depth = (*global)[2].getAsInteger();
      if (width && height && depth && *width > 0 && *width <= 8192 &&
          *height > 0 && *height <= 8192 && *depth == 1)
        dispatchShape = std::make_pair(*width, *height);
    }
  std::set<int64_t> ids;
  std::set<std::string> names;
  for (const json::Value &value : *buffers) {
    auto *entry = value.getAsObject();
    auto id = entry ? entry->getInteger("id") : std::nullopt;
    auto name = entry ? entry->getString("name") : std::nullopt;
    auto bytes = entry ? entry->getInteger("byte_size") : std::nullopt;
    auto logicalBytes =
        entry ? entry->getInteger("logical_byte_size") : std::nullopt;
    auto elementType = entry ? entry->getString("element_type") : std::nullopt;
    auto publicAccess =
        entry ? entry->getString("public_access") : std::nullopt;
    auto publicIndex = entry ? entry->getInteger("public_index") : std::nullopt;
    constexpr int64_t maximumBytes = 8192LL * 8192LL * 4LL;
    // The access of a public buffer is checked where the buffer is
    // described (vrc-describe-canonical-unity-buffers, or the ONNX plan
    // buffers derived with a fixed access).
    bool semanticBinding = elementType && publicAccess &&
                           (*elementType == "float32" ||
                            *elementType == "int32" || *elementType == "int64");
    if (!id || *id <= 0 || *id > INT32_MAX || !name || !identifier(*name) ||
        !bytes || *bytes <= 0 || *bytes > maximumBytes ||
        (logicalBytes && (*logicalBytes < 0 || *logicalBytes > *bytes)) ||
        !ids.insert(*id).second || !names.insert(name->str()).second ||
        ((elementType || publicAccess || publicIndex) && !semanticBinding) ||
        (publicIndex && *publicIndex < 0)) {
      errs() << "invalid or duplicate Unity Device buffer\n";
      return failure();
    }
    int64_t pixels = (*bytes + 3) / 4;
    bool shapeFits =
        dispatchShape && dispatchShape->first * dispatchShape->second >= pixels;
    int64_t width =
        shapeFits ? dispatchShape->first : std::min<int64_t>(pixels, 8192);
    int64_t height =
        shapeFits ? dispatchShape->second : (pixels + width - 1) / width;
    // Word texels: a one-row buffer is padded to whole texels, as its
    // kernels' one-row targets are.
    if (auto format = config.getString("device_texel_format");
        format && height == 1)
      if (unsigned texelWords = ::vrc::vrcgpu::gatherTexelWords(*format))
        width = llvm::alignTo(width, texelWords);
    result.push_back({*id, *bytes, logicalBytes.value_or(*bytes), width, height,
                      name->str(), "__udoncl_buffer_" + std::to_string(*id),
                      elementType ? elementType->str() : "",
                      publicAccess ? publicAccess->str() : "", publicIndex,
                      entry->get("source")
                          ? std::optional<json::Value>(*entry->get("source"))
                          : std::nullopt});
  }
  return result;
}

LogicalResult reconcileStaticUnityBuffers(OwningOpRef<ModuleOp> &ownedModule,
                                          ArrayRef<UnityBuffer> buffers) {
  ModuleOp module = *ownedModule;
  bool hasInventory = module->hasAttr("vrc.static_device_resource_inventory");
  bool hasAliases = module->hasAttr("vrc.static_public_resource_aliases");
  if (!hasInventory && !hasAliases)
    return success();
  if (hasInventory != hasAliases) {
    module.emitError(
        "static Unity buffer reconciliation metadata is incomplete");
    return failure();
  }
  Builder builder(module.getContext());
  SmallVector<mlir::Attribute> entries;
  entries.reserve(buffers.size());
  for (const UnityBuffer &buffer : buffers) {
    SmallVector<NamedAttribute> fields{
        builder.getNamedAttr("buffer_id", builder.getI32IntegerAttr(buffer.id)),
        builder.getNamedAttr("capacity_bytes",
                             builder.getI64IntegerAttr(buffer.allocationBytes)),
        builder.getNamedAttr("heap_slot",
                             builder.getStringAttr(buffer.heapSlot)),
    };
    if (!buffer.publicAccess.empty())
      fields.push_back(builder.getNamedAttr(
          "public_access", builder.getStringAttr(buffer.publicAccess)));
    if (buffer.publicIndex)
      fields.push_back(builder.getNamedAttr(
          "public_index", builder.getI32IntegerAttr(*buffer.publicIndex)));
    entries.push_back(builder.getDictionaryAttr(fields));
  }
  module->setAttr(
      "vrc.unity_configured_buffers",
      builder.getDictionaryAttr({
          builder.getNamedAttr("buffers", builder.getArrayAttr(entries)),
          builder.getNamedAttr("schema_version", builder.getI32IntegerAttr(1)),
      }));
  PassManager reconciliation(module.getContext());
  reconciliation.addPass(::vrc::createReconcileStaticUnityBuffersPass());
  reconciliation.addPass(::vrc::createPlanStaticUnityBufferLayoutPass());
  if (failed(reconciliation.run(module)))
    return failure();

  OwningOpRef<ModuleOp> preview = cast<ModuleOp>(module.clone());
  PassManager readiness(module.getContext());
  ::vrc::buildStaticUnityResourceReadinessPipeline(readiness);
  if (failed(readiness.run(*preview)))
    return failure();
  auto physicalPlan =
      (*preview)->getAttr("vrc.static_physical_allocation_specialization_plan");
  if (!physicalPlan)
    return module.emitError(
        "static physical allocation preview produced no plan");
  module->setAttr("vrc.static_physical_allocation_specialization_plan",
                  physicalPlan);
  for (const char *name : {
           "vrc.static_physical_allocation_specialization",
           "vrc.static_physical_allocation_materialization",
           "vrc.static_physical_resolve_materialization",
           "vrc.fixed_physical_release_materialization",
       })
    if (mlir::Attribute value = (*preview)->getAttr(name))
      module->setAttr(name, value);
  auto logicalMaterialization =
      (*preview)->getAttr("vrc.static_logical_buffer_materialization");
  if (logicalMaterialization)
    module->setAttr("vrc.static_logical_buffer_materialization",
                    logicalMaterialization);
  auto result = (*preview)->getAttrOfType<DictionaryAttr>(
      "vrc.static_unity_resource_lowering_readiness");
  if (!result)
    return module.emitError(
        "static Unity resource readiness preview produced no result");
  auto poolReplacementReady = result.getAs<BoolAttr>("pool_replacement_ready");
  if (!poolReplacementReady)
    return module.emitError(
        "static Unity resource readiness result has no replacement decision");
  SmallVector<NamedAttribute> fields(result.begin(), result.end());
  bool apply = poolReplacementReady.getValue();
  fields.push_back(builder.getNamedAttr("applied", builder.getBoolAttr(apply)));
  fields.push_back(
      builder.getNamedAttr("preview", builder.getBoolAttr(!apply)));
  auto readinessResult = builder.getDictionaryAttr(fields);
  if (apply) {
    (*preview)->setAttr("vrc.static_unity_resource_lowering_readiness",
                        readinessResult);
    ownedModule = std::move(preview);
  } else {
    module->setAttr("vrc.static_unity_resource_lowering_readiness",
                    readinessResult);
  }
  return success();
}

std::optional<json::Value> mlirAttributeToJson(mlir::Attribute attribute);

/// The Device buffers of a Host whose static Unity resource lowering is
/// applied, as vrc-describe-canonical-unity-buffers describes them, replace
/// the `buffers` of `config`.
LogicalResult materializeCanonicalUnityBufferConfig(ModuleOp module,
                                                    json::Object &config) {
  PassManager describe(module.getContext());
  describe.addPass(::vrc::createDescribeCanonicalUnityBuffersPass());
  if (failed(describe.run(module)))
    return failure();
  auto buffers =
      module->getAttrOfType<ArrayAttr>("vrc.canonical_unity_buffers");
  if (!buffers)
    return success();
  // The description is an input of the artifact, not Host IR.
  module->removeAttr("vrc.canonical_unity_buffers");
  auto value = mlirAttributeToJson(buffers);
  if (!value) {
    module.emitError("canonical Unity buffers are not serializable");
    return failure();
  }
  config["buffers"] = std::move(*value);
  return success();
}

/// The Device buffers of an ONNX Host, as vrc-derive-onnx-unity-buffers
/// derives them from its execution plan and binding, replace the `buffers`
/// of `config`. A Host without ONNX binding metadata leaves `config`
/// unchanged.
LogicalResult deriveOnnxUnityBufferConfig(ModuleOp module,
                                          json::Object &config) {
  PassManager derive(module.getContext());
  derive.enableVerifier(false);
  derive.addPass(::vrc::createDeriveONNXUnityBuffersPass());
  if (failed(derive.run(module)))
    return failure();
  auto buffers = module->getAttrOfType<ArrayAttr>("vrc.onnx_unity_buffers");
  if (!buffers)
    return success();
  // The description is an input of the artifact, not Host IR.
  module->removeAttr("vrc.onnx_unity_buffers");
  auto value = mlirAttributeToJson(buffers);
  if (!value) {
    module.emitError("ONNX Unity buffers are not serializable");
    return failure();
  }
  config["buffers"] = std::move(*value);
  return success();
}

/// The `device_kernels` entries of the library and user Device kernels the
/// Unity Host dispatches, as vrc-describe-unity-device-kernels derives them
/// from their dispatch sites and library contracts.
FailureOr<json::Array>
describeUnityDeviceKernels(ModuleOp module,
                           ArrayRef<int64_t> synthesizedKernelIds,
                           const std::map<int64_t, std::string> &userKernels) {
  ::vrc::DescribeUnityDeviceKernelsPassOptions options;
  options.synthesizedKernelIds.assign(synthesizedKernelIds.begin(),
                                      synthesizedKernelIds.end());
  for (const auto &[id, name] : userKernels) {
    options.userKernelIds.push_back(id);
    options.userKernelNames.push_back(name);
  }
  PassManager describe(module.getContext());
  describe.enableVerifier(false);
  describe.addPass(::vrc::createDescribeUnityDeviceKernelsPass(options));
  if (failed(describe.run(module)))
    return failure();
  auto kernels = module->getAttrOfType<ArrayAttr>("vrc.unity_device_kernels");
  if (!kernels)
    return failure();
  // The description is an input of the artifact, not Host IR.
  module->removeAttr("vrc.unity_device_kernels");
  auto value = mlirAttributeToJson(kernels);
  if (!value || !value->getAsArray()) {
    module.emitError("Unity Device kernels are not serializable");
    return failure();
  }
  return std::move(*value->getAsArray());
}

FailureOr<SmallVector<const json::Object *>>
parseUnityDeviceKernels(const json::Object &config) {
  SmallVector<const json::Object *> result;
  if (auto *kernels = config.getArray("device_kernels"))
    for (const json::Value &value : *kernels) {
      auto *kernel = value.getAsObject();
      if (!kernel) {
        errs() << "derived Unity Device kernels must be objects\n";
        return failure();
      }
      result.push_back(kernel);
    }
  std::set<int64_t> ids;
  std::set<std::string> names;
  for (const json::Object *kernel : result) {
    auto id = kernel->getInteger("kernel_id");
    auto name = kernel->getString("kernel");
    if (!id || *id <= 0 || !name || !identifier(*name) ||
        !ids.insert(*id).second || !names.insert(name->str()).second) {
      errs() << "invalid or duplicate Unity Device kernel\n";
      return failure();
    }
  }
  return result;
}

void appendUnityBufferPoolSlot(ModuleOp module, ArrayRef<UnityBuffer> buffers) {
  if (buffers.empty())
    return;
  auto data = module->getAttrOfType<ArrayAttr>("udonasm.data");
  SmallVector<mlir::Attribute> entries(data.begin(), data.end());
  OpBuilder builder(module.getContext());
  bool exists = llvm::any_of(entries, [&](mlir::Attribute attribute) {
    auto entry = dyn_cast<DictionaryAttr>(attribute);
    auto name = entry ? entry.getAs<StringAttr>("name") : StringAttr();
    return name && name.getValue() == "__udoncl_buffer_pool";
  });
  if (!exists)
    entries.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("name",
                              builder.getStringAttr("__udoncl_buffer_pool")),
         builder.getNamedAttr(
             "type", builder.getStringAttr("UnityEngineRenderTextureArray")),
         builder.getNamedAttr("initial", builder.getStringAttr("null")),
         builder.getNamedAttr("public", builder.getUnitAttr())}));
  module->setAttr("udonasm.data", builder.getArrayAttr(entries));
}

struct UnityBufferBinding {
  bool usesCopy = false;
};

FailureOr<UnityBufferBinding>
bindUnityBufferOperations(ModuleOp module, ArrayRef<UnityBuffer> buffers,
                          ArrayRef<int64_t> configuredKernelIds,
                          ArrayAttr staticTensors = {}) {
  SmallVector<::vrc::vrc::BufferAllocOp> allocations;
  SmallVector<::vrc::vrc::AssignedBufferOp> assignedBuffers;
  SmallVector<::vrc::vrc::BufferCopyOp> copies;
  SmallVector<::vrc::vrc::BufferFillOp> fills;
  SmallVector<::vrc::vrc::BufferUploadOp> uploads;
  SmallVector<::vrc::vrc::DispatchOp> dispatches;
  module.walk([&](::vrc::vrc::BufferAllocOp operation) {
    allocations.push_back(operation);
  });
  module.walk([&](::vrc::vrc::AssignedBufferOp operation) {
    assignedBuffers.push_back(operation);
  });
  module.walk(
      [&](::vrc::vrc::BufferCopyOp operation) { copies.push_back(operation); });
  module.walk(
      [&](::vrc::vrc::BufferFillOp operation) { fills.push_back(operation); });
  module.walk([&](::vrc::vrc::BufferUploadOp operation) {
    uploads.push_back(operation);
  });
  module.walk([&](::vrc::vrc::DispatchOp operation) {
    dispatches.push_back(operation);
  });
  // Multiple functions may acquire the same statically configured Unity
  // buffer. Count constant logical identities rather than allocation ops so
  // those acquisitions do not look like distinct resources. Dynamic logical
  // ids select from this same configured pool at runtime and therefore do not
  // imply an additional physical buffer per call site.
  std::set<int64_t> constantAllocationIds;
  for (auto allocation : allocations) {
    auto logicalId =
        allocation.getLogicalId().getDefiningOp<arith::ConstantIntOp>();
    if (logicalId)
      constantAllocationIds.insert(logicalId.value());
  }
  size_t requiredBuffers = constantAllocationIds.size();
  if (requiredBuffers > buffers.size()) {
    module.emitError() << "Unity library requires " << requiredBuffers
                       << " distinct buffer identities but only "
                       << buffers.size() << " configured Device buffers";
    return failure();
  }
  OpBuilder builder(module.getContext());
  SmallVector<mlir::Attribute> poolEntries;
  for (const UnityBuffer &buffer : buffers)
    poolEntries.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("id", builder.getI32IntegerAttr(buffer.id)),
         builder.getNamedAttr("logical_bytes", builder.getI32IntegerAttr(
                                                   buffer.allocationBytes)),
         builder.getNamedAttr("heap_slot",
                              builder.getStringAttr(buffer.heapSlot))}));
  module->setAttr("vrc.unity_buffer_pool", builder.getArrayAttr(poolEntries));
  llvm::DenseMap<Operation *, const UnityBuffer *> bound;
  llvm::DenseMap<int32_t, const UnityBuffer *> assignedResources;
  for (auto assigned : assignedBuffers) {
    const UnityBuffer *buffer = nullptr;
    for (const UnityBuffer &candidate : buffers)
      if (candidate.id == assigned.getTextureId()) {
        buffer = &candidate;
        break;
      }
    if (!buffer || buffer->allocationBytes != assigned.getCapacityBytes() ||
        assigned.getFormat() != "rgba8_unorm")
      return assigned.emitError(
          "assigned Unity buffer must match its configured texture id, "
          "capacity, and format");
    assigned->setAttr("vrc.unity_heap_slot",
                      builder.getStringAttr(buffer->heapSlot));
    bound.try_emplace(assigned.getOperation(), buffer);
    auto [resource, inserted] = assignedResources.try_emplace(
        static_cast<int32_t>(assigned.getResourceId()), buffer);
    if (!inserted && resource->second != buffer)
      return assigned.emitError(
          "one static resource cannot name multiple Unity buffers");
  }
  bool staticLoweringApplied = false;
  if (auto readiness = module->getAttrOfType<DictionaryAttr>(
          "vrc.static_unity_resource_lowering_readiness")) {
    auto version = readiness.getAs<IntegerAttr>("schema_version");
    auto applied = readiness.getAs<BoolAttr>("applied");
    auto poolReady = readiness.getAs<BoolAttr>("pool_replacement_ready");
    if (applied && applied.getValue()) {
      if (!version || version.getInt() != 1 || !poolReady ||
          !poolReady.getValue())
        return module.emitError(
            "applied static Unity lowering has invalid readiness metadata");
      staticLoweringApplied = true;
    }
  }
  for (auto allocation : allocations) {
    auto logicalId =
        allocation.getLogicalId().getDefiningOp<arith::ConstantIntOp>();
    auto constant =
        allocation.getByteSize().getDefiningOp<arith::ConstantIntOp>();
    const UnityBuffer *buffer = nullptr;
    if (logicalId)
      for (const UnityBuffer &candidate : buffers)
        if (candidate.id == logicalId.value()) {
          buffer = &candidate;
          break;
        }
    if (!logicalId)
      continue;
    if (!buffer || !constant || constant.value() != buffer->allocationBytes)
      return allocation.emitError(
          "constant Unity buffer allocation must match its configured logical "
          "id and byte size");
    allocation->setAttr("vrc.unity_buffer_id",
                        builder.getI32IntegerAttr(buffer->id));
    allocation->setAttr("vrc.unity_heap_slot",
                        builder.getStringAttr(buffer->heapSlot));
    allocation->setAttr("vrc.unity_logical_bytes",
                        builder.getI32IntegerAttr(buffer->allocationBytes));
    bound.try_emplace(allocation.getOperation(), buffer);
  }
  auto resolveBoundBuffer = [&](mlir::Value value) -> const UnityBuffer * {
    while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1)
        return nullptr;
      value = cast.getInputs().front();
    }
    Operation *definition = value.getDefiningOp();
    auto found = definition ? bound.find(definition) : bound.end();
    return found == bound.end() ? nullptr : found->second;
  };
  auto validateOrSetCopySlot = [&](Operation *operation, StringRef name,
                                   StringRef value) -> LogicalResult {
    if (auto existing = operation->getAttrOfType<StringAttr>(name)) {
      if (existing.getValue() != value) {
        operation->emitError()
            << name << " contradicts the statically bound Unity buffer";
        return failure();
      }
      return success();
    }
    operation->setAttr(name, builder.getStringAttr(value));
    return success();
  };
  for (auto copy : copies) {
    auto sourceOffset =
        copy.getSourceOffset().getDefiningOp<arith::ConstantIntOp>();
    auto destinationOffset =
        copy.getDestinationOffset().getDefiningOp<arith::ConstantIntOp>();
    auto byteSize = copy.getByteSize().getDefiningOp<arith::ConstantIntOp>();
    const UnityBuffer *sourceBuffer = resolveBoundBuffer(copy.getSource());
    const UnityBuffer *destinationBuffer =
        resolveBoundBuffer(copy.getDestination());
    // A copy of two whole, equally shaped, statically bound surfaces is one
    // whole-surface Blit. Every other copy is a word-range Blit through the
    // transfer material's range-copy mode, which rewrites only the
    // destination words; its buffers may be bound at run time.
    bool wholeSurface =
        sourceBuffer && destinationBuffer && sourceOffset &&
        sourceOffset.value() == 0 && destinationOffset &&
        destinationOffset.value() == 0 && byteSize &&
        byteSize.value() == sourceBuffer->allocationBytes &&
        byteSize.value() == destinationBuffer->allocationBytes &&
        sourceBuffer->width == destinationBuffer->width &&
        sourceBuffer->height == destinationBuffer->height;
    bool unaligned = false;
    for (auto constant : {sourceOffset, destinationOffset, byteSize})
      unaligned |= constant && constant.value() % 4 != 0;
    if ((sourceBuffer && sourceBuffer == destinationBuffer) || unaligned ||
        (sourceBuffer && sourceOffset && byteSize &&
         sourceOffset.value() + byteSize.value() >
             sourceBuffer->allocationBytes) ||
        (destinationBuffer && destinationOffset && byteSize &&
         destinationOffset.value() + byteSize.value() >
             destinationBuffer->allocationBytes)) {
      copy.emitError("Unity CopyBuffer requires distinct buffers and a "
                     "word-aligned range inside both of them");
      return failure();
    }
    copy->setAttr("vrc.unity_copy_material_slot",
                  builder.getStringAttr("__udoncl_buffer_copy_material"));
    if (!wholeSurface)
      copy->setAttr("vrc.unity_copy_word_range", builder.getUnitAttr());
    if ((sourceBuffer &&
         failed(validateOrSetCopySlot(copy, "vrc.unity_copy_source_heap_slot",
                                      sourceBuffer->heapSlot))) ||
        (destinationBuffer && failed(validateOrSetCopySlot(
                                  copy, "vrc.unity_copy_destination_heap_slot",
                                  destinationBuffer->heapSlot))))
      return failure();
  }
  // Fills and small uploads write their words through the same transfer
  // material as copies; a fill of a statically bound buffer also loads its
  // destination from the buffer's heap slot.
  constexpr StringLiteral transferMaterial = "__udoncl_buffer_copy_material";
  for (auto fill : fills) {
    fill->setAttr("vrc.unity_copy_material_slot",
                  builder.getStringAttr(transferMaterial));
    if (const UnityBuffer *destination =
            resolveBoundBuffer(fill.getDestination()))
      if (failed(validateOrSetCopySlot(fill,
                                       "vrc.unity_fill_destination_heap_slot",
                                       destination->heapSlot)))
        return failure();
  }
  // A small upload is patched in place only when the program has a Unity
  // buffer pool to select its destination from; otherwise it stays a binding
  // command.
  bool patchesUploads = !uploads.empty() && !buffers.empty();
  if (patchesUploads)
    for (auto upload : uploads)
      upload->setAttr("vrc.unity_copy_material_slot",
                      builder.getStringAttr(transferMaterial));
  UnityBufferBinding result;
  result.usesCopy = !copies.empty() || !fills.empty() || patchesUploads;
  auto staticTensorMatches = [&](int64_t id, int64_t bytes) {
    if (!staticTensors)
      return false;
    for (mlir::Attribute attribute : staticTensors) {
      auto tensor = dyn_cast<DictionaryAttr>(attribute);
      auto tensorId = tensor ? tensor.getAs<IntegerAttr>("id") : IntegerAttr();
      auto byteLength =
          tensor ? tensor.getAs<IntegerAttr>("byte_length") : IntegerAttr();
      if (tensorId && tensorId.getInt() == id)
        return byteLength && byteLength.getInt() == bytes;
    }
    return false;
  };
  auto staticTensorExists = [&](int64_t id) {
    if (!staticTensors)
      return false;
    return llvm::any_of(staticTensors, [&](mlir::Attribute attribute) {
      auto tensor = dyn_cast<DictionaryAttr>(attribute);
      auto tensorId = tensor ? tensor.getAs<IntegerAttr>("id") : IntegerAttr();
      return tensorId && tensorId.getInt() == id;
    });
  };
  for (auto dispatch : dispatches) {
    auto descriptor =
        dispatch.getDesc().getDefiningOp<::vrc::vrc::DispatchDescCreateOp>();
    SmallVector<StringRef> scalarKinds;
    for (mlir::Attribute value : dispatch.getArgumentKinds()) {
      auto kind = dyn_cast<StringAttr>(value);
      if (kind && kind.getValue() != "buffer")
        scalarKinds.push_back(kind.getValue());
    }
    bool descriptorMatches =
        descriptor && descriptor.getKernelId() == dispatch.getKernelId() &&
        descriptor.getScalarKinds().size() == scalarKinds.size();
    if (descriptorMatches)
      for (auto [expected, actual] :
           llvm::zip(scalarKinds, descriptor.getScalarKinds())) {
        auto kind = dyn_cast<StringAttr>(actual);
        descriptorMatches &= kind && kind.getValue() == expected;
      }
    size_t bufferKindCount =
        llvm::count_if(dispatch.getArgumentKinds(), [](mlir::Attribute value) {
          auto kind = dyn_cast<StringAttr>(value);
          return kind && kind.getValue() == "buffer";
        });
    bool resourcesMatch = dispatch.getResources().size() == bufferKindCount;
    llvm::DenseMap<unsigned, const UnityBuffer *> snapshotBuffers;
    llvm::DenseMap<unsigned, int64_t> snapshotStaticTensors;
    auto snapshot = dispatch->getAttrOfType<DictionaryAttr>(
        "vrc.static_kernel_binding_snapshot");
    if (staticLoweringApplied && !snapshot)
      return dispatch.emitError(
          "applied static Unity dispatch has no binding snapshot");
    if (snapshot) {
      auto version =
          snapshot ? snapshot.getAs<IntegerAttr>("schema_version") : nullptr;
      auto kernelId =
          snapshot ? snapshot.getAs<IntegerAttr>("kernel_id") : nullptr;
      auto resources =
          snapshot ? snapshot.getAs<ArrayAttr>("resources") : nullptr;
      if (!version || version.getInt() != 1 || !kernelId ||
          kernelId.getInt() != dispatch.getKernelId() || !resources)
        return dispatch.emitError(
            "applied static Unity dispatch has an invalid binding snapshot");
      llvm::DenseSet<unsigned> seen;
      for (mlir::Attribute attribute : resources) {
        auto resource = dyn_cast<DictionaryAttr>(attribute);
        auto kind =
            resource ? resource.getAs<StringAttr>("kind") : StringAttr();
        auto index = resource ? resource.getAs<IntegerAttr>("operand_index")
                              : IntegerAttr();
        auto id = resource ? resource.getAs<IntegerAttr>("resource_id")
                           : IntegerAttr();
        auto access =
            resource ? resource.getAs<StringAttr>("access") : StringAttr();
        if (!kind || !index || index.getInt() < 0 ||
            index.getInt() >= dispatch.getResources().size() || !id ||
            id.getInt() < 0 || !access ||
            !seen.insert(static_cast<unsigned>(index.getInt())).second)
          return dispatch.emitError(
              "applied static Unity dispatch snapshot is malformed");
        auto expectedAccess = dyn_cast<StringAttr>(
            dispatch.getResourceAccesses()[index.getInt()]);
        if (!expectedAccess || expectedAccess != access)
          return dispatch.emitError(
              "applied static Unity dispatch access is inconsistent");
        unsigned operandIndex = static_cast<unsigned>(index.getInt());
        if (kind.getValue() == "allocation") {
          auto found =
              assignedResources.find(static_cast<int32_t>(id.getInt()));
          if (found != assignedResources.end())
            snapshotBuffers[operandIndex] = found->second;
          else if (staticLoweringApplied)
            return dispatch.emitError(
                "static dispatch allocation has no bound Unity buffer");
        } else if (kind.getValue() == "static_tensor") {
          if (!staticTensorExists(id.getInt()))
            return dispatch.emitError(
                "static dispatch tensor is absent from the artifact");
          snapshotStaticTensors[operandIndex] = id.getInt();
        } else if (staticLoweringApplied) {
          return dispatch.emitError(
              "applied static Unity dispatch retains a dynamic resource");
        }
      }
      if (seen.size() != dispatch.getResources().size())
        return dispatch.emitError(
            "applied static Unity dispatch snapshot is incomplete");
    }
    SmallVector<mlir::Attribute> resourceHeapSlots;
    SmallVector<mlir::Attribute> resourceStaticTensorIds;
    for (auto [resourceIndex, resource] :
         llvm::enumerate(dispatch.getResources())) {
      auto staticBuffer = resource.getDefiningOp<::vrc::vrc::StaticBufferOp>();
      const UnityBuffer *boundBuffer = resolveBoundBuffer(resource);
      auto snapshotBuffer =
          snapshotBuffers.find(static_cast<unsigned>(resourceIndex));
      if (snapshotBuffer != snapshotBuffers.end()) {
        if (boundBuffer && boundBuffer != snapshotBuffer->second)
          return dispatch.emitError(
              "static dispatch SSA resource disagrees with its snapshot");
        boundBuffer = snapshotBuffer->second;
      }
      const UnityBuffer *constantBuffer = nullptr;
      mlir::Value token = resource;
      while (auto cast = token.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (cast.getInputs().size() != 1)
          break;
        token = cast.getInputs().front();
      }
      if (!staticBuffer)
        staticBuffer = token.getDefiningOp<::vrc::vrc::StaticBufferOp>();
      if (auto logicalId = token.getDefiningOp<arith::ConstantIntOp>())
        for (const UnityBuffer &candidate : buffers)
          if (candidate.id == logicalId.value()) {
            constantBuffer = &candidate;
            break;
          }
      if (staticBuffer) {
        resourcesMatch &= staticTensorMatches(staticBuffer.getResourceId(),
                                              staticBuffer.getByteSize());
        resourceHeapSlots.push_back(builder.getStringAttr(""));
        resourceStaticTensorIds.push_back(
            builder.getI32IntegerAttr(staticBuffer.getResourceId()));
      } else if (auto found = snapshotStaticTensors.find(
                     static_cast<unsigned>(resourceIndex));
                 found != snapshotStaticTensors.end()) {
        resourceHeapSlots.push_back(builder.getStringAttr(""));
        resourceStaticTensorIds.push_back(
            builder.getI32IntegerAttr(found->second));
      } else if (boundBuffer) {
        resourceHeapSlots.push_back(
            builder.getStringAttr(boundBuffer->heapSlot));
        resourceStaticTensorIds.push_back(builder.getI32IntegerAttr(-1));
      } else if (constantBuffer) {
        resourceHeapSlots.push_back(
            builder.getStringAttr(constantBuffer->heapSlot));
        resourceStaticTensorIds.push_back(builder.getI32IntegerAttr(-1));
      } else {
        resourcesMatch &= !buffers.empty();
        resourceHeapSlots.push_back(builder.getStringAttr(""));
        resourceStaticTensorIds.push_back(builder.getI32IntegerAttr(-1));
      }
    }
    std::optional<size_t> writeResource;
    for (auto [index, attribute] :
         llvm::enumerate(dispatch.getResourceAccesses())) {
      auto access = dyn_cast<StringAttr>(attribute);
      if (access && access.getValue() == "write")
        writeResource = index;
    }
    if (writeResource) {
      auto writeSlot = dyn_cast<StringAttr>(resourceHeapSlots[*writeResource]);
      for (auto [index, attribute] :
           llvm::enumerate(dispatch.getResourceAccesses())) {
        auto access = dyn_cast<StringAttr>(attribute);
        auto readSlot = dyn_cast<StringAttr>(resourceHeapSlots[index]);
        if (index != *writeResource && access && access.getValue() == "read" &&
            writeSlot && readSlot && !writeSlot.getValue().empty() &&
            writeSlot == readSlot) {
          dispatch.emitError()
              << "kernel " << dispatch.getKernelId() << " resource " << index
              << " reads Unity heap slot " << readSlot.getValue()
              << " also written by resource " << *writeResource;
          return failure();
        }
      }
    }
    if (!llvm::is_contained(configuredKernelIds, dispatch.getKernelId()) ||
        !descriptorMatches || dispatch.getArgumentKinds().empty() ||
        !isa<StringAttr>(*dispatch.getArgumentKinds().begin()) ||
        cast<StringAttr>(*dispatch.getArgumentKinds().begin()).getValue() !=
            "buffer" ||
        !resourcesMatch) {
      dispatch.emitError()
          << "Unity dispatch requires the configured kernel, logical buffers "
             "from the fixed pool, and a copied f32/i32 scalar descriptor "
          << "(kernel_id=" << dispatch.getKernelId() << ", configured="
          << llvm::is_contained(configuredKernelIds, dispatch.getKernelId())
          << ", descriptor=" << descriptorMatches
          << ", resources=" << resourcesMatch << ")";
      return failure();
    }
    int32_t kernelId = static_cast<int32_t>(dispatch.getKernelId());
    dispatch->setAttr("vrc.unity_dispatch_material_slot",
                      builder.getStringAttr("__udoncl_kernel_material_" +
                                            std::to_string(kernelId)));
    dispatch->setAttr("vrc.unity_resource_heap_slots",
                      builder.getArrayAttr(resourceHeapSlots));
    dispatch->setAttr("vrc.unity_static_tensor_ids",
                      builder.getArrayAttr(resourceStaticTensorIds));
    if (!writeResource) {
      dispatch.emitError("Unity dispatch requires one write resource");
      return failure();
    }
    dispatch->setAttr(
        "vrc.unity_output_resource_index",
        builder.getI32IntegerAttr(static_cast<int32_t>(*writeResource)));
  }
  return result;
}

void appendUnityCopyMaterialSlot(ModuleOp module) {
  auto data = module->getAttrOfType<ArrayAttr>("udonasm.data");
  SmallVector<mlir::Attribute> entries(data.begin(), data.end());
  OpBuilder builder(module.getContext());
  if (llvm::any_of(entries, [](mlir::Attribute attribute) {
        auto entry = dyn_cast<DictionaryAttr>(attribute);
        auto name = entry ? entry.getAs<StringAttr>("name") : StringAttr();
        return name && name.getValue() == "__udoncl_buffer_copy_material";
      }))
    return;
  entries.push_back(builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "name", builder.getStringAttr("__udoncl_buffer_copy_material")),
       builder.getNamedAttr("type",
                            builder.getStringAttr("UnityEngineMaterial")),
       builder.getNamedAttr("initial", builder.getStringAttr("null")),
       builder.getNamedAttr("public", builder.getUnitAttr())}));
  module->setAttr("udonasm.data", builder.getArrayAttr(entries));
}

/// Lists in the manifest every dispatch material the Udon program declares,
/// with the kernel whose shader it renders: the Unity importer creates one
/// material per entry and binds it to the program's public slot.
LogicalResult appendUnityDispatchMaterialManifest(ModuleOp module,
                                                  json::Object &manifest) {
  llvm::StringSet<> declared;
  if (auto data = module->getAttrOfType<ArrayAttr>("udonasm.data"))
    for (mlir::Attribute attribute : data)
      if (auto entry = dyn_cast<DictionaryAttr>(attribute))
        if (auto name = entry.getAs<StringAttr>("name"))
          declared.insert(name.getValue());
  json::Array materials;
  if (auto entries =
          module->getAttrOfType<ArrayAttr>("vrc.unity_dispatch_materials"))
    for (mlir::Attribute attribute : entries) {
      auto entry = dyn_cast<DictionaryAttr>(attribute);
      auto slot = entry ? entry.getAs<StringAttr>("slot") : StringAttr();
      auto kernelId =
          entry ? entry.getAs<IntegerAttr>("kernel_id") : IntegerAttr();
      if (!slot || !kernelId) {
        errs() << "malformed vrc.unity_dispatch_materials entry\n";
        return failure();
      }
      // A dispatch the scalar lowering dropped as unreachable declares no
      // material.
      if (!declared.contains(slot.getValue()))
        continue;
      materials.push_back(json::Object{{"slot", slot.getValue().str()},
                                       {"kernel_id", kernelId.getInt()}});
    }
  manifest["dispatch_materials"] = std::move(materials);
  return success();
}

/// The Device buffers' textures. `width` and `height` are the texture extent
/// in texels; a texel holds `gatherTexelWords(format)` words, so a row of
/// `width` texels holds width * words-per-texel consecutive words.
json::Array unityBufferManifest(ArrayRef<UnityBuffer> buffers,
                                StringRef texelFormat) {
  json::Array result;
  int64_t texelWords = ::vrc::vrcgpu::gatherTexelWords(texelFormat);
  for (const UnityBuffer &buffer : buffers) {
    json::Object entry{
        {"id", buffer.id},
        {"name", buffer.name},
        {"heap_slot", buffer.heapSlot},
        {"logical_bytes", buffer.logicalBytes},
        {"capacity_bytes", buffer.allocationBytes},
        {"width", buffer.width / texelWords},
        {"height", buffer.height},
        {"format", texelFormat.str()},
        {"srgb", false},
        {"mipmaps", false},
        {"filter", "point"},
    };
    if (!buffer.elementType.empty()) {
      entry["element_type"] = buffer.elementType;
      entry["public_access"] = buffer.publicAccess;
      if (buffer.source)
        entry["source"] = *buffer.source;
    }
    result.push_back(std::move(entry));
  }
  return result;
}

std::optional<json::Value> mlirAttributeToJson(mlir::Attribute attribute) {
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute)) {
    json::Object object;
    for (NamedAttribute field : dictionary) {
      auto value = mlirAttributeToJson(field.getValue());
      if (!value)
        return std::nullopt;
      object[field.getName().getValue()] = std::move(*value);
    }
    return json::Value(std::move(object));
  }
  if (auto array = dyn_cast<ArrayAttr>(attribute)) {
    json::Array values;
    for (mlir::Attribute element : array) {
      auto value = mlirAttributeToJson(element);
      if (!value)
        return std::nullopt;
      values.push_back(std::move(*value));
    }
    return json::Value(std::move(values));
  }
  if (auto string = dyn_cast<StringAttr>(attribute))
    return json::Value(string.getValue());
  if (auto integer = dyn_cast<IntegerAttr>(attribute)) {
    if (!integer.getValue().isSignedIntN(64))
      return std::nullopt;
    return json::Value(integer.getInt());
  }
  if (isa<UnitAttr>(attribute))
    return json::Value(true);
  return std::nullopt;
}

/// The `onnx_binding` entry of the manifest of an ONNX Host, as
/// vrc-describe-onnx-binding-manifest describes it, or nothing for a Host
/// without ONNX binding metadata.
FailureOr<std::optional<json::Value>>
unityOnnxBindingManifest(ModuleOp module) {
  PassManager describe(module.getContext());
  describe.enableVerifier(false);
  describe.addPass(::vrc::createDescribeONNXBindingManifestPass());
  if (failed(describe.run(module)))
    return failure();
  auto entry =
      module->getAttrOfType<DictionaryAttr>("vrc.onnx_binding_manifest");
  if (!entry)
    return std::optional<json::Value>();
  // The description is an input of the artifact, not Host IR.
  module->removeAttr("vrc.onnx_binding_manifest");
  auto value = mlirAttributeToJson(entry);
  if (!value) {
    module.emitError("ONNX binding manifest is not serializable");
    return failure();
  }
  return std::optional<json::Value>(std::move(*value));
}

/// Reads the UdonCL annotations of a linked Host LLVM module and erases their
/// metadata before the module is imported.
FailureOr<udoncl::HostAnnotations>
collectUdonCLHostAnnotations(llvm::Module &module) {
  auto annotations = udoncl::readHostAnnotations(module, errs());
  if (failed(annotations))
    return failure();
  udoncl::eraseHostAnnotations(module);
  udoncl::internalizePublicGlobals(module, *annotations);
  return annotations;
}

/// Re-attaches the per-function UdonCL annotations to the recovered Host
/// (vrc-attach-host-annotations).
LogicalResult
markRecoveredHostAnnotations(ModuleOp module,
                             const udoncl::HostAnnotations &annotations) {
  Builder builder(module.getContext());
  auto strings = [&](auto &&values) {
    SmallVector<StringRef> list(values.begin(), values.end());
    return builder.getStrArrayAttr(list);
  };
  SmallVector<mlir::Attribute> publicGlobals;
  for (const udoncl::PublicGlobal &global : annotations.publicGlobals)
    publicGlobals.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("element_type",
                             builder.getStringAttr(global.elementType)),
        builder.getNamedAttr("symbol", builder.getStringAttr(global.symbol)),
    }));
  SmallVector<NamedAttribute> parameterNames;
  for (const auto &[symbol, names] : annotations.parameterNames)
    parameterNames.push_back(builder.getNamedAttr(symbol, strings(names)));
  SmallVector<NamedAttribute> entries{
      builder.getNamedAttr("constructors", strings(annotations.constructors)),
      builder.getNamedAttr("destructors", strings(annotations.destructors)),
      builder.getNamedAttr("exports", strings(annotations.exports)),
      builder.getNamedAttr("parameter_names",
                           builder.getDictionaryAttr(parameterNames)),
      builder.getNamedAttr("public", builder.getArrayAttr(publicGlobals)),
      builder.getNamedAttr("while_busy", strings(annotations.whileBusy)),
  };
  if (annotations.program)
    entries.push_back(builder.getNamedAttr(
        "program", builder.getStringAttr(*annotations.program)));
  module->setAttr("udoncl.host_annotations",
                  builder.getDictionaryAttr(entries));
  PassManager attach(module.getContext());
  attach.addPass(::vrc::createAttachHostAnnotationsPass());
  return attach.run(module);
}

/// The Unity library configuration as an MLIR attribute, key by key.
mlir::Attribute jsonAttribute(Builder &builder, const json::Value &value) {
  if (auto object = value.getAsObject()) {
    SmallVector<mlir::NamedAttribute> entries;
    for (const auto &[key, entry] : *object)
      entries.push_back(
          builder.getNamedAttr(key.str(), jsonAttribute(builder, entry)));
    return builder.getDictionaryAttr(entries);
  }
  if (auto array = value.getAsArray()) {
    SmallVector<mlir::Attribute> elements;
    for (const json::Value &element : *array)
      elements.push_back(jsonAttribute(builder, element));
    return builder.getArrayAttr(elements);
  }
  if (auto string = value.getAsString())
    return builder.getStringAttr(*string);
  if (auto boolean = value.getAsBoolean())
    return builder.getBoolAttr(*boolean);
  if (auto integer = value.getAsInteger())
    return builder.getI64IntegerAttr(*integer);
  if (auto number = value.getAsNumber())
    return builder.getF64FloatAttr(*number);
  return builder.getUnitAttr();
}

/// Selects and names the U# exports of the recovered Host
/// (vrc-configure-unity-exports) from the Unity library configuration.
bool configureUnityExports(ModuleOp module, const json::Object &config,
                           bool finalizeReachability = true) {
  Builder builder(module.getContext());
  module->setAttr("udoncl.unity_library_config",
                  jsonAttribute(builder, json::Value(json::Object(config))));
  PassManager manager(module.getContext());
  ::vrc::ConfigureUnityExportsPassOptions options;
  options.finalizeReachability = finalizeReachability;
  manager.addPass(::vrc::createConfigureUnityExportsPass(options));
  return succeeded(manager.run(module));
}

LogicalResult optimizeUnityHostStorage(ModuleOp module) {
  // Preserve may-suspend call boundaries for distinct static coroutine frames.
  // Synchronous helpers remain available to whole-program scalar promotion.
  // The Func inliner honors no_inline on the callable.
  SmallVector<std::string> protectedFunctions;
  const bool debugCompilerPhases = module->hasAttr("vrc.debug_compiler_phases");
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto effect = function->getAttrOfType<StringAttr>("vrc.async_effect");
    const bool maySuspend = effect && effect.getValue() == "may_suspend";
    if (maySuspend && !function.getNoInline()) {
      function.setNoInline(true);
      protectedFunctions.push_back(function.getSymName().str());
      if (debugCompilerPhases)
        llvm::errs() << "udonclc-inline=preserve function="
                     << function.getSymName() << " reason=may-suspend\n";
    }
  }
  PassManager optimization(module.getContext());
  optimization.addPass(createInlinerPass());
  optimization.addPass(createSROA());
  optimization.addPass(createMem2Reg());
  optimization.addPass(createCanonicalizerPass());
  optimization.addPass(createSCCPPass());
  optimization.addPass(createCSEPass());
  // Canonical constants are what makes a decision edge recognizable; the
  // following canonicalization merges the blocks threading leaves behind.
  optimization.addPass(::vrc::createThreadConstantBranchesPass());
  optimization.addPass(createCanonicalizerPass());
  optimization.addPass(createSymbolDCEPass());
  // Before the async transform: a suspension spills a live address, and a
  // spilled address no longer names its global.
  optimization.addPass(::vrc::createPromoteHostGlobalsToUdonVariablesPass());
  LogicalResult result = optimization.run(module);
  for (const std::string &name : protectedFunctions)
    if (auto function = module.lookupSymbol<func::FuncOp>(name))
      function.setNoInline(false);
  return result;
}

LogicalResult lowerUnityHostObjectIntrinsics(ModuleOp module,
                                             MLIRContext &context) {
  PassManager manager(&context);
  manager.addPass(::vrc::createLowerHostObjectIntrinsicsPass());
  if (failed(manager.run(module)))
    return failure();
  return verify(module);
}

void printCompilerIRStats(ModuleOp module, StringRef stage) {
  if (!module->hasAttr("vrc.debug_compiler_phases"))
    return;
  uint64_t operations = 0;
  uint64_t blocks = 0;
  uint64_t functions = 0;
  uint64_t asyncFunctions = 0;
  module.walk([&](Operation *operation) {
    ++operations;
    for (Region &region : operation->getRegions())
      blocks += region.getBlocks().size();
    functions += isa<func::FuncOp>(operation);
    asyncFunctions += isa<async::FuncOp>(operation);
  });
  llvm::errs() << "udonclc-ir-stats stage=" << stage
               << " operations=" << operations << " blocks=" << blocks
               << " functions=" << functions
               << " async_functions=" << asyncFunctions << '\n';
}

LogicalResult propagateUnityHostAsyncEffects(ModuleOp module) {
  bool changed;
  do {
    changed = false;
    for (func::FuncOp caller : module.getOps<func::FuncOp>()) {
      if (auto effect = caller->getAttrOfType<StringAttr>("vrc.async_effect");
          effect && effect.getValue() == "may_suspend")
        continue;
      bool maySuspend = false;
      caller.walk([&](func::CallOp call) {
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        auto effect =
            callee ? callee->getAttrOfType<StringAttr>("vrc.async_effect")
                   : StringAttr();
        maySuspend |= effect && effect.getValue() == "may_suspend";
      });
      if (!maySuspend)
        continue;
      caller->setAttr("vrc.async_effect",
                      StringAttr::get(module.getContext(), "may_suspend"));
      changed = true;
    }
  } while (changed);
  // The binding runs an export allowed while busy in the middle of another
  // export's suspension; it has no frame to suspend in itself.
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto effect = function->getAttrOfType<StringAttr>("vrc.async_effect");
    if (function->hasAttr("vrc.allowed_while_busy") && effect &&
        effect.getValue() == "may_suspend") {
      errs() << "UDONCL_EXPORT_WHILE_BUSY export may suspend: "
             << function.getSymName()
             << "; an export allowed while busy must not wait for Device "
                "work\n";
      return failure();
    }
  }
  return success();
}

/// The OpenCL API of a Unity Host is implemented by the linked OpenCL AOT
/// runtime (UdonCL::OpenCLRuntime, passed as --host-library-llvm). A Host that
/// calls the API, or the removed <udoncl/unity_buffer.h> intrinsics, without
/// the runtime is rejected with the calls it makes.
LogicalResult rejectUnlinkedOpenCLHostCalls(const llvm::Module &module) {
  SmallVector<StringRef> openCLCalls, unityBufferCalls;
  for (const llvm::Function &function : module) {
    if (!function.isDeclaration() || function.use_empty())
      continue;
    StringRef name = function.getName();
    if (name.size() > 2 && name.starts_with("cl") && llvm::isUpper(name[2]))
      openCLCalls.push_back(name);
    else if (name == "__udoncl_buffer_alloc" || name == "__udoncl_buffer_copy")
      unityBufferCalls.push_back(name);
  }
  if (!unityBufferCalls.empty()) {
    llvm::sort(unityBufferCalls);
    errs() << "the Unity Host calls " << llvm::join(unityBufferCalls, ", ")
           << " of the removed <udoncl/unity_buffer.h>; allocate and copy "
              "buffers with clCreateBuffer and clEnqueueCopyBuffer and link "
              "UdonCL::OpenCLRuntime\n";
    return failure();
  }
  if (!openCLCalls.empty()) {
    llvm::sort(openCLCalls);
    errs() << "the Unity Host calls the OpenCL API ("
           << llvm::join(openCLCalls, ", ")
           << ") but links no OpenCL runtime; add UdonCL::OpenCLRuntime to "
              "HOST_LIBRARIES of udoncl_add_unity_library (udonclc "
              "--host-library-llvm)\n";
    return failure();
  }
  return success();
}

FailureOr<OwningOpRef<ModuleOp>> recoverUnityHostLLVM(StringRef filename,
                                                      MLIRContext &context,
                                                      bool linkedRuntime) {
  LLVMContext llvmContext;
  SMDiagnostic diagnostic;
  auto inputModule = parseIRFile(filename, diagnostic, llvmContext);
  if (!inputModule) {
    diagnostic.print("udonclc", errs());
    return failure();
  }
  if (inputModule->getTargetTriple().str() != "spir-unknown-unknown" ||
      !inputModule->getDataLayout().isLittleEndian() ||
      inputModule->getDataLayout().getPointerSizeInBits() != 32) {
    errs() << "Unity Host library requires 32-bit little-endian SPIR input\n";
    return failure();
  }
  normalizeUnityHostCallingConventions(*inputModule);
  auto annotations = collectUdonCLHostAnnotations(*inputModule);
  if (failed(annotations))
    return failure();
  bool hasRuntime =
      inputModule->getFunction("__vrc_opencl_buffer_alloc") != nullptr;
  // The user OpenCL C kernels the Host C launches through the runtime.
  if (!linkedRuntime && !hasRuntime && !deviceLLVM.empty()) {
    auto deviceModule = parseIRFile(deviceLLVM, diagnostic, llvmContext);
    if (!deviceModule) {
      diagnostic.print("udonclc", errs());
      return failure();
    }
    auto kernels =
        udoncl::discoverOpenCLUserKernels(*inputModule, *deviceModule, errs());
    if (failed(kernels) || failed(udoncl::bindOpenCLUserKernelLaunches(
                               *inputModule, kernels->kernels, errs())))
      return failure();
    std::map<std::string, std::string> publicGlobals;
    for (const udoncl::PublicGlobal &global : annotations->publicGlobals)
      publicGlobals[global.symbol] = global.elementType;
    if (failed(udoncl::checkPublicKernelArgumentTypes(
            *inputModule, kernels->kernels, publicGlobals, errs())))
      return failure();
  }
  if (linkedRuntime && !hasRuntime) {
    errs() << "--host-library-llvm requires the linked OpenCL AOT runtime\n";
    return failure();
  }
  if (!linkedRuntime && hasRuntime) {
    errs() << "--host-llvm must be runtime-free when "
              "--host-library-llvm is provided\n";
    return failure();
  }
  auto library = ::vrc::importLLVM(std::move(inputModule), context);
  if (!library || failed(verify(*library)))
    return failure();
  for (const std::string &symbol : annotations->exports) {
    auto function = library->lookupSymbol<LLVM::LLVMFuncOp>(symbol);
    if (!function) {
      errs() << "UDONCL_EXPORT definition was lost while importing Host "
                "library: "
             << symbol << "\n";
      return failure();
    }
    function->setAttr("udoncl.export", UnitAttr::get(&context));
  }
  PassManager cleanup(&context);
  cleanup.addPass(createSROA());
  cleanup.addPass(createMem2Reg());
  cleanup.addPass(createCanonicalizerPass());
  if (failed(cleanup.run(*library)))
    return failure();
  PassManager recovery(&context);
  recovery.enableVerifier(false);
  recovery.addPass(createCanonicalizerPass());
  ::vrc::RaiseLLVMMemoryPassOptions memoryOptions;
  memoryOptions.allowRawFallback = true;
  recovery.addPass(::vrc::createRaiseLLVMMemoryPass(memoryOptions));
  recovery.addPass(::vrc::createRaiseLLVMRawMemoryPass());
  recovery.addPass(::vrc::createRaiseLLVMToScalarPass());
  recovery.addPass(createCanonicalizerPass());
  if (linkedRuntime)
    recovery.addPass(::vrc::createLowerOpenCLAOTIntrinsicsPass());
  if (failed(recovery.run(*library)) || failed(verify(*library)))
    return failure();
  if (failed(markRecoveredHostAnnotations(*library, *annotations)))
    return failure();
  return library;
}

FailureOr<OwningOpRef<ModuleOp>>
recoverLinkedUnityHostLibrary(StringRef filename, MLIRContext &context) {
  return recoverUnityHostLLVM(filename, context, true);
}

FailureOr<OwningOpRef<ModuleOp>>
recoverRuntimeFreeUnityHost(StringRef filename, MLIRContext &context) {
  return recoverUnityHostLLVM(filename, context, false);
}

/// Copies each configured kernel's optional `fragment_cost` declaration onto
/// its dispatch sites as `vrc.fragment_cost`, completing every argument
/// reference with the declared `argument_bounds` maximum used when the site
/// argument is not a constant:
///   "fragment_cost": {"loads": L, "active": [i, ...],
///                     "per_argument": [{"index": i, "loads": l}, ...]}
/// Loads per fragment are L + sum(argument_i * l) over the fragments selected
/// by the product of the `active` arguments (all launched fragments when
/// absent).
LogicalResult
attachDeviceFragmentCosts(ModuleOp module,
                          ArrayRef<const json::Object *> deviceConfigs) {
  Builder builder(module.getContext());
  llvm::DenseMap<int64_t, DictionaryAttr> costs;
  for (const json::Object *config : deviceConfigs) {
    const json::Object *declared = config->getObject("fragment_cost");
    if (!declared)
      continue;
    int64_t kernelId = *config->getInteger("kernel_id");
    auto fail = [&](const Twine &message) {
      errs() << "kernel " << kernelId << " fragment_cost: " << message << "\n";
      return failure();
    };
    llvm::DenseMap<int64_t, int64_t> maxima;
    if (const json::Array *bounds = config->getArray("argument_bounds"))
      for (const json::Value &bound : *bounds)
        if (const json::Object *entry = bound.getAsObject())
          if (auto index = entry->getInteger("index"))
            if (auto maximum = entry->getInteger("max"))
              maxima[*index] = *maximum;
    auto maximumOf = [&](int64_t index) -> std::optional<int64_t> {
      auto found = maxima.find(index);
      if (found == maxima.end())
        return std::nullopt;
      return found->second;
    };
    std::optional<int64_t> loads = declared->getInteger("loads");
    if (!loads || *loads < 0)
      return fail("requires non-negative integer loads");
    SmallVector<NamedAttribute> entries{
        builder.getNamedAttr("loads", builder.getI64IntegerAttr(*loads))};
    if (const json::Array *active = declared->getArray("active")) {
      SmallVector<mlir::Attribute> terms;
      for (const json::Value &value : *active) {
        std::optional<int64_t> index = value.getAsInteger();
        std::optional<int64_t> maximum =
            index ? maximumOf(*index) : std::nullopt;
        if (!maximum)
          return fail("active arguments need argument_bounds");
        terms.push_back(builder.getDictionaryAttr(
            {builder.getNamedAttr("index", builder.getI64IntegerAttr(*index)),
             builder.getNamedAttr("max",
                                  builder.getI64IntegerAttr(*maximum))}));
      }
      entries.push_back(
          builder.getNamedAttr("active", builder.getArrayAttr(terms)));
    }
    if (const json::Array *perArgument = declared->getArray("per_argument")) {
      SmallVector<mlir::Attribute> terms;
      for (const json::Value &value : *perArgument) {
        const json::Object *entry = value.getAsObject();
        std::optional<int64_t> index =
            entry ? entry->getInteger("index") : std::nullopt;
        std::optional<int64_t> unit =
            entry ? entry->getInteger("loads") : std::nullopt;
        std::optional<int64_t> maximum =
            index ? maximumOf(*index) : std::nullopt;
        if (!unit || *unit < 0 || !maximum)
          return fail("per_argument entries need non-negative loads and "
                      "argument_bounds");
        terms.push_back(builder.getDictionaryAttr(
            {builder.getNamedAttr("index", builder.getI64IntegerAttr(*index)),
             builder.getNamedAttr("loads", builder.getI64IntegerAttr(*unit)),
             builder.getNamedAttr("max",
                                  builder.getI64IntegerAttr(*maximum))}));
      }
      entries.push_back(
          builder.getNamedAttr("per_argument", builder.getArrayAttr(terms)));
    }
    costs[kernelId] = builder.getDictionaryAttr(entries);
  }
  if (costs.empty())
    return success();
  module.walk([&](::vrc::vrc::DispatchOp dispatch) {
    auto found = costs.find(dispatch.getKernelId());
    if (found != costs.end())
      dispatch->setAttr("vrc.fragment_cost", found->second);
  });
  return success();
}

/// The largest frame budget setting, in microseconds or continuations. The
/// generated binding accepts the same range at run time.
constexpr int64_t kDeviceWorkFrameSettingMaximum = 1000000;

/// How much the generated binding resumes within one frame: a continuation
/// that is ready in the same Udon event (a Device work slice or a
/// synchronously completed command) runs at once while both limits allow.
/// Every Host slice of a frame is charged, the first one included, and a
/// zero limit resumes one slice per frame. The binding can change both at
/// run time; these are its defaults: 8 ms of a 90 Hz frame's 11.1 ms, and
/// no practical continuation count. A count is only a proxy for time, and
/// any fixed one stops a frame early once slices get cheaper than the budget
/// divided by the count (32 did once slices took 0.12 to 0.25 ms), so the
/// default is the setting maximum: reaching it within 8 ms would take 8 ns
/// slices, so the time budget alone ends a busy frame. A library can still
/// set a smaller count, for example for deterministic pacing.
struct DeviceWorkFrameBudget {
  int64_t microseconds = 8000;
  int64_t continuations = kDeviceWorkFrameSettingMaximum;
};

/// Reads the optional per-library slice and frame budgets:
///   "device_work_budget": {"device_cost": N, "host_cost": N,
///                          "fragment_weight": N, "frame_microseconds": N,
///                          "frame_continuations": N}
LogicalResult
readDeviceWorkBudget(const json::Object &config,
                     ::vrc::ScheduleUdonDeviceWorkPassOptions &out,
                     DeviceWorkFrameBudget &frame) {
  const json::Value *value = config.get("device_work_budget");
  if (!value)
    return success();
  StringRef budgetSetting = "--device-work-budget (DEVICE_WORK_BUDGET)";
  const json::Object *budget = value->getAsObject();
  if (!budget) {
    errs() << budgetSetting << " must be an object\n";
    return failure();
  }
  for (const auto &[key, field] : *budget) {
    std::optional<int64_t> number = field.getAsInteger();
    if (!number) {
      errs() << budgetSetting << " " << key.str() << " must be an integer\n";
      return failure();
    }
    if (key == "device_cost")
      out.deviceCostBudget = *number;
    else if (key == "host_cost")
      out.hostCostBudget = *number;
    else if (key == "fragment_weight")
      out.fragmentWeight = *number;
    else if (key == "frame_microseconds" || key == "frame_continuations") {
      if (*number < 0 || *number > kDeviceWorkFrameSettingMaximum) {
        errs() << budgetSetting << " " << key.str() << " must be between 0 and "
               << kDeviceWorkFrameSettingMaximum << "\n";
        return failure();
      }
      (key == "frame_microseconds" ? frame.microseconds : frame.continuations) =
          *number;
    } else {
      errs() << "unknown " << budgetSetting << " key " << key.str() << "\n";
      return failure();
    }
  }
  return success();
}

/// Prints the start and the duration of every pass of a pipeline under
/// --debug-compiler-phases.
struct PhaseTimingInstrumentation : PassInstrumentation {
  std::vector<std::chrono::steady_clock::time_point> started;

  void runBeforePass(Pass *pass, Operation *) override {
    errs() << "udonclc-unity-prepare-pass=" << pass->getArgument()
           << " state=start\n";
    started.push_back(std::chrono::steady_clock::now());
  }
  void runAfterPass(Pass *pass, Operation *) override {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - started.back())
                       .count();
    started.pop_back();
    errs() << "udonclc-unity-prepare-pass=" << pass->getArgument()
           << " state=complete elapsed_ms=" << elapsed << "\n";
  }
  void runAfterPassFailed(Pass *, Operation *) override { started.pop_back(); }
};

LogicalResult discoverStaticDeviceResources(ModuleOp module) {
  constexpr StringLiteral debugProvenanceAttribute =
      "vrc.debug_device_resource_provenance";
  if (debugCompilerPhases)
    module->setAttr(debugProvenanceAttribute,
                    UnitAttr::get(module.getContext()));
  PassManager manager(module.getContext());
  ::vrc::buildStaticDeviceResourceDiscoveryPipeline(manager);
  if (debugCompilerPhases)
    manager.addInstrumentation(std::make_unique<PhaseTimingInstrumentation>());
  if (failed(manager.run(module)))
    return failure();
  module->removeAttr(debugProvenanceAttribute);
  return verify(module);
}

LogicalResult linkUnityHostLibrary(ModuleOp destination, ModuleOp library) {
  for (NamedAttribute attribute : library->getAttrs())
    if (!destination->hasAttr(attribute.getName()))
      destination->setAttr(attribute.getName(), attribute.getValue());

  SymbolTable symbols(destination);
  for (Operation &operation :
       llvm::make_early_inc_range(library.getBody()->getOperations())) {
    auto symbol = dyn_cast<SymbolOpInterface>(&operation);
    if (!symbol) {
      operation.moveBefore(destination.getBody(), destination.getBody()->end());
      continue;
    }
    Operation *existing = symbols.lookup(symbol.getName());
    if (existing) {
      auto existingFunction = dyn_cast<func::FuncOp>(existing);
      auto libraryFunction = dyn_cast<func::FuncOp>(&operation);
      if (existingFunction && libraryFunction &&
          existingFunction.isDeclaration()) {
        if (existingFunction.getFunctionType() !=
            libraryFunction.getFunctionType())
          return existingFunction.emitError(
                     "Host library definition has a mismatched ABI for ")
                 << symbol.getName() << ": expected "
                 << existingFunction.getFunctionType() << ", recovered "
                 << libraryFunction.getFunctionType();
        symbols.erase(existingFunction);
      } else {
        operation.erase();
        continue;
      }
    }
    operation.moveBefore(destination.getBody(), destination.getBody()->end());
    symbols.insert(&operation);
  }
  return verify(destination);
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

/// The program's transfer shader. Mode 0 (the material default) copies the
/// whole source surface and serves whole-buffer copies and binding uploads.
/// Modes 1 and 2 write only the destination words [First, First + Count) of a
/// surface Width texels wide and discard every other fragment, so the rest of
/// the destination keeps its contents: mode 1 copies source word
/// (word + Delta), mode 2 writes pattern word ((word - First) mod Period).
/// Bytes are carried through RGBA8 texels exactly, as in every kernel.
std::string unityLibraryBufferCopyShader(StringRef program) {
  std::string words, properties;
  raw_string_ostream wordList(words), propertyList(properties);
  for (int index = 0; index < 32; ++index) {
    propertyList << "    _VRC_TransferWord" << index << " (\"Word " << index
                 << "\", Integer) = 0\n";
    wordList << (index ? ", " : "") << "asuint(_VRC_TransferWord" << index
             << ")";
  }
  std::string declarations;
  raw_string_ostream declarationList(declarations);
  for (StringRef name : {"Mode", "Width", "First", "Count", "Period", "Delta"})
    declarationList << "    int _VRC_Transfer" << name << ";\n";
  for (int index = 0; index < 32; ++index)
    declarationList << "    int _VRC_TransferWord" << index << ";\n";
  return (Twine("Shader \"Hidden/UdonCL/") + program +
          "/BufferCopy\" {\n"
          "  Properties {\n"
          "    _MainTex (\"Source\", 2D) = \"black\" {}\n"
          "    _VRC_TransferMode (\"Mode\", Integer) = 0\n"
          "    _VRC_TransferWidth (\"Destination width\", Integer) = 1\n"
          "    _VRC_TransferFirst (\"First word\", Integer) = 0\n"
          "    _VRC_TransferCount (\"Word count\", Integer) = 0\n"
          "    _VRC_TransferPeriod (\"Pattern words\", Integer) = 1\n"
          "    _VRC_TransferDelta (\"Source word delta\", Integer) = 0\n" +
          properties +
          "  }\n"
          "  SubShader { Pass { ZTest Always Cull Off ZWrite Off Blend Off\n"
          "    HLSLPROGRAM\n"
          "    #pragma target 5.0\n"
          "    #pragma vertex vert_img\n"
          "    #pragma fragment frag\n"
          "    #include \"UnityCG.cginc\"\n"
          "    Texture2D<float4> _MainTex;\n" +
          declarations +
          "    float4 frag(v2f_img input) : SV_Target {\n"
          "      uint width, height;\n"
          "      if (_VRC_TransferMode == 0) {\n"
          "        _MainTex.GetDimensions(width, height);\n"
          "        uint2 p = min(uint2(input.pos.xy), "
          "uint2(width - 1, height - 1));\n"
          "        return _MainTex.Load(int3(p, 0));\n"
          "      }\n"
          "      uint2 position = uint2(input.pos.xy);\n"
          "      uint index = position.y * uint(_VRC_TransferWidth) + "
          "position.x;\n"
          "      uint offset = index - uint(_VRC_TransferFirst);\n"
          "      if (index < uint(_VRC_TransferFirst) || "
          "offset >= uint(_VRC_TransferCount))\n"
          "        discard;\n"
          "      if (_VRC_TransferMode == 2) {\n"
          "        uint words[32] = { " +
          words +
          " };\n"
          "        uint word = words[offset % uint(max(_VRC_TransferPeriod, "
          "1))];\n"
          "        return float4(word & 255u, (word >> 8u) & 255u, "
          "(word >> 16u) & 255u, word >> 24u) / 255.0;\n"
          "      }\n"
          "      _MainTex.GetDimensions(width, height);\n"
          "      uint source = index + uint(_VRC_TransferDelta);\n"
          "      return _MainTex.Load(int3(source % width, source / width, "
          "0));\n"
          "    }\n"
          "  ENDHLSL } }\n"
          "}\n")
      .str();
}

/// Reads the `gpu.func` named by a native Device bundle config from a module
/// of compiler-synthesized gather kernels. A synthesized kernel carries its
/// own launch contract, so the config must agree with it and only assigns the
/// kernel id. A native bundle holds one kernel; the others are dropped.
/// The transfer shader of a program whose buffers hold four words per
/// RGBA32 float texel, read and written as bits. It has the RGBA8 shader's
/// modes and properties (Width counts destination texels per row) and three
/// passes, chosen by the caller:
///  - pass 0 writes word texels from an RGBA8 word source (one word per
///    texel, a binding's upload staging texture);
///  - pass 1 writes word texels from a word-texel source (Device copies,
///    fills, and the snapshot of a destination);
///  - pass 2 writes RGBA8 words from a word-texel source, whole surface
///    (a binding's readback staging texture).
/// Modes 1 and 2 write a texel only if some of its words are in
/// [First, First + Count); its other words come from _VRC_TransferPrevious,
/// a snapshot of the destination the caller made with pass 1 mode 0.
std::string unityLibraryWordTexelCopyShader(StringRef program) {
  std::string words, properties;
  raw_string_ostream wordList(words), propertyList(properties);
  for (int index = 0; index < 32; ++index) {
    propertyList << "    _VRC_TransferWord" << index << " (\"Word " << index
                 << "\", Integer) = 0\n";
    wordList << (index ? ", " : "") << "asuint(_VRC_TransferWord" << index
             << ")";
  }
  std::string declarations;
  raw_string_ostream declarationList(declarations);
  for (StringRef name : {"Mode", "Width", "First", "Count", "Period", "Delta"})
    declarationList << "    int _VRC_Transfer" << name << ";\n";
  for (int index = 0; index < 32; ++index)
    declarationList << "    int _VRC_TransferWord" << index << ";\n";
  std::string common =
      "    #pragma target 5.0\n"
      "    #pragma vertex vert_img\n"
      "    #pragma fragment frag\n"
      "    #include \"UnityCG.cginc\"\n" +
      declarations +
      "    uint TransferLane(uint4 t, uint c) {\n"
      "      return c < 2u ? (c == 0u ? t.x : t.y) : (c == 2u ? t.z : t.w);\n"
      "    }\n";
  // Writes word texels; `SourceWord(word)` and `SourceTexel(position)` read
  // the source.
  std::string writeTexels =
      "    Texture2D<float4> _VRC_TransferPrevious;\n"
      "    float4 frag(v2f_img input) : SV_Target {\n"
      "      uint2 position = uint2(input.pos.xy);\n"
      "      if (_VRC_TransferMode == 0)\n"
      "        return asfloat(SourceTexel(position));\n"
      "      uint4 previous = asuint(_VRC_TransferPrevious.Load(int3(position, "
      "0)));\n"
      "      uint first = position.y * uint(_VRC_TransferWidth) * 4u + "
      "position.x * 4u;\n"
      "      uint words[32] = { " +
      words +
      " };\n"
      "      uint4 result = previous;\n"
      "      bool written = false;\n"
      "      [unroll] for (uint lane = 0u; lane < 4u; ++lane) {\n"
      "        uint index = first + lane;\n"
      "        uint offset = index - uint(_VRC_TransferFirst);\n"
      "        if (index >= uint(_VRC_TransferFirst) && "
      "offset < uint(_VRC_TransferCount)) {\n"
      "          uint word = _VRC_TransferMode == 2\n"
      "              ? words[offset % uint(max(_VRC_TransferPeriod, 1))]\n"
      "              : SourceWord(index + uint(_VRC_TransferDelta));\n"
      "          result = lane == 0u ? uint4(word, result.yzw)\n"
      "                 : lane == 1u ? uint4(result.x, word, result.zw)\n"
      "                 : lane == 2u ? uint4(result.xy, word, result.w)\n"
      "                              : uint4(result.xyz, word);\n"
      "          written = true;\n"
      "        }\n"
      "      }\n"
      "      if (!written)\n"
      "        discard;\n"
      "      return asfloat(result);\n"
      "    }\n";
  std::string fromBytes =
      "    Texture2D<float4> _MainTex;\n"
      "    uint SourceWordAt(uint2 p) {\n"
      "      float4 t = _MainTex.Load(int3(p, 0));\n"
      "      uint2 halves = uint2(t.yw * 65280.0 + t.xz * 255.0 + 0.5);\n"
      "      return halves.x | (halves.y << 16u);\n"
      "    }\n"
      "    uint SourceWord(uint word) {\n"
      "      uint width, height;\n"
      "      _MainTex.GetDimensions(width, height);\n"
      "      return SourceWordAt(uint2(word % width, word / width));\n"
      "    }\n"
      "    uint4 SourceTexel(uint2 position) {\n"
      "      uint width, height;\n"
      "      _MainTex.GetDimensions(width, height);\n"
      "      uint4 texel = (uint4)0;\n"
      "      [unroll] for (uint lane = 0u; lane < 4u; ++lane) {\n"
      "        uint2 p = min(uint2(position.x * 4u + lane, position.y), "
      "uint2(width - 1u, height - 1u));\n"
      "        uint word = SourceWordAt(p);\n"
      "        texel = lane == 0u ? uint4(word, texel.yzw)\n"
      "              : lane == 1u ? uint4(texel.x, word, texel.zw)\n"
      "              : lane == 2u ? uint4(texel.xy, word, texel.w)\n"
      "                           : uint4(texel.xyz, word);\n"
      "      }\n"
      "      return texel;\n"
      "    }\n";
  std::string fromWords =
      "    Texture2D<float4> _MainTex;\n"
      "    uint SourceWord(uint word) {\n"
      "      uint width, height;\n"
      "      _MainTex.GetDimensions(width, height);\n"
      "      uint row = width * 4u, column = word % row;\n"
      "      return TransferLane(asuint(_MainTex.Load(int3(column >> 2u, "
      "word / row, 0))), column & 3u);\n"
      "    }\n"
      "    uint4 SourceTexel(uint2 position) {\n"
      "      uint width, height;\n"
      "      _MainTex.GetDimensions(width, height);\n"
      "      uint2 p = min(position, uint2(width - 1u, height - 1u));\n"
      "      return asuint(_MainTex.Load(int3(p, 0)));\n"
      "    }\n";
  std::string toBytes =
      "    Texture2D<float4> _MainTex;\n"
      "    float4 frag(v2f_img input) : SV_Target {\n"
      "      uint width, height;\n"
      "      _MainTex.GetDimensions(width, height);\n"
      "      uint2 p = min(uint2(input.pos.xy), uint2(width * 4u - 1u, "
      "height - 1u));\n"
      "      uint word = TransferLane(asuint(_MainTex.Load(int3(p.x >> 2u, "
      "p.y, 0))), p.x & 3u);\n"
      "      return float4(word & 255u, (word >> 8u) & 255u, (word >> 16u) & "
      "255u, word >> 24u) / 255.0;\n"
      "    }\n";
  auto pass = [&](StringRef body) {
    return ("  Pass { ZTest Always Cull Off ZWrite Off Blend Off\n"
            "    HLSLPROGRAM\n" +
            common + body.str() + "  ENDHLSL }\n");
  };
  return (Twine("Shader \"Hidden/UdonCL/") + program +
          "/BufferCopy\" {\n"
          "  Properties {\n"
          "    _MainTex (\"Source\", 2D) = \"black\" {}\n"
          "    _VRC_TransferPrevious (\"Destination snapshot\", 2D) = "
          "\"black\" {}\n"
          "    _VRC_TransferMode (\"Mode\", Integer) = 0\n"
          "    _VRC_TransferWidth (\"Destination width\", Integer) = 1\n"
          "    _VRC_TransferFirst (\"First word\", Integer) = 0\n"
          "    _VRC_TransferCount (\"Word count\", Integer) = 0\n"
          "    _VRC_TransferPeriod (\"Pattern words\", Integer) = 1\n"
          "    _VRC_TransferDelta (\"Source word delta\", Integer) = 0\n" +
          properties + "  }\n  SubShader {\n" + pass(fromBytes + writeTexels) +
          pass(fromWords + writeTexels) + pass(toBytes) + "  }\n}\n")
      .str();
}

OwningOpRef<ModuleOp> importSynthesizedDeviceKernel(StringRef filename,
                                                    const json::Object &config,
                                                    MLIRContext &context) {
  auto fail = [](const Twine &message) {
    errs() << "synthesized Device kernel: " << message << "\n";
    return OwningOpRef<ModuleOp>();
  };
  auto dimensions = [&](StringRef key) -> std::optional<SmallVector<int32_t>> {
    const json::Array *array = config.getArray(key);
    if (!array)
      return std::nullopt;
    SmallVector<int32_t> values;
    for (const json::Value &value : *array) {
      std::optional<int64_t> integer = value.getAsInteger();
      if (!integer || *integer < INT32_MIN || *integer > INT32_MAX)
        return std::nullopt;
      values.push_back(static_cast<int32_t>(*integer));
    }
    return values;
  };
  std::optional<StringRef> name = config.getString("kernel");
  std::optional<int64_t> kernelId = config.getInteger("kernel_id");
  std::optional<int64_t> words = config.getInteger("words");
  auto globalSize = dimensions("global_size");
  if (config.getInteger("schema_version") != 1 ||
      config.getString("profile") != "gather-f32" || !name || !kernelId ||
      *kernelId <= 0 || *kernelId > INT32_MAX || !words || !globalSize ||
      dimensions("local_size") != SmallVector<int32_t>{1, 1, 1} ||
      dimensions("global_offset") != SmallVector<int32_t>{0, 0, 0})
    return fail("config requires schema_version 1, profile gather-f32, a "
                "kernel name, a positive kernel_id, words, global_size, unit "
                "local_size and zero global_offset");

  context
      .loadDialect<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                   math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(filename, &context);
  if (!module)
    return {};
  gpu::GPUFuncOp kernel;
  SmallVector<Operation *> dropped;
  for (Operation &op : *module->getBody()) {
    auto gpuModule = dyn_cast<gpu::GPUModuleOp>(op);
    if (!gpuModule)
      return fail("the module may only contain gpu.module operations");
    auto found = kernel ? gpu::GPUFuncOp()
                        : gpuModule.lookupSymbol<gpu::GPUFuncOp>(*name);
    if (!found) {
      dropped.push_back(&op);
      continue;
    }
    kernel = found;
    for (Operation &member : *gpuModule.getBody())
      if (&member != kernel.getOperation())
        dropped.push_back(&member);
  }
  if (!kernel || !kernel->hasAttr("vrc.synthesized_kernel"))
    return fail("no gpu.func @" + *name + " marked vrc.synthesized_kernel");
  for (Operation *op : dropped)
    op->erase();

  // The kernel is the source of truth for its launch contract.
  Builder builder(&context);
  auto grid = kernel->getAttrOfType<DenseI32ArrayAttr>("known_grid_size");
  auto kernelWords = kernel->getAttrOfType<IntegerAttr>("vrc.words");
  if (!grid || grid.asArrayRef() != ArrayRef<int32_t>(*globalSize))
    return fail("config global_size does not match known_grid_size");
  if (!kernelWords || kernelWords.getInt() != *words)
    return fail("config words does not match vrc.words");
  if (const json::Array *inputWords = config.getArray("input_words")) {
    SmallVector<int64_t> declared;
    for (mlir::Type type : kernel.getFunctionType().getInputs())
      if (auto buffer = dyn_cast<MemRefType>(type))
        declared.push_back(buffer.hasStaticShape() ? buffer.getNumElements()
                                                   : int64_t{-1});
    // The last buffer is the output.
    if (!declared.empty())
      declared.pop_back();
    bool matches = inputWords->size() == declared.size();
    for (size_t index = 0; matches && index < declared.size(); ++index)
      matches = (*inputWords)[index].getAsInteger() == declared[index];
    if (!matches)
      return fail("config input_words does not match the input buffer sizes");
  }
  if (const json::Array *argumentBounds = config.getArray("argument_bounds")) {
    SmallVector<mlir::Attribute> entries;
    for (const json::Value &value : *argumentBounds) {
      const json::Object *entry = value.getAsObject();
      auto index = entry ? entry->getInteger("index") : std::nullopt;
      auto minimum = entry ? entry->getInteger("min") : std::nullopt;
      auto maximum = entry ? entry->getInteger("max") : std::nullopt;
      if (!index || !minimum || !maximum)
        return fail("invalid config argument_bounds entry");
      entries.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("index", builder.getI32IntegerAttr(*index)),
          builder.getNamedAttr("min", builder.getI32IntegerAttr(*minimum)),
          builder.getNamedAttr("max", builder.getI32IntegerAttr(*maximum)),
      }));
    }
    if (kernel->getAttr("vrc.argument_bounds") != builder.getArrayAttr(entries))
      return fail("config argument_bounds does not match "
                  "vrc.argument_bounds");
  }
  IntegerAttr id = builder.getI32IntegerAttr(static_cast<int32_t>(*kernelId));
  if (mlir::Attribute assigned = kernel->getAttr("vrc.kernel_id");
      assigned && assigned != id)
    return fail("config kernel_id does not match vrc.kernel_id");
  kernel->setAttr("vrc.kernel_id", id);
  // A proof marker read from a file is a claim, not a proof.
  kernel->removeAttr("vrc.gather_proven");
  return module;
}

/// Kernel ids below this are left to hand-written library kernels, whose ids
/// are fixed by their dispatch intrinsics and the library configs.
constexpr int64_t firstSynthesizedKernelId = 4096;

/// A synthesized kernel taken out of a Unity Host module, alone in its
/// `gpu.module @device_gpu_<id>`.
struct SynthesizedUnityKernel {
  gpu::GPUModuleOp module;
  std::string name;
  int32_t id;
  int64_t width;
  int64_t words;
};

/// Resolves the generated kernel channel of a Unity Host module: assigns ids
/// that avoid the configured library kernels, proves the kernels, and moves
/// them out of `host` into the returned module, which is null when the Host
/// carries none. The Host keeps only self-describing dispatch declarations.
FailureOr<OwningOpRef<ModuleOp>> extractSynthesizedUnityKernels(
    ModuleOp host, ArrayRef<const json::Object *> deviceConfigs,
    SmallVectorImpl<SynthesizedUnityKernel> &kernels) {
  ::vrc::AssignSynthesizedKernelIdsPassOptions ids;
  ids.firstId = firstSynthesizedKernelId;
  for (const json::Object *config : deviceConfigs)
    ids.reservedIds.push_back(*config->getInteger("kernel_id"));
  PassManager resolution(host.getContext());
  // The caller verifies the Host once kernels have left it; a Host without
  // any is not changed.
  resolution.enableVerifier(false);
  resolution.addPass(::vrc::createAssignSynthesizedKernelIdsPass(ids));
  // No LLVM-level gather proof covers a synthesized kernel.
  resolution.addPass(::vrc::createProveSynthesizedGatherKernelsPass());
  if (failed(resolution.run(host)))
    return failure();
  OwningOpRef<ModuleOp> device;
  for (gpu::GPUModuleOp gpuModule :
       llvm::make_early_inc_range(host.getOps<gpu::GPUModuleOp>())) {
    auto functions = gpuModule.getOps<gpu::GPUFuncOp>();
    if (!llvm::hasSingleElement(functions) ||
        !(*functions.begin())->hasAttr("vrc.synthesized_kernel"))
      continue;
    gpu::GPUFuncOp kernel = *functions.begin();
    if (!device)
      device = ModuleOp::create(host.getLoc());
    gpuModule->moveBefore(device->getBody(), device->getBody()->end());
    kernels.push_back(
        {gpuModule, kernel.getName().str(),
         static_cast<int32_t>(
             kernel->getAttrOfType<IntegerAttr>("vrc.kernel_id").getInt()),
         kernel->getAttrOfType<DenseI32ArrayAttr>("known_grid_size")[0],
         kernel->getAttrOfType<IntegerAttr>("vrc.words").getInt()});
  }
  return device;
}
} // namespace
int main(int argc, char **argv) {
  InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                              "udoncl C/OpenCL C artifact compiler\n");
  auto phase = [](StringRef name) {
    if (debugCompilerPhases) {
      errs() << "udonclc-phase=" << name << "\n";
      errs().flush();
    }
  };
  if (int(emitBundle) + int(emitLLVMBundle) + int(checkRequest) +
          int(emitNativeDevice) + int(emitUnityLibrary) +
          int(planHostBufferCapacity) + int(emitOpenCLUserKernels) !=
      1) {
    errs() << "select exactly one boundary mode\n";
    return 1;
  }
  if (!deviceMLIR.empty() && (!emitNativeDevice || !deviceLLVM.empty())) {
    errs() << "--device-mlir replaces --device-llvm and requires "
              "--emit-native-device-bundle\n";
    return 1;
  }
  if (runtimeValidation && noRuntimeValidation) {
    errs() << "--runtime-validation and --no-runtime-validation are mutually "
              "exclusive\n";
    return 1;
  }
  if (emitOpenCLUserKernels) {
    if (outputDirectory.empty() || deviceLLVM.empty()) {
      errs() << "OpenCL user kernel declarations require a Host LLVM input, "
                "--device-llvm and -o directory\n";
      return 1;
    }
    LLVMContext llvmContext;
    SMDiagnostic diagnostic;
    auto hostModule = parseIRFile(inputFilename, diagnostic, llvmContext);
    if (!hostModule) {
      diagnostic.print(argv[0], errs());
      return 1;
    }
    auto deviceModule = parseIRFile(deviceLLVM, diagnostic, llvmContext);
    if (!deviceModule) {
      diagnostic.print(argv[0], errs());
      return 1;
    }
    auto kernels =
        udoncl::discoverOpenCLUserKernels(*hostModule, *deviceModule, errs());
    return failed(kernels) || failed(udoncl::writeOpenCLUserKernelHeaders(
                                  *kernels, outputDirectory, errs()))
               ? 1
               : 0;
  }
  if (planHostBufferCapacity) {
    if (outputDirectory.empty() || capacityHeader.empty()) {
      errs() << "Host buffer planning requires -o plan.json and "
                "--capacity-header\n";
      return 1;
    }
    LLVMContext llvmContext;
    SMDiagnostic diagnostic;
    auto inputModule = parseIRFile(inputFilename, diagnostic, llvmContext);
    if (!inputModule) {
      diagnostic.print(argv[0], errs());
      return 1;
    }
    auto plan = udoncl::planHostBuffers(*inputModule, baseMutableHandleCapacity,
                                        errs());
    return failed(plan) || failed(udoncl::writeHostBufferPlan(
                               *plan, outputDirectory, capacityHeader, errs()))
               ? 1
               : 0;
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
  func::registerInlinerExtension(registry);
  registry.insert<::vrc::udonasm::UdonAsmDialect, ::vrc::udonhl::UdonHLDialect,
                  ::vrc::vrc::VRCDialect, ::vrc::vrcgpu::VRCGPUDialect,
                  arith::ArithDialect, async::AsyncDialect,
                  cf::ControlFlowDialect, func::FuncDialect, gpu::GPUDialect,
                  math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module;
  std::map<std::string, std::string> files;
  std::map<std::string, std::string> externalFiles;
  bool gatherProfile = false;
  bool unityGatherProfile = false;
  std::optional<json::Object> unityConfig;
  OwningOpRef<ModuleOp> unityDeviceModule;
  // Synthesized kernels taken out of the Host MLIR; they join the library
  // kernels in unityDeviceModule.
  OwningOpRef<ModuleOp> synthesizedDeviceModule;
  SmallVector<const json::Object *> unityDeviceConfigs;
  SmallVector<SynthesizedUnityKernel> unitySynthesizedKernels;
  // The user OpenCL C kernels the linked OpenCL AOT runtime dispatches, by id.
  std::map<int64_t, std::string> unityUserKernels;
  SmallVector<UnityBuffer> unityBuffers;
  bool unityUsesBufferCopy = false;
  // The texel format of every Device buffer and immutable tensor texture of
  // a Unity library (`device_texel_format`).
  std::string unityTexelFormat = "rgba8_unorm";
  if (!emitUnityLibrary &&
      (!bindingNamespace.empty() || !bindingClass.empty() ||
       !deviceTexelFormat.empty() || !deviceWorkBudget.empty())) {
    errs() << "--binding-namespace, --binding-class, --device-texel-format "
              "and --device-work-budget require --emit-unity-library-bundle\n";
    return 1;
  }
  if (inputFilename.empty() && !emitUnityLibrary) {
    errs() << "this mode requires an input file\n";
    return 1;
  }
  if (emitUnityLibrary) {
    phase("unity-prepare");
    // The Device libraries are imported only once the Host is recovered, so
    // the dialects their import used to load come first: the Host recovery
    // builds their operations, and the Host MLIR may hold LLVM types.
    context.loadAllAvailableDialects();
    context.getOrLoadDialect<LLVM::LLVMDialect>();
    if (hostLLVM.empty() && hostMLIR.empty()) {
      errs() << "Unity library mode requires at least one of --host-llvm or "
                "--host-mlir; --host-library-llvm supplies the separately "
                "linked OpenCL AOT runtime\n";
      return 1;
    }
    // The Host source and the command-line settings describe the library;
    // the Unity library config file is removed.
    if (!inputFilename.empty()) {
      errs()
          << "a Unity library takes no config file (" << inputFilename
          << "): name the program with UDONCL_PROGRAM(\"name\"), the "
             "exports with UDONCL_EXPORT and the public buffers with "
             "UDONCL_PUBLIC in the Host source, and pass the binding, "
             "texel format and Device work budget as udoncl_add_unity_library "
             "arguments (--binding-namespace, --binding-class, "
             "--device-texel-format, --device-work-budget)\n";
      return 1;
    }
    unityConfig = json::Object{};
    if (failed(applyUnityLibraryOptions(*unityConfig)))
      return 1;
    auto texelFormat =
        configTexelFormat(*unityConfig, "device_texel_format",
                          "--device-texel-format (DEVICE_TEXEL_FORMAT)");
    if (failed(texelFormat))
      return 1;
    unityTexelFormat = *texelFormat;
    // The Device kernels are derived from the Host's dispatches once it is
    // recovered.
    // The Host MLIR may carry synthesized Device kernels; they leave it here
    // and join the configured library kernels.
    if (!hostMLIR.empty()) {
      module = parseSourceFile<ModuleOp>(hostMLIR, &context);
      if (!module || failed(verify(*module)))
        return 1;
      auto extracted = extractSynthesizedUnityKernels(
          *module, unityDeviceConfigs, unitySynthesizedKernels);
      if (failed(extracted))
        return 1;
      synthesizedDeviceModule = std::move(*extracted);
      if (synthesizedDeviceModule && failed(verify(*module)))
        return 1;
    }
    if (!hostMLIR.empty()) {
      if (!hostLLVM.empty()) {
        auto program = recoverRuntimeFreeUnityHost(hostLLVM, context);
        if (failed(program) || failed(linkUnityHostLibrary(*module, **program)))
          return 1;
      }
      if (!configureUnityExports(*module, *unityConfig,
                                 /*finalizeReachability=*/false) ||
          failed(discoverStaticDeviceResources(*module)))
        return 1;
      if (!hostLibraryLLVM.empty()) {
        auto library = recoverLinkedUnityHostLibrary(hostLibraryLLVM, context);
        if (failed(library) || failed(linkUnityHostLibrary(*module, **library)))
          return 1;
      }
      {
        auto described =
            udoncl::describeOpenCLUserDispatches(*module, hostLLVM, errs());
        if (failed(described))
          return 1;
        unityUserKernels = std::move(*described);
      }
      {
        PassManager linkedHostRecovery(&context);
        linkedHostRecovery.enableVerifier(false);
        linkedHostRecovery.addPass(::vrc::createLowerOpenCLAOTIntrinsicsPass());
        if (failed(linkedHostRecovery.run(*module)) || failed(verify(*module)))
          return 1;
      }
      if (failed(lowerUnityHostObjectIntrinsics(*module, context)) ||
          !configureUnityExports(*module, *unityConfig))
        return 1;
    } else if (!hostLibraryLLVM.empty()) {
      auto program = recoverRuntimeFreeUnityHost(hostLLVM, context);
      if (failed(program))
        return 1;
      module = std::move(*program);
      if (!configureUnityExports(*module, *unityConfig,
                                 /*finalizeReachability=*/false) ||
          failed(discoverStaticDeviceResources(*module)))
        return 1;
      auto library = recoverLinkedUnityHostLibrary(hostLibraryLLVM, context);
      if (failed(library) || failed(linkUnityHostLibrary(*module, **library)))
        return 1;
      {
        auto described =
            udoncl::describeOpenCLUserDispatches(*module, hostLLVM, errs());
        if (failed(described))
          return 1;
        unityUserKernels = std::move(*described);
      }
      {
        PassManager linkedHostRecovery(&context);
        linkedHostRecovery.enableVerifier(false);
        linkedHostRecovery.addPass(::vrc::createLowerOpenCLAOTIntrinsicsPass());
        if (failed(linkedHostRecovery.run(*module)) || failed(verify(*module)))
          return 1;
      }
      if (failed(lowerUnityHostObjectIntrinsics(*module, context)) ||
          !configureUnityExports(*module, *unityConfig))
        return 1;
    } else {
      LLVMContext llvmContext;
      SMDiagnostic diagnostic;
      auto inputModule = parseIRFile(hostLLVM, diagnostic, llvmContext);
      if (!inputModule) {
        diagnostic.print(argv[0], errs());
        return 1;
      }
      normalizeUnityHostCallingConventions(*inputModule);
      auto annotations = collectUdonCLHostAnnotations(*inputModule);
      if (failed(annotations))
        return 1;
      if (inputModule->getTargetTriple().str() != "spir-unknown-unknown" ||
          !inputModule->getDataLayout().isLittleEndian() ||
          inputModule->getDataLayout().getPointerSizeInBits() != 32) {
        errs()
            << "Unity Host library requires 32-bit little-endian SPIR input\n";
        return 1;
      }
      bool linkedOpenCLAOT =
          inputModule->getFunction("__vrc_opencl_buffer_alloc") != nullptr;
      if (!linkedOpenCLAOT &&
          failed(rejectUnlinkedOpenCLHostCalls(*inputModule)))
        return 1;
      module = ::vrc::importLLVM(std::move(inputModule), context);
      if (!module || failed(verify(*module)))
        return 1;
      PassManager cleanup(&context);
      cleanup.addPass(createSROA());
      cleanup.addPass(createMem2Reg());
      cleanup.addPass(createCanonicalizerPass());
      if (failed(cleanup.run(*module)))
        return 1;
      PassManager recovery(&context);
      recovery.enableVerifier(false);
      recovery.addPass(createCanonicalizerPass());
      ::vrc::RaiseLLVMMemoryPassOptions memoryOptions;
      memoryOptions.allowRawFallback = true;
      recovery.addPass(::vrc::createRaiseLLVMMemoryPass(memoryOptions));
      recovery.addPass(::vrc::createRaiseLLVMRawMemoryPass());
      recovery.addPass(::vrc::createRaiseLLVMToScalarPass());
      recovery.addPass(createCanonicalizerPass());
      if (linkedOpenCLAOT)
        recovery.addPass(::vrc::createLowerOpenCLAOTIntrinsicsPass());
      if (failed(recovery.run(*module)) ||
          failed(markRecoveredHostAnnotations(*module, *annotations)) ||
          failed(lowerUnityHostObjectIntrinsics(*module, context)) ||
          !configureUnityExports(*module, *unityConfig))
        return 1;
      // Without the static Device resource analysis nothing resolves a
      // UDONCL_PUBLIC variable; the pass rejects one.
      PassManager publicResources(&context);
      publicResources.addPass(
          ::vrc::createClassifyAnnotatedPublicDeviceResourcesPass());
      if (failed(publicResources.run(*module)))
        return 1;
    }
    // Every Device kernel the Host dispatches is known once the Host has
    // reached its exported form: the library kernels it dispatches are
    // derived from their dispatch sites (vrc-describe-unity-device-kernels).
    phase("unity-device-kernels");
    {
      SmallVector<int64_t> synthesizedKernelIds;
      for (const SynthesizedUnityKernel &kernel : unitySynthesizedKernels)
        synthesizedKernelIds.push_back(kernel.id);
      auto derived = describeUnityDeviceKernels(*module, synthesizedKernelIds,
                                                unityUserKernels);
      if (failed(derived))
        return 1;
      if (!derived->empty() && deviceLLVM.empty()) {
        errs() << "the Unity Host dispatches the Device library kernels";
        for (const json::Value &kernel : *derived)
          errs() << " " << *kernel.getAsObject()->getString("kernel");
        errs() << " but no Device library is linked; add their Device "
                  "libraries to DEVICE_LIBRARIES (--device-llvm)\n";
        return 1;
      }
      if (!derived->empty()) {
        json::Array kernels;
        for (json::Value &kernel : *derived)
          kernels.push_back(std::move(kernel));
        (*unityConfig)["device_kernels"] = std::move(kernels);
        auto merged = parseUnityDeviceKernels(*unityConfig);
        if (failed(merged))
          return 1;
        unityDeviceConfigs = std::move(*merged);
      }
      if (!deviceLLVM.empty() && unityDeviceConfigs.empty()) {
        errs() << "--device-llvm is given but the Unity Host dispatches no "
                  "Device kernel\n";
        return 1;
      }
    }
    // The Device buffers of an ONNX Host come from its execution plan and
    // binding (vrc-derive-onnx-unity-buffers). The Device import below takes
    // its texture shape and row pitch decisions from these buffers.
    phase("unity-buffers");
    if (!hostBufferPlan.empty() &&
        failed(udoncl::applyHostBufferPlan(*module, hostBufferPlan)))
      return 1;
    if (failed(deriveOnnxUnityBufferConfig(*module, *unityConfig)))
      return 1;
    auto parsedBuffers = parseUnityBuffers(*unityConfig);
    if (failed(parsedBuffers))
      return 1;
    unityBuffers = std::move(*parsedBuffers);
    if (!deviceLLVM.empty()) {
      unityDeviceModule = ModuleOp::create(UnknownLoc::get(&context));
      unityGatherProfile = true;
      for (const json::Object *deviceConfig : unityDeviceConfigs) {
        bool gather = deviceConfig->getString("profile") == "gather-f32";
        if (unityDeviceConfigs.size() > 1 && !gather) {
          errs() << "multi-kernel Unity libraries currently require the "
                    "gather-f32 profile\n";
          return 1;
        }
        unityGatherProfile &= gather;
        auto recovered =
            ::vrc::importCBoundary({}, deviceLLVM, *deviceConfig, context);
        if (!recovered || failed(verify(*recovered)))
          return 1;
        PassManager deviceRecovery(&context);
        deviceRecovery.addNestedPass<ModuleOp>(createMem2Reg());
        if (gather) {
          deviceRecovery.addNestedPass<ModuleOp>(createCanonicalizerPass());
          deviceRecovery.addNestedPass<ModuleOp>(
              ::vrc::createRaiseGatherDevicePass());
        } else
          deviceRecovery.addNestedPass<ModuleOp>(
              ::vrc::createRaiseLLVMBoundaryDevicePass());
        deviceRecovery.addNestedPass<ModuleOp>(
            ::vrc::createRaiseLLVMToScalarPass());
        deviceRecovery.addNestedPass<ModuleOp>(createCanonicalizerPass());
        if (gather) {
          deviceRecovery.addNestedPass<ModuleOp>(
              createLiftControlFlowToSCFPass());
          deviceRecovery.addNestedPass<ModuleOp>(createCanonicalizerPass());
        }
        deviceRecovery.addPass(::vrc::createPrepareNativeDevicePass());
        if (failed(deviceRecovery.run(*recovered)) ||
            failed(verify(*recovered)))
          return 1;
        auto modules = recovered->getOps<gpu::GPUModuleOp>();
        if (!llvm::hasSingleElement(modules)) {
          errs() << "recovered Unity Device kernel is missing its GPU module\n";
          return 1;
        }
        auto gpuModule = *modules.begin();
        int64_t kernelId = *deviceConfig->getInteger("kernel_id");
        gpuModule->setAttr(
            SymbolTable::getSymbolAttrName(),
            StringAttr::get(&context,
                            "device_gpu_" + std::to_string(kernelId)));
        gpuModule->moveBefore(unityDeviceModule->getBody(),
                              unityDeviceModule->getBody()->end());
      }
    }
    if (synthesizedDeviceModule) {
      if (!unityDeviceModule) {
        unityDeviceModule = ModuleOp::create(UnknownLoc::get(&context));
        unityGatherProfile = true;
      } else if (!unityGatherProfile) {
        errs() << "synthesized kernels require library kernels of the "
                  "gather-f32 profile\n";
        return 1;
      }
    }
    if (unityDeviceModule) {
      int64_t maximumKernelWords = 0;
      std::optional<int64_t> commonKernelWidth;
      bool commonKernelWidthValid = true;
      for (const json::Object *deviceConfig : unityDeviceConfigs) {
        if (auto words = deviceConfig->getInteger("words"))
          maximumKernelWords = std::max(maximumKernelWords, *words);
        auto *global = deviceConfig->getArray("global_size");
        auto width = global && global->size() == 3 ? (*global)[0].getAsInteger()
                                                   : std::optional<int64_t>();
        if (!width)
          commonKernelWidthValid = false;
        else if (!commonKernelWidth)
          commonKernelWidth = *width;
        else if (*commonKernelWidth != *width)
          commonKernelWidthValid = false;
      }
      // A synthesized kernel states its own launch shape. It takes part in
      // the same shape decisions.
      for (const SynthesizedUnityKernel &kernel : unitySynthesizedKernels) {
        maximumKernelWords = std::max(maximumKernelWords, kernel.words);
        if (!commonKernelWidth)
          commonKernelWidth = kernel.width;
        else if (*commonKernelWidth != kernel.width)
          commonKernelWidthValid = false;
      }
      if (!unityBuffers.empty() && commonKernelWidthValid &&
          commonKernelWidth &&
          llvm::all_of(unityBuffers, [&](const UnityBuffer &buffer) {
            return buffer.width == unityBuffers.front().width &&
                   buffer.width == *commonKernelWidth;
          }))
        (*unityDeviceModule)
            ->setAttr("vrc.texture_row_pitch",
                      mlir::IntegerAttr::get(
                          mlir::IntegerType::get(&context, 32),
                          static_cast<int32_t>(unityBuffers.front().width)));
      if (!unityBuffers.empty() &&
          llvm::all_of(unityBuffers,
                       [&](const UnityBuffer &buffer) {
                         return buffer.width == unityBuffers.front().width &&
                                buffer.height == unityBuffers.front().height;
                       }) &&
          unityBuffers.front().width * unityBuffers.front().height >=
              maximumKernelWords)
        (*unityDeviceModule)
            ->setAttr("vrc.texture_shape",
                      DenseI32ArrayAttr::get(
                          &context,
                          {static_cast<int32_t>(unityBuffers.front().width),
                           static_cast<int32_t>(unityBuffers.front().height)}));
      for (const SynthesizedUnityKernel &kernel : unitySynthesizedKernels)
        kernel.module->moveBefore(unityDeviceModule->getBody(),
                                  unityDeviceModule->getBody()->end());
      setModuleTexelFormat(*unityDeviceModule, unityTexelFormat);
      if (failed(verify(*unityDeviceModule)))
        return 1;
    }
    if ((*module)->hasAttr("vrc.static_device_resource_provenance")) {
      PassManager kernelBindings(&context);
      // Export configuration has erased unreachable Host functions since the
      // transfer bindings were first resolved. Recompute them so their
      // summary describes only the transfers that survive into the artifact.
      kernelBindings.addPass(::vrc::createResolveStaticTransferBindingsPass());
      kernelBindings.addPass(::vrc::createResolveStaticKernelBindingsPass());
      if (failed(kernelBindings.run(*module)) || failed(verify(*module)))
        return 1;
    }
    if (failed(reconcileStaticUnityBuffers(module, unityBuffers)))
      return 1;
    // A Host whose Device buffers are already assigned (hand-written Host
    // MLIR) and whose config lists none has exactly the buffers it names.
    if (!unityConfig->get("buffers")) {
      PassManager assignedBuffers(&context);
      assignedBuffers.addPass(::vrc::createPlanAssignedUnityBufferLayoutPass());
      if (failed(assignedBuffers.run(*module)))
        return 1;
    }
    if (failed(materializeCanonicalUnityBufferConfig(*module, *unityConfig)))
      return 1;
    if (reportStaticResources) {
      ScopedDiagnosticHandler report(&context, [](Diagnostic &diagnostic) {
        if (diagnostic.getSeverity() != mlir::DiagnosticSeverity::Remark)
          return failure();
        errs() << "udonclc-static-resources: " << diagnostic << '\n';
        return success();
      });
      PassManager reporting(&context);
      reporting.addPass(::vrc::createReportStaticUnityResourceLoweringPass());
      if (failed(reporting.run(*module)))
        return 1;
    }
    {
      PassManager required(&context);
      required.addPass(::vrc::createRequireStaticUnityResourceLoweringPass());
      if (failed(required.run(*module)))
        return 1;
    }
    auto generatedBuffers = parseUnityBuffers(*unityConfig);
    if (failed(generatedBuffers))
      return 1;
    unityBuffers = std::move(*generatedBuffers);
    // The texture shape the Unity buffers share must hold the output of a
    // synthesized kernel.
    if (unityDeviceModule) {
      ::vrc::CheckSynthesizedKernelTextureCapacityPassOptions capacity;
      for (const UnityBuffer &buffer : unityBuffers) {
        capacity.bufferWidths.push_back(buffer.width);
        capacity.bufferHeights.push_back(buffer.height);
      }
      PassManager textureCapacity(&context);
      textureCapacity.addPass(
          ::vrc::createCheckSynthesizedKernelTextureCapacityPass(capacity));
      if (failed(textureCapacity.run(*unityDeviceModule)))
        return 1;
    }
    if (unityTexelFormat != "rgba8_unorm") {
      int64_t texelWords = ::vrc::vrcgpu::gatherTexelWords(unityTexelFormat);
      for (const UnityBuffer &buffer : unityBuffers)
        if (buffer.width % texelWords != 0) {
          errs() << "Device buffer " << buffer.name << " is " << buffer.width
                 << " words wide; " << unityTexelFormat
                 << " texels need a multiple of " << texelWords << "\n";
          return 1;
        }
      setModuleTexelFormat(*module, unityTexelFormat);
    }
    if (!omitIntermediateIR) {
      raw_string_ostream recovered(files["frontend.recovered.mlir"]);
      module->print(recovered);
      recovered << "\n";
    }
  } else if (emitNativeDevice && !deviceMLIR.empty()) {
    auto input = MemoryBuffer::getFile(inputFilename);
    if (!input) {
      errs() << "native Device bundle mode requires a JSON config\n";
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
    gatherProfile = true;
    auto texelFormat = configTexelFormat(*config->getAsObject(), "texel_format",
                                         "texel_format");
    if (failed(texelFormat))
      return 1;
    module = importSynthesizedDeviceKernel(deviceMLIR, *config->getAsObject(),
                                           context);
    if (!module || failed(verify(*module)))
      return 1;
    setModuleTexelFormat(*module, *texelFormat);
    raw_string_ostream imported(files["frontend.imported.mlir"]);
    module->print(imported);
    imported << "\n";
    // No LLVM-level gather proof covers a synthesized kernel; the lowering
    // below only accepts it once its buffer accesses are proven here.
    PassManager proof(&context);
    proof.addPass(::vrc::createProveSynthesizedGatherKernelsPass());
    if (failed(proof.run(*module)))
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
    auto texelFormat = configTexelFormat(*config->getAsObject(), "texel_format",
                                         "texel_format");
    if (failed(texelFormat))
      return 1;
    module = ::vrc::importCBoundary(
        emitNativeDevice ? StringRef() : StringRef(hostLLVM), deviceLLVM,
        *config->getAsObject(), context);
    if (!module || failed(verify(*module)))
      return 1;
    setModuleTexelFormat(*module, *texelFormat);
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
  (*module)->setAttr("vrc.runtime_validation.read_write_alias",
                     BoolAttr::get(&context, runtimeValidation));
  if (debugCompilerPhases)
    (*module)->setAttr("vrc.debug_compiler_phases", UnitAttr::get(&context));
  PassManager manager(&context);
  if (emitUnityLibrary) {
    // Immutable tensor payloads can be hundreds of megabytes. StaticBufferOps
    // already carry the IDs and sizes required by Host lowering, so retaining
    // DenseElementsAttr payloads through module cloning, async passes and
    // textual intermediate-IR emission only multiplies memory and compile
    // time. Keep the uniqued attribute for validation and final artifact
    // packing, but remove it from the executable IR before those operations.
    ArrayAttr unityStaticTensors =
        (*module)->getAttrOfType<ArrayAttr>("udoncl.static_tensors");
    (*module)->removeAttr("udoncl.static_tensors");
    SmallVector<int64_t> configuredKernelIds;
    for (const json::Object *deviceKernel : unityDeviceConfigs) {
      int64_t kernelId = *deviceKernel->getInteger("kernel_id");
      configuredKernelIds.push_back(kernelId);
    }
    for (const SynthesizedUnityKernel &kernel : unitySynthesizedKernels) {
      configuredKernelIds.push_back(kernel.id);
    }
    DeviceWorkFrameBudget frameBudget;
    phase("host-object-intrinsics");
    {
      PassManager hostObjects(&context);
      hostObjects.addPass(::vrc::createLowerHostObjectIntrinsicsPass());
      if (failed(hostObjects.run(*module)) || failed(verify(*module)))
        return 1;
    }
    // A static constructor is evaluated once every allocation is a compiler-
    // assigned buffer and while it is still a plain function: what it leaves
    // of itself decides the async effects computed next.
    DictionaryAttr evaluatedConstructor;
    if (auto initialization = (*module)->getAttrOfType<DictionaryAttr>(
            "vrc.static_initialization")) {
      phase("static-initialization");
      // The evaluation is read before SymbolDCE: a constructor that is not
      // also an export leaves the module once it has run.
      PassManager staticInitialization(&context);
      staticInitialization.addPass(::vrc::createEvaluateHostConstructorsPass());
      if (failed(staticInitialization.run(*module)) || failed(verify(*module)))
        return 1;
      if (auto constructor = module->lookupSymbol<func::FuncOp>(
              initialization.getAs<StringAttr>("constructor").getValue()))
        evaluatedConstructor = constructor->getAttrOfType<DictionaryAttr>(
            "vrc.evaluated_static_constructor");
      PassManager unreachable(&context);
      unreachable.addPass(createSymbolDCEPass());
      if (failed(unreachable.run(*module)) || failed(verify(*module)))
        return 1;
      if (!evaluatedConstructor) {
        errs() << "the Host constructor (__attribute__((constructor))) "
                  "was not evaluated\n";
        return 1;
      }
      if (!omitIntermediateIR) {
        raw_string_ostream evaluated(files["host.static-initialized.mlir"]);
        module->print(evaluated);
        evaluated << "\n";
      }
    }
    phase("host-effects");
    if (failed(propagateUnityHostAsyncEffects(*module)))
      return 1;
    // After the async effects fix the export ABI: Device commands become
    // enqueue-only and yield by slice budget instead of one frame each.
    phase("schedule-device-work");
    {
      if (failed(attachDeviceFragmentCosts(*module, unityDeviceConfigs)))
        return 1;
      ::vrc::ScheduleUdonDeviceWorkPassOptions budget;
      if (failed(readDeviceWorkBudget(*unityConfig, budget, frameBudget)))
        return 1;
      PassManager deviceWork(&context);
      deviceWork.addPass(::vrc::createScheduleUdonDeviceWorkPass(budget));
      if (failed(deviceWork.run(*module)) || failed(verify(*module)))
        return 1;
    }
    phase("host-storage");
    if (failed(optimizeUnityHostStorage(*module)) || failed(verify(*module)))
      return 1;
    phase("async-clone");
    OwningOpRef<ModuleOp> asyncPreview = cast<ModuleOp>((*module)->clone());
    printCompilerIRStats(*asyncPreview, "host-storage");
    auto runAsyncStage = [&](StringRef name,
                             std::unique_ptr<Pass> pass) -> LogicalResult {
      phase(name);
      PassManager stage(&context);
      stage.addPass(std::move(pass));
      if (failed(stage.run(*asyncPreview)) || failed(verify(*asyncPreview)))
        return failure();
      printCompilerIRStats(*asyncPreview, name);
      return success();
    };
    if (failed(runAsyncStage("host-to-async",
                             ::vrc::createLowerHostToAsyncPass())) ||
        failed(runAsyncStage("async-func-to-runtime",
                             createAsyncFuncToAsyncRuntimePass())) ||
        failed(runAsyncStage("prepare-udon-async-runtime",
                             ::vrc::createPrepareUdonAsyncRuntimePass())) ||
        failed(
            runAsyncStage("discover-host-rematerialization",
                          ::vrc::createDiscoverHostRematerializationPass())) ||
        failed(runAsyncStage("allocate-host-storage",
                             ::vrc::createAllocateHostStoragePass())) ||
        failed(runAsyncStage("analyze-host-storage",
                             ::vrc::createAnalyzeHostStoragePass())))
      return 1;
    if (!omitIntermediateIR) {
      raw_string_ostream asyncIR(files["host.async-runtime.mlir"]);
      asyncPreview->print(asyncIR);
      asyncIR << "\n";
    }

    PassManager udonAsyncPipeline(&context);
    udonAsyncPipeline.addPass(::vrc::createLowerUdonAsyncRuntimePass());
    phase("async-to-udon-runtime");
    if (failed(udonAsyncPipeline.run(*asyncPreview)) ||
        failed(verify(*asyncPreview)))
      return 1;
    phase("bind-async-buffers");
    auto asyncBufferBinding = bindUnityBufferOperations(
        *asyncPreview, unityBuffers, configuredKernelIds, unityStaticTensors);
    if (failed(asyncBufferBinding))
      return 1;
    (*asyncPreview)
        ->setAttr("vrc.external_raw_heap_initialization",
                  UnitAttr::get(&context));
    if (!omitIntermediateIR) {
      raw_string_ostream udonAsyncIR(files["host.udon-async.mlir"]);
      asyncPreview->print(udonAsyncIR);
      udonAsyncIR << "\n";
    }

    phase("host-to-udon-hl");
    manager.addPass(::vrc::createLowerHostToUdonHLPass());
    manager.addPass(::vrc::createSubstituteRuntimeIntrinsicsPass());
    if (failed(manager.run(*module)) || failed(verify(*module)))
      return 1;
    phase("bind-sync-buffers");
    auto bufferBinding = bindUnityBufferOperations(
        *module, unityBuffers, configuredKernelIds, unityStaticTensors);
    if (failed(bufferBinding))
      return 1;
    unityUsesBufferCopy = bufferBinding->usesCopy;
    auto manifest = ::vrc::buildBoundaryManifest(*module);
    auto onnxBinding = unityOnnxBindingManifest(*module);
    if (failed(onnxBinding))
      return 1;
    if (*onnxBinding)
      manifest["onnx_binding"] = std::move(**onnxBinding);
    bool usesAsyncCommands = false;
    module->walk([&](Operation *operation) {
      usesAsyncCommands |=
          isa<::vrc::vrc::BufferUploadOp, ::vrc::vrc::BufferReadbackOp,
              ::vrc::vrc::AwaitOp>(operation);
    });
    // vrc-configure-unity-exports resolved the program name from
    // UDONCL_PROGRAM or the configuration.
    auto programAttr = (*module)->getAttrOfType<StringAttr>("vrc.program");
    if (!programAttr) {
      errs() << "Unity library export configuration named no program\n";
      return 1;
    }
    std::optional<StringRef> program = programAttr.getValue();
    manifest["artifact_stage"] = "unity-udoncl-library";
    json::Object programManifest{{"name", *program},
                                 {"assembly", "host.uasm"},
                                 {"display_shader", "display.shader"}};
    if (!omitIntermediateIR) {
      programManifest["async_ir"] = "host.async-runtime.mlir";
      programManifest["udon_async_ir"] = "host.udon-async.mlir";
      programManifest["ir"] = "host.udonasm.mlir";
    }
    manifest["program"] = std::move(programManifest);
    if (auto initialization = (*module)->getAttrOfType<DictionaryAttr>(
            "vrc.static_initialization")) {
      json::Array omitted;
      for (mlir::Attribute symbol :
           initialization.getAs<ArrayAttr>("omitted_exports"))
        omitted.push_back(cast<StringAttr>(symbol).getValue().str());
      manifest["static_initialization"] = json::Object{
          {"constructor",
           initialization.getAs<StringAttr>("constructor").getValue().str()},
          {"omitted_exports", std::move(omitted)},
          {"evaluated_operations",
           evaluatedConstructor.getAs<IntegerAttr>("evaluated_operations")
               .getInt()},
          {"residual_commands",
           evaluatedConstructor.getAs<IntegerAttr>("residual_commands")
               .getInt()}};
    }
    if (usesAsyncCommands)
      (*manifest["program"].getAsObject())["async_commands"] = json::Object{
          {"completion_status_slot", "__udoncl_async_completion_status"},
          {"pending_object_slot", "__udoncl_async_pending_object"},
          {"pending_yield_slot", "__udoncl_async_pending_yield"},
          {"completion_object_slot", "__udoncl_async_completion_object"},
          {"completion_event", "__udoncl_async_complete"},
          {"command_kind_slot", "__udoncl_async_command_kind"},
          {"command_buffer_slot", "__udoncl_async_command_buffer"},
          {"command_offset_slot", "__udoncl_async_command_offset"},
          {"command_size_slot", "__udoncl_async_command_size"},
          {"command_pointer_slot", "__udoncl_async_command_pointer"},
          {"raw_heap_slot", "__vrc_raw_heap"},
          {"upload_kind", 1},
          {"readback_kind", 2},
          {"frame_budget_microseconds", frameBudget.microseconds},
          {"frame_continuation_limit", frameBudget.continuations},
      };
    auto *device = manifest["device"].getAsObject();
    if (!device) {
      errs() << "Unity library manifest is missing its Device section\n";
      return 1;
    }
    (*device)["buffers"] = unityBufferManifest(unityBuffers, unityTexelFormat);
    if (unityTexelFormat != "rgba8_unorm") {
      // Word-texel buffers: a transfer that writes part of a texel first
      // copies the destination to the scratch texture and keeps the words
      // outside its range from it; see `unityLibraryBufferCopyShader`.
      auto *programManifest = manifest["program"].getAsObject();
      (*programManifest)["texel_format"] = unityTexelFormat;
      if (!unityBuffers.empty()) {
        int64_t texelWords = ::vrc::vrcgpu::gatherTexelWords(unityTexelFormat);
        int64_t width = 0, height = 0;
        for (const UnityBuffer &buffer : unityBuffers) {
          width = std::max(width, buffer.width / texelWords);
          height = std::max(height, buffer.height);
        }
        (*programManifest)["transfer_scratch"] =
            json::Object{{"slot", "__udoncl_transfer_scratch"},
                         {"width", width},
                         {"height", height},
                         {"format", unityTexelFormat}};
      }
    }
    if (!unityBuffers.empty())
      (*manifest["program"].getAsObject())["buffer_pool_slot"] =
          "__udoncl_buffer_pool";
    if (auto capacity = (*module)->getAttrOfType<IntegerAttr>(
            "udoncl.mutable_handle_capacity"))
      (*device)["mutable_handle_capacity"] = capacity.getInt();
    // Every Unity library ships its U# binding, named by
    // vrc-configure-unity-exports, except one whose async exports have no
    // Device buffer to carry the async command runtime: no binding runs
    // that.
    bool asyncExports = false;
    if (const json::Array *exports = manifest.getArray("exports"))
      for (const json::Value &value : *exports)
        if (const json::Object *entry = value.getAsObject())
          asyncExports |= entry->getBoolean("async").value_or(false);
    if (!asyncExports || !unityBuffers.empty()) {
      auto bindingNamespaceAttr =
          (*module)->getAttrOfType<StringAttr>("vrc.binding_namespace");
      auto bindingClassAttr =
          (*module)->getAttrOfType<StringAttr>("vrc.binding_class");
      if (!bindingNamespaceAttr || !bindingClassAttr) {
        errs() << "Unity library export configuration named no binding\n";
        return 1;
      }
      std::string bindingNamespace = bindingNamespaceAttr.getValue().str();
      std::string bindingClass = bindingClassAttr.getValue().str();
      auto *programManifest = manifest["program"].getAsObject();
      (*programManifest)["binding_namespace"] = bindingNamespace;
      (*programManifest)["binding_class"] = bindingClass;
      ::vrc::refreshPublicABIFingerprint(manifest);
      auto binding = ::vrc::buildUdonSharpBindingSource(
          manifest, bindingNamespace, bindingClass);
      if (!binding) {
        errs() << toString(binding.takeError()) << "\n";
        return 1;
      }
      files["binding.cs"] = std::move(*binding);
      (*programManifest)["binding_source"] = "binding.cs";
    }
    if (unityUsesBufferCopy || !unityBuffers.empty())
      (*manifest["program"].getAsObject())["buffer_copy_shader"] =
          "buffer-copy.shader";
    if (unityUsesBufferCopy)
      (*manifest["program"].getAsObject())["buffer_copy_program_slot"] = true;
    manifest["ready_for_unity"] = true;
    if (unityDeviceModule) {
      phase("device-lowering");
      PassManager deviceLowering(&context);
      if (unityGatherProfile)
        deviceLowering.addPass(::vrc::createLowerGatherDevicePass());
      else
        deviceLowering.addPass(::vrc::createLowerDeviceSingleInvocationPass());
      if (failed(deviceLowering.run(*unityDeviceModule)) ||
          failed(verify(*unityDeviceModule)))
        return 1;
      for (auto kernel : unityDeviceModule->getOps<::vrc::vrcgpu::KernelOp>()) {
        std::string shader, hlsl;
        raw_string_ostream shaderOut(shader), hlslOut(hlsl);
        if (failed(::vrc::emitShaderLab(kernel, shaderOut)) ||
            failed(::vrc::emitFragmentHLSL(kernel, hlslOut)))
          return 1;
        shaderOut.flush();
        hlslOut.flush();
        std::string prefix =
            "kernels/kernel_" + std::to_string(kernel.getKernelId());
        files[prefix + ".shader"] = std::move(shader);
        files[prefix + ".hlsl"] = std::move(hlsl);
      }
      auto deviceManifest = ::vrc::buildBoundaryManifest(*unityDeviceModule);
      auto *compiledDevice = deviceManifest["device"].getAsObject();
      auto *kernels =
          compiledDevice ? compiledDevice->getArray("kernels") : nullptr;
      if (!kernels || kernels->empty()) {
        errs() << "Unity Device module did not produce a kernel manifest\n";
        return 1;
      }
      (*device)["kernels"] = std::move(*kernels);
    }
    ModuleOp physicalModule = usesAsyncCommands ? *asyncPreview : *module;
    physicalModule->setAttr("vrc.external_raw_heap_initialization",
                            UnitAttr::get(&context));
    PassManager udon(&context);
    if (usesAsyncCommands) {
      udon.addPass(::vrc::createLowerHostToUdonHLPass());
      udon.addPass(::vrc::createSubstituteRuntimeIntrinsicsPass());
    }
    phase("udon-hl-to-asm");
    // Again for the globals the async transform introduced.
    udon.addPass(::vrc::createPromoteHostGlobalsToUdonVariablesPass());
    // One material per dispatch site, so the scalar lowering binds each
    // site's fixed arguments once.
    udon.addPass(::vrc::createAssignUnityDispatchMaterialsPass());
    ::vrc::buildUdonHostStorageFinalizationPipeline(udon);
    if (failed(udon.run(physicalModule)) || failed(verify(physicalModule)))
      return 1;
    if (auto initializer = physicalModule->getAttrOfType<DenseI8ArrayAttr>(
            "udonasm.raw_heap_initializer")) {
      ArrayRef<int8_t> bytes = initializer.asArrayRef();
      files["raw-heap.bytes"] = std::string(
          reinterpret_cast<const char *>(bytes.data()), bytes.size());
      auto *programManifest = manifest["program"].getAsObject();
      (*programManifest)["raw_heap_initializer"] = "raw-heap.bytes";
      (*programManifest)["raw_heap_slot"] = "__vrc_raw_heap";
      (*programManifest)["raw_heap_words"] =
          static_cast<int64_t>(llvm::divideCeil(bytes.size(), size_t{4}));
      json::Array *resources = manifest.getArray("resources");
      if (!resources) {
        manifest["resources"] = json::Array();
        resources = manifest.getArray("resources");
      }
      resources->push_back(json::Object{
          {"name", "raw_heap_initializer"},
          {"kind", "bytes"},
          {"path", "raw-heap.bytes"},
          {"bytes", static_cast<int64_t>(bytes.size())},
      });
    }
    if (ArrayAttr tensors = unityStaticTensors) {
      phase("static-tensor-packaging");
      if (auto data =
              physicalModule->getAttrOfType<ArrayAttr>("udonasm.data")) {
        bool hasStaticTensorTable =
            llvm::any_of(data, [](mlir::Attribute attribute) {
              auto entry = dyn_cast<DictionaryAttr>(attribute);
              auto name =
                  entry ? entry.getAs<StringAttr>("name") : StringAttr();
              return name && name.getValue() == "__udoncl_static_tensors";
            });
        if (hasStaticTensorTable) {
          auto *programManifest = manifest["program"].getAsObject();
          (*programManifest)["static_tensor_table_slot"] =
              "__udoncl_static_tensors";
        }
      }
      json::Array tensorManifest;
      json::Array *resources = manifest.getArray("resources");
      if (!resources) {
        manifest["resources"] = json::Array();
        resources = manifest.getArray("resources");
      }
      for (mlir::Attribute attribute : tensors) {
        auto tensor = dyn_cast<DictionaryAttr>(attribute);
        auto id = tensor ? tensor.getAs<IntegerAttr>("id") : IntegerAttr();
        auto name = tensor ? tensor.getAs<StringAttr>("name") : StringAttr();
        auto dataType =
            tensor ? tensor.getAs<StringAttr>("data_type") : StringAttr();
        auto shape = tensor ? tensor.getAs<DenseI64ArrayAttr>("shape")
                            : DenseI64ArrayAttr();
        auto layout =
            tensor ? tensor.getAs<StringAttr>("layout") : StringAttr();
        auto alignment =
            tensor ? tensor.getAs<IntegerAttr>("alignment") : IntegerAttr();
        auto byteStrides = tensor
                               ? tensor.getAs<DenseI64ArrayAttr>("byte_strides")
                               : DenseI64ArrayAttr();
        auto byteLength =
            tensor ? tensor.getAs<IntegerAttr>("byte_length") : IntegerAttr();
        auto sha256 =
            tensor ? tensor.getAs<StringAttr>("sha256") : StringAttr();
        auto data = tensor ? tensor.getAs<DenseElementsAttr>("data")
                           : DenseElementsAttr();
        FailureOr<std::string> inlineBytes = failure();
        std::unique_ptr<MemoryBuffer> externalBuffer;
        std::string externalPath;
        StringRef bytes;
        if (data) {
          inlineBytes = denseTensorBytes(data);
          if (succeeded(inlineBytes))
            bytes = *inlineBytes;
        } else if (id && !staticTensorDirectory.empty()) {
          SmallString<256> sourcePath(staticTensorDirectory);
          sys::path::append(sourcePath,
                            "tensor_" + std::to_string(id.getInt()) + ".bytes");
          auto source = MemoryBuffer::getFile(sourcePath, false, false);
          if (source) {
            externalBuffer = std::move(*source);
            externalPath = sourcePath.str().str();
            bytes = externalBuffer->getBuffer();
          }
        }
        if (!id || !name || !dataType || !shape || !layout || !alignment ||
            !byteStrides || !byteLength || !sha256 || bytes.empty() ||
            layout.getValue() != "row_major_contiguous" ||
            alignment.getInt() < 1 || byteStrides.size() != shape.size() ||
            byteLength.getInt() < 1 ||
            static_cast<uint64_t>(byteLength.getInt()) != bytes.size() ||
            llvm::toHex(llvm::SHA256::hash(ArrayRef<uint8_t>(
                            reinterpret_cast<const uint8_t *>(bytes.data()),
                            bytes.size())),
                        true) != sha256.getValue()) {
          errs() << "immutable static tensor metadata is malformed\n";
          return 1;
        }
        int64_t tensorId = id.getInt();
        std::string resourceName = "static_tensor_" + std::to_string(tensorId);
        std::string path =
            "static-tensors/tensor_" + std::to_string(tensorId) + ".bytes";
        if (externalBuffer)
          externalFiles[path] = std::move(externalPath);
        else
          files[path] = std::move(*inlineBytes);
        // Words fill rows of 8192 (a shorter tensor is one row); a texel
        // holds `texelWords` consecutive words of its row.
        int64_t pixels = llvm::divideCeil(byteLength.getInt(), int64_t{4});
        int64_t rowWords = std::min<int64_t>(pixels, 8192);
        int64_t height = llvm::divideCeil(pixels, rowWords);
        int64_t texelWords = ::vrc::vrcgpu::gatherTexelWords(unityTexelFormat);
        int64_t width = llvm::divideCeil(rowWords, texelWords);
        if (height > 8192) {
          errs() << "immutable static tensor " << tensorId
                 << " exceeds one 8192x8192 RGBA32 texture\n";
          return 1;
        }
        json::Array dimensions;
        for (int64_t dimension : shape.asArrayRef())
          dimensions.push_back(dimension);
        json::Array strides;
        for (int64_t stride : byteStrides.asArrayRef())
          strides.push_back(stride);
        tensorManifest.push_back(json::Object{
            {"id", tensorId},
            {"name", name.getValue().str()},
            {"resource", resourceName},
            {"data_type", dataType.getValue().str()},
            {"shape", std::move(dimensions)},
            {"layout", layout.getValue().str()},
            {"alignment", alignment.getInt()},
            {"byte_strides", std::move(strides)},
            {"byte_length", byteLength.getInt()},
            {"sha256", sha256.getValue().str()},
            {"texture",
             json::Object{{"abi", texelWords == 1 ? "udoncl-static-rgba32-v1"
                                                  : "udoncl-static-words-v1"},
                          {"width", width},
                          {"height", height},
                          {"format", texelWords == 1 ? std::string("rgba32")
                                                     : unityTexelFormat},
                          {"srgb", false},
                          {"mipmaps", false},
                          {"filter", "point"}}},
        });
        resources->push_back(json::Object{
            {"name", resourceName},
            {"kind", "bytes"},
            {"path", path},
            {"bytes", byteLength.getInt()},
        });
      }
      manifest["static_tensors"] = std::move(tensorManifest);
    }
    if (auto report = physicalModule->getAttrOfType<DictionaryAttr>(
            "udonasm.storage_report")) {
      auto serialized = mlirAttributeToJson(report);
      if (!serialized) {
        errs()
            << "Udon Host storage report contains an unsupported attribute\n";
        return 1;
      }
      std::string reportText = formatv("{0:2}\n", *serialized).str();
      files["storage-report.json"] = reportText;
      auto *programManifest = manifest["program"].getAsObject();
      (*programManifest)["storage_report"] = "storage-report.json";
      json::Array *resources = manifest.getArray("resources");
      if (!resources) {
        manifest["resources"] = json::Array();
        resources = manifest.getArray("resources");
      }
      resources->push_back(json::Object{
          {"name", "host_storage_report"},
          {"kind", "json"},
          {"path", "storage-report.json"},
          {"bytes", static_cast<int64_t>(reportText.size())},
      });
    }
    appendUnityBufferPoolSlot(physicalModule, unityBuffers);
    if (unityUsesBufferCopy)
      appendUnityCopyMaterialSlot(physicalModule);
    if (failed(appendUnityDispatchMaterialManifest(physicalModule, manifest)))
      return 1;
    // The Host program declares the transfer scratch slot only when it
    // lowered a partial transfer; the finalizer binds it then.
    if (auto *scratch = (*manifest["program"].getAsObject())
                            .getObject("transfer_scratch")) {
      bool declared = false;
      if (auto data = physicalModule->getAttrOfType<ArrayAttr>("udonasm.data"))
        for (mlir::Attribute attribute : data)
          if (auto entry = dyn_cast<DictionaryAttr>(attribute))
            if (auto name = entry.getAs<StringAttr>("name"))
              declared |= name.getValue() == "__udoncl_transfer_scratch";
      (*scratch)["program_slot"] = declared;
    }
    phase("emit-udon-assembly");
    std::string assembly;
    raw_string_ostream assemblyOut(assembly);
    if (failed(::vrc::emitUdonAssembly(physicalModule, assemblyOut)))
      return 1;
    files["host.uasm"] = std::move(assembly);
    files["display.shader"] = unityLibraryDisplayShader(*program);
    if (unityUsesBufferCopy || !unityBuffers.empty())
      files["buffer-copy.shader"] =
          unityTexelFormat == "rgba8_unorm"
              ? unityLibraryBufferCopyShader(*program)
              : unityLibraryWordTexelCopyShader(*program);
    if (!omitIntermediateIR) {
      raw_string_ostream lowered(files["host.udonasm.mlir"]);
      physicalModule.print(lowered);
      lowered << "\n";
    }
    files["manifest.json"] =
        formatv("{0:2}\n", json::Value(std::move(manifest))).str();
    phase("publish");
    if (!publish(files, externalFiles))
      return 1;
    outs() << "Unity Host library emitted to " << outputDirectory << "\n";
    return 0;
  }
  if (!emitNativeDevice) {
    manager.addPass(::vrc::createLowerHostObjectIntrinsicsPass());
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
