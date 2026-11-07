#include "VRC/ABI/Manifest/Boundary.h"
#include "VRC/ABI/Manifest/BindingSource.h"
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
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"
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
cl::opt<bool> omitIntermediateIR(
    "omit-intermediate-ir",
    cl::desc("Omit diagnostic MLIR snapshots from a Unity library bundle"));
cl::opt<bool> runtimeValidation(
    "runtime-validation",
    cl::desc("Record that read/write physical-resource alias validation is "
             "compiled into the Host runtime"));
cl::opt<bool> noRuntimeValidation(
    "no-runtime-validation",
    cl::desc("Record that read/write physical-resource alias validation is "
             "compiled out of the Host runtime"));
cl::opt<std::string> hostLLVM("host-llvm", cl::desc("Host LLVM IR from C"));
cl::opt<std::string>
    hostMLIR("host-mlir", cl::desc("Typed Host MLIR for compiler validation"));
cl::opt<std::string> hostLibraryLLVM(
    "host-library-llvm",
    cl::desc("Linked Host library LLVM IR used to resolve Host MLIR calls"));
cl::opt<std::string> staticTensorDirectory(
    "static-tensor-directory",
    cl::desc("Directory containing external ONNX static tensor payloads"));
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

struct UnityBuffer {
  int64_t id;
  int64_t logicalBytes;
  int64_t width;
  int64_t height;
  std::string name;
  std::string heapSlot;
  std::string elementType;
  std::string publicAccess;
};

FailureOr<SmallVector<UnityBuffer>>
parseUnityBuffers(const json::Object &config) {
  SmallVector<UnityBuffer> result;
  auto *buffers = config.getArray("buffers");
  if (!buffers)
    return result;
  std::optional<std::pair<int64_t, int64_t>> dispatchShape;
  const json::Object *shapeKernel = config.getObject("device_kernel");
  if (!shapeKernel)
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
    auto explicitWidth = entry ? entry->getInteger("width") : std::nullopt;
    auto explicitHeight = entry ? entry->getInteger("height") : std::nullopt;
    auto elementType = entry ? entry->getString("element_type") : std::nullopt;
    auto publicAccess =
        entry ? entry->getString("public_access") : std::nullopt;
    constexpr int64_t maximumBytes = 8192LL * 8192LL * 4LL;
    bool semanticBinding =
        elementType && publicAccess &&
        (*elementType == "float32" || *elementType == "int32") &&
        (*publicAccess == "input" || *publicAccess == "output");
    if (!id || *id <= 0 || *id > INT32_MAX || !name || !identifier(*name) ||
        !bytes || *bytes <= 0 || *bytes > maximumBytes ||
        (explicitWidth.has_value() != explicitHeight.has_value()) ||
        !ids.insert(*id).second || !names.insert(name->str()).second ||
        ((elementType || publicAccess) && !semanticBinding)) {
      errs() << "invalid or duplicate Unity Device buffer\n";
      return failure();
    }
    int64_t pixels = (*bytes + 3) / 4;
    if (explicitWidth &&
        (*explicitWidth <= 0 || *explicitWidth > 8192 || *explicitHeight <= 0 ||
         *explicitHeight > 8192 || *explicitWidth * *explicitHeight < pixels)) {
      errs() << "invalid Unity Device buffer texture dimensions\n";
      return failure();
    }
    bool shapeFits =
        dispatchShape && dispatchShape->first * dispatchShape->second >= pixels;
    int64_t width = explicitWidth ? *explicitWidth
                    : shapeFits   ? dispatchShape->first
                                  : std::min<int64_t>(pixels, 8192);
    int64_t height = explicitHeight ? *explicitHeight
                     : shapeFits    ? dispatchShape->second
                                    : (pixels + width - 1) / width;
    result.push_back({*id, *bytes, width, height, name->str(),
                      "__udoncl_buffer_" + std::to_string(*id),
                      elementType ? elementType->str() : "",
                      publicAccess ? publicAccess->str() : ""});
  }
  return result;
}

FailureOr<SmallVector<const json::Object *>>
parseUnityDeviceKernels(const json::Object &config) {
  SmallVector<const json::Object *> result;
  auto *single = config.getObject("device_kernel");
  auto *multiple = config.getArray("device_kernels");
  if (single && multiple) {
    errs() << "Unity library config cannot contain both device_kernel and "
              "device_kernels\n";
    return failure();
  }
  if (single)
    result.push_back(single);
  if (multiple) {
    if (multiple->empty()) {
      errs() << "Unity library device_kernels must not be empty\n";
      return failure();
    }
    for (const json::Value &value : *multiple) {
      auto *kernel = value.getAsObject();
      if (!kernel) {
        errs() << "Unity library device_kernels entries must be objects\n";
        return failure();
      }
      result.push_back(kernel);
    }
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

void appendUnityBufferSlots(ModuleOp module, ArrayRef<UnityBuffer> buffers) {
  auto data = module->getAttrOfType<ArrayAttr>("udonasm.data");
  SmallVector<mlir::Attribute> entries(data.begin(), data.end());
  OpBuilder builder(module.getContext());
  for (const UnityBuffer &buffer : buffers) {
    bool exists = llvm::any_of(entries, [&](mlir::Attribute attribute) {
      auto entry = dyn_cast<DictionaryAttr>(attribute);
      auto name = entry ? entry.getAs<StringAttr>("name") : StringAttr();
      return name && name.getValue() == buffer.heapSlot;
    });
    if (exists)
      continue;
    entries.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("name", builder.getStringAttr(buffer.heapSlot)),
         builder.getNamedAttr(
             "type", builder.getStringAttr("UnityEngineRenderTexture")),
         builder.getNamedAttr("initial", builder.getStringAttr("null")),
         builder.getNamedAttr("public", builder.getUnitAttr())}));
  }
  module->setAttr("udonasm.data", builder.getArrayAttr(entries));
}

struct UnityBufferBinding {
  bool usesCopy = false;
  SmallVector<int32_t> dispatchKernelIds;
};

FailureOr<UnityBufferBinding>
bindUnityBufferOperations(ModuleOp module, ArrayRef<UnityBuffer> buffers,
                          ArrayRef<int64_t> configuredKernelIds,
                          ArrayAttr staticTensors = {}) {
  SmallVector<::vrc::vrc::BufferAllocOp> allocations;
  SmallVector<::vrc::vrc::BufferCopyOp> copies;
  SmallVector<::vrc::vrc::DispatchOp> dispatches;
  module.walk([&](::vrc::vrc::BufferAllocOp operation) {
    allocations.push_back(operation);
  });
  module.walk(
      [&](::vrc::vrc::BufferCopyOp operation) { copies.push_back(operation); });
  module.walk([&](::vrc::vrc::DispatchOp operation) {
    dispatches.push_back(operation);
  });
  if (allocations.size() > buffers.size()) {
    module.emitError() << "Unity library has " << allocations.size()
                       << " buffer allocation sites but only " << buffers.size()
                       << " configured Device buffers";
    return failure();
  }
  OpBuilder builder(module.getContext());
  SmallVector<mlir::Attribute> poolEntries;
  for (const UnityBuffer &buffer : buffers)
    poolEntries.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("id", builder.getI32IntegerAttr(buffer.id)),
         builder.getNamedAttr("logical_bytes",
                              builder.getI32IntegerAttr(buffer.logicalBytes)),
         builder.getNamedAttr("heap_slot",
                              builder.getStringAttr(buffer.heapSlot))}));
  module->setAttr("vrc.unity_buffer_pool", builder.getArrayAttr(poolEntries));
  llvm::DenseMap<Operation *, const UnityBuffer *> bound;
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
    if (!buffer || !constant || constant.value() != buffer->logicalBytes)
      return allocation.emitError(
          "constant Unity buffer allocation must match its configured logical "
          "id and byte size");
    allocation->setAttr("vrc.unity_buffer_id",
                        builder.getI32IntegerAttr(buffer->id));
    allocation->setAttr("vrc.unity_heap_slot",
                        builder.getStringAttr(buffer->heapSlot));
    allocation->setAttr("vrc.unity_logical_bytes",
                        builder.getI32IntegerAttr(buffer->logicalBytes));
    bound.try_emplace(allocation.getOperation(), buffer);
  }
  for (auto copy : copies) {
    auto source = copy.getSource().getDefiningOp<::vrc::vrc::BufferAllocOp>();
    auto destination =
        copy.getDestination().getDefiningOp<::vrc::vrc::BufferAllocOp>();
    auto sourceOffset =
        copy.getSourceOffset().getDefiningOp<arith::ConstantIntOp>();
    auto destinationOffset =
        copy.getDestinationOffset().getDefiningOp<arith::ConstantIntOp>();
    auto byteSize = copy.getByteSize().getDefiningOp<arith::ConstantIntOp>();
    auto sourceBuffer =
        source ? bound.find(source.getOperation()) : bound.end();
    auto destinationBuffer =
        destination ? bound.find(destination.getOperation()) : bound.end();
    if (!source || !destination || source == destination ||
        sourceBuffer == bound.end() || destinationBuffer == bound.end() ||
        !sourceOffset || sourceOffset.value() != 0 || !destinationOffset ||
        destinationOffset.value() != 0 || !byteSize ||
        byteSize.value() != sourceBuffer->second->logicalBytes ||
        byteSize.value() != destinationBuffer->second->logicalBytes ||
        sourceBuffer->second->width != destinationBuffer->second->width ||
        sourceBuffer->second->height != destinationBuffer->second->height) {
      copy.emitError("initial Unity CopyBuffer requires distinct direct "
                     "allocations, zero offsets, and equal full buffer shapes");
      return failure();
    }
    copy->setAttr("vrc.unity_copy_material_slot",
                  builder.getStringAttr("__udoncl_buffer_copy_material"));
  }
  UnityBufferBinding result;
  result.usesCopy = !copies.empty();
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
    SmallVector<mlir::Attribute> resourceHeapSlots;
    for (mlir::Value resource : dispatch.getResources()) {
      auto allocation = resource.getDefiningOp<::vrc::vrc::BufferAllocOp>();
      auto staticBuffer = resource.getDefiningOp<::vrc::vrc::StaticBufferOp>();
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
      } else if (allocation) {
        resourcesMatch &= bound.contains(allocation.getOperation());
        auto found = bound.find(allocation.getOperation());
        resourceHeapSlots.push_back(builder.getStringAttr(
            found == bound.end() ? "" : found->second->heapSlot));
      } else if (constantBuffer) {
        resourceHeapSlots.push_back(
            builder.getStringAttr(constantBuffer->heapSlot));
      } else {
        resourcesMatch &= !buffers.empty();
        resourceHeapSlots.push_back(builder.getStringAttr(""));
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
    if (!writeResource) {
      dispatch.emitError("Unity dispatch requires one write resource");
      return failure();
    }
    dispatch->setAttr(
        "vrc.unity_output_resource_index",
        builder.getI32IntegerAttr(static_cast<int32_t>(*writeResource)));
    result.dispatchKernelIds.push_back(kernelId);
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

void appendUnityDispatchMaterialSlots(ModuleOp module,
                                      ArrayRef<int32_t> kernelIds) {
  auto data = module->getAttrOfType<ArrayAttr>("udonasm.data");
  SmallVector<mlir::Attribute> entries(data.begin(), data.end());
  OpBuilder builder(module.getContext());
  for (int32_t kernelId : kernelIds) {
    std::string name = "__udoncl_kernel_material_" + std::to_string(kernelId);
    if (llvm::any_of(entries, [&](mlir::Attribute attribute) {
          auto entry = dyn_cast<DictionaryAttr>(attribute);
          auto existing =
              entry ? entry.getAs<StringAttr>("name") : StringAttr();
          return existing && existing.getValue() == name;
        }))
      continue;
    entries.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("name", builder.getStringAttr(name)),
         builder.getNamedAttr("type",
                              builder.getStringAttr("UnityEngineMaterial")),
         builder.getNamedAttr("initial", builder.getStringAttr("null")),
         builder.getNamedAttr("public", builder.getUnitAttr())}));
  }
  module->setAttr("udonasm.data", builder.getArrayAttr(entries));
}

json::Array unityBufferManifest(ArrayRef<UnityBuffer> buffers) {
  json::Array result;
  for (const UnityBuffer &buffer : buffers) {
    json::Object entry{
        {"id", buffer.id},
        {"name", buffer.name},
        {"heap_slot", buffer.heapSlot},
        {"logical_bytes", buffer.logicalBytes},
        {"width", buffer.width},
        {"height", buffer.height},
        {"format", "rgba8_unorm"},
        {"srgb", false},
        {"mipmaps", false},
        {"filter", "point"},
    };
    if (!buffer.elementType.empty()) {
      entry["element_type"] = buffer.elementType;
      entry["public_access"] = buffer.publicAccess;
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

std::optional<json::Object> unityOnnxBindingManifest(ModuleOp module) {
  auto binding = module->getAttrOfType<DictionaryAttr>("udoncl.onnx_binding");
  if (!binding)
    return std::nullopt;
  auto abi = binding.getAs<StringAttr>("abi");
  auto modelIndex = binding.getAs<IntegerAttr>("model_index");
  auto modelName = binding.getAs<StringAttr>("model_name");
  auto inputs = binding.getAs<ArrayAttr>("inputs");
  auto outputs = binding.getAs<ArrayAttr>("outputs");
  auto physical = binding.getAs<DictionaryAttr>("physical");
  if (!abi || !modelIndex || !modelName || !inputs || !outputs || !physical)
    return std::nullopt;
  auto serializeIo = [](ArrayAttr values) -> std::optional<json::Array> {
    json::Array result;
    for (mlir::Attribute value : values) {
      auto entry = dyn_cast<DictionaryAttr>(value);
      auto index = entry ? entry.getAs<IntegerAttr>("index") : IntegerAttr();
      auto name = entry ? entry.getAs<StringAttr>("name") : StringAttr();
      auto dataType =
          entry ? entry.getAs<StringAttr>("data_type") : StringAttr();
      auto rank = entry ? entry.getAs<IntegerAttr>("rank") : IntegerAttr();
      auto dimensions = entry ? entry.getAs<DenseI64ArrayAttr>("dimensions")
                              : DenseI64ArrayAttr();
      if (!index || !name || !dataType || !rank || !dimensions)
        return std::nullopt;
      json::Array shape;
      for (int64_t dimension : dimensions.asArrayRef())
        shape.push_back(dimension);
      result.push_back(json::Object{{"index", index.getInt()},
                                    {"name", name.getValue()},
                                    {"data_type", dataType.getValue()},
                                    {"rank", rank.getInt()},
                                    {"dimensions", std::move(shape)}});
    }
    return result;
  };
  auto serializedInputs = serializeIo(inputs);
  auto serializedOutputs = serializeIo(outputs);
  if (!serializedInputs || !serializedOutputs)
    return std::nullopt;
  json::Object physicalJson;
  for (StringRef name :
       {"create_symbol", "bind_tensor_symbol", "enqueue_initialize_symbol",
        "enqueue_run_symbol", "dispose_symbol"}) {
    auto value = physical.getAs<StringAttr>(name);
    if (!value)
      return std::nullopt;
    physicalJson[name] = value.getValue();
  }
  return json::Object{{"abi", abi.getValue()},
                      {"model_index", modelIndex.getInt()},
                      {"model_name", modelName.getValue()},
                      {"inputs", std::move(*serializedInputs)},
                      {"outputs", std::move(*serializedOutputs)},
                      {"physical", std::move(physicalJson)}};
}

FailureOr<std::set<std::string>>
collectUdonCLAnnotatedExports(llvm::Module &module) {
  std::set<std::string> exports;
  GlobalVariable *annotations =
      module.getNamedGlobal("llvm.global.annotations");
  if (!annotations)
    return exports;
  auto *entries =
      dyn_cast_or_null<ConstantArray>(annotations->getInitializer());
  if (!entries) {
    errs() << "llvm.global.annotations has an unsupported initializer\n";
    return failure();
  }
  for (Use &use : entries->operands()) {
    auto *entry = dyn_cast<ConstantStruct>(use.get());
    if (!entry || entry->getNumOperands() < 2) {
      errs() << "llvm.global.annotations has a malformed entry\n";
      return failure();
    }
    auto *function =
        dyn_cast<Function>(entry->getOperand(0)->stripPointerCasts());
    auto *annotationGlobal = dyn_cast<GlobalVariable>(
        getUnderlyingObject(entry->getOperand(1), /*MaxLookup=*/6));
    auto *annotation = annotationGlobal
                           ? dyn_cast_or_null<ConstantDataArray>(
                                 annotationGlobal->getInitializer())
                           : nullptr;
    if (!function || !annotation || !annotation->isCString()) {
      errs() << "llvm.global.annotations entry is not a function annotation\n";
      return failure();
    }
    if (annotation->getAsCString() != "udoncl.export")
      continue;
    if (function->isDeclaration()) {
      errs() << "UDONCL_EXPORT requires a linked Host definition: "
             << function->getName() << "\n";
      return failure();
    }
    if (!exports.insert(function->getName().str()).second) {
      errs() << "duplicate UDONCL_EXPORT annotation: " << function->getName()
             << "\n";
      return failure();
    }
  }
  // The annotation has been converted to compiler-owned export metadata.  The
  // MLIR LLVM importer otherwise preserves the appending metadata global as an
  // ordinary llvm.mlir.global, which is not Host memory and must not reach raw
  // memory recovery.
  annotations->eraseFromParent();
  return exports;
}

int32_t stableExportId(StringRef publicName, mlir::FunctionType type) {
  std::string canonical;
  raw_string_ostream stream(canonical);
  stream << publicName << ':';
  type.print(stream);
  uint32_t hash = 2166136261u;
  for (unsigned char byte : canonical) {
    hash ^= byte;
    hash *= 16777619u;
  }
  int32_t id = static_cast<int32_t>(hash & 0x7fffffffu);
  return id == 0 ? 1 : id;
}

bool configureUnityExports(ModuleOp module, const json::Object &config,
                           const std::set<std::string> &annotatedExports = {}) {
  auto program = config.getString("program");
  auto schema = config.getInteger("schema_version");
  if (!schema || (*schema != 1 && *schema != 2) || !program ||
      !identifier(*program)) {
    errs() << "Unity library config requires schema_version 1 or 2 and a "
              "safe program name\n";
    return false;
  }
  auto *exports =
      config.getArray(*schema == 1 ? "exports" : "export_overrides");
  if (*schema == 1 && (!exports || exports->empty())) {
    errs() << "Unity library schema 1 requires non-empty exports\n";
    return false;
  }
  if (*schema == 2 && annotatedExports.empty()) {
    errs() << "Unity library schema 2 requires at least one UDONCL_EXPORT "
              "Host definition\n";
    return false;
  }
  std::set<std::string> selected, publicNames;
  std::set<int64_t> ids;
  OpBuilder builder(module.getContext());
  auto configure = [&](StringRef symbol, StringRef publicName, int64_t id,
                       const json::Array *parameters) -> bool {
    if (!identifier(symbol) || !identifier(publicName) || id <= 0 ||
        id > INT32_MAX || !selected.insert(symbol.str()).second ||
        !publicNames.insert(publicName.str()).second ||
        !ids.insert(id).second) {
      errs() << "invalid or duplicate Unity library export\n";
      return false;
    }
    auto function = module.lookupSymbol<func::FuncOp>(symbol);
    if (!function || function.isExternal() || function.isPrivate() ||
        (parameters && parameters->size() != function.getNumArguments())) {
      errs() << "Unity library export does not resolve to a public definition: "
             << symbol << "\n";
      return false;
    }
    SmallVector<mlir::Attribute> names;
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      StringRef name;
      if (parameters) {
        auto configured = (*parameters)[index].getAsString();
        if (!configured || !identifier(*configured)) {
          errs() << "invalid Unity library parameter name\n";
          return false;
        }
        name = *configured;
      } else {
        names.push_back(builder.getStringAttr("arg" + std::to_string(index)));
        continue;
      }
      names.push_back(builder.getStringAttr(name));
    }
    function->setAttr("vrc.export_id", builder.getI32IntegerAttr(id));
    function->setAttr("vrc.public_name", builder.getStringAttr(publicName));
    function->setAttr("vrc.parameter_names", builder.getArrayAttr(names));
    return true;
  };
  if (*schema == 1) {
    for (const json::Value &value : *exports) {
      auto *entry = value.getAsObject();
      auto symbol = entry ? entry->getString("symbol") : std::nullopt;
      auto publicName = entry ? entry->getString("public_name") : std::nullopt;
      auto id = entry ? entry->getInteger("id") : std::nullopt;
      auto *parameters = entry ? entry->getArray("parameters") : nullptr;
      if (!symbol || !publicName || !id || !parameters ||
          !configure(*symbol, *publicName, *id, parameters)) {
        errs() << "invalid or duplicate Unity library export\n";
        return false;
      }
    }
  } else {
    std::map<std::string, const json::Object *> overrides;
    std::set<std::string> internalExports;
    if (exports)
      for (const json::Value &value : *exports) {
        auto *entry = value.getAsObject();
        auto symbol = entry ? entry->getString("symbol") : std::nullopt;
        if (!symbol || !identifier(*symbol) ||
            !overrides.emplace(symbol->str(), entry).second ||
            !annotatedExports.count(symbol->str())) {
          errs() << "schema 2 export override must name one annotated Host "
                    "definition\n";
          return false;
        }
      }
    if (auto *internal = config.getArray("internal_exports"))
      for (const json::Value &value : *internal) {
        auto symbol = value.getAsString();
        if (!symbol || !identifier(*symbol) ||
            !internalExports.insert(symbol->str()).second ||
            !annotatedExports.count(symbol->str()) ||
            overrides.count(symbol->str())) {
          errs() << "schema 2 internal export must name one non-overridden "
                    "annotated Host definition\n";
          return false;
        }
      }
    for (const std::string &symbol : annotatedExports) {
      if (internalExports.count(symbol))
        continue;
      auto function = module.lookupSymbol<func::FuncOp>(symbol);
      if (!function || function.isExternal() || function.isPrivate()) {
        errs() << "UDONCL_EXPORT does not resolve after Host recovery: "
               << symbol << "\n";
        return false;
      }
      const json::Object *entry =
          overrides.count(symbol) ? overrides.at(symbol) : nullptr;
      auto publicName = entry ? entry->getString("public_name") : std::nullopt;
      StringRef resolvedName = publicName ? *publicName : StringRef(symbol);
      auto *parameters = entry ? entry->getArray("parameters") : nullptr;
      if (!configure(symbol, resolvedName,
                     stableExportId(resolvedName, function.getFunctionType()),
                     parameters))
        return false;
    }
    if (selected.empty()) {
      errs() << "Unity library schema 2 cannot hide every UDONCL_EXPORT "
                "definition\n";
      return false;
    }
  }
  if (auto *binding = config.getObject("binding")) {
    auto nameSpace = binding->getString("namespace");
    auto className = binding->getString("class_name");
    if (!nameSpace || !dottedIdentifier(*nameSpace) || !className ||
        !identifier(*className)) {
      errs() << "binding requires namespace and safe class_name\n";
      return false;
    }
    module->setAttr("vrc.binding_namespace", builder.getStringAttr(*nameSpace));
    module->setAttr("vrc.binding_class", builder.getStringAttr(*className));
  }
  std::set<std::string> reachable = selected;
  SmallVector<std::string> worklist(selected.begin(), selected.end());
  while (!worklist.empty()) {
    std::string symbol = std::move(worklist.pop_back_val());
    auto function = module.lookupSymbol<func::FuncOp>(symbol);
    if (!function)
      continue;
    for (func::CallOp call : function.getOps<func::CallOp>()) {
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      if (!callee) {
        call.emitError("Unity library helper does not resolve in the linked "
                       "Host module");
        return false;
      }
      if (callee->hasAttr("vrc.intrinsic"))
        continue;
      if (callee.isExternal()) {
        call.emitError("Unity library helper requires a definition: ")
            << call.getCallee();
        return false;
      }
      if (reachable.insert(call.getCallee().str()).second)
        worklist.push_back(call.getCallee().str());
    }
  }
  for (func::FuncOp function :
       llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
    if (!reachable.count(function.getSymName().str()) &&
        !function->hasAttr("vrc.intrinsic"))
      function.erase();
    else if (reachable.count(function.getSymName().str()) &&
             !selected.count(function.getSymName().str()) &&
             !function.isExternal())
      function.setPrivate();
  module->setAttr("vrc.target", builder.getStringAttr("vrc-pc"));
  return true;
}

LogicalResult optimizeUnityHostStorage(ModuleOp module) {
  // All Host modules are linked at this point. Inline before scalar promotion
  // so fixed buffers, descriptors, shapes, OpenCL objects and wrapper calls
  // become visible to whole-program constant propagation and dead-code
  // elimination instead of surviving as generic Udon runtime work.
  PassManager optimization(module.getContext());
  optimization.addPass(createInlinerPass());
  optimization.addPass(createSROA());
  optimization.addPass(createMem2Reg());
  optimization.addPass(createCanonicalizerPass());
  optimization.addPass(createSCCPPass());
  optimization.addPass(createCSEPass());
  optimization.addPass(createCanonicalizerPass());
  optimization.addPass(createSymbolDCEPass());
  return optimization.run(module);
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
  return success();
}

LogicalResult recoverUnityCBufferIntrinsics(ModuleOp module) {
  constexpr StringLiteral allocName = "__udoncl_buffer_alloc";
  constexpr StringLiteral copyName = "__udoncl_buffer_copy";
  SmallVector<func::CallOp> allocations, copies;
  module.walk([&](func::CallOp call) {
    if (call.getCallee() == allocName)
      allocations.push_back(call);
    else if (call.getCallee() == copyName)
      copies.push_back(call);
  });
  if (allocations.empty() && copies.empty())
    return success();
  auto allocPrototype = module.lookupSymbol<func::FuncOp>(allocName);
  auto copyPrototype = module.lookupSymbol<func::FuncOp>(copyName);
  auto i32 = mlir::IntegerType::get(module.getContext(), 32);
  bool invalidAllocation =
      !allocations.empty() &&
      (!allocPrototype || !allocPrototype.isExternal() ||
       allocPrototype.getArgumentTypes() != ArrayRef<mlir::Type>{i32, i32} ||
       allocPrototype.getResultTypes() != ArrayRef<mlir::Type>{i32});
  bool invalidCopy =
      !copies.empty() &&
      (!copyPrototype || !copyPrototype.isExternal() ||
       copyPrototype.getArgumentTypes() !=
           ArrayRef<mlir::Type>{i32, i32, i32, i32, i32} ||
       copyPrototype.getResultTypes() != ArrayRef<mlir::Type>{i32});
  if (invalidAllocation || invalidCopy)
    return module.emitError(
        "Unity C buffer intrinsics have an invalid ILP32 signature");
  llvm::DenseSet<Operation *> copySites;
  for (func::CallOp copy : copies)
    copySites.insert(copy.getOperation());
  for (func::CallOp allocation : allocations) {
    mlir::Value handle = allocation.getResult(0);
    if (handle.use_empty() ||
        !llvm::all_of(handle.getUsers(), [&](Operation *user) {
          return copySites.contains(user);
        }))
      return allocation.emitError(
          "initial Unity C buffer handles must flow directly to one or more "
          "CopyBuffer calls");
  }
  for (func::CallOp copy : copies)
    for (unsigned index : {0u, 1u}) {
      auto allocation = copy.getOperand(index).getDefiningOp<func::CallOp>();
      if (!allocation || allocation.getCallee() != allocName)
        return copy.emitError(
            "initial Unity C CopyBuffer operands must be direct allocations");
    }

  IRRewriter rewriter(module.getContext());
  llvm::DenseMap<mlir::Value, mlir::Value> buffers;
  for (func::CallOp allocation : allocations) {
    rewriter.setInsertionPoint(allocation);
    OperationState state(allocation.getLoc(),
                         ::vrc::vrc::BufferAllocOp::getOperationName());
    state.addOperands({allocation.getOperand(0), allocation.getOperand(1)});
    state.addTypes({::vrc::vrc::BufferType::get(module.getContext()), i32});
    state.addAttribute("format", rewriter.getStringAttr("rgba8_unorm"));
    state.addAttribute("usage", rewriter.getStringAttr("storage"));
    Operation *operation = rewriter.create(state);
    buffers.try_emplace(allocation.getResult(0), operation->getResult(0));
  }
  for (func::CallOp copy : copies) {
    rewriter.setInsertionPoint(copy);
    OperationState state(copy.getLoc(),
                         ::vrc::vrc::BufferCopyOp::getOperationName());
    state.addOperands({buffers.lookup(copy.getOperand(0)),
                       buffers.lookup(copy.getOperand(1)), copy.getOperand(2),
                       copy.getOperand(3), copy.getOperand(4)});
    state.addTypes(::vrc::vrc::EventType::get(module.getContext()));
    rewriter.create(state);
    mlir::Value success =
        rewriter.create<arith::ConstantIntOp>(copy.getLoc(), 0, 32);
    rewriter.replaceOp(copy, success);
  }
  for (func::CallOp allocation : allocations)
    rewriter.eraseOp(allocation);
  if (allocPrototype)
    rewriter.eraseOp(allocPrototype);
  if (copyPrototype)
    rewriter.eraseOp(copyPrototype);
  return success();
}

constexpr StringLiteral openCLAllocName = "__udoncl_opencl_buffer_alloc";
constexpr StringLiteral openCLCopyName = "__udoncl_opencl_buffer_copy";
constexpr StringLiteral openCLDeviceName = "__udoncl_opencl_device";
constexpr StringLiteral openCLContextName = "__udoncl_opencl_context";
constexpr StringLiteral openCLQueueName = "__udoncl_opencl_queue";

std::optional<int64_t> llvmIRConstantInt(const llvm::Value *value) {
  auto constant = dyn_cast<llvm::ConstantInt>(value);
  return constant ? std::optional<int64_t>(constant->getSExtValue())
                  : std::nullopt;
}

const llvm::Value *llvmIRStoredSlot(const llvm::Value *value) {
  for (const llvm::User *user : value->users())
    if (auto store = dyn_cast<llvm::StoreInst>(user))
      if (store->getValueOperand() == value)
        return store->getPointerOperand()->stripPointerCasts();
  // Optimized Host IR keeps non-escaping handles in SSA instead of forcing a
  // temporary alloca. Treat the defining value itself as its provenance root.
  return value;
}

bool llvmIRLoadsFrom(const llvm::Value *value, const llvm::Value *slot) {
  if (value && slot && value->stripPointerCasts() == slot->stripPointerCasts())
    return true;
  auto load = dyn_cast<llvm::LoadInst>(value);
  return load && slot &&
         load->getPointerOperand()->stripPointerCasts() ==
             slot->stripPointerCasts();
}

bool llvmIRSameSlot(const llvm::Value *left, const llvm::Value *right) {
  return left && right &&
         left->stripPointerCasts() == right->stripPointerCasts();
}

bool llvmIRSlotStoresConstant(const llvm::Value *slot, int64_t expected) {
  if (!slot)
    return false;
  for (const llvm::User *user : slot->users())
    if (auto store = dyn_cast<llvm::StoreInst>(user))
      if (llvmIRSameSlot(store->getPointerOperand(), slot) &&
          llvmIRConstantInt(store->getValueOperand()) == expected)
        return true;
  return false;
}

OwningOpRef<ModuleOp>
recoverUnityOpenCLDispatchProgram(const llvm::Module &input,
                                  const json::Object &config,
                                  MLIRContext &context) {
  auto *kernelConfig = config.getObject("device_kernel");
  auto *exports = config.getArray("exports");
  auto *buffers = config.getArray("buffers");
  auto kernelName =
      kernelConfig ? kernelConfig->getString("kernel") : std::nullopt;
  auto kernelId =
      kernelConfig ? kernelConfig->getInteger("kernel_id") : std::nullopt;
  auto *argumentKinds =
      kernelConfig ? kernelConfig->getArray("argument_kinds") : nullptr;
  auto scalarKind = argumentKinds && argumentKinds->size() == 2
                        ? (*argumentKinds)[1].getAsString()
                        : std::nullopt;
  auto *bufferConfig =
      buffers && buffers->size() == 1 ? (*buffers)[0].getAsObject() : nullptr;
  auto bufferBytes =
      bufferConfig ? bufferConfig->getInteger("byte_size") : std::nullopt;
  auto publicElementType =
      bufferConfig ? bufferConfig->getString("element_type") : std::nullopt;
  auto publicAccess =
      bufferConfig ? bufferConfig->getString("public_access") : std::nullopt;
  if (!kernelConfig || !exports || exports->size() != 1 || !buffers ||
      buffers->size() != 1 || !kernelName || !kernelId || !argumentKinds ||
      argumentKinds->size() != 2 || !scalarKind || !bufferBytes ||
      *bufferBytes != 4 || (*argumentKinds)[0].getAsString() != "buffer" ||
      (*scalarKind != "f32" && *scalarKind != "i32")) {
    errs() << "initial Unity OpenCL dispatch requires one configured "
              "four-byte buffer and one f32/i32 scalar kernel\n";
    return {};
  }
  if ((publicElementType || publicAccess) &&
      (!publicElementType || !publicAccess || *publicAccess != "output" ||
       *publicElementType != (*scalarKind == "f32" ? "float32" : "int32"))) {
    errs() << "initial Unity OpenCL dispatch public buffer semantics do not "
              "match the write-only kernel output\n";
    return {};
  }
  SmallVector<int32_t> configuredGlobal, configuredLocal, configuredOffset;
  auto readShape = [&](StringRef key, SmallVectorImpl<int32_t> &values) {
    auto *array = kernelConfig->getArray(key);
    if (!array || array->size() != 3)
      return false;
    for (const json::Value &value : *array) {
      auto integer = value.getAsInteger();
      if (!integer || *integer < std::numeric_limits<int32_t>::min() ||
          *integer > std::numeric_limits<int32_t>::max())
        return false;
      values.push_back(static_cast<int32_t>(*integer));
    }
    return true;
  };
  if (!readShape("global_size", configuredGlobal) ||
      !readShape("local_size", configuredLocal) ||
      !readShape("global_offset", configuredOffset) ||
      configuredGlobal != SmallVector<int32_t>{1, 1, 1} ||
      configuredLocal != SmallVector<int32_t>{1, 1, 1} ||
      configuredOffset != SmallVector<int32_t>{0, 0, 0}) {
    errs() << "initial Unity OpenCL dispatch requires configured global/local "
              "size 1 and zero offset\n";
    return {};
  }
  auto *exportConfig = (*exports)[0].getAsObject();
  auto exportName =
      exportConfig ? exportConfig->getString("symbol") : std::nullopt;
  auto *function = exportName ? input.getFunction(*exportName) : nullptr;
  if (!function || function->isDeclaration() || function->size() != 1 ||
      !function->getReturnType()->isVoidTy() || function->arg_size() != 1 ||
      (*scalarKind == "f32" && !function->getArg(0)->getType()->isFloatTy()) ||
      (*scalarKind == "i32" &&
       !function->getArg(0)->getType()->isIntegerTy(32))) {
    errs() << "initial Unity OpenCL dispatch export must be a straight-line "
              "void(f32/i32) definition matching argument_kinds\n";
    return {};
  }
  SmallVector<const llvm::CallBase *> calls;
  for (const llvm::Instruction &instruction : function->front())
    if (auto *call = dyn_cast<llvm::CallBase>(&instruction))
      if (const llvm::Function *callee = call->getCalledFunction())
        if (callee->getName().starts_with("cl"))
          calls.push_back(call);
  constexpr StringLiteral expected[] = {
      "clGetPlatformIDs",       "clGetDeviceIDs",
      "clCreateContext",        "clCreateCommandQueue",
      "clCreateBuffer",         "clCreateProgramWithSource",
      "clBuildProgram",         "clCreateKernel",
      "clSetKernelArg",         "clSetKernelArg",
      "clEnqueueNDRangeKernel", "clReleaseKernel",
      "clReleaseProgram",       "clReleaseMemObject",
      "clReleaseCommandQueue",  "clReleaseContext",
  };
  if (calls.size() != std::size(expected)) {
    errs() << "initial Unity OpenCL dispatch requires the complete ordered "
              "OpenCL 1.2 setup/dispatch/release sequence\n";
    return {};
  }
  for (auto [call, name] : llvm::zip(calls, expected))
    if (call->getCalledFunction()->getName() != name) {
      errs() << "initial Unity OpenCL dispatch call order mismatch at " << name
             << "\n";
      return {};
    }
  auto require = [&](bool condition, StringRef message) {
    if (!condition)
      errs() << "initial Unity OpenCL dispatch " << message << "\n";
    return condition;
  };
  StringRef actualKernelName;
  const llvm::Value *platformSlot = calls[0]->getArgOperand(1);
  const llvm::Value *deviceSlot = calls[1]->getArgOperand(3);
  const llvm::Value *contextSlot = llvmIRStoredSlot(calls[2]);
  const llvm::Value *queueSlot = llvmIRStoredSlot(calls[3]);
  const llvm::Value *bufferSlot = llvmIRStoredSlot(calls[4]);
  const llvm::Value *programSlot = llvmIRStoredSlot(calls[5]);
  const llvm::Value *kernelSlot = llvmIRStoredSlot(calls[7]);
  const llvm::Value *scalarSlot = llvmIRStoredSlot(function->getArg(0));
  bool valid =
      require(llvmIRConstantInt(calls[0]->getArgOperand(0)) == 1,
              "requires one platform") &&
      require(llvmIRConstantInt(calls[1]->getArgOperand(1)) == 4 &&
                  llvmIRConstantInt(calls[1]->getArgOperand(2)) == 1,
              "requires one GPU device") &&
      require(llvmIRConstantInt(calls[2]->getArgOperand(1)) == 1,
              "requires one context device") &&
      require(llvmIRConstantInt(calls[3]->getArgOperand(2)) == 0,
              "requires an in-order queue") &&
      require(llvmIRConstantInt(calls[4]->getArgOperand(1)) == 2 &&
                  llvmIRConstantInt(calls[4]->getArgOperand(2)) == 4,
              "requires one four-byte write-only output") &&
      require(llvmIRConstantInt(calls[5]->getArgOperand(1)) == 1,
              "requires one static program source") &&
      require(llvm::getConstantStringInfo(calls[7]->getArgOperand(1),
                                          actualKernelName) &&
                  actualKernelName == *kernelName,
              "kernel name does not match device_kernel") &&
      require(llvmIRConstantInt(calls[8]->getArgOperand(1)) == 0 &&
                  llvmIRConstantInt(calls[8]->getArgOperand(2)) == 4 &&
                  llvmIRConstantInt(calls[9]->getArgOperand(1)) == 1 &&
                  llvmIRConstantInt(calls[9]->getArgOperand(2)) == 4,
              "requires buffer arg 0 and one four-byte scalar arg 1") &&
      require(llvmIRConstantInt(calls[10]->getArgOperand(2)) == 1 &&
                  isa<llvm::ConstantPointerNull>(calls[10]->getArgOperand(3)) &&
                  llvmIRSlotStoresConstant(calls[10]->getArgOperand(4), 1) &&
                  llvmIRSlotStoresConstant(calls[10]->getArgOperand(5), 1) &&
                  llvmIRConstantInt(calls[10]->getArgOperand(6)) == 0 &&
                  isa<llvm::ConstantPointerNull>(calls[10]->getArgOperand(7)) &&
                  isa<llvm::ConstantPointerNull>(calls[10]->getArgOperand(8)),
              "requires a one-dimensional dispatch without wait events") &&
      require(llvmIRLoadsFrom(calls[1]->getArgOperand(0), platformSlot) &&
                  llvmIRSameSlot(calls[2]->getArgOperand(2), deviceSlot) &&
                  llvmIRLoadsFrom(calls[3]->getArgOperand(0), contextSlot) &&
                  llvmIRLoadsFrom(calls[3]->getArgOperand(1), deviceSlot) &&
                  llvmIRLoadsFrom(calls[4]->getArgOperand(0), contextSlot) &&
                  llvmIRLoadsFrom(calls[5]->getArgOperand(0), contextSlot),
              "context and queue handles must flow from their setup calls") &&
      require(llvmIRLoadsFrom(calls[6]->getArgOperand(0), programSlot) &&
                  llvmIRLoadsFrom(calls[7]->getArgOperand(0), programSlot) &&
                  llvmIRLoadsFrom(calls[8]->getArgOperand(0), kernelSlot) &&
                  llvmIRLoadsFrom(calls[9]->getArgOperand(0), kernelSlot) &&
                  llvmIRSameSlot(calls[8]->getArgOperand(3), bufferSlot) &&
                  llvmIRSameSlot(calls[9]->getArgOperand(3), scalarSlot),
              "program, kernel, buffer, and scalar arguments must keep their "
              "declared provenance") &&
      require(llvmIRLoadsFrom(calls[10]->getArgOperand(0), queueSlot) &&
                  llvmIRLoadsFrom(calls[10]->getArgOperand(1), kernelSlot) &&
                  llvmIRLoadsFrom(calls[11]->getArgOperand(0), kernelSlot) &&
                  llvmIRLoadsFrom(calls[12]->getArgOperand(0), programSlot) &&
                  llvmIRLoadsFrom(calls[13]->getArgOperand(0), bufferSlot) &&
                  llvmIRLoadsFrom(calls[14]->getArgOperand(0), queueSlot) &&
                  llvmIRLoadsFrom(calls[15]->getArgOperand(0), contextSlot),
              "enqueue and release handles must match their created objects");
  if (!valid)
    return {};

  auto owner = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(owner.getBodyRegion());
  mlir::Type scalarType = *scalarKind == "f32"
                              ? mlir::Type(builder.getF32Type())
                              : mlir::Type(builder.getI32Type());
  auto functionType = builder.getFunctionType({scalarType}, {});
  auto host =
      func::FuncOp::create(builder, owner.getLoc(), *exportName, functionType);
  Block *entry = host.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto bytes = arith::ConstantIntOp::create(builder, owner.getLoc(), 4, 32);
  auto logicalId = arith::ConstantIntOp::create(builder, owner.getLoc(), 1, 32);
  OperationState allocation(owner.getLoc(),
                            ::vrc::vrc::BufferAllocOp::getOperationName());
  allocation.addOperands({logicalId.getResult(), bytes.getResult()});
  allocation.addTypes(
      {::vrc::vrc::BufferType::get(&context), builder.getI32Type()});
  allocation.addAttribute("format", builder.getStringAttr("rgba8_unorm"));
  allocation.addAttribute("usage", builder.getStringAttr("storage"));
  Operation *buffer = builder.create(allocation);
  OperationState descriptor(
      owner.getLoc(), ::vrc::vrc::DispatchDescCreateOp::getOperationName());
  descriptor.addOperands(entry->getArgument(0));
  descriptor.addTypes(::vrc::vrc::DispatchDescType::get(&context));
  descriptor.addAttribute("kernel_id", builder.getI32IntegerAttr(*kernelId));
  descriptor.addAttribute(
      "scalar_kinds",
      builder.getArrayAttr({builder.getStringAttr(*scalarKind)}));
  Operation *desc = builder.create(descriptor);
  OperationState dispatch(owner.getLoc(),
                          ::vrc::vrc::DispatchOp::getOperationName());
  dispatch.addOperands({desc->getResult(0), buffer->getResult(0)});
  dispatch.addTypes(::vrc::vrc::EventType::get(&context));
  dispatch.addAttribute("kernel_id", builder.getI32IntegerAttr(*kernelId));
  dispatch.addAttribute(
      "argument_kinds",
      builder.getArrayAttr({builder.getStringAttr("buffer"),
                            builder.getStringAttr(*scalarKind)}));
  dispatch.addAttribute("resource_accesses",
                        builder.getArrayAttr({builder.getStringAttr("write")}));
  dispatch.addAttribute("global_size",
                        builder.getDenseI32ArrayAttr(configuredGlobal));
  dispatch.addAttribute("local_size",
                        builder.getDenseI32ArrayAttr(configuredLocal));
  dispatch.addAttribute("global_offset",
                        builder.getDenseI32ArrayAttr(configuredOffset));
  builder.create(dispatch);
  func::ReturnOp::create(builder, owner.getLoc());
  return OwningOpRef<ModuleOp>(owner);
}

OwningOpRef<ModuleOp>
recoverUnityOpenCLGatherDispatchProgram(const llvm::Module &input,
                                        const json::Object &config,
                                        MLIRContext &context) {
  auto *kernelConfig = config.getObject("device_kernel");
  auto *exports = config.getArray("exports");
  auto *buffers = config.getArray("buffers");
  auto *argumentKinds =
      kernelConfig ? kernelConfig->getArray("argument_kinds") : nullptr;
  auto kernelName =
      kernelConfig ? kernelConfig->getString("kernel") : std::nullopt;
  auto kernelId =
      kernelConfig ? kernelConfig->getInteger("kernel_id") : std::nullopt;
  auto words = kernelConfig ? kernelConfig->getInteger("words") : std::nullopt;
  if (!kernelConfig || kernelConfig->getString("profile") != "gather-f32" ||
      !exports || exports->size() != 1 || !buffers || buffers->empty() ||
      !argumentKinds || argumentKinds->size() < 3 ||
      argumentKinds->size() > 16 || !kernelName || !kernelId || !words ||
      *words < 4 || *words > 32768) {
    errs() << "Unity OpenCL gather dispatch requires one configured export, "
              "kernel, and 3-16 arguments\n";
    return {};
  }
  SmallVector<StringRef> kinds, scalarKinds;
  SmallVector<int32_t> bufferBytes;
  bool sawScalar = false;
  for (const json::Value &value : *argumentKinds) {
    auto kind = value.getAsString();
    if (!kind || (*kind != "buffer" && *kind != "f32" && *kind != "i32") ||
        (*kind == "buffer" && sawScalar)) {
      errs() << "Unity OpenCL gather arguments require buffers before f32/i32 "
                "scalars\n";
      return {};
    }
    kinds.push_back(*kind);
    if (*kind != "buffer") {
      sawScalar = true;
      scalarKinds.push_back(*kind);
    }
  }
  if (kinds.size() - scalarKinds.size() != buffers->size() ||
      scalarKinds.empty()) {
    errs() << "Unity OpenCL gather argument_kinds do not match buffers\n";
    return {};
  }
  for (const json::Value &value : *buffers) {
    auto *object = value.getAsObject();
    auto bytes = object ? object->getInteger("byte_size") : std::nullopt;
    if (!bytes || *bytes != *words * 4 || *bytes > INT32_MAX) {
      errs() << "Unity OpenCL gather buffers must each contain words * 4 "
                "bytes\n";
      return {};
    }
    bufferBytes.push_back(static_cast<int32_t>(*bytes));
  }
  SmallVector<int32_t> global, local, offset;
  auto readShape = [&](StringRef key, SmallVectorImpl<int32_t> &values) {
    auto *array = kernelConfig->getArray(key);
    if (!array || array->size() != 3)
      return false;
    for (const json::Value &value : *array) {
      auto integer = value.getAsInteger();
      if (!integer || *integer < 0 || *integer > INT32_MAX)
        return false;
      values.push_back(static_cast<int32_t>(*integer));
    }
    return true;
  };
  if (!readShape("global_size", global) || !readShape("local_size", local) ||
      !readShape("global_offset", offset) ||
      local != SmallVector<int32_t>{1, 1, 1} ||
      offset != SmallVector<int32_t>{0, 0, 0} || global[0] <= 0 ||
      global[1] <= 0 || global[2] != 1 ||
      int64_t(global[0]) * global[1] != *words) {
    errs() << "Unity OpenCL gather launch does not match words\n";
    return {};
  }
  auto *exportConfig = (*exports)[0].getAsObject();
  auto exportName =
      exportConfig ? exportConfig->getString("symbol") : std::nullopt;
  auto *function = exportName ? input.getFunction(*exportName) : nullptr;
  if (!function || function->isDeclaration() || function->size() != 1 ||
      !function->getReturnType()->isVoidTy() ||
      function->arg_size() != scalarKinds.size()) {
    errs() << "Unity OpenCL gather export must be straight-line "
              "void(f32/i32...)\n";
    return {};
  }
  for (auto [argument, kind] : llvm::zip(function->args(), scalarKinds))
    if ((kind == "f32" && !argument.getType()->isFloatTy()) ||
        (kind == "i32" && !argument.getType()->isIntegerTy(32))) {
      errs() << "Unity OpenCL gather export scalar signature mismatch\n";
      return {};
    }
  SmallVector<const llvm::CallBase *> calls;
  for (const llvm::Instruction &instruction : function->front())
    if (auto *call = dyn_cast<llvm::CallBase>(&instruction))
      if (const llvm::Function *callee = call->getCalledFunction())
        if (callee->getName().starts_with("cl"))
          calls.push_back(call);
  SmallVector<StringRef> expected{"clGetPlatformIDs", "clGetDeviceIDs",
                                  "clCreateContext", "clCreateCommandQueue"};
  for (size_t i = 0; i < buffers->size(); ++i)
    expected.push_back("clCreateBuffer");
  for (StringRef name :
       {"clCreateProgramWithSource", "clBuildProgram", "clCreateKernel"})
    expected.push_back(name);
  for (size_t i = 0; i < argumentKinds->size(); ++i)
    expected.push_back("clSetKernelArg");
  for (StringRef name :
       {"clEnqueueNDRangeKernel", "clReleaseKernel", "clReleaseProgram"})
    expected.push_back(name);
  for (size_t i = 0; i < buffers->size(); ++i)
    expected.push_back("clReleaseMemObject");
  expected.push_back("clReleaseCommandQueue");
  expected.push_back("clReleaseContext");
  if (calls.size() != expected.size()) {
    errs() << "Unity OpenCL gather requires the complete ordered setup, "
              "dispatch, and release sequence\n";
    return {};
  }
  for (auto [call, name] : llvm::zip(calls, expected))
    if (call->getCalledFunction()->getName() != name) {
      errs() << "Unity OpenCL gather call order mismatch at " << name << "\n";
      return {};
    }
  auto require = [&](bool condition, StringRef message) {
    if (!condition)
      errs() << "Unity OpenCL gather " << message << "\n";
    return condition;
  };
  size_t bufferBase = 4;
  size_t programIndex = bufferBase + buffers->size();
  size_t buildIndex = programIndex + 1;
  size_t kernelIndex = programIndex + 2;
  size_t setArgBase = programIndex + 3;
  size_t enqueueIndex = setArgBase + argumentKinds->size();
  size_t releaseKernelIndex = enqueueIndex + 1;
  size_t releaseProgramIndex = enqueueIndex + 2;
  size_t releaseMemBase = enqueueIndex + 3;
  size_t releaseQueueIndex = releaseMemBase + buffers->size();
  size_t releaseContextIndex = releaseQueueIndex + 1;
  const llvm::Value *platformSlot = calls[0]->getArgOperand(1);
  const llvm::Value *deviceSlot = calls[1]->getArgOperand(3);
  const llvm::Value *contextSlot = llvmIRStoredSlot(calls[2]);
  const llvm::Value *queueSlot = llvmIRStoredSlot(calls[3]);
  SmallVector<const llvm::Value *> bufferSlots, scalarSlots;
  for (size_t i = 0; i < buffers->size(); ++i)
    bufferSlots.push_back(llvmIRStoredSlot(calls[bufferBase + i]));
  for (llvm::Argument &argument : function->args())
    scalarSlots.push_back(llvmIRStoredSlot(&argument));
  const llvm::Value *programSlot = llvmIRStoredSlot(calls[programIndex]);
  const llvm::Value *kernelSlot = llvmIRStoredSlot(calls[kernelIndex]);
  StringRef actualKernelName;
  bool valid =
      require(llvmIRConstantInt(calls[0]->getArgOperand(0)) == 1,
              "requires one platform") &&
      require(llvmIRConstantInt(calls[1]->getArgOperand(1)) == 4 &&
                  llvmIRConstantInt(calls[1]->getArgOperand(2)) == 1,
              "requires one GPU device") &&
      require(llvmIRConstantInt(calls[2]->getArgOperand(1)) == 1 &&
                  llvmIRConstantInt(calls[3]->getArgOperand(2)) == 0,
              "requires one context and in-order queue") &&
      require(llvmIRLoadsFrom(calls[1]->getArgOperand(0), platformSlot) &&
                  llvmIRSameSlot(calls[2]->getArgOperand(2), deviceSlot) &&
                  llvmIRLoadsFrom(calls[3]->getArgOperand(0), contextSlot) &&
                  llvmIRLoadsFrom(calls[3]->getArgOperand(1), deviceSlot),
              "platform/device/context/queue provenance mismatch") &&
      require(llvmIRConstantInt(calls[programIndex]->getArgOperand(1)) == 1 &&
                  llvmIRLoadsFrom(calls[programIndex]->getArgOperand(0),
                                  contextSlot) &&
                  llvmIRLoadsFrom(calls[buildIndex]->getArgOperand(0),
                                  programSlot) &&
                  llvmIRLoadsFrom(calls[kernelIndex]->getArgOperand(0),
                                  programSlot) &&
                  llvm::getConstantStringInfo(
                      calls[kernelIndex]->getArgOperand(1), actualKernelName) &&
                  actualKernelName == *kernelName,
              "program/kernel provenance or configured name mismatch");
  std::optional<size_t> outputBuffer;
  for (size_t i = 0; valid && i < buffers->size(); ++i) {
    auto flags = llvmIRConstantInt(calls[bufferBase + i]->getArgOperand(1));
    auto *bufferObject = (*buffers)[i].getAsObject();
    auto publicElementType =
        bufferObject ? bufferObject->getString("element_type") : std::nullopt;
    auto publicAccess =
        bufferObject ? bufferObject->getString("public_access") : std::nullopt;
    if (flags == 2 && !outputBuffer)
      outputBuffer = i;
    else if (flags != 4)
      valid =
          require(false, "requires read-only inputs and one write-only output");
    if (publicElementType || publicAccess)
      valid = valid &&
              require(publicElementType && *publicElementType == "float32" &&
                          publicAccess &&
                          *publicAccess == (*flags == 2 ? "output" : "input"),
                      "public buffer semantics do not match OpenCL access");
    valid = valid &&
            require(llvmIRLoadsFrom(calls[bufferBase + i]->getArgOperand(0),
                                    contextSlot) &&
                        flags && (*flags == 2 || *flags == 4) &&
                        llvmIRConstantInt(calls[bufferBase + i]->getArgOperand(
                            2)) == bufferBytes[i] &&
                        isa<llvm::ConstantPointerNull>(
                            calls[bufferBase + i]->getArgOperand(3)),
                    "buffer allocation must match configured size and context");
  }
  valid = valid && require(outputBuffer.has_value(),
                           "requires exactly one write-only output buffer");
  size_t bufferOrdinal = 0, scalarOrdinal = 0;
  for (size_t i = 0; valid && i < kinds.size(); ++i) {
    const llvm::Value *slot = kinds[i] == "buffer"
                                  ? bufferSlots[bufferOrdinal++]
                                  : scalarSlots[scalarOrdinal++];
    const llvm::CallBase *set = calls[setArgBase + i];
    valid =
        require(llvmIRLoadsFrom(set->getArgOperand(0), kernelSlot) &&
                    llvmIRConstantInt(set->getArgOperand(1)) == int64_t(i) &&
                    llvmIRConstantInt(set->getArgOperand(2)) == 4 &&
                    llvmIRSameSlot(set->getArgOperand(3), slot),
                "SetKernelArg order, size, or provenance mismatch");
  }
  const llvm::CallBase *enqueue = calls[enqueueIndex];
  valid =
      valid &&
      require(
          llvmIRLoadsFrom(enqueue->getArgOperand(0), queueSlot) &&
              llvmIRLoadsFrom(enqueue->getArgOperand(1), kernelSlot) &&
              llvmIRConstantInt(enqueue->getArgOperand(2)) == 1 &&
              isa<llvm::ConstantPointerNull>(enqueue->getArgOperand(3)) &&
              llvmIRSlotStoresConstant(enqueue->getArgOperand(4), global[0]) &&
              llvmIRSlotStoresConstant(enqueue->getArgOperand(5), 1) &&
              llvmIRConstantInt(enqueue->getArgOperand(6)) == 0 &&
              isa<llvm::ConstantPointerNull>(enqueue->getArgOperand(7)) &&
              isa<llvm::ConstantPointerNull>(enqueue->getArgOperand(8)),
          "enqueue does not match configured 1D launch") &&
      require(llvmIRLoadsFrom(calls[releaseKernelIndex]->getArgOperand(0),
                              kernelSlot) &&
                  llvmIRLoadsFrom(calls[releaseProgramIndex]->getArgOperand(0),
                                  programSlot) &&
                  llvmIRLoadsFrom(calls[releaseQueueIndex]->getArgOperand(0),
                                  queueSlot) &&
                  llvmIRLoadsFrom(calls[releaseContextIndex]->getArgOperand(0),
                                  contextSlot),
              "released handles must match their created objects");
  for (size_t i = 0; valid && i < buffers->size(); ++i)
    valid = require(llvmIRLoadsFrom(calls[releaseMemBase + i]->getArgOperand(0),
                                    bufferSlots[i]),
                    "released buffer must match its created object");
  if (!valid)
    return {};

  auto owner = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(owner.getBodyRegion());
  SmallVector<mlir::Type> scalarTypes;
  SmallVector<mlir::Attribute> scalarKindAttrs, kindAttrs;
  for (StringRef kind : scalarKinds) {
    scalarTypes.push_back(kind == "f32" ? mlir::Type(builder.getF32Type())
                                        : mlir::Type(builder.getI32Type()));
    scalarKindAttrs.push_back(builder.getStringAttr(kind));
  }
  for (StringRef kind : kinds)
    kindAttrs.push_back(builder.getStringAttr(kind));
  auto host =
      func::FuncOp::create(builder, owner.getLoc(), *exportName,
                           builder.getFunctionType(scalarTypes, TypeRange{}));
  Block *entry = host.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  SmallVector<Operation *> allocatedBuffers;
  for (auto [bufferIndex, byteSize] : llvm::enumerate(bufferBytes)) {
    auto logicalId = arith::ConstantIntOp::create(
        builder, owner.getLoc(), static_cast<int64_t>(bufferIndex + 1), 32);
    auto bytes =
        arith::ConstantIntOp::create(builder, owner.getLoc(), byteSize, 32);
    OperationState allocation(owner.getLoc(),
                              ::vrc::vrc::BufferAllocOp::getOperationName());
    allocation.addOperands({logicalId.getResult(), bytes.getResult()});
    allocation.addTypes(
        {::vrc::vrc::BufferType::get(&context), builder.getI32Type()});
    allocation.addAttribute("format", builder.getStringAttr("rgba8_unorm"));
    allocation.addAttribute("usage", builder.getStringAttr("storage"));
    allocatedBuffers.push_back(builder.create(allocation));
  }
  OperationState descriptor(
      owner.getLoc(), ::vrc::vrc::DispatchDescCreateOp::getOperationName());
  descriptor.addOperands(entry->getArguments());
  descriptor.addTypes(::vrc::vrc::DispatchDescType::get(&context));
  descriptor.addAttribute("kernel_id", builder.getI32IntegerAttr(*kernelId));
  descriptor.addAttribute("scalar_kinds",
                          builder.getArrayAttr(scalarKindAttrs));
  Operation *desc = builder.create(descriptor);
  OperationState dispatch(owner.getLoc(),
                          ::vrc::vrc::DispatchOp::getOperationName());
  SmallVector<mlir::Value> operands{desc->getResult(0)};
  for (Operation *buffer : allocatedBuffers)
    operands.push_back(buffer->getResult(0));
  dispatch.addOperands(operands);
  dispatch.addTypes(::vrc::vrc::EventType::get(&context));
  dispatch.addAttribute("kernel_id", builder.getI32IntegerAttr(*kernelId));
  dispatch.addAttribute("argument_kinds", builder.getArrayAttr(kindAttrs));
  SmallVector<mlir::Attribute> resourceAccesses(allocatedBuffers.size(),
                                                builder.getStringAttr("read"));
  resourceAccesses[*outputBuffer] = builder.getStringAttr("write");
  dispatch.addAttribute("resource_accesses",
                        builder.getArrayAttr(resourceAccesses));
  dispatch.addAttribute(
      "vrc.unity_output_resource_index",
      builder.getI32IntegerAttr(static_cast<int32_t>(*outputBuffer)));
  dispatch.addAttribute("global_size", builder.getDenseI32ArrayAttr(global));
  dispatch.addAttribute("local_size", builder.getDenseI32ArrayAttr(local));
  dispatch.addAttribute("global_offset", builder.getDenseI32ArrayAttr(offset));
  builder.create(dispatch);
  func::ReturnOp::create(builder, owner.getLoc());
  return OwningOpRef<ModuleOp>(owner);
}

std::optional<int64_t> llvmConstantInt(mlir::Value value) {
  auto constant = value.getDefiningOp<LLVM::ConstantOp>();
  auto integer =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
  return integer ? std::optional<int64_t>(integer.getInt()) : std::nullopt;
}

bool llvmNull(mlir::Value value) {
  return !!value.getDefiningOp<LLVM::ZeroOp>();
}

LLVM::CallOp createLLVMCall(IRRewriter &rewriter, LLVM::CallOp source,
                            StringRef callee, TypeRange results,
                            ValueRange operands) {
  OperationState state(source.getLoc(), LLVM::CallOp::getOperationName());
  NamedAttrList attributes(source->getAttrs());
  attributes.set("callee", SymbolRefAttr::get(rewriter.getContext(), callee));
  attributes.set("operandSegmentSizes",
                 rewriter.getDenseI32ArrayAttr(
                     {static_cast<int32_t>(operands.size()), 0}));
  state.addAttributes(attributes);
  state.addOperands(operands);
  state.addTypes(results);
  return cast<LLVM::CallOp>(rewriter.create(state));
}

void storeOpenCLSuccess(IRRewriter &rewriter, Location location,
                        mlir::Value pointer) {
  if (llvmNull(pointer))
    return;
  auto zero = rewriter.create<LLVM::ConstantOp>(location, rewriter.getI32Type(),
                                                rewriter.getI32IntegerAttr(0));
  rewriter.create<LLVM::StoreOp>(location, zero, pointer);
}

mlir::Value openCLSentinel(IRRewriter &rewriter, Location location,
                           int32_t value) {
  auto integer = rewriter.create<LLVM::ConstantOp>(
      location, rewriter.getI32Type(), rewriter.getI32IntegerAttr(value));
  return rewriter.create<LLVM::IntToPtrOp>(
      location, LLVM::LLVMPointerType::get(rewriter.getContext()),
      ValueRange{integer.getResult()});
}

LogicalResult normalizeUnityOpenCLHostCalls(ModuleOp module) {
  SmallVector<LLVM::CallOp> calls;
  module.walk([&](LLVM::CallOp call) {
    if (call.getCallee() && call.getCallee()->starts_with("cl"))
      calls.push_back(call);
  });
  if (calls.empty())
    return success();

  auto invalid = [](LLVM::CallOp call, const Twine &message) {
    return call.emitError() << "Unity OpenCL AOT subset " << message;
  };
  std::set<Operation *> hostFunctions;
  for (LLVM::CallOp call : calls)
    hostFunctions.insert(call->getParentOfType<LLVM::LLVMFuncOp>());
  for (Operation *operation : hostFunctions) {
    bool hasConditionalBranch = false;
    operation->walk([&](LLVM::CondBrOp) { hasConditionalBranch = true; });
    if (hasConditionalBranch)
      return invalid(calls.front(),
                     "does not support control flow around setup calls; use a "
                     "straight-line AOT setup function");
  }
  for (LLVM::CallOp call : calls) {
    StringRef name = *call.getCallee();
    auto operands = call.getCalleeOperands();
    if (name == "clGetPlatformIDs") {
      if (operands.size() != 3 || llvmConstantInt(operands[0]) != 1 ||
          llvmNull(operands[1]))
        return invalid(call, "requires clGetPlatformIDs(1, &platform, ...)");
    } else if (name == "clGetDeviceIDs") {
      if (operands.size() != 5 || llvmConstantInt(operands[1]) != 4 ||
          llvmConstantInt(operands[2]) != 1 || llvmNull(operands[3]))
        return invalid(call, "requires one CL_DEVICE_TYPE_GPU device request");
    } else if (name == "clCreateContext") {
      if (operands.size() != 6 || !llvmNull(operands[0]) ||
          llvmConstantInt(operands[1]) != 1 || llvmNull(operands[2]) ||
          !llvmNull(operands[3]) || !llvmNull(operands[4]))
        return invalid(call, "requires one device and no context properties or "
                             "callback");
    } else if (name == "clCreateCommandQueue") {
      if (operands.size() != 4 || llvmConstantInt(operands[2]) != 0)
        return invalid(call, "requires an in-order command queue");
    } else if (name == "clCreateBuffer") {
      auto flags =
          operands.size() == 5 ? llvmConstantInt(operands[1]) : std::nullopt;
      if (operands.size() != 5 ||
          (!flags || (*flags != 1 && *flags != 2 && *flags != 4)) ||
          !llvmConstantInt(operands[2]) || *llvmConstantInt(operands[2]) <= 0 ||
          !llvmNull(operands[3]))
        return invalid(call, "requires a positive constant size, one standard "
                             "access flag, and a null host pointer");
    } else if (name == "clEnqueueCopyBuffer") {
      if (operands.size() != 9 || llvmConstantInt(operands[6]) != 0 ||
          !llvmNull(operands[7]) || !llvmNull(operands[8]))
        return invalid(
            call, "requires an empty event wait list and null output event");
    } else if (name != "clReleaseMemObject" &&
               name != "clReleaseCommandQueue" && name != "clReleaseContext") {
      return invalid(call, "does not support Host API call ") << name;
    }
  }

  IRRewriter rewriter(module.getContext());
  rewriter.setInsertionPointToEnd(module.getBody());
  auto pointer = LLVM::LLVMPointerType::get(module.getContext());
  auto i32 = rewriter.getI32Type();
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(openCLAllocName))
    LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), openCLAllocName,
                             LLVM::LLVMFunctionType::get(
                                 LLVM::LLVMVoidType::get(module.getContext()),
                                 {pointer, i32, i32}, false));
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(openCLCopyName))
    LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), openCLCopyName,
                             LLVM::LLVMFunctionType::get(
                                 LLVM::LLVMVoidType::get(module.getContext()),
                                 {pointer, pointer, pointer, i32, i32, i32},
                                 false));
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(openCLDeviceName))
    LLVM::LLVMFuncOp::create(
        rewriter, module.getLoc(), openCLDeviceName,
        LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(module.getContext()), {pointer}, false));
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(openCLContextName))
    LLVM::LLVMFuncOp::create(
        rewriter, module.getLoc(), openCLContextName,
        LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(module.getContext()), {pointer}, false));
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(openCLQueueName))
    LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), openCLQueueName,
                             LLVM::LLVMFunctionType::get(
                                 LLVM::LLVMVoidType::get(module.getContext()),
                                 {pointer, pointer}, false));
  int32_t nextBufferHandle = 16;
  for (LLVM::CallOp call : calls) {
    rewriter.setInsertionPoint(call);
    StringRef name = *call.getCallee();
    auto operands = call.getCalleeOperands();
    auto success = [&]() -> mlir::Value {
      return rewriter.create<LLVM::ConstantOp>(
          call.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    };
    if (name == "clGetPlatformIDs") {
      rewriter.create<LLVM::StoreOp>(call.getLoc(),
                                     openCLSentinel(rewriter, call.getLoc(), 1),
                                     operands[1]);
      if (!llvmNull(operands[2])) {
        auto one = rewriter.create<LLVM::ConstantOp>(
            call.getLoc(), rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(1));
        rewriter.create<LLVM::StoreOp>(call.getLoc(), one, operands[2]);
      }
      rewriter.replaceOp(call, success());
    } else if (name == "clGetDeviceIDs") {
      createLLVMCall(rewriter, call, openCLDeviceName, TypeRange{},
                     ValueRange{operands[0]});
      rewriter.create<LLVM::StoreOp>(call.getLoc(),
                                     openCLSentinel(rewriter, call.getLoc(), 2),
                                     operands[3]);
      if (!llvmNull(operands[4])) {
        auto one = rewriter.create<LLVM::ConstantOp>(
            call.getLoc(), rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(1));
        rewriter.create<LLVM::StoreOp>(call.getLoc(), one, operands[4]);
      }
      rewriter.replaceOp(call, success());
    } else if (name == "clCreateContext") {
      auto device =
          rewriter.create<LLVM::LoadOp>(call.getLoc(), pointer, operands[2]);
      createLLVMCall(rewriter, call, openCLContextName, TypeRange{},
                     ValueRange{device.getResult()});
      storeOpenCLSuccess(rewriter, call.getLoc(), operands[5]);
      rewriter.replaceOp(call, openCLSentinel(rewriter, call.getLoc(), 3));
    } else if (name == "clCreateCommandQueue") {
      createLLVMCall(rewriter, call, openCLQueueName, TypeRange{},
                     ValueRange{operands[0], operands[1]});
      storeOpenCLSuccess(rewriter, call.getLoc(), operands[3]);
      rewriter.replaceOp(call, openCLSentinel(rewriter, call.getLoc(), 4));
    } else if (name == "clCreateBuffer") {
      storeOpenCLSuccess(rewriter, call.getLoc(), operands[4]);
      auto handle = rewriter.create<LLVM::ConstantOp>(
          call.getLoc(), i32, rewriter.getI32IntegerAttr(nextBufferHandle));
      SmallVector<mlir::Value> arguments = {operands[0], handle.getResult(),
                                            operands[2]};
      createLLVMCall(rewriter, call, openCLAllocName, TypeRange{}, arguments);
      rewriter.replaceOp(
          call, openCLSentinel(rewriter, call.getLoc(), nextBufferHandle));
      ++nextBufferHandle;
    } else if (name == "clEnqueueCopyBuffer") {
      SmallVector<mlir::Value> arguments = {operands[0], operands[1],
                                            operands[2], operands[3],
                                            operands[4], operands[5]};
      createLLVMCall(rewriter, call, openCLCopyName, TypeRange{}, arguments);
      rewriter.replaceOp(call, success());
    } else {
      rewriter.replaceOp(call, success());
    }
  }
  return success();
}

LogicalResult recoverUnityOpenCLBufferCalls(ModuleOp module) {
  SmallVector<LLVM::CallOp> allocations, copies, devices, contexts, queues;
  module.walk([&](LLVM::CallOp call) {
    StringRef name = call.getCallee().value_or("");
    if (name == openCLAllocName)
      allocations.push_back(call);
    else if (name == openCLCopyName)
      copies.push_back(call);
    else if (name == openCLDeviceName)
      devices.push_back(call);
    else if (name == openCLContextName)
      contexts.push_back(call);
    else if (name == openCLQueueName)
      queues.push_back(call);
  });
  if (allocations.empty() && copies.empty() && devices.empty() &&
      contexts.empty() && queues.empty())
    return success();

  llvm::DenseMap<int64_t, mlir::Value> buffers;
  IRRewriter rewriter(module.getContext());
  auto i32 = rewriter.getI32Type();
  auto getHandle = [](mlir::Value pointer) -> std::optional<int64_t> {
    auto cast = pointer.getDefiningOp<LLVM::IntToPtrOp>();
    return cast ? llvmConstantInt(cast.getArg()) : std::nullopt;
  };
  for (LLVM::CallOp device : devices)
    if (getHandle(device.getOperand(0)) != 1)
      return device.emitError(
          "clGetDeviceIDs must use the platform returned by clGetPlatformIDs");
  for (LLVM::CallOp context : contexts)
    if (getHandle(context.getOperand(0)) != 2)
      return context.emitError(
          "clCreateContext must use the device returned by clGetDeviceIDs");
  for (LLVM::CallOp queue : queues)
    if (getHandle(queue.getOperand(0)) != 3 ||
        getHandle(queue.getOperand(1)) != 2)
      return queue.emitError("clCreateCommandQueue must use the context and "
                             "device returned by the AOT setup sequence");
  for (auto [allocationIndex, allocation] : llvm::enumerate(allocations)) {
    if (getHandle(allocation.getOperand(0)) != 3)
      return allocation.emitError(
          "clCreateBuffer must use the context returned by clCreateContext");
    auto handle = llvmConstantInt(allocation.getOperand(1));
    if (!handle)
      return allocation.emitError("internal OpenCL buffer handle must be "
                                  "a compiler-generated constant");
    rewriter.setInsertionPoint(allocation);
    auto logicalId = arith::ConstantIntOp::create(
        rewriter, allocation.getLoc(),
        static_cast<int64_t>(allocationIndex + 1), 32);
    OperationState state(allocation.getLoc(),
                         ::vrc::vrc::BufferAllocOp::getOperationName());
    state.addOperands({logicalId.getResult(), allocation.getOperand(2)});
    state.addTypes({::vrc::vrc::BufferType::get(module.getContext()), i32});
    state.addAttribute("format", rewriter.getStringAttr("rgba8_unorm"));
    state.addAttribute("usage", rewriter.getStringAttr("storage"));
    Operation *operation = rewriter.create(state);
    if (!buffers.try_emplace(*handle, operation->getResult(0)).second)
      return allocation.emitError("duplicate internal OpenCL buffer handle");
  }
  for (LLVM::CallOp copy : copies) {
    if (getHandle(copy.getOperand(0)) != 4)
      return copy.emitError(
          "clEnqueueCopyBuffer must use the queue returned by "
          "clCreateCommandQueue");
    auto sourceHandle = getHandle(copy.getOperand(1));
    auto destinationHandle = getHandle(copy.getOperand(2));
    mlir::Value source =
        sourceHandle ? buffers.lookup(*sourceHandle) : mlir::Value();
    mlir::Value destination =
        destinationHandle ? buffers.lookup(*destinationHandle) : mlir::Value();
    if (!source || !destination)
      return copy.emitError("standard OpenCL buffer handles must flow directly "
                            "from clCreateBuffer to clEnqueueCopyBuffer");
    rewriter.setInsertionPoint(copy);
    OperationState state(copy.getLoc(),
                         ::vrc::vrc::BufferCopyOp::getOperationName());
    state.addOperands({source, destination, copy.getOperand(3),
                       copy.getOperand(4), copy.getOperand(5)});
    state.addTypes(::vrc::vrc::EventType::get(module.getContext()));
    rewriter.create(state);
    rewriter.eraseOp(copy);
  }
  for (LLVM::CallOp allocation : allocations) {
    if (!allocation->use_empty())
      return allocation.emitError(
          "standard OpenCL buffer handles cannot escape the fixed CopyBuffer "
          "subset");
    rewriter.eraseOp(allocation);
  }
  for (LLVM::CallOp call : devices)
    rewriter.eraseOp(call);
  for (LLVM::CallOp call : contexts)
    rewriter.eraseOp(call);
  for (LLVM::CallOp call : queues)
    rewriter.eraseOp(call);
  constexpr StringLiteral prototypes[] = {
      "clGetPlatformIDs",     "clGetDeviceIDs",        "clCreateContext",
      "clCreateCommandQueue", "clCreateBuffer",        "clEnqueueCopyBuffer",
      "clReleaseMemObject",   "clReleaseCommandQueue", "clReleaseContext",
      openCLAllocName,        openCLCopyName,          openCLDeviceName,
      openCLContextName,      openCLQueueName,
  };
  for (StringLiteral name : prototypes)
    if (auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
      if (function.isExternal())
        rewriter.eraseOp(function);
  return success();
}

FailureOr<OwningOpRef<ModuleOp>>
recoverLinkedUnityHostLibrary(StringRef filename, MLIRContext &context) {
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
  auto annotatedExports = collectUdonCLAnnotatedExports(*inputModule);
  if (failed(annotatedExports))
    return failure();
  if (!inputModule->getFunction("__vrc_opencl_buffer_alloc")) {
    errs() << "--host-library-llvm requires the linked OpenCL AOT runtime\n";
    return failure();
  }
  auto library = ::vrc::importLLVM(std::move(inputModule), context);
  if (!library || failed(verify(*library)))
    return failure();
  for (const std::string &symbol : *annotatedExports) {
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
  recovery.addPass(::vrc::createLowerOpenCLAOTIntrinsicsPass());
  if (failed(recovery.run(*library)) || failed(verify(*library)))
    return failure();
  for (const std::string &symbol : *annotatedExports) {
    auto function = library->lookupSymbol<func::FuncOp>(symbol);
    if (!function) {
      errs() << "UDONCL_EXPORT definition was lost while recovering Host "
                "library: "
             << symbol << "\n";
      return failure();
    }
    function->setAttr("udoncl.export", UnitAttr::get(&context));
  }
  return library;
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

std::string unityLibraryBufferCopyShader(StringRef program) {
  return (Twine("Shader \"Hidden/UdonCL/") + program +
          "/BufferCopy\" {\n"
          "  Properties { _MainTex (\"Source\", 2D) = \"black\" {} }\n"
          "  SubShader { Pass { ZTest Always Cull Off ZWrite Off\n"
          "    HLSLPROGRAM\n"
          "    #pragma vertex vert_img\n"
          "    #pragma fragment frag\n"
          "    #include \"UnityCG.cginc\"\n"
          "    Texture2D<float4> _MainTex;\n"
          "    float4 frag(v2f_img input) : SV_Target {\n"
          "      uint width, height; _MainTex.GetDimensions(width, height);\n"
          "      uint2 p = min(uint2(input.pos.xy), "
          "uint2(width - 1, height - 1));\n"
          "      return _MainTex.Load(int3(p, 0));\n"
          "    }\n"
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
  if (runtimeValidation && noRuntimeValidation) {
    errs() << "--runtime-validation and --no-runtime-validation are mutually "
              "exclusive\n";
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
  func::registerInlinerExtension(registry);
  registry
      .insert<::vrc::udonasm::UdonAsmDialect, ::vrc::udonhl::UdonHLDialect,
              ::vrc::vrc::VRCDialect, ::vrc::vrcgpu::VRCGPUDialect,
              arith::ArithDialect, async::AsyncDialect, cf::ControlFlowDialect,
              func::FuncDialect, gpu::GPUDialect, memref::MemRefDialect>();
  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module;
  std::map<std::string, std::string> files;
  bool gatherProfile = false;
  bool unityGatherProfile = false;
  std::optional<json::Object> unityConfig;
  OwningOpRef<ModuleOp> unityDeviceModule;
  SmallVector<const json::Object *> unityDeviceConfigs;
  SmallVector<UnityBuffer> unityBuffers;
  bool unityUsesBufferCopy = false;
  SmallVector<int32_t> unityConfiguredKernelIds;
  std::set<std::string> annotatedHostExports;
  if (emitUnityLibrary) {
    auto input = MemoryBuffer::getFile(inputFilename);
    if (!input || (hostLLVM.empty() == hostMLIR.empty()) ||
        (!hostLibraryLLVM.empty() && hostMLIR.empty())) {
      errs() << "Unity library mode requires a JSON config and exactly one of "
                "--host-llvm or --host-mlir; --host-library-llvm is valid "
                "with --host-mlir; --device-llvm is optional\n";
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
    auto parsedDeviceConfigs = parseUnityDeviceKernels(*unityConfig);
    if (failed(parsedDeviceConfigs))
      return 1;
    unityDeviceConfigs = std::move(*parsedDeviceConfigs);
    if (unityDeviceConfigs.empty() != deviceLLVM.empty()) {
      errs() << "Unity library Device kernel config and --device-llvm must be "
                "provided together\n";
      return 1;
    }
    auto parsedBuffers = parseUnityBuffers(*unityConfig);
    if (failed(parsedBuffers))
      return 1;
    unityBuffers = std::move(*parsedBuffers);
    if (!deviceLLVM.empty()) {
      unityDeviceModule = ModuleOp::create(UnknownLoc::get(&context));
      if (!unityBuffers.empty() &&
          llvm::all_of(unityBuffers, [&](const UnityBuffer &buffer) {
            return buffer.width == unityBuffers.front().width;
          }))
        (*unityDeviceModule)
            ->setAttr("vrc.texture_row_pitch",
                      mlir::IntegerAttr::get(
                          mlir::IntegerType::get(&context, 32),
                          static_cast<int32_t>(unityBuffers.front().width)));
      if (!unityBuffers.empty() &&
          llvm::all_of(unityBuffers, [&](const UnityBuffer &buffer) {
            return buffer.width == unityBuffers.front().width &&
                   buffer.height == unityBuffers.front().height;
          }))
        (*unityDeviceModule)
            ->setAttr("vrc.texture_shape",
                      DenseI32ArrayAttr::get(
                          &context,
                          {static_cast<int32_t>(unityBuffers.front().width),
                           static_cast<int32_t>(unityBuffers.front().height)}));
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
      if (failed(verify(*unityDeviceModule)))
        return 1;
    }
    if (!hostMLIR.empty()) {
      module = parseSourceFile<ModuleOp>(hostMLIR, &context);
      if (!module || failed(verify(*module)))
        return 1;
      if (unityConfig->getBoolean("require_device_graph").value_or(false)) {
        auto deviceGraph =
            (*module)->getAttrOfType<BoolAttr>("udoncl.onnx_device_graph");
        if (!deviceGraph || !deviceGraph.getValue()) {
          errs() << "Unity library config requires a fully device-resident "
                    "ONNX graph";
          if (auto unsupported = (*module)->getAttrOfType<ArrayAttr>(
                  "udoncl.onnx_device_unsupported"))
            for (mlir::Attribute attribute : unsupported) {
              auto entry = cast<DictionaryAttr>(attribute);
              errs() << "\n  "
                     << entry.getAs<StringAttr>("operation").getValue() << "@"
                     << entry.getAs<StringAttr>("provider").getValue() << ": "
                     << entry.getAs<IntegerAttr>("count").getInt();
            }
          errs() << "\n";
          return 1;
        }
      }
      if (!hostLibraryLLVM.empty()) {
        auto library = recoverLinkedUnityHostLibrary(hostLibraryLLVM, context);
        if (failed(library) || failed(linkUnityHostLibrary(*module, **library)))
          return 1;
      }
      {
        PassManager linkedHostRecovery(&context);
        linkedHostRecovery.enableVerifier(false);
        linkedHostRecovery.addPass(::vrc::createLowerOpenCLAOTIntrinsicsPass());
        if (failed(linkedHostRecovery.run(*module)) || failed(verify(*module)))
          return 1;
      }
      for (func::FuncOp function : module->getOps<func::FuncOp>())
        if (function->hasAttr("udoncl.export"))
          annotatedHostExports.insert(function.getSymName().str());
      if (failed(verify(*module)) ||
          !configureUnityExports(*module, *unityConfig, annotatedHostExports))
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
      auto discoveredExports = collectUdonCLAnnotatedExports(*inputModule);
      if (failed(discoveredExports))
        return 1;
      annotatedHostExports = std::move(*discoveredExports);
      if (inputModule->getTargetTriple().str() != "spir-unknown-unknown" ||
          !inputModule->getDataLayout().isLittleEndian() ||
          inputModule->getDataLayout().getPointerSizeInBits() != 32) {
        errs()
            << "Unity Host library requires 32-bit little-endian SPIR input\n";
        return 1;
      }
      auto *enqueue = inputModule->getFunction("clEnqueueNDRangeKernel");
      bool linkedOpenCLAOT =
          inputModule->getFunction("__vrc_opencl_buffer_alloc") != nullptr;
      if (enqueue && !enqueue->use_empty() && !linkedOpenCLAOT) {
        if (unityDeviceConfigs.size() != 1) {
          errs() << "legacy OpenCL recovery supports one configured kernel\n";
          return 1;
        }
        auto *device = unityDeviceConfigs.front();
        module = device && device->getString("profile") == "gather-f32"
                     ? recoverUnityOpenCLGatherDispatchProgram(
                           *inputModule, *unityConfig, context)
                     : recoverUnityOpenCLDispatchProgram(*inputModule,
                                                         *unityConfig, context);
        if (!module || failed(verify(*module)) ||
            !configureUnityExports(*module, *unityConfig, annotatedHostExports))
          return 1;
      } else {
        module = ::vrc::importLLVM(std::move(inputModule), context);
        if (!module || failed(verify(*module)))
          return 1;
        if (!linkedOpenCLAOT && failed(normalizeUnityOpenCLHostCalls(*module)))
          return 1;
        PassManager cleanup(&context);
        cleanup.addPass(createSROA());
        cleanup.addPass(createMem2Reg());
        cleanup.addPass(createCanonicalizerPass());
        if (failed(cleanup.run(*module)) ||
            (!linkedOpenCLAOT &&
             failed(recoverUnityOpenCLBufferCalls(*module))))
          return 1;
        PassManager recovery(&context);
        // The first canonicalization removes dead opaque OpenCL handles while
        // typed buffer operations still temporarily live in llvm.func.  The
        // following pass immediately raises that function to the Host runtime
        // path, where the VRC operation verifier accepts them.
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
            (!linkedOpenCLAOT &&
             failed(recoverUnityCBufferIntrinsics(*module))) ||
            !configureUnityExports(*module, *unityConfig, annotatedHostExports))
          return 1;
      }
    }
    if (!omitIntermediateIR) {
      raw_string_ostream recovered(files["frontend.recovered.mlir"]);
      module->print(recovered);
      recovered << "\n";
    }
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
  (*module)->setAttr("vrc.runtime_validation.read_write_alias",
                     BoolAttr::get(&context, runtimeValidation));
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
      unityConfiguredKernelIds.push_back(static_cast<int32_t>(kernelId));
    }
    if (failed(propagateUnityHostAsyncEffects(*module)) ||
        failed(optimizeUnityHostStorage(*module)) || failed(verify(*module)))
      return 1;
    OwningOpRef<ModuleOp> asyncPreview = cast<ModuleOp>((*module)->clone());
    PassManager asyncPipeline(&context);
    asyncPipeline.addPass(::vrc::createLowerHostToAsyncPass());
    asyncPipeline.addPass(createAsyncFuncToAsyncRuntimePass());
    asyncPipeline.addPass(::vrc::createPrepareUdonAsyncRuntimePass());
    if (failed(asyncPipeline.run(*asyncPreview)) ||
        failed(verify(*asyncPreview)))
      return 1;
    if (!omitIntermediateIR) {
      raw_string_ostream asyncIR(files["host.async-runtime.mlir"]);
      asyncPreview->print(asyncIR);
      asyncIR << "\n";
    }

    PassManager udonAsyncPipeline(&context);
    udonAsyncPipeline.addPass(::vrc::createLowerUdonAsyncRuntimePass());
    if (failed(udonAsyncPipeline.run(*asyncPreview)) ||
        failed(verify(*asyncPreview)))
      return 1;
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

    manager.addPass(::vrc::createLowerHostToUdonHLPass());
    manager.addPass(::vrc::createSubstituteRuntimeIntrinsicsPass());
    if (failed(manager.run(*module)) || failed(verify(*module)))
      return 1;
    auto bufferBinding = bindUnityBufferOperations(
        *module, unityBuffers, configuredKernelIds, unityStaticTensors);
    if (failed(bufferBinding))
      return 1;
    unityUsesBufferCopy = bufferBinding->usesCopy;
    auto manifest = ::vrc::buildBoundaryManifest(*module);
    if (auto onnxBinding = unityOnnxBindingManifest(*module))
      manifest["onnx_binding"] = std::move(*onnxBinding);
    bool usesAsyncCommands = false;
    module->walk([&](Operation *operation) {
      usesAsyncCommands |=
          isa<::vrc::vrc::BufferUploadOp, ::vrc::vrc::BufferReadbackOp,
              ::vrc::vrc::AwaitOp>(operation);
    });
    auto program = unityConfig->getString("program");
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
    if (usesAsyncCommands)
      (*manifest["program"].getAsObject())["async_commands"] = json::Object{
          {"completion_status_slot", "__udoncl_async_completion_status"},
          {"pending_object_slot", "__udoncl_async_pending_object"},
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
      };
    auto *device = manifest["device"].getAsObject();
    if (!device) {
      errs() << "Unity library manifest is missing its Device section\n";
      return 1;
    }
    (*device)["buffers"] = unityBufferManifest(unityBuffers);
    if (auto bindingNamespace =
            (*module)->getAttrOfType<StringAttr>("vrc.binding_namespace")) {
      auto bindingClass =
          (*module)->getAttrOfType<StringAttr>("vrc.binding_class");
      if (!bindingClass) {
        errs() << "binding namespace requires a binding class\n";
        return 1;
      }
      auto *programManifest = manifest["program"].getAsObject();
      (*programManifest)["binding_namespace"] =
          bindingNamespace.getValue().str();
      (*programManifest)["binding_class"] = bindingClass.getValue().str();
      ::vrc::refreshPublicABIFingerprint(manifest);
      auto binding = ::vrc::buildUdonSharpBindingSource(
          manifest, bindingNamespace.getValue(), bindingClass.getValue());
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
    udon.addPass(::vrc::createLowerUdonHLToUdonAsmPass());
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
        FailureOr<std::string> bytes = failure();
        if (data) {
          bytes = denseTensorBytes(data);
        } else if (id && !staticTensorDirectory.empty()) {
          SmallString<256> sourcePath(staticTensorDirectory);
          sys::path::append(sourcePath,
                            "tensor_" + std::to_string(id.getInt()) + ".bytes");
          auto source = MemoryBuffer::getFile(sourcePath, false, false);
          if (source)
            bytes = std::string((*source)->getBuffer());
        }
        if (!id || !name || !dataType || !shape || !layout || !alignment ||
            !byteStrides || !byteLength || !sha256 || failed(bytes) ||
            layout.getValue() != "row_major_contiguous" ||
            alignment.getInt() < 1 || byteStrides.size() != shape.size() ||
            byteLength.getInt() < 1 ||
            static_cast<uint64_t>(byteLength.getInt()) != bytes->size() ||
            llvm::toHex(llvm::SHA256::hash(ArrayRef<uint8_t>(
                            reinterpret_cast<const uint8_t *>(bytes->data()),
                            bytes->size())),
                        true) != sha256.getValue()) {
          errs() << "immutable static tensor metadata is malformed\n";
          return 1;
        }
        int64_t tensorId = id.getInt();
        std::string resourceName = "static_tensor_" + std::to_string(tensorId);
        std::string path =
            "static-tensors/tensor_" + std::to_string(tensorId) + ".bytes";
        files[path] = std::move(*bytes);
        int64_t pixels = llvm::divideCeil(byteLength.getInt(), int64_t{4});
        int64_t width = std::min<int64_t>(pixels, 8192);
        int64_t height = llvm::divideCeil(pixels, width);
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
            {"texture", json::Object{{"abi", "udoncl-static-rgba32-v1"},
                                     {"width", width},
                                     {"height", height},
                                     {"format", "rgba32"},
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
    appendUnityBufferSlots(physicalModule, unityBuffers);
    if (unityUsesBufferCopy)
      appendUnityCopyMaterialSlot(physicalModule);
    appendUnityDispatchMaterialSlots(physicalModule, unityConfiguredKernelIds);
    std::string assembly;
    raw_string_ostream assemblyOut(assembly);
    if (failed(::vrc::emitUdonAssembly(physicalModule, assemblyOut)))
      return 1;
    files["host.uasm"] = std::move(assembly);
    files["display.shader"] = unityLibraryDisplayShader(*program);
    if (unityUsesBufferCopy || !unityBuffers.empty())
      files["buffer-copy.shader"] = unityLibraryBufferCopyShader(*program);
    if (!omitIntermediateIR) {
      raw_string_ostream lowered(files["host.udonasm.mlir"]);
      physicalModule.print(lowered);
      lowered << "\n";
    }
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
