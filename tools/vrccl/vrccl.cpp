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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/PatternMatch.h"
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
cl::opt<std::string>
    hostMLIR("host-mlir", cl::desc("Typed Host MLIR for compiler validation"));
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

struct UnityBuffer {
  int64_t id;
  int64_t logicalBytes;
  int64_t width;
  int64_t height;
  std::string name;
  std::string heapSlot;
};

FailureOr<SmallVector<UnityBuffer>>
parseUnityBuffers(const json::Object &config) {
  SmallVector<UnityBuffer> result;
  auto *buffers = config.getArray("buffers");
  if (!buffers)
    return result;
  std::set<int64_t> ids;
  std::set<std::string> names;
  for (const json::Value &value : *buffers) {
    auto *entry = value.getAsObject();
    auto id = entry ? entry->getInteger("id") : std::nullopt;
    auto name = entry ? entry->getString("name") : std::nullopt;
    auto bytes = entry ? entry->getInteger("byte_size") : std::nullopt;
    constexpr int64_t maximumBytes = 8192LL * 8192LL * 4LL;
    if (!id || *id <= 0 || *id > INT32_MAX || !name || !identifier(*name) ||
        !bytes || *bytes <= 0 || *bytes > maximumBytes ||
        !ids.insert(*id).second || !names.insert(name->str()).second) {
      errs() << "invalid or duplicate Unity Device buffer\n";
      return failure();
    }
    int64_t pixels = (*bytes + 3) / 4;
    int64_t width = std::min<int64_t>(pixels, 8192);
    int64_t height = (pixels + width - 1) / width;
    result.push_back({*id, *bytes, width, height, name->str(),
                      "__udoncl_buffer_" + std::to_string(*id)});
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
                          std::optional<int64_t> configuredKernelId) {
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
    module.emitError("Unity library has more buffer allocation sites than "
                     "configured Device buffers");
    return failure();
  }
  OpBuilder builder(module.getContext());
  llvm::DenseMap<Operation *, const UnityBuffer *> bound;
  for (auto [index, allocation] : llvm::enumerate(allocations)) {
    auto constant =
        allocation.getByteSize().getDefiningOp<arith::ConstantIntOp>();
    const UnityBuffer &buffer = buffers[index];
    if (!constant || constant.value() != buffer.logicalBytes) {
      allocation.emitError()
          << "Unity buffer allocation " << index
          << " must have constant byte size " << buffer.logicalBytes
          << " from the configured pool order";
      return failure();
    }
    allocation->setAttr("vrc.unity_buffer_id",
                        builder.getI32IntegerAttr(buffer.id));
    allocation->setAttr("vrc.unity_heap_slot",
                        builder.getStringAttr(buffer.heapSlot));
    allocation->setAttr("vrc.unity_logical_bytes",
                        builder.getI32IntegerAttr(buffer.logicalBytes));
    bound.try_emplace(allocation.getOperation(), &buffer);
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
    auto output = dispatch.getResources().size() == 1
                      ? dispatch.getResources()
                            .front()
                            .getDefiningOp<::vrc::vrc::BufferAllocOp>()
                      : ::vrc::vrc::BufferAllocOp();
    if (!configuredKernelId || dispatch.getKernelId() != *configuredKernelId ||
        !descriptorMatches || dispatch.getArgumentKinds().empty() ||
        !isa<StringAttr>(*dispatch.getArgumentKinds().begin()) ||
        cast<StringAttr>(*dispatch.getArgumentKinds().begin()).getValue() !=
            "buffer" ||
        !output || !bound.contains(output.getOperation())) {
      dispatch.emitError(
          "initial Unity dispatch requires the configured kernel, one direct "
          "output allocation, and a copied f32/i32 scalar descriptor");
      return failure();
    }
    int32_t kernelId = static_cast<int32_t>(dispatch.getKernelId());
    dispatch->setAttr("vrc.unity_dispatch_material_slot",
                      builder.getStringAttr("__udoncl_kernel_material_" +
                                            std::to_string(kernelId)));
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
  for (const UnityBuffer &buffer : buffers)
    result.push_back(json::Object{
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
    });
  return result;
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
          auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
          if (callee && callee->hasAttr("vrc.intrinsic"))
            continue;
          call.emitError("Unity library helper calls must be inlined before "
                         "Udon lowering");
          return false;
        }
  for (func::FuncOp function :
       llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
    if (!selected.count(function.getSymName().str()) &&
        !function->hasAttr("vrc.intrinsic"))
      function.erase();
  module->setAttr("vrc.target", builder.getStringAttr("vrc-pc"));
  return true;
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
       allocPrototype.getArgumentTypes() != ArrayRef<mlir::Type>{i32} ||
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
    state.addOperands(allocation.getOperand(0));
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
  for (LLVM::CallOp allocation : allocations) {
    if (getHandle(allocation.getOperand(0)) != 3)
      return allocation.emitError(
          "clCreateBuffer must use the context returned by clCreateContext");
    auto handle = llvmConstantInt(allocation.getOperand(1));
    if (!handle)
      return allocation.emitError("internal OpenCL buffer handle must be "
                                  "a compiler-generated constant");
    rewriter.setInsertionPoint(allocation);
    OperationState state(allocation.getLoc(),
                         ::vrc::vrc::BufferAllocOp::getOperationName());
    state.addOperands(allocation.getOperand(2));
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
  OwningOpRef<ModuleOp> unityDeviceModule;
  SmallVector<UnityBuffer> unityBuffers;
  bool unityUsesBufferCopy = false;
  SmallVector<int32_t> unityDispatchKernelIds;
  if (emitUnityLibrary) {
    auto input = MemoryBuffer::getFile(inputFilename);
    if (!input || (hostLLVM.empty() == hostMLIR.empty())) {
      errs() << "Unity library mode requires a JSON config and exactly one of "
                "--host-llvm or --host-mlir; --device-llvm is optional\n";
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
    if ((unityConfig->getObject("device_kernel") != nullptr) !=
        !deviceLLVM.empty()) {
      errs() << "Unity library device_kernel and --device-llvm must be "
                "provided together\n";
      return 1;
    }
    auto parsedBuffers = parseUnityBuffers(*unityConfig);
    if (failed(parsedBuffers))
      return 1;
    unityBuffers = std::move(*parsedBuffers);
    if (!deviceLLVM.empty()) {
      auto *deviceConfig = unityConfig->getObject("device_kernel");
      unityDeviceModule =
          ::vrc::importCBoundary({}, deviceLLVM, *deviceConfig, context);
      if (!unityDeviceModule || failed(verify(*unityDeviceModule)))
        return 1;
      PassManager deviceRecovery(&context);
      deviceRecovery.addNestedPass<ModuleOp>(createMem2Reg());
      deviceRecovery.addNestedPass<ModuleOp>(
          ::vrc::createRaiseLLVMBoundaryDevicePass());
      deviceRecovery.addNestedPass<ModuleOp>(
          ::vrc::createRaiseLLVMToScalarPass());
      deviceRecovery.addNestedPass<ModuleOp>(createCanonicalizerPass());
      deviceRecovery.addPass(::vrc::createPrepareNativeDevicePass());
      if (failed(deviceRecovery.run(*unityDeviceModule)) ||
          failed(verify(*unityDeviceModule)))
        return 1;
    }
    if (!hostMLIR.empty()) {
      module = parseSourceFile<ModuleOp>(hostMLIR, &context);
      if (!module || failed(verify(*module)) ||
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
      if (inputModule->getTargetTriple().str() != "spir-unknown-unknown" ||
          !inputModule->getDataLayout().isLittleEndian() ||
          inputModule->getDataLayout().getPointerSizeInBits() != 32) {
        errs()
            << "Unity Host library requires 32-bit little-endian SPIR input\n";
        return 1;
      }
      module = ::vrc::importLLVM(std::move(inputModule), context);
      if (!module || failed(verify(*module)))
        return 1;
      if (failed(normalizeUnityOpenCLHostCalls(*module)))
        return 1;
      PassManager cleanup(&context);
      cleanup.addPass(createMem2Reg());
      cleanup.addPass(createCanonicalizerPass());
      if (failed(cleanup.run(*module)) ||
          failed(recoverUnityOpenCLBufferCalls(*module)))
        return 1;
      PassManager recovery(&context);
      // The first canonicalization removes dead opaque OpenCL handles while
      // typed buffer operations still temporarily live in llvm.func.  The
      // following pass immediately raises that function to the Host runtime
      // path, where the VRC operation verifier accepts them.
      recovery.enableVerifier(false);
      recovery.addPass(createCanonicalizerPass());
      recovery.addPass(::vrc::createRaiseLLVMToScalarPass());
      recovery.addPass(createCanonicalizerPass());
      if (failed(recovery.run(*module)) ||
          failed(recoverUnityCBufferIntrinsics(*module)) ||
          !configureUnityExports(*module, *unityConfig))
        return 1;
    }
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
    manager.addPass(::vrc::createSubstituteRuntimeIntrinsicsPass());
    if (failed(manager.run(*module)) || failed(verify(*module)))
      return 1;
    auto *deviceKernel = unityConfig->getObject("device_kernel");
    std::optional<int64_t> configuredKernelId =
        deviceKernel ? deviceKernel->getInteger("kernel_id") : std::nullopt;
    auto bufferBinding =
        bindUnityBufferOperations(*module, unityBuffers, configuredKernelId);
    if (failed(bufferBinding))
      return 1;
    unityUsesBufferCopy = bufferBinding->usesCopy;
    unityDispatchKernelIds = std::move(bufferBinding->dispatchKernelIds);
    auto manifest = ::vrc::buildBoundaryManifest(*module);
    auto program = unityConfig->getString("program");
    manifest["artifact_stage"] = "unity-udoncl-library";
    manifest["program"] = json::Object{{"name", *program},
                                       {"assembly", "host.uasm"},
                                       {"display_shader", "display.shader"},
                                       {"ir", "host.udonasm.mlir"}};
    if (unityUsesBufferCopy)
      (*manifest["program"].getAsObject())["buffer_copy_shader"] =
          "buffer-copy.shader";
    manifest["ready_for_unity"] = true;
    auto *device = manifest["device"].getAsObject();
    if (!device) {
      errs() << "Unity library manifest is missing its Device section\n";
      return 1;
    }
    (*device)["buffers"] = unityBufferManifest(unityBuffers);
    if (unityDeviceModule) {
      PassManager deviceLowering(&context);
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
    PassManager udon(&context);
    udon.addPass(::vrc::createLowerUdonHLToUdonAsmPass());
    if (failed(udon.run(*module)) || failed(verify(*module)))
      return 1;
    appendUnityBufferSlots(*module, unityBuffers);
    if (unityUsesBufferCopy)
      appendUnityCopyMaterialSlot(*module);
    appendUnityDispatchMaterialSlots(*module, unityDispatchKernelIds);
    std::string assembly;
    raw_string_ostream assemblyOut(assembly);
    if (failed(::vrc::emitUdonAssembly(*module, assemblyOut)))
      return 1;
    files["host.uasm"] = std::move(assembly);
    files["display.shader"] = unityLibraryDisplayShader(*program);
    if (unityUsesBufferCopy)
      files["buffer-copy.shader"] = unityLibraryBufferCopyShader(*program);
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
