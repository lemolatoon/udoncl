#include "VRC/Conversion/BoundaryFrontend/Frontend.h"
#include "VRC/Conversion/LLVMToHighLevel/Import.h"
#include "VRC/Conversion/LLVMToHighLevel/Passes.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_RAISELLVMBOUNDARYDEVICEPASS
#define GEN_PASS_DEF_LINKCBOUNDARYPASS
#define GEN_PASS_DEF_PREPARENATIVEDEVICEPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace runtime = ::vrc::vrc;
void fail(Operation *op, StringRef reason) {
  op->emitError() << "C boundary recovery: " << reason;
}
bool profile(const llvm::json::Object &config) {
  auto dims = [&](StringRef key, int64_t expected) {
    auto *array = config.getArray(key);
    return array && array->size() == 3 &&
           llvm::all_of(*array, [&](const auto &v) {
             return v.getAsInteger() == expected;
           });
  };
  auto words = config.getInteger("words");
  bool gather = config.getString("profile") == "gather-f32";
  auto *global = config.getArray("global_size");
  auto width = global && global->size() == 3 ? (*global)[0].getAsInteger()
                                             : std::optional<int64_t>();
  auto height = global && global->size() == 3 ? (*global)[1].getAsInteger()
                                              : std::optional<int64_t>();
  bool launch = gather ? words && *words >= 1 && *words <= 131072 && width &&
                             height && *width > 0 && *height > 0 &&
                             *width <= 16384 && *height <= 16384 &&
                             *width * *height == *words &&
                             (*global)[2].getAsInteger() == 1
                       : dims("global_size", 1);
  return config.getInteger("schema_version") == 1 && launch &&
         dims("local_size", 1) && dims("global_offset", 0);
}
bool validContract(DictionaryAttr config) {
  if (!config)
    return false;
  for (StringRef key : {"export", "request", "intrinsic", "kernel"}) {
    auto attr = config.getAs<StringAttr>(key);
    if (!attr || attr.getValue().empty())
      return false;
  }
  for (StringRef key : {"export_id", "kernel_id"}) {
    auto attr = config.getAs<IntegerAttr>(key);
    if (!attr || !attr.getType().isInteger(32) || attr.getInt() <= 0)
      return false;
  }
  for (StringRef key : {"global_size", "local_size", "global_offset"}) {
    auto attr = config.getAs<DenseI32ArrayAttr>(key);
    int32_t expected = key == "global_offset" ? 0 : 1;
    if (key == "global_size" &&
        config.getAs<StringAttr>("profile") ==
            StringAttr::get(config.getContext(), "gather-f32")) {
      auto words = config.getAs<IntegerAttr>("words");
      auto values = attr ? attr.asArrayRef() : ArrayRef<int32_t>();
      if (!words || words.getInt() < 1 || words.getInt() > 131072 || !attr ||
          values.size() != 3 || values[0] <= 0 || values[1] <= 0 ||
          values[2] != 1 || int64_t(values[0]) * values[1] != words.getInt())
        return false;
      continue;
    }
    if (!attr || attr.size() != 3 ||
        !llvm::all_of(attr.asArrayRef(),
                      [&](int32_t value) { return value == expected; }))
      return false;
  }
  auto scalarKind = config.getAs<StringAttr>("scalar_kind");
  if (scalarKind && scalarKind != StringAttr::get(config.getContext(), "f32") &&
      scalarKind != StringAttr::get(config.getContext(), "i32"))
    return false;
  return true;
}
struct RaiseLLVMBoundaryDevicePass
    : impl::RaiseLLVMBoundaryDevicePassBase<RaiseLLVMBoundaryDevicePass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto config = module->getAttrOfType<DictionaryAttr>("vrc.c_boundary");
    if (module.getSymName() != "device")
      return;
    if (!validContract(config)) {
      fail(module, "invalid explicit ABI contract");
      signalPassFailure();
      return;
    }
    auto kernelName = config.getAs<StringAttr>("kernel").getValue();
    auto source = module.lookupSymbol<LLVM::LLVMFuncOp>(kernelName);
    auto reject = [&](StringRef message) {
      fail(module, message);
      signalPassFailure();
    };
    if (!source || source.isExternal() ||
        source.getCConv() != LLVM::cconv::CConv::SPIR_KERNEL ||
        source.isVarArg() || source.getBody().getBlocks().size() != 1 ||
        source.getNumArguments() != 2 ||
        !isa<LLVM::LLVMVoidType>(source.getFunctionType().getReturnType())) {
      reject("expected a defined two-argument SPIR kernel");
      return;
    }
    auto pointer =
        dyn_cast<LLVM::LLVMPointerType>(source.getArgument(0).getType());
    auto scalarKind = config.getAs<StringAttr>("scalar_kind");
    bool integer = scalarKind && scalarKind.getValue() == "i32";
    Type scalarType = integer ? Type(IntegerType::get(&getContext(), 32))
                              : Type(Float32Type::get(&getContext()));
    if (!pointer || pointer.getAddressSpace() != 1 ||
        source.getArgument(1).getType() != scalarType) {
      reject(integer ? "kernel signature must be (__global int*, int)"
                     : "kernel signature must be (__global float*, float)");
      return;
    }
    for (unsigned index = 0; index < source.getNumArguments(); ++index) {
      for (auto attr : cast<FunctionOpInterface>(source.getOperation())
                           .getArgAttrs(index)) {
        if (attr.getName() == "llvm.noundef")
          continue;
        auto alignment = dyn_cast<IntegerAttr>(attr.getValue());
        if (index == 0 && attr.getName() == "llvm.align" && alignment &&
            alignment.getInt() <= 4)
          continue;
        reject("unproven Device argument attribute");
        return;
      }
    }
    // Metadata disambiguates the pointee type of LLVM's opaque pointer.
    auto frontend = module->getAttrOfType<StringAttr>("vrc.frontend");
    auto json =
        frontend
            ? llvm::json::parse(frontend.getValue())
            : llvm::Expected<llvm::json::Value>(llvm::json::Value(nullptr));
    if (!json) {
      llvm::consumeError(json.takeError());
      reject("missing kernel metadata");
      return;
    }
    auto *doc = json->getAsObject();
    auto *functions = doc ? doc->getArray("functions") : nullptr;
    bool metadataOK = false;
    if (functions)
      for (const auto &item : *functions) {
        auto *function = item.getAsObject();
        if (!function || function->getString("name") != kernelName)
          continue;
        auto *meta = function->getObject("metadata");
        auto *types = meta ? meta->getArray("kernel_arg_type") : nullptr;
        auto *spaces = meta ? meta->getArray("kernel_arg_addr_space") : nullptr;
        metadataOK =
            function->getBoolean("is_kernel") == true && types && spaces &&
            types->size() == 2 && spaces->size() == 2 &&
            (*types)[0].getAsString() == (integer ? "int*" : "float*") &&
            (*types)[1].getAsString() == (integer ? "int" : "float") &&
            (*spaces)[0].getAsInteger() == 1 &&
            (*spaces)[1].getAsInteger() == 0;
        if (meta && meta->get("reqd_work_group_size")) {
          auto *required = meta->getArray("reqd_work_group_size");
          metadataOK &= required && required->size() == 3 &&
                        llvm::all_of(*required, [](const auto &v) {
                          return v.getAsInteger() == 1;
                        });
        }
      }
    if (!metadataOK) {
      reject("OpenCL pointee/launch metadata does not match the boundary ABI");
      return;
    }
    // Preflight the complete body. The sole pointer use must be the complete
    // output word store. There are no reads, aliases, offsets or barriers.
    LLVM::StoreOp output;
    for (auto &op : source.getBody().front()) {
      if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
        if (output || store.getAddr() != source.getArgument(0) ||
            store.getValue().getType() != scalarType || store.getVolatile_() ||
            store.getOrdering() != LLVM::AtomicOrdering::not_atomic ||
            store.getAlignment().value_or(4) > 4) {
          reject("requires one non-atomic complete scalar store to output[0]");
          return;
        }
        output = store;
      } else if (!isa<LLVM::ConstantOp, LLVM::FAddOp, LLVM::FSubOp,
                      LLVM::FMulOp, LLVM::AddOp, LLVM::SubOp, LLVM::MulOp,
                      LLVM::ReturnOp>(op)) {
        reject("unsupported kernel operation; pointer offsets, reads and "
               "barriers need further proofs");
        return;
      }
    }
    if (!output || !source.getArgument(0).hasOneUse()) {
      reject("output pointer must have exactly one complete store use");
      return;
    }
    // Additional non-kernel functions are discarded only if unreferenced. Clang
    // 23 emits an unused __clang_ocl_kern_imp_* implementation alongside
    // kernels.
    SmallVector<Operation *> dead;
    for (auto &op : module.getBody()->getOperations()) {
      if (&op == source)
        continue;
      auto function = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!function || function.getCConv() == LLVM::cconv::CConv::SPIR_KERNEL ||
          !SymbolTable::symbolKnownUseEmpty(function, module)) {
        reject("unexpected live Device symbol or module operation");
        return;
      }
      dead.push_back(&op);
    }
    // Reject unknown target/FP behavior before changing the signature. Keep
    // the source attributes as provenance for the recovered function.
    auto attrs = source->getAttrs();
    for (auto attr : attrs) {
      auto name = attr.getName().getValue();
      if (name == "passthrough") {
        for (auto value : cast<ArrayAttr>(attr.getValue())) {
          if (auto str = dyn_cast<StringAttr>(value)) {
            if (str == "norecurse" || str == "mustprogress" ||
                str == "uniform-work-group-size")
              continue;
          }
          auto pair = dyn_cast<ArrayAttr>(value);
          if (pair && pair.size() == 2 && isa<StringAttr>(pair[0]) &&
              isa<StringAttr>(pair[1])) {
            auto key = cast<StringAttr>(pair[0]).getValue(),
                 val = cast<StringAttr>(pair[1]).getValue();
            if ((key == "no-trapping-math" && val == "true") ||
                key == "stack-protector-buffer-size")
              continue;
          }
          reject("unproven Device target/passthrough attribute");
          return;
        }
      } else if (!llvm::is_contained(
                     ArrayRef<StringRef>{
                         "CConv", "function_type", "linkage", "sym_name",
                         "visibility_", "sym_visibility", "arg_attrs",
                         "res_attrs", "dso_local", "convergent", "no_inline",
                         "always_inline", "no_unwind", "will_return",
                         "memory_effects", "unnamed_addr", "frame_pointer"},
                     name)) {
        source.emitError()
            << "C boundary recovery: unproven Device function attribute "
            << name;
        signalPassFailure();
        return;
      }
    }
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPoint(source);
    auto type = rewriter.getFunctionType(
        {MemRefType::get({1}, scalarType), scalarType}, {});
    auto recovered =
        func::FuncOp::create(rewriter, source.getLoc(), kernelName, type);
    recovered->setAttr("vrc.llvm_function", rewriter.getDictionaryAttr(attrs));
    auto *block = recovered.addEntryBlock();
    rewriter.setInsertionPointToEnd(block);
    IRMapping mapping;
    mapping.map(source.getArgument(1), block->getArgument(1));
    auto index = arith::ConstantIndexOp::create(rewriter, source.getLoc(), 0);
    for (auto &op : source.getBody().front()) {
      if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
        memref::StoreOp::create(rewriter, store.getLoc(),
                                mapping.lookup(store.getValue()),
                                block->getArgument(0), ValueRange{index});
      } else if (isa<LLVM::ReturnOp>(op)) {
        func::ReturnOp::create(rewriter, op.getLoc());
      } else
        rewriter.clone(op, mapping);
    }
    rewriter.eraseOp(source);
    for (auto *op : dead)
      rewriter.eraseOp(op);
  }
};
struct PrepareNativeDevicePass
    : impl::PrepareNativeDevicePassBase<PrepareNativeDevicePass> {
  void runOnOperation() override {
    auto root = getOperation();
    auto device = root.lookupSymbol<ModuleOp>("device");
    auto config = root->getAttrOfType<DictionaryAttr>("vrc.c_boundary");
    if (!device || !validContract(config)) {
      fail(root, "missing native Device contract");
      signalPassFailure();
      return;
    }
    auto kernel = device.lookupSymbol<func::FuncOp>(
        config.getAs<StringAttr>("kernel").getValue());
    if (!kernel || kernel.isExternal() ||
        kernel.getBody().getBlocks().size() != 1 ||
        std::distance(device.getBody()->begin(), device.getBody()->end()) !=
            1) {
      fail(root, "native Device bundle requires one recovered kernel");
      signalPassFailure();
      return;
    }
    OpBuilder builder(root.getBodyRegion());
    auto gpuModule =
        gpu::GPUModuleOp::create(builder, kernel.getLoc(), "device_gpu");
    builder.setInsertionPointToStart(gpuModule.getBody());
    auto gpuKernel = gpu::GPUFuncOp::create(
        builder, kernel.getLoc(), kernel.getName(), kernel.getFunctionType());
    gpuKernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       builder.getUnitAttr());
    gpuKernel->setAttr("vrc.kernel_id", config.get("kernel_id"));
    gpuKernel->setAttr("known_block_size",
                       builder.getDenseI32ArrayAttr({1, 1, 1}));
    gpuKernel->setAttr("known_grid_size", config.get("global_size"));
    if (auto profile = config.getAs<StringAttr>("profile")) {
      gpuKernel->setAttr("vrc.profile", profile);
      gpuKernel->setAttr("vrc.words", config.get("words"));
      if (auto bounds = config.get("argument_bounds"))
        gpuKernel->setAttr("vrc.argument_bounds", bounds);
      if (kernel->hasAttr("vrc.gather_proven"))
        gpuKernel->setAttr("vrc.gather_proven", builder.getUnitAttr());
    }
    gpuKernel.getBody().takeBody(kernel.getBody());
    auto ret =
        cast<func::ReturnOp>(gpuKernel.getBody().front().getTerminator());
    builder.setInsertionPoint(ret);
    gpu::ReturnOp::create(builder, ret.getLoc());
    ret.erase();
    root->setAttr("vrc.device_frontend", device->getAttr("vrc.frontend"));
    device.erase();
  }
};
struct LinkCBoundaryPass : impl::LinkCBoundaryPassBase<LinkCBoundaryPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto config = module->getAttrOfType<DictionaryAttr>("vrc.c_boundary");
    auto host = module.lookupSymbol<ModuleOp>("host"),
         device = module.lookupSymbol<ModuleOp>("device");
    auto reject = [&](StringRef message) {
      fail(module, message);
      signalPassFailure();
    };
    if (!validContract(config) || !host || !device) {
      reject("expected imported host/device and explicit ABI contract");
      return;
    }
    auto name = [&](StringRef key) {
      return config.getAs<StringAttr>(key).getValue();
    };
    auto exported = host.lookupSymbol<func::FuncOp>(name("export"));
    auto request = host.lookupSymbol<func::FuncOp>(name("request"));
    auto intrinsic = host.lookupSymbol<func::FuncOp>(name("intrinsic"));
    auto kernel = device.lookupSymbol<func::FuncOp>(name("kernel"));
    if (!exported || !request || !intrinsic || !kernel || exported == request ||
        exported == intrinsic || request == intrinsic) {
      reject("configured C symbols must exist and have distinct ABI roles");
      return;
    }
    auto u32sig = [&](func::FuncOp f) {
      return f.getNumArguments() == 2 &&
             f.getFunctionType().getNumResults() == 1 &&
             llvm::all_of(f.getArgumentTypes(),
                          [](Type t) { return t.isInteger(32); }) &&
             f.getResultTypes()[0].isInteger(32);
    };
    if (request.isExternal() || request.getBody().getBlocks().size() != 1 ||
        !intrinsic.isExternal() || !u32sig(request) || !u32sig(intrinsic)) {
      reject("internal request/intrinsic ABI must be (uint32,uint32)->uint32");
      return;
    }
    func::CallOp dispatch;
    func::ReturnOp result;
    for (auto &op : request.getBody().front()) {
      if (auto call = dyn_cast<func::CallOp>(op)) {
        if (dispatch || call.getCallee() != intrinsic.getName()) {
          reject("request must directly forward to the configured intrinsic");
          return;
        }
        dispatch = call;
      } else if (auto ret = dyn_cast<func::ReturnOp>(op))
        result = ret;
      else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        if (!constant->use_empty()) {
          reject("request must forward opaque handles without arithmetic");
          return;
        }
      } else {
        reject("request must forward opaque handles without arithmetic");
        return;
      }
    }
    if (!dispatch || !result || dispatch.getNumOperands() != 2 ||
        dispatch.getOperand(0) != request.getArgument(1) ||
        dispatch.getOperand(1) != request.getArgument(0) ||
        result.getNumOperands() != 1 ||
        result.getOperand(0) != dispatch.getResult(0)) {
      reject("request descriptor/resource/result forwarding order is invalid");
      return;
    }
    // No other C caller may reinterpret these handles as integers.
    for (auto function : host.getOps<func::FuncOp>())
      for (auto call : function.getOps<func::CallOp>())
        if (call.getCallee() == request.getName() ||
            (call.getCallee() == intrinsic.getName() && call != dispatch)) {
          reject("additional handle callers require general Host ABI recovery");
          return;
        }
    for (auto &op : device.getBody()->getOperations())
      if (&op != kernel) {
        reject("Device recovery must produce one kernel");
        return;
      }
    if (kernel.isExternal() || kernel.getBody().getBlocks().size() != 1 ||
        kernel.getArgumentTypes() !=
            ArrayRef<Type>{
                MemRefType::get({1}, Float32Type::get(&getContext())),
                Float32Type::get(&getContext())} ||
        !kernel.getResultTypes().empty()) {
      reject("recovered Device signature is invalid");
      return;
    }
    OpBuilder builder(&getContext());
    auto buffer = runtime::BufferType::get(&getContext());
    auto desc = runtime::DispatchDescType::get(&getContext());
    auto event = runtime::EventType::get(&getContext());
    request.setFunctionType(builder.getFunctionType({buffer, desc}, {event}));
    request.getArgument(0).setType(buffer);
    request.getArgument(1).setType(desc);
    request.setPrivate();
    intrinsic.setFunctionType(builder.getFunctionType({desc, buffer}, {event}));
    dispatch.getResult(0).setType(event);
    intrinsic->setAttr("vrc.intrinsic", builder.getStringAttr("dispatch"));
    intrinsic->setAttr("vrc.kernel_id", config.get("kernel_id"));
    intrinsic->setAttr("vrc.argument_kinds",
                       builder.getArrayAttr({builder.getStringAttr("buffer"),
                                             builder.getStringAttr("f32")}));
    intrinsic->setAttr("vrc.resource_accesses",
                       builder.getArrayAttr({builder.getStringAttr("write")}));
    for (StringRef key : {"global_size", "local_size", "global_offset"})
      intrinsic->setAttr(("vrc." + key).str(), config.get(key));
    exported->setAttr("vrc.export_id", config.get("export_id"));
    builder.setInsertionPointToEnd(module.getBody());
    auto gpuModule =
        gpu::GPUModuleOp::create(builder, kernel.getLoc(), "device");
    builder.setInsertionPointToStart(gpuModule.getBody());
    auto gpuKernel = gpu::GPUFuncOp::create(
        builder, kernel.getLoc(), kernel.getName(), kernel.getFunctionType());
    gpuKernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       builder.getUnitAttr());
    gpuKernel->setAttr("vrc.kernel_id", config.get("kernel_id"));
    gpuKernel->setAttr("known_block_size",
                       builder.getDenseI32ArrayAttr({1, 1, 1}));
    gpuKernel->setAttr("known_grid_size", config.get("global_size"));
    if (auto profile = config.getAs<StringAttr>("profile")) {
      gpuKernel->setAttr("vrc.profile", profile);
      gpuKernel->setAttr("vrc.words", config.get("words"));
      if (auto bounds = config.get("argument_bounds"))
        gpuKernel->setAttr("vrc.argument_bounds", bounds);
    }
    gpuKernel.getBody().takeBody(kernel.getBody());
    auto ret =
        cast<func::ReturnOp>(gpuKernel.getBody().front().getTerminator());
    builder.setInsertionPoint(ret);
    gpu::ReturnOp::create(builder, ret.getLoc());
    ret.erase();
    for (auto &op : llvm::make_early_inc_range(host.getBody()->getOperations()))
      op.moveBefore(gpuModule);
    module->setAttr("vrc.host_frontend", host->getAttr("vrc.frontend"));
    module->setAttr("vrc.device_frontend", device->getAttr("vrc.frontend"));
    host.erase();
    device.erase();
  }
};
} // namespace
OwningOpRef<ModuleOp> importCBoundary(llvm::StringRef hostFile,
                                      llvm::StringRef deviceFile,
                                      const llvm::json::Object &config,
                                      MLIRContext &context) {
  context.loadDialect<func::FuncDialect, arith::ArithDialect,
                      memref::MemRefDialect>();
  OwningOpRef<ModuleOp> owner(ModuleOp::create(UnknownLoc::get(&context)));
  auto root = *owner;
  OpBuilder builder(&context);
  if (!profile(config)) {
    fail(root, "unsupported config version or launch profile");
    return {};
  }
  NamedAttrList attrs;
  for (StringRef key : {"export", "request", "intrinsic", "kernel"}) {
    auto value = config.getString(key);
    if (!value && hostFile.empty() && key != "kernel")
      value = "unused_native_device";
    if (!value || value->empty()) {
      fail(root, "missing configured C symbol");
      return {};
    }
    attrs.append(key, builder.getStringAttr(*value));
  }
  for (StringRef key : {"export_id", "kernel_id"}) {
    auto value = config.getInteger(key);
    if (!value && hostFile.empty() && key == "export_id")
      value = 1;
    if (!value || *value <= 0 || *value > INT32_MAX) {
      fail(root, "invalid configured ABI id");
      return {};
    }
    attrs.append(key, builder.getI32IntegerAttr(*value));
  }
  for (StringRef key : {"global_size", "local_size", "global_offset"}) {
    SmallVector<int32_t> values;
    for (const auto &v : *config.getArray(key))
      values.push_back(*v.getAsInteger());
    attrs.append(key, builder.getDenseI32ArrayAttr(values));
  }
  if (auto *kinds = config.getArray("argument_kinds");
      kinds && config.getString("profile") != "gather-f32") {
    auto scalar = kinds->size() == 2 ? (*kinds)[1].getAsString()
                                     : std::optional<StringRef>();
    if (!scalar || (*scalar != "f32" && *scalar != "i32")) {
      fail(root, "unsupported configured scalar argument kind");
      return {};
    }
    attrs.append("scalar_kind", builder.getStringAttr(*scalar));
  }
  if (config.getString("profile") == "gather-f32") {
    attrs.append("profile", builder.getStringAttr("gather-f32"));
    attrs.append("words",
                 builder.getI32IntegerAttr(*config.getInteger("words")));
    if (auto *inputWords = config.getArray("input_words")) {
      SmallVector<int32_t> values;
      for (const llvm::json::Value &value : *inputWords) {
        auto words = value.getAsInteger();
        if (!words || *words <= 0 || *words > INT32_MAX) {
          fail(root, "invalid gather input_words capacity");
          return {};
        }
        values.push_back(static_cast<int32_t>(*words));
      }
      attrs.append("input_words", builder.getDenseI32ArrayAttr(values));
    }
    if (auto *argumentBounds = config.getArray("argument_bounds")) {
      SmallVector<Attribute> values;
      llvm::SmallDenseSet<int64_t> indices;
      for (const llvm::json::Value &value : *argumentBounds) {
        const auto *entry = value.getAsObject();
        auto index = entry ? entry->getInteger("index") : std::nullopt;
        auto minimum = entry ? entry->getInteger("min") : std::nullopt;
        auto maximum = entry ? entry->getInteger("max") : std::nullopt;
        if (!index || !minimum || !maximum || *index < 0 ||
            *index > INT32_MAX || *minimum < INT32_MIN ||
            *maximum > INT32_MAX || *minimum > *maximum ||
            !indices.insert(*index).second) {
          fail(root, "invalid gather argument_bounds entry");
          return {};
        }
        values.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("index", builder.getI32IntegerAttr(*index)),
            builder.getNamedAttr("min", builder.getI32IntegerAttr(*minimum)),
            builder.getNamedAttr("max", builder.getI32IntegerAttr(*maximum)),
        }));
      }
      attrs.append("argument_bounds", builder.getArrayAttr(values));
    }
  }
  root->setAttr("vrc.c_boundary", attrs.getDictionary(&context));
  for (auto entry : {std::make_pair(hostFile, StringRef("host")),
                     std::make_pair(deviceFile, StringRef("device"))}) {
    if (entry.first.empty())
      continue;
    llvm::LLVMContext llvmContext;
    llvm::SMDiagnostic diagnostic;
    auto input = llvm::parseIRFile(entry.first, diagnostic, llvmContext);
    if (!input) {
      diagnostic.print("vrccl", llvm::errs());
      return {};
    }
    if (input->getTargetTriple().str() != "spir-unknown-unknown" ||
        !input->getDataLayout().isLittleEndian() ||
        input->getDataLayout().getPointerSizeInBits() != 32) {
      fail(root, "example requires 32-bit little-endian SPIR input; this is "
                 "not the final Host TargetInfo");
      return {};
    }
    if (entry.second == "device" &&
        config.getString("profile") == "gather-f32") {
      const auto &layout = input->getDataLayout();
      if (layout.getPointerSizeInBits(1) != 32 ||
          layout.getIndexSizeInBits(1) != 32 ||
          layout.getTypeAllocSize(llvm::Type::getFloatTy(llvmContext))
                  .getFixedValue() != 4) {
        fail(root, "gather requires four-byte f32 and 32-bit global "
                   "pointer/index layout");
        return {};
      }
    }
    // Clang lowers source annotations to this appending metadata global.  The
    // Unity-library frontend consumes udoncl.export before import; the older
    // paired Host/Device boundary has an explicit export contract and must
    // discard the same metadata rather than recover it as program memory.
    if (auto *annotations = input->getNamedGlobal("llvm.global.annotations"))
      annotations->eraseFromParent();
    for (llvm::GlobalVariable &global :
         llvm::make_early_inc_range(input->globals())) {
      if (global.getSection() != "llvm.metadata")
        continue;
      global.removeDeadConstantUsers();
      if (global.use_empty())
        global.eraseFromParent();
    }
    auto imported = importLLVM(std::move(input), context);
    if (!imported)
      return {};
    imported->setSymName(entry.second);
    imported->getOperation()->setAttr("vrc.c_boundary",
                                      attrs.getDictionary(&context));
    root.getBody()->push_back(imported.release());
  }
  return owner;
}
void buildCBoundaryRecoveryPipeline(OpPassManager &manager) {
  manager.addNestedPass<ModuleOp>(createMem2Reg());
  manager.addNestedPass<ModuleOp>(createRaiseLLVMBoundaryDevicePass());
  RaiseLLVMMemoryPassOptions memoryOptions;
  memoryOptions.allowRawFallback = true;
  manager.addNestedPass<ModuleOp>(createRaiseLLVMMemoryPass(memoryOptions));
  manager.addNestedPass<ModuleOp>(createRaiseLLVMRawMemoryPass());
  manager.addNestedPass<ModuleOp>(createRaiseLLVMToScalarPass());
  manager.addNestedPass<ModuleOp>(createCanonicalizerPass());
  manager.addPass(createLinkCBoundaryPass());
}
void registerCBoundaryRecoveryPipeline() {
  PassPipelineRegistration<>(
      "vrc-recover-c-boundary",
      "Recover and link the restricted C/OpenCL boundary inputs",
      buildCBoundaryRecoveryPipeline);
}
} // namespace vrc
