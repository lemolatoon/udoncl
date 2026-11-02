#include "VRC/ABI/Manifest/Boundary.h"
#include "VRC/ABI/Manifest/UdonSharpABI.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SHA256.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_VERIFYHOSTDEVICEBOUNDARYPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace runtime = ::vrc::vrc;
namespace fragment = ::vrc::vrcgpu;
std::string kind(Type type) {
  if (type.isInteger(32))
    return "i32";
  if (type.isF32())
    return "f32";
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref || memref.getRank() != 1 || !memref.getLayout().isIdentity())
    return "unsupported";
  if (memref.getElementType().isInteger(32))
    return "i32_array";
  if (memref.getElementType().isF32())
    return "f32_array";
  return "unsupported";
}
UdonSharpMethodABI methodABI(func::FuncOp function,
                             UdonSharpABIBuilder &abiBuilder) {
  auto publicName = function->getAttrOfType<StringAttr>("vrc.public_name");
  SmallVector<std::string> ownedNames;
  SmallVector<StringRef> parameterNames;
  if (auto names = function->getAttrOfType<ArrayAttr>("vrc.parameter_names")) {
    for (Attribute name : names)
      parameterNames.push_back(cast<StringAttr>(name).getValue());
  } else {
    ownedNames.reserve(function.getNumArguments());
    parameterNames.reserve(function.getNumArguments());
    for (unsigned index = 0; index < function.getNumArguments(); ++index)
      ownedNames.push_back(defaultUdonSharpParameterName(index));
    for (const std::string &name : ownedNames)
      parameterNames.push_back(name);
  }
  return abiBuilder.buildMethod(publicName ? publicName.getValue()
                                           : function.getSymName(),
                                parameterNames, function.getNumResults() != 0);
}
SmallVector<std::string> parameterNames(func::FuncOp function) {
  SmallVector<std::string> result;
  result.reserve(function.getNumArguments());
  if (auto names = function->getAttrOfType<ArrayAttr>("vrc.parameter_names")) {
    for (Attribute name : names)
      result.push_back(cast<StringAttr>(name).getValue().str());
  } else {
    for (unsigned index = 0; index < function.getNumArguments(); ++index)
      result.push_back(defaultUdonSharpParameterName(index));
  }
  return result;
}
llvm::json::Object typeSchema(Type type) {
  if (type.isInteger(32))
    return llvm::json::Object{{"category", "signed-integer"}, {"bits", 32}};
  if (type.isF32())
    return llvm::json::Object{{"category", "float"}, {"bits", 32}};
  auto memref = dyn_cast<MemRefType>(type);
  if (memref && memref.getRank() == 1 && memref.getLayout().isIdentity() &&
      (memref.getElementType().isInteger(32) ||
       memref.getElementType().isF32()))
    return llvm::json::Object{
        {"category", "array"},
        {"element_category",
         memref.getElementType().isInteger(32) ? "signed-integer" : "float"},
        {"element_bits", 32},
        {"rank", 1},
        {"elements", memref.hasStaticShape() ? memref.getNumElements() : -1}};
  return llvm::json::Object{{"category", "unsupported"}};
}
llvm::json::Array dimensions(ArrayRef<int32_t> values) {
  llvm::json::Array result;
  for (auto value : values)
    result.push_back(value);
  return result;
}
} // namespace
LogicalResult verifyHostDeviceBoundary(ModuleOp module) {
  auto abi = module->getAttrOfType<IntegerAttr>("vrc.abi_version");
  if (!abi || abi.getInt() != 1)
    return module.emitError("unsupported Host/Device ABI version");
  auto host = module->getAttrOfType<StringAttr>("vrc.host_stage");
  auto device = module->getAttrOfType<StringAttr>("vrc.device_stage");
  auto intrinsics = module->getAttrOfType<StringAttr>("vrc.intrinsics_stage");
  if (!host || host != "udonhl" || !intrinsics || intrinsics != "vrc" ||
      !device || device != "vrcgpu-single-invocation")
    return module.emitError(
        "boundary verification requires lowered Host and Device stages");
  llvm::DenseMap<int32_t, fragment::KernelOp> kernels;
  for (auto kernel : module.getOps<fragment::KernelOp>())
    if (!kernels.try_emplace(kernel.getKernelId(), kernel).second)
      return kernel.emitError("duplicate kernel id at Host/Device boundary");
  if (kernels.empty())
    return module.emitError("boundary requires at least one Device kernel");
  llvm::SmallDenseSet<int64_t> ids;
  llvm::StringSet<> events;
  llvm::StringSet<> publicNames;
  SmallVector<func::FuncOp> exportedFunctions;
  for (auto function : module.getOps<func::FuncOp>()) {
    if (function->hasAttr("vrc.export_id")) {
      if (!function->getAttrOfType<IntegerAttr>("vrc.export_id"))
        return function.emitError("Host export id must be an integer");
      if (Attribute value = function->getAttr("vrc.public_name")) {
        auto name = dyn_cast<StringAttr>(value);
        if (!name || !isUdonSharpIdentifier(name.getValue()))
          return function.emitError("invalid UdonSharp public export name");
      }
      if (Attribute value = function->getAttr("vrc.parameter_names")) {
        auto names = dyn_cast<ArrayAttr>(value);
        if (!names || names.size() != function.getNumArguments() ||
            !llvm::all_of(names, [](Attribute name) {
              auto string = dyn_cast<StringAttr>(name);
              return string && isUdonSharpIdentifier(string.getValue());
            }))
          return function.emitError("invalid UdonSharp parameter names");
      }
      exportedFunctions.push_back(function);
    }
  }
  llvm::sort(exportedFunctions, [](func::FuncOp lhs, func::FuncOp rhs) {
    return lhs->getAttrOfType<IntegerAttr>("vrc.export_id").getInt() <
           rhs->getAttrOfType<IntegerAttr>("vrc.export_id").getInt();
  });
  llvm::DenseMap<Operation *, UdonSharpMethodABI> methodABIs;
  UdonSharpABIBuilder abiBuilder;
  for (func::FuncOp function : exportedFunctions)
    methodABIs.try_emplace(function.getOperation(),
                           methodABI(function, abiBuilder));
  bool exported = false;
  for (auto function : module.getOps<func::FuncOp>()) {
    auto effect = function->getAttrOfType<StringAttr>("vrc.async_effect");
    if (function.isExternal() || function->hasAttr("vrc.intrinsic") ||
        (effect && effect != "none"))
      return function.emitError(
          "unresolved intrinsic or continuation lowering at the boundary");
    for (auto &op : function.getBody().front())
      if (isa<func::CallOp>(op))
        return op.emitError(
            "Host calls must pass through UdonHL before boundary verification");
    if (auto id = function->getAttrOfType<IntegerAttr>("vrc.export_id")) {
      const auto &abi = methodABIs.at(function.getOperation());
      if (id.getInt() <= 0 || !ids.insert(id.getInt()).second ||
          function.isPrivate() || !isUdonSharpIdentifier(abi.publicName) ||
          !publicNames.insert(abi.publicName).second ||
          function.getFunctionType().getNumResults() > 1 ||
          !llvm::all_of(function.getArgumentTypes(),
                        [](Type t) { return kind(t) != "unsupported"; }) ||
          !llvm::all_of(function.getResultTypes(),
                        [](Type t) { return kind(t) != "unsupported"; }))
        return function.emitError("invalid or duplicate Host export ABI");
      if (!events.insert(abi.eventName).second)
        return function.emitError("generated export event name collision");
      exported = true;
    }
  }
  if (!exported)
    return module.emitError("boundary requires a public Host export");
  llvm::SmallDenseSet<int32_t> referenced;
  auto status = module.walk([&](runtime::DispatchOp dispatch) {
    auto found = kernels.find(dispatch.getKernelId());
    if (found == kernels.end()) {
      dispatch.emitError("Host dispatch references a missing Device kernel id");
      return WalkResult::interrupt();
    }
    auto kernel = found->second;
    size_t bufferArguments =
        llvm::count_if(kernel.getArgumentKinds(), [](Attribute value) {
          auto kind = dyn_cast<StringAttr>(value);
          return kind && kind.getValue() == "buffer";
        });
    if (dispatch.getArgumentKinds() != kernel.getArgumentKinds() ||
        dispatch.getResources().size() != bufferArguments ||
        dispatch.getGlobalSize() != kernel.getGlobalSize() ||
        dispatch.getLocalSize() != kernel.getLocalSize() ||
        dispatch.getGlobalOffset() != kernel.getGlobalOffset()) {
      dispatch.emitError("Host descriptor/resource/NDRange schema does not "
                         "match Device kernel ABI");
      return WalkResult::interrupt();
    }
    auto function = dispatch->getParentOfType<func::FuncOp>();
    if (!function || function->getParentOp() != module) {
      dispatch.emitError("GPU commands must belong to the Host runtime path");
      return WalkResult::interrupt();
    }
    referenced.insert(dispatch.getKernelId());
    return WalkResult::advance();
  });
  if (status.wasInterrupted())
    return failure();
  if (referenced.size() != kernels.size())
    return module.emitError(
        "every initial Device kernel must have a matching Host dispatch");
  return success();
}
llvm::json::Object buildBoundaryManifest(ModuleOp module) {
  llvm::json::Array exports, kernels, dispatches;
  SmallVector<func::FuncOp> exportedFunctions;
  for (auto function : module.getOps<func::FuncOp>())
    if (function->hasAttr("vrc.export_id"))
      exportedFunctions.push_back(function);
  llvm::sort(exportedFunctions, [](func::FuncOp lhs, func::FuncOp rhs) {
    return lhs->getAttrOfType<IntegerAttr>("vrc.export_id").getInt() <
           rhs->getAttrOfType<IntegerAttr>("vrc.export_id").getInt();
  });
  std::string canonicalABI = "udoncl-public-abi-v1\n";
  UdonSharpABIBuilder abiBuilder;
  for (auto function : exportedFunctions) {
    auto id = function->getAttrOfType<IntegerAttr>("vrc.export_id");
    auto abi = methodABI(function, abiBuilder);
    SmallVector<std::string> names = parameterNames(function);
    canonicalABI += std::to_string(id.getInt()) + "|" + abi.publicName + "|";
    llvm::json::Array args;
    for (auto indexed : llvm::enumerate(function.getArgumentTypes())) {
      canonicalABI += kind(indexed.value()) + ",";
      args.push_back(llvm::json::Object{
          {"index", int64_t(indexed.index())},
          {"name", names[indexed.index()]},
          {"kind", kind(indexed.value())},
          {"type_schema", typeSchema(indexed.value())},
          {"heap_slot", abi.parameterSlots[indexed.index()]}});
    }
    canonicalABI += "->";
    llvm::json::Value result = nullptr;
    if (!function.getResultTypes().empty()) {
      canonicalABI += kind(function.getResultTypes()[0]);
      result = llvm::json::Object{
          {"kind", kind(function.getResultTypes()[0])},
          {"type_schema", typeSchema(function.getResultTypes()[0])},
          {"heap_slot", *abi.resultSlot}};
    } else
      canonicalABI += "void";
    canonicalABI += "\n";
    exports.push_back(
        llvm::json::Object{{"id", id.getInt()},
                           {"c_symbol", function.getName().str()},
                           {"public_name", abi.publicName},
                           {"udon_event", abi.eventName},
                           {"parameters", std::move(args)},
                           {"result", std::move(result)},
                           {"status_slot", udonclStatusSlot(id.getInt())},
                           {"async", false}});
  }
  auto publicDigest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(canonicalABI.data()),
      canonicalABI.size()));
  for (auto kernel : module.getOps<fragment::KernelOp>()) {
    std::string shader =
        "kernels/kernel_" + std::to_string(kernel.getKernelId()) + ".shader";
    llvm::json::Array args;
    args.push_back(llvm::json::Object{{"index", 0},
                                      {"kind", "buffer"},
                                      {"access", "write"},
                                      {"resource_index", 0},
                                      {"required_bytes", 4},
                                      {"alignment", 4},
                                      {"binding", "SV_Target"}});
    args.push_back(llvm::json::Object{{"index", 1},
                                      {"kind", "f32"},
                                      {"scalar_payload_offset", 0},
                                      {"byte_size", 4},
                                      {"binding", "_VRC_Arg1"}});
    kernels.push_back(llvm::json::Object{
        {"id", kernel.getKernelId()},
        {"name", kernel.getEntryName().str()},
        {"source_module", kernel.getSourceModule().str()},
        {"symbol", kernel.getSymName().str()},
        {"shader", shader},
        {"args", std::move(args)},
        {"global_size", dimensions(kernel.getGlobalSize())},
        {"local_size", dimensions(kernel.getLocalSize())},
        {"global_offset", dimensions(kernel.getGlobalOffset())},
        {"descriptor",
         llvm::json::Object{
             {"scalar_payload_bytes", 4},
             {"byte_order", "little"},
             {"resource_order", llvm::json::Array{0}},
             {"argument_copy_policy", "required_at_clSetKernelArg"}}},
        {"target_texture", llvm::json::Object{{"width", 1},
                                              {"height", 1},
                                              {"format", "rgba8_unorm"},
                                              {"srgb", false},
                                              {"mipmaps", false},
                                              {"filter", "point"},
                                              {"validated_on_client", false}}},
        {"passes", llvm::json::Array{llvm::json::Object{
                       {"id", 0},
                       {"output_arg", 0},
                       {"reads", llvm::json::Array{}},
                       {"writes", llvm::json::Array{0}}}}},
        {"cost", llvm::json::Object{{"kind", "static"},
                                    {"pass_count", 1},
                                    {"texture_read_bytes", 0},
                                    {"texture_write_bytes", 4},
                                    {"scratch_bytes", 0}}}});
  }
  // Gather kernels retain the same byte-preserving texture representation,
  // with one word per pixel in every read-only input and the output.
  unsigned kernelIndex = 0;
  for (auto kernel : module.getOps<fragment::KernelOp>()) {
    auto &entry = *kernels[kernelIndex++].getAsObject();
    auto profile = kernel->getAttrOfType<StringAttr>("vrc.profile");
    if (!profile || profile != "gather-f32")
      continue;
    int64_t words = kernel->getAttrOfType<IntegerAttr>("vrc.words").getInt();
    auto global = kernel.getGlobalSize();
    llvm::json::Array args;
    llvm::DenseMap<unsigned, fragment::ResourceOp> resources;
    for (auto resource :
         kernel.getBody().front().getOps<fragment::ResourceOp>())
      resources[resource.getArgIndex()] = resource;
    llvm::json::Array resourceOrder, reads, writes;
    std::optional<unsigned> outputArgument;
    unsigned scalarOffset = 0;
    for (unsigned i = 0; i < kernel.getArgumentKinds().size(); ++i) {
      auto resource = resources.lookup(i);
      if (resource && resource.getKind() != "uniform") {
        bool output = resource.getKind() == "target";
        std::string binding = output   ? "SV_Target"
                              : i == 0 ? "_VRC_Input"
                                       : "_VRC_Input" + std::to_string(i);
        args.push_back(llvm::json::Object{{"index", i},
                                          {"kind", "buffer"},
                                          {"access", output ? "write" : "read"},
                                          {"resource_index", i},
                                          {"required_bytes", words * 4},
                                          {"alignment", 4},
                                          {"binding", std::move(binding)}});
        resourceOrder.push_back(i);
        if (output) {
          outputArgument = i;
          writes.push_back(i);
        } else
          reads.push_back(i);
      } else {
        args.push_back(llvm::json::Object{
            {"index", i},
            {"kind",
             cast<StringAttr>(kernel.getArgumentKinds()[i]).getValue().str()},
            {"scalar_payload_offset", scalarOffset},
            {"byte_size", 4},
            {"binding", "_VRC_Arg" + std::to_string(i)}});
        scalarOffset += 4;
      }
    }
    entry["profile"] = "gather-f32";
    entry["args"] = std::move(args);
    entry["descriptor"] = llvm::json::Object{
        {"scalar_payload_bytes", scalarOffset},
        {"byte_order", "little"},
        {"resource_order", std::move(resourceOrder)},
        {"distinct_resources", true},
        {"argument_copy_policy", "required_at_clSetKernelArg"}};
    (*entry.getObject("target_texture"))["width"] = global[0];
    (*entry.getObject("target_texture"))["height"] = global[1];
    entry["passes"] =
        llvm::json::Array{llvm::json::Object{{"id", 0},
                                             {"output_arg", *outputArgument},
                                             {"reads", std::move(reads)},
                                             {"writes", std::move(writes)}}};
    // Dynamic gather counts depend on control flow; do not invent a static
    // read cost from the allocation size.
    entry["cost"] = llvm::json::Object{{"kind", "bounded-dynamic"},
                                       {"pass_count", 1},
                                       {"texture_write_bytes", words * 4},
                                       {"scratch_bytes", 0}};
  }
  module.walk([&](runtime::DispatchOp dispatch) {
    dispatches.push_back(llvm::json::Object{
        {"function", dispatch->getParentOfType<func::FuncOp>().getName().str()},
        {"kernel_id", dispatch.getKernelId()},
        {"descriptor_abi", 1},
        {"resources", static_cast<int64_t>(dispatch.getResources().size())},
        {"completion", "pending_event"}});
  });
  return llvm::json::Object{
      {"schema_version", 1},
      {"compiler_version", "0.1.0-dev"},
      {"runtime_abi_version", 1},
      {"public_abi_fingerprint", llvm::toHex(publicDigest, true)},
      {"physical_export_abi", "udonsharp-worlds-sdk-3.10.5"},
      {"artifact_stage", "boundary-prototype"},
      {"sdk_compatibility", llvm::json::Object{{"sdk_version", "3.10.5"},
                                               {"unity_version", "2022.3.22f1"},
                                               {"extern_db_hash", nullptr},
                                               {"validated", false}}},
      {"target", llvm::json::Object{{"host_pointer_bits", 32},
                                    {"device_pointer_bits", 32},
                                    {"endianness", "little"},
                                    {"platform", "VRChat-PC"}}},
      {"program", llvm::json::Object{{"name", "boundary_probe"},
                                     {"assembly", nullptr},
                                     {"ir", "host-device.mlir"}}},
      {"exports", std::move(exports)},
      {"device", llvm::json::Object{{"kernels", std::move(kernels)}}},
      {"dispatch_sites", std::move(dispatches)},
      {"ready_for_unity", false}};
}
bool validateBoundaryRequest(const llvm::json::Object &manifest,
                             const llvm::json::Object &request,
                             llvm::raw_ostream &error) {
  auto fail = [&](llvm::StringRef text) {
    error << text << "\n";
    return false;
  };
  if (manifest.getInteger("schema_version") != 1 ||
      manifest.getInteger("runtime_abi_version") != 1 ||
      request.getInteger("runtime_abi_version") != 1)
    return fail("unsupported manifest/runtime ABI version");
  if (manifest.getString("artifact_stage") != "boundary-prototype" ||
      manifest.getBoolean("ready_for_unity") != false)
    return fail("unsupported artifact stage/readiness claim");
  auto *target = manifest.getObject("target");
  if (!target || target->getInteger("host_pointer_bits") != 32 ||
      target->getInteger("device_pointer_bits") != 32 ||
      target->getString("endianness") != "little" ||
      target->getString("platform") != "VRChat-PC")
    return fail("unsupported Host/Device address or byte-order ABI");
  auto *device = manifest.getObject("device");
  auto *kernels = device ? device->getArray("kernels") : nullptr;
  auto id = request.getInteger("kernel_id"),
       generation = request.getInteger("generation");
  if (!kernels || !id || *id <= 0 || *id > INT32_MAX || !generation ||
      *generation <= 0 || *generation > UINT32_MAX)
    return fail("invalid kernel or generation id");
  const llvm::json::Object *selected = nullptr;
  llvm::SmallDenseSet<int64_t> ids;
  for (const auto &value : *kernels) {
    auto *kernel = value.getAsObject();
    auto key = kernel ? kernel->getInteger("id") : std::nullopt;
    if (!key || *key <= 0 || *key > INT32_MAX || !ids.insert(*key).second)
      return fail("invalid or duplicate kernel id");
    auto shader = kernel->getString("shader");
    if (!shader ||
        *shader != "kernels/kernel_" + std::to_string(*key) + ".shader")
      return fail("shader path is not the canonical safe relative asset path");
    if (*key == *id)
      selected = kernel;
  }
  if (!selected)
    return fail("request references a missing kernel id");
  auto dimensions = [](const llvm::json::Array *values, int64_t required) {
    return values && values->size() == 3 &&
           llvm::all_of(*values, [&](const llvm::json::Value &v) {
             return v.getAsInteger() == required;
           });
  };
  for (const auto *object : {selected, &request})
    if (!dimensions(object->getArray("global_size"), 1) ||
        !dimensions(object->getArray("local_size"), 1) ||
        !dimensions(object->getArray("global_offset"), 0))
      return fail("descriptor NDRange does not match the single-invocation "
                  "Device profile");
  auto *args = selected->getArray("args");
  auto *descriptor = selected->getObject("descriptor");
  auto *order = descriptor ? descriptor->getArray("resource_order") : nullptr;
  if (!args || args->size() != 2 || !descriptor ||
      descriptor->getInteger("scalar_payload_bytes") != 4 ||
      descriptor->getString("byte_order") != "little" ||
      descriptor->getString("argument_copy_policy") !=
          "required_at_clSetKernelArg" ||
      !order || order->size() != 1 || (*order)[0].getAsInteger() != 0)
    return fail("unsupported Device descriptor schema");
  auto *buffer = (*args)[0].getAsObject(), *scalar = (*args)[1].getAsObject();
  if (!buffer || !scalar || buffer->getInteger("index") != 0 ||
      buffer->getString("kind") != "buffer" ||
      buffer->getString("access") != "write" ||
      buffer->getInteger("resource_index") != 0 ||
      buffer->getInteger("required_bytes") != 4 ||
      buffer->getInteger("alignment") != 4 ||
      buffer->getString("binding") != "SV_Target" ||
      scalar->getString("binding") != "_VRC_Arg1" ||
      scalar->getInteger("index") != 1 || scalar->getString("kind") != "f32" ||
      scalar->getInteger("scalar_payload_offset") != 0 ||
      scalar->getInteger("byte_size") != 4)
    return fail(
        "Device argument kinds/layout do not match the boundary profile");
  auto *resources = request.getArray("resources");
  auto *payload = request.getArray("scalar_payload");
  if (!resources || resources->size() != 1 || !payload || payload->size() != 4)
    return fail("request resource count or scalar payload length mismatch");
  auto *resource = (*resources)[0].getAsObject();
  auto handle = resource ? resource->getInteger("handle") : std::nullopt;
  if (!resource || resource->getString("kind") != "buffer" ||
      resource->getInteger("arg_index") != 0 || !handle || *handle <= 0 ||
      *handle > UINT32_MAX || resource->getInteger("logical_bytes") != 4 ||
      resource->getInteger("generation") != generation)
    return fail("invalid resource handle/kind/order/size or stale generation");
  auto *texture = resource->getObject("texture");
  auto *targetTexture = selected->getObject("target_texture");
  for (const auto *object : {texture, targetTexture})
    if (!object || object->getInteger("width") != 1 ||
        object->getInteger("height") != 1 ||
        object->getString("format") != "rgba8_unorm" ||
        object->getBoolean("srgb") != false ||
        object->getBoolean("mipmaps") != false ||
        object->getString("filter") != "point")
      return fail(
          "texture layout/format/color-space does not match the Device ABI");
  for (const auto &value : *payload) {
    auto byte = value.getAsInteger();
    if (!byte || *byte < 0 || *byte > 255)
      return fail("scalar payload must contain four copied bytes");
  }
  return true;
}

namespace {
struct VerifyHostDeviceBoundaryPass
    : impl::VerifyHostDeviceBoundaryPassBase<VerifyHostDeviceBoundaryPass> {
  void runOnOperation() override {
    if (failed(verifyHostDeviceBoundary(getOperation())))
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
