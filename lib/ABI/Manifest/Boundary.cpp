#include "VRC/ABI/Manifest/Boundary.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  return type.isInteger(32) ? "i32" : type.isF32() ? "f32" : "unsupported";
}
llvm::json::Array dimensions(ArrayRef<int32_t> values) {
  llvm::json::Array result;
  for (auto value : values)
    result.push_back(value);
  return result;
}
std::string eventName(func::FuncOp function) {
  std::string signature = function.getName().str();
  for (auto type : function.getArgumentTypes())
    signature += ":" + kind(type);
  signature += "->";
  for (auto type : function.getResultTypes())
    signature += kind(type);
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(signature.data()), signature.size()));
  return "VRC_Call_" + llvm::toHex(digest, true).substr(0, 16);
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
  bool exported = false;
  for (auto function : module.getOps<func::FuncOp>()) {
    auto effect = function->getAttrOfType<StringAttr>("vrc.async_effect");
    if (function.isExternal() || function->hasAttr("vrc.intrinsic") ||
        (effect && effect != "none"))
      return function.emitError(
          "unresolved intrinsic or async/frame lowering at the boundary");
    for (auto &op : function.getBody().front())
      if (isa<func::CallOp>(op))
        return op.emitError(
            "Host calls must pass through UdonHL before boundary verification");
    if (auto id = function->getAttrOfType<IntegerAttr>("vrc.export_id")) {
      if (id.getInt() <= 0 || !ids.insert(id.getInt()).second ||
          function.isPrivate() ||
          function.getFunctionType().getNumResults() > 1 ||
          !llvm::all_of(function.getArgumentTypes(),
                        [](Type t) { return kind(t) != "unsupported"; }) ||
          !llvm::all_of(function.getResultTypes(),
                        [](Type t) { return kind(t) != "unsupported"; }))
        return function.emitError("invalid or duplicate Host export ABI");
      if (!events.insert(eventName(function)).second)
        return function.emitError("generated export event name collision");
      exported = true;
    }
  }
  if (!exported)
    return module.emitError("boundary requires a scalar Host export");
  llvm::SmallDenseSet<int32_t> referenced;
  auto status = module.walk([&](runtime::DispatchOp dispatch) {
    auto found = kernels.find(dispatch.getKernelId());
    if (found == kernels.end()) {
      dispatch.emitError("Host dispatch references a missing Device kernel id");
      return WalkResult::interrupt();
    }
    auto kernel = found->second;
    if (dispatch.getArgumentKinds() != kernel.getArgumentKinds() ||
        dispatch.getResources().size() != 1 ||
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
  for (auto function : module.getOps<func::FuncOp>()) {
    auto id = function->getAttrOfType<IntegerAttr>("vrc.export_id");
    if (!id)
      continue;
    std::string suffix = std::to_string(id.getInt());
    llvm::json::Array args;
    for (auto indexed : llvm::enumerate(function.getArgumentTypes()))
      args.push_back(llvm::json::Object{
          {"index", int64_t(indexed.index())},
          {"kind", kind(indexed.value())},
          {"heap_slot",
           "VRC_ARG_" + suffix + "_" + std::to_string(indexed.index())}});
    llvm::json::Value result = nullptr;
    if (!function.getResultTypes().empty())
      result = llvm::json::Object{{"kind", kind(function.getResultTypes()[0])},
                                  {"heap_slot", "VRC_RET_" + suffix}};
    exports.push_back(
        llvm::json::Object{{"id", id.getInt()},
                           {"c_symbol", function.getName().str()},
                           {"udon_event", eventName(function)},
                           {"parameters", std::move(args)},
                           {"result", std::move(result)},
                           {"status_slot", "VRC_STATUS_" + suffix},
                           {"async", false}});
  }
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
  // with one input word and one output word per pixel.
  unsigned kernelIndex = 0;
  for (auto kernel : module.getOps<fragment::KernelOp>()) {
    auto &entry = *kernels[kernelIndex++].getAsObject();
    auto profile = kernel->getAttrOfType<StringAttr>("vrc.profile");
    if (!profile || profile != "gather-f32")
      continue;
    int64_t words = kernel->getAttrOfType<IntegerAttr>("vrc.words").getInt();
    llvm::json::Array args;
    for (unsigned i = 0; i < 2; ++i)
      args.push_back(
          llvm::json::Object{{"index", i},
                             {"kind", "buffer"},
                             {"access", i == 0 ? "read" : "write"},
                             {"resource_index", i},
                             {"required_bytes", words * 4},
                             {"alignment", 4},
                             {"binding", i == 0 ? "_VRC_Input" : "SV_Target"}});
    for (unsigned i = 2; i < kernel.getArgumentKinds().size(); ++i)
      args.push_back(llvm::json::Object{
          {"index", i},
          {"kind",
           cast<StringAttr>(kernel.getArgumentKinds()[i]).getValue().str()},
          {"scalar_payload_offset", (i - 2) * 4},
          {"byte_size", 4},
          {"binding", "_VRC_Arg" + std::to_string(i)}});
    entry["profile"] = "gather-f32";
    entry["args"] = std::move(args);
    entry["descriptor"] = llvm::json::Object{
        {"scalar_payload_bytes", (kernel.getArgumentKinds().size() - 2) * 4},
        {"byte_order", "little"},
        {"resource_order", llvm::json::Array{0, 1}},
        {"distinct_resources", true},
        {"argument_copy_policy", "required_at_clSetKernelArg"}};
    (*entry.getObject("target_texture"))["width"] = words;
    entry["passes"] =
        llvm::json::Array{llvm::json::Object{{"id", 0},
                                             {"output_arg", 1},
                                             {"reads", llvm::json::Array{0}},
                                             {"writes", llvm::json::Array{1}}}};
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
        {"resources", 1},
        {"completion", "pending_event"}});
  });
  return llvm::json::Object{
      {"schema_version", 1},
      {"compiler_version", "0.1.0-dev"},
      {"runtime_abi_version", 1},
      {"artifact_stage", "boundary-prototype"},
      {"sdk_compatibility", llvm::json::Object{{"sdk_version", nullptr},
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
