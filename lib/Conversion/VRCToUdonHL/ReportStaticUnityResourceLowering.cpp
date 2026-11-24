#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/Twine.h"
#include <optional>
#include <set>
#include <string>
#include <utility>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_REPORTSTATICUNITYRESOURCELOWERINGPASS
#define GEN_PASS_DEF_REQUIRESTATICUNITYRESOURCELOWERINGPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral readinessAttribute =
    "vrc.static_unity_resource_lowering_readiness";
constexpr StringLiteral layoutAttribute = "vrc.static_unity_buffer_layout_plan";
constexpr StringLiteral reconciliationAttribute =
    "vrc.static_unity_buffer_reconciliation";
constexpr StringLiteral aliasesAttribute = "vrc.static_public_resource_aliases";
constexpr StringLiteral inventoryAttribute =
    "vrc.static_device_resource_inventory";

int64_t integer(DictionaryAttr value, StringRef name) {
  auto attribute = value ? value.getAs<IntegerAttr>(name) : IntegerAttr();
  return attribute ? attribute.getInt() : 0;
}

std::optional<bool> boolean(DictionaryAttr value, StringRef name) {
  auto attribute = value ? value.getAs<BoolAttr>(name) : BoolAttr();
  if (!attribute)
    return std::nullopt;
  return attribute.getValue();
}

std::string endpoint(DictionaryAttr entry) {
  auto access = entry.getAs<StringAttr>("public_access");
  auto index = entry.getAs<IntegerAttr>("public_index");
  std::string text = access ? access.getValue().str() : "private";
  if (index)
    text += " " + std::to_string(index.getInt());
  return text;
}

/// Calls `blocker` once per reason the static Unity resource lowering of
/// `module` is not applied. It reads only the summaries the static resource
/// passes left on the module.
void forEachBlocker(ModuleOp module, DictionaryAttr readiness,
                    function_ref<void(const Twine &)> blocker) {
  auto layout = module->getAttrOfType<DictionaryAttr>(layoutAttribute);
  auto inventory = module->getAttrOfType<DictionaryAttr>(inventoryAttribute);
  auto aliases = module->getAttrOfType<DictionaryAttr>(aliasesAttribute);
  auto reconciliation =
      module->getAttrOfType<DictionaryAttr>(reconciliationAttribute);

  // Public allocations of the inventory without a public alias.
  std::set<int64_t> aliased;
  auto aliasEntries =
      aliases ? aliases.getAs<ArrayAttr>("aliases") : ArrayAttr();
  if (aliasEntries)
    for (Attribute attribute : aliasEntries)
      if (auto alias = dyn_cast<DictionaryAttr>(attribute))
        if (auto id = alias.getAs<IntegerAttr>("resource_id"))
          aliased.insert(id.getInt());
  if (auto resources =
          inventory ? inventory.getAs<ArrayAttr>("resources") : ArrayAttr())
    for (Attribute attribute : resources) {
      auto resource = dyn_cast<DictionaryAttr>(attribute);
      auto id = resource ? resource.getAs<IntegerAttr>("resource_id") : nullptr;
      if (!id || !boolean(resource, "public").value_or(false) ||
          aliased.count(id.getInt()))
        continue;
      auto name = resource.getAs<StringAttr>("name");
      blocker("public resource " + Twine(id.getInt()) + " (" +
              (name ? name.getValue() : StringRef("unnamed")) +
              ") has no public alias");
    }

  // Configured public buffers no static public resource resolves, and
  // static public resources no configured buffer uses.
  std::set<int64_t> resolved;
  if (auto buffers = reconciliation ? reconciliation.getAs<ArrayAttr>("buffers")
                                    : ArrayAttr())
    for (Attribute attribute : buffers) {
      auto buffer = dyn_cast<DictionaryAttr>(attribute);
      auto resolution =
          buffer ? buffer.getAs<StringAttr>("resolution") : nullptr;
      if (!resolution)
        continue;
      if (resolution == "static_alias") {
        if (auto id = buffer.getAs<IntegerAttr>("resource_id"))
          resolved.insert(id.getInt());
      } else if (resolution == "legacy_public") {
        blocker("configured public buffer " +
                Twine(integer(buffer, "buffer_id")) + " (" + endpoint(buffer) +
                ") resolves to no static public resource");
      }
    }
  std::set<int64_t> declared;
  if (auto attribute = module->getAttrOfType<DictionaryAttr>(
          "vrc.static_declared_public_resources"))
    if (auto ids = attribute.getAs<DenseI32ArrayAttr>("resource_ids"))
      declared.insert(ids.asArrayRef().begin(), ids.asArrayRef().end());
  if (aliasEntries && reconciliation)
    for (Attribute attribute : aliasEntries) {
      auto alias = dyn_cast<DictionaryAttr>(attribute);
      auto id = alias ? alias.getAs<IntegerAttr>("resource_id") : nullptr;
      if (!id || resolved.count(id.getInt()) || declared.count(id.getInt()))
        continue;
      blocker("static public resource " + Twine(id.getInt()) + " (" +
              endpoint(alias) + ") has no configured buffer");
    }
  if (layout && !boolean(layout, "ready").value_or(true) &&
      integer(layout, "blocker_count") > 0)
    blocker("the canonical buffer layout has " +
            Twine(integer(layout, "blocker_count")) + " blocker(s)");

  // Device commands that still name a resource at run time.
  std::pair<StringLiteral, StringLiteral> counts[] = {
      {"dynamic_allocation_count", "dynamic Device allocation(s)"},
      {"dynamic_release_count", "dynamic Device release(s)"},
      {"dynamic_resolve_count", "dynamic physical buffer resolve(s)"},
      {"unresolved_dispatch_operand_count",
       "dispatch operand(s) without an assigned buffer"},
      {"unresolved_typed_transfer_operand_count",
       "transfer operand(s) without an assigned buffer"},
      {"transfer_call_count", "unlowered OpenCL transfer call(s)"},
  };
  for (auto [name, description] : counts)
    if (int64_t count = integer(readiness, name))
      blocker(Twine(count) + " " + description);
  if (!boolean(readiness, "dispatch_ready").value_or(true))
    blocker("Device dispatches are not ready");
}

/// Emits one remark per reason the static Unity resource lowering of the
/// module is not applied, after a summary remark. It changes nothing.
struct ReportStaticUnityResourceLoweringPass
    : impl::ReportStaticUnityResourceLoweringPassBase<
          ReportStaticUnityResourceLoweringPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto readiness = module->getAttrOfType<DictionaryAttr>(readinessAttribute);
    if (!readiness) {
      module.emitRemark("static Unity resource lowering was not assessed");
      return;
    }
    std::optional<bool> applied = boolean(readiness, "applied");
    std::optional<bool> ready = boolean(readiness, "pool_replacement_ready");
    StringRef state = applied ? (*applied ? "applied" : "not applied")
                              : (ready.value_or(false) ? "ready" : "not ready");
    module.emitRemark("static Unity resource lowering ") << state;
    forEachBlocker(module, readiness, [&](const Twine &message) {
      module.emitRemark("static Unity resource lowering blocker: ") << message;
    });
  }
};

/// Whether the module's ONNX binding is the transitional sequence-state
/// binding (udoncl.onnx_binding.transitional_sequence_state names an advance
/// symbol): the only exception to the required static Unity resource
/// lowering, kept by the TinyLlama Conversation and Chat programs until
/// stage 3 binds their KV cache through the public UdonONNX API.
bool banksOnnxSequenceState(ModuleOp module) {
  auto binding = module->getAttrOfType<DictionaryAttr>("udoncl.onnx_binding");
  auto transitional =
      binding ? binding.getAs<DictionaryAttr>("transitional_sequence_state")
              : DictionaryAttr();
  auto advance =
      transitional
          ? transitional.getAs<StringAttr>("advance_sequence_state_symbol")
          : StringAttr();
  return advance && !advance.getValue().empty();
}

/// Whether the module still allocates a Device buffer at run time.
bool allocatesDeviceBuffers(ModuleOp module) {
  bool allocates = false;
  module.walk([&](Operation *operation) {
    if (isa<::vrc::vrc::BufferAllocOp>(operation)) {
      allocates = true;
      return WalkResult::interrupt();
    }
    auto call = dyn_cast<func::CallOp>(operation);
    auto callee = call ? module.lookupSymbol<func::FuncOp>(call.getCallee())
                       : func::FuncOp();
    auto intrinsic =
        callee ? callee->getAttrOfType<StringAttr>("vrc.intrinsic") : nullptr;
    if (intrinsic && intrinsic.getValue() == "buffer_alloc") {
      allocates = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return allocates;
}

/// Fails unless every Device buffer of the module resolves at compile time
/// to one assigned buffer, listing the blockers in the error.
struct RequireStaticUnityResourceLoweringPass
    : impl::RequireStaticUnityResourceLoweringPassBase<
          RequireStaticUnityResourceLoweringPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto readiness = module->getAttrOfType<DictionaryAttr>(readinessAttribute);
    std::optional<bool> applied =
        readiness ? boolean(readiness, "applied") : std::nullopt;
    if (applied.value_or(false))
      return;
    // Only a Device buffer must resolve: a module without one (for example
    // one that only reads immutable static tensors) passes.
    auto inventory = module->getAttrOfType<DictionaryAttr>(inventoryAttribute);
    auto resources =
        inventory ? inventory.getAs<ArrayAttr>("resources") : ArrayAttr();
    if ((!resources || resources.empty()) && !allocatesDeviceBuffers(module))
      return;
    // Transitional exception (removed in stage 3): a Host on the private
    // sequence-state binding keeps the run-time Device buffer pool.
    if (banksOnnxSequenceState(module))
      return;
    InFlightDiagnostic error = module.emitError(
        "every Unity Device buffer must resolve at compile time to one "
        "assigned buffer, but the static Unity resource lowering ");
    if (!readiness) {
      error << "was not assessed";
    } else {
      error << "is not applied";
      StringRef separator = ": ";
      forEachBlocker(module, readiness, [&](const Twine &message) {
        error << separator << message;
        separator = "; ";
      });
    }
    // Only the transitional sequence-state programs are exempt; name the
    // program that is not.
    if (auto program = module->getAttrOfType<StringAttr>("udoncl.program"))
      error << " (program \"" << program.getValue() << "\")";
    signalPassFailure();
  }
};

} // namespace
} // namespace vrc
