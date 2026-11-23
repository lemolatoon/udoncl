#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MATERIALIZESTATICDISPATCHRESOURCESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral inventoryAttribute =
    "vrc.static_device_resource_inventory";
constexpr StringLiteral assignmentsAttribute = "vrc.unity_texture_assignments";
constexpr StringLiteral snapshotAttribute =
    "vrc.static_kernel_binding_snapshot";
constexpr StringLiteral summaryAttribute =
    "vrc.static_dispatch_resource_materialization";

struct ResourcePlan {
  int64_t capacity;
  int32_t textureId;
  StringAttr format;
};

/// The `vrc.assigned_buffer` that defines `value` through single-operand
/// casts, if any.
::vrc::vrc::AssignedBufferOp assignedBuffer(Value value) {
  while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1)
      return {};
    value = cast.getInputs().front();
  }
  return value.getDefiningOp<::vrc::vrc::AssignedBufferOp>();
}

bool hasSchemaVersion(DictionaryAttr attribute) {
  auto version =
      attribute ? attribute.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

FailureOr<DenseMap<int32_t, ResourcePlan>> readResourcePlan(ModuleOp module) {
  auto inventory = module->getAttrOfType<DictionaryAttr>(inventoryAttribute);
  auto assignments =
      module->getAttrOfType<DictionaryAttr>(assignmentsAttribute);
  auto resources =
      inventory ? inventory.getAs<ArrayAttr>("resources") : nullptr;
  auto textures =
      assignments ? assignments.getAs<ArrayAttr>("assignments") : nullptr;
  if (!hasSchemaVersion(inventory) || !resources ||
      !hasSchemaVersion(assignments) || !textures) {
    module.emitError("static dispatch materialization requires keyed inventory "
                     "and Texture assignments");
    return failure();
  }

  DenseMap<int32_t, std::pair<int64_t, StringAttr>> inventoryById;
  for (Attribute attribute : resources) {
    auto resource = dyn_cast<DictionaryAttr>(attribute);
    auto kind =
        resource ? resource.getAs<StringAttr>("resource_kind") : nullptr;
    auto id = resource ? resource.getAs<IntegerAttr>("resource_id") : nullptr;
    auto capacity =
        resource ? resource.getAs<IntegerAttr>("capacity_bytes") : nullptr;
    auto format = resource ? resource.getAs<StringAttr>("format") : nullptr;
    if (!kind || kind != "allocation")
      continue;
    if (!id || id.getInt() < 0 || !capacity || capacity.getInt() <= 0 ||
        !format || format != "rgba8_unorm" ||
        inventoryById.contains(static_cast<int32_t>(id.getInt()))) {
      module.emitError("invalid allocation in static resource inventory");
      return failure();
    }
    inventoryById[static_cast<int32_t>(id.getInt())] = {capacity.getInt(),
                                                        format};
  }

  DenseMap<int32_t, ResourcePlan> result;
  DenseSet<int32_t> textureIds;
  for (Attribute attribute : textures) {
    auto assignment = dyn_cast<DictionaryAttr>(attribute);
    auto kind =
        assignment ? assignment.getAs<StringAttr>("resource_kind") : nullptr;
    auto id =
        assignment ? assignment.getAs<IntegerAttr>("resource_id") : nullptr;
    auto textureId =
        assignment ? assignment.getAs<IntegerAttr>("texture_id") : nullptr;
    auto capacity =
        assignment ? assignment.getAs<IntegerAttr>("capacity_bytes") : nullptr;
    auto format = assignment ? assignment.getAs<StringAttr>("format") : nullptr;
    if (!kind || kind != "allocation")
      continue;
    auto inventoryEntry =
        id ? inventoryById.find(static_cast<int32_t>(id.getInt()))
           : inventoryById.end();
    if (!id || id.getInt() < 0 || !textureId || textureId.getInt() <= 0 ||
        !capacity || capacity.getInt() <= 0 || !format ||
        format != "rgba8_unorm" || inventoryEntry == inventoryById.end() ||
        inventoryEntry->second.first != capacity.getInt() ||
        inventoryEntry->second.second != format ||
        result.contains(static_cast<int32_t>(id.getInt())) ||
        !textureIds.insert(static_cast<int32_t>(textureId.getInt())).second) {
      module.emitError(
          "Texture assignment does not match static resource inventory");
      return failure();
    }
    result[static_cast<int32_t>(id.getInt())] = {
        capacity.getInt(), static_cast<int32_t>(textureId.getInt()), format};
  }
  return result;
}

struct MaterializeStaticDispatchResourcesPass
    : impl::MaterializeStaticDispatchResourcesPassBase<
          MaterializeStaticDispatchResourcesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(summaryAttribute);

    int32_t candidateDispatches = 0;
    SmallVector<::vrc::vrc::DispatchOp> dispatches;
    module.walk([&](::vrc::vrc::DispatchOp dispatch) {
      ++candidateDispatches;
      if (dispatch->hasAttr(snapshotAttribute))
        dispatches.push_back(dispatch);
    });
    if (candidateDispatches == 0)
      return;

    FailureOr<DenseMap<int32_t, ResourcePlan>> plans =
        DenseMap<int32_t, ResourcePlan>();
    if (!dispatches.empty()) {
      plans = readResourcePlan(module);
      if (failed(plans)) {
        signalPassFailure();
        return;
      }
    }

    int32_t materialized = 0;
    int32_t unresolved = 0;
    for (::vrc::vrc::DispatchOp dispatch : dispatches) {
      auto snapshot =
          dispatch->getAttrOfType<DictionaryAttr>(snapshotAttribute);
      auto version =
          snapshot ? snapshot.getAs<IntegerAttr>("schema_version") : nullptr;
      auto resources =
          snapshot ? snapshot.getAs<ArrayAttr>("resources") : nullptr;
      if (!version || version.getInt() != 1 || !resources) {
        dispatch.emitError("invalid static kernel binding snapshot");
        signalPassFailure();
        return;
      }
      DenseSet<int32_t> indices;
      DenseMap<int32_t, Value> values;
      for (Attribute attribute : resources) {
        auto resource = dyn_cast<DictionaryAttr>(attribute);
        auto kind = resource ? resource.getAs<StringAttr>("kind") : nullptr;
        auto index =
            resource ? resource.getAs<IntegerAttr>("operand_index") : nullptr;
        auto id =
            resource ? resource.getAs<IntegerAttr>("resource_id") : nullptr;
        auto access = resource ? resource.getAs<StringAttr>("access") : nullptr;
        if (!kind || !index || index.getInt() < 0 ||
            index.getInt() >= dispatch.getResources().size() || !id ||
            id.getInt() < 0 || !access ||
            !indices.insert(static_cast<int32_t>(index.getInt())).second) {
          dispatch.emitError("invalid resource in static kernel snapshot");
          signalPassFailure();
          return;
        }
        auto expectedAccess = dyn_cast<StringAttr>(
            dispatch.getResourceAccesses()[index.getInt()]);
        if (!expectedAccess || expectedAccess != access) {
          dispatch.emitError("static kernel snapshot access is inconsistent");
          signalPassFailure();
          return;
        }
        if (kind == "static_tensor")
          continue;
        if (kind == "logical_buffer") {
          ++unresolved;
          continue;
        }
        if (kind != "allocation") {
          dispatch.emitError("static kernel snapshot resource kind is invalid");
          signalPassFailure();
          return;
        }
        int32_t resourceId = static_cast<int32_t>(id.getInt());
        // An operand the compiler already assigned keeps its buffer.
        if (auto assigned =
                assignedBuffer(dispatch.getResources()[index.getInt()]);
            assigned && assigned.getResourceId() == resourceId) {
          dispatch->setOperand(static_cast<unsigned>(index.getInt() + 1),
                               assigned.getResult());
          ++materialized;
          continue;
        }
        auto plan = plans->find(resourceId);
        if (plan == plans->end()) {
          dispatch.emitError(
              "static dispatch allocation has no Texture assignment");
          signalPassFailure();
          return;
        }
        Value assigned;
        auto existing = values.find(resourceId);
        if (existing != values.end()) {
          assigned = existing->second;
        } else {
          OpBuilder opBuilder(dispatch);
          OperationState state(
              dispatch.getLoc(),
              ::vrc::vrc::AssignedBufferOp::getOperationName());
          state.addTypes(::vrc::vrc::BufferType::get(module.getContext()));
          state.addAttribute("resource_id",
                             builder.getI32IntegerAttr(resourceId));
          state.addAttribute("texture_id",
                             builder.getI32IntegerAttr(plan->second.textureId));
          state.addAttribute("capacity_bytes",
                             builder.getI64IntegerAttr(plan->second.capacity));
          state.addAttribute("format", plan->second.format);
          assigned = opBuilder.create(state)->getResult(0);
          values[resourceId] = assigned;
        }
        dispatch->setOperand(static_cast<unsigned>(index.getInt() + 1),
                             assigned);
        ++materialized;
      }
    }
    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr(
                "candidate_dispatch_count",
                builder.getI32IntegerAttr(candidateDispatches)),
            builder.getNamedAttr("materialized_operand_count",
                                 builder.getI32IntegerAttr(materialized)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("snapshotted_dispatch_count",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(dispatches.size()))),
            builder.getNamedAttr("unresolved_dispatch_count",
                                 builder.getI32IntegerAttr(
                                     candidateDispatches -
                                     static_cast<int32_t>(dispatches.size()))),
            builder.getNamedAttr("unresolved_operand_count",
                                 builder.getI32IntegerAttr(unresolved)),
        }));
  }
};

} // namespace
} // namespace vrc
