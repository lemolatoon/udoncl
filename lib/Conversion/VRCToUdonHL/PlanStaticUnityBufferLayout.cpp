#include "VRC/Conversion/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <map>
#include <set>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_PLANSTATICUNITYBUFFERLAYOUTPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral inventoryAttribute =
    "vrc.static_device_resource_inventory";
constexpr StringLiteral assignmentsAttribute = "vrc.unity_texture_assignments";
constexpr StringLiteral aliasesAttribute = "vrc.static_public_resource_aliases";
constexpr StringLiteral reconciliationAttribute =
    "vrc.static_unity_buffer_reconciliation";
constexpr StringLiteral planAttribute = "vrc.static_unity_buffer_layout_plan";

struct Resource {
  int32_t id;
  int64_t capacity;
  StringAttr format;
  BoolAttr isPublic;
  BoolAttr persistent;
  StringAttr publicAccess;
  StringAttr name;
  StringAttr dataType;
  IntegerAttr logicalBytes;
};

struct Assignment {
  Resource resource;
  int32_t textureId;
};

struct Alias {
  StringAttr access;
  int32_t index;
  int32_t textureId;
};

bool schemaV1(DictionaryAttr attribute) {
  auto version =
      attribute ? attribute.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

struct PlanStaticUnityBufferLayoutPass
    : impl::PlanStaticUnityBufferLayoutPassBase<
          PlanStaticUnityBufferLayoutPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(planAttribute);

    auto inventory = module->getAttrOfType<DictionaryAttr>(inventoryAttribute);
    auto assignments =
        module->getAttrOfType<DictionaryAttr>(assignmentsAttribute);
    auto aliases = module->getAttrOfType<DictionaryAttr>(aliasesAttribute);
    auto reconciliation =
        module->getAttrOfType<DictionaryAttr>(reconciliationAttribute);
    auto resources =
        inventory ? inventory.getAs<ArrayAttr>("resources") : nullptr;
    auto textures =
        assignments ? assignments.getAs<ArrayAttr>("assignments") : nullptr;
    auto publicAliases =
        aliases ? aliases.getAs<ArrayAttr>("aliases") : nullptr;
    auto legacyBuffers =
        reconciliation ? reconciliation.getAs<ArrayAttr>("buffers") : nullptr;
    auto configuredCount =
        reconciliation ? reconciliation.getAs<IntegerAttr>("configured_count")
                       : nullptr;
    auto resolvedAliasCount =
        reconciliation
            ? reconciliation.getAs<IntegerAttr>("resolved_alias_count")
            : nullptr;
    auto unresolvedPublicCount =
        reconciliation
            ? reconciliation.getAs<IntegerAttr>("unresolved_public_count")
            : nullptr;
    auto unusedAliasCount =
        reconciliation ? reconciliation.getAs<IntegerAttr>("unused_alias_count")
                       : nullptr;
    auto declaredAliasCount =
        reconciliation
            ? reconciliation.getAs<IntegerAttr>("declared_alias_count")
            : nullptr;
    if (!schemaV1(inventory) || !resources || !schemaV1(assignments) ||
        !textures || !schemaV1(aliases) || !publicAliases ||
        !schemaV1(reconciliation) || !legacyBuffers || !configuredCount ||
        !resolvedAliasCount || !unresolvedPublicCount || !unusedAliasCount ||
        configuredCount.getInt() < 0 || resolvedAliasCount.getInt() < 0 ||
        unresolvedPublicCount.getInt() < 0 || unusedAliasCount.getInt() < 0 ||
        configuredCount.getInt() !=
            static_cast<int64_t>(legacyBuffers.size())) {
      module.emitError("static Unity buffer layout requires inventory, "
                       "assignments, aliases and reconciliation schema v1");
      signalPassFailure();
      return;
    }

    std::map<int32_t, Resource> inventoryById;
    std::set<std::pair<std::string, int32_t>> identities;
    for (Attribute attribute : resources) {
      auto resource = dyn_cast<DictionaryAttr>(attribute);
      auto kind =
          resource ? resource.getAs<StringAttr>("resource_kind") : nullptr;
      auto id = resource ? resource.getAs<IntegerAttr>("resource_id") : nullptr;
      if (!kind || !id || id.getInt() < 0 ||
          !identities
               .insert(
                   {kind.getValue().str(), static_cast<int32_t>(id.getInt())})
               .second) {
        module.emitError("invalid keyed static resource inventory");
        signalPassFailure();
        return;
      }
      if (kind != "allocation")
        continue;
      auto capacity = resource.getAs<IntegerAttr>("capacity_bytes");
      auto format = resource.getAs<StringAttr>("format");
      auto isPublic = resource.getAs<BoolAttr>("public");
      auto persistent = resource.getAs<BoolAttr>("persistent");
      auto access = resource.getAs<StringAttr>("public_access");
      auto name = resource.getAs<StringAttr>("name");
      auto dataType = resource.getAs<StringAttr>("data_type");
      auto logicalBytes = resource.getAs<IntegerAttr>("logical_bytes");
      bool validPublicMetadata =
          !isPublic || !isPublic.getValue() ||
          (capacity && name && !name.getValue().empty() && dataType &&
           (dataType == "float32" || dataType == "int32" ||
            dataType == "int64") &&
           logicalBytes && logicalBytes.getInt() >= 0 &&
           logicalBytes.getInt() <= capacity.getInt());
      if (!capacity || capacity.getInt() <= 0 || !format ||
          format != "rgba8_unorm" || !isPublic || !persistent ||
          (isPublic.getValue() &&
           (!access ||
            (access != "input" && access != "output" && access != "inout"))) ||
          (!isPublic.getValue() &&
           (access || name || dataType || logicalBytes)) ||
          !validPublicMetadata) {
        module.emitError("invalid allocation in static resource inventory");
        signalPassFailure();
        return;
      }
      int32_t resourceId = static_cast<int32_t>(id.getInt());
      inventoryById.emplace(resourceId,
                            Resource{resourceId, capacity.getInt(), format,
                                     isPublic, persistent, access, name,
                                     dataType, logicalBytes});
    }

    SmallVector<Assignment> ordered;
    std::set<int32_t> assignedResources;
    std::set<int32_t> textureIds;
    std::map<int32_t, int32_t> textureByResource;
    for (Attribute attribute : textures) {
      auto assignment = dyn_cast<DictionaryAttr>(attribute);
      auto kind =
          assignment ? assignment.getAs<StringAttr>("resource_kind") : nullptr;
      auto id =
          assignment ? assignment.getAs<IntegerAttr>("resource_id") : nullptr;
      auto textureId =
          assignment ? assignment.getAs<IntegerAttr>("texture_id") : nullptr;
      auto capacity = assignment
                          ? assignment.getAs<IntegerAttr>("capacity_bytes")
                          : nullptr;
      auto format =
          assignment ? assignment.getAs<StringAttr>("format") : nullptr;
      auto isPublic =
          assignment ? assignment.getAs<BoolAttr>("public") : nullptr;
      auto persistent =
          assignment ? assignment.getAs<BoolAttr>("persistent") : nullptr;
      auto access =
          assignment ? assignment.getAs<StringAttr>("public_access") : nullptr;
      auto found = id ? inventoryById.find(static_cast<int32_t>(id.getInt()))
                      : inventoryById.end();
      if (!kind || kind != "allocation" || !id || id.getInt() < 0 ||
          !textureId || textureId.getInt() <= 0 || !capacity || !format ||
          !isPublic || !persistent || found == inventoryById.end() ||
          capacity.getInt() != found->second.capacity ||
          format != found->second.format ||
          isPublic != found->second.isPublic ||
          persistent != found->second.persistent ||
          access != found->second.publicAccess ||
          !assignedResources.insert(static_cast<int32_t>(id.getInt())).second ||
          !textureIds.insert(static_cast<int32_t>(textureId.getInt())).second) {
        module.emitError(
            "Unity Texture assignment does not match static inventory");
        signalPassFailure();
        return;
      }
      ordered.push_back(
          {found->second, static_cast<int32_t>(textureId.getInt())});
      textureByResource[static_cast<int32_t>(id.getInt())] =
          static_cast<int32_t>(textureId.getInt());
    }
    if (assignedResources.size() != inventoryById.size()) {
      module.emitError(
          "Unity Texture assignments do not cover all allocations");
      signalPassFailure();
      return;
    }
    llvm::sort(ordered, [](const Assignment &lhs, const Assignment &rhs) {
      return lhs.textureId < rhs.textureId;
    });
    for (auto [index, assignment] : llvm::enumerate(ordered))
      if (assignment.textureId != static_cast<int32_t>(index + 1)) {
        module.emitError("Unity Texture IDs must be dense from one");
        signalPassFailure();
        return;
      }

    std::map<int32_t, Alias> aliasByResource;
    std::set<std::pair<std::string, int32_t>> endpoints;
    for (Attribute attribute : publicAliases) {
      auto alias = dyn_cast<DictionaryAttr>(attribute);
      auto kind = alias ? alias.getAs<StringAttr>("resource_kind") : nullptr;
      auto id = alias ? alias.getAs<IntegerAttr>("resource_id") : nullptr;
      auto textureId = alias ? alias.getAs<IntegerAttr>("texture_id") : nullptr;
      auto access = alias ? alias.getAs<StringAttr>("public_access") : nullptr;
      auto index = alias ? alias.getAs<IntegerAttr>("public_index") : nullptr;
      auto found = id ? inventoryById.find(static_cast<int32_t>(id.getInt()))
                      : inventoryById.end();
      auto assigned =
          id ? textureByResource.find(static_cast<int32_t>(id.getInt()))
             : textureByResource.end();
      if (!kind || kind != "allocation" || !id || id.getInt() < 0 ||
          !textureId || textureId.getInt() <= 0 || !access ||
          (access != "input" && access != "output" && access != "inout") ||
          !index || index.getInt() < 0 || found == inventoryById.end() ||
          assigned == textureByResource.end() ||
          assigned->second != textureId.getInt() ||
          !found->second.isPublic.getValue() ||
          found->second.publicAccess != access ||
          !aliasByResource
               .emplace(static_cast<int32_t>(id.getInt()),
                        Alias{access, static_cast<int32_t>(index.getInt()),
                              static_cast<int32_t>(textureId.getInt())})
               .second ||
          !endpoints
               .insert({access.getValue().str(),
                        static_cast<int32_t>(index.getInt())})
               .second) {
        module.emitError("invalid static public resource alias");
        signalPassFailure();
        return;
      }
    }

    int32_t missingPublic = 0;
    SmallVector<Attribute> plannedBuffers;
    for (const Assignment &assignment : ordered) {
      auto alias = aliasByResource.find(assignment.resource.id);
      if (assignment.resource.isPublic.getValue() &&
          (alias == aliasByResource.end() ||
           alias->second.textureId != assignment.textureId))
        ++missingPublic;
      SmallVector<NamedAttribute> fields{
          builder.getNamedAttr("buffer_id",
                               builder.getI32IntegerAttr(assignment.textureId)),
          builder.getNamedAttr(
              "capacity_bytes",
              builder.getI64IntegerAttr(assignment.resource.capacity)),
          builder.getNamedAttr("format", assignment.resource.format),
          builder.getNamedAttr(
              "heap_slot",
              builder.getStringAttr("__udoncl_buffer_" +
                                    std::to_string(assignment.textureId))),
          builder.getNamedAttr("persistent", assignment.resource.persistent),
          builder.getNamedAttr("public", assignment.resource.isPublic),
          builder.getNamedAttr(
              "resource_id", builder.getI32IntegerAttr(assignment.resource.id)),
          builder.getNamedAttr("resource_kind",
                               builder.getStringAttr("allocation")),
          builder.getNamedAttr("texture_id",
                               builder.getI32IntegerAttr(assignment.textureId)),
      };
      if (alias != aliasByResource.end()) {
        fields.push_back(
            builder.getNamedAttr("data_type", assignment.resource.dataType));
        fields.push_back(builder.getNamedAttr(
            "logical_bytes", assignment.resource.logicalBytes));
        fields.push_back(
            builder.getNamedAttr("name", assignment.resource.name));
        fields.push_back(
            builder.getNamedAttr("public_access", alias->second.access));
        fields.push_back(builder.getNamedAttr(
            "public_index", builder.getI32IntegerAttr(alias->second.index)));
      }
      plannedBuffers.push_back(builder.getDictionaryAttr(fields));
    }

    int32_t unresolvedPublic =
        static_cast<int32_t>(unresolvedPublicCount.getInt());
    int32_t unusedAliases = static_cast<int32_t>(unusedAliasCount.getInt());
    int32_t resolvedAliases = static_cast<int32_t>(resolvedAliasCount.getInt());
    // An alias of a declared (UDONCL_PUBLIC) resource needs no configured
    // buffer.
    int32_t declaredAliases =
        declaredAliasCount ? static_cast<int32_t>(declaredAliasCount.getInt())
                           : 0;
    if (declaredAliases < 0 ||
        resolvedAliases + unusedAliases + declaredAliases !=
            static_cast<int32_t>(publicAliases.size())) {
      module.emitError("public alias reconciliation count is inconsistent");
      signalPassFailure();
      return;
    }
    int32_t blockers = missingPublic + unresolvedPublic + unusedAliases;
    bool ready = blockers == 0;
    int32_t planned = static_cast<int32_t>(plannedBuffers.size());
    int32_t legacy = static_cast<int32_t>(configuredCount.getInt());
    int32_t removable = ready ? std::max(legacy - planned, 0) : 0;
    int32_t additional = ready ? std::max(planned - legacy, 0) : 0;
    module->setAttr(
        planAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("additional_planned_count",
                                 builder.getI32IntegerAttr(additional)),
            builder.getNamedAttr("blocker_count",
                                 builder.getI32IntegerAttr(blockers)),
            builder.getNamedAttr("buffers",
                                 builder.getArrayAttr(plannedBuffers)),
            builder.getNamedAttr("legacy_configured_count",
                                 builder.getI32IntegerAttr(legacy)),
            builder.getNamedAttr("missing_public_alias_count",
                                 builder.getI32IntegerAttr(missingPublic)),
            builder.getNamedAttr("planned_count",
                                 builder.getI32IntegerAttr(planned)),
            builder.getNamedAttr("ready", builder.getBoolAttr(ready)),
            builder.getNamedAttr("removable_legacy_count",
                                 builder.getI32IntegerAttr(removable)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("unresolved_public_count",
                                 builder.getI32IntegerAttr(unresolvedPublic)),
            builder.getNamedAttr("unused_alias_count",
                                 builder.getI32IntegerAttr(unusedAliases)),
        }));
  }
};

} // namespace
} // namespace vrc
