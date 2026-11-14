#include "VRC/Conversion/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <set>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_RECONCILESTATICUNITYBUFFERSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral configuredAttribute = "vrc.unity_configured_buffers";
constexpr StringLiteral inventoryAttribute =
    "vrc.static_device_resource_inventory";
constexpr StringLiteral aliasesAttribute = "vrc.static_public_resource_aliases";
constexpr StringLiteral reconciliationAttribute =
    "vrc.static_unity_buffer_reconciliation";

using Endpoint = std::pair<std::string, int32_t>;

struct Alias {
  int32_t resourceId;
  int32_t textureId;
  int64_t capacityBytes;
};

bool hasSchemaVersion(DictionaryAttr attribute) {
  auto version =
      attribute ? attribute.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

struct ReconcileStaticUnityBuffersPass
    : impl::ReconcileStaticUnityBuffersPassBase<
          ReconcileStaticUnityBuffersPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(reconciliationAttribute);

    auto configured =
        module->getAttrOfType<DictionaryAttr>(configuredAttribute);
    auto inventory = module->getAttrOfType<DictionaryAttr>(inventoryAttribute);
    auto aliases = module->getAttrOfType<DictionaryAttr>(aliasesAttribute);
    auto configuredBuffers =
        configured ? configured.getAs<ArrayAttr>("buffers") : nullptr;
    auto resources =
        inventory ? inventory.getAs<ArrayAttr>("resources") : nullptr;
    auto resolvedAliases =
        aliases ? aliases.getAs<ArrayAttr>("aliases") : nullptr;
    if (!hasSchemaVersion(configured) || !configuredBuffers ||
        !hasSchemaVersion(inventory) || !resources ||
        !hasSchemaVersion(aliases) || !resolvedAliases) {
      module.emitError("static Unity buffer reconciliation requires configured "
                       "buffers, inventory and public aliases");
      signalPassFailure();
      return;
    }

    std::map<int32_t, int64_t> capacityByResource;
    for (Attribute attribute : resources) {
      auto resource = dyn_cast<DictionaryAttr>(attribute);
      auto kind =
          resource ? resource.getAs<StringAttr>("resource_kind") : nullptr;
      if (!kind || kind != "allocation")
        continue;
      auto id = resource.getAs<IntegerAttr>("resource_id");
      auto capacity = resource.getAs<IntegerAttr>("capacity_bytes");
      if (!id || id.getInt() < 0 || !capacity || capacity.getInt() <= 0 ||
          capacityByResource.find(static_cast<int32_t>(id.getInt())) !=
              capacityByResource.end()) {
        module.emitError("invalid allocation in static resource inventory");
        signalPassFailure();
        return;
      }
      capacityByResource[static_cast<int32_t>(id.getInt())] = capacity.getInt();
    }

    std::map<Endpoint, Alias> aliasByEndpoint;
    std::set<int32_t> aliasTextureIds;
    for (Attribute attribute : resolvedAliases) {
      auto alias = dyn_cast<DictionaryAttr>(attribute);
      auto access = alias ? alias.getAs<StringAttr>("public_access") : nullptr;
      auto index = alias ? alias.getAs<IntegerAttr>("public_index") : nullptr;
      auto kind = alias ? alias.getAs<StringAttr>("resource_kind") : nullptr;
      auto resourceId =
          alias ? alias.getAs<IntegerAttr>("resource_id") : nullptr;
      auto textureId = alias ? alias.getAs<IntegerAttr>("texture_id") : nullptr;
      if (!access || (access != "input" && access != "output") || !index ||
          index.getInt() < 0 || !kind || kind != "allocation" || !resourceId ||
          resourceId.getInt() < 0 || !textureId || textureId.getInt() <= 0) {
        module.emitError("invalid static public resource alias");
        signalPassFailure();
        return;
      }
      int32_t resource = static_cast<int32_t>(resourceId.getInt());
      auto capacity = capacityByResource.find(resource);
      Endpoint endpoint{access.getValue().str(),
                        static_cast<int32_t>(index.getInt())};
      if (capacity == capacityByResource.end() ||
          aliasByEndpoint.find(endpoint) != aliasByEndpoint.end() ||
          !aliasTextureIds.insert(static_cast<int32_t>(textureId.getInt()))
               .second) {
        module.emitError("static public resource alias is inconsistent with "
                         "the resource inventory");
        signalPassFailure();
        return;
      }
      aliasByEndpoint[endpoint] = {
          resource, static_cast<int32_t>(textureId.getInt()), capacity->second};
    }

    struct Buffer {
      int32_t id;
      StringAttr heapSlot;
      int64_t capacityBytes;
      StringAttr access;
      std::optional<int32_t> publicIndex;
    };
    SmallVector<Buffer> buffers;
    std::set<int32_t> bufferIds;
    std::set<std::string> heapSlots;
    std::set<Endpoint> configuredEndpoints;
    for (Attribute attribute : configuredBuffers) {
      auto buffer = dyn_cast<DictionaryAttr>(attribute);
      auto id = buffer ? buffer.getAs<IntegerAttr>("buffer_id") : nullptr;
      auto heapSlot = buffer ? buffer.getAs<StringAttr>("heap_slot") : nullptr;
      auto capacity =
          buffer ? buffer.getAs<IntegerAttr>("capacity_bytes") : nullptr;
      auto access =
          buffer ? buffer.getAs<StringAttr>("public_access") : nullptr;
      auto index = buffer ? buffer.getAs<IntegerAttr>("public_index") : nullptr;
      bool validAccess = !access || access == "input" || access == "output";
      if (!id || id.getInt() <= 0 || !heapSlot || heapSlot.empty() ||
          !capacity || capacity.getInt() <= 0 || !validAccess ||
          (index && (!access || index.getInt() < 0)) ||
          !bufferIds.insert(static_cast<int32_t>(id.getInt())).second ||
          !heapSlots.insert(heapSlot.getValue().str()).second) {
        module.emitError("invalid configured Unity buffer");
        signalPassFailure();
        return;
      }
      std::optional<int32_t> publicIndex;
      if (index) {
        publicIndex = static_cast<int32_t>(index.getInt());
        if (!configuredEndpoints.insert({access.getValue().str(), *publicIndex})
                 .second) {
          module.emitError("duplicate configured public Device endpoint");
          signalPassFailure();
          return;
        }
      }
      buffers.push_back({static_cast<int32_t>(id.getInt()), heapSlot,
                         capacity.getInt(), access, publicIndex});
    }
    llvm::sort(buffers, [](const Buffer &lhs, const Buffer &rhs) {
      return lhs.id < rhs.id;
    });

    std::set<Endpoint> usedAliases;
    SmallVector<Attribute> output;
    int32_t resolvedCount = 0;
    int32_t unresolvedPublicCount = 0;
    int32_t unresolvedPrivateCount = 0;
    for (const Buffer &buffer : buffers) {
      SmallVector<NamedAttribute> fields{
          builder.getNamedAttr("buffer_id",
                               builder.getI32IntegerAttr(buffer.id)),
          builder.getNamedAttr("heap_slot", buffer.heapSlot),
      };
      if (buffer.access)
        fields.push_back(builder.getNamedAttr("public_access", buffer.access));
      if (buffer.publicIndex)
        fields.push_back(builder.getNamedAttr(
            "public_index", builder.getI32IntegerAttr(*buffer.publicIndex)));

      const Alias *resolved = nullptr;
      Endpoint endpoint;
      if (buffer.access && buffer.publicIndex) {
        endpoint = {buffer.access.getValue().str(), *buffer.publicIndex};
        auto found = aliasByEndpoint.find(endpoint);
        if (found != aliasByEndpoint.end())
          resolved = &found->second;
      }
      if (resolved) {
        if (resolved->capacityBytes != buffer.capacityBytes) {
          module.emitError("configured Unity buffer capacity does not match "
                           "its static public resource");
          signalPassFailure();
          return;
        }
        usedAliases.insert(endpoint);
        ++resolvedCount;
        fields.push_back(builder.getNamedAttr(
            "resolution", builder.getStringAttr("static_alias")));
        fields.push_back(builder.getNamedAttr(
            "resource_kind", builder.getStringAttr("allocation")));
        fields.push_back(builder.getNamedAttr(
            "resource_id", builder.getI32IntegerAttr(resolved->resourceId)));
        fields.push_back(builder.getNamedAttr(
            "texture_id", builder.getI32IntegerAttr(resolved->textureId)));
      } else if (buffer.access) {
        ++unresolvedPublicCount;
        fields.push_back(builder.getNamedAttr(
            "resolution", builder.getStringAttr("legacy_public")));
      } else {
        ++unresolvedPrivateCount;
        fields.push_back(builder.getNamedAttr(
            "resolution", builder.getStringAttr("legacy_private")));
      }
      output.push_back(builder.getDictionaryAttr(fields));
    }

    module->setAttr(
        reconciliationAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("buffers", builder.getArrayAttr(output)),
            builder.getNamedAttr("configured_count",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(buffers.size()))),
            builder.getNamedAttr("resolved_alias_count",
                                 builder.getI32IntegerAttr(resolvedCount)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr(
                "unresolved_private_count",
                builder.getI32IntegerAttr(unresolvedPrivateCount)),
            builder.getNamedAttr(
                "unresolved_public_count",
                builder.getI32IntegerAttr(unresolvedPublicCount)),
            builder.getNamedAttr(
                "unused_alias_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(
                    aliasByEndpoint.size() - usedAliases.size()))),
        }));
  }
};

} // namespace
} // namespace vrc
