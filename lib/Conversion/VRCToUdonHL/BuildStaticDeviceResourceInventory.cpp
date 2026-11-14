#include "VRC/Conversion/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_BUILDSTATICDEVICERESOURCEINVENTORYPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral requirementsAttribute =
    "vrc.static_device_resource_requirements";
constexpr StringLiteral persistenceAttribute =
    "vrc.static_device_resource_persistence";
constexpr StringLiteral publicAttribute =
    "vrc.static_device_resource_public_bindings";
constexpr StringLiteral inventoryAttribute =
    "vrc.static_device_resource_inventory";

struct Requirement {
  int64_t capacity;
  int64_t memoryFlags;
  StringAttr format;
  StringAttr usage;
};

struct PublicBinding {
  StringAttr access;
  int32_t index;
  StringAttr name;
  StringAttr dataType;
  int64_t logicalBytes;
  int64_t capacityBytes;
};

bool hasSchemaVersion(DictionaryAttr attribute) {
  auto version =
      attribute ? attribute.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

FailureOr<int32_t> allocationIdentity(DictionaryAttr resource) {
  auto kind = resource ? resource.getAs<StringAttr>("resource_kind") : nullptr;
  auto id = resource ? resource.getAs<IntegerAttr>("resource_id") : nullptr;
  if (!kind || kind != "allocation" || !id || id.getInt() < 0)
    return failure();
  return static_cast<int32_t>(id.getInt());
}

struct BuildStaticDeviceResourceInventoryPass
    : impl::BuildStaticDeviceResourceInventoryPassBase<
          BuildStaticDeviceResourceInventoryPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(inventoryAttribute);

    auto discovery = module->getAttrOfType<DictionaryAttr>(discoveryAttribute);
    auto requirements =
        module->getAttrOfType<DictionaryAttr>(requirementsAttribute);
    auto persistence =
        module->getAttrOfType<DictionaryAttr>(persistenceAttribute);
    auto contexts =
        discovery ? discovery.getAs<ArrayAttr>("contexts") : nullptr;
    auto requirementResources =
        requirements ? requirements.getAs<ArrayAttr>("resources") : nullptr;
    auto persistenceResources =
        persistence ? persistence.getAs<ArrayAttr>("resources") : nullptr;
    if (!hasSchemaVersion(discovery) || !contexts ||
        !hasSchemaVersion(requirements) || !requirementResources ||
        !hasSchemaVersion(persistence) || !persistenceResources) {
      module.emitError("static resource inventory inputs are incomplete");
      signalPassFailure();
      return;
    }

    llvm::DenseSet<int32_t> contextIds;
    for (Attribute attribute : contexts) {
      auto context = dyn_cast<DictionaryAttr>(attribute);
      auto id = context ? context.getAs<IntegerAttr>("context_id") : nullptr;
      if (!id || id.getInt() < 0 ||
          !contextIds.insert(static_cast<int32_t>(id.getInt())).second) {
        module.emitError("invalid discovered allocation context");
        signalPassFailure();
        return;
      }
    }

    llvm::DenseMap<int32_t, Requirement> requirementById;
    for (Attribute attribute : requirementResources) {
      auto resource = dyn_cast<DictionaryAttr>(attribute);
      auto id = allocationIdentity(resource);
      auto capacity =
          resource ? resource.getAs<IntegerAttr>("capacity_bytes") : nullptr;
      auto flags =
          resource ? resource.getAs<IntegerAttr>("memory_flags") : nullptr;
      auto format = resource ? resource.getAs<StringAttr>("format") : nullptr;
      auto usage = resource ? resource.getAs<StringAttr>("usage") : nullptr;
      if (failed(id) || !capacity || capacity.getInt() <= 0 || !flags ||
          flags.getInt() < 0 || !format || format != "rgba8_unorm" || !usage ||
          usage != "storage" || !contextIds.contains(*id) ||
          requirementById.contains(*id)) {
        module.emitError("invalid static allocation requirement");
        signalPassFailure();
        return;
      }
      requirementById[*id] = {capacity.getInt(), flags.getInt(), format, usage};
    }

    llvm::DenseMap<int32_t, bool> persistenceById;
    for (Attribute attribute : persistenceResources) {
      auto resource = dyn_cast<DictionaryAttr>(attribute);
      auto id = allocationIdentity(resource);
      auto persistent =
          resource ? resource.getAs<BoolAttr>("persistent") : nullptr;
      if (failed(id) || !persistent || !contextIds.contains(*id) ||
          persistenceById.contains(*id)) {
        module.emitError("invalid static allocation persistence");
        signalPassFailure();
        return;
      }
      persistenceById[*id] = persistent.getValue();
    }

    auto publicBindings =
        module->getAttrOfType<DictionaryAttr>(publicAttribute);
    if (module->hasAttr("udoncl.onnx_binding") && !publicBindings) {
      module.emitError(
          "ONNX resource inventory requires public classification");
      signalPassFailure();
      return;
    }
    llvm::DenseMap<int32_t, PublicBinding> publicById;
    if (publicBindings) {
      auto resources = publicBindings.getAs<ArrayAttr>("resources");
      if (!hasSchemaVersion(publicBindings) || !resources) {
        module.emitError("invalid public resource classification schema");
        signalPassFailure();
        return;
      }
      for (Attribute attribute : resources) {
        auto resource = dyn_cast<DictionaryAttr>(attribute);
        auto id = allocationIdentity(resource);
        auto access =
            resource ? resource.getAs<StringAttr>("public_access") : nullptr;
        auto index =
            resource ? resource.getAs<IntegerAttr>("public_index") : nullptr;
        auto name = resource ? resource.getAs<StringAttr>("name") : nullptr;
        auto dataType =
            resource ? resource.getAs<StringAttr>("data_type") : nullptr;
        auto logical =
            resource ? resource.getAs<IntegerAttr>("logical_bytes") : nullptr;
        auto capacity =
            resource ? resource.getAs<IntegerAttr>("capacity_bytes") : nullptr;
        if (failed(id) || !contextIds.contains(*id) ||
            publicById.contains(*id) || !access ||
            (access != "input" && access != "output") || !index ||
            index.getInt() < 0 || !name || name.getValue().empty() ||
            !dataType || dataType.getValue().empty() || !logical ||
            logical.getInt() <= 0 || !capacity ||
            capacity.getInt() < logical.getInt()) {
          module.emitError("invalid public allocation binding");
          signalPassFailure();
          return;
        }
        publicById[*id] = {access,
                           static_cast<int32_t>(index.getInt()),
                           name,
                           dataType,
                           logical.getInt(),
                           capacity.getInt()};
      }
    }

    if (requirementById.size() != contextIds.size() ||
        persistenceById.size() != contextIds.size()) {
      module.emitError("allocation inventory inputs do not cover discovery");
      signalPassFailure();
      return;
    }

    SmallVector<int32_t> ids(contextIds.begin(), contextIds.end());
    llvm::sort(ids);
    SmallVector<Attribute> output;
    for (int32_t id : ids) {
      const Requirement &requirement = requirementById[id];
      auto publicFound = publicById.find(id);
      bool isPublic = publicFound != publicById.end();
      if (isPublic &&
          publicFound->second.capacityBytes > requirement.capacity) {
        module.emitError("public binding exceeds physical allocation capacity");
        signalPassFailure();
        return;
      }
      SmallVector<NamedAttribute> fields = {
          builder.getNamedAttr("capacity_bytes",
                               builder.getI64IntegerAttr(requirement.capacity)),
          builder.getNamedAttr("format", requirement.format),
          builder.getNamedAttr("memory_flags", builder.getI64IntegerAttr(
                                                   requirement.memoryFlags)),
          builder.getNamedAttr(
              "persistent",
              builder.getBoolAttr(persistenceById[id] || isPublic)),
          builder.getNamedAttr("public", builder.getBoolAttr(isPublic)),
          builder.getNamedAttr("resource_id", builder.getI32IntegerAttr(id)),
          builder.getNamedAttr("resource_kind",
                               builder.getStringAttr("allocation")),
          builder.getNamedAttr("usage", requirement.usage),
      };
      if (isPublic) {
        const PublicBinding &binding = publicFound->second;
        fields.append({
            builder.getNamedAttr(
                "binding_capacity_bytes",
                builder.getI64IntegerAttr(binding.capacityBytes)),
            builder.getNamedAttr("data_type", binding.dataType),
            builder.getNamedAttr("logical_bytes", builder.getI64IntegerAttr(
                                                      binding.logicalBytes)),
            builder.getNamedAttr("name", binding.name),
            builder.getNamedAttr("public_index",
                                 builder.getI32IntegerAttr(binding.index)),
            builder.getNamedAttr("public_access", binding.access),
        });
      }
      output.push_back(builder.getDictionaryAttr(fields));
    }
    module->setAttr(
        inventoryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("discovery_schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("persistence_schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("requirements_schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("resources", builder.getArrayAttr(output)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
