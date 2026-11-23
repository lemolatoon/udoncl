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
#define GEN_PASS_DEF_RESOLVESTATICPUBLICRESOURCEALIASESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral inventoryAttribute =
    "vrc.static_device_resource_inventory";
constexpr StringLiteral assignmentsAttribute = "vrc.unity_texture_assignments";
constexpr StringLiteral aliasesAttribute = "vrc.static_public_resource_aliases";

struct Assignment {
  int32_t textureId;
  int64_t capacityBytes;
  StringAttr format;
  BoolAttr isPublic;
  StringAttr publicAccess;
};

struct Alias {
  StringAttr access;
  int32_t publicIndex;
  int32_t resourceId;
  int32_t textureId;
};

bool hasSchemaVersion(DictionaryAttr attribute) {
  auto version =
      attribute ? attribute.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

struct ResolveStaticPublicResourceAliasesPass
    : impl::ResolveStaticPublicResourceAliasesPassBase<
          ResolveStaticPublicResourceAliasesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(aliasesAttribute);

    auto inventory = module->getAttrOfType<DictionaryAttr>(inventoryAttribute);
    auto assignments =
        module->getAttrOfType<DictionaryAttr>(assignmentsAttribute);
    auto resources =
        inventory ? inventory.getAs<ArrayAttr>("resources") : nullptr;
    auto textures =
        assignments ? assignments.getAs<ArrayAttr>("assignments") : nullptr;
    if (!hasSchemaVersion(inventory) || !resources ||
        !hasSchemaVersion(assignments) || !textures) {
      module.emitError("static public resource alias resolution requires "
                       "inventory and Texture assignments");
      signalPassFailure();
      return;
    }

    DenseMap<int32_t, Assignment> assignmentByResource;
    DenseSet<int32_t> textureIds;
    for (Attribute attribute : textures) {
      auto assignment = dyn_cast<DictionaryAttr>(attribute);
      auto kind =
          assignment ? assignment.getAs<StringAttr>("resource_kind") : nullptr;
      if (!kind || kind != "allocation")
        continue;
      auto id = assignment.getAs<IntegerAttr>("resource_id");
      auto textureId = assignment.getAs<IntegerAttr>("texture_id");
      auto capacity = assignment.getAs<IntegerAttr>("capacity_bytes");
      auto format = assignment.getAs<StringAttr>("format");
      auto isPublic = assignment.getAs<BoolAttr>("public");
      auto publicAccess = assignment.getAs<StringAttr>("public_access");
      if (!id || id.getInt() < 0 || !textureId || textureId.getInt() <= 0 ||
          !capacity || capacity.getInt() <= 0 || !format ||
          format != "rgba8_unorm" || !isPublic ||
          (isPublic.getValue() &&
           (!publicAccess ||
            (publicAccess != "input" && publicAccess != "output" &&
             publicAccess != "inout"))) ||
          (!isPublic.getValue() && publicAccess) ||
          assignmentByResource.contains(static_cast<int32_t>(id.getInt())) ||
          !textureIds.insert(static_cast<int32_t>(textureId.getInt())).second) {
        module.emitError("invalid allocation Texture assignment for public "
                         "resource alias resolution");
        signalPassFailure();
        return;
      }
      assignmentByResource[static_cast<int32_t>(id.getInt())] = {
          static_cast<int32_t>(textureId.getInt()), capacity.getInt(), format,
          isPublic, publicAccess};
    }

    SmallVector<Alias> aliases;
    DenseSet<int32_t> inventoryResourceIds;
    for (Attribute attribute : resources) {
      auto resource = dyn_cast<DictionaryAttr>(attribute);
      auto kind =
          resource ? resource.getAs<StringAttr>("resource_kind") : nullptr;
      if (!kind || kind != "allocation")
        continue;
      auto id = resource.getAs<IntegerAttr>("resource_id");
      auto capacity = resource.getAs<IntegerAttr>("capacity_bytes");
      auto format = resource.getAs<StringAttr>("format");
      auto isPublic = resource.getAs<BoolAttr>("public");
      if (!id || id.getInt() < 0 || !capacity || capacity.getInt() <= 0 ||
          !format || format != "rgba8_unorm" || !isPublic ||
          !inventoryResourceIds.insert(static_cast<int32_t>(id.getInt()))
               .second) {
        module.emitError("invalid allocation in static resource inventory");
        signalPassFailure();
        return;
      }
      if (!isPublic.getValue())
        continue;
      auto access = resource.getAs<StringAttr>("public_access");
      auto index = resource.getAs<IntegerAttr>("public_index");
      int32_t resourceId = static_cast<int32_t>(id.getInt());
      if (!access ||
          (access != "input" && access != "output" && access != "inout") ||
          !index || index.getInt() < 0) {
        module.emitError(
            "invalid public endpoint in static resource inventory");
        signalPassFailure();
        return;
      }
      if (llvm::any_of(aliases, [&](const Alias &alias) {
            return alias.access == access &&
                   alias.publicIndex == index.getInt();
          })) {
        module.emitError("duplicate public Device endpoint");
        signalPassFailure();
        return;
      }
      auto found = assignmentByResource.find(resourceId);
      if (found == assignmentByResource.end()) {
        module.emitError("public allocation has no Texture assignment");
        signalPassFailure();
        return;
      }
      const Assignment &assignment = found->second;
      if (!assignment.isPublic.getValue() ||
          assignment.publicAccess != access ||
          assignment.capacityBytes != capacity.getInt() ||
          assignment.format != format) {
        module.emitError(
            "public allocation and Texture assignment are inconsistent");
        signalPassFailure();
        return;
      }
      aliases.push_back({access, static_cast<int32_t>(index.getInt()),
                         resourceId, assignment.textureId});
    }

    llvm::sort(aliases, [](const Alias &lhs, const Alias &rhs) {
      if (lhs.access != rhs.access)
        return lhs.access.getValue() < rhs.access.getValue();
      return lhs.publicIndex < rhs.publicIndex;
    });
    SmallVector<Attribute> output;
    for (const Alias &alias : aliases)
      output.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("public_access", alias.access),
          builder.getNamedAttr("public_index",
                               builder.getI32IntegerAttr(alias.publicIndex)),
          builder.getNamedAttr("resource_id",
                               builder.getI32IntegerAttr(alias.resourceId)),
          builder.getNamedAttr("resource_kind",
                               builder.getStringAttr("allocation")),
          builder.getNamedAttr("texture_id",
                               builder.getI32IntegerAttr(alias.textureId)),
      }));
    module->setAttr(
        aliasesAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("aliases", builder.getArrayAttr(output)),
            builder.getNamedAttr("resolved_count",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(aliases.size()))),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
