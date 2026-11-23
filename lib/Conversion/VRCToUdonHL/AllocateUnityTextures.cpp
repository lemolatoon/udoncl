#include "VRC/Conversion/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <set>
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ALLOCATEUNITYTEXTURESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral inventoryAttribute =
    "vrc.static_device_resource_inventory";
constexpr StringLiteral assignmentAttribute = "vrc.unity_texture_assignments";

struct AllocationResource {
  int32_t resourceId;
  int64_t capacityBytes;
  StringAttr format;
  BoolAttr isPublic;
  StringAttr publicAccess;
  BoolAttr persistent;
};

struct AllocateUnityTexturesPass
    : impl::AllocateUnityTexturesPassBase<AllocateUnityTexturesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(assignmentAttribute);

    auto inventory = module->getAttrOfType<DictionaryAttr>(inventoryAttribute);
    if (!inventory)
      return;
    auto schemaVersion = inventory.getAs<IntegerAttr>("schema_version");
    auto resources = inventory.getAs<ArrayAttr>("resources");
    if (!schemaVersion || schemaVersion.getInt() != 1 || !resources) {
      module.emitError("invalid static device resource inventory schema");
      signalPassFailure();
      return;
    }

    auto discovery = module->getAttrOfType<DictionaryAttr>(discoveryAttribute);
    auto contexts =
        discovery ? discovery.getAs<ArrayAttr>("contexts") : nullptr;
    llvm::DenseSet<int32_t> contextIds;
    if (!contexts) {
      module.emitError("Unity Texture allocation requires resource discovery");
      signalPassFailure();
      return;
    }
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

    std::set<std::pair<std::string, int32_t>> identities;
    SmallVector<AllocationResource> allocations;
    for (Attribute attribute : resources) {
      auto resource = dyn_cast<DictionaryAttr>(attribute);
      auto kind =
          resource ? resource.getAs<StringAttr>("resource_kind") : nullptr;
      auto id = resource ? resource.getAs<IntegerAttr>("resource_id") : nullptr;
      if (!kind || !id || id.getInt() < 0) {
        module.emitError("invalid keyed static resource identity");
        signalPassFailure();
        return;
      }
      auto identity = std::make_pair(kind.getValue().str(),
                                     static_cast<int32_t>(id.getInt()));
      if (!identities.insert(identity).second) {
        module.emitError("duplicate keyed static resource identity");
        signalPassFailure();
        return;
      }
      if (kind != "allocation")
        continue;
      if (!contextIds.contains(identity.second)) {
        module.emitError("allocation inventory references unknown context");
        signalPassFailure();
        return;
      }
      auto capacity = resource.getAs<IntegerAttr>("capacity_bytes");
      auto format = resource.getAs<StringAttr>("format");
      auto isPublic = resource.getAs<BoolAttr>("public");
      auto persistent = resource.getAs<BoolAttr>("persistent");
      auto publicAccess = resource.getAs<StringAttr>("public_access");
      if (!capacity || capacity.getInt() <= 0 || !format ||
          format != "rgba8_unorm" || !isPublic || !persistent ||
          (isPublic.getValue() &&
           (!publicAccess ||
            (publicAccess != "input" && publicAccess != "output" &&
             publicAccess != "inout"))) ||
          (!isPublic.getValue() && publicAccess)) {
        module.emitError("invalid allocation Texture requirements");
        signalPassFailure();
        return;
      }
      allocations.push_back({identity.second, capacity.getInt(), format,
                             isPublic, publicAccess, persistent});
    }

    llvm::sort(allocations, [](const AllocationResource &lhs,
                               const AllocationResource &rhs) {
      return lhs.resourceId < rhs.resourceId;
    });
    SmallVector<Attribute> assignments;
    for (auto [index, allocation] : llvm::enumerate(allocations)) {
      SmallVector<NamedAttribute> fields = {
          builder.getNamedAttr("capacity_bytes", builder.getI64IntegerAttr(
                                                     allocation.capacityBytes)),
          builder.getNamedAttr("format", allocation.format),
          builder.getNamedAttr("persistent", allocation.persistent),
          builder.getNamedAttr("public", allocation.isPublic),
          builder.getNamedAttr(
              "resource_id", builder.getI32IntegerAttr(allocation.resourceId)),
          builder.getNamedAttr("resource_kind",
                               builder.getStringAttr("allocation")),
          builder.getNamedAttr(
              "texture_id",
              builder.getI32IntegerAttr(static_cast<int32_t>(index + 1))),
      };
      if (allocation.publicAccess)
        fields.push_back(
            builder.getNamedAttr("public_access", allocation.publicAccess));
      assignments.push_back(builder.getDictionaryAttr(fields));
    }
    module->setAttr(assignmentAttribute,
                    builder.getDictionaryAttr({
                        builder.getNamedAttr("assignments",
                                             builder.getArrayAttr(assignments)),
                        builder.getNamedAttr("schema_version",
                                             builder.getI32IntegerAttr(1)),
                    }));
  }
};

} // namespace
} // namespace vrc
