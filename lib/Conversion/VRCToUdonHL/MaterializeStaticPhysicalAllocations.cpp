#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MATERIALIZESTATICPHYSICALALLOCATIONSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral specializationAttribute =
    "vrc.static_physical_allocation_specialization";
constexpr StringLiteral assignmentAttribute =
    "vrc.static_physical_allocation_assignment";
constexpr StringLiteral inventoryAttribute =
    "vrc.static_device_resource_inventory";
constexpr StringLiteral texturesAttribute = "vrc.unity_texture_assignments";
constexpr StringLiteral summaryAttribute =
    "vrc.static_physical_allocation_materialization";

struct Assignment {
  int64_t capacity;
  int32_t textureId;
  StringAttr format;
};

bool schemaV1(DictionaryAttr value) {
  auto version = value ? value.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

FailureOr<DenseMap<int32_t, Assignment>> readAssignments(ModuleOp module) {
  auto inventory = module->getAttrOfType<DictionaryAttr>(inventoryAttribute);
  auto textures = module->getAttrOfType<DictionaryAttr>(texturesAttribute);
  auto resources =
      inventory ? inventory.getAs<ArrayAttr>("resources") : nullptr;
  auto assignments =
      textures ? textures.getAs<ArrayAttr>("assignments") : nullptr;
  if (!schemaV1(inventory) || !resources || !schemaV1(textures) ||
      !assignments) {
    module.emitError("static physical allocation materialization requires "
                     "inventory and Texture assignments");
    return failure();
  }
  DenseMap<int32_t, std::pair<int64_t, StringAttr>> requirements;
  for (Attribute attribute : resources) {
    auto item = dyn_cast<DictionaryAttr>(attribute);
    auto kind = item ? item.getAs<StringAttr>("resource_kind") : nullptr;
    if (!kind || kind != "allocation")
      continue;
    auto id = item.getAs<IntegerAttr>("resource_id");
    auto capacity = item.getAs<IntegerAttr>("capacity_bytes");
    auto format = item.getAs<StringAttr>("format");
    auto persistent = item.getAs<BoolAttr>("persistent");
    if (!id || id.getInt() < 0 || !capacity || capacity.getInt() <= 0 ||
        !format || format.empty() || !persistent) {
      module.emitError("invalid allocation inventory entry");
      return failure();
    }
    if (!persistent.getValue())
      continue;
    requirements[static_cast<int32_t>(id.getInt())] = {capacity.getInt(),
                                                       format};
  }
  DenseMap<int32_t, Assignment> result;
  for (Attribute attribute : assignments) {
    auto item = dyn_cast<DictionaryAttr>(attribute);
    auto kind = item ? item.getAs<StringAttr>("resource_kind") : nullptr;
    if (!kind || kind != "allocation")
      continue;
    auto id = item.getAs<IntegerAttr>("resource_id");
    auto texture = item.getAs<IntegerAttr>("texture_id");
    auto capacity = item.getAs<IntegerAttr>("capacity_bytes");
    auto format = item.getAs<StringAttr>("format");
    auto persistent = item.getAs<BoolAttr>("persistent");
    auto requirement = id ? requirements.find(static_cast<int32_t>(id.getInt()))
                          : requirements.end();
    if (!id || id.getInt() < 0 || !texture || texture.getInt() <= 0 ||
        !capacity || capacity.getInt() <= 0 || !format || format.empty() ||
        !persistent) {
      module.emitError("invalid allocation Texture assignment");
      return failure();
    }
    if (!persistent.getValue())
      continue;
    if (requirement == requirements.end() ||
        requirement->second.first != capacity.getInt() ||
        requirement->second.second != format ||
        result.contains(static_cast<int32_t>(id.getInt()))) {
      module.emitError(
          "Texture assignment does not match persistent allocation inventory");
      return failure();
    }
    result[static_cast<int32_t>(id.getInt())] = {
        capacity.getInt(), static_cast<int32_t>(texture.getInt()), format};
  }
  return result;
}

struct MaterializeStaticPhysicalAllocationsPass
    : impl::MaterializeStaticPhysicalAllocationsPassBase<
          MaterializeStaticPhysicalAllocationsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    if (auto summary =
            module->getAttrOfType<DictionaryAttr>(summaryAttribute)) {
      if (!schemaV1(summary)) {
        module.emitError(
            "static physical allocation materialization summary is invalid");
        signalPassFailure();
      }
      return;
    }
    auto specialization =
        module->getAttrOfType<DictionaryAttr>(specializationAttribute);
    auto expected =
        specialization
            ? specialization.getAs<IntegerAttr>("annotated_sink_count")
            : nullptr;
    auto contexts =
        specialization
            ? specialization.getAs<IntegerAttr>("specialized_context_count")
            : nullptr;
    if (!schemaV1(specialization) || !expected || expected.getInt() < 0 ||
        !contexts || contexts.getInt() != expected.getInt()) {
      module.emitError("static physical allocation materialization requires "
                       "specialization schema v1");
      signalPassFailure();
      return;
    }
    auto assignments = readAssignments(module);
    if (failed(assignments)) {
      signalPassFailure();
      return;
    }

    SmallVector<::vrc::vrc::BufferAllocOp> sinks;
    module.walk([&](::vrc::vrc::BufferAllocOp allocation) {
      if (allocation->hasAttr(assignmentAttribute))
        sinks.push_back(allocation);
    });
    if (sinks.size() != static_cast<size_t>(expected.getInt())) {
      module.emitError(
          "annotated physical allocation count does not match specialization");
      signalPassFailure();
      return;
    }
    DenseSet<int32_t> resources;
    for (::vrc::vrc::BufferAllocOp allocation : sinks) {
      auto annotation =
          allocation->getAttrOfType<DictionaryAttr>(assignmentAttribute);
      auto resource =
          annotation ? annotation.getAs<IntegerAttr>("resource_id") : nullptr;
      auto texture =
          annotation ? annotation.getAs<IntegerAttr>("texture_id") : nullptr;
      auto capacity = annotation
                          ? annotation.getAs<IntegerAttr>("capacity_bytes")
                          : nullptr;
      auto format =
          annotation ? annotation.getAs<StringAttr>("format") : nullptr;
      auto version = annotation
                         ? annotation.getAs<IntegerAttr>("schema_version")
                         : nullptr;
      auto expectedAssignment =
          resource ? assignments->find(static_cast<int32_t>(resource.getInt()))
                   : assignments->end();
      if (!version || version.getInt() != 1 || !resource ||
          resource.getInt() < 0 || !texture || texture.getInt() <= 0 ||
          !capacity || capacity.getInt() <= 0 || !format ||
          expectedAssignment == assignments->end() ||
          expectedAssignment->second.textureId != texture.getInt() ||
          expectedAssignment->second.capacity != capacity.getInt() ||
          expectedAssignment->second.format != format ||
          allocation.getFormat() != format ||
          allocation.getUsage() != "storage" ||
          !resources.insert(static_cast<int32_t>(resource.getInt())).second) {
        allocation.emitError("static physical assignment is inconsistent");
        signalPassFailure();
        return;
      }
      if (auto constant =
              allocation.getByteSize().getDefiningOp<arith::ConstantIntOp>();
          constant && constant.value() != capacity.getInt()) {
        allocation.emitError(
            "constant physical allocation size does not match assignment");
        signalPassFailure();
        return;
      }
    }

    for (::vrc::vrc::BufferAllocOp allocation : sinks) {
      auto annotation =
          allocation->getAttrOfType<DictionaryAttr>(assignmentAttribute);
      OpBuilder opBuilder(allocation);
      OperationState state(allocation.getLoc(),
                           ::vrc::vrc::AssignedBufferOp::getOperationName());
      state.addTypes(::vrc::vrc::BufferType::get(module.getContext()));
      state.addAttribute("resource_id",
                         annotation.getAs<IntegerAttr>("resource_id"));
      state.addAttribute("texture_id",
                         annotation.getAs<IntegerAttr>("texture_id"));
      state.addAttribute("capacity_bytes",
                         annotation.getAs<IntegerAttr>("capacity_bytes"));
      state.addAttribute("format", annotation.getAs<StringAttr>("format"));
      Value assigned = opBuilder.create(state)->getResult(0);
      Value success =
          arith::ConstantIntOp::create(opBuilder, allocation.getLoc(), 0, 32);
      allocation.getBuffer().replaceAllUsesWith(assigned);
      allocation.getStatus().replaceAllUsesWith(success);
      allocation.erase();
    }
    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr(
                "materialized_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(sinks.size()))),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
