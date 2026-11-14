#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_PLANSTATICPHYSICALALLOCATIONSPECIALIZATIONPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral allocationSiteAttribute =
    "vrc.static_device_allocation_site";
constexpr StringLiteral resultProvenanceAttribute =
    "vrc.static_device_result_provenance";
constexpr StringLiteral inventoryAttribute =
    "vrc.static_device_resource_inventory";
constexpr StringLiteral assignmentsAttribute = "vrc.unity_texture_assignments";
constexpr StringLiteral planAttribute =
    "vrc.static_physical_allocation_specialization_plan";

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
  auto assignments =
      module->getAttrOfType<DictionaryAttr>(assignmentsAttribute);
  auto resources =
      inventory ? inventory.getAs<ArrayAttr>("resources") : nullptr;
  auto textures =
      assignments ? assignments.getAs<ArrayAttr>("assignments") : nullptr;
  if (!schemaV1(inventory) || !resources || !schemaV1(assignments) ||
      !textures) {
    module.emitError("static physical allocation planning requires keyed "
                     "inventory and Texture assignments");
    return failure();
  }

  DenseMap<int32_t, std::pair<int64_t, StringAttr>> requirements;
  for (Attribute attribute : resources) {
    auto resource = dyn_cast<DictionaryAttr>(attribute);
    auto kind =
        resource ? resource.getAs<StringAttr>("resource_kind") : nullptr;
    if (!kind || kind != "allocation")
      continue;
    auto id = resource.getAs<IntegerAttr>("resource_id");
    auto capacity = resource.getAs<IntegerAttr>("capacity_bytes");
    auto format = resource.getAs<StringAttr>("format");
    auto persistent = resource.getAs<BoolAttr>("persistent");
    if (!id || id.getInt() < 0 || !capacity || capacity.getInt() <= 0 ||
        !format || format.empty() || !persistent ||
        requirements.contains(static_cast<int32_t>(id.getInt()))) {
      module.emitError("invalid allocation inventory entry");
      return failure();
    }
    if (!persistent.getValue())
      continue;
    requirements[static_cast<int32_t>(id.getInt())] = {capacity.getInt(),
                                                       format};
  }

  DenseMap<int32_t, Assignment> result;
  DenseSet<int32_t> textureIds;
  for (Attribute attribute : textures) {
    auto texture = dyn_cast<DictionaryAttr>(attribute);
    auto kind = texture ? texture.getAs<StringAttr>("resource_kind") : nullptr;
    if (!kind || kind != "allocation")
      continue;
    auto id = texture.getAs<IntegerAttr>("resource_id");
    auto textureId = texture.getAs<IntegerAttr>("texture_id");
    auto capacity = texture.getAs<IntegerAttr>("capacity_bytes");
    auto format = texture.getAs<StringAttr>("format");
    auto persistent = texture.getAs<BoolAttr>("persistent");
    auto requirement = id ? requirements.find(static_cast<int32_t>(id.getInt()))
                          : requirements.end();
    if (!id || id.getInt() < 0 || !textureId || textureId.getInt() <= 0 ||
        !capacity || !format || !persistent) {
      module.emitError("invalid allocation Texture assignment");
      return failure();
    }
    if (!persistent.getValue())
      continue;
    if (requirement == requirements.end() ||
        requirement->second.first != capacity.getInt() ||
        requirement->second.second != format ||
        result.contains(static_cast<int32_t>(id.getInt())) ||
        !textureIds.insert(static_cast<int32_t>(textureId.getInt())).second) {
      module.emitError(
          "Texture assignment does not match persistent allocation inventory");
      return failure();
    }
    result[static_cast<int32_t>(id.getInt())] = {
        capacity.getInt(), static_cast<int32_t>(textureId.getInt()), format};
  }
  return result;
}

std::optional<int32_t> singletonResource(func::CallOp call) {
  auto provenance = call->getAttrOfType<ArrayAttr>(resultProvenanceAttribute);
  if (!provenance)
    return std::nullopt;
  DictionaryAttr result;
  for (Attribute attribute : provenance) {
    auto candidate = dyn_cast<DictionaryAttr>(attribute);
    auto index = candidate ? candidate.getAs<IntegerAttr>("index") : nullptr;
    if (index && index.getInt() == 0) {
      result = candidate;
      break;
    }
  }
  auto cases = result ? result.getAs<ArrayAttr>("cases") : nullptr;
  if (!cases || cases.empty())
    return std::nullopt;
  std::optional<int32_t> resource;
  for (Attribute attribute : cases) {
    auto item = dyn_cast<DictionaryAttr>(attribute);
    auto unknown = item ? item.getAs<BoolAttr>("unknown") : nullptr;
    auto ids = item ? item.getAs<ArrayAttr>("resource_ids") : nullptr;
    if (!unknown || unknown.getValue() || !ids || ids.size() != 1)
      return std::nullopt;
    auto id = dyn_cast<IntegerAttr>(ids[0]);
    if (!id || id.getInt() < 0)
      return std::nullopt;
    int32_t value = static_cast<int32_t>(id.getInt());
    if (resource && *resource != value)
      return std::nullopt;
    resource = value;
  }
  return resource;
}

int32_t callOccurrence(func::CallOp target) {
  auto function = target->getParentOfType<func::FuncOp>();
  int32_t occurrence = 0;
  WalkResult result = function.walk([&](func::CallOp call) {
    if (call == target)
      return WalkResult::interrupt();
    if (call.getCallee() == target.getCallee())
      ++occurrence;
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? occurrence : -1;
}

DictionaryAttr frame(Builder &builder, func::CallOp call) {
  auto caller = call->getParentOfType<func::FuncOp>();
  return builder.getDictionaryAttr({
      builder.getNamedAttr("callee", call.getCalleeAttr()),
      builder.getNamedAttr(
          "caller",
          FlatSymbolRefAttr::get(builder.getContext(), caller.getSymName())),
      builder.getNamedAttr("occurrence",
                           builder.getI32IntegerAttr(callOccurrence(call))),
  });
}

void findAllocationPaths(func::FuncOp function,
                         SmallVector<DictionaryAttr> &path,
                         SmallVector<SmallVector<DictionaryAttr>> &paths,
                         DenseSet<Operation *> &active, bool &unsafe,
                         Builder &builder) {
  if (!active.insert(function).second) {
    unsafe = true;
    return;
  }
  SmallVector<::vrc::vrc::BufferAllocOp> allocations;
  function.walk([&](::vrc::vrc::BufferAllocOp allocation) {
    allocations.push_back(allocation);
  });
  if (!allocations.empty()) {
    if (allocations.size() != 1)
      unsafe = true;
    else
      paths.push_back(path);
    active.erase(function);
    return;
  }

  function.walk([&](func::CallOp call) {
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (!callee || callee.isExternal())
      return;
    if (!callee.isPrivate() || callee->hasAttr("udoncl.export")) {
      unsafe = true;
      return;
    }
    path.push_back(frame(builder, call));
    findAllocationPaths(callee, path, paths, active, unsafe, builder);
    path.pop_back();
  });
  active.erase(function);
}

struct PlanStaticPhysicalAllocationSpecializationPass
    : impl::PlanStaticPhysicalAllocationSpecializationPassBase<
          PlanStaticPhysicalAllocationSpecializationPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(planAttribute);

    auto assignments = readAssignments(module);
    if (failed(assignments)) {
      signalPassFailure();
      return;
    }

    int32_t candidates = 0;
    int32_t blockers = 0;
    DenseSet<int32_t> plannedResources;
    SmallVector<Attribute> contexts;
    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp call) {
      if (call->hasAttr(allocationSiteAttribute))
        calls.push_back(call);
    });
    for (func::CallOp call : calls) {
      ++candidates;
      auto site = call->getAttrOfType<DictionaryAttr>(allocationSiteAttribute);
      auto function =
          site ? site.getAs<FlatSymbolRefAttr>("function") : nullptr;
      auto occurrence = site ? site.getAs<IntegerAttr>("occurrence") : nullptr;
      auto primitive = site ? site.getAs<StringAttr>("primitive") : nullptr;
      auto caller = call->getParentOfType<func::FuncOp>();
      auto resource = singletonResource(call);
      if (!function || function.getValue() != caller.getSymName() ||
          !occurrence || occurrence.getInt() < 0 || !primitive ||
          occurrence.getInt() != callOccurrence(call) ||
          primitive.getValue() != call.getCallee() || !resource ||
          !assignments->contains(*resource) ||
          !plannedResources.insert(*resource).second) {
        ++blockers;
        continue;
      }
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.isExternal() || !callee.isPrivate() ||
          callee->hasAttr("udoncl.export")) {
        ++blockers;
        continue;
      }

      SmallVector<DictionaryAttr> path{frame(builder, call)};
      SmallVector<SmallVector<DictionaryAttr>> paths;
      DenseSet<Operation *> active;
      bool unsafe = false;
      findAllocationPaths(callee, path, paths, active, unsafe, builder);
      if (unsafe || paths.size() != 1) {
        ++blockers;
        continue;
      }
      const Assignment &assignment = assignments->lookup(*resource);
      SmallVector<Attribute> pathAttributes(paths.front().begin(),
                                            paths.front().end());
      contexts.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("call_path",
                               builder.getArrayAttr(pathAttributes)),
          builder.getNamedAttr("capacity_bytes",
                               builder.getI64IntegerAttr(assignment.capacity)),
          builder.getNamedAttr("format", assignment.format),
          builder.getNamedAttr("resource_id",
                               builder.getI32IntegerAttr(*resource)),
          builder.getNamedAttr("texture_id",
                               builder.getI32IntegerAttr(assignment.textureId)),
      }));
    }

    llvm::sort(contexts, [](Attribute lhs, Attribute rhs) {
      return cast<DictionaryAttr>(lhs)
                 .getAs<IntegerAttr>("resource_id")
                 .getInt() < cast<DictionaryAttr>(rhs)
                                 .getAs<IntegerAttr>("resource_id")
                                 .getInt();
    });
    module->setAttr(
        planAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("blocker_count",
                                 builder.getI32IntegerAttr(blockers)),
            builder.getNamedAttr("candidate_count",
                                 builder.getI32IntegerAttr(candidates)),
            builder.getNamedAttr("contexts", builder.getArrayAttr(contexts)),
            builder.getNamedAttr("planned_count",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(contexts.size()))),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
