#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MATERIALIZESTATICPHYSICALRESOLVESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral operandProvenanceAttribute =
    "vrc.static_device_operand_provenance";
constexpr StringLiteral inventoryAttribute =
    "vrc.static_device_resource_inventory";
constexpr StringLiteral texturesAttribute = "vrc.unity_texture_assignments";
constexpr StringLiteral summaryAttribute =
    "vrc.static_physical_resolve_materialization";

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
    module.emitError("static physical resolve materialization requires "
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

FailureOr<std::optional<int32_t>> resolveNonNullSingleton(func::CallOp call) {
  auto provenance = call->getAttrOfType<ArrayAttr>(operandProvenanceAttribute);
  if (!provenance)
    return std::optional<int32_t>();
  DictionaryAttr operand;
  for (Attribute attribute : provenance) {
    auto candidate = dyn_cast<DictionaryAttr>(attribute);
    auto index = candidate ? candidate.getAs<IntegerAttr>("index") : nullptr;
    if (index && index.getInt() == 0) {
      operand = candidate;
      break;
    }
  }
  if (!operand)
    return std::optional<int32_t>();
  auto cases = operand.getAs<ArrayAttr>("cases");
  if (!cases || cases.empty())
    return call.emitError("physical resolve provenance has no cases");

  std::optional<int32_t> identity;
  for (Attribute attribute : cases) {
    auto item = dyn_cast<DictionaryAttr>(attribute);
    auto unknown = item ? item.getAs<BoolAttr>("unknown") : nullptr;
    auto nullable = item ? item.getAs<BoolAttr>("may_be_null") : nullptr;
    auto ids = item ? item.getAs<ArrayAttr>("resource_ids") : nullptr;
    if (!item || !unknown || !nullable || !ids)
      return call.emitError("physical resolve provenance is invalid");
    if (unknown.getValue() || nullable.getValue() || ids.size() != 1)
      return std::optional<int32_t>();
    auto id = dyn_cast<IntegerAttr>(ids[0]);
    if (!id || id.getInt() < 0)
      return call.emitError("physical resolve resource identity is invalid");
    int32_t value = static_cast<int32_t>(id.getInt());
    if (identity && *identity != value)
      return std::optional<int32_t>();
    identity = value;
  }
  return identity;
}

bool isExactOutputCell(Value value) {
  auto allocation = value.getDefiningOp<::vrc::udonhl::AllocaOp>();
  if (!allocation || allocation.getAlignment() != 4)
    return false;
  auto bytes = allocation.getByteSize().getDefiningOp<arith::ConstantIntOp>();
  return bytes && bytes.value() == 4;
}

struct MaterializeStaticPhysicalResolvesPass
    : impl::MaterializeStaticPhysicalResolvesPassBase<
          MaterializeStaticPhysicalResolvesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    if (auto summary =
            module->getAttrOfType<DictionaryAttr>(summaryAttribute)) {
      auto materialized = summary.getAs<IntegerAttr>("materialized_count");
      auto unresolved = summary.getAs<IntegerAttr>("unresolved_count");
      if (!schemaV1(summary) || !materialized || materialized.getInt() < 0 ||
          !unresolved || unresolved.getInt() < 0) {
        module.emitError(
            "static physical resolve materialization summary is invalid");
        signalPassFailure();
      }
      return;
    }

    SmallVector<func::CallOp> resolves;
    module.walk([&](func::CallOp call) {
      if (call.getCallee() == "udonclResolvePhysicalBuffer")
        resolves.push_back(call);
    });
    if (resolves.empty()) {
      module->setAttr(summaryAttribute,
                      builder.getDictionaryAttr({
                          builder.getNamedAttr("materialized_count",
                                               builder.getI32IntegerAttr(0)),
                          builder.getNamedAttr("schema_version",
                                               builder.getI32IntegerAttr(1)),
                          builder.getNamedAttr("unresolved_count",
                                               builder.getI32IntegerAttr(0)),
                      }));
      return;
    }

    auto assignments = readAssignments(module);
    if (failed(assignments)) {
      signalPassFailure();
      return;
    }

    struct Candidate {
      func::CallOp call;
      int32_t resourceId;
    };
    SmallVector<Candidate> candidates;
    int32_t unresolved = 0;
    for (func::CallOp call : resolves) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      auto pointer = ::vrc::udonhl::PtrType::get(module.getContext(), 0);
      if (!callee || call.getNumOperands() != 2 || call.getNumResults() != 1 ||
          call.getOperand(0).getType() != pointer ||
          call.getOperand(1).getType() != pointer ||
          !call.getResult(0).getType().isInteger(32) ||
          callee.getFunctionType() !=
              builder.getFunctionType({pointer, pointer},
                                      {builder.getI32Type()})) {
        call.emitError("udonclResolvePhysicalBuffer has invalid signature");
        signalPassFailure();
        return;
      }
      auto identity = resolveNonNullSingleton(call);
      if (failed(identity)) {
        signalPassFailure();
        return;
      }
      if (!*identity || !isExactOutputCell(call.getOperand(1)) ||
          !assignments->contains(**identity)) {
        ++unresolved;
        continue;
      }
      candidates.push_back({call, **identity});
    }

    for (Candidate &candidate : candidates) {
      const Assignment &assignment = assignments->lookup(candidate.resourceId);
      OpBuilder opBuilder(candidate.call);
      OperationState state(candidate.call.getLoc(),
                           ::vrc::vrc::AssignedBufferOp::getOperationName());
      state.addTypes(::vrc::vrc::BufferType::get(module.getContext()));
      state.addAttribute("resource_id",
                         builder.getI32IntegerAttr(candidate.resourceId));
      state.addAttribute("texture_id",
                         builder.getI32IntegerAttr(assignment.textureId));
      state.addAttribute("capacity_bytes",
                         builder.getI64IntegerAttr(assignment.capacity));
      state.addAttribute("format", assignment.format);
      Value assigned = opBuilder.create(state)->getResult(0);
      auto token = UnrealizedConversionCastOp::create(
          opBuilder, candidate.call.getLoc(), TypeRange{builder.getI32Type()},
          ValueRange{assigned});
      ::vrc::udonhl::StoreOp::create(opBuilder, candidate.call.getLoc(),
                                     token.getResult(0),
                                     candidate.call.getOperand(1), 4);
      Value success = arith::ConstantIntOp::create(
          opBuilder, candidate.call.getLoc(), 0, 32);
      candidate.call.getResult(0).replaceAllUsesWith(success);
      candidate.call.erase();
    }

    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("materialized_count",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(candidates.size()))),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("unresolved_count",
                                 builder.getI32IntegerAttr(unresolved)),
        }));
  }
};

} // namespace
} // namespace vrc
