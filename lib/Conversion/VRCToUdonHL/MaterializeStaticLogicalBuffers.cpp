#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MATERIALIZESTATICLOGICALBUFFERSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral reconciliationAttribute =
    "vrc.static_unity_buffer_reconciliation";
constexpr StringLiteral layoutAttribute = "vrc.static_unity_buffer_layout_plan";
constexpr StringLiteral summaryAttribute =
    "vrc.static_logical_buffer_materialization";

struct LogicalBufferPlan {
  int32_t resourceId;
  int32_t textureId;
  int64_t capacity;
  StringAttr format;
};

bool hasSchemaVersion(DictionaryAttr attribute) {
  auto version =
      attribute ? attribute.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

FailureOr<DenseMap<int32_t, LogicalBufferPlan>> readPlans(ModuleOp module) {
  auto reconciliation =
      module->getAttrOfType<DictionaryAttr>(reconciliationAttribute);
  auto layout = module->getAttrOfType<DictionaryAttr>(layoutAttribute);
  auto reconciled =
      reconciliation ? reconciliation.getAs<ArrayAttr>("buffers") : nullptr;
  auto buffers = layout ? layout.getAs<ArrayAttr>("buffers") : nullptr;
  if (!hasSchemaVersion(reconciliation) || !reconciled ||
      !hasSchemaVersion(layout) || !buffers) {
    module.emitError("static logical buffer materialization requires "
                     "reconciliation and canonical layout metadata");
    return failure();
  }

  DenseMap<int32_t, int32_t> resourceByBuffer;
  for (Attribute attribute : reconciled) {
    auto entry = dyn_cast<DictionaryAttr>(attribute);
    auto resolution = entry ? entry.getAs<StringAttr>("resolution") : nullptr;
    if (!resolution || resolution != "static_alias")
      continue;
    auto bufferId = entry.getAs<IntegerAttr>("buffer_id");
    auto resourceId = entry.getAs<IntegerAttr>("resource_id");
    auto textureId = entry.getAs<IntegerAttr>("texture_id");
    if (!bufferId || bufferId.getInt() <= 0 || !resourceId ||
        resourceId.getInt() < 0 || !textureId || textureId.getInt() <= 0 ||
        resourceByBuffer.contains(static_cast<int32_t>(bufferId.getInt()))) {
      module.emitError("invalid static alias in Unity buffer reconciliation");
      return failure();
    }
    resourceByBuffer[static_cast<int32_t>(bufferId.getInt())] =
        static_cast<int32_t>(resourceId.getInt());
  }

  DenseMap<int32_t, LogicalBufferPlan> result;
  DenseSet<int32_t> layoutResourceIds;
  for (Attribute attribute : buffers) {
    auto entry = dyn_cast<DictionaryAttr>(attribute);
    auto bufferId = entry ? entry.getAs<IntegerAttr>("buffer_id") : nullptr;
    auto resourceId = entry ? entry.getAs<IntegerAttr>("resource_id") : nullptr;
    auto textureId = entry ? entry.getAs<IntegerAttr>("texture_id") : nullptr;
    auto capacity =
        entry ? entry.getAs<IntegerAttr>("capacity_bytes") : nullptr;
    auto format = entry ? entry.getAs<StringAttr>("format") : nullptr;
    if (!bufferId || bufferId.getInt() <= 0 || !resourceId ||
        resourceId.getInt() < 0 || !textureId || textureId.getInt() <= 0 ||
        !capacity || capacity.getInt() <= 0 || !format ||
        format != "rgba8_unorm" ||
        !layoutResourceIds.insert(static_cast<int32_t>(resourceId.getInt()))
             .second) {
      module.emitError("invalid buffer in canonical Unity layout");
      return failure();
    }
    auto alias = resourceByBuffer.find(static_cast<int32_t>(bufferId.getInt()));
    if (alias == resourceByBuffer.end())
      continue;
    if (alias->second != resourceId.getInt()) {
      module.emitError(
          "reconciled logical buffer does not match canonical Unity layout");
      return failure();
    }
    result[static_cast<int32_t>(bufferId.getInt())] = {
        static_cast<int32_t>(resourceId.getInt()),
        static_cast<int32_t>(textureId.getInt()), capacity.getInt(), format};
  }
  for (const auto &[bufferId, resourceId] : resourceByBuffer)
    if (!result.contains(bufferId)) {
      module.emitError("reconciled logical buffer has no canonical layout");
      return failure();
    }
  return result;
}

struct MaterializeStaticLogicalBuffersPass
    : impl::MaterializeStaticLogicalBuffersPassBase<
          MaterializeStaticLogicalBuffersPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(summaryAttribute);

    SmallVector<::vrc::vrc::BufferAllocOp> allocations;
    module.walk([&](::vrc::vrc::BufferAllocOp allocation) {
      allocations.push_back(allocation);
    });
    if (allocations.empty())
      return;

    auto plans = readPlans(module);
    if (failed(plans)) {
      signalPassFailure();
      return;
    }

    int32_t materializedCount = 0;
    int32_t unresolvedCount = 0;
    for (::vrc::vrc::BufferAllocOp allocation : allocations) {
      auto logicalId =
          allocation.getLogicalId().getDefiningOp<arith::ConstantIntOp>();
      auto byteSize =
          allocation.getByteSize().getDefiningOp<arith::ConstantIntOp>();
      if (!logicalId || logicalId.value() <= 0 || !byteSize ||
          byteSize.value() <= 0) {
        ++unresolvedCount;
        continue;
      }
      auto plan = plans->find(static_cast<int32_t>(logicalId.value()));
      if (plan == plans->end()) {
        ++unresolvedCount;
        continue;
      }
      if (allocation.getFormatAttr() != plan->second.format ||
          allocation.getUsage() != "storage" ||
          byteSize.value() > plan->second.capacity) {
        allocation.emitError(
            "logical buffer allocation does not match canonical Unity layout");
        signalPassFailure();
        return;
      }

      OpBuilder opBuilder(allocation);
      OperationState state(allocation.getLoc(),
                           ::vrc::vrc::AssignedBufferOp::getOperationName());
      state.addTypes(::vrc::vrc::BufferType::get(module.getContext()));
      state.addAttribute("resource_id",
                         builder.getI32IntegerAttr(plan->second.resourceId));
      state.addAttribute("texture_id",
                         builder.getI32IntegerAttr(plan->second.textureId));
      state.addAttribute("capacity_bytes",
                         builder.getI64IntegerAttr(plan->second.capacity));
      state.addAttribute("format", plan->second.format);
      Value assigned = opBuilder.create(state)->getResult(0);
      Value success = arith::ConstantIntOp::create(
          opBuilder, allocation.getLoc(), builder.getI32Type(), 0);
      allocation.getBuffer().replaceAllUsesWith(assigned);
      allocation.getStatus().replaceAllUsesWith(success);
      allocation.erase();
      ++materializedCount;
    }

    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("candidate_count",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(allocations.size()))),
            builder.getNamedAttr("materialized_count",
                                 builder.getI32IntegerAttr(materializedCount)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("unresolved_count",
                                 builder.getI32IntegerAttr(unresolvedCount)),
        }));
  }
};

} // namespace
} // namespace vrc
