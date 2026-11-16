#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_EXPOSEHOSTBACKENDLITERALUSESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral completionAttribute =
    "vrc.host_backend_literal_use_version";
constexpr StringLiteral snapshotAttribute =
    "vrc.static_kernel_binding_snapshot";
constexpr StringLiteral roleAttribute = "vrc.host_literal_role";
constexpr StringLiteral literalVersionAttribute =
    "vrc.host_literal_generated_version";
constexpr int32_t schemaVersion = 1;

struct StaticIndexPlan {
  int32_t position;
  int32_t value;
};

struct DispatchPlan {
  ::vrc::vrc::DispatchOp dispatch;
  SmallVector<StaticIndexPlan> indices;
};

std::optional<int32_t> traceStaticResourceId(Value resource) {
  llvm::DenseSet<Value> visited;
  while (visited.insert(resource).second) {
    if (auto buffer = resource.getDefiningOp<::vrc::vrc::StaticBufferOp>())
      return static_cast<int32_t>(buffer.getResourceId());
    auto cast = resource.getDefiningOp<UnrealizedConversionCastOp>();
    if (!cast || cast.getInputs().size() != 1)
      break;
    resource = cast.getInputs().front();
  }
  return std::nullopt;
}

LogicalResult validateGeneratedRepresentation(DispatchPlan &plan) {
  auto positions = plan.dispatch.getStaticResourcePositions();
  auto indices = plan.dispatch.getStaticResourceIndices();
  if (plan.indices.empty()) {
    if (!positions && indices.empty())
      return success();
    return plan.dispatch.emitError(
        "generated static resource index representation is stale");
  }
  if (!positions || positions->size() != plan.indices.size() ||
      indices.size() != plan.indices.size())
    return plan.dispatch.emitError(
        "generated static resource index representation is incomplete");
  for (auto [index, expected] : llvm::enumerate(plan.indices)) {
    if ((*positions)[index] != expected.position)
      return plan.dispatch.emitError(
          "generated static resource positions are stale");
    auto constant = indices[index].getDefiningOp<arith::ConstantIntOp>();
    auto role = constant ? constant->getAttrOfType<StringAttr>(roleAttribute)
                         : StringAttr();
    auto version =
        constant ? constant->getAttrOfType<IntegerAttr>(literalVersionAttribute)
                 : IntegerAttr();
    if (!constant || !constant.getType().isInteger(32) ||
        constant.value() != expected.value || role != "static_resource_index" ||
        !version || version.getInt() != schemaVersion ||
        constant->getBlock() != plan.dispatch->getBlock() ||
        !constant->hasOneUse() ||
        *constant->getUsers().begin() != plan.dispatch.getOperation())
      return plan.dispatch.emitError(
          "generated static resource index literal is stale");
  }
  Operation *previous = plan.dispatch->getPrevNode();
  for (size_t reverseIndex = plan.indices.size(); reverseIndex > 0;
       --reverseIndex) {
    auto constant =
        indices[reverseIndex - 1].getDefiningOp<arith::ConstantIntOp>();
    if (!constant || previous != constant.getOperation())
      return plan.dispatch.emitError(
          "generated static resource index literals are not canonical");
    previous = previous->getPrevNode();
  }
  return success();
}

FailureOr<DispatchPlan> planDispatch(::vrc::vrc::DispatchOp dispatch) {
  DispatchPlan plan{dispatch, {}};
  auto snapshot = dispatch->getAttrOfType<DictionaryAttr>(snapshotAttribute);
  if (!snapshot)
    return plan;
  auto version = snapshot.getAs<IntegerAttr>("schema_version");
  auto kernelId = snapshot.getAs<IntegerAttr>("kernel_id");
  auto resources = snapshot.getAs<ArrayAttr>("resources");
  auto scalars = snapshot.getAs<ArrayAttr>("scalars");
  auto compatibility =
      dispatch->getAttrOfType<ArrayAttr>("vrc.unity_static_tensor_ids");
  if (!version || version.getInt() != 1 || !kernelId ||
      kernelId != dispatch.getKernelIdAttr() || !resources || !scalars ||
      resources.size() != dispatch.getResources().size() ||
      (compatibility && compatibility.size() != resources.size()))
    return dispatch.emitError("invalid static kernel binding snapshot");

  llvm::DenseSet<int32_t> claimed;
  for (Attribute attribute : resources) {
    auto resource = dyn_cast<DictionaryAttr>(attribute);
    auto position =
        resource ? resource.getAs<IntegerAttr>("operand_index") : IntegerAttr();
    auto resourceId =
        resource ? resource.getAs<IntegerAttr>("resource_id") : IntegerAttr();
    auto kind = resource ? resource.getAs<StringAttr>("kind") : StringAttr();
    auto access =
        resource ? resource.getAs<StringAttr>("access") : StringAttr();
    if (!position || position.getInt() < 0 ||
        position.getInt() >= dispatch.getResources().size() || !resourceId ||
        resourceId.getInt() < 0 || resourceId.getInt() > INT32_MAX || !kind ||
        (kind != "static_tensor" && kind != "allocation" &&
         kind != "logical_buffer") ||
        !access ||
        access != dispatch.getResourceAccesses()[position.getInt()] ||
        !claimed.insert(static_cast<int32_t>(position.getInt())).second)
      return dispatch.emitError("invalid resource in static kernel snapshot");
    IntegerAttr compatibleId;
    if (compatibility) {
      compatibleId = dyn_cast<IntegerAttr>(compatibility[position.getInt()]);
      if (!compatibleId || !compatibleId.getType().isInteger(32) ||
          compatibleId.getInt() < -1)
        return dispatch.emitError(
            "invalid bound static tensor compatibility identity");
    }
    if (kind != "static_tensor") {
      if (compatibleId && compatibleId.getInt() != -1)
        return dispatch.emitError(
            "non-static resource has a bound static tensor identity");
      continue;
    }
    if (access != "read")
      return dispatch.emitError("static tensor dispatch resource must be read");
    auto traced =
        traceStaticResourceId(dispatch.getResources()[position.getInt()]);
    if (traced && *traced != resourceId.getInt())
      return dispatch.emitError(
          "static tensor snapshot resource id does not match its operand");
    if (compatibleId && compatibleId.getInt() != resourceId.getInt())
      return dispatch.emitError()
             << "static tensor snapshot resource id " << resourceId.getInt()
             << " does not match bound compatibility identity "
             << compatibleId.getInt() << " at resource " << position.getInt();
    if (!traced && !compatibleId)
      return dispatch.emitError(
          "static tensor snapshot has neither SSA nor bound provenance");
    plan.indices.push_back({static_cast<int32_t>(position.getInt()),
                            static_cast<int32_t>(resourceId.getInt())});
  }
  llvm::sort(plan.indices,
             [](const StaticIndexPlan &left, const StaticIndexPlan &right) {
               return left.position < right.position;
             });
  return plan;
}

struct ExposeHostBackendLiteralUsesPass
    : impl::ExposeHostBackendLiteralUsesPassBase<
          ExposeHostBackendLiteralUsesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto completed = module->getAttrOfType<IntegerAttr>(completionAttribute);
    if (completed && completed.getInt() != schemaVersion) {
      module.emitError("unsupported Host backend literal use version");
      signalPassFailure();
      return;
    }

    SmallVector<DispatchPlan> plans;
    WalkResult status = module.walk([&](::vrc::vrc::DispatchOp dispatch) {
      auto plan = planDispatch(dispatch);
      if (failed(plan))
        return WalkResult::interrupt();
      plans.push_back(std::move(*plan));
      return WalkResult::advance();
    });
    if (status.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    if (completed) {
      for (DispatchPlan &plan : plans)
        if (failed(validateGeneratedRepresentation(plan))) {
          signalPassFailure();
          return;
        }
      return;
    }

    for (DispatchPlan &plan : plans)
      if (!plan.dispatch.getStaticResourceIndices().empty() ||
          plan.dispatch.getStaticResourcePositions()) {
        plan.dispatch.emitError(
            "unversioned static resource index operands are not allowed");
        signalPassFailure();
        return;
      }

    Builder builder(module.getContext());
    for (DispatchPlan &plan : plans) {
      if (plan.indices.empty())
        continue;
      OpBuilder opBuilder(plan.dispatch);
      SmallVector<Value> values;
      SmallVector<int32_t> positions;
      for (const StaticIndexPlan &index : plan.indices) {
        auto constant = opBuilder.create<arith::ConstantIntOp>(
            plan.dispatch.getLoc(), index.value, 32);
        constant->setAttr(roleAttribute,
                          builder.getStringAttr("static_resource_index"));
        constant->setAttr(literalVersionAttribute,
                          builder.getI32IntegerAttr(schemaVersion));
        values.push_back(constant);
        positions.push_back(index.position);
      }
      plan.dispatch.getStaticResourceIndicesMutable().append(values);
      plan.dispatch->setAttr("static_resource_positions",
                             builder.getDenseI32ArrayAttr(positions));
    }
    module->setAttr(completionAttribute,
                    builder.getI32IntegerAttr(schemaVersion));
  }
};

} // namespace
} // namespace vrc
