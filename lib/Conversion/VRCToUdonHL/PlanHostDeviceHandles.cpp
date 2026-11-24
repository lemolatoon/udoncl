// Plans the OpenCL buffer handles of a Unity Host: one per static allocation
// context, with the capacity inferred for it.
#include "VRC/Conversion/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_PLANHOSTDEVICEHANDLESPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral requirementsAttribute =
    "vrc.static_device_resource_requirements";
constexpr StringLiteral planAttribute = "vrc.host_device_handle_plan";

struct PlanHostDeviceHandlesPass
    : impl::PlanHostDeviceHandlesPassBase<PlanHostDeviceHandlesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->removeAttr(planAttribute);
    if (failed(plan(module)))
      signalPassFailure();
  }

  LogicalResult plan(ModuleOp module) {
    auto discovery = module->getAttrOfType<DictionaryAttr>(discoveryAttribute);
    auto contexts =
        discovery ? discovery.getAs<ArrayAttr>("contexts") : nullptr;
    auto requirements =
        module->getAttrOfType<DictionaryAttr>(requirementsAttribute);
    auto resources =
        requirements ? requirements.getAs<ArrayAttr>("resources") : nullptr;
    if (!contexts || !resources)
      return module.emitError(
          "the Host device handle plan requires static allocation contexts "
          "and their requirements");
    llvm::DenseMap<int64_t, int64_t> capacities;
    for (Attribute attribute : resources) {
      auto resource = dyn_cast<DictionaryAttr>(attribute);
      auto id = resource ? resource.getAs<IntegerAttr>("resource_id") : nullptr;
      auto kind =
          resource ? resource.getAs<StringAttr>("resource_kind") : nullptr;
      auto capacity =
          resource ? resource.getAs<IntegerAttr>("capacity_bytes") : nullptr;
      if (!id || !kind || !capacity || kind != "allocation" ||
          capacity.getInt() <= 0 ||
          !capacities.try_emplace(id.getInt(), capacity.getInt()).second)
        return module.emitError("invalid static allocation requirement");
    }
    SmallVector<int64_t> bytes;
    for (Attribute attribute : contexts) {
      auto context = dyn_cast<DictionaryAttr>(attribute);
      auto id = context ? context.getAs<IntegerAttr>("context_id") : nullptr;
      auto capacity = id ? capacities.find(id.getInt()) : capacities.end();
      if (capacity == capacities.end())
        return module.emitError(
            "static allocation context has no inferred capacity");
      bytes.push_back(capacity->second);
    }
    Builder builder(module.getContext());
    module->setAttr(
        planAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("buffer_bytes",
                                 builder.getDenseI64ArrayAttr(bytes)),
            builder.getNamedAttr(
                "handles",
                builder.getI64IntegerAttr(static_cast<int64_t>(bytes.size()))),
        }));
    return success();
  }
};
} // namespace
} // namespace vrc
