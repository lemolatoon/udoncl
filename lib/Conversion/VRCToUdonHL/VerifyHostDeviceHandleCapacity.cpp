// Verifies that the OpenCL runtime of a Unity library has a buffer handle
// for every allocation its Host may make: each static allocation context is
// live at most once, so the runtime needs one handle per context.
#include "VRC/Conversion/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_VERIFYHOSTDEVICEHANDLECAPACITYPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
struct VerifyHostDeviceHandleCapacityPass
    : impl::VerifyHostDeviceHandleCapacityPassBase<
          VerifyHostDeviceHandleCapacityPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto capacity =
        module->getAttrOfType<IntegerAttr>("udoncl.mutable_handle_capacity");
    auto discovery = module->getAttrOfType<DictionaryAttr>(
        "vrc.static_device_resource_discovery");
    if (!capacity || !discovery)
      return;
    auto contexts = discovery.getAs<ArrayAttr>("contexts");
    if (!contexts) {
      module.emitError("static Device resource discovery has no contexts");
      return signalPassFailure();
    }
    int64_t required = static_cast<int64_t>(contexts.size());
    if (capacity.getInt() >= required)
      return;
    auto plan = module->getAttrOfType<DictionaryAttr>("vrc.host_buffer_plan");
    auto path = plan ? plan.getAs<StringAttr>("path") : nullptr;
    InFlightDiagnostic diagnostic =
        module.emitError() << "the OpenCL runtime has " << capacity.getInt()
                           << " buffer handles but the Host has " << required
                           << " allocation contexts, each of which needs one";
    if (path)
      diagnostic << " (Host buffer plan " << path.getValue() << ")";
    else
      diagnostic << " (no Host buffer plan)";
    signalPassFailure();
  }
};
} // namespace
} // namespace vrc
