// Keep the Device buffers of a generated ONNX binding private.
//
// The ONNX Host finalizer marks every buffer its generated binding allocates
// for a graph input or output `vrc.static_public_resource_binding`, so the
// static Device resource analysis can expose it as a public buffer of the
// Unity library. A library that does not expose its ONNX tensors (the
// transitional ONNX_PUBLIC_TENSOR_BUFFERS setting is off) runs this pass
// before the analysis: the buffers stay private instead of becoming public
// buffers that no Unity buffer backs.
#include "VRC/Conversion/Passes.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_PRIVATIZEONNXBINDINGRESOURCESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

struct PrivatizeONNXBindingResourcesPass
    : impl::PrivatizeONNXBindingResourcesPassBase<
          PrivatizeONNXBindingResourcesPass> {
  void runOnOperation() override {
    getOperation().walk([](Operation *operation) {
      operation->removeAttr("vrc.static_public_resource_binding");
    });
  }
};

} // namespace
} // namespace vrc
