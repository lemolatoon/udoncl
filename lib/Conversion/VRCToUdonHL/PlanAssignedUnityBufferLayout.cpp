// Plan the Unity buffers of a Host whose Device buffers are already
// compiler-assigned.
//
// A Host that reaches the Unity library compiler with `vrc.assigned_buffer`
// operations instead of Device buffer allocations (for example hand-written
// Host MLIR) has no allocation for the static Device resource analysis to
// assess. Its buffers are exactly the textures those operations name, so
// this pass records them as an applied static lowering with a canonical
// buffer layout, from which udonclc derives the Unity buffer list.
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <map>
#include <set>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_PLANASSIGNEDUNITYBUFFERLAYOUTPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral readinessAttribute =
    "vrc.static_unity_resource_lowering_readiness";
constexpr StringLiteral layoutAttribute = "vrc.static_unity_buffer_layout_plan";

struct PlanAssignedUnityBufferLayoutPass
    : impl::PlanAssignedUnityBufferLayoutPassBase<
          PlanAssignedUnityBufferLayoutPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // The static Device resource analysis already applied its own layout,
    // unless it found no allocation to lay out.
    if (auto readiness =
            module->getAttrOfType<DictionaryAttr>(readinessAttribute)) {
      auto applied = readiness.getAs<BoolAttr>("applied");
      auto layout = module->getAttrOfType<DictionaryAttr>(layoutAttribute);
      auto planned = layout ? layout.getAs<ArrayAttr>("buffers") : nullptr;
      if (applied && applied.getValue() && (!planned || !planned.empty()))
        return;
    }
    // A Host that still allocates Device buffers is not assigned yet.
    bool allocates = false;
    module.walk([&](func::CallOp call) {
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      auto intrinsic =
          callee ? callee->getAttrOfType<StringAttr>("vrc.intrinsic") : nullptr;
      allocates |= intrinsic && intrinsic.getValue() == "buffer_alloc";
    });
    if (allocates)
      return;
    std::map<int32_t, ::vrc::vrc::AssignedBufferOp> textures;
    std::set<int32_t> resources;
    WalkResult walked = module.walk([&](::vrc::vrc::AssignedBufferOp buffer) {
      int32_t texture = buffer.getTextureId();
      auto [found, inserted] = textures.try_emplace(texture, buffer);
      if (inserted) {
        if (!resources.insert(buffer.getResourceId()).second) {
          buffer.emitError("assigned Unity buffers of textures ")
              << texture << " and another share resource "
              << buffer.getResourceId();
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      ::vrc::vrc::AssignedBufferOp first = found->second;
      if (first.getResourceId() != buffer.getResourceId() ||
          first.getCapacityBytes() != buffer.getCapacityBytes() ||
          first.getFormat() != buffer.getFormat()) {
        buffer.emitError("assigned Unity buffer texture ")
            << texture << " has conflicting resource, capacity or format";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walked.wasInterrupted())
      return signalPassFailure();
    if (textures.empty())
      return;
    Builder builder(module.getContext());
    SmallVector<Attribute> buffers;
    int32_t expected = 1;
    for (auto &[texture, buffer] : textures) {
      if (texture != expected++) {
        buffer.emitError("assigned Unity buffer textures must be numbered "
                         "densely from 1; texture ")
            << texture << " has no predecessor";
        return signalPassFailure();
      }
      if (buffer.getFormat() != "rgba8_unorm") {
        buffer.emitError("assigned Unity buffer texture ")
            << texture << " must be rgba8_unorm";
        return signalPassFailure();
      }
      buffers.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("buffer_id", builder.getI32IntegerAttr(texture)),
          builder.getNamedAttr("capacity_bytes", buffer.getCapacityBytesAttr()),
          builder.getNamedAttr("format", buffer.getFormatAttr()),
          builder.getNamedAttr("heap_slot",
                               builder.getStringAttr("__udoncl_buffer_" +
                                                     std::to_string(texture))),
          builder.getNamedAttr("persistent", builder.getBoolAttr(true)),
          builder.getNamedAttr("public", builder.getBoolAttr(false)),
          builder.getNamedAttr("resource_id", buffer.getResourceIdAttr()),
          builder.getNamedAttr("texture_id", buffer.getTextureIdAttr()),
      }));
    }
    module->setAttr(
        layoutAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("buffers", builder.getArrayAttr(buffers)),
            builder.getNamedAttr("ready", builder.getBoolAttr(true)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
    module->setAttr(
        readinessAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("applied", builder.getBoolAttr(true)),
            builder.getNamedAttr("pool_replacement_ready",
                                 builder.getBoolAttr(true)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
