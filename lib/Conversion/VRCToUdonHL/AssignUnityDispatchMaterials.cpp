// Give every Unity dispatch site its own kernel material.
//
// On the Unity target a vrc.dispatch is one Blit through a Material whose
// shader is the kernel; the kernel arguments are material properties and a
// material keeps them between Blits. When all dispatches of a kernel share
// one material, each of them rebinds every argument that differs from the
// previous dispatch of that kernel anywhere in the program, although most
// arguments of one dispatch site are the same on every run (constants, static
// tensors, statically planned buffers).
//
// The pass assigns each dispatch with a `vrc.unity_dispatch_material_slot`
// its own public slot `__udoncl_dispatch_material_<n>` in module order, and
// records every slot with its kernel id in `vrc.unity_dispatch_materials`.
// The scalar Udon lowering then binds a site's fixed arguments once and
// remembers the ones that change; the Unity importer creates one material per
// listed slot from the kernel's shader. A dispatch that later passes clone
// shares its material with the clone, which the lowering's per-material plan
// handles like any shared material.
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ASSIGNUNITYDISPATCHMATERIALSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
constexpr StringLiteral kMaterialSlotAttr = "vrc.unity_dispatch_material_slot";
constexpr StringLiteral kMaterialsAttr = "vrc.unity_dispatch_materials";

struct AssignUnityDispatchMaterialsPass
    : impl::AssignUnityDispatchMaterialsPassBase<
          AssignUnityDispatchMaterialsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    // Material slot -> kernel id, in first-use order.
    llvm::MapVector<StringAttr, int32_t> materials;
    int64_t assigned = 0;
    WalkResult result = module.walk([&](::vrc::vrc::DispatchOp dispatch) {
      auto slot = dispatch->getAttrOfType<StringAttr>(kMaterialSlotAttr);
      if (!slot || slot.getValue().empty())
        return WalkResult::advance();
      int32_t kernelId = static_cast<int32_t>(dispatch.getKernelId());
      if (maxMaterials <= 0 || assigned < maxMaterials) {
        slot = builder.getStringAttr("__udoncl_dispatch_material_" +
                                     std::to_string(assigned++));
        dispatch->setAttr(kMaterialSlotAttr, slot);
        ++numSiteMaterials;
      }
      auto [entry, inserted] = materials.try_emplace(slot, kernelId);
      if (!inserted && entry->second != kernelId) {
        dispatch.emitError()
            << "material slot " << slot.getValue() << " is shared by kernels "
            << entry->second << " and " << kernelId;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
    numSharedMaterials = materials.size() - static_cast<size_t>(assigned);
    SmallVector<Attribute> entries;
    for (auto [slot, kernelId] : materials)
      entries.push_back(builder.getDictionaryAttr(
          {builder.getNamedAttr("slot", slot),
           builder.getNamedAttr("kernel_id",
                                builder.getI32IntegerAttr(kernelId))}));
    module->setAttr(kMaterialsAttr, builder.getArrayAttr(entries));
  }
};
} // namespace
} // namespace vrc
