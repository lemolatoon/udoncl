#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ANALYZEHOSTSTORAGEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

struct AnalyzeHostStoragePass
    : impl::AnalyzeHostStoragePassBase<AnalyzeHostStoragePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (function->hasAttr("vrc.async_frame_id"))
        functions.push_back(function);
    llvm::sort(functions, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });

    SmallVector<Attribute> placements;
    bool invalid = false;
    auto appendPlan = [&](func::FuncOp function, StringRef planName,
                          StringRef classification) {
      auto plan = function->getAttrOfType<ArrayAttr>(planName);
      if (!plan)
        return;
      for (Attribute attribute : plan) {
        auto source = dyn_cast<DictionaryAttr>(attribute);
        if (!source) {
          function.emitError()
              << planName << " requires dictionary placement entries";
          invalid = true;
          return;
        }
        NamedAttrList fields(source);
        fields.set("function", builder.getStringAttr(function.getSymName()));
        fields.set("classification", builder.getStringAttr(classification));
        // This pass describes the current representation without blessing it
        // as the target allocator policy. Later allocation passes replace
        // legacy_raw with direct_slot or typed_spill_array.
        if (!source.getAs<StringAttr>("placement"))
          fields.set("placement", builder.getStringAttr("legacy_raw"));
        placements.push_back(builder.getDictionaryAttr(fields));
      }
    };

    for (func::FuncOp function : functions) {
      appendPlan(function, "vrc.async_spill_slots", "async_persistent");
      appendPlan(function, "vrc.async_objects", "async_object");
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    module->setAttr("vrc.host_storage_placements",
                    builder.getArrayAttr(placements));
    module->setAttr("vrc.host_storage_placement_version",
                    builder.getI32IntegerAttr(1));
  }
};

} // namespace
} // namespace vrc
