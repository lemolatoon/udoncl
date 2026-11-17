#include "ONNXHostGraph.h"
#include "ONNXHostMaterialization.h"
#include "ONNXHostPlan.h"
#include "VRC/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_CLEANUPONNXHOSTSOURCEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
struct CleanupONNXHostSourcePass
    : impl::CleanupONNXHostSourcePassBase<CleanupONNXHostSourcePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(onnxHostSourceCleanedAttr)) {
      module.emitError("ONNX Host source has already been cleaned");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(onnxHostFinalizedAttr)) {
      module.emitError("ONNX Host source cleanup requires finalization");
      signalPassFailure();
      return;
    }

    llvm::SmallPtrSet<Operation *, 16> sources;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function.getSymName() == "main_graph" ||
          function->hasAttr(onnxHostSourceAttr) ||
          (function->hasAttr("udon.onnx.operation") &&
           function->hasAttr("udon.onnx.provider")) ||
          function->hasAttr("udoncl.static_tensor"))
        sources.insert(function);
    }
    for (Operation *source : sources) {
      auto uses = SymbolTable::getSymbolUses(source, module);
      if (!uses) {
        source->emitError("cannot prove ONNX Host source symbol uses");
        signalPassFailure();
        return;
      }
      for (const SymbolTable::SymbolUse &use : *uses) {
        bool removedWithSource = false;
        for (Operation *owner = use.getUser(); owner && owner != module;
             owner = owner->getParentOp())
          if (sources.contains(owner)) {
            removedWithSource = true;
            break;
          }
        if (!removedWithSource) {
          auto diagnostic = source->emitError()
                            << "ONNX Host source symbol remains referenced by "
                            << use.getUser()->getName();
          if (auto function = use.getUser()->getParentOfType<func::FuncOp>())
            diagnostic << " in @" << function.getSymName();
          signalPassFailure();
          return;
        }
      }
    }
    for (Operation *source : sources)
      source->erase();
    module->removeAttr(hostGraphValidatedAttr);
    module->removeAttr(hostGraphPlanAttr);
    module->removeAttr(onnxHostGlobalsMaterializedAttr);
    module->removeAttr(onnxHostModelRegistryMaterializedAttr);
    module->setAttr(onnxHostSourceCleanedAttr,
                    UnitAttr::get(module.getContext()));
  }
};
} // namespace
} // namespace vrc
