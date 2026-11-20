#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ELIMINATEDEADPHYSICALALLOCATIONFALLBACKSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

bool isRawTransfer(func::CallOp call) {
  return call.getCallee() == "clEnqueueWriteBuffer" ||
         call.getCallee() == "clEnqueueReadBuffer" ||
         call.getCallee() == "clEnqueueCopyBuffer" ||
         call.getCallee() == "clEnqueueFillBuffer";
}

bool containsProtectedWork(func::FuncOp function) {
  bool protectedWork = false;
  function.walk([&](Operation *operation) {
    protectedWork |= isa<::vrc::vrc::DispatchOp, ::vrc::vrc::BufferUploadOp,
                         ::vrc::vrc::BufferReadbackOp, ::vrc::vrc::BufferCopyOp,
                         ::vrc::vrc::BufferFillOp>(operation);
    if (auto call = dyn_cast<func::CallOp>(operation))
      protectedWork |= isRawTransfer(call);
  });
  return protectedWork;
}

bool containsFallbackAllocation(func::FuncOp function) {
  bool found = false;
  function.walk([&](::vrc::vrc::BufferAllocOp) { found = true; });
  return found;
}

struct EliminateDeadPhysicalAllocationFallbacksPass
    : impl::EliminateDeadPhysicalAllocationFallbacksPassBase<
          EliminateDeadPhysicalAllocationFallbacksPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    DenseSet<Operation *> candidates;
    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      functions.push_back(function);
      if (!function.isExternal() && function.isPrivate() &&
          !containsProtectedWork(function) &&
          containsFallbackAllocation(function))
        candidates.insert(function.getOperation());
    }

    bool changed = true;
    while (changed) {
      changed = false;
      for (func::FuncOp function : functions) {
        if (function.isExternal() || !function.isPrivate() ||
            candidates.contains(function.getOperation()) ||
            containsProtectedWork(function))
          continue;
        bool callsCandidate = false;
        function.walk([&](func::CallOp call) {
          auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
          callsCandidate |=
              callee && candidates.contains(callee.getOperation());
        });
        if (callsCandidate) {
          candidates.insert(function.getOperation());
          changed = true;
        }
      }
    }

    changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> rejected;
      for (Operation *candidate : candidates) {
        auto uses = SymbolTable::getSymbolUses(candidate, module);
        if (!uses) {
          rejected.push_back(candidate);
          continue;
        }
        for (const SymbolTable::SymbolUse &use : *uses) {
          auto owner = use.getUser()->getParentOfType<func::FuncOp>();
          if (!owner || !candidates.contains(owner.getOperation())) {
            rejected.push_back(candidate);
            break;
          }
        }
      }
      for (Operation *candidate : rejected)
        changed |= candidates.erase(candidate);
    }

    for (func::FuncOp function : functions) {
      if (candidates.contains(function.getOperation()))
        function.erase();
    }
  }
};

} // namespace
} // namespace vrc
