#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_SUBSTITUTERUNTIMEINTRINSICSPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace host = ::vrc::udonhl;
namespace runtime = ::vrc::vrc;
struct SubstituteRuntimeIntrinsicsPass
    : impl::SubstituteRuntimeIntrinsicsPassBase<
          SubstituteRuntimeIntrinsicsPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto stage = module->getAttrOfType<StringAttr>("vrc.host_stage");
    if (!stage || stage != "udonhl") {
      module.emitError(
          "runtime intrinsic substitution must follow UdonHL construction");
      signalPassFailure();
      return;
    }
    SmallVector<std::pair<host::CallOp, func::FuncOp>> sites;
    SmallVector<func::FuncOp> prototypes;
    for (auto prototype : module.getOps<func::FuncOp>()) {
      auto name = prototype->getAttrOfType<StringAttr>("vrc.intrinsic");
      if (!name)
        continue;
      auto args = prototype.getArgumentTypes(),
           results = prototype.getResultTypes();
      if (name != "dispatch" || !prototype.isExternal() ||
          !prototype.isPrivate() || args.empty() ||
          !isa<runtime::DispatchDescType>(args[0]) ||
          !llvm::all_of(args.drop_front(),
                        [](Type t) { return isa<runtime::BufferType>(t); }) ||
          results.size() != 1 || !isa<runtime::EventType>(results[0]) ||
          !prototype->getAttrOfType<IntegerAttr>("vrc.kernel_id") ||
          !prototype->getAttrOfType<ArrayAttr>("vrc.argument_kinds") ||
          !prototype->getAttrOfType<DenseI32ArrayAttr>("vrc.global_size") ||
          !prototype->getAttrOfType<DenseI32ArrayAttr>("vrc.local_size") ||
          !prototype->getAttrOfType<DenseI32ArrayAttr>("vrc.global_offset")) {
        prototype.emitError(
            "invalid typed runtime dispatch intrinsic contract");
        signalPassFailure();
        return;
      }
      prototypes.push_back(prototype);
    }
    if (module
            .walk([&](host::CallOp call) {
              auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                  call, call.getCalleeAttr());
              if (!callee || !callee->hasAttr("vrc.intrinsic"))
                return WalkResult::advance();
              auto caller = call->getParentOfType<func::FuncOp>();
              if (!caller || caller->getParentOp() != module ||
                  call.getCallPolicy() != host::CallPolicy::Normal) {
                call.emitError("runtime dispatch intrinsic is a normal "
                               "internal Host call");
                return WalkResult::interrupt();
              }
              sites.emplace_back(call, callee);
              return WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }
    IRRewriter rewriter(&getContext());
    for (auto site : sites) {
      auto call = site.first;
      auto callee = site.second;
      rewriter.setInsertionPoint(call);
      OperationState state(call.getLoc(),
                           runtime::DispatchOp::getOperationName());
      state.addOperands(call.getArgs());
      state.addTypes(call.getResultTypes());
      for (StringRef name : {"kernel_id", "argument_kinds", "global_size",
                             "local_size", "global_offset"})
        state.addAttribute(name, callee->getAttr(("vrc." + name).str()));
      auto *dispatch = rewriter.create(state);
      rewriter.replaceOp(call, dispatch->getResults());
    }
    for (auto prototype : prototypes) {
      if (!SymbolTable::symbolKnownUseEmpty(prototype, module)) {
        prototype.emitError("unsubstituted runtime intrinsic use");
        signalPassFailure();
        return;
      }
      rewriter.eraseOp(prototype);
    }
    module->setAttr("vrc.intrinsics_stage",
                    StringAttr::get(&getContext(), "vrc"));
  }
};
} // namespace
} // namespace vrc
