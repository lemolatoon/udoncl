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
      bool valid = false;
      if (name == "dispatch")
        valid =
            args.size() >= 1 && isa<runtime::DispatchDescType>(args[0]) &&
            llvm::all_of(
                args.drop_front(),
                [](Type type) { return isa<runtime::BufferType>(type); }) &&
            results.size() == 1 && isa<runtime::EventType>(results[0]) &&
            prototype->getAttrOfType<IntegerAttr>("vrc.kernel_id") &&
            prototype->getAttrOfType<ArrayAttr>("vrc.argument_kinds") &&
            prototype->getAttrOfType<DenseI32ArrayAttr>("vrc.global_size") &&
            prototype->getAttrOfType<DenseI32ArrayAttr>("vrc.local_size") &&
            prototype->getAttrOfType<DenseI32ArrayAttr>("vrc.global_offset");
      else if (name == "buffer_alloc")
        valid =
            args == ArrayRef<Type>{IntegerType::get(&getContext(), 32)} &&
            results == ArrayRef<Type>{runtime::BufferType::get(&getContext()),
                                      IntegerType::get(&getContext(), 32)} &&
            prototype->getAttrOfType<StringAttr>("vrc.format") &&
            prototype->getAttrOfType<StringAttr>("vrc.usage");
      else if (name == "buffer_copy")
        valid =
            args == ArrayRef<Type>{runtime::BufferType::get(&getContext()),
                                   runtime::BufferType::get(&getContext()),
                                   IntegerType::get(&getContext(), 32),
                                   IntegerType::get(&getContext(), 32),
                                   IntegerType::get(&getContext(), 32)} &&
            results == ArrayRef<Type>{runtime::EventType::get(&getContext())};
      else if (name == "await" || name == "event_status")
        valid =
            args == ArrayRef<Type>{runtime::EventType::get(&getContext())} &&
            results == ArrayRef<Type>{IntegerType::get(&getContext(), 32)} &&
            (name != "await" || prototype->getAttrOfType<StringAttr>(
                                    "vrc.async_effect") == "may_suspend");
      if (!prototype.isExternal() || !prototype.isPrivate() || !valid) {
        prototype.emitError("invalid typed runtime intrinsic contract");
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
              auto name = callee->getAttrOfType<StringAttr>("vrc.intrinsic");
              auto expectedPolicy = name == "await"
                                        ? host::CallPolicy::MaySuspend
                                        : host::CallPolicy::Normal;
              if (!caller || caller->getParentOp() != module ||
                  call.getCallPolicy() != expectedPolicy) {
                call.emitError(
                    "runtime intrinsic call policy does not match its Host "
                    "suspension contract");
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
      auto name = callee->getAttrOfType<StringAttr>("vrc.intrinsic");
      rewriter.setInsertionPoint(call);
      StringRef operationName =
          name == "dispatch"       ? runtime::DispatchOp::getOperationName()
          : name == "buffer_alloc" ? runtime::BufferAllocOp::getOperationName()
          : name == "buffer_copy"  ? runtime::BufferCopyOp::getOperationName()
          : name == "await"        ? runtime::AwaitOp::getOperationName()
                                   : runtime::EventStatusOp::getOperationName();
      OperationState state(call.getLoc(), operationName);
      state.addOperands(call.getArgs());
      state.addTypes(call.getResultTypes());
      if (name == "dispatch")
        for (StringRef attribute :
             {"kernel_id", "argument_kinds", "global_size", "local_size",
              "global_offset"})
          state.addAttribute(attribute,
                             callee->getAttr(("vrc." + attribute).str()));
      else if (name == "buffer_alloc")
        for (StringRef attribute : {"format", "usage"})
          state.addAttribute(attribute,
                             callee->getAttr(("vrc." + attribute).str()));
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
