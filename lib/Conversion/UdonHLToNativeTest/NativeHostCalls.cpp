#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_RAISENATIVEHOSTCALLSPASS
#define GEN_PASS_DEF_LOWERNATIVEHOSTCALLSPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace host = ::vrc::udonhl;
struct RaiseNativeHostCallsPass
    : impl::RaiseNativeHostCallsPassBase<RaiseNativeHostCallsPass> {
  void runOnOperation() override {
    auto module = getOperation();
    if (module->hasAttr("vrc.native_host_calls")) {
      module.emitError("native Host calls already raised");
      signalPassFailure();
      return;
    }
    bool invalid = false;
    module.walk([&](LLVM::LLVMFuncOp f) {
      if (f.getCConv() != LLVM::cconv::CConv::C || f.isVarArg()) {
        f.emitError("native Host adapter requires nonvariadic C functions");
        invalid = true;
      }
    });
    module.walk([&](LLVM::CallOp call) {
      if (!call.getCallee() || call.getCConv() != LLVM::cconv::CConv::C ||
          call.getVarCalleeType() ||
          call.getTailCallKind() != LLVM::TailCallKind::None ||
          !call.getOpBundleSizes().empty()) {
        call.emitError("native Host adapter requires direct C calls without "
                       "tail/bundle semantics");
        invalid = true;
      }
    });
    module.walk([&](LLVM::AddressOfOp address) {
      if (module.lookupSymbol<LLVM::LLVMFuncOp>(address.getGlobalName())) {
        address.emitError(
            "native Host function pointer recovery is not implemented");
        invalid = true;
      }
    });
    if (!invalid) {
      llvm::DenseSet<Operation *> active, done;
      auto visit = [&](auto &&self,
                       LLVM::LLVMFuncOp function) -> LogicalResult {
        if (done.contains(function))
          return success();
        if (!active.insert(function).second)
          return function.emitError(
              "native Host adapter requires acyclic direct C calls");
        for (auto call : function.getOps<LLVM::CallOp>()) {
          auto callee =
              module.lookupSymbol<LLVM::LLVMFuncOp>(*call.getCallee());
          if (callee && failed(self(self, callee)))
            return failure();
        }
        active.erase(function);
        done.insert(function);
        return success();
      };
      for (auto function : module.getOps<LLVM::LLVMFuncOp>())
        if (failed(visit(visit, function))) {
          invalid = true;
          break;
        }
    }
    if (invalid) {
      signalPassFailure();
      return;
    }
    IRRewriter rewriter(&getContext());
    for (auto source :
         llvm::make_early_inc_range(module.getOps<LLVM::LLVMFuncOp>())) {
      rewriter.setInsertionPoint(source);
      auto f = func::FuncOp::create(
          rewriter, source.getLoc(), source.getName(),
          rewriter.getFunctionType(source.getArgumentTypes(),
                                   source.getResultTypes()));
      f->setAttr("vrc.native_llvm_function",
                 rewriter.getDictionaryAttr(source->getAttrs()));
      if (source.isExternal() || source.getLinkage() != LLVM::Linkage::External)
        f.setPrivate();
      f.getBody().takeBody(source.getBody());
      source.erase();
    }
    SmallVector<LLVM::CallOp> calls;
    module.walk([&](LLVM::CallOp call) { calls.push_back(call); });
    for (auto call : calls) {
      rewriter.setInsertionPoint(call);
      OperationState state(call.getLoc(), host::CallOp::getOperationName());
      state.addOperands(call.getCalleeOperands());
      state.addTypes(call.getResultTypes());
      state.addAttribute("callee", call.getCalleeAttr());
      state.addAttribute(
          "call_policy",
          host::CallPolicyAttr::get(&getContext(), host::CallPolicy::Normal));
      state.addAttribute("vrc.native_llvm_call",
                         rewriter.getDictionaryAttr(call->getAttrs()));
      rewriter.replaceOp(call, rewriter.create(state)->getResults());
    }
    SmallVector<LLVM::ReturnOp> returns;
    module.walk([&](LLVM::ReturnOp ret) {
      if (ret->getParentOfType<func::FuncOp>())
        returns.push_back(ret);
    });
    for (auto ret : returns) {
      rewriter.setInsertionPoint(ret);
      rewriter.replaceOpWithNewOp<func::ReturnOp>(ret, ret.getOperands());
    }
    module->setAttr("vrc.native_host_calls", rewriter.getUnitAttr());
  }
};
struct LowerNativeHostCallsPass
    : impl::LowerNativeHostCallsPassBase<LowerNativeHostCallsPass> {
  void runOnOperation() override {
    auto module = getOperation();
    if (!module->hasAttr("vrc.native_host_calls")) {
      module.emitError("requires native Host call recovery");
      signalPassFailure();
      return;
    }
    bool invalid = false;
    module.walk([&](host::CallOp call) {
      if (call.getCallPolicy() != host::CallPolicy::Normal ||
          !call->getAttrOfType<DictionaryAttr>("vrc.native_llvm_call")) {
        call.emitError(
            "native call must retain its normal policy and original LLVM ABI");
        invalid = true;
      }
    });
    for (auto f : module.getOps<func::FuncOp>())
      if (!f->getAttrOfType<DictionaryAttr>("vrc.native_llvm_function")) {
        f.emitError("missing native LLVM function ABI");
        invalid = true;
      }
    if (invalid) {
      signalPassFailure();
      return;
    }
    IRRewriter rewriter(&getContext());
    SmallVector<host::CallOp> calls;
    module.walk([&](host::CallOp call) { calls.push_back(call); });
    for (auto call : calls) {
      rewriter.setInsertionPoint(call);
      OperationState state(call.getLoc(), LLVM::CallOp::getOperationName());
      state.addAttributes(
          call->getAttrOfType<DictionaryAttr>("vrc.native_llvm_call")
              .getValue());
      state.addOperands(call.getArgs());
      state.addTypes(call.getResultTypes());
      rewriter.replaceOp(call, rewriter.create(state)->getResults());
    }
    SmallVector<func::ReturnOp> returns;
    module.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
    for (auto ret : returns) {
      rewriter.setInsertionPoint(ret);
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(ret, ret.getOperands());
    }
    for (auto source :
         llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      auto attributes =
          source->getAttrOfType<DictionaryAttr>("vrc.native_llvm_function");
      auto type = cast<LLVM::LLVMFunctionType>(
          cast<TypeAttr>(attributes.get("function_type")).getValue());
      rewriter.setInsertionPoint(source);
      auto f = LLVM::LLVMFuncOp::create(rewriter, source.getLoc(),
                                        source.getName(), type);
      f->setAttrs(attributes.getValue());
      f.getBody().takeBody(source.getBody());
      source.erase();
    }
    module->removeAttr("vrc.native_host_calls");
  }
};
} // namespace
} // namespace vrc
