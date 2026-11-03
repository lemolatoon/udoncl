#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_LOWERHOSTTOUDONHLPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace host = ::vrc::udonhl;
namespace runtime = ::vrc::vrc;
bool scalar(Type type) { return type.isInteger(32) || type.isF32(); }
bool internalScalar(Type type) {
  return scalar(type) || type.isInteger(1) || type.isIndex();
}
bool publicArray(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.getLayout().isIdentity() &&
         (memref.getElementType().isInteger(32) ||
          memref.getElementType().isF32());
}
bool publicType(Type type) { return scalar(type) || publicArray(type); }
bool internalType(Type type) {
  return internalScalar(type) || publicArray(type) ||
         isa<runtime::BufferType, runtime::DispatchDescType,
             runtime::EventType>(type);
}
struct CallPattern : OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OperationState state(op.getLoc(), host::CallOp::getOperationName());
    state.addOperands(adaptor.getOperands());
    state.addTypes(op.getResultTypes());
    state.addAttribute("callee", op.getCalleeAttr());
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        op, op.getCalleeAttr());
    auto effect = callee ? callee->getAttrOfType<StringAttr>("vrc.async_effect")
                         : StringAttr();
    state.addAttribute("call_policy",
                       host::CallPolicyAttr::get(
                           getContext(), effect && effect == "may_suspend"
                                             ? host::CallPolicy::MaySuspend
                                             : host::CallPolicy::Normal));
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    return success();
  }
};
struct LowerHostToUdonHLPass
    : impl::LowerHostToUdonHLPassBase<LowerHostToUdonHLPass> {
  void runOnOperation() override {
    auto module = getOperation();
    if (auto abi = module->getAttrOfType<IntegerAttr>("vrc.abi_version"))
      if (abi.getInt() != 1) {
        module.emitError("unsupported Host/Device ABI version");
        signalPassFailure();
        return;
      }
    if (auto target = module->getAttrOfType<StringAttr>("vrc.target"))
      if (target != "vrc-pc") {
        module.emitError("initial boundary target is vrc-pc");
        signalPassFailure();
        return;
      }
    SmallVector<func::FuncOp> functions;
    llvm::SmallDenseSet<int64_t> exportIds;
    for (auto function : module.getOps<func::FuncOp>()) {
      functions.push_back(function);
      auto fail = [&](StringRef message) {
        function.emitError(message);
        signalPassFailure();
      };
      if (function->hasAttr("vrc.export_id") &&
          !function->getAttrOfType<IntegerAttr>("vrc.export_id")) {
        fail("vrc.export_id must be an integer");
        return;
      }
      auto effect = function->getAttrOfType<StringAttr>("vrc.async_effect");
      if (effect && effect != "none" && effect != "may_suspend") {
        fail("unsupported Host async effect");
        return;
      }
      if (function.isExternal()) {
        auto intrinsic = function->getAttrOfType<StringAttr>("vrc.intrinsic");
        if (!intrinsic ||
            (intrinsic != "dispatch" && intrinsic != "buffer_alloc" &&
             intrinsic != "buffer_copy" && intrinsic != "buffer_upload" &&
             intrinsic != "buffer_readback" && intrinsic != "await" &&
             intrinsic != "event_status") ||
            function->hasAttr("vrc.export_id")) {
          fail("external Host calls require a recognized internal runtime "
               "intrinsic");
          return;
        }
        continue;
      }
      if (!llvm::all_of(function.getArgumentTypes(), internalType) ||
          !llvm::all_of(function.getResultTypes(), internalType)) {
        fail("unsupported Host ABI type; pointers, memory and aggregate "
             "lowering remain pending");
        return;
      }
      if (auto id = function->getAttrOfType<IntegerAttr>("vrc.export_id")) {
        if (id.getInt() <= 0 || id.getInt() > INT32_MAX ||
            !exportIds.insert(id.getInt()).second ||
            !llvm::all_of(function.getArgumentTypes(), publicType) ||
            !llvm::all_of(function.getResultTypes(), publicType) ||
            function.getFunctionType().getNumResults() > 1 ||
            function.isPrivate()) {
          fail("export id/signature must be unique and use public scalar or "
               "one-dimensional array types; opaque resources are internal "
               "only");
          return;
        }
      }
      for (Block &block : function.getBody()) {
        if (!llvm::all_of(block.getArgumentTypes(), internalType)) {
          fail("unsupported Host block argument type");
          return;
        }
        for (Operation &op : block) {
          if (!isa<func::CallOp, func::ReturnOp, cf::BranchOp, cf::CondBranchOp,
                   runtime::DispatchDescCreateOp, runtime::DispatchOp,
                   runtime::BufferAllocOp, runtime::BufferCopyOp,
                   runtime::BufferUploadOp, runtime::BufferReadbackOp,
                   memref::LoadOp, memref::StoreOp, memref::AllocaOp,
                   memref::GetGlobalOp, arith::ConstantOp, arith::AddIOp,
                   arith::SubIOp, arith::MulIOp, arith::AddFOp, arith::SubFOp,
                   arith::MulFOp, arith::DivFOp, arith::CmpIOp, arith::CmpFOp>(
                  op)) {
            op.emitError(
                "operation is outside the initial Host lowering pattern");
            signalPassFailure();
            return;
          }
          if (!llvm::all_of(op.getResultTypes(), internalType)) {
            op.emitError("unsupported Host intermediate type");
            signalPassFailure();
            return;
          }
        }
      }
    }
    // Direct calls may carry a proven may_suspend effect, but recursive SCCs
    // still require explicit frame lowering before conversion.
    llvm::DenseSet<Operation *> active, done;
    auto visit = [&](auto &&self, func::FuncOp function) -> LogicalResult {
      if (done.contains(function))
        return success();
      if (!active.insert(function).second)
        return function.emitError(
            "recursive call graph needs SCC/frame lowering");
      for (auto call : function.getOps<func::CallOp>()) {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, call.getCalleeAttr());
        if (!callee || callee->getParentOp() != module)
          return call.emitError(
              "Host callee must resolve within the same module");
        if (failed(self(self, callee)))
          return failure();
      }
      active.erase(function);
      done.insert(function);
      return success();
    };
    for (auto function : functions)
      if (failed(visit(visit, function))) {
        signalPassFailure();
        return;
      }
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp call) {
      auto function = call->getParentOfType<func::FuncOp>();
      return !function || function->getParentOp() != module;
    });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    RewritePatternSet patterns(&getContext());
    patterns.add<CallPattern>(&getContext());
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    module->setAttr("vrc.host_stage", StringAttr::get(&getContext(), "udonhl"));
    module->setAttr("vrc.abi_version",
                    IntegerAttr::get(IntegerType::get(&getContext(), 32), 1));
  }
};
} // namespace
} // namespace vrc
