#include "VRC/ABI/Manifest/Boundary.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_LOWERHOSTTONATIVETESTPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace host = ::vrc::udonhl;
namespace runtime = ::vrc::vrc;
constexpr StringLiteral dispatchSymbol = "__vrc_native_test_dispatch";
struct CallPattern : OpConversionPattern<host::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(host::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getCallPolicy() != host::CallPolicy::Normal)
      return op.emitError(
          "native test backend requires normal non-suspending calls");
    SmallVector<Type> results;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), results)))
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), results,
                                              adaptor.getArgs());
    return success();
  }
};
struct DispatchPattern : OpConversionPattern<runtime::DispatchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(runtime::DispatchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto id = arith::ConstantIntOp::create(rewriter, op.getLoc(),
                                           op.getKernelId(), 32);
    SmallVector<Value> args{id, adaptor.getDesc()};
    llvm::append_range(args, adaptor.getResources());
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, dispatchSymbol, TypeRange{rewriter.getI32Type()}, args);
    return success();
  }
};
struct LowerHostToNativeTestPass
    : impl::LowerHostToNativeTestPassBase<LowerHostToNativeTestPass> {
  void runOnOperation() override {
    auto module = getOperation();
    if (failed(verifyHostDeviceBoundary(module))) {
      signalPassFailure();
      return;
    }
    if (SymbolTable::lookupSymbolIn(module, dispatchSymbol)) {
      module.emitError("native test runtime symbol collision");
      signalPassFailure();
      return;
    }
    bool invalid = false;
    module.walk([&](host::CallOp call) {
      if (call.getCallPolicy() != host::CallPolicy::Normal) {
        call.emitError(
            "native test backend requires normal non-suspending calls");
        invalid = true;
      }
    });
    if (invalid) {
      signalPassFailure();
      return;
    }
    auto config = module->getAttrOfType<DictionaryAttr>("vrc.c_boundary");
    auto requestName =
        config ? config.getAs<StringAttr>("request") : StringAttr();
    auto request =
        requestName ? module.lookupSymbol<func::FuncOp>(requestName.getValue())
                    : func::FuncOp();
    if (!request || request.isExternal() || request.getNumArguments() != 2 ||
        !isa<runtime::BufferType>(request.getArgumentTypes()[0]) ||
        !isa<runtime::DispatchDescType>(request.getArgumentTypes()[1]) ||
        request.getResultTypes().size() != 1 ||
        !isa<runtime::EventType>(request.getResultTypes()[0])) {
      module.emitError(
          "native test backend requires the configured internal request entry");
      signalPassFailure();
      return;
    }
    // Expose the internal request to this test harness only. Production ABI
    // visibility remains unchanged in the original boundary bundle.
    request.setPublic();
    for (auto kernel :
         llvm::make_early_inc_range(module.getOps<::vrc::vrcgpu::KernelOp>()))
      kernel.erase();
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([&](runtime::BufferType) {
      return IntegerType::get(&getContext(), 32);
    });
    converter.addConversion([&](runtime::DispatchDescType) {
      return IntegerType::get(&getContext(), 32);
    });
    converter.addConversion([&](runtime::EventType) {
      return IntegerType::get(&getContext(), 32);
    });
    OpBuilder builder(module.getBodyRegion());
    auto runtimeDecl = func::FuncOp::create(
        builder, module.getLoc(), dispatchSymbol,
        builder.getFunctionType(
            {builder.getI32Type(), builder.getI32Type(), builder.getI32Type()},
            {builder.getI32Type()}));
    runtimeDecl.setPrivate();
    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalOp<func::CallOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp f) {
      return converter.isSignatureLegal(f.getFunctionType()) &&
             converter.isLegal(&f.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp ret) {
      return converter.isLegal(ret.getOperation());
    });
    target.addIllegalDialect<host::UdonHLDialect, runtime::VRCDialect>();
    RewritePatternSet patterns(&getContext());
    patterns.add<CallPattern, DispatchPattern>(converter, &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    // The native test binary must use its real target layout, not the temporary
    // SPIR32 frontend layout. This does not change the explicit uint32 handles.
    module->removeAttr("dlti.dl_spec");
    module->removeAttr("llvm.target_triple");
    module->removeAttr("vrc.llvm_data_layout");
    module->setAttr("vrc.native_test_backend", builder.getUnitAttr());
  }
};
} // namespace
} // namespace vrc
