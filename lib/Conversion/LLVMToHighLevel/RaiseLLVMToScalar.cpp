#include "VRC/Conversion/LLVMToHighLevel/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_RAISELLVMTOSCALARPASS
#include "VRC/Conversion/LLVMToHighLevel/Passes.h.inc"

namespace {
bool isScalar(Type type) {
  return isa<IntegerType, ::vrc::udonhl::PtrType>(type) || type.isF32() ||
         type.isF64();
}

// Match operation names explicitly. Unknown LLVM operations are never legal.
StringRef raisedName(StringRef name) {
  return llvm::StringSwitch<StringRef>(name)
      .Case("llvm.add", "arith.addi")
      .Case("llvm.sub", "arith.subi")
      .Case("llvm.mul", "arith.muli")
      .Case("llvm.sdiv", "arith.divsi")
      .Case("llvm.udiv", "arith.divui")
      .Case("llvm.srem", "arith.remsi")
      .Case("llvm.urem", "arith.remui")
      .Case("llvm.and", "arith.andi")
      .Case("llvm.or", "arith.ori")
      .Case("llvm.xor", "arith.xori")
      .Case("llvm.shl", "arith.shli")
      .Case("llvm.lshr", "arith.shrui")
      .Case("llvm.ashr", "arith.shrsi")
      .Case("llvm.zext", "arith.extui")
      .Case("llvm.sext", "arith.extsi")
      .Case("llvm.trunc", "arith.trunci")
      .Case("llvm.fadd", "arith.addf")
      .Case("llvm.fsub", "arith.subf")
      .Case("llvm.fmul", "arith.mulf")
      .Case("llvm.fneg", "arith.negf")
      .Case("llvm.fdiv", "arith.divf")
      .Case("llvm.uitofp", "arith.uitofp")
      .Case("llvm.sitofp", "arith.sitofp")
      .Case("llvm.fptoui", "arith.fptoui")
      .Case("llvm.bitcast", "arith.bitcast")
      .Case("llvm.select", "arith.select")
      .Default("");
}

LogicalResult reject(Operation *op, StringRef reason) {
  return op->emitError() << "cannot raise " << op->getName() << ": " << reason;
}

LogicalResult checkArgumentAttrs(Operation *op, ArrayAttr attributes) {
  if (!attributes)
    return success();
  for (Attribute value : attributes)
    for (NamedAttribute attr : cast<DictionaryAttr>(value))
      if (attr.getName() != "llvm.noundef")
        return reject(
            op, "ABI argument/result attributes require explicit recovery");
  return success();
}

LogicalResult checkFunction(LLVM::LLVMFuncOp function) {
  if (function.getCConv() != LLVM::cconv::CConv::C)
    return reject(function, "only C calling convention is proven; OpenCL "
                            "GPU/ABI recovery is pending");
  if (function.isVarArg())
    return reject(function, "variadic ABI recovery is pending");
  if (function.getLinkage() != LLVM::Linkage::External &&
      function.getLinkage() != LLVM::Linkage::Internal &&
      function.getLinkage() != LLVM::Linkage::Private)
    return reject(function, "linkage recovery is pending for this linkage");
  if (!llvm::all_of(function.getArgumentTypes(), isScalar) ||
      !llvm::all_of(function.getResultTypes(), isScalar))
    return reject(function,
                  "pointer/aggregate ABI requires memory/provenance recovery");
  for (NamedAttribute attr : function->getAttrs()) {
    auto name = attr.getName().getValue();
    if (name.starts_with("vrc."))
      continue;
    bool known =
        llvm::StringSwitch<bool>(name)
            .Cases({"CConv", "function_type", "linkage", "sym_name"}, true)
            .Cases({"arg_attrs", "res_attrs", "unnamed_addr", "visibility_"},
                   true)
            .Cases({"sym_visibility", "dso_local", "frame_pointer"}, true)
            .Cases({"no_inline", "always_inline", "no_unwind", "optimize_none"},
                   true)
            .Cases({"optimize_for_size", "min_size", "will_return",
                    "memory_effects"},
                   true)
            .Case("passthrough", true)
            .Default(false);
    if (!known)
      return function.emitError()
             << "cannot raise llvm.func: unproven function attribute '" << name
             << "'";
  }
  // Unknown target attributes may change arithmetic or ABI semantics. Accept
  // only the default C frontend attributes that have no effect on this subset.
  if (auto passthrough = function->getAttrOfType<ArrayAttr>("passthrough")) {
    for (auto attr : passthrough) {
      if (auto string = dyn_cast<StringAttr>(attr)) {
        if (string.getValue() == "norecurse" ||
            string.getValue() == "mustprogress")
          continue;
      }
      if (auto pair = dyn_cast<ArrayAttr>(attr)) {
        // These frontend tuning attributes do not change the accepted scalar
        // operations. Stack-protector enablement itself is not accepted.
        if (pair.size() == 2 && isa<StringAttr>(pair[0]) &&
            isa<StringAttr>(pair[1])) {
          auto key = cast<StringAttr>(pair[0]).getValue();
          auto value = cast<StringAttr>(pair[1]).getValue();
          if ((key == "no-trapping-math" && value == "true") ||
              key == "stack-protector-buffer-size")
            continue;
        }
      }
      return reject(function,
                    "target/passthrough function attributes are not proven");
    }
  }
  if (failed(checkArgumentAttrs(function, function.getArgAttrsAttr())) ||
      failed(checkArgumentAttrs(function, function.getResAttrsAttr())))
    return failure();
  return success();
}

LogicalResult checkOperation(Operation *op) {
  if (auto function = dyn_cast<LLVM::LLVMFuncOp>(op))
    return checkFunction(function);
  for (Type type : op->getOperandTypes())
    if (!isScalar(type))
      return reject(op,
                    "non-scalar operand requires memory/provenance recovery");
  for (Type type : op->getResultTypes())
    if (!isScalar(type))
      return reject(op,
                    "non-scalar result requires memory/provenance recovery");
  if (auto fastmath = dyn_cast<LLVM::FastmathFlagsInterface>(op))
    if (fastmath.getFastmathAttr().getValue() != LLVM::FastmathFlags::none)
      return reject(op, "fast-math recovery is pending");
  // Dropping these flags would erase poison conditions.
  for (StringRef name : {"isExact", "nonNeg", "isDisjoint", "samesign"})
    if (op->hasAttr(name))
      return reject(op, "poison-generating flag has no implemented mapping");
  if (auto call = dyn_cast<LLVM::CallOp>(op)) {
    if (!call.getCallee() || call.getCConv() != LLVM::cconv::CConv::C ||
        call.getTailCallKind() != LLVM::TailCallKind::None ||
        !call.getOpBundleSizes().empty() || call.getVarCalleeType())
      return reject(op, "only direct nonvariadic C calls without tail/bundle "
                        "semantics are proven");
    if (failed(checkArgumentAttrs(op, call.getArgAttrsAttr())) ||
        failed(checkArgumentAttrs(op, call.getResAttrsAttr())))
      return failure();
    for (NamedAttribute attr : op->getAttrs()) {
      auto name = attr.getName().getValue();
      if (!llvm::StringSwitch<bool>(name)
               .Cases({"callee", "CConv", "TailCallKind", "fastmathFlags"},
                      true)
               .Cases({"arg_attrs", "res_attrs", "operandSegmentSizes",
                       "op_bundle_sizes"},
                      true)
               .Default(false))
        return reject(op, "unproven instruction attribute on scalar call");
    }
    return success();
  }
  for (NamedAttribute attr : op->getAttrs()) {
    auto name = attr.getName().getValue();
    if (name.starts_with("vrc."))
      continue;
    if (!llvm::StringSwitch<bool>(name)
             .Cases({"value", "predicate", "overflowFlags", "fastmathFlags"},
                    true)
             .Cases(
                 {"operandSegmentSizes", "branch_weights", "loop_annotation"},
                 true)
             .Default(false))
      return op->emitError()
             << "cannot raise " << op->getName()
             << ": unproven instruction attribute '" << name << "'";
  }
  if (auto constant = dyn_cast<LLVM::ConstantOp>(op)) {
    if (!isa<IntegerAttr, FloatAttr>(constant.getValue()))
      return reject(op, "only scalar integer/float constants are supported");
    return success();
  }
  if (isa<LLVM::BitcastOp>(op)) {
    Type source = op->getOperand(0).getType();
    Type target = op->getResult(0).getType();
    bool proven = (source.isF32() && target.isInteger(32)) ||
                  (source.isInteger(32) && target.isF32());
    if (!proven)
      return reject(op, "only bit-exact f32/i32 scalar casts are proven");
    return success();
  }
  if (!raisedName(op->getName().getStringRef()).empty() ||
      isa<LLVM::FCmpOp, LLVM::ICmpOp, LLVM::BrOp, LLVM::CondBrOp,
          LLVM::ReturnOp>(op))
    return success();
  return reject(op, "unsupported semantics; memory, calls, poison and GPU "
                    "recovery are pending");
}

struct FunctionPattern : OpConversionPattern<LLVM::LLVMFuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type =
        rewriter.getFunctionType(op.getArgumentTypes(), op.getResultTypes());
    auto function =
        func::FuncOp::create(rewriter, op.getLoc(), op.getSymName(), type);
    if (op.isExternal() || op.getLinkage() != LLVM::Linkage::External)
      function.setPrivate();
    // Keep the source contract for later ABI/debug work, without pretending
    // LLVM attributes automatically have func-dialect semantics.
    function->setAttr("vrc.llvm_function",
                      rewriter.getDictionaryAttr(op->getAttrs()));
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("vrc."))
        function->setAttr(attr.getName(), attr.getValue());
    rewriter.inlineRegionBefore(op.getBody(), function.getBody(),
                                function.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ScalarPattern : ConversionPattern {
  ScalarPattern(StringRef source, StringRef target, MLIRContext *context)
      : ConversionPattern(source, 1, context), target(target) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperationState state(op->getLoc(), target);
    state.addOperands(operands);
    state.addTypes(op->getResultTypes());
    if (auto flags = dyn_cast<LLVM::IntegerOverflowFlagsInterface>(op)) {
      auto mapped = arith::IntegerOverflowFlags::none;
      auto source = flags.getOverflowFlags();
      if (LLVM::bitEnumContainsAll(source, LLVM::IntegerOverflowFlags::nsw))
        mapped = mapped | arith::IntegerOverflowFlags::nsw;
      if (LLVM::bitEnumContainsAll(source, LLVM::IntegerOverflowFlags::nuw))
        mapped = mapped | arith::IntegerOverflowFlags::nuw;
      state.addAttribute("overflowFlags", arith::IntegerOverflowFlagsAttr::get(
                                              getContext(), mapped));
    }
    auto *raised = rewriter.create(state);
    rewriter.replaceOp(op, raised->getResults());
    return success();
  }
  StringRef target;
};
struct CallPattern : OpConversionPattern<LLVM::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, *op.getCallee(), op.getResultTypes(), adaptor.getOperands());
    return success();
  }
};
struct ConstantPattern : OpConversionPattern<LLVM::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, op.getType(), cast<TypedAttr>(op.getValue()));
    return success();
  }
};
struct ComparePattern : OpConversionPattern<LLVM::ICmpOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto predicate = arith::symbolizeCmpIPredicate(
        LLVM::stringifyICmpPredicate(op.getPredicate()));
    if (!predicate)
      return failure();
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, *predicate, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};
struct FloatComparePattern : OpConversionPattern<LLVM::FCmpOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::FCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto predicate = arith::symbolizeCmpFPredicate(
        LLVM::stringifyFCmpPredicate(op.getPredicate()));
    if (!predicate)
      return failure();
    rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, *predicate, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};
struct BranchPattern : OpConversionPattern<LLVM::BrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::BrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                              adaptor.getDestOperands());
    return success();
  }
};
struct CondBranchPattern : OpConversionPattern<LLVM::CondBrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::CondBrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    return success();
  }
};
struct ReturnPattern : OpConversionPattern<LLVM::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct RaiseLLVMToScalarPass
    : impl::RaiseLLVMToScalarPassBase<RaiseLLVMToScalarPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Preflight before rewriting: diagnose at the original operation/location.
    if (module
            .walk([](Operation *op) {
              if (op->getName().getDialectNamespace() == "llvm" &&
                  failed(checkOperation(op)))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }
    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, cf::ControlFlowDialect,
        memref::MemRefDialect, math::MathDialect, gpu::GPUDialect,
        ::vrc::vrc::VRCDialect, ::vrc::udonhl::UdonHLDialect>();
    target.addIllegalDialect<LLVM::LLVMDialect>();
    RewritePatternSet patterns(&getContext());
    patterns.add<FunctionPattern, ConstantPattern, ComparePattern,
                 FloatComparePattern, BranchPattern, CondBranchPattern,
                 ReturnPattern, CallPattern>(&getContext());
    for (auto source :
         {"llvm.add",    "llvm.sub",    "llvm.mul",    "llvm.sdiv",
          "llvm.udiv",   "llvm.srem",   "llvm.urem",   "llvm.and",
          "llvm.or",     "llvm.xor",    "llvm.shl",    "llvm.lshr",
          "llvm.ashr",   "llvm.zext",   "llvm.sext",   "llvm.trunc",
          "llvm.fadd",   "llvm.fsub",   "llvm.fmul",   "llvm.fneg",
          "llvm.select", "llvm.fdiv",   "llvm.uitofp", "llvm.sitofp",
          "llvm.fptoui", "llvm.bitcast"})
      patterns.add<ScalarPattern>(source, raisedName(source), &getContext());
    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void registerHighLevelRecoveryPipeline() {
  PassPipelineRegistration<>(
      "vrc-recover-high-level",
      "Recover proven LLVM memory/scalars and structure the recovered CFG",
      [](OpPassManager &manager) {
        manager.addPass(mlir::createMem2Reg());
        RaiseLLVMMemoryPassOptions memoryOptions;
        memoryOptions.allowRawFallback = true;
        manager.addPass(createRaiseLLVMMemoryPass(memoryOptions));
        manager.addPass(createRaiseLLVMRawMemoryPass());
        manager.addPass(createRaiseLLVMToScalarPass());
        manager.addPass(mlir::createLiftControlFlowToSCFPass());
      });
}
} // namespace vrc
