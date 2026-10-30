#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_LOWERDEVICESINGLEINVOCATIONPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace fragment = ::vrc::vrcgpu;
Operation *make(OpBuilder &builder, Location loc, StringRef name,
                TypeRange types, ValueRange operands,
                ArrayRef<NamedAttribute> attrs, unsigned regions = 0) {
  OperationState state(loc, name);
  state.addTypes(types);
  state.addOperands(operands);
  state.addAttributes(attrs);
  for (unsigned i = 0; i < regions; ++i)
    state.addRegion();
  return builder.create(state);
}
LogicalResult validate(gpu::GPUFuncOp function) {
  auto fail = [&](StringRef message) { return function.emitError(message); };
  if (!function.isKernel() || function.getBody().getBlocks().size() != 1 ||
      function.getNumArguments() != 2 ||
      function.getFunctionType().getNumResults() ||
      function.getBody().front().getNumArguments() != 2)
    return fail("initial Device pattern requires one kernel block, output "
                "buffer and f32 scalar");
  auto output = dyn_cast<MemRefType>(function.getArgument(0).getType());
  if (!output || output.getShape() != ArrayRef<int64_t>({1}) ||
      !output.getElementType().isF32() || !output.getLayout().isIdentity() ||
      output.getMemorySpace() ||
      !function.getArgument(1).getType().isF32())
    return fail(
        "initial Device ABI is (memref<1xf32>, f32) with identity layout");
  auto id = function->getAttrOfType<IntegerAttr>("vrc.kernel_id");
  auto block = function.getKnownBlockSize(), grid = function.getKnownGridSize();
  if (!id || id.getInt() <= 0 || id.getInt() > INT32_MAX || !block || !grid ||
      *block != ArrayRef<int32_t>({1, 1, 1}) ||
      *grid != ArrayRef<int32_t>({1, 1, 1}))
    return fail("one-invocation proof requires a positive kernel id and known "
                "block/grid sizes [1,1,1]");
  unsigned stores = 0;
  for (auto &op : function.getBody().front()) {
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      auto index =
          store.getIndices().size() == 1
              ? store.getIndices()[0].getDefiningOp<arith::ConstantIndexOp>()
              : arith::ConstantIndexOp();
      if (++stores != 1 || store.getMemref() != function.getArgument(0) ||
          !index || index.value() != 0 || store.getNontemporal() ||
          store.getAlignment().value_or(4) > 4)
        return store.emitError("initial fragment ownership requires one "
                               "complete f32 store to output[0]");
    } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      if (!constant.getType().isF32() && !constant.getType().isIndex())
        return constant.emitError("unsupported Device constant type");
    } else if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(op)) {
      auto flags = op.getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
      if (!op.getResult(0).getType().isF32() ||
          (flags && flags.getValue() != arith::FastMathFlags::none))
        return op.emitError(
            "initial Device arithmetic requires f32 without fast-math");
    } else if (!isa<gpu::ReturnOp>(op)) {
      return op.emitError("operation requires additional GPU "
                          "legality/barrier/dependence analysis");
    }
  }
  if (stores != 1)
    return fail("kernel must write exactly one output");
  // The output allocation must not be read, captured, or used by other ops.
  for (auto &use : function.getArgument(0).getUses())
    if (!isa<memref::StoreOp>(use.getOwner()) || use.getOperandNumber() != 1)
      return fail("initial output ownership forbids reads and pointer escape");
  return success();
}
void lower(gpu::GPUModuleOp parent, gpu::GPUFuncOp source,
           IRRewriter &rewriter) {
  auto loc = source.getLoc();
  rewriter.setInsertionPoint(parent);
  auto kinds = rewriter.getArrayAttr(
      {rewriter.getStringAttr("buffer"), rewriter.getStringAttr("f32")});
  auto one = rewriter.getDenseI32ArrayAttr({1, 1, 1}),
       zero = rewriter.getDenseI32ArrayAttr({0, 0, 0});
  auto id = source->getAttrOfType<IntegerAttr>("vrc.kernel_id");
  auto symbol = rewriter.getStringAttr(
      (parent.getName() + "__" + source.getName()).str());
  auto *kernel =
      make(rewriter, loc, fragment::KernelOp::getOperationName(), {}, {},
           {rewriter.getNamedAttr("sym_name", symbol),
            rewriter.getNamedAttr("entry_name",
                                  rewriter.getStringAttr(source.getName())),
            rewriter.getNamedAttr("source_module",
                                  rewriter.getStringAttr(parent.getName())),
            rewriter.getNamedAttr("kernel_id",
                                  rewriter.getI32IntegerAttr(id.getInt())),
            rewriter.getNamedAttr("argument_kinds", kinds),
            rewriter.getNamedAttr("global_size", one),
            rewriter.getNamedAttr("local_size", one),
            rewriter.getNamedAttr("global_offset", zero)},
           1);
  auto *body = new Block();
  kernel->getRegion(0).push_back(body);
  rewriter.setInsertionPointToEnd(body);
  for (int i = 0; i < 2; ++i) {
    SmallVector<NamedAttribute> attrs{
        rewriter.getNamedAttr("sym_name",
                              rewriter.getStringAttr(i == 0 ? "arg0" : "arg1")),
        rewriter.getNamedAttr(
            "kind", rewriter.getStringAttr(i == 0 ? "target" : "uniform")),
        rewriter.getNamedAttr("arg_index", rewriter.getI32IntegerAttr(i)),
        rewriter.getNamedAttr("binding", rewriter.getI32IntegerAttr(i)),
        rewriter.getNamedAttr("value_type",
                              TypeAttr::get(rewriter.getF32Type()))};
    if (!i)
      attrs.push_back(rewriter.getNamedAttr(
          "format", rewriter.getStringAttr("rgba8_unorm")));
    make(rewriter, loc, fragment::ResourceOp::getOperationName(), {}, {},
         attrs);
  }
  auto version = fragment::VersionType::get(rewriter.getContext(), 0);
  auto *pass =
      make(rewriter, loc, fragment::PassOp::getOperationName(),
           TypeRange{version}, {},
           {rewriter.getNamedAttr("output", FlatSymbolRefAttr::get(
                                                rewriter.getContext(), "arg0")),
            rewriter.getNamedAttr("pass_id", rewriter.getI32IntegerAttr(0))},
           1);
  auto *passBody = new Block();
  pass->getRegion(0).push_back(passBody);
  rewriter.setInsertionPointToEnd(passBody);
  auto *uniform = make(
      rewriter, loc, fragment::UniformLoadOp::getOperationName(),
      TypeRange{rewriter.getF32Type()}, {},
      {rewriter.getNamedAttr(
          "resource", FlatSymbolRefAttr::get(rewriter.getContext(), "arg1"))});
  IRMapping mapping;
  mapping.map(source.getArgument(1), uniform->getResult(0));
  memref::StoreOp output;
  for (auto &op : source.getBody().front()) {
    if (auto store = dyn_cast<memref::StoreOp>(op))
      output = store;
    else if (isa<arith::ConstantOp, arith::AddFOp, arith::SubFOp,
                 arith::MulFOp>(op) &&
             !op.getResult(0).getType().isIndex())
      rewriter.clone(op, mapping);
  }
  auto *packed = make(
      rewriter, output.getLoc(), fragment::RawPackOp::getOperationName(),
      TypeRange{VectorType::get({4}, rewriter.getF32Type())},
      ValueRange{mapping.lookup(output.getValue())},
      {rewriter.getNamedAttr("byte_offset", rewriter.getI32IntegerAttr(0)),
       rewriter.getNamedAttr("format", rewriter.getStringAttr("rgba8_unorm"))});
  make(rewriter, output.getLoc(), fragment::OutputOp::getOperationName(), {},
       packed->getResults(), {});
}
struct LowerDeviceSingleInvocationPass
    : impl::LowerDeviceSingleInvocationPassBase<
          LowerDeviceSingleInvocationPass> {
  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<gpu::GPUModuleOp> modules;
    llvm::SmallDenseSet<int64_t> ids;
    for (auto parent : module.getOps<gpu::GPUModuleOp>()) {
      unsigned count = 0;
      for (auto &op : parent.getBody()->getOperations()) {
        auto function = dyn_cast<gpu::GPUFuncOp>(op);
        if (!function) {
          op.emitError("initial GPU module may contain only gpu.func kernels");
          signalPassFailure();
          return;
        }
        if (failed(validate(function))) {
          signalPassFailure();
          return;
        }
        auto id =
            function->getAttrOfType<IntegerAttr>("vrc.kernel_id").getInt();
        auto name = (parent.getName() + "__" + function.getName()).str();
        if (!ids.insert(id).second ||
            SymbolTable::lookupSymbolIn(module, name)) {
          function.emitError("kernel id or output symbol collision");
          signalPassFailure();
          return;
        }
        ++count;
      }
      if (!count) {
        parent.emitError("GPU module has no kernels");
        signalPassFailure();
        return;
      }
      modules.push_back(parent);
    }
    IRRewriter rewriter(&getContext());
    for (auto parent : modules) {
      for (auto function : parent.getOps<gpu::GPUFuncOp>())
        lower(parent, function, rewriter);
      rewriter.eraseOp(parent);
    }
    module->setAttr("vrc.device_stage",
                    StringAttr::get(&getContext(), "vrcgpu-single-invocation"));
  }
};
} // namespace
} // namespace vrc
