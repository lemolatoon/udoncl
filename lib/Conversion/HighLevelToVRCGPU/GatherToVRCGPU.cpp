#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include <algorithm>
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_LOWERGATHERDEVICEPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace fragment = ::vrc::vrcgpu;
Operation *make(OpBuilder &builder, Location loc, StringRef name,
                TypeRange results, ValueRange operands,
                ArrayRef<NamedAttribute> attributes, unsigned regions = 0) {
  OperationState state(loc, name);
  state.addTypes(results);
  state.addOperands(operands);
  state.addAttributes(attributes);
  while (regions--)
    state.addRegion();
  return builder.create(state);
}
struct LowerGatherDevicePass
    : impl::LowerGatherDevicePassBase<LowerGatherDevicePass> {
  void runOnOperation() override {
    auto root = getOperation();
    IRRewriter b(&getContext());
    auto textureShape =
        root->getAttrOfType<DenseI32ArrayAttr>("vrc.texture_shape");
    if (textureShape && (textureShape.size() != 2 || textureShape[0] <= 0 ||
                         textureShape[1] <= 0)) {
      root.emitError("invalid physical texture shape");
      signalPassFailure();
      return;
    }
    SmallVector<gpu::GPUModuleOp> modules;
    llvm::SmallDenseSet<int64_t> ids;
    for (auto module : root.getOps<gpu::GPUModuleOp>()) {
      modules.push_back(module);
      for (auto source : module.getOps<gpu::GPUFuncOp>()) {
        auto words = source->getAttrOfType<IntegerAttr>("vrc.words");
        auto grid = source->getAttrOfType<DenseI32ArrayAttr>("known_grid_size");
        if (!source->hasAttr("vrc.gather_proven") || !words || !grid ||
            grid.size() != 3 || !source.getBody().hasOneBlock()) {
          source.emitError("gather lowering requires bounded memory/loop "
                           "recovery and structured CFG");
          signalPassFailure();
          return;
        }
        auto id = source->getAttrOfType<IntegerAttr>("vrc.kernel_id");
        if (!id || id.getInt() <= 0 || !ids.insert(id.getInt()).second) {
          source.emitError("invalid or duplicate kernel id");
          signalPassFailure();
          return;
        }
        memref::StoreOp store;
        for (auto candidate : source.getOps<memref::StoreOp>()) {
          if (store) {
            source.emitError("multiple output stores");
            signalPassFailure();
            return;
          }
          store = candidate;
        }
        if (!store) {
          source.emitError("missing proven output store");
          signalPassFailure();
          return;
        }
        std::optional<unsigned> outputArgument;
        SmallVector<unsigned> inputArguments;
        for (unsigned i = 0; i < source.getNumArguments(); ++i) {
          if (!isa<MemRefType>(source.getArgument(i).getType()))
            continue;
          if (store.getMemref() == source.getArgument(i))
            outputArgument = i;
          else
            inputArguments.push_back(i);
        }
        if (!outputArgument || inputArguments.empty()) {
          source.emitError("gather requires read-only inputs and one output");
          signalPassFailure();
          return;
        }
        b.setInsertionPoint(module);
        auto loc = source.getLoc();
        SmallVector<Attribute> kinds;
        for (unsigned i = 0; i < source.getNumArguments(); ++i)
          kinds.push_back(b.getStringAttr(
              isa<MemRefType>(source.getArgument(i).getType()) ? "buffer"
              : source.getArgument(i).getType().isF32()        ? "f32"
                                                               : "i32"));
        SmallVector<NamedAttribute> kernelAttrs{
            b.getNamedAttr(
                "sym_name",
                b.getStringAttr(
                    (module.getName() + "__" + source.getName()).str())),
            b.getNamedAttr("entry_name", b.getStringAttr(source.getName())),
            b.getNamedAttr("source_module", b.getStringAttr(module.getName())),
            b.getNamedAttr("kernel_id", id),
            b.getNamedAttr("argument_kinds", b.getArrayAttr(kinds)),
            b.getNamedAttr("global_size", grid),
            b.getNamedAttr("local_size", b.getDenseI32ArrayAttr({1, 1, 1})),
            b.getNamedAttr("global_offset", b.getDenseI32ArrayAttr({0, 0, 0})),
            b.getNamedAttr("vrc.profile", b.getStringAttr("gather-f32")),
            b.getNamedAttr("vrc.words", words),
            b.getNamedAttr("vrc.gather_proven", b.getUnitAttr())};
        if (auto bounds = source->getAttr("vrc.argument_bounds"))
          kernelAttrs.push_back(b.getNamedAttr("vrc.argument_bounds", bounds));
        if (textureShape)
          kernelAttrs.push_back(
              b.getNamedAttr("vrc.texture_shape", textureShape));
        auto *kernel = make(b, loc, fragment::KernelOp::getOperationName(), {},
                            {}, kernelAttrs, 1);
        auto *body = new Block;
        kernel->getRegion(0).push_back(body);
        b.setInsertionPointToEnd(body);
        SmallVector<Value> inputVersions;
        for (unsigned i : inputArguments)
          inputVersions.push_back(body->addArgument(
              fragment::VersionType::get(&getContext(), i), loc));
        for (unsigned i = 0; i < source.getNumArguments(); ++i) {
          bool buffer = isa<MemRefType>(source.getArgument(i).getType());
          bool output = i == *outputArgument;
          SmallVector<NamedAttribute> attrs{
              b.getNamedAttr("sym_name",
                             b.getStringAttr("arg" + std::to_string(i))),
              b.getNamedAttr(
                  "kind",
                  b.getStringAttr(buffer ? (output ? "target" : "texture")
                                         : "uniform")),
              b.getNamedAttr("arg_index", b.getI32IntegerAttr(i)),
              b.getNamedAttr("binding", b.getI32IntegerAttr(i)),
              b.getNamedAttr(
                  "value_type",
                  TypeAttr::get(
                      buffer ? cast<MemRefType>(source.getArgument(i).getType())
                                   .getElementType()
                             : source.getArgument(i).getType()))};
          if (buffer)
            attrs.push_back(
                b.getNamedAttr("format", b.getStringAttr("rgba8_unorm")));
          make(b, loc, fragment::ResourceOp::getOperationName(), {}, {}, attrs);
        }
        auto *pass =
            make(b, loc, fragment::PassOp::getOperationName(),
                 TypeRange{fragment::VersionType::get(&getContext(),
                                                      *outputArgument)},
                 inputVersions,
                 {b.getNamedAttr("output",
                                 FlatSymbolRefAttr::get(
                                     &getContext(),
                                     "arg" + std::to_string(*outputArgument))),
                  b.getNamedAttr("pass_id", b.getI32IntegerAttr(0))},
                 1);
        auto *block = new Block;
        pass->getRegion(0).push_back(block);
        b.setInsertionPointToEnd(block);
        IRMapping mapping;
        for (unsigned i = 0; i < source.getNumArguments(); ++i) {
          if (isa<MemRefType>(source.getArgument(i).getType()))
            continue;
          auto *uniform = make(
              b, loc, fragment::UniformLoadOp::getOperationName(),
              TypeRange{source.getArgument(i).getType()}, {},
              {b.getNamedAttr("resource",
                              FlatSymbolRefAttr::get(
                                  &getContext(), "arg" + std::to_string(i)))});
          mapping.map(source.getArgument(i), uniform->getResult(0));
        }
        auto *coord =
            make(b, loc, fragment::FragmentCoordOp::getOperationName(),
                 TypeRange{b.getI32Type(), b.getI32Type()}, {}, {});
        int32_t physicalWidth =
            textureShape ? textureShape.asArrayRef()[0] : grid.asArrayRef()[0];
        auto physicalWidthValue =
            arith::ConstantIntOp::create(b, loc, physicalWidth, 32);
        Value physicalLinear = arith::AddIOp::create(
            b, loc,
            arith::MulIOp::create(b, loc, coord->getResult(1),
                                  physicalWidthValue),
            coord->getResult(0));
        auto logicalWidth =
            arith::ConstantIntOp::create(b, loc, grid.asArrayRef()[0], 32);
        Value logicalX =
            arith::RemUIOp::create(b, loc, physicalLinear, logicalWidth);
        Value logicalY =
            arith::DivUIOp::create(b, loc, physicalLinear, logicalWidth);
        for (auto &op : source.getBody().front())
          if (!isa<gpu::ReturnOp, memref::StoreOp>(op))
            b.clone(op, mapping);
        SmallVector<gpu::GlobalIdOp> globals;
        pass->walk([&](gpu::GlobalIdOp op) { globals.push_back(op); });
        for (auto op : globals) {
          unsigned coordinate = op.getDimension() == gpu::Dimension::x   ? 0
                                : op.getDimension() == gpu::Dimension::y ? 1
                                                                         : 2;
          if (coordinate == 2) {
            op.emitError("only x/y global ids are fragment-legal");
            signalPassFailure();
            return;
          }
          b.setInsertionPoint(op);
          b.replaceOpWithNewOp<arith::IndexCastOp>(
              op, b.getIndexType(), coordinate == 0 ? logicalX : logicalY);
        }
        SmallVector<memref::LoadOp> loads;
        pass->walk([&](memref::LoadOp op) { loads.push_back(op); });
        for (auto op : loads) {
          std::optional<unsigned> inputArgument;
          for (unsigned i : inputArguments)
            if (op.getMemref() == source.getArgument(i))
              inputArgument = i;
          if (!inputArgument || op.getIndices().size() != 1) {
            op.emitError("unproven input read");
            signalPassFailure();
            return;
          }
          b.setInsertionPoint(op);
          Value x = arith::IndexCastOp::create(b, op.getLoc(), b.getI32Type(),
                                               op.getIndices()[0]);
          int64_t inputElements =
              cast<MemRefType>(op.getMemref().getType()).getNumElements();
          int32_t inputWidth =
              static_cast<int32_t>(std::min<int64_t>(inputElements, 8192));
          auto width =
              arith::ConstantIntOp::create(b, op.getLoc(), inputWidth, 32);
          Value y = arith::DivUIOp::create(b, op.getLoc(), x, width);
          x = arith::RemUIOp::create(b, op.getLoc(), x, width);
          auto *texel = make(
              b, op.getLoc(), fragment::TextureLoadOp::getOperationName(),
              TypeRange{VectorType::get({4}, b.getF32Type())}, ValueRange{x, y},
              {b.getNamedAttr(
                   "resource",
                   FlatSymbolRefAttr::get(
                       &getContext(), "arg" + std::to_string(*inputArgument))),
               b.getNamedAttr("element_count",
                              b.getI32IntegerAttr(inputElements))});
          auto *value = make(
              b, op.getLoc(), fragment::RawUnpackOp::getOperationName(),
              TypeRange{
                  cast<MemRefType>(op.getMemref().getType()).getElementType()},
              texel->getResults(),
              {b.getNamedAttr("byte_offset", b.getI32IntegerAttr(0)),
               b.getNamedAttr("format", b.getStringAttr("rgba8_unorm"))});
          SmallVector<Value> keys;
          for (const auto &entry : mapping.getValueMap())
            if (entry.second == op.getResult())
              keys.push_back(entry.first);
          for (auto key : keys)
            mapping.map(key, value->getResult(0));
          b.replaceOp(op, value->getResults());
        }
        b.setInsertionPointToEnd(block);
        auto *packed =
            make(b, store.getLoc(), fragment::RawPackOp::getOperationName(),
                 TypeRange{VectorType::get({4}, b.getF32Type())},
                 ValueRange{mapping.lookup(store.getValue())},
                 {b.getNamedAttr("byte_offset", b.getI32IntegerAttr(0)),
                  b.getNamedAttr("format", b.getStringAttr("rgba8_unorm"))});
        make(b, loc, fragment::OutputOp::getOperationName(), {},
             packed->getResults(), {});
      }
    }
    for (auto module : modules)
      b.eraseOp(module);
    root->setAttr("vrc.device_stage", b.getStringAttr("vrcgpu-gather-f32"));
  }
};
} // namespace
} // namespace vrc
