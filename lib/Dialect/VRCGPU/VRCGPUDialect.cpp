#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#define GET_TYPEDEF_CLASSES
#include "VRC/Dialect/VRCGPU/VRCGPUDialect.cpp.inc"
#include "VRC/Dialect/VRCGPU/VRCGPUTypes.cpp.inc"
void vrc::vrcgpu::VRCGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "VRC/Dialect/VRCGPU/VRCGPUTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "VRC/Dialect/VRCGPU/VRCGPUOps.cpp.inc"
      >();
}
#define GET_OP_CLASSES
#include "VRC/Dialect/VRCGPU/VRCGPUOps.cpp.inc"

namespace vrc::vrcgpu {
using namespace mlir;
LogicalResult KernelOp::verify() {
  if (getBody().empty() || !getBody().hasOneBlock() ||
      getBody().front().getNumArguments())
    return emitOpError("kernel requires one resource block without arguments");
  if (getEntryName().empty() || getSourceModule().empty())
    return emitOpError(
        "kernel must retain source module and public entry name");
  if (getKernelIdAttr().getInt() <= 0 || getArgumentKinds().size() != 2 ||
      getArgumentKinds()[0] != StringAttr::get(getContext(), "buffer") ||
      getArgumentKinds()[1] != StringAttr::get(getContext(), "f32"))
    return emitOpError(
        "initial kernel ABI requires a positive id and [buffer, f32]");
  if (getGlobalSize() != ArrayRef<int32_t>({1, 1, 1}) ||
      getLocalSize() != ArrayRef<int32_t>({1, 1, 1}) ||
      getGlobalOffset() != ArrayRef<int32_t>({0, 0, 0}))
    return emitOpError(
        "initial fragment pattern requires one invocation with zero offset");
  unsigned resources = 0, passes = 0;
  llvm::SmallDenseSet<int32_t> indices, bindings;
  for (auto &op : getBody().front()) {
    if (auto resource = dyn_cast<ResourceOp>(op)) {
      ++resources;
      if (!indices.insert(resource.getArgIndex()).second ||
          !bindings.insert(resource.getBinding()).second)
        return emitOpError("resource argument/binding collision");
    } else if (isa<PassOp>(op)) {
      ++passes;
    } else
      return emitOpError("kernel body contains an unsupported operation");
  }
  if (resources != 2 || passes != 1)
    return emitOpError(
        "initial pattern requires two resources and one output pass");
  return success();
}
LogicalResult ResourceOp::verify() {
  if (!(*this)->getParentOfType<KernelOp>() || getBinding() != getArgIndex() ||
      !getValueType().isF32())
    return emitOpError("resource must have a kernel, f32 value type and stable "
                       "argument binding");
  if (getKind() == "target") {
    if (getArgIndex() != 0 || !getFormat() || *getFormat() != "rgba8_unorm")
      return emitOpError(
          "initial target is argument zero with rgba8_unorm format");
  } else if (getKind() == "uniform") {
    if (getArgIndex() != 1 || getFormat())
      return emitOpError(
          "initial f32 uniform is argument one without a texture format");
  } else
    return emitOpError("unsupported resource kind");
  return success();
}
LogicalResult PassOp::verify() {
  auto target =
      SymbolTable::lookupNearestSymbolFrom<ResourceOp>(*this, getOutputAttr());
  if (!target || target.getKind() != "target" || getPassId() != 0 ||
      getVersion().getType().getBufferId() != uint32_t(target.getArgIndex()))
    return emitOpError(
        "output/version must identify the sole target argument and pass zero");
  if (getBody().empty() || !getBody().hasOneBlock())
    return emitOpError("fragment requires one block");
  auto &block = getBody().front();
  for (auto &op : block) {
    if (!isa<UniformLoadOp, RawPackOp, OutputOp, arith::ConstantOp,
             arith::AddFOp, arith::SubFOp, arith::MulFOp>(op))
      return emitOpError(
          "operation is not fragment-safe in the initial pattern");
    if (auto constant = dyn_cast<arith::ConstantOp>(op))
      if (!constant.getType().isF32())
        return emitOpError("fragment constant must be f32");
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(op)) {
      auto flags = op.getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
      if (!op.getResult(0).getType().isF32() ||
          (flags && flags.getValue() != arith::FastMathFlags::none))
        return emitOpError("fragment arithmetic must be f32 without fast-math");
    }
  }
  if (block.empty() || !isa<OutputOp>(block.back()))
    return emitOpError("fragment region must terminate with vrcgpu.output");
  if (block.getNumArguments())
    return emitOpError(
        "initial pass uses symbol-bound resources, not region arguments");
  auto output = cast<OutputOp>(block.back());
  if (!output.getValue().getDefiningOp<RawPackOp>())
    return emitOpError("RGBA8 output must be produced by raw_pack");
  return success();
}
LogicalResult UniformLoadOp::verify() {
  auto resource = SymbolTable::lookupNearestSymbolFrom<ResourceOp>(
      *this, getResourceAttr());
  if (!resource || resource.getKind() != "uniform" ||
      resource.getValueType() != getValue().getType())
    return emitOpError("uniform reference/type must match a declared binding");
  if (!(*this)->getParentOfType<PassOp>())
    return emitOpError("uniform load must belong to a fragment pass");
  return success();
}
LogicalResult RawPackOp::verify() {
  if (getByteOffset() != 0 || getFormat() != "rgba8_unorm" ||
      !(*this)->getParentOfType<PassOp>())
    return emitOpError(
        "initial raw_pack encodes one complete f32 word at byte offset zero");
  return success();
}
} // namespace vrc::vrcgpu
