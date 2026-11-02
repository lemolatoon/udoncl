#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
static bool gather(KernelOp kernel) {
  auto profile = kernel->getAttrOfType<StringAttr>("vrc.profile");
  return profile && profile == "gather-f32";
}
static bool scalar(Type t) {
  return t.isF32() || t.isInteger(32) || t.isInteger(1) || t.isIndex();
}
static bool legalArithmetic(Operation *op) {
  return llvm::is_contained(
             ArrayRef<StringRef>{
                 "arith.constant",     "arith.addf",   "arith.subf",
                 "arith.mulf",         "arith.divf",   "arith.negf",
                 "arith.addi",         "arith.subi",   "arith.muli",
                 "arith.divui",        "arith.remui",  "arith.andi",
                 "arith.ori",          "arith.xori",   "arith.shli",
                 "arith.shrui",        "arith.shrsi",  "arith.cmpi",
                 "arith.cmpf",         "arith.select", "arith.uitofp",
                 "arith.sitofp",       "arith.fptoui", "arith.index_cast",
                 "arith.index_castui", "arith.trunci", "arith.bitcast",
                 "math.floor",         "math.sqrt",    "math.cos",
                 "math.sin",           "math.exp",     "math.log"},
             op->getName().getStringRef()) &&
         llvm::all_of(op->getResultTypes(), scalar);
}
LogicalResult KernelOp::verify() {
  if (getBody().empty() || !getBody().hasOneBlock() ||
      (getBody().front().getNumArguments() && !gather(*this)))
    return emitOpError("kernel requires one resource block without arguments");
  if (getEntryName().empty() || getSourceModule().empty())
    return emitOpError(
        "kernel must retain source module and public entry name");
  if (gather(*this)) {
    auto words = (*this)->getAttrOfType<IntegerAttr>("vrc.words");
    auto global = getGlobalSize();
    if (!words || words.getInt() < 4 || words.getInt() > 16384 ||
        getKernelId() <= 0 || getArgumentKinds().size() < 2 ||
        getArgumentKinds().size() > 16 || global.size() != 3 ||
        global[0] <= 0 || global[1] <= 0 || global[2] != 1 ||
        int64_t(global[0]) * global[1] != words.getInt() ||
        getLocalSize() != ArrayRef<int32_t>{1, 1, 1} ||
        getGlobalOffset() != ArrayRef<int32_t>{0, 0, 0} ||
        !(*this)->hasAttr("vrc.gather_proven"))
      return emitOpError("invalid proven gather launch/argument contract");
    unsigned bufferCount = 0;
    bool sawScalar = false;
    for (auto kind : getArgumentKinds().getValue()) {
      if (kind == StringAttr::get(getContext(), "buffer")) {
        if (sawScalar)
          return emitOpError("gather buffers must precede scalar arguments");
        ++bufferCount;
      } else if (kind == StringAttr::get(getContext(), "f32") ||
                 kind == StringAttr::get(getContext(), "i32"))
        sawScalar = true;
      else
        return emitOpError("unsupported gather scalar ABI");
    }
    if (bufferCount < 2)
      return emitOpError("gather requires at least one input and one output");
    llvm::SmallDenseSet<int32_t> indices;
    llvm::SmallDenseSet<int32_t> inputs;
    unsigned passes = 0, targets = 0;
    for (auto &op : getBody().front()) {
      if (auto r = dyn_cast<ResourceOp>(op)) {
        if (!indices.insert(r.getArgIndex()).second)
          return emitOpError("resource argument collision");
        if (r.getKind() == "texture")
          inputs.insert(r.getArgIndex());
        else if (r.getKind() == "target")
          ++targets;
      } else if (isa<PassOp>(op))
        ++passes;
      else
        return emitOpError("unsupported gather kernel operation");
    }
    if (indices.size() != getArgumentKinds().size() || passes != 1 ||
        targets != 1 || inputs.size() + targets != bufferCount ||
        getBody().front().getNumArguments() != inputs.size())
      return emitOpError("gather requires all argument bindings and one pass");
    llvm::SmallDenseSet<int32_t> versions;
    for (auto argument : getBody().front().getArguments()) {
      auto version = dyn_cast<VersionType>(argument.getType());
      if (!version || !inputs.contains(version.getBufferId()) ||
          !versions.insert(version.getBufferId()).second)
        return emitOpError("gather input versions must match texture bindings");
    }
    return success();
  }
  if (getKernelIdAttr().getInt() <= 0 || getArgumentKinds().size() != 2 ||
      getArgumentKinds()[0] != StringAttr::get(getContext(), "buffer") ||
      (getArgumentKinds()[1] != StringAttr::get(getContext(), "f32") &&
       getArgumentKinds()[1] != StringAttr::get(getContext(), "i32")))
    return emitOpError(
        "initial kernel ABI requires a positive id and [buffer, f32/i32]");
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
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (kernel && gather(kernel)) {
    int32_t i = getArgIndex();
    if (getBinding() != i || i < 0 ||
        size_t(i) >= kernel.getArgumentKinds().size())
      return emitOpError("invalid gather argument binding");
    auto argumentKind = kernel.getArgumentKinds()[i];
    if (argumentKind == StringAttr::get(getContext(), "buffer")) {
      if (!getValueType().isF32() ||
          (getKind() != "texture" && getKind() != "target") || !getFormat() ||
          *getFormat() != "rgba8_unorm")
        return emitOpError("invalid input/target format");
    } else if (getKind() != "uniform" || getFormat() ||
               (getValueType().isF32()
                    ? kernel.getArgumentKinds()[i] !=
                          StringAttr::get(getContext(), "f32")
                    : !getValueType().isInteger(32) ||
                          kernel.getArgumentKinds()[i] !=
                              StringAttr::get(getContext(), "i32")))
      return emitOpError("invalid gather uniform type");
    return success();
  }
  if (!kernel || getArgIndex() < 0 || getArgIndex() > 1 ||
      getBinding() != getArgIndex() ||
      (getValueType().isF32()
           ? kernel.getArgumentKinds()[getArgIndex()] !=
                 StringAttr::get(getContext(), getArgIndex() ? "f32" : "buffer")
           : !getValueType().isInteger(32) ||
                 kernel.getArgumentKinds()[getArgIndex()] !=
                     StringAttr::get(getContext(),
                                     getArgIndex() ? "i32" : "buffer")))
    return emitOpError("resource type must match the stable kernel argument "
                       "binding");
  if (getKind() == "target") {
    if (getArgIndex() != 0 || !getFormat() || *getFormat() != "rgba8_unorm")
      return emitOpError(
          "initial target is argument zero with rgba8_unorm format");
  } else if (getKind() == "uniform") {
    if (getArgIndex() != 1 || getFormat())
      return emitOpError(
          "initial scalar uniform is argument one without a texture format");
  } else
    return emitOpError("unsupported resource kind");
  return success();
}
LogicalResult PassOp::verify() {
  auto target =
      SymbolTable::lookupNearestSymbolFrom<ResourceOp>(*this, getOutputAttr());
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (kernel && gather(kernel)) {
    if (!target || target.getKind() != "target" ||
        getVersion().getType().getBufferId() !=
            uint32_t(target.getArgIndex()) ||
        getPassId() != 0 ||
        getReads().size() != kernel.getBody().front().getNumArguments() ||
        getBody().empty() || !getBody().hasOneBlock() ||
        getBody().front().getNumArguments())
      return emitOpError(
          "gather pass requires distinct input/output versions and one block");
    for (auto pair :
         llvm::zip(getReads(), kernel.getBody().front().getArguments()))
      if (std::get<0>(pair) != std::get<1>(pair))
        return emitOpError(
            "input versions must name every bound read-only resource");
    bool invalid = false;
    getBody().walk([&](Operation *op) {
      if (isa<FragmentCoordOp, TextureLoadOp, RawUnpackOp, RawPackOp,
              UniformLoadOp, OutputOp, scf::IfOp, scf::WhileOp, scf::YieldOp,
              scf::ConditionOp, scf::ExecuteRegionOp>(op) ||
          legalArithmetic(op)) {
        if (auto flags =
                op->getAttrOfType<arith::FastMathFlagsAttr>("fastmath"))
          if (flags.getValue() != arith::FastMathFlags::none)
            invalid = true;
      } else {
        op->emitError("operation is not proven fragment-safe: ")
            << op->getName();
        invalid = true;
      }
    });
    auto &body = getBody().front();
    auto output = body.empty() ? OutputOp() : dyn_cast<OutputOp>(body.back());
    if (invalid || !output || !output.getValue().getDefiningOp<RawPackOp>())
      return emitOpError(
          "gather region contains an operation not proven fragment-safe");
    return success();
  }
  if (!target || target.getKind() != "target" || getPassId() != 0 ||
      getVersion().getType().getBufferId() != uint32_t(target.getArgIndex()))
    return emitOpError(
        "output/version must identify the sole target argument and pass zero");
  if (getBody().empty() || !getBody().hasOneBlock())
    return emitOpError("fragment requires one block");
  auto &block = getBody().front();
  for (auto &op : block) {
    if (!isa<UniformLoadOp, RawPackOp, OutputOp, arith::ConstantOp,
             arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::AddIOp,
             arith::SubIOp, arith::MulIOp>(op))
      return emitOpError(
          "operation is not fragment-safe in the initial pattern");
    if (auto constant = dyn_cast<arith::ConstantOp>(op))
      if (!constant.getType().isF32() && !constant.getType().isInteger(32))
        return emitOpError("fragment constant must be f32 or i32");
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(op)) {
      auto flags = op.getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
      if (!op.getResult(0).getType().isF32() ||
          (flags && flags.getValue() != arith::FastMathFlags::none))
        return emitOpError("fragment arithmetic must be f32 without fast-math");
    }
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op) &&
        !op.getResult(0).getType().isInteger(32))
      return emitOpError("fragment integer arithmetic must be i32");
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
      (!getValue().getType().isF32() && !getValue().getType().isInteger(32)) ||
      !(*this)->getParentOfType<PassOp>())
    return emitOpError("initial raw_pack encodes one complete f32/i32 word at "
                       "byte offset zero");
  return success();
}
LogicalResult FragmentCoordOp::verify() {
  auto k = (*this)->getParentOfType<KernelOp>();
  if (!k || !gather(k) || !(*this)->getParentOfType<PassOp>())
    return emitOpError("fragment coordinate requires a gather pass");
  return success();
}
LogicalResult TextureLoadOp::verify() {
  auto k = (*this)->getParentOfType<KernelOp>();
  auto r = SymbolTable::lookupNearestSymbolFrom<ResourceOp>(*this,
                                                            getResourceAttr());
  auto x = getX().getDefiningOp<arith::RemUIOp>();
  auto y = getY().getDefiningOp<arith::DivUIOp>();
  auto width =
      x ? x.getRhs().getDefiningOp<arith::ConstantOp>() : arith::ConstantOp();
  auto widthValue =
      width ? dyn_cast<IntegerAttr>(width.getValue()) : IntegerAttr();
  auto words = k ? k->getAttrOfType<IntegerAttr>("vrc.words") : IntegerAttr();
  if (!k || !gather(k) || !r || r.getKind() != "texture" || !words ||
      words.getInt() != getElementCount() || !x || !y ||
      x.getLhs() != y.getLhs() || x.getRhs() != y.getRhs() || !widthValue ||
      widthValue.getInt() != k.getGlobalSize()[0])
    return emitOpError("load must address the read-only 2D input extent");
  return success();
}
LogicalResult RawUnpackOp::verify() {
  if (getByteOffset() != 0 || getFormat() != "rgba8_unorm" ||
      !getTexel().getDefiningOp<TextureLoadOp>())
    return emitOpError(
        "raw_unpack requires a complete point-loaded RGBA8 word");
  return success();
}
} // namespace vrc::vrcgpu
