#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "VRC/Dialect/UdonHL/UdonHLEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "VRC/Dialect/UdonHL/UdonHLAttrs.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "VRC/Dialect/UdonHL/UdonHLTypes.cpp.inc"
#include "VRC/Dialect/UdonHL/UdonHLDialect.cpp.inc"
void vrc::udonhl::UdonHLDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "VRC/Dialect/UdonHL/UdonHLAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "VRC/Dialect/UdonHL/UdonHLTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "VRC/Dialect/UdonHL/UdonHLOps.cpp.inc"
      >();
}
#define GET_OP_CLASSES
#include "VRC/Dialect/UdonHL/UdonHLOps.cpp.inc"

mlir::LogicalResult vrc::udonhl::CallOp::verify() {
  if (!(*this)->getParentOfType<mlir::func::FuncOp>() ||
      !mlir::isa<mlir::ModuleOp>(
          (*this)->getParentOfType<mlir::func::FuncOp>()->getParentOp()))
    return emitOpError("C call must belong to a Host func.func");
  if (!getCallee().size())
    return emitOpError("requires a nonempty callee");
  return mlir::success();
}
mlir::LogicalResult
vrc::udonhl::CallOp::verifySymbolUses(mlir::SymbolTableCollection &symbols) {
  auto callee = symbols.lookupNearestSymbolFrom<mlir::func::FuncOp>(
      *this, getCalleeAttr());
  if (!callee)
    return emitOpError("callee must resolve to func.func");
  auto type = callee.getFunctionType();
  if (getOperandTypes() != type.getInputs() ||
      getResultTypes() != type.getResults())
    return emitOpError("callee argument/result types do not match");
  auto effect = callee->getAttrOfType<mlir::StringAttr>("vrc.async_effect");
  if (effect && effect.getValue() == "may_suspend" &&
      getCallPolicy() != CallPolicy::MaySuspend)
    return emitOpError("may_suspend callee requires may_suspend call policy");
  if (getCallPolicy() == CallPolicy::MaySuspend) {
    auto caller = (*this)->getParentOfType<mlir::func::FuncOp>();
    auto callerEffect =
        caller ? caller->getAttrOfType<mlir::StringAttr>("vrc.async_effect")
               : mlir::StringAttr();
    if (!callerEffect || callerEffect.getValue() != "may_suspend")
      return emitOpError("may_suspend must propagate to the caller");
  }
  return mlir::success();
}

namespace {
mlir::LogicalResult verifyFrameLayout(mlir::Operation *op, int64_t version,
                                      mlir::ArrayAttr slots) {
  if (version != 1)
    return op->emitOpError("requires continuation frame layout version 1");
  for (auto [index, slot] : llvm::enumerate(slots)) {
    auto type = mlir::dyn_cast<mlir::TypeAttr>(slot);
    if (!type)
      return op->emitOpError("slot_types must contain only type attributes");
    auto valueType = type.getValue();
    if (!valueType.isInteger(32) && !valueType.isF32())
      return op->emitOpError("frame slot ")
             << index << " has a non-persistable type " << valueType;
  }
  return mlir::success();
}

mlir::LogicalResult verifyFrameAccess(mlir::Operation *op, mlir::Value frame,
                                      int64_t version, int64_t slot,
                                      mlir::ArrayAttr slots,
                                      mlir::Type valueType) {
  if (mlir::failed(verifyFrameLayout(op, version, slots)))
    return mlir::failure();
  if (slot < 0 || static_cast<size_t>(slot) >= slots.size())
    return op->emitOpError("frame slot is outside slot_types");
  auto expected = mlir::cast<mlir::TypeAttr>(slots[slot]).getValue();
  if (expected != valueType)
    return op->emitOpError("frame slot type does not match the accessed value");
  if (auto create = frame.getDefiningOp<vrc::udonhl::FrameCreateOp>())
    if (create.getLayoutVersion() != version || create.getSlotTypes() != slots)
      return op->emitOpError("frame access layout does not match frame_create");
  return mlir::success();
}

bool isVRCEvent(mlir::Type type) {
  std::string spelling;
  llvm::raw_string_ostream stream(spelling);
  type.print(stream);
  return stream.str() == "!vrc.event";
}

bool maySuspend(mlir::func::FuncOp function) {
  auto effect = function->getAttrOfType<mlir::StringAttr>("vrc.async_effect");
  return effect && effect == "may_suspend";
}
} // namespace

mlir::LogicalResult vrc::udonhl::FrameCreateOp::verify() {
  return verifyFrameLayout(*this, getLayoutVersion(), getSlotTypes());
}

mlir::LogicalResult vrc::udonhl::FrameStoreOp::verify() {
  return verifyFrameAccess(*this, getFrame(), getLayoutVersion(), getSlot(),
                           getSlotTypes(), getValue().getType());
}

mlir::LogicalResult vrc::udonhl::FrameLoadOp::verify() {
  return verifyFrameAccess(*this, getFrame(), getLayoutVersion(), getSlot(),
                           getSlotTypes(), getValue().getType());
}

mlir::LogicalResult vrc::udonhl::ResumeDispatchOp::verify() {
  auto function = (*this)->getParentOfType<mlir::func::FuncOp>();
  if (!function || !maySuspend(function))
    return emitOpError("requires a may_suspend Host function");
  auto states = getRegisteredStates();
  if (states.empty() || states.size() != getDestinations().size())
    return emitOpError("requires one destination per registered resume state");
  llvm::SmallDenseSet<int32_t> unique;
  for (auto indexedState : llvm::enumerate(states)) {
    size_t index = indexedState.index();
    int32_t resumeId = indexedState.value();
    if (resumeId < 0 || !unique.insert(resumeId).second)
      return emitOpError(
          "registered resume states must be unique and nonnegative");
    mlir::Block *destination = getDestinations()[index];
    if (destination->getParent() != (*this)->getBlock()->getParent() ||
        destination->getNumArguments() != 0)
      return emitOpError("resume destinations must be argument-free blocks in "
                         "the same function");
  }
  return mlir::success();
}

mlir::LogicalResult vrc::udonhl::SuspendOp::verify() {
  auto function = (*this)->getParentOfType<mlir::func::FuncOp>();
  if (!function || !maySuspend(function))
    return emitOpError("requires a may_suspend Host function");
  if (getResumeState() <= 0)
    return emitOpError("resume_state must be positive");
  if (getCompletionSlot() < 0)
    return emitOpError("completion_slot must be nonnegative");
  if (auto create = getFrame().getDefiningOp<FrameCreateOp>()) {
    auto slots = create.getSlotTypes();
    if (static_cast<size_t>(getCompletionSlot()) >= slots.size() ||
        !mlir::cast<mlir::TypeAttr>(slots[getCompletionSlot()])
             .getValue()
             .isInteger(32))
      return emitOpError("completion_slot must name an i32 frame slot");
  }
  if (getReason() == SuspendReason::Gpu &&
      (!getToken() || !isVRCEvent(getToken().getType())))
    return emitOpError("gpu suspension requires a !vrc.event token");
  if (getReason() == SuspendReason::Service &&
      (!getToken() || !getToken().getType().isInteger(32)))
    return emitOpError("service suspension requires an i32 token");
  if (getReason() == SuspendReason::Budget && getToken() &&
      !getToken().getType().isInteger(32))
    return emitOpError("budget suspension token must be i32 when present");
  bool registered = false;
  function.walk([&](ResumeDispatchOp dispatch) {
    registered |=
        llvm::is_contained(dispatch.getRegisteredStates(), getResumeState());
  });
  if (!registered)
    return emitOpError("resume_state is not registered by resume_dispatch");
  return mlir::success();
}
