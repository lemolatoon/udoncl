#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Support/HostLiteral.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "udoncl/array.h"
#include "udoncl/string.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/InliningUtils.h"
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
namespace {
struct UdonHLInlinerInterface final : mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const override {
    return true;
  }
};
} // namespace
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
  addInterfaces<UdonHLInlinerInterface>();
}
#define GET_OP_CLASSES
#include "VRC/Dialect/UdonHL/UdonHLOps.cpp.inc"

namespace {
bool isPowerOfTwo(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

mlir::LogicalResult verifyAlignment(mlir::Operation *operation,
                                    int64_t alignment) {
  if (!isPowerOfTwo(alignment) || alignment > (1LL << 31))
    return operation->emitOpError(
        "alignment must be a positive power of two no larger than 2^31");
  return mlir::success();
}

bool isMemoryAccessType(mlir::Type type) {
  if (mlir::isa<vrc::udonhl::PtrType, vrc::vrc::BufferType, vrc::vrc::EventType,
                vrc::vrc::DispatchDescType>(type))
    return true;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth() == 1 || integer.getWidth() == 8 ||
           integer.getWidth() == 16 || integer.getWidth() == 32 ||
           integer.getWidth() == 64;
  if (type.isF16() || type.isF32() || type.isF64())
    return true;
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type))
    return vector.hasStaticShape() &&
           isMemoryAccessType(vector.getElementType());
  return false;
}

mlir::LogicalResult verifyMatchingAddressSpaces(mlir::Operation *operation,
                                                mlir::Type destination,
                                                mlir::Type source) {
  auto destinationPointer = mlir::cast<vrc::udonhl::PtrType>(destination);
  auto sourcePointer = mlir::cast<vrc::udonhl::PtrType>(source);
  if (destinationPointer.getAddressSpace() != sourcePointer.getAddressSpace())
    return operation->emitOpError(
        "source and destination address spaces must match");
  return mlir::success();
}
} // namespace

mlir::LogicalResult vrc::udonhl::LiteralMaterializeOp::verify() {
  auto literal = mlir::dyn_cast<mlir::TypedAttr>(getLiteral());
  if (!literal || literal.getType() != getResult().getType())
    return emitOpError("literal type must match the result type");
  if (getInstanceId().empty())
    return emitOpError("instance_id must be nonempty");
  if (getGeneratedVersion() != 1)
    return emitOpError("generated_version must be 1");
  if (getStorageDomain() != "writable" || getColorAttr().getInt() < 0)
    return emitOpError("requires a nonnegative writable storage color");
  if (getHelperScratchAttr().getInt() < 0)
    return emitOpError("helper_scratch must be nonnegative");

  ::vrc::HostLiteralKey key =
      ::vrc::getHostLiteralKey(literal.getType(), literal);
  if (!key.supported || getLiteralKey() != key.id ||
      getPhysicalType() != key.physicalType)
    return emitOpError("literal key or physical type is not canonical");
  if (getRecipe() != "rematerialize_i32")
    return emitOpError("unsupported literal materialization recipe");
  auto expected = ::vrc::getHostLiteralI32Recipe(literal.getType(), literal);
  if (!expected || getHelperScratch() != 0)
    return emitOpError(
        "rematerialize_i32 requires a SystemInt32 literal and no helper "
        "scratch");
  if (getSeedBindings().size() != expected->seedValues.size())
    return emitOpError("seed_bindings do not match the selected recipe");
  llvm::DenseSet<int32_t> seedColors;
  for (auto [attribute, seed] :
       llvm::zip(getSeedBindings(), expected->seedValues)) {
    auto binding = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
    auto literalKey = binding ? binding.getAs<mlir::StringAttr>("literal_key")
                              : mlir::StringAttr();
    auto domain = binding ? binding.getAs<mlir::StringAttr>("storage_domain")
                          : mlir::StringAttr();
    auto physicalType = binding
                            ? binding.getAs<mlir::StringAttr>("physical_type")
                            : mlir::StringAttr();
    auto color = binding ? binding.getAs<mlir::IntegerAttr>("color")
                         : mlir::IntegerAttr();
    ::vrc::HostLiteralKey seedKey = ::vrc::getHostLiteralKey(
        mlir::IntegerType::get(getContext(), 32),
        mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 32), seed));
    if (!binding || !literalKey || literalKey.getValue() != seedKey.id ||
        !domain || domain.getValue() != "resident" || !physicalType ||
        physicalType.getValue() != "SystemInt32" || !color ||
        color.getInt() < 0 ||
        !seedColors.insert(static_cast<int32_t>(color.getInt())).second)
      return emitOpError("seed binding is noncanonical or aliases a seed");
  }

  if (getRequiredExterns().size() != expected->requiredExterns.size())
    return emitOpError("required_externs do not match the selected recipe");
  for (auto [attribute, expected] :
       llvm::zip(getRequiredExterns(), expected->requiredExterns)) {
    auto name = mlir::dyn_cast<mlir::StringAttr>(attribute);
    if (!name || name.getValue() != expected)
      return emitOpError("required_externs do not match the selected recipe");
  }
  return mlir::success();
}

mlir::LogicalResult vrc::udonhl::GlobalOp::verify() {
  if (!mlir::isa<mlir::ModuleOp>((*this)->getParentOp()))
    return emitOpError("must be defined directly inside a module");
  if (getLinkage() != "internal" && getLinkage() != "external")
    return emitOpError("linkage must be 'internal' or 'external'");
  if (getAddressSpace() < 0)
    return emitOpError("address_space must be nonnegative");
  if (mlir::failed(verifyAlignment(*this, getAlignment())))
    return mlir::failure();
  if (getLinkage() == "external" && getInitialValue())
    return emitOpError("external globals cannot have an initializer");
  if (getConstant() && !getInitialValue())
    return emitOpError("constant globals require an initializer");
  if (auto initial =
          mlir::dyn_cast_if_present<mlir::TypedAttr>(getInitialValueAttr()))
    if (initial.getType() != getStorageType() &&
        !(mlir::isa<PtrType>(getStorageType()) &&
          initial.getType().isInteger(32)) &&
        !mlir::isa<mlir::DenseI8ArrayAttr>(initial))
      return emitOpError("initializer type must match storage_type");
  return mlir::success();
}

mlir::LogicalResult vrc::udonhl::AddressOfOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbols) {
  auto global =
      symbols.lookupNearestSymbolFrom<GlobalOp>(*this, getGlobalAttr());
  if (!global)
    return emitOpError("global must resolve to udonhl.global");
  if (static_cast<int64_t>(getResult().getType().getAddressSpace()) !=
      global.getAddressSpace())
    return emitOpError("result address space must match the global");
  return mlir::success();
}

mlir::LogicalResult vrc::udonhl::AllocaOp::verify() {
  return verifyAlignment(*this, getAlignment());
}

mlir::LogicalResult vrc::udonhl::HeapAllocOp::verify() {
  return verifyAlignment(*this, getAlignment());
}

mlir::LogicalResult vrc::udonhl::PtrAddOp::verify() {
  if (getPointer().getType() != getResult().getType())
    return emitOpError("result address space must match the input pointer");
  return mlir::success();
}

mlir::LogicalResult vrc::udonhl::RefCmpOp::verify() {
  if (getLhs().getType() != getRhs().getType())
    return emitOpError("operands must have the same reference type");
  uint32_t addressSpace = getLhs().getType().getAddressSpace();
  if (addressSpace != UDONCL_STRING_ADDRESS_SPACE &&
      addressSpace != UDONCL_INT32_ARRAY_ADDRESS_SPACE)
    return emitOpError("operands must use a managed object address space");
  if (getPredicate() != "eq" && getPredicate() != "ne")
    return emitOpError("predicate must be 'eq' or 'ne'");
  return mlir::success();
}

mlir::LogicalResult vrc::udonhl::LoadOp::verify() {
  if (!isMemoryAccessType(getResult().getType()))
    return emitOpError("result must be a fixed-width scalar or vector");
  return verifyAlignment(*this, getAlignment());
}

mlir::LogicalResult vrc::udonhl::StoreOp::verify() {
  if (!isMemoryAccessType(getValue().getType()))
    return emitOpError() << "value type " << getValue().getType()
                         << " must be a fixed-width scalar or vector";
  return verifyAlignment(*this, getAlignment());
}

mlir::LogicalResult vrc::udonhl::CheckedAssignedBufferResolveOp::verify() {
  auto function = (*this)->getParentOfType<mlir::func::FuncOp>();
  if (!function || !mlir::isa<mlir::ModuleOp>(function->getParentOp()))
    return emitOpError("must belong to the Host runtime path");
  if (getLogicalHandle().getType().getAddressSpace() != 0 ||
      getOutputCell().getType().getAddressSpace() != 0)
    return emitOpError("requires raw address-space-zero pointers");
  if (!mlir::isa<::vrc::vrc::BufferType>(getAssignedBuffer().getType()))
    return emitOpError("requires a !vrc.buffer assigned-buffer operand");
  if (!getAssignedBuffer().getDefiningOp<::vrc::vrc::AssignedBufferOp>())
    return emitOpError("requires a directly assigned Unity buffer");
  auto output = getOutputCell().getDefiningOp<AllocaOp>();
  auto bytes =
      output ? output.getByteSize().getDefiningOp<mlir::arith::ConstantIntOp>()
             : mlir::arith::ConstantIntOp();
  if (!output || output.getAlignment() != 4 || !bytes || bytes.value() != 4)
    return emitOpError("requires an exact aligned four-byte output cell");
  return mlir::success();
}

mlir::LogicalResult vrc::udonhl::AsyncBindEventOp::verify() {
  if (!mlir::isa<::vrc::vrc::EventType>(getEvent().getType()))
    return emitOpError("event must have !vrc.event type");
  if (getObjectId() <= 0)
    return emitOpError("object_id must be positive");
  return mlir::success();
}

mlir::LogicalResult vrc::udonhl::MemcpyOp::verify() {
  return verifyMatchingAddressSpaces(*this, getDestination().getType(),
                                     getSource().getType());
}

mlir::LogicalResult vrc::udonhl::MemmoveOp::verify() {
  return verifyMatchingAddressSpaces(*this, getDestination().getType(),
                                     getSource().getType());
}

mlir::LogicalResult vrc::udonhl::ArrayViewOp::verify() {
  if (getElementByteSize() <= 0)
    return emitOpError("element_byte_size must be positive");
  if (getMemoryKind() == MemoryKind::Raw)
    return emitOpError("an Udon typed array view cannot declare raw backing");
  return mlir::success();
}

namespace {
mlir::LogicalResult verifyObjectAddressSpace(mlir::Operation *operation,
                                             vrc::udonhl::PtrType pointer,
                                             uint32_t expected) {
  if (pointer.getAddressSpace() != expected)
    return operation->emitOpError("object has the wrong address space");
  return mlir::success();
}
} // namespace

mlir::LogicalResult vrc::udonhl::StringLengthOp::verify() {
  return verifyObjectAddressSpace(*this, getString().getType(),
                                  UDONCL_STRING_ADDRESS_SPACE);
}
mlir::LogicalResult vrc::udonhl::StringCodeUnitOp::verify() {
  return verifyObjectAddressSpace(*this, getString().getType(),
                                  UDONCL_STRING_ADDRESS_SPACE);
}
mlir::LogicalResult vrc::udonhl::Int32ArrayLengthOp::verify() {
  return verifyObjectAddressSpace(*this, getArray().getType(),
                                  UDONCL_INT32_ARRAY_ADDRESS_SPACE);
}
mlir::LogicalResult vrc::udonhl::Int32ArrayGetOp::verify() {
  return verifyObjectAddressSpace(*this, getArray().getType(),
                                  UDONCL_INT32_ARRAY_ADDRESS_SPACE);
}
mlir::LogicalResult vrc::udonhl::Int32ArraySetOp::verify() {
  return verifyObjectAddressSpace(*this, getArray().getType(),
                                  UDONCL_INT32_ARRAY_ADDRESS_SPACE);
}

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
