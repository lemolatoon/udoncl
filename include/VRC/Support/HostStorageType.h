#ifndef VRC_SUPPORT_HOSTSTORAGETYPE_H
#define VRC_SUPPORT_HOSTSTORAGETYPE_H

#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LogicalResult.h"
#include "udoncl/array.h"
#include "udoncl/string.h"
#include <string>

namespace vrc {

inline mlir::FailureOr<llvm::StringRef> hostStorageUdonType(mlir::Type type) {
  if (type.isInteger(1))
    return llvm::StringRef("SystemBoolean");
  if (type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
      type.isIndex())
    return llvm::StringRef("SystemInt32");
  if (type.isInteger(64))
    return llvm::StringRef("SystemInt64");
  if (type.isF32())
    return llvm::StringRef("SystemSingle");
  if (mlir::isa<::vrc::vrc::BufferType, ::vrc::vrc::EventType,
                ::vrc::vrc::DispatchDescType, ::vrc::udonhl::FrameType>(type))
    return llvm::StringRef("SystemInt32");
  if (auto pointer = mlir::dyn_cast<::vrc::udonhl::PtrType>(type)) {
    if (pointer.getAddressSpace() == UDONCL_STRING_ADDRESS_SPACE)
      return llvm::StringRef("SystemString");
    if (pointer.getAddressSpace() == UDONCL_INT32_ARRAY_ADDRESS_SPACE)
      return llvm::StringRef("SystemInt32Array");
    return llvm::StringRef("SystemInt32");
  }
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
    if (memref.getRank() != 1 || !memref.getLayout().isIdentity())
      return mlir::failure();
    if (memref.getElementType().isF32())
      return llvm::StringRef("SystemSingleArray");
    if (memref.getElementType().isInteger(32))
      return llvm::StringRef("SystemInt32Array");
  }
  return mlir::failure();
}

inline mlir::FailureOr<std::string> hostStorageKey(mlir::Type type) {
  auto physicalType = hostStorageUdonType(type);
  if (mlir::failed(physicalType))
    return mlir::failure();
  std::string key = physicalType->str();
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
    key += "_n";
    if (memref.hasStaticShape())
      key += std::to_string(memref.getNumElements());
    else
      key += "dynamic";
  }
  return key;
}

} // namespace vrc

#endif // VRC_SUPPORT_HOSTSTORAGETYPE_H
