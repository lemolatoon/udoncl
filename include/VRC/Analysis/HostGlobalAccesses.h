#ifndef VRC_ANALYSIS_HOSTGLOBALACCESSES_H
#define VRC_ANALYSIS_HOSTGLOBALACCESSES_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace vrc {

/// A load or store of one typed cell of a Host global: the bytes
/// [byteOffset, byteOffset + byteWidth) from the start of the global.
struct HostGlobalAccess {
  mlir::Operation *operation;
  int64_t byteOffset;
  uint64_t byteWidth;
  mlir::Type type;
  bool isStore;
};

/// Every use of the address of one `udonhl.global`.
struct HostGlobalAccessSummary {
  /// Loads and stores through the address plus a constant byte offset.
  llvm::SmallVector<HostGlobalAccess> accesses;
  /// The `udonhl.address_of` and constant `udonhl.ptr_add` operations that
  /// compute the accessed addresses.
  llvm::SmallVector<mlir::Operation *> addressComputations;
  /// Why the global needs a byte address, or empty when `accesses` are all the
  /// uses of its address. The address is then unobservable: the global is a
  /// set of independent cells.
  llvm::StringRef escape;

  bool hasStore() const;
};

/// Bytes a `udonhl.load` or `udonhl.store` of `type` covers in the raw
/// memory model, or std::nullopt for a type without a fixed scalar layout.
std::optional<uint64_t> hostMemoryAccessBytes(mlir::Type type);

/// Summarizes the address uses of every `udonhl.global` of `module`, keyed
/// by symbol name. A global whose address is never taken has an empty
/// summary.
llvm::StringMap<HostGlobalAccessSummary>
analyzeHostGlobalAccesses(mlir::ModuleOp module);

} // namespace vrc

#endif // VRC_ANALYSIS_HOSTGLOBALACCESSES_H
