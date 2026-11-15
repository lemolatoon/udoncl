#ifndef VRC_ANALYSIS_HOSTSTORAGELIVENESS_H
#define VRC_ANALYSIS_HOSTSTORAGELIVENESS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace vrc {

struct HostStorageInterference {
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value>>
      functionValues;
  llvm::DenseMap<mlir::Operation *,
                 llvm::SmallVector<llvm::SmallVector<mlir::Value>>>
      simultaneousValueGroups;
  llvm::DenseMap<mlir::Operation *, std::string> operationIds;
  llvm::DenseMap<mlir::Value, std::string> valueIds;
  llvm::DenseMap<mlir::Value, std::string> storageKeys;
  llvm::DenseMap<mlir::Value, unsigned> ordinals;
  llvm::DenseMap<mlir::Value, llvm::DenseSet<mlir::Value>> interference;
};

mlir::FailureOr<HostStorageInterference> analyzeHostStorageInterference(
    mlir::ModuleOp module, llvm::ArrayRef<mlir::func::FuncOp> functions,
    llvm::function_ref<mlir::FailureOr<std::string>(mlir::Type)>
        physicalStorageKey,
    const llvm::DenseSet<mlir::Operation *> &materializedConstants,
    bool includeNestedRegions = false);

} // namespace vrc

#endif // VRC_ANALYSIS_HOSTSTORAGELIVENESS_H
