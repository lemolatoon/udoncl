#ifndef UDONCL_TOOLS_VRCCL_HOSTBUFFERPLAN_H
#define UDONCL_TOOLS_VRCCL_HOSTBUFFERPLAN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace llvm {
class Module;
class raw_ostream;
} // namespace llvm

namespace udoncl {

struct HostBufferPlan {
  int64_t baseCapacity = 0;
  llvm::SmallVector<int64_t> extraBufferBytes;
};

mlir::FailureOr<HostBufferPlan> planHostBuffers(llvm::Module &module,
                                                int64_t baseCapacity,
                                                llvm::raw_ostream &diagnostics);

mlir::LogicalResult writeHostBufferPlan(const HostBufferPlan &plan,
                                        llvm::StringRef jsonPath,
                                        llvm::StringRef headerPath,
                                        llvm::raw_ostream &diagnostics);

mlir::LogicalResult applyHostBufferPlan(mlir::ModuleOp module,
                                        llvm::StringRef jsonPath);

} // namespace udoncl

#endif
