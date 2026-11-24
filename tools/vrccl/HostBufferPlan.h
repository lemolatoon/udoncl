#ifndef UDONCL_TOOLS_VRCCL_HOSTBUFFERPLAN_H
#define UDONCL_TOOLS_VRCCL_HOSTBUFFERPLAN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace udoncl {

/// The handles of a Unity library's OpenCL runtime: the `baseCapacity`
/// handles of its ONNX Host, then one Device buffer of each extra byte size
/// per allocation context of its C Host.
struct HostBufferPlan {
  int64_t baseCapacity = 0;
  llvm::SmallVector<int64_t> extraBufferBytes;
};

mlir::LogicalResult writeHostBufferPlan(const HostBufferPlan &plan,
                                        llvm::StringRef jsonPath,
                                        llvm::StringRef headerPath,
                                        llvm::raw_ostream &diagnostics);

/// Reads the plan file into `vrc.host_buffer_plan` on `module`, which
/// vrc-apply-host-buffer-plan applies.
mlir::LogicalResult readHostBufferPlan(mlir::ModuleOp module,
                                       llvm::StringRef jsonPath);

} // namespace udoncl

#endif
