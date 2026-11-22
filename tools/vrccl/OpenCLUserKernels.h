#ifndef UDONCL_TOOLS_VRCCL_OPENCLUSERKERNELS_H
#define UDONCL_TOOLS_VRCCL_OPENCLUSERKERNELS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llvm {
class Module;
class raw_ostream;
} // namespace llvm

namespace udoncl {

/// An OpenCL C kernel of the Device libraries that the Host C creates by name
/// (`clCreateKernel`) and that no UdonCL library kernel of the OpenCL AOT
/// runtime provides. The linked runtime dispatches it through the intrinsic
/// `__vrc_opencl_dispatch_user_<id>_<bufferCount>_<name>`, whose operands are
/// the physical tokens of its buffer arguments followed by its scalars.
struct OpenCLUserKernel {
  std::string name;
  int32_t id;
  /// The leading global buffer arguments; the last one is written.
  unsigned bufferCount;
  /// 'i' (int or uint) or 'f' (float) for every scalar after the buffers.
  std::string scalarKinds;
};

/// User kernels take the ids from here up, in name order; library kernels
/// stay below and synthesized kernels start at 4096.
inline constexpr int32_t firstOpenCLUserKernelId = 1024;
inline constexpr int32_t lastOpenCLUserKernelId = 4095;

struct OpenCLUserKernelSet {
  std::vector<OpenCLUserKernel> kernels;
  /// The UdonCL library kernels the Device libraries define.
  std::vector<std::string> libraryKernels;
};

/// Finds the user kernels of `device` that `host` creates by a constant name.
/// Fails, naming the kernel, when its arguments are not global buffers
/// followed by int, uint or float scalars.
mlir::FailureOr<OpenCLUserKernelSet>
discoverOpenCLUserKernels(const llvm::Module &host, const llvm::Module &device,
                          llvm::raw_ostream &diagnostics);

/// Writes `udoncl_aot_user_kernels.h`, which OpenCLAOT.c is compiled with
/// (`-include`), and `udoncl_aot_user_dispatch.h`, which it includes when
/// there are user kernels, into `directory`. Without user kernels the first
/// header is empty and the runtime is unchanged; with them, it also leaves
/// out the dispatch paths of the UdonCL libraries the Device libraries do not
/// define a kernel of.
mlir::LogicalResult
writeOpenCLUserKernelHeaders(const OpenCLUserKernelSet &set,
                             llvm::StringRef directory,
                             llvm::raw_ostream &diagnostics);

/// Binds every clEnqueueNDRangeKernel of a user kernel in the Host C `host`
/// to the runtime entry point `__vrc_enqueue_user_<id>_<bufferCount>_<name>`
/// with the values the last dominating clSetKernelArg of each argument set,
/// so the static Device resource analysis sees every buffer the launch
/// reads and writes. Fails, naming the kernel and the function, when a launch
/// may run more than one kernel, has a global work offset, or an argument is
/// not set by a dominating four-byte clSetKernelArg in its function.
mlir::LogicalResult
bindOpenCLUserKernelLaunches(llvm::Module &host,
                             llvm::ArrayRef<OpenCLUserKernel> kernels,
                             llvm::raw_ostream &diagnostics);

/// Describes every user kernel dispatch intrinsic of the linked Unity Host
/// `host` with `vrc.device_dispatch`, so LowerOpenCLAOTIntrinsics lowers it
/// like a library kernel. The NDRange is the `global_size` of a hand-written
/// `device_kernels` entry of that kernel or, without one, the constant
/// global work size every `clEnqueueNDRangeKernel` of the kernel in the Host C
/// module `hostLLVMPath` passes. Returns the user kernel names by id.
mlir::FailureOr<std::map<int64_t, std::string>> describeOpenCLUserDispatches(
    mlir::ModuleOp host, llvm::StringRef hostLLVMPath,
    llvm::ArrayRef<const llvm::json::Object *> explicitKernels,
    llvm::raw_ostream &diagnostics);

} // namespace udoncl

#endif
