#ifndef UDONCL_TOOLS_VRCCL_UNITYDEVICEKERNELS_H
#define UDONCL_TOOLS_VRCCL_UNITYDEVICEKERNELS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <map>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace udoncl {

/// Derives the `device_kernels` entries of the library Device kernels that
/// the Unity Host `host` dispatches.
///
/// Every `vrc.dispatch` of `host` whose kernel id is neither in
/// `explicitKernelIds` (the hand-written entries, which take precedence)
/// nor in `synthesizedKernelIds` names a kernel of the UdonCL Device
/// libraries through the OpenCL AOT intrinsic it was lowered from. Its entry
/// takes the launch (argument kinds, NDRange, words) from the dispatch sites
/// and the access contract (input words, argument bounds, fragment cost)
/// from the library kernel contract, which holds the element limits the ONNX
/// Host planner and the OpenCL AOT runtime check before they dispatch the
/// kernel. The table an axis-0 Gather reads is the one contract that grows
/// with the constant scalars of its dispatch sites.
///
/// Fails, after printing to `diagnostics` which entry to write by hand, when
/// a dispatched kernel id has no library contract, when the dispatch sites of
/// one kernel disagree on its launch, when a constant scalar of a site lies
/// outside the argument bounds of the contract, or when a site-derived
/// contract meets a scalar that is not a compile-time constant. The result is
/// ordered by kernel id.
///
/// `userKernels` names the user OpenCL C kernels the linked OpenCL AOT
/// runtime dispatches (OpenCLUserKernels.h) by id. Their entries hold only
/// the launch: the gather profile when they read a buffer, otherwise one
/// invocation.
mlir::FailureOr<llvm::json::Array>
deriveUnityDeviceKernels(mlir::ModuleOp host,
                         llvm::ArrayRef<int64_t> explicitKernelIds,
                         llvm::ArrayRef<int64_t> synthesizedKernelIds,
                         const std::map<int64_t, std::string> &userKernels,
                         llvm::raw_ostream &diagnostics);

} // namespace udoncl

#endif
