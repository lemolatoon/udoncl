#ifndef UDONCL_TOOLS_VRCCL_UNITYONNXBUFFERS_H
#define UDONCL_TOOLS_VRCCL_UNITYONNXBUFFERS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <set>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace udoncl {

/// The Unity buffer name of the ONNX public tensor `name`: its characters
/// outside [A-Za-z0-9_] become '_', a name that does not start with a letter
/// or '_' gets the prefix `tensor_`, and a name already in `used` gets the
/// suffix `_input_<index>` or `_output_<index>`. Returns an empty string
/// when the suffixed name is taken as well; inserts the result into `used`.
std::string onnxPublicBufferName(llvm::StringRef name, bool output,
                                 int64_t index, std::set<std::string> &used);

/// Derives the `buffers` of a Unity library whose Host is an ONNX Host (a
/// module with `udoncl.onnx_binding`) and stores them in `config`.
///
/// The ONNX execution plan fixes every Device buffer of the Host: buffer ids
/// 1 to `udoncl.mutable_handle_capacity` with the capacities of
/// `udoncl.mutable_buffer_bytes` (the planned ONNX buffers, then the ones the
/// Host buffer plan adds for the user C Host). With
/// `generate_onnx_public_buffers`, the ONNX inputs and then the outputs, in
/// graph order, are buffers 1 to N; each gets its public name, element type,
/// logical size, access and index from the ONNX binding. Every other buffer
/// is private and named `private_<id>`.
///
/// Leaves `config` unchanged for a Host without ONNX binding metadata, and
/// fails when `generate_onnx_public_buffers` (--onnx-public-tensor-buffers)
/// is set there.
mlir::LogicalResult deriveOnnxUnityBuffers(mlir::ModuleOp host,
                                           llvm::json::Object &config,
                                           llvm::raw_ostream &diagnostics);

} // namespace udoncl

#endif
