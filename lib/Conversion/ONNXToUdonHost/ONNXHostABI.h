#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTABI_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTABI_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <string>

namespace vrc {
inline int32_t onnxHostDataType(mlir::Type type) {
  if (type.isInteger(1) || type.isInteger(8))
    return 1;
  if (type.isInteger(32))
    return 2;
  if (type.isInteger(64))
    return 3;
  if (type.isF16())
    return 4;
  if (type.isBF16())
    return 5;
  if (type.isF32())
    return 6;
  return 0;
}

inline unsigned onnxHostElementByteSize(mlir::Type type) {
  if (type.isInteger(1) || type.isInteger(8))
    return 1;
  if (type.isF16() || type.isBF16())
    return 2;
  if (type.isInteger(32) || type.isF32())
    return 4;
  if (type.isInteger(64))
    return 8;
  return 0;
}

inline llvm::StringRef onnxHostDataTypeName(mlir::Type type) {
  if (type.isInteger(1))
    return "bool";
  if (type.isInteger(8))
    return "uint8";
  if (type.isInteger(32))
    return "int32";
  if (type.isInteger(64))
    return "int64";
  if (type.isF16())
    return "float16";
  if (type.isBF16())
    return "bfloat16";
  if (type.isF32())
    return "float32";
  return "unknown";
}

inline constexpr llvm::StringLiteral onnxHostOperatorScratchSymbol =
    "__udon_onnx_operator_scratch";
inline constexpr llvm::StringLiteral onnxHostOperatorScratchOwnedSymbol =
    "__udon_onnx_operator_scratch_owned";
inline constexpr llvm::StringLiteral onnxHostModelStorageSymbol =
    "__udon_onnx_model_storage";
inline constexpr llvm::StringLiteral onnxHostInputMetadataSymbol =
    "__udon_onnx_input_metadata";
inline constexpr llvm::StringLiteral onnxHostOutputMetadataSymbol =
    "__udon_onnx_output_metadata";
inline constexpr llvm::StringLiteral onnxHostModelNameSymbol =
    "__udon_onnx_model_name";
inline constexpr llvm::StringLiteral onnxHostHasRunSymbol =
    "__udon_onnx_has_run";

inline constexpr int32_t onnxHostMetadataStorageBytes = 64;
inline constexpr int32_t onnxHostModelStorageBytes = 64;

inline std::string onnxHostIoNameSymbol(bool output, unsigned index) {
  return (llvm::Twine("__udon_onnx_") + (output ? "output" : "input") +
          "_name_" + llvm::Twine(index))
      .str();
}

inline std::string onnxHostIoDimensionsSymbol(bool output, unsigned index) {
  return (llvm::Twine("__udon_onnx_") + (output ? "output" : "input") +
          "_dimensions_" + llvm::Twine(index))
      .str();
}

inline std::string onnxHostInputBindingSymbol(unsigned index) {
  return (llvm::Twine("__udon_onnx_input_binding_") + llvm::Twine(index)).str();
}

inline std::string onnxHostDeviceBufferSymbol(unsigned index) {
  return (llvm::Twine("__udon_onnx_device_buffer_") + llvm::Twine(index)).str();
}

inline std::string onnxHostDeviceBufferOwnedSymbol(unsigned index) {
  return (llvm::Twine("__udon_onnx_device_buffer_owned_") + llvm::Twine(index))
      .str();
}

/// The `cl_mem` a UdonONNX session (udononnxBindInput/udononnxBindOutput)
/// bound to graph input or output `index`, or null while it is unbound.
inline std::string onnxHostBoundMemorySymbol(bool output, unsigned index) {
  return (llvm::Twine("__udon_onnx_bound_") + (output ? "output" : "input") +
          "_" + llvm::Twine(index))
      .str();
}

/// The largest extent each axis of graph input or output `index` takes: the
/// bound of its dimension plus its offset on the dynamic axis, 0 elsewhere.
inline std::string onnxHostIoDimensionBoundsSymbol(bool output,
                                                   unsigned index) {
  return (llvm::Twine("__udon_onnx_") + (output ? "output" : "input") +
          "_dimension_bounds_" + llvm::Twine(index))
      .str();
}

/// The run-time extent of the dynamic axis of graph input or output `index`
/// (an axis of a bounded dimension) a UdonONNX session bound.
inline std::string onnxHostBoundExtentSymbol(bool output, unsigned index) {
  return (llvm::Twine("__udon_onnx_bound_") + (output ? "output" : "input") +
          "_extent_" + llvm::Twine(index))
      .str();
}

/// The pitch of that dynamic axis in the bound buffer (buffer_dims).
inline std::string onnxHostBoundPitchSymbol(bool output, unsigned index) {
  return (llvm::Twine("__udon_onnx_bound_") + (output ? "output" : "input") +
          "_pitch_" + llvm::Twine(index))
      .str();
}

/// The value of bounded dimension `index` (in GraphRecord::dimensionBounds)
/// the current run derived from its bindings.
inline std::string onnxHostDimensionSymbol(unsigned index) {
  return (llvm::Twine("__udon_onnx_dimension_") + llvm::Twine(index)).str();
}
} // namespace vrc

#endif
