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
inline constexpr llvm::StringLiteral onnxHostBindingStateSymbol =
    "__udon_onnx_binding_state";
inline constexpr llvm::StringLiteral onnxHostBindingInputSymbol =
    "__udon_onnx_binding_inputs";
inline constexpr llvm::StringLiteral onnxHostBindingOutputSymbol =
    "__udon_onnx_binding_outputs";
inline constexpr llvm::StringLiteral onnxHostBindingEventSymbol =
    "__udon_onnx_binding_event";
inline constexpr llvm::StringLiteral onnxHostBindingIoHandlesSymbol =
    "__udon_onnx_io_handles";
inline constexpr llvm::StringLiteral onnxHostSequenceStateOwnedSymbol =
    "__udon_onnx_sequence_state_owned";
inline constexpr llvm::StringLiteral onnxHostSequenceStatePendingSymbol =
    "__udon_onnx_sequence_state_pending";
inline constexpr llvm::StringLiteral onnxHostSequenceStatePoisonedSymbol =
    "__udon_onnx_sequence_state_poisoned";
inline constexpr llvm::StringLiteral onnxHostSequenceStateZeroSymbol =
    "__udon_onnx_sequence_state_zero";
inline constexpr llvm::StringLiteral onnxHostHasRunSymbol =
    "__udon_onnx_has_run";

inline constexpr int32_t onnxHostMetadataStorageBytes = 64;
inline constexpr int32_t onnxHostModelStorageBytes = 64;
inline constexpr int32_t onnxHostBindingStateStorageBytes = 256;
inline constexpr int32_t onnxHostBindingTensorStorageBytes = 128;
inline constexpr int32_t onnxHostBindingBufferHandleBytes = 8;
inline constexpr int32_t onnxHostMaximumTensorRank = 8;

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
} // namespace vrc

#endif
