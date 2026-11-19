#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTMATERIALIZATION_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTMATERIALIZATION_H

#include "llvm/ADT/StringRef.h"

namespace vrc {
inline constexpr llvm::StringLiteral onnxHostSourceAttr =
    "udoncl.onnx_host_source";
inline constexpr llvm::StringLiteral onnxHostGlobalsMaterializedAttr =
    "udoncl.onnx_host_globals_materialized";
inline constexpr llvm::StringLiteral onnxHostModelRegistryMaterializedAttr =
    "udoncl.onnx_host_model_registry_materialized";
inline constexpr llvm::StringLiteral onnxHostMetadataAttachedAttr =
    "udoncl.onnx_host_metadata_attached";
inline constexpr llvm::StringLiteral onnxHostFinalizedAttr =
    "udoncl.onnx_host_finalized";
inline constexpr llvm::StringLiteral onnxHostSourceCleanedAttr =
    "udoncl.onnx_host_source_cleaned";
} // namespace vrc

#endif
