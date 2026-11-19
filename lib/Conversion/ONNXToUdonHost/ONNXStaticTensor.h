#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXSTATICTENSOR_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXSTATICTENSOR_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace vrc {
/// Module attribute listing every immutable static tensor record.
inline constexpr llvm::StringLiteral onnxStaticTensorsAttr =
    "udoncl.static_tensors";
/// Declaration attribute carrying one static tensor record.
inline constexpr llvm::StringLiteral onnxStaticTensorAttr =
    "udoncl.static_tensor";

/// Returns true when `value` has a representation an immutable static tensor
/// can hold: a ranked, static-shape, non-empty tensor of a byte-sized
/// supported element type.
bool isStaticTensorRepresentable(mlir::DenseElementsAttr value);

/// Replaces the single-result `operation` producing `value` with a call to a
/// new private declaration carrying a static tensor record with resource
/// `id`. When `directory` is non-empty the payload is written there as
/// `tensor_<id>.bytes`; the dense data stays inline in the record when
/// `keepInlineData` is set or no directory is given. Returns the record.
mlir::FailureOr<mlir::DictionaryAttr>
replaceWithStaticTensor(mlir::ModuleOp module, mlir::Operation *operation,
                        mlir::DenseElementsAttr value, int64_t id,
                        llvm::StringRef name, llvm::StringRef directory,
                        bool keepInlineData);
} // namespace vrc

#endif
