#include "VRC/Conversion/Passes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace vrc {
#define GEN_PASS_DEF_EMITONNXHOSTMETADATAPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
struct EmitONNXHostMetadataPass
    : impl::EmitONNXHostMetadataPassBase<EmitONNXHostMetadataPass> {
  EmitONNXHostMetadataPass() = default;
  explicit EmitONNXHostMetadataPass(EmitONNXHostMetadataPassOptions options)
      : Base(std::move(options)) {}
  EmitONNXHostMetadataPass(const EmitONNXHostMetadataPass &other)
      : Base(other) {}

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    auto capacity = module->getAttrOfType<mlir::IntegerAttr>(
        "udoncl.mutable_handle_capacity");
    if (outputPath.empty()) {
      module.emitError("ONNX Host metadata output path is empty");
      signalPassFailure();
      return;
    }
    if (!capacity || capacity.getInt() < 0) {
      module.emitError("ONNX Host mutable handle capacity is missing");
      signalPassFailure();
      return;
    }

    std::error_code error;
    llvm::raw_fd_ostream output(outputPath, error, llvm::sys::fs::OF_Text);
    if (error) {
      module.emitError() << "cannot write ONNX Host metadata to " << outputPath
                         << ": " << error.message();
      signalPassFailure();
      return;
    }
    output << "{\n"
              "  \"schema_version\": 1,\n"
              "  \"mutable_handle_capacity\": "
           << capacity.getInt() << "\n}\n";
  }
};
} // namespace
} // namespace vrc
