# Generate logical bindings and artifact capacity from ONNX MLIR

After ONNX import, C++ passes generate every model Logical Tensor Binding,
unrolled physical allocation/bind operation, and the maximum live mutable
handle capacity. Empty logical tensors may use nonempty capacity allocations,
and padded surfaces keep exact logical lengths; physical packing and reuse stay
private. This avoids handwritten 47/45 C or JSON tables, fixed global capacity
increases, and model-name branches while leaving the public lifecycle in an
ordinary user-authored C Host.
