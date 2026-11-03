# ONNX MNIST Dense through public UdonBLAS

This example generates the trained MNIST ONNX model, selects its
`[1,144] x [144,10]` Dense tail, and lowers it directly to the public
`udonblasSgemm(..., m=1, n=10, k=144, ...)` Host API. It does not emit or
interpret an ONNX execution-plan JSON.

The CMake target currently emits both `host.c`, which remains the executable
compatibility path, and `host.mlir`. The MLIR artifact contains the model
weights and bias as `memref.global` values and calls the public UdonBLAS ABI
directly. It is parse/verify gated now; linking that MLIR module with the
UdonBLAS implementation is the next migration step before the C generator is
removed.

Configure and build it against an installed UdonCL package in the same way as
the public `udonblas-sgemm` example. The resulting artifact exports scalar
input writes, the asynchronous Dense run, and scalar output reads so the whole
path can be exercised from the generated U# binding.
