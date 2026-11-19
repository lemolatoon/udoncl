# Operator ownership

Public APIs follow the component that implements the corresponding PyTorch
CUDA operation. GPU execution alone does not make an operation cuBLAS or
cuDNN.

| Operation | PyTorch CUDA owner | Udon public owner | Interface contract |
| --- | --- | --- | --- |
| GEMM, GEMV, and DOT | cuBLAS-shaped ATen backend calls | UdonBLAS | CUDA 13.0 cuBLAS argument, layout, enum, and status semantics |
| Explicit cuDNN convolution, activation, pooling, softmax, batch normalization, and OpTensor calls | cuDNN Legacy API | UdonDNN | matching cuDNN API after the prefix |
| Add, subtract, multiply, ReLU, sigmoid, tanh | ATen pointwise schemas and TensorIterator CUDA kernels | UdonATen | corresponding ATen tensor argument order, alpha, broadcasting, and unary semantics |
| Embedding lookup | ATen `embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)`; implemented through `index_select` | UdonATen | weight and indices first, followed by the three ATen options |
| Gather/indexing | ATen CUDA scatter/gather and indexing kernels | UdonATen | ATen gather/index-select semantics; do not disguise it as cuDNN |
| RMSNorm | ATen `rms_norm(input, normalized_shape, weight=None, eps=None)` and the fused CUDA layer-norm kernel | UdonATen | normalized shape and optional weight/epsilon semantics; no cuDNN alpha/beta |
| Tile/repeat | ATen tensor-shape/repeat implementation | UdonATen | `tile(input, dims)` semantics |
| ONNX Runtime weight-only linear | `com.microsoft::MatMulNBits` extension | UdonONNXOperators | ONNX Runtime bits/block-size, packed or unpacked zero point, optional `g_idx`/bias, and FP32/FP16/BF16 activation contract |

Authoritative sources:

- NVIDIA CUDA 13.0 cuBLAS Level-2 and Level-3 APIs:
  <https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html>
- PyTorch ATen native operator registration and dispatch:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md>
- PyTorch CUDA BLAS backend declarations (`gemm`, `gemv`, and `dot`), which
  preserve the cuBLAS-shaped argument contracts:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.h>
- ATen pointwise schemas, including `add.Tensor`, `sub.Tensor`, `relu`, `tanh`,
  and their TensorIterator dispatch:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>
- ATen ReLU's `clamp_min(self, 0)` implementation:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp>
- ATen CUDA multiply, sigmoid, and tanh kernels:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/BinaryMulKernel.cu>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/UnarySpecialOpsKernel.cu>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/UnaryGeometricTanhKernel.cu>
- Embedding implementation:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Embedding.cpp>
- CUDA gather/scatter implementation:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScatterGatherKernel.cu>
- RMSNorm frontend and fused implementation:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu>
- ATen `tile` schema and its `repeat` decomposition (leading-one promotion,
  followed by a parallel copy):
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorShape.cpp>
- TorchAO INT4 weight-only configuration and dispatch:
  <https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py>
- ATen INT4 schema and CUDA implementation (`[M,K]` input, `[M,N]` output,
  group sizes 32/64/128/256):
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu>

Embedding now exposes only `udonatenEmbedding`; the former public
`udondnnExtGatherForward` shortcut has been removed. Its Device kernel is named
and packaged as UdonATen, following PyTorch's `embedding` delegation to
CUDA `index_select`. ONNX plans and generated Host code expose that same
UdonATen contract. Both the small packed RMSNorm kernel and the large two-pass
path are UdonATen-owned and are no longer named or packaged as UdonDNN
extensions. Its arena
profile executes up to 16,384 words with normalized rows up to 4,096 words as
an UdonATen-owned two-pass reduction and normalization, without CPU readback.
The optional ATen weight is a separate readonly texture input rather than a
packed-state offset. The same rule applies
to pointwise Add, Sub, Mul, ReLU, Sigmoid, and Tanh: their public calls are
UdonATen-shaped. Large ONNX pointwise tensors use the private
`udonaten_arena_pointwise` Device backend; this is an ATen execution detail and
does not add an UdonDNN extension or public API. Smaller validated paths may
still reuse compatible UdonDNN kernels privately. ONNX Gemm and Softmax arena
profiles are private Device backends of the existing `udonblasSgemm` and
`udondnnSoftmaxForward` contracts; they do not add public APIs. Tile is implemented directly
as the ATen-shaped `udonatenTile` operation. ONNX Tile uses the private
`udonaten_arena_tile` backend to write at texture-arena offset zero, without
adding a UdonDNN extension. ONNX Runtime extension `MatMulNBits` belongs to
`UdonONNXOperators`; its packed weights and zero-point formats are not exposed
as an ATen, UdonBLAS, or UdonDNN public extension. The superseded
`udonatenWeightInt4packMm` and private `udonblasBackendSgemvQ4` paths were
removed rather than retained as competing quantized-matrix APIs.
The Device-resident Q4 kernel `udononnx_matmul_nbits_q4` is a private
UdonONNXOperators schedule of that ONNX Runtime contract, not a public API:
one dispatch accumulates up to 256 K blocks of every output word in ascending
K order, and the Host splits a longer K into compile-time segments that pass
the partial sum through a scratch surface. Changing that schedule keeps the
`MatMulNBits` semantics and adds no `Ext` entry point.

The general ONNX MNIST path follows the same boundary: arena convolution and
pooling are private backends of `udondnnConvolutionForward` and
`udondnnPoolingForward`, while coordinate-aware broadcast Add remains an
UdonATen backend. These kernels only expand physical texture capacity; they do
not define alternate public signatures.
# ArgMax

`ArgMax` is owned by ATen native reduction, not cuBLAS, cuDNN, or TorchAO.
UdonATen follows PyTorch's CUDA `argmax_kernel_cuda` comparison semantics from
`aten/src/ATen/native/cuda/ReduceArgMaxKernel.cu` and
`aten/src/ATen/native/SharedReduceOps.h`: NaN wins over non-NaN and ties select
the lowest index.  The Udon backend may separately request rejection of any
non-finite input for conversation-result validation; that policy does not alter
the public ATen-compatible operation.
