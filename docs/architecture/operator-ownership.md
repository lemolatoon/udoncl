# Operator ownership

Public APIs follow the component that implements the corresponding PyTorch
CUDA operation. GPU execution alone does not make an operation cuBLAS or
cuDNN.

| Operation | PyTorch CUDA owner | Udon public owner | Interface contract |
| --- | --- | --- | --- |
| GEMM | cuBLAS-shaped ATen backend call | UdonBLAS | cuBLAS argument and status semantics |
| Explicit cuDNN convolution, activation, pooling, softmax, batch normalization, and OpTensor calls | cuDNN Legacy API | UdonDNN | matching cuDNN API after the prefix |
| Add, multiply, sigmoid | ATen pointwise schemas and TensorIterator CUDA kernels | UdonATen | `add.Tensor(self, other, *, alpha=1)`, `mul.Tensor(self, other)`, and `sigmoid(self)` semantics |
| Embedding lookup | ATen `embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)`; implemented through `index_select` | UdonATen | weight and indices first, followed by the three ATen options |
| Gather/indexing | ATen CUDA scatter/gather and indexing kernels | UdonATen | ATen gather/index-select semantics; do not disguise it as cuDNN |
| RMSNorm | ATen `rms_norm(input, normalized_shape, weight=None, eps=None)` and the fused CUDA layer-norm kernel | UdonATen | normalized shape and optional weight/epsilon semantics; no cuDNN alpha/beta |
| Tile/repeat | ATen tensor-shape/repeat implementation | UdonATen | `tile(input, dims)` semantics |
| INT4 weight-only linear | TorchAO quantized linear dispatch to ATen `_weight_int4pack_mm(input, packed_weight, group_size, scales_and_zeros)` | UdonATen quantized operator | separate packed weight and scale/zero metadata; not a cuBLAS extension |

Authoritative sources:

- PyTorch ATen native operator registration and dispatch:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md>
- ATen pointwise schemas, including `add.Tensor` and its TensorIterator
  dispatch:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>
- ATen CUDA multiply and sigmoid kernels:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/BinaryMulKernel.cu>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/UnarySpecialOpsKernel.cu>
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

The existing `udondnnExtGatherForward` and
`udondnnExtRmsNormalizationForward` names are provisional backend details.
UdonATen may reuse their Device kernels internally during migration, but ONNX
plans and generated Host code expose UdonATen contracts. The same rule applies
to pointwise Add, Mul, and Sigmoid: their public calls are UdonATen-shaped even
while their validated Device implementation privately reuses UdonDNN OpTensor
and Activation kernels. Tile is implemented
directly as the ATen-shaped `udonatenTile` operation and Device kernel, without
adding a UdonDNN extension. INT4 AOT now uses separate
packed-weight and `[K/group_size,N,2]` bfloat16 scale/zero inputs and the
`udonatenWeightInt4packMm` contract. Its Device implementation remains
an UdonATen-owned kernel, with one fragment per output feature and numerical
coverage for non-zero zero points. External-weight paging is still required to
execute the complete 8B plan. The public call has no beta parameter; private
Device tiles use replace-then-add accumulation to preserve whole-operator ATen
semantics across K pages.
