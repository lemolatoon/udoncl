# Operator ownership

Public APIs follow the component that implements the corresponding PyTorch
CUDA operation. GPU execution alone does not make an operation cuBLAS or
cuDNN.

| Operation | PyTorch CUDA owner | Udon public owner | Interface contract |
| --- | --- | --- | --- |
| GEMM | cuBLAS-shaped ATen backend call | UdonBLAS | cuBLAS argument and status semantics |
| Convolution, activation, pooling, softmax, batch normalization, OpTensor | cuDNN-shaped ATen backend calls | UdonDNN | matching cuDNN API after the prefix |
| Embedding lookup | ATen `embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)`; implemented through `index_select` | UdonATen | weight and indices first, followed by the three ATen options |
| Gather/indexing | ATen CUDA scatter/gather and indexing kernels | UdonATen | ATen gather/index-select semantics; do not disguise it as cuDNN |
| RMSNorm | ATen `rms_norm(input, normalized_shape, weight=None, eps=None)` and the fused CUDA layer-norm kernel | UdonATen | normalized shape and optional weight/epsilon semantics; no cuDNN alpha/beta |
| Tile/repeat | ATen tensor-shape/repeat implementation | UdonATen | `tile(input, dims)` semantics |
| INT4 weight-only linear | TorchAO quantized linear dispatch to ATen `_weight_int4pack_mm(input, packed_weight, group_size, scales_and_zeros)` | UdonATen quantized operator | separate packed weight and scale/zero metadata; not a cuBLAS extension |

Authoritative sources:

- PyTorch ATen native operator registration and dispatch:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md>
- Embedding implementation:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Embedding.cpp>
- CUDA gather/scatter implementation:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScatterGatherKernel.cu>
- RMSNorm frontend and fused implementation:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu>
- TorchAO INT4 weight-only configuration and dispatch:
  <https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py>

The existing `udondnnExtGatherForward`, `udondnnExtRmsNormalizationForward`,
and `udondnnExtTileForward` names are provisional backend details. UdonATen may
reuse their Device kernels internally during migration, but ONNX plans and
generated Host code expose UdonATen contracts. INT4 AOT now uses separate
packed-weight and `[K/group_size,N,2]` bfloat16 scale/zero inputs and the
`udonatenWeightInt4packMm` contract. Its Device implementation remains
explicitly `NOT_SUPPORTED` until paging and the tiled kernel are connected.
