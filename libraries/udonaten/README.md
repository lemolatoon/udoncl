# UdonATen

UdonATen owns tensor operators that PyTorch implements in ATen CUDA kernels
rather than cuBLAS or cuDNN. Its public C surface mirrors ATen operator names,
argument order, and semantics as closely as the explicit-buffer Udon runtime
allows.

The first supported inference operators are:

- `udonatenEmbedding`, corresponding to
  `embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)`. The
  current backend supports `padding_idx=-1`, `scale_grad_by_freq=false`, and
  `sparse=false`.
- `udonatenRmsNorm`, corresponding to
  `rms_norm(input, normalized_shape, weight=None, eps=None)`. The current
  backend requires a weight tensor and supports normalized regions through the
  existing 512-element Device profile.
- `udonatenWeightInt4packMm`, corresponding exactly in semantic argument order
  to ATen `_weight_int4pack_mm(input, packed_weight, group_size,
  scales_and_zeros)`. The external-data and AOT contracts are implemented;
  the Device entry point currently returns `NOT_SUPPORTED` until its tiled
  kernel and paging path are connected.

The implementation currently adapts the already validated UdonDNN prototype
kernels. Those functions are backend details, not the public ownership model.
Tile/repeat and the INT4 Device implementation remain to be migrated before
the Llama 8B operator plan can emit an executable Host.

See [operator ownership](../../docs/architecture/operator-ownership.md) for
the source audit and compatibility rules.
