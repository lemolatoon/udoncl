# UdonATen

UdonATen owns tensor operators that PyTorch implements in ATen CUDA kernels
rather than cuBLAS or cuDNN. Its public C surface mirrors ATen operator names,
argument order, and semantics as closely as the explicit-buffer Udon runtime
allows.

The first supported inference operators are:

- `udonatenAdd`, `udonatenSub`, `udonatenMul`, `udonatenRelu`,
  `udonatenSigmoid`, and `udonatenTanh`, corresponding to the ATen pointwise
  tensor schemas. Add and Sub retain ATen's `alpha` argument; the binary calls
  support the broadcast shapes accepted by the current four-axis backend.
  Exact-shape tensors larger than 512 elements are composed internally from
  independent 512-word Device tiles without changing the public ATen-shaped
  call; the 14,336-element Llama MLP shape is covered by the native Device
  test. Broadcast operands above 512 output elements still return
  `NOT_SUPPORTED` until coordinate-aware tile addressing is implemented.
  Their public ownership follows PyTorch's TensorIterator CUDA implementation;
  the private Device path currently reuses validated UdonDNN OpTensor and
  Activation kernels without exposing those cuDNN-shaped calls to ONNX plans.
- `udonatenEmbedding`, corresponding to
  `embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)`. The
  current backend supports `padding_idx=-1`, `scale_grad_by_freq=false`, and
  `sparse=false`. Its Device kernel and artifact are UdonATen-owned; no Gather
  extension is exposed from the cuDNN-compatible UdonDNN API.
- `udonatenRmsNorm`, corresponding to
  `rms_norm(input, normalized_shape, weight=None, eps=None)`. Optional weight
  is supported. A two-pass UdonATen Device path reduces 64-element chunks into
  a partial-sum texture, then binds the original state and partial sums as two
  read-only textures while normalizing up to 4,096 elements in parallel. Large
  multi-row tensors remain `NOT_SUPPORTED`; no CPU fallback is used.
- `udonatenTile`, corresponding to ATen `tile(input, dims)`. Repeat dimensions
  align to the trailing input dimensions and omitted leading repeats are one,
  matching ATen's `tile_symint` delegation to `repeat`. The current four-axis
  Device profile copies up to 4,096 float elements with one independent
  fragment per output element; ranks above four and zero-sized outputs return
  `NOT_SUPPORTED`.
Pointwise operations currently adapt validated UdonDNN backend kernels as a
private implementation detail. Embedding and both the small packed and large
two-pass RMSNorm Device paths are implemented and packaged directly in
UdonATen.
Tile has a UdonATen-owned Device kernel. ONNX Runtime extension quantized
matrix multiplication belongs to `UdonONNXOperators`, not UdonATen.
`libraries/udondnn/run_device.py` packages Tile as the reusable
`unity-udonaten-tile-bundle`; the Unity importer creates
`UdonAtenTile.prefab`, and its SDK IUdonVM lifecycle fixture validates all 72
outputs for trailing-dimension repeats before removing generated assets.

See [operator ownership](../../docs/architecture/operator-ownership.md) for
the source audit and compatibility rules.
