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
  `sparse=false`.
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
- `udonatenWeightInt4packMm`, corresponding exactly in semantic argument order
  to ATen `_weight_int4pack_mm(input, packed_weight, group_size,
  scales_and_zeros)`. Its Device kernel applies signed INT4 dequantization as
  `value * scale + zero`; the public input/output contract is `[M,K] → [M,N]`
  and one fragment independently owns each output feature for a matrix row.
  Each active Device tile supports group size 32, up to 64 outputs and 128
  input features. The private backend composes larger public calls and every M
  row from those tiles; the first K tile replaces the output and later K tiles
  add their partial sums without adding a non-ATen beta parameter to the public
  API. ATen-recognized group sizes 64, 128, and 256 currently return
  `NOT_SUPPORTED` rather than being misclassified as malformed calls.

`libraries/udondnn/run_device.py` also emits the
`unity-udonaten-int4-mm-bundle` reusable Unity artifact. Its manifest retains
the semantic input, packed-weight, `[K/group_size,N,2]` scale/zero, and output
shapes while the fixture packs two 8x128 Device pages into one compact
Q4/BF16 atlas for an 8x256 public operation. A fragment compose pass expands
the selected atlas row and copies the runtime activation slice and prior GPU
output into each 8,192-word active state; one Host `_run` executes both pages
without a CPU readback or a prefilled partial-sum fixture.
The package importer creates `UdonAtenWeightInt4packMm.prefab`; the Editor
lifecycle test executes kernel 208 through SDK IUdonVM and validates non-zero
zero points before deleting all generated assets.

Pointwise operations, Embedding, and RMSNorm currently adapt already validated
UdonDNN backend kernels. Those functions are backend details, not the public
ownership model.
Tile and INT4 matrix multiplication have UdonATen-owned Device kernels.
`libraries/udondnn/run_device.py` packages Tile as the reusable
`unity-udonaten-tile-bundle`; the Unity importer creates
`UdonAtenTile.prefab`, and its SDK IUdonVM lifecycle fixture validates all 72
outputs for trailing-dimension repeats before removing generated assets.
The fixed two-page artifact validates the runtime texture-copy and accumulation
mechanism. `udon-onnx-page --all` now materializes and dispatches arbitrary M/K
tile grids for one call, including edge tiles and M-tile result scatter. Its
counter-based Host remains bounded by atlas count rather than page count. Binding
all call artifacts and the full mixed-library schedule into one model bundle is
also available through `--all-calls`. A single combined Udon Host and Unity
binding still remain before the Llama 8B operator plan can run end to end.

See [operator ownership](../../docs/architecture/operator-ownership.md) for
the source audit and compatibility rules.
