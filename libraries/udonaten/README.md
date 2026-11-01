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
  scales_and_zeros)`. Its Device kernel applies signed INT4 dequantization as
  `value * scale + zero`; one fragment independently owns each output feature.
  Each active Device tile supports group size 32, up to 64 outputs and 128
  input features. The private backend composes larger public calls from those
  tiles; the first K tile replaces the output and later K tiles add their
  partial sums without adding a non-ATen beta parameter to the public API.

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

The implementation currently adapts the already validated UdonDNN prototype
kernels. Those functions are backend details, not the public ownership model.
The fixed two-page artifact validates the runtime texture-copy and accumulation
mechanism. `udon-onnx-page --all` now materializes and dispatches arbitrary M/K
tile grids for one call, including edge tiles and M-tile result scatter. Its
counter-based Host remains bounded by atlas count rather than page count. Binding
all call artifacts and the full mixed-library schedule into one model bundle is
also available through `--all-calls`. A single combined Udon Host and Unity
binding still remain before the Llama 8B operator plan can run end to end.

See [operator ownership](../../docs/architecture/operator-ownership.md) for
the source audit and compatibility rules.
