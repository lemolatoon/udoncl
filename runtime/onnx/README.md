# Udon ONNX runtime

`udon_onnx.py` is the dependency-free compiler half of the public Udon ONNX
Runtime. It provides a protobuf reader, typed shape validator, safe ONNX
external-data metadata handling, an extensible `ExecutionProviderRegistry`,
and a serializable Device execution plan. It skips unknown protobuf fields,
rejects malformed tensors and unsafe external paths, and records the UdonDNN
or UdonBLAS call selected for every pass.

The first supported end-to-end graph is the trained MNIST network:

```text
Conv -> Add -> Relu -> AveragePool -> Flatten -> Gemm -> Add
```

Its execution provider maps those nodes to UdonDNN convolution, OpTensor,
activation and pooling plus UdonBLAS SGEMM. The current large-tensor provider
fuses the graph into three existing fragment passes: 9,216 independently owned
convolution outputs, 144 pooling outputs, and 10 logits. Model initializers are
read back from `TensorProto.raw_data` to construct the Device state; the native
CPU oracle still uses the original C header, making weight conversion part of
the differential test.

The built-in providers currently are:

- `UdonMnistExecutionProvider` for the standard-ONNX MNIST fixture.
- `UdonLlamaExecutionProvider` for a fused Llama decoder custom operator. Its
  tiny float32 profile is numerically executed on Device; its Llama 3.1 8B Q4
  profile validates 517 external tensors and emits a 162-pass schedule. All
  225 quantized weight projections are assigned to
  `udonatenWeightInt4packMm` with separate packed weights and bfloat16
  scale/zero metadata; attention activation products remain on
  `udonblasSgemm`.

The general provider also lowers Add, Mul, Sub, Relu, Sigmoid, Tanh, and
Softmax to their cuDNN-compatible UdonDNN entry points. The checked-in
reinforcement-learning actor fixture uses these operators and Gemm entirely
through `UdonOperatorExecutionProvider`, without a model-specific provider.
The Tiny Llama fixture also emits a standard-operator, single-token decoder
through this provider. For sequence length one, causal attention has one
possible value and therefore reduces exactly to the value projection; the
graph still exercises embedding Gather, RMSNormalization, projected residuals,
SiLU expressed as Sigmoid+Mul, and the final vocabulary projection.
Rank-2 through rank-4 ONNX MatMul, including broadcast batch dimensions used
by attention, is expanded AOT into one or more tiled `udonblasSgemm` calls per
output batch matrix.
The general provider also accepts the explicit
`com.lemolatoon.udoncl::WeightInt4packMM` library extension. Llama 3.1 8B's
embedding-independent single-token decoder expands to 546 operator nodes and
225 tiled `udonatenWeightInt4packMm` calls instead of one fused `LlamaDecoder`
node.
Each call records its 64x128 tile grid and K-tile accumulation contract. Each
call also records the exact packed-weight row stride, scale/zero group
stride, external byte ranges, and 8,192-word kernel-208 active-state layout.
`udon-onnx-page` validates those ranges and materializes any requested 64x128
tile without loading a complete tensor or model. It can also place a prior
output tile at word 7,168 for the kernel's private accumulation argument. The
Unity artifact proves two-page GPU texture composition and accumulation in one
Host `_run`. General Host emission still rejects these plans until it can emit
that dispatcher for arbitrary M/K tile grids and bind all external pages,
rather than generating an executable that cannot supply them.
Identity, Reshape, Squeeze, and Unsqueeze are compiled as typed zero-copy views.
Their embedded int32/int64 shape inputs are evaluated at AOT time and retained
with the correct C ABI type rather than being treated as float buffers.
ONNX opset-23 RMSNormalization lowers to the ATen-shaped `udonatenRmsNorm`
operator, not to cuDNN. Rank-2, axis-zero ONNX Gather used for token lookup
lowers to `udonatenEmbedding`, following ATen's
`embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)` argument
order. Their current Device backend reuses the earlier UdonDNN prototype
kernels internally and accepts regions of at most 512 float elements; those
kernels are not part of the public cuDNN-compatibility surface.

Consumers can construct `ExecutionProviderRegistry`, register one graph
predicate and compiler callback per provider, and pass it to `compile_model`.
Standard graphs made only from the currently supported operators can also use
`UdonOperatorExecutionProvider`; it preserves topological node order and emits
an explicit `aot.calls` list of public UdonBLAS/UdonDNN entry points. The
`--emit-host-c` option turns that list into C Host source with descriptor
lifecycle management and error cleanup, ready for the normal UdonCL Host
pipeline. SGEMM dimensions are tiled at AOT time over the current 16x16x16
Device profile without changing the cuBLAS-shaped public call.
`--emit-host-uasm` emits the executable Udon Host side of the same plan: it
exports the ordered pass Materials and ping-pong state textures, binds each
pass input, and dispatches it through `VRCGraphics.Blit`. This removes the need
for a hand-maintained demo Host when packaging a compiled model. The execution
plan, generated C Host, and Udon Host share the same physical SGEMM tile
schedule, including K-tile accumulation. ONNX Gemm forms that are not yet
implemented fail during compilation instead of being accepted with different
numerical semantics. Gemm's optional broadcast C input and its `alpha` and
`beta` coefficients are lowered to tiled `udonblasSgemm` followed by a
broadcasting `udondnnOpTensor` bias pass.
Standard operator plans preserve validated ONNX external initializers as
explicit location, offset, length, type, and shape metadata instead of loading
large weights into the compiler process.

The Unity package imports every artifact exposing the generic `onnx_model` and
`onnx_execution_plan` resources to stable `Models/model.onnx.bytes` and
`Models/execution-plan.json.bytes` paths, independent of the demo stage.

The compiler can be installed independently for tooling and CI:

```text
python -m pip install runtime/onnx
udon-onnx model.onnx -o execution-plan.json
udon-onnx model.onnx -o execution-plan.json --emit-host-c generated-host.c
udon-onnx model.onnx -o execution-plan.json --emit-host-uasm generated-host.uasm
udon-onnx model.onnx -o execution-plan.json --execution-provider operators
```

`--execution-provider operators` deliberately bypasses graph-specific fusion.
It is used to migrate complete models, including the MNIST fixture, onto the
same `UdonOperatorExecutionProvider` path used by arbitrary standard graphs.

It has no runtime Python dependencies. Applications can instead import
`parse_model`, `validate_external_data`, `ExecutionProviderRegistry`, and
`compile_model` from the installed `udon_onnx` module.

Checkpoint conversion is an optional build-time component:

```text
python -m pip install 'runtime/onnx[weights]'
udon-onnx-pack /path/to/safetensors weights-index.json -o weights.q4
udon-onnx-page execution-plan.json . --call 0 --m-tile 0 --k-tile 0 \
  -o page-0-0.bytes
```

The packer accepts indexed or single-file safetensors in float32, float16, or
bfloat16, validates every shape/range, converts matrix tensors to separate
signed-INT4 packed weights and `[K/group_size,N,2]` bfloat16 scale/zero tensors,
preserves normalization tensors as bfloat16, emits
per-tensor SHA-256 evidence, and replaces the destination only after the full
conversion succeeds.

The page tool is dependency-free. It gathers signed-INT4 row slices and
bfloat16 `[K/group_size,N,2]` scale/zero slices directly from validated ONNX
external ranges, converts only the selected metadata to float32, and emits the
same active-state layout consumed by the UnityPackage kernel. An optional full
float32 activation vector fills the selected K slice at word 6,144; an optional
full output accumulator fills the selected N slice at word 7,168.

This is not full ONNX operator coverage yet. The MNIST provider still contains
model-level fusion, while the Llama provider deliberately isolates its custom
fusion behind the registry. General emitted tiled dispatch and external-weight
page binding remain the next runtime layer.
