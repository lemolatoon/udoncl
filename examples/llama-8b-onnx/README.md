# Llama 3.1 8B ONNX profile

This example emits a real ONNX `ModelProto` for the public Udon ONNX Runtime.
The graph contains a `com.lemolatoon.udoncl::LlamaDecoder` execution-provider
node and 517 ONNX external-data tensors describing the canonical Llama 3.1 8B
shape: 32 decoder layers, hidden size 4096, intermediate size 14336, 32 query
heads, eight KV heads, and vocabulary size 128256.

The user-facing build command emits both the fused descriptor and an
embedding-independent, single-token decoder graph through the general
`UdonOperatorExecutionProvider`:

```powershell
python examples/llama-8b-onnx/build.py `
  --output build/llama-8b-onnx
```

That graph has no fused `LlamaDecoder` node. It contains 546 explicit operator
nodes, including all 225 Q4 projections as `WeightInt4packMM`
library-extension nodes,
standard RMSNormalization/Add/Sigmoid/Mul/Reshape/Tile nodes, and records the
64x128 `udonatenWeightInt4packMm` tile grid for every projection. Its input is the
post-embedding `[1,4096]` hidden state. It is a resource-complete AOT plan.
The generated `materialization-plan.json` records 225 paged Q4 calls, 322
direct Device passes, five reusable activation textures, and the exact artifact
capacity without reading licensed weights. The general Host emitter schedules
the page runners and direct passes together; the legacy fused descriptor
remains available for execution-provider compatibility.

The runtime validates 8,030,261,248 logical parameters and plans 162 Device
passes. Matrix weights use signed Q4 blocks of 32 values, while scale and zero
metadata is stored separately as bfloat16 `[K/32,N,2]`, matching ATen's
`_weight_int4pack_mm` contract. Normalization weights remain bfloat16. The
resulting external payload is 5,019,279,360 bytes before 64 KiB alignment gaps.

Every quantized projection in the plan selects `udonatenWeightInt4packMm`. The
plan names its packed weight and scale/zero tensors and records 64-output by
128-input tiles, including the number of M and K tiles and the beta=0/beta=1
accumulation contract. Attention
score/value products remain activation-to-activation `udonblasSgemm` calls.
This distinction lets a Unity runtime stream one compact external-weight page
at a time without pretending Q4 tensors are ordinary float32 SGEMM inputs.

The repository does not redistribute Meta weights. `weights.q4` must be
created from a checkpoint the user is licensed to use and must match the
generated index:

```text
python -m pip install 'runtime/onnx[weights]'
udon-onnx-pack /path/to/Meta-Llama-3.1-8B-Instruct \
  build/llama-8b-onnx/weights-index.json \
  -o build/llama-8b-onnx/weights.q4
python examples/llama-8b-onnx/build.py \
  --output build/llama-8b-onnx \
  --external-data-directory build/llama-8b-onnx \
  --materialize
```

The last command validates all 517 ONNX external tensors before reading them,
checks available disk space, creates `unity-artifact`, and may take a long time:
the compact Q4 atlases alone contain 4,690,411,520 bytes. To reuse an existing
shader template, pass `--unity-template-directory
build/udondnn-device-run/unity-udonaten-int4-mm-bundle`.

Install the UdonCL VPM package in a Worlds project, then choose **Tools →
udoncl → Import Artifacts...** and select
`build/llama-8b-onnx/unity-artifact`. The generated runtime prefab exposes the
general typed Write/Run/async Read API. This operator graph accepts a
post-embedding `hidden_state` float array and returns `logits`; tokenization and
embedding lookup remain application responsibilities.

Supplying the descriptor alone is compilation and resource
planning, not pretrained-model inference. The numerically executable
`examples/tiny-llama` fixture uses the same public Llama execution provider,
the same seven-pass layer schedule, and the real Device backend at a small
shape suitable for CI and Unity MCP verification.
