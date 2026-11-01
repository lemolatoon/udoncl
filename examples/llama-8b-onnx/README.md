# Llama 3.1 8B ONNX profile

This example emits a real ONNX `ModelProto` for the public Udon ONNX Runtime.
The graph contains a `com.lemolatoon.udoncl::LlamaDecoder` execution-provider
node and 291 ONNX external-data tensors describing the canonical Llama 3.1 8B
shape: 32 decoder layers, hidden size 4096, intermediate size 14336, 32 query
heads, eight KV heads, and vocabulary size 128256.

The generator can additionally emit an embedding-independent, single-token
decoder graph through the general `UdonOperatorExecutionProvider`:

```text
python examples/llama-8b-onnx/generate_onnx.py \
  -o build/llama-8b-onnx/model.onnx \
  --operator-output build/llama-8b-onnx/operators.onnx \
  --operator-plan build/llama-8b-onnx/operators-plan.json
```

That graph has no fused `LlamaDecoder` node. It contains 546 explicit operator
nodes, including all 225 Q4 projections as `Q4Gemm` library-extension nodes,
standard RMSNormalization/Add/Sigmoid/Mul/Reshape/Tile nodes, and records the
64x128 `udonblasSgemvQ4` tile grid for every projection. Its input is the
post-embedding `[1,4096]` hidden state. It is a resource-complete AOT plan, but
the C/Udon Host emitter intentionally rejects it until external-weight paging
can feed those tile grids; the legacy fused plan remains available meanwhile.

```text
python examples/llama-8b-onnx/generate_onnx.py \
  -o build/llama-8b-onnx/model.onnx \
  --index build/llama-8b-onnx/weights-index.json \
  --plan build/llama-8b-onnx/execution-plan.json
```

The runtime validates 8,030,261,248 logical parameters and plans 162 Device
passes. Matrix weights use signed Q4 blocks of 32 values with one fp16 scale;
normalization weights remain bfloat16. The resulting external payload is
4,517,404,672 bytes before 64 KiB alignment gaps.

Every quantized projection in the plan selects `udonblasSgemvQ4`. The plan
names its weight tensor and records 64-output by 128-input tiles, including the
number of M and K tiles and the beta=0/beta=1 accumulation contract. Attention
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
python runtime/onnx/udon_onnx.py build/llama-8b-onnx/model.onnx \
  --external-data-dir build/llama-8b-onnx \
  -o build/llama-8b-onnx/validated-plan.json
```

Supplying the descriptor alone is compilation and resource
planning, not pretrained-model inference. The numerically executable
`examples/tiny-llama` fixture uses the same public Llama execution provider,
the same seven-pass layer schedule, and the real Device backend at a small
shape suitable for CI and Unity MCP verification.
