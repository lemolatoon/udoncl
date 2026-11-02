# Tiny LLaMA

This native-first compiler fixture implements one complete LLaMA-style block:
RMSNorm, rotary position embeddings, causal two-head attention, residual
connections, SwiGLU, a final RMSNorm, and an untied language-model head.

The intentionally tiny shape (`vocab=16`, `dim=8`, `hidden=16`, `context=4`)
keeps future fragment passes inspectable. It is an inference/compiler fixture,
not a useful pretrained language model. Fixed-seed weights make the C reference,
the independent Python oracle, and later Device passes deterministic.

Run the original C implementation and its native UdonHL-adapter round trip:

```text
python examples/tiny-llama/run.py --build-dir build/udoncl
```

Run the same block as seven ordered Device passes (embedding, QKV with RoPE,
causal attention, attention projection, SwiGLU, FFN down projection, and the
language-model head):

```text
python examples/tiny-llama/run_device.py --build-dir build/udoncl
```

The Device runner first generates a real ONNX model, compiles its
`com.lemolatoon.udoncl::LlamaDecoder` node through the public
`UdonLlamaExecutionProvider`, and derives all seven kernel ids and names from
that execution plan. The checked CPU/Device logits therefore validate the
same provider schedule used by the 8B external-weight profile rather than a
parallel hand-maintained list.

## Standard ONNX Unity artifact

Build the same single-token block from only standard ONNX operators and
materialize an importable general-runtime artifact:

```text
python examples/tiny-llama/build_onnx.py --output build/tiny-llama-onnx
```

Import `build/tiny-llama-onnx/unity-artifact` with **Tools → udoncl → Import
Artifacts...**, then import the **ONNX Tiny Llama** Package Sample. This path
uses the stable typed int32 upload, 18 shader-backed Device passes, and
asynchronous float32 logits readback. It covers Gather, RMSNormalization,
Gemm, Add, Sigmoid, Mul, and Reshape without using the disabled general ONNX
Pooling path.

Each active output is owned by a separate fragment. The native fragment test
compares all 16 logits for four prompts and eight greedy generation steps
against the C model.

The generated Unity prefab also contains a world-space chat panel. A player
types a prompt locally and then interacts with the panel. That client takes
ownership, maps the prompt into the fixture's 16-token vocabulary, runs the
seven Device passes, and obtains the Device argmax through
`VRCAsyncGPUReadback`. Only the completed prompt/response turn and four-token
context are manually synchronized. Other clients update their history in
`_onDeserialization`; they do not repeat the inference.

This UI proves the ownership, asynchronous continuation, and shared-history
shape needed by a larger model. Its replies remain deliberately toy-sized and
should not be presented as a useful pretrained conversational model.
