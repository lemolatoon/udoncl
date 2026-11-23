# TinyLlama 1.1B Device Prefill Plan

> Status: active. This is the standalone implementation plan for making the
> pinned real TinyLlama 1.1B Q4 model execute through the generic UdonCL ONNX
> Device path. `TINYLLAMA_AOT_CONVERSATION_PLAN.md` remains the later
> tokenizer, decode, and two-turn acceptance plan.

## 1. Completion boundary

The first gate is complete only when the pinned four-token, empty-KV prefill:

1. imports through the locked ONNX-MLIR frontend and C++ UdonCL passes;
2. emits a fully Device-resident artifact with no Host execution fallback;
3. initializes all immutable weights without copying them into the Udon raw
   heap or mutable OpenCL buffer table;
4. binds all 47 inputs and 45 outputs through the model-generic generated
   binding;
5. executes on the UdonCL Device path using the existing serial in-order
   waits; and
6. matches the pinned ONNX Runtime logits and all 44 present-K/V outputs.

The synthetic hidden-size-8 Transformer, the old seven-pass Tiny LLaMA-style
fixture, artifact generation without inference, and CPU-only ORT execution do
not satisfy this gate.

## 2. Pinned fixture

- Model: `onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX`
- Revision: `cb463df0db21d713728366575b5a71f80de7adb2`
- File: `onnx/model_q4.onnx`
- SHA-256:
  `399360b3b9174b505eedfc71a305f48b98207c51fff4c4643c722b7e37f137eb`
- Specialization: batch 1, sequence 4, past sequence 0
- Public ABI: 47 inputs and 45 outputs
- Oracle tokens: `[1, 450, 4996, 17354]`
- Oracle argmax: `[29958, 390, 17354, 1701]`

The model is cached by content hash. Cache reuse may skip download and
compilation, but it must not skip the numerical inference comparison.

## 3. Autonomous grilling closure

The user requested that every unresolved choice use the recommended answer.
The current design frontier is therefore closed as follows.

1. **Keep synchronization correctness before event optimization.** The first
   real-model path keeps the current serial waits between Device dispatches.
   Event elision/coalescing is a general OpenCL optimization and is not a
   prerequisite for numerical prefill correctness.
2. **Do not specialize compiler decisions for TinyLlama.** Compiler and runtime
   behavior is selected from operator semantics, dtype, shape, resource role,
   and proven lifetime. The pinned model appears only in the example and tests.
3. **Use the generated model ABI, not a handwritten 47/45 switch.** The
   compiler-generated metadata and binding own tensor names, dtypes, shapes,
   byte lengths, and indices. The existing three-input/three-output `host.c`
   remains only the synthetic fixture and is not extended into a real-model
   compatibility table.
4. **Make immutable static handles table-free.** A generated static tensor has
   AOT-known resource id, byte length, context, and lifetime. Represent its
   opaque `cl_mem` with a tagged immutable handle derived from its resource id
   rather than allocating one runtime `_cl_mem` slot and one Host global per
   tensor. Retain/release are validated no-ops for this artifact-owned lifetime.
   Physical-token resolution preserves the existing negative static-resource
   encoding. This is an implementation of opaque `cl_mem`, not a public
   UdonBLAS/UdonDNN API change.
5. **Keep mutable buffers as ordinary `cl_mem`.** Inputs, outputs,
   intermediates, scratch, and K/V capacity allocations keep normal reference,
   context, bounds, command-retention, and release semantics.
6. **Size mutable capacity for the compiled artifact.** The real prefill needs
   more than the current 32 runtime entries. Past and present K/V use the same
   caller-owned capacity allocations only when the schedule proves that final
   output copies occur after all reads. Internal liveness reuse remains
   compiler-controlled. Capacity is an artifact/build property with a checked
   upper bound, not a model-name branch.
7. **Preserve semantic K/V separation.** Past and present remain distinct ONNX
   values and bindings. Sharing their backing allocation is allowed only after
   lifetime and same-dispatch read/write-alias validation succeeds.
8. **Use the low-level generated U# binding for the prefill gate.** Tokenizer,
   chat template, greedy decode loop, and conversation C Host are later gates.
   The prefill test supplies already-tokenized tensors and reads logits/K/V
   through generated public APIs; it never accesses IUdonVM heap slots.
9. **Validate progressively but without weakening final acceptance.** First
   prove artifact structure and initialize/dispose, then logits, then all K/V.
   The final gate still compares every required output against ORT.
10. **Keep the real-model gate out of the mandatory precheck until it is
    deterministic and cache-bounded.** Focused development tests may run it
    explicitly. Once green, precheck must execute inference on every run while
    reusing only content-addressed model/compiler artifacts.

## 4. Current measured blockers

The current repository already imports and canonicalizes the real model. A
fresh configure of `examples/onnx-transformer-tiny` with the cached model
completes successfully. The remaining blockers are in artifact/runtime
orchestration:

- the example C Host is fixed to three inputs, three outputs, and six buffers;
- the AOT mutable `_cl_mem` table has capacity 32;
- the AOT immutable `_cl_mem` table has capacity 256, while the current graph
  has about 1,033 static tensors;
- generated Host finalization currently creates an i32 global and calls
  `udonclAdoptStaticBuffer` for every root static tensor;
- the generated real-model public outputs cannot all be bound through the toy
  Unity buffer declaration; and
- no real-model Device numerical runner or generated-U# Unity acceptance test
  exists.

Increasing the static table to 1,033 is rejected: it consumes raw heap and
still leaves one persistent Host global per weight. Increasing only the event
capacity is unrelated to these blockers.

## 5. Implementation phases

### Phase A: table-free immutable `cl_mem`

1. Add one authoritative tagged-handle codec in the AOT runtime.
2. Make static adoption return the deterministic opaque handle without
   consuming `udoncl_static_buffers`.
3. Resolve context, immutable identity, physical token, and byte bounds from
   generated static metadata or immediate AOT facts.
4. Update retain/release, alias checks, queue validation, and Device argument
   recovery to accept tagged immutable handles.
5. Stop emitting per-static-tensor writable Host globals where the handle is
   AOT-derivable.
6. Prove MNIST static weights still pass generated-U# numerical acceptance.

### Phase B: artifact-sized mutable buffer capacity

1. Compute the required maximum live mutable handles from generated public I/O,
   internal liveness slots, scratch, and library temporaries.
2. Pass a checked capacity into the artifact runtime build. The default remains
   small for ordinary programs; the real model does not impose its capacity on
   every package consumer.
3. Reject an artifact at compile time if the configured capacity cannot hold
   the proven maximum.
4. Keep raw heap and Udon data-slot reports within their backend limits.

### Phase C: model-generic real prefill binding

1. Separate the synthetic fixture target from a real-model target.
2. Generate public input/output buffers and binding methods from model metadata.
3. Allocate capacity-sized K/V storage, bind zero-length past views for the
   first prefill, and bind distinct logical present outputs.
4. Bind input IDs, attention mask, and position IDs with their declared INT64
   dtypes; do not reuse the synthetic INT32 ABI.
5. Remove the real target's dependency on the three-input/three-output
   `host.c` and fixed `unity-library.json` buffer list. (Done: the Unity
   library config file has since been removed altogether.)

### Phase D: numerical execution

1. Add a focused non-Unity Device runner for the complete graph.
2. Compare 128,000 logits and 44 K/V tensors with the cached ORT oracle.
3. Record maximum absolute/relative error, argmax, dispatch count, memory, and
   wall time.
4. Add a Unity generated-U# test using only public binding APIs.
5. Require callback completion, all outputs, zero console errors, and complete
   temporary asset cleanup.

### Phase E: integration

1. Add the cached real-model inference gate to `scripts/precommit.py` only
   after focused Native/Device/Unity runs are stable.
2. Run the complete precheck before every commit.
3. Continue with incremental decode, K/V reuse, tokenizer, conversation C Host,
   and two-turn generated-U# acceptance in
   `TINYLLAMA_AOT_CONVERSATION_PLAN.md`.

## 6. Required regression evidence

- Tagged static handles: identity, byte bounds, context validation, immutable
  write rejection, retain/release, alias checks, and more than 256 resources.
- Capacity planning: exact proven requirement, insufficient-capacity compile
  failure, no silent runtime exhaustion, and ordinary examples retaining their
  small default.
- MNIST: existing Native and generated-U# numerical tests remain green.
- Real TinyLlama: full Device graph, no Host tensor fallback, no weight bytes in
  raw heap, 47/45 generated ABI, and every output compared with ORT.
- Unity: generated U# only, callback completion, no direct heap access, no
  timeout, zero console errors, and cleanup verified.

## 7. Commit gates

Before every commit, stage only the intended changes and run the single
authoritative command:

```text
python scripts/precommit.py
```

Commit only when it prints `precommit=passed`. Unity MCP validation remains
delegated to the required Luna/xhigh agent through that precheck workflow.
