# TinyLlama AOT Conversation Plan

## 1. Purpose and completion boundary

This file is the source of truth for running the pinned real TinyLlama 1.1B Q4
model through the generic UdonCL ONNX path. The implementation is complete only
when a generated U# binding can accept text in Unity, preserve and reuse KV
cache across two conversation turns, and produce numerically verified output.
Generating an artifact, completing one prefill, or running the synthetic
hidden-size-8 Transformer does not by itself complete this plan.

The implementation must remain model-independent below the example C Host.
Model names, 22-layer structure, hidden sizes, and TinyLlama graph patterns must
not appear in compiler or runtime dispatch decisions. Generic canonicalization
and fusion based on operator semantics, dtype, rank, shape, strides, and proven
aliasing are allowed.

## 2. Pinned inputs and current facts

### 2.1 Model and oracle

- Repository: `onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX`
- Revision: `cb463df0db21d713728366575b5a71f80de7adb2`
- File: `onnx/model_q4.onnx`
- Size: `909750422` bytes
- SHA-256:
  `399360b3b9174b505eedfc71a305f48b98207c51fff4c4643c722b7e37f137eb`
- The model contains embedded block-32 Q4 weights, FP32 activations, 155
  `com.microsoft::MatMulNBits` operations, dynamic sequence length, and 22
  layers of past/present K/V.
- ONNX Runtime CPU is the numerical oracle. The existing four-token/empty-KV
  oracle has 128,000 logits, 44 present K/V outputs, and argmax
  `[29958, 390, 17354, 1701]`.
- The model and generated artifacts use content-addressed caches. Cache hits may
  avoid download, packing, and compilation, but must not skip inference or
  oracle comparison.

### 2.2 Frontend checkpoint and actual blockers

- ONNX is imported into ONNX MLIR and then handled exclusively by C++ passes.
  Python may download/cache inputs, invoke tools, and run ORT oracles; it must
  not parse the graph or emit Host C/MLIR after import.
- The canonical real-model MLIR has already been produced without textualizing
  the large initializers and retains all 155 `MatMulNBits` operations.
- Generic Gather already supports INT32/INT64 indices, arbitrary rank/axis,
  scalar storage types, and large source tensors by transferring selected
  slices. The old statement that INT64/large Gather is the next blocker is
  stale.
- The real model now completes ONNX Dialect -> portable Host MLIR -> finalized
  Host MLIR and can be emitted as a Unity bundle. Host scalar allocation uses
  442 of 512 data slots after buffer handles and single-use constants are
  colored/spilled instead of receiving permanent slots.
- The emitted bundle contains 1,033 immutable static tensors totaling about
  909 MB. It is not yet an inference success: the graph fails the fully
  Device-resident gate and falls back to Host tensor storage, duplicating about
  939 MB into `raw-heap.bytes` and producing zero whole-graph dispatch sites.
- The compiler records operator counts and unsupported Device operator/provider
  pairs. A Unity config can set `require_device_graph` so this fallback fails
  immediately instead of producing a misleading large artifact. Persistent
  buffer dispatch is implemented for `Cos`, `Sin`, `Neg`, bounded `Mul`, and
  bounded `Sigmoid`/`Tanh`, in addition to the earlier convolution, pooling,
  activation, rank-two matrix product, Add, and metadata aliases. On the pinned
  sequence-4 specialization, expanding the proven pointwise view to 32,768
  words removes every `Cos`/`Sin`/`Neg`, `Add`, `Mul`, `Sigmoid`, and `Tanh`
  node from the unsupported set. After the Device Q4, RmsNorm, and Softmax
  checkpoints below, the remaining measured set is Cast 2, Concat 45, Equal 1,
  Expand 46, Gather 1, MatMul 45, ScatterND 1, Slice 111, Transpose 111, and
  Where 1.
- The generic correctness-oriented `MatMulNBits` path still uses small
  Host-managed tiles. A first Device-resident specialization now covers the
  pinned model's three-input unsigned Q4 form (`bits=4`, `block_size=32`,
  implicit zero point 8, FP32 activation/scales/output, `M<=4`, `K<=5632`,
  `N<=32000`). It reads immutable packed weights and scales directly, covers
  all output columns per dispatch, tiles K by 128, and ping-pongs a
  131,072-word Device surface without Host partial readback. A standalone
  static-initializer graph
  verifies generated allocation, kernel-235 dispatch, final public-output copy,
  and disposal. This is not yet wired through all remaining TinyLlama
  operators and does not replace the generic 2-8-bit Host path. Re-finalizing
  the real sequence-4 graph confirms that all 155 `MatMulNBits` nodes,
  including the 32,000-column `lm_head`, now satisfy this Device contract;
  `MatMulNBits` is absent from the measured unsupported set.
- The existing UdonATen two-pass arena RMSNorm backend is now used directly by
  the generated Device graph. Static epsilon and normalized extent are folded
  by the C++ finalization pass, the immutable weight stays Device resident, and
  reduce/finalize share the generic operator scratch allocation with Q4. The
  pinned sequence-4 graph accepts all 45 RmsNorm nodes without Host tensor
  readback.
- Static FP32 Softmax tensors up to 16,384 words and 512 values per reduction
  group now use the existing UdonDNN arena backend. Any static ONNX axis is
  mapped generically to `outer x axis x inner`; no model shape is embedded in
  the compiler. The standalone Device numerical suite passes, and all 22
  Softmax nodes disappear from the pinned graph's measured unsupported set.
- Production-sized diagnostic bundles may omit intermediate textual MLIR with
  `--omit-intermediate-ir`; assembly, manifest, binding, static tensors, and the
  storage report remain. This avoids publishing several model-sized debug
  snapshots but does not remove the still-pending compiler clone/probe cost.
- The AOT OpenCL runtime now implements `clEnqueueCopyBuffer` through the typed
  `vrc.buffer_copy` intrinsic. This is required when an output view cannot alias
  the public output buffer and fixes the previously missing Host link symbol.
- Device intermediate storage now uses compile-time liveness and capacity
  matching. A slot is never reused by the dispatch that still reads it, graph
  outputs remain live through the final public-buffer copy, and a six-stage
  unary chain reduces five intermediate allocations to two. The AOT intrinsic
  lowering also emits the actual configured 2D launch shapes: large
  convolution/pooling use 8192x2 and large activation/op-tensor use 8192x4,
  instead of the obsolete 10800x1 MNIST launch.
- Gather-profile recovery now accepts a separately proven word capacity for
  each read-only input. The recovered memref, texture width, HLSL coordinate
  calculation, and manifest `required_bytes` retain that capacity instead of
  incorrectly inheriting the output dispatch extent. An i32 scalar may also
  carry a configured inclusive min/max contract. LLVM index-range recovery
  uses that contract to prove `base + gid`, rejects an overflowing view at
  compile time, and preserves the same minimum/maximum in the kernel manifest
  for generated dispatch validation. The focused regression proves a
  16-element output view at bases 8..16 into a 64-word input and rejects bases
  up to 60. Read-only inputs may now also be `const uint*`: the fragment path
  reconstructs a little-endian uint32 directly from RGBA8 bytes, preserves
  `uint32` in the manifest, and supports shader bit extraction without first
  interpreting packed Q4 data as float. The first Q4 dispatch builds on this:
  kernel 235 binds the complete immutable packed/scales resources, applies
  generated scalar range contracts, validates dynamic resource lengths in the
  OpenCL facade, and tiles N/K while keeping partials on Device. Dynamic
  sequence-length specialization, generic packed/unpacked zero points,
  sub-word/unaligned views, and integration with the full real graph remain.

### 2.3 Initial target

- Unity 2022.3, VRChat Worlds, PC, D3D11 only.
- Batch size 1.
- Build-time configurable maximum context length. The first example uses 128
  tokens and generates at most 32 new tokens per call. The design must permit a
  later build with the model maximum of 2048.
- The final acceptance fixture performs two conversation turns. It must append
  new K/V rather than recomputing the first turn.
- Quest/Android, WebGL, and other graphics APIs are outside this plan.

## 3. Immutable AOT static tensors

### 3.1 Common facility

Static weights are not an ONNX-only hidden mechanism. UdonCL artifacts gain a
generic immutable Device tensor facility, with ONNX initializer externalization
as its first consumer. The same manifest/resource mechanism may later be used
by ordinary C/OpenCL Hosts.

After ONNX import, a C++ MLIR pass replaces initializer payloads with immutable
static tensor symbols containing:

- stable symbol/resource identity;
- content hash and byte length;
- dtype, logical shape, layout, and required alignment;
- artifact resource index.

Weight bytes must not be duplicated into generated C arrays, Udon raw heap, or
session initialization uploads.

### 3.2 Physical Unity representation

- The artifact keeps one logical immutable tensor and one lossless Unity
  texture per ONNX initializer. Unity textures are already validated up to
  8192x8192 RGBA32; the current `gather-f32` compiler proof, not the physical
  texture, exposes only 16,384 four-byte words (64 KiB) through one view.
  Initializers larger than this retain one physical texture and gain
  deterministic bounded tile views with byte base/length metadata. Packed Q4
  weight, scale, and zero point remain distinct logical resources.
- The compiler/CMake artifact contains exact static tensor bytes, texture
  layout metadata, manifest entries, and hashes. The Unity importer creates the
  texture assets and serialized references but does not parse ONNX or repeat
  compiler decisions.
- Mipmaps, color-space conversion, and Unity texture compression are disabled.
  Imported bytes are hash-checked.
- Static resources use a generated immutable resource table separate from the
  32-entry mutable OpenCL buffer table. A dispatch binds only the few textures
  needed by that operator.
- Static and mutable storage share one typed Device view ABI. A view records
  storage kind/index, byte offset/length, dtype, shape, and strides. Read-only
  static views reject writes.
- Session initialization resolves and validates references, hashes, bounds,
  alignment, dtype, and layout, but performs no Host-to-Device weight upload.
- Cache keys include model content, packer ABI, texture format/layout,
  alignment, compiler version, and graphics backend.

### 3.3 MNIST gate

Before the 900 MiB model is attempted, MNIST must use the same static tensor
path. Its weight and bias must be texture assets, absent from raw heap and large
C arrays, require no initialization upload, and produce the existing oracle
result through the generated U# binding. Hash failure, view validation,
reimport, console checks, and cleanup are part of this gate.

## 4. Device-resident execution and memory planning

- Activations, intermediates, K/V, logits, and workspaces remain Device
  resident between operators. Host readback is limited to public outputs,
  test/debug observations, and the scalar/token result needed by generation.
- UdonBLAS, UdonDNN, UdonATen, and UdonONNXOperators consume the same Device
  view ABI and share one context and in-order queue.
- C++ MLIR bufferization performs liveness analysis and assigns intermediates
  to a bounded Device workspace. Non-overlapping tensors reuse storage. Public
  I/O and K/V are persistent.
- ONNX past and present K/V remain distinct at the semantic level. When alias
  analysis proves it safe, bufferization aliases them to one capacity-sized
  allocation and appends only the new slice. No TinyLlama-specific KV kernel is
  permitted.
- `Reshape`, `Flatten`, `Unsqueeze`, `Squeeze`, and compatible `Slice` or
  `Transpose` operations become metadata-only views. A generic materialization
  is inserted only when a consumer cannot support the proven strides/layout.
- Static shape expressions are folded before kernel selection. Truly dynamic
  shapes remain Host scalar IR bounded by the configured capacity.
- All actual GPU waits use the existing may-suspend/MLIR Async/Udon continuation
  path. Cooperative checkpoints stay disabled by default; busy waits and long
  synchronous Host loops are forbidden.

## 5. Operators and optimization

### 5.1 Ownership

- GEMM and batched GEMM lower to UdonBLAS.
- Convolution, pooling, activation, and softmax lower to UdonDNN where their
  pinned compatible APIs own the operation.
- Tensor, reduction, and indexing operations lower to UdonATen when ATen owns
  them.
- ONNX-specific semantics and ONNX Runtime extensions, including
  `MatMulNBits`, live in UdonONNXOperators.
- Unsupported domain/opset/operator/dtype/rank/shape combinations fail during
  compilation with node location and a concrete reason. There is no Python or
  execution-plan fallback.

### 5.2 Device-resident `MatMulNBits`

- Preserve the generic semantics already tested for bits 2 through 8,
  power-of-two block sizes of at least 16, packed/unpacked zero points,
  optional `g_idx` and bias, and logical FP32/FP16/BF16. The current
  Device-resident checkpoint intentionally accepts only the pinned unsigned-Q4
  subset described in section 2.2 and rejects other forms from the Device
  graph; broadening it must not silently change those generic semantics.
- Read packed weight, scale, and zero point from immutable Device resources.
  Do not expand the complete model to FP32. Read-only input capacities, raw
  uint32 reconstruction, correlated index-range proof, and scalar min/max
  contracts now permit a bounded whole-resource Q4 kernel without weakening
  access proof.
- Keep input, partial sums, and output on Device; dequantize in the shader and
  accumulate in FP32.
- Build an AOT tile schedule from generic tensor layout, dtype, block size,
  bits, and proven resource capacity. The initial Q4 specialization binds the
  immutable packed-weight and scale resources plus activation and ping-pong
  output surfaces. Its fragment lowering proves all linear accesses against
  independently configured input capacities; the OpenCL facade additionally
  validates the concrete resource byte lengths. K and N tile selection is
  generated at compile time and no weight bytes are repacked by the runtime
  Host. Extend the same contract with zero-point and other generic views rather
  than introducing a model profile.
- Resource limits therefore require multiple Device-only dispatches per
  operator. K partial sums remain in a persistent Device workspace and never
  round-trip through the Host. N tiles write disjoint output ranges. The final
  32,000-logit projection is tiled by the same generic rule.

### 5.3 Fusion policy

The first correct path uses generic unfused operators except for the required
Device-resident `MatMulNBits` replacement and metadata-only view elimination.
Dispatch count and time are then measured. Attention or other fusions may be
added only as model-independent MLIR optimization passes. Native tests retain
an unfused reference path and compare fused, unfused, and ORT results.

## 6. Tokenizer and conversation Host

### 6.1 Tokenizer component

The pinned tokenizer is not another ONNX model. It is a Llama tokenizer
described by `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, and
`special_tokens_map.json`: vocabulary 32,000, 61,249 BPE merges, byte fallback,
and a chat template.

Create an independent `UdonTokenizers` C library/package. It contains a generic
byte-fallback BPE engine and consumes model-specific generated tables. It is
not part of UdonONNXOperators or the generic ONNX runtime.

At build time, tokenizer JSON/config is compiled to compact arrays containing:

- normal vocabulary pieces already decoded to UTF-16;
- byte-fallback value to token-ID mappings;
- integer merge-pair keys and ranks;
- special token IDs and chat-template data.

Runtime JSON parsing, `DataDictionary`, and string-concatenation BPE are not
used. Input is read directly from an immutable Udon string reference. Normal
pieces stay UTF-16. Encode creates at most four logical UTF-8 fallback bytes
for an unmatched Unicode scalar. Decode appends normal UTF-16 pieces directly
and uses a small UTF-8 state machine, preserved across tokens, only for byte
fallback pieces. This preserves the semantics verified in ShaderGPT while
removing its runtime JSON and whole-string UTF-8 conversion costs.

Tokenizer tests compare token IDs and decoded strings with Hugging Face for
ASCII, Japanese/BMP input, surrogate pairs/emoji, combining characters,
whitespace/newlines, fallback input, special tokens, chat templates, and
multi-byte sequences split across tokens.

### 6.2 String ABI

`char *` retains ordinary C byte-pointer semantics and is never silently
reinterpreted as `System.String`. UdonCL Host Runtime adds an explicit target
extension `udon_string_t`, represented by an immutable Udon string reference
and by an equivalent Native runtime object. Generated U# bindings expose it as
`string`. Its reference is kept live in the coroutine frame across suspension.
Tokenizer accessors expose UTF-16 length/code units/scalars without copying the
whole string. Native and Udon targets must produce identical results.

### 6.3 C Host and generated binding

The example C Host, not handwritten U#, owns initialization, chat-template
application, tokenization, prefill, decode loop, KV state, greedy token
selection, and result decoding. It exports operations equivalent to:

- initialize and validate resources;
- reset conversation;
- append user text;
- generate up to the requested token count;
- read the response.

The exact C signatures follow normal `UDONCL_EXPORT` rules and use
`udon_string_t` for U# strings. May-suspend exports become the normal generated
`Start*`/callback binding surface. The Unity test calls that generated binding;
it does not implement a second decode loop or KV manager in U#.

The first gate uses deterministic greedy selection. ArgMax runs on Device and
only the selected token is returned during normal conversation. Test/debug
mode can read all logits. Streaming callbacks, cancellation, top-k/top-p, and
temperature sampling are later work.

For the second turn, incremental chat-template tokens are appended only when
their sequence equals a full official-tokenizer re-encode of the conversation.
Otherwise the Host explicitly resets and performs a full prefill rather than
silently reusing invalid K/V.

## 7. Tests, performance, and caching

### 7.1 Numerical acceptance

For prefill and every one of 32 decode steps, Native, Device, and Unity paths
compare all 32,000 logits and all 22 layers of K/V with ORT. Tests also require
finite values, exact greedy token IDs, argmax/top-k agreement, and recorded
absolute/relative error. Tolerances are fixed from the first independently
correct end-to-end run and are not loosened per backend for convenience.

The Unity fixture performs two turns through generated U# bindings and checks
that K/V capacity and length advance without recomputing the previous turn.
Console errors, callback status, temporary asset cleanup, and lifecycle cleanup
are mandatory.

### 7.2 Performance observations

Initial correctness is not gated on a guessed tokens/second value. Every run
records initialization, tokenization, prefill, per-token decode time, dispatch
count, Device/workspace/KV memory, artifact size, and maximum continuous Udon
execution time. GPU waits must yield frames from the beginning. A measured
baseline establishes a later timeout that distinguishes hangs from slow runs;
until then tests emit phase heartbeat/progress and use a deliberately generous
upper bound.

### 7.3 Required execution

The final `scripts/precommit.py` runs the real-model tests every time. The model,
static textures, canonical MLIR, and compiled artifact may be reused from a
content-addressed cache, but ORT and UdonCL inference comparisons are not
skipped. GitHub Actions runs cached ORT/compiler/Native/Device gates. Unity is
run by the local precheck until a non-self-hosted licensed Unity CI path is
available.

Generated model/texture/build artifacts are not committed to Git or bundled in
the normal VPM package. The library package distributes compiler/runtime/
operator/tokenizer support; the example CMake build obtains the pinned model
and generates its project-local artifact.

## 8. Implementation and commit gates

Implement in this order, keeping each gate independently testable:

1. Add the generic MLIR/static-resource representation and artifact schema.
2. Move MNIST weight/bias to immutable Device textures and pass Native/Device/
   generated-U# Unity numerical tests with no weight upload.
3. Externalize the pinned TinyLlama initializers and import its static resource
   table without placing bytes in Host globals.
4. Implement static shape/view canonicalization and Device workspace planning.
5. Replace `MatMulNBits` with the Device-resident implementation.
6. Implement and test every remaining generic operator required by the
   canonical model, including batched MatMul and generic ScatterND/cache update.
7. Pass full-model Native prefill, KV, and iterative decode against ORT.
8. Pass the same full-model Device path and record the first performance and
   resource baseline.
9. Add `udon_string_t`, UdonTokenizers, generated tokenizer tables, and official
   tokenizer conformance tests.
10. Add the example C conversation Host and its generated binding.
11. Pass the Unity two-turn, 32-token, full-logit/full-KV gate and integrate it
    into the mandatory precheck.
12. After all consumers pass, remove the old ONNX execution-plan JSON,
    Material/arena scheduler, `static_inputs`, old runtime, and toy TinyLlama
    sample. Do not remove the only working comparison path earlier.

## 9. Completion checklist

The work is complete only when all of the following are true:

- generic C++ ONNX-to-MLIR lowering handles the pinned graph without a
  model-profile fallback;
- MNIST and TinyLlama use the same immutable AOT Device tensor facility;
- Q4 weights, activations, intermediates, and K/V remain Device resident;
- K/V is reused across decode steps and two conversation turns;
- the independent C tokenizer/chat-template path matches Hugging Face;
- the C conversation Host is exposed through compiler-generated U# bindings;
- Unity accepts text and returns a two-turn response;
- every prefill/decode logit and K/V comparison passes against ORT;
- the real-model gate runs on every final precheck with cache reuse only for
  inputs and compilation;
- Unity reports zero console errors and cleans temporary assets;
- stale execution-plan and toy-runtime consumers are removed after migration.
