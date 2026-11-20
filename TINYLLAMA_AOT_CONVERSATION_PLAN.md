# TinyLlama AOT Conversation Plan

## 1. Purpose and completion boundary

This file is the source of truth for running the pinned real TinyLlama 1.1B Q4
model through the generic UdonCL ONNX path. The current completion boundary is
one externally-tokenized prompt: a generated U# binding accepts token IDs,
runs the same statically specialized `M=1` artifact once per prompt token,
reuses bounded Device-resident K/V, then greedily decodes until EOS or the
requested limit and returns generated token IDs. Generating an artifact,
completing one fixed batch prefill, or running the synthetic hidden-size-8
Transformer does not by itself complete this plan.

The first compiler/runtime acceptance milestone remains the externally
tokenized token-array path. On top of it the separate Chat Artifact adds the
text layer: chat-template and special-token handling for the prompt, and
token-to-text decoding that streams the reply while it is generated
(sections 6.6, 6.9 and 6.10). `UdonTokenizers` remains an independent
package: the generic ONNX compiler/runtime must not depend on it, and the
low-level Conversation Artifact must continue to accept and return token IDs
without requiring text services. The Chat Artifact continues one
conversation over several turns within the Context Capacity (section 6.11);
a conversation longer than the capacity (ring buffer, M8) and transactional
rollback remain later work.

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
- The real model completes ONNX Dialect -> portable Host MLIR -> finalized Host
  MLIR. An earlier allocator path reached a historical 752-slot measurement
  against capacity 503, dominated by lowering-created i32 static-resource
  indices. That number is not the current production requirement and must not
  be reused as an acceptance baseline: later finalization stopped earlier on a
  generated-Host-literal ownership check, before exact capacity measurement.
- Commit `86437ad` fixes that issue generically. Backend-generated literals,
  including immutable static-resource indices, have a dedicated ownership and
  materialization path and are excluded from ordinary source-constant pooling
  and spill selection. The general regression and mandatory precommit pass;
  no model name, tensor shape, layer count, or TinyLlama-specific allocation
  rule was added. The earlier finite-buffer-view provenance scaling issue
  remains separately handled by exact copy-on-write state and explicit layout
  metadata; literal ownership is not coupled to resource provenance.
- Fresh real-model artifact generation and numerical U# execution have not yet
  accepted this change. The next gate is to generate a fresh focused Prefill
  Validation Artifact, then compare all 128,000 logits and all 44 present K/V
  outputs with the ORT oracle through the generated U# binding.
- The compiler records operator counts and unsupported Device operator/provider
  pairs. A Unity config can set `require_device_graph` so this fallback fails
  immediately instead of producing a misleading large artifact. Persistent
  buffer dispatch is implemented for `Cos`, `Sin`, `Neg`, bounded `Mul`, and
  bounded `Sigmoid`/`Tanh`, in addition to the earlier convolution, pooling,
  activation, rank-two matrix product, Add, and metadata aliases. On the pinned
  sequence-4 specialization, expanding the proven pointwise view to 32,768
  words removes every `Cos`/`Sin`/`Neg`, `Add`, `Mul`, `Sigmoid`, and `Tanh`
  node from the unsupported set. After the Device Q4, RmsNorm, and Softmax
  checkpoints below and the generic Device reindex/Gather checkpoints
  described below, plus generic rank-zero-through-four FP32 Equal/Where
  broadcast kernels with word-addressable FP32 0/1 Device masks and a generic
  rank-one-through-four ScatterND kernel, the remaining measured unsupported
  Device operator set is empty. An earlier artifact and the real generic
  47-input/45-output public binding passed structural validation, but that does
  not accept the current compiler. The current next gate is fresh artifact
  generation followed by numerical prefill through the ordinary C Host exports
  and generated U# binding. An earlier structural real-model artifact
  provisioned the mutable Device pool for 35 measured allocation sites, and
  public Q4 logits used a logical view over a padded 131,072-word physical
  surface. These counts remain diagnostic history until the current compiler
  emits a fresh accepted artifact.
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
- UdonBLAS now has a Device-resident strided-batched SGEMM arena backend for
  contiguous static FP32 tensors up to 16,384 words and K up to 256. Zero input
  stride implements whole-matrix broadcast; equal batches use ordinary matrix
  strides, while output remains contiguous as required by ONNX bufferization
  and the fragment proof. Native numerical tests cover the pinned attention
  shape and broadcast. All 45 MatMul nodes in the pinned sequence-4 graph now
  satisfy this generic UdonBLAS contract.
- UdonONNXOperators now materializes static contiguous FP32 Transpose tensors
  of rank one through five and at most 16,384 elements with a generic Device
  reindex kernel. Shape and permutation are packed as validated dispatch
  scalars; the compiler contains no attention- or model-specific case. Native
  numerical tests cover the pinned attention layouts, the OpenCL AOT facade
  validates shape/permutation/buffer bounds, fragment recovery succeeds, and
  all 111 Transpose nodes disappear from the pinned graph's measured
  unsupported set.
- The same generic UdonONNXOperators reindex facility now materializes static
  contiguous FP32 Slice tensors of rank one through five and at most 16,384
  elements. The C++ pass normalizes ONNX starts/ends/axes/steps, proves the
  inferred output shape and every input coordinate, and packs nonnegative
  starts and positive steps into the dispatch ABI. Other valid Slice forms,
  including negative steps, retain public Host semantics but fail the fully
  Device-resident gate. Native positive/strided numerical tests, OpenCL facade
  bounds checks, fragment recovery, and the pinned graph verify the path; all
  111 pinned Slice nodes disappear from the measured unsupported set.
- Static broadcast-compatible FP32 Expand tensors of rank zero through five
  and at most 16,384 elements now use a generic Device reindex kernel. Input
  dimensions are right-aligned with output dimensions and only extent-one
  axes broadcast; the OpenCL facade revalidates ranks, extents, logical buffer
  sizes, and the inferred output product. Native rank-five repeated-head and
  leading-axis broadcast tests, AOT runtime execution, C++ lowering, fragment
  recovery, and the real sequence-4 graph pass; all 46 pinned Expand nodes
  disappear from the measured unsupported set.
- Static contiguous two-input FP32 Concat tensors of rank one through five and
  at most 16,384 output elements now use one generic Device dispatch. The C++
  pass folds the axis into inner-word and left/right extent scalars; the kernel
  selects either input without reading or preserving the render target. Native
  last-axis and KV-axis numerical tests, AOT facade bounds checks, C++
  lowering, fragment recovery, and the real graph pass; all 45 pinned Concat
  nodes disappear from the measured unsupported set. Other valid variadic
  Concat forms retain the public Host implementation until a fragment-safe
  generic schedule is added.
- Immutable FP32 axis-zero Gather now uses a generic Device kernel rather than
  the old 512-element embedding staging path. INT32 and INT64 indices are read
  as one or two little-endian uint32 words, so fragment HLSL does not require
  64-bit integer arithmetic; negative ONNX indices are normalized and every
  texture address is made safe before sampling. Native INT32/INT64 numerical
  tests, the AOT facade, C++ lowering, fragment recovery, and the real
  sequence-4 graph pass. The single pinned embedding Gather is absent from the
  measured unsupported set. Other axes and mutable/non-FP32 data retain the
  public Host implementation.
- INT64-to-FP32 Cast now uses a generic Device conversion kernel. It reads two
  little-endian uint32 words, handles positive and negative two's-complement
  values, and performs the final conversion in FP32 without requiring shader
  int64 support. Native numerical tests, the AOT facade, C++ lowering,
  fragment recovery, and the real graph pass cover the path; both pinned Cast
  nodes are absent from the measured unsupported set.
- Production-sized diagnostic bundles may omit intermediate textual MLIR with
  `--omit-intermediate-ir`; assembly, manifest, binding, static tensors, and the
  storage report remain. This avoids publishing several model-sized debug
  snapshots but does not remove the still-pending compiler clone/probe cost.
- The AOT OpenCL runtime now implements `clEnqueueCopyBuffer` through the typed
  `vrc.buffer_copy` intrinsic. This is required when an output view cannot alias
  the public output buffer and fixes the previously missing Host link symbol.
- Device allocations can now expose zero-length typed views, and ONNX Runtime
  accepts static zero tensor dimensions. This represents the real model's
  initial `[1, 4, 0, 64]` past K/V inputs without inventing a non-empty tensor,
  while the owning allocation may retain capacity for later decode steps.
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
- The bounded Transformer fixture now executes through the compiler-generated
  U# binding in Unity Play Mode rather than only through direct `IUdonVM`
  calls. Its two-turn acceptance test compares both turns' logits and present
  K/V values. The C Host's `CommitKv` export uses the model-independent
  generated binding ABI to enqueue whole-buffer output-to-input Device copies;
  the second turn writes only its token input and does not re-upload K/V from
  Host memory. The test still reads present K/V solely as an acceptance oracle.
  Dynamic append/slice handling for the real model remains required.

### 2.3 Acceptance status

Items 1-4 below passed on 2026-11-20 through the generated U# bindings in
Unity (commits `3fad51f..49cd65c`, on top of `6d1485c`):

- Prefill Validation Artifact: all 128,000 logits and all 44 present K/V
  tensors match the ORT sidecar.
- `M=1`, capacity-64 Conversation Artifact: the externally tokenized prompt
  `[1, 450, 4996, 17354]` greedily generates `[1701, 29916, 432, 17204]`
  twice with `Reset` in between, then `Shutdown` succeeds. The `M=1` step
  gate (`49cd65c`) compares steps 3 and 4 against ORT: all logits and all
  present K/V match with maximum absolute error about `5e-5`.
- Measured cost: 56,339 frames for two generations of 4 prompt + 4 generated
  steps, about 4,000 frames and 4,000 device dispatches per token step.

Real-model validation found three generic bug classes, each fixed below the
example Host with a smaller regression:

- unwritten graph-constant Device buffers: graph-local ONNX constants are now
  externalized as immutable static tensors (`073d5fe`), and a Device graph
  operand that nothing writes is rejected at compile time (`2f3715e`);
- async resume re-entering the dispatcher's static return slot: waiters are
  resumed through a non-nesting trampoline (`78f9474`), and fixed entry
  arguments stay out of async persistent slots (`a59f378`);
- consecutive non-blocking Host transfers overwriting the single Udon command
  slot: an outstanding transfer is drained before the next is issued
  (`d1ca036`), and an unexecuted command is never overwritten (`86efdba`).

Other fixes in the same range: static ONNX folding and canonicalization for
the `M=1` graph (`bedcc35`, `541319e`, `cab5d46`, `5a6c85d`, `fa3e709`,
`1be8479`, `402736a`), strided-view provenance (`d21631b`, `6940e1b`,
`6a88197`, `407b270`), buffer resolution at each use and transfer binding
after export pruning (`6b21d1b`, `a21aedc`), and the Unity fixtures
(`80b5392`, `432d165`, `54bf4d0`, `e279ae9`, `49cd65c`).

Item 5 is implemented and awaits its Unity run (section 6.7): UdonTokenizers
gained added-token splitting, a streaming encoder, compiled chat templates and
an incremental decoder (`5ffcc06`); the example gained the optional text layer
and a Context Capacity build parameter (`d7d4768`) and a text-in/text-out
Unity fixture for the separate Chat Artifact (`1b7323b`). The original list
follows; each failure must still
be fixed in a model-independent compiler, runtime, library, or importer
component and covered by a smaller regression before a real-model gate is
retried.

1. Generate a fresh Prefill Validation Artifact from the current compiler and
   record its actual storage report, immutable resources, mutable allocations,
   canonical Unity pool, and absence of readiness fallbacks. Historical
   artifacts and synthetic 10-buffer results are not evidence for this step.
2. Prove that canonical pool materialization handles the real 47-input/
   45-output public binding plus its private buffers. HEAD `0d6c512` currently
   proves config/manifest replacement only on the bounded synthetic Transformer.
3. Import that fresh artifact and run the focused generated-U# lifecycle. Bulk
   upload the three fixed prefill inputs, execute through callbacks, bulk-read
   all outputs, and compare all 128,000 logits and 44 present K/V tensors with
   the ORT sidecar. Structural validation alone does not complete this step.
4. Generate the static `M=1`, capacity-64 Conversation Artifact and first prove
   one decode token against ORT. Then prove the complete externally-tokenized
   prompt/decode loop, including Device-resident K/V advancement, exact greedy
   token sequence, EOS/limit behavior, callbacks, and cleanup.
5. Only after the token-array conversation passes, add the independent
   token-to-text, chat-template, and special-token consumer layer. None of this
   text work may enter generic ONNX or OpenCL lowering.

### 2.4 Initial target

- Unity 2022.3, VRChat Worlds, PC, D3D11 only.
- Batch size 1.
- Query length `M=1` for both prefill and decode. Prompt length is arbitrary up
  to a build-time configurable Context Capacity; the Host loops over the
  externally supplied token IDs.
- The first conversation artifact specializes the ONNX graph to
  `past_sequence_length = 63` and `attention_mask = [1, 64]`. Before the
  logical sequence fills the capacity, unused past rows are zero-filled on the
  left and masked with zero.
- Every semantic past/present pair has distinct physical storage. After each
  successful run that has a following step, a compiler-generated unrolled
  helper materializes `present[:, :, 1:64, :]` into the paired past buffer with
  the generic Device Slice kernel. In row-major `[1, H, C, D]` layout this
  axis-two suffix is one segment per head, so it is not a single-offset view or
  a contiguous buffer copy. The separate Slice dispatch reads present and
  writes past without same-dispatch resource aliasing.
- The first example uses capacity 64 and generates at most 32 new tokens.
  Before every run the Host requires `prompt_count + generated_count <= 64`;
  reaching capacity before EOS fails explicitly instead of silently applying
  a sliding-window policy.
- The final acceptance fixture performs one complete externally-tokenized
  prompt and greedy decode through the generated U# API. It must stop at EOS
  or the requested limit and must not recompute prior prompt positions.
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
- ONNX past and present K/V remain distinct semantically and physically for
  this milestone. The generic sequence-state analysis identifies their
  relationship, and bufferization generates a generic Slice materialization
  after a successful run. No TinyLlama-specific KV kernel is permitted.
- `Reshape`, `Flatten`, `Unsqueeze`, `Squeeze`, and compatible `Slice` or
  `Transpose` operations become metadata-only views. A generic materialization
  is inserted when a consumer cannot support the proven strides/layout. In
  particular, `present[:, :, 1:capacity, :]` is non-contiguous across heads in
  row-major `[1, H, C, D]` and is materialized into contiguous past storage.
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
- One Q4 dispatch accumulates the whole K of every output word for up to 256
  blocks of 32 (K <= 8192) with constant, guarded Device loops; a longer K is
  split at compile time into straight-line dispatches whose partial sums
  ping-pong through a persistent Device workspace and never round-trip through
  the Host. The result is bit-identical to the former 128-wide K tiles, which
  cost ceil(K / 128) dispatches (3,096 of about 4,019 per token step); every
  TinyLlama projection, including the 32,000-logit one, is now one dispatch.

### 5.3 Fusion policy

The first correct path uses generic unfused operators except for the required
Device-resident `MatMulNBits` replacement and metadata-only view elimination.
Dispatch count and time are then measured. Attention or other fusions may be
added only as model-independent MLIR optimization passes. Native tests retain
an unfused reference path and compare fused, unfused, and ORT results.

## 6. External token boundary and generation Host

### 6.1 Tokenizer component

The pinned tokenizer is not another ONNX model. It is a Llama tokenizer
described by `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, and
`special_tokens_map.json`: vocabulary 32,000, 61,249 BPE merges, byte fallback,
and a chat template.

`UdonTokenizers` is an independent C library/package. It contains the
allocation-free generic byte-fallback BPE encoder, the incremental decoder,
added/special-token handling and the build-time table and chat-template
compiler, and it consumes model-specific immutable tables. It is not part of
UdonONNXOperators or the generic ONNX runtime.

At build time, tokenizer JSON/config is compiled to compact arrays containing:

- normal vocabulary pieces already decoded to UTF-16;
- byte-fallback value to token-ID mappings;
- integer merge-pair keys and ranks, plus an open-addressing hash index of
  the pairs (at most half full, so a lookup is about one probe);
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

The first executable acceptance fixture uses Japanese BMP input, a surrogate
pair/emoji, UTF-8 byte fallback, and ranked BPE merges. It is intentionally a
small generic-model fixture, not a TinyLlama-specific tokenizer shortcut.

### 6.2 String ABI

`char *` retains ordinary C byte-pointer semantics and is never silently
reinterpreted as `System.String`. UdonCL Host Runtime adds an explicit target
extension `udon_string_t`, represented by an immutable Udon string reference
and by an equivalent Native runtime object. Generated U# bindings expose it as
`string`. Its reference is kept live in the coroutine frame across suspension.
Tokenizer accessors expose UTF-16 length/code units/scalars without copying the
whole string. Native and Udon targets must produce identical results.
`udon_int32_array_t` similarly denotes the caller-owned Udon `int[]` object;
Host C reads and writes it only through typed accessors. Reserved object
references are lowered by a dedicated Host-object-intrinsics pass and never
enter the raw C byte-memory model.

### 6.3 C Host and generated binding

The example C Host, not handwritten U#, owns initialization, bounded prompt
token storage, the `M=1` prefill/decode loop, K/V state, greedy token selection,
EOS handling, and generated-token storage. Tokenization and text decoding are
caller concerns for the token-array Conversation Artifact; the Chat Artifact
links the optional text layer of section 6.6 into the same Host. The Unity
test calls only the generated binding; it does not implement another decode
loop or K/V manager in U#.

The first gate uses deterministic greedy selection. ArgMax runs on Device and
only the selected token is returned during normal conversation. Test/debug
mode can read all logits. Reply text is streamed by polling synchronous
exports (section 6.10); C-to-U# streaming callbacks, cancellation,
top-k/top-p, and temperature sampling are later work.

### 6.4 Agreed generation lifecycle and C ABI

One generated Behaviour owns one generation instance: one ONNX session, one
bounded prompt workspace, one generated-token workspace, and one K/V state.
The API has no instance handle; multiple concurrent generations require
multiple Behaviour instances. The example C Host exports the following
ordinary C surface:

```c
UDONCL_EXPORT udononnxStatus_t tinyllama_initialize(void);
UDONCL_EXPORT udononnxStatus_t tinyllama_reset(void);
UDONCL_EXPORT udononnxStatus_t tinyllama_set_prompt_tokens(
    udon_int32_array_t token_ids, uint32_t token_count);
UDONCL_EXPORT udononnxStatus_t
tinyllama_generate(uint32_t max_new_tokens); /* may suspend */
UDONCL_EXPORT uint32_t tinyllama_generated_token_count(void);
UDONCL_EXPORT udononnxStatus_t tinyllama_read_generated_tokens(
    udon_int32_array_t destination);
UDONCL_EXPORT udononnxStatus_t tinyllama_dispose(void);
```

The compiler adapts `tinyllama_generate` to the normal generated
`StartGenerate(maxNewTokens, receiver, completionEvent)` surface. The other
exports become synchronous generated `int[]`/scalar methods. C exports never
contain U# receiver or event parameters.

`SetPromptTokens` synchronously copies the caller-owned `int[]` into the
build-time-bounded Host workspace. `Generate` runs one model invocation per
prompt token, then feeds each greedy result into the next invocation until EOS
or `max_new_tokens`. Reset clears prompt/generated tokens, K/V logical length,
position, masks, and the distinct past buffers while retaining initialized
immutable model resources. Present outputs are overwritten by the next run.
Dispose releases the complete instance.

Extend `udononnxStatus_t` with `INVALID_STATE` and `CAPACITY_EXCEEDED`.
`INVALID_VALUE` is reserved for malformed arguments, `INVALID_STATE` for a
valid operation in the wrong lifecycle state, `BUSY` for an operation in
flight, `CAPACITY_EXCEEDED` for a request beyond the artifact's build-time
Context Capacity, and `DISPOSED` for use after disposal. Capacity failure never
truncates history or silently selects a sliding-window policy.

Initialize is valid only for an empty instance. Reset, prompt replacement, and
dispose are rejected with `BUSY` while generation is active. Reset is otherwise
reusable and keeps immutable model resources initialized; dispose is terminal.
Generate requires a nonempty prompt and a positive token limit. Reading tokens
requires successful completion.

### 6.5 Tokenizer C contract

This section documents the independent `UdonTokenizers` package only. It is
not part of the token-array Conversation Artifact or its completion gate.

`UdonTokenizers` owns its independent status enum because encoding and table
validation are not ONNX session states. Its generated Tokenizer Model is
immutable and shareable. Encoding uses a caller-owned encoder state and a
caller-owned Tokenizer Workspace of six `int32` words per pre-merge symbol
(`UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS`); the token IDs end up in the first
words of that workspace. Decoding uses a caller-owned output array and
explicit incremental UTF-8 fallback state. The library performs no dynamic
allocation and has no mutable global session. Conversation Host maps
tokenizer failures to its public ONNX status at the example boundary while
retaining the tokenizer status for diagnostics.

The build compiles the pinned tokenizer files into project-local generated
tables keyed by their content hashes and compiler ABI. The VPM package ships
the generic tokenizer compiler/runtime, not the pinned tables or downloaded
model assets.

The table compiler also derives bounded token, turn-reference, and UTF-16
response workspace capacities from Context Capacity, the configured maximum
new tokens, and the concrete Tokenizer Model. Exhaustion is explicit and never
falls back to dynamic allocation or repeated string concatenation.

The generated low-level U# surface of the small tokenizer fixtures is
`TryTokenize(string text, int[] tokenIds, out int result)`. `tokenIds` is the
caller-owned result storage; `result` is the number of valid token IDs, while
negative values in the fixture facade encode tokenizer status. The encoder
workspace is C storage of the fixture Host. The conversation Host maps the
independent tokenizer status enum to its public status contract instead of
exposing that fixture convention. One-shot `udontokenizersEncode` (Udon
`int[]` output) and `udontokenizersEncodeRaw` (C array output) share the
streaming encoder and copy the finished tokens out; a C array is not cast to
`udon_int32_array_t`.

### 6.6 Text layer as implemented

These decisions fill the parts of 6.1 and 6.5 that the plan left open.

- Encoding matches Hugging Face `Tokenizer.encode(text)` of the pinned
  `tokenizer.json`: the post-processor BOS, then added tokens (`<unk>`, `<s>`,
  `</s>`) matched on the raw text leftmost-longest, also inside user text.
  The text between them forms independent segments, each normalized with its
  own `Prepend("▁")`, so `</s>\n<|assistant|>` encodes as `</s>`, `▁`,
  `<0x0A>`, ... exactly as Hugging Face does. Added tokens that are normalized,
  stripped or single-word are rejected by the table compiler.
- The chat template is compiled at build time, not interpreted. The table
  compiler renders the Jinja template with the Hugging Face settings
  (`trim_blocks`, `lstrip_blocks`, sandboxed) using sentinel contents and
  accepts it only if every probe conversation of one to three messages over
  the rendered roles equals the concatenation of per-role
  `prefix + content + suffix` plus a generation-prompt suffix. It also rejects
  literals that an added token could straddle. For TinyLlama this yields
  `<|role|>\n` + content + `</s>\n` and the generation prompt
  `<|assistant|>\n`. The runtime streams the literals and the caller's
  immutable strings into one logical text; BPE segments continue across
  pieces, so no string is concatenated.
- The chat prompt is BOS + rendered template, i.e. `Tokenizer.encode` of
  `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`.
  With no system message the template costs 17 tokens; a user message of
  "What is your name?" gives 22 prompt tokens, "Hello" gives 18.
- Decoding implements the pinned decoder sequence Replace(`▁`→` `),
  ByteFallback, Fuse, Strip(1 leading space) with explicit state: a run of
  byte-fallback pieces that is not valid UTF-8 (including truncated runs and
  overlong or surrogate encodings) becomes one U+FFFD per byte, as in
  Hugging Face, by rewriting the run's own output. Special tokens are skipped
  on request. The leading strip is applied to the first units that become
  final, so decoded text is usable before decoding finishes (section 6.10).
- There is no Udon char-array ABI. Decoded text is returned as UTF-16 code
  units, one per element of a caller-owned `int[]`; a U# caller builds the
  string from it. Workspaces are bounded by
  `max_new_tokens * MAX_PIECE_CODE_UNITS` (16 for TinyLlama), emitted by the
  table compiler.
- The example C Host keeps its token-array exports unchanged and exposes a
  small internal C interface to its prompt and generated-token workspaces.
  `conversation_text.c` adds `SetChatPrompt(systemText, userText)` (an empty
  system text omits the system message), its bounded form `BeginChatPrompt` /
  `ContinueChatPrompt(systemText, userText, maxWork)` / `ChatPromptPending`
  (section 6.9), `PromptTokenCount`, `ReadPromptTokens`, `StreamedTextLength`,
  `ReadStreamedText(destination, offset, count)` (section 6.10),
  `GeneratedTextLength`, `ReadGeneratedText` and `LastTokenizerStatus`, and
  for multi-turn chat `BeginChatTurn` / `ContinueChatTurn(userText, maxWork)`
  and `ResetChat` (section 6.11). A
  prompt beyond the Context Capacity maps to `CAPACITY_EXCEEDED`. The
  text layer is linked only into the separate Chat Artifact
  (`tinyllama_real_chat`, `UdonclTinyLlamaChatBinding`); the Conversation
  Artifact stays free of tokenizer code.
- Context Capacity and the new-token limit are CMake cache parameters of the
  example (`UDONCL_TINYLLAMA_CONTEXT_CAPACITY`, default 64, and
  `UDONCL_TINYLLAMA_MAX_NEW_TOKENS`, default 32). They specialize the graph to
  `past_sequence_length = capacity - 1` and pass the same bounds to the C
  Host; the compiler has no knowledge of the value, and generic kernel limits
  decide whether a capacity compiles. The SGEMM contract (K at most 8192,
  attention-times-V has K = capacity) no longer caps it at 256; other
  capacities are inferred to compile, and only capacity 64 has been compiled. A capacity of 64 holds a short exchange: 22 prompt tokens plus up
  to 32 new tokens.

### 6.7 Text acceptance fixture

`tests/boundary/tinyllama_chat_oracle.py` renders the chat prompt, runs ONNX
Runtime with the same left-padded fixed-capacity `M=1` profile, decodes the
greedy reply with Hugging Face and records the greedy margins. For "What is
your name?" at capacity 64 the oracle is prompt `[1, 529, 29989, 1792, ...]`
(22 tokens), reply tokens `[3421, 1024, 338, 19235, 29889, 2]` and the text
"My name is Sarah." with a minimum greedy margin of 0.66. The Unity fixture
`PublicTinyLlamaChatLifecycleTests.GeneratedUSharpBindingAnswersChatPromptWithText`
passes the two strings through the generated binding and compares the prompt
tokens, the generated tokens and the decoded UTF-16 text with the oracle. Its
controller tokenizes the prompt in bounded slices over several frames and
polls the streamed text once per frame during the generation; the test
requires monotone lengths, a stable polled prefix and a last polled text
equal to the oracle reply. This revision of the fixture has not been run in
Unity yet.

### 6.8 Generated prefill binding and physical storage

The Prefill Validation Artifact and Conversation Artifact are separate. The
former exposes all 47 inputs and 45 outputs as generated Logical Tensor
Bindings so every logit and present K/V value can be compared with ORT. The
latter keeps those tensors private and exports only the conversation API.
Full tensor observation may be enabled by an explicit test/debug build but is
not part of the normal conversation ABI.

After ONNX import, the C++ pipeline is the sole owner of model I/O names,
dtypes, logical shapes, byte lengths, capacity bounds, and indices. It emits
unrolled physical allocation/bind operations and the public binding metadata;
Python and handwritten 47/45 C or JSON tables are forbidden. As with the MNIST
Host, the ordinary C Host selects the OpenCL platform/device and owns the
`cl_context` and in-order `cl_command_queue`. It passes both to a
model-independent generated physical create helper, which borrows them and
uses the Host context for session creation and generated I/O allocation. The
generated helpers own only their generated buffers and model/session state;
they never release the Host queue or context. Initialize, bind, run, and
generated-buffer disposal remain unrolled compiler output, so the public
lifecycle is authored in C while model-specific tensor enumeration remains in
generated MLIR. The Host disposes generated ONNX state before releasing its
queue and context.

Input upload, output readback, and Device-to-Device copies use ordinary OpenCL
operations on `cl_mem`. No ONNX-specific WriteBuffer/ReadBuffer facade is
introduced. Model-independent generated resolve helpers expose each bound
input/output as a borrowed `cl_mem`, byte offset, and logical byte length to
the C Host; the handles remain generated-buffer-owned. `CL_MEM_READ_ONLY` and
`CL_MEM_WRITE_ONLY` retain their OpenCL
kernel-access meaning and do not block Host transfers. Generated U# whole-array
methods may still use the Unity bulk texture upload/readback transport for
public validation buffers, but they address the same bound physical buffers and
do not change the C/OpenCL ownership contract.

Logical tensor shape and physical capacity are distinct metadata. The fixed
four-token Prefill Validation Artifact has empty past K/V inputs with a
zero-length logical view over nonzero storage. The Conversation Artifact
instead has static `[1, H, 63, D]` past tensors whose unused prefix is zero and
masked. Padded logits similarly expose their exact logical length over a larger
Device surface. Public INT64 tensors use generated U# `long[]` whole-buffer methods;
the physical upload packs each element as two little-endian 32-bit words in
one bulk transfer rather than exposing an `int[]` pair ABI.

The compiler computes the maximum simultaneously live mutable handles from
public I/O, internal liveness slots, operator scratch, and distinct past and
present sequence-state allocations plus their Slice dispatch resources.
It emits an artifact-specific runtime capacity and fails compilation if the
proven requirement exceeds the backend limit. Fixed global increases and
model-name capacity branches are forbidden. Physical sharing and slot reuse
remain invisible behind Logical Tensor Bindings and require proven non-overlap,
liveness, and fragment read/write legality.

### 6.9 Encode cost and bounded work

Hugging Face BPE merges the lowest-ranked adjacent pair first, leftmost on a
tie, until no pair has a merge. The first encoder implemented that definition
literally: every merge rescanned all adjacent pairs with a 16-step binary
search each, which is quadratic in the text length. The encoder now uses the
Hugging Face algorithm itself: a binary min-heap of `(rank, position)` keys
over a doubly linked symbol list, with stale heap entries discarded when they
surface. Rank and position are packed into one `int32` key. Added tokens are
stored as negative symbols that never merge, so one pass covers all segments
of a text. Pair lookup goes through the hash index of the merge table.

Udon instructions for one synchronous call, measured with the Udon emulator
on a small artifact that links only the tokenizer and the pinned TinyLlama
tables (the library source and tables that the Chat Artifact links):

| input | tokens | before | after |
|---|---:|---:|---:|
| encode, 18 chars English | 6 | 332,544 | 76,869 |
| encode, 175 chars English | 37 | 26,213,253 | 856,972 |
| encode, 36 UTF-16 units Japanese + emoji | 42 | 139,889 | 80,901 |
| encode, 879 chars English | 181 | 647,395,865 | 4,611,347 |
| chat prompt, 18 chars | 22 | 836,481 | 162,292 |
| chat prompt, 175 chars | 53 | 28,772,700 | 948,774 |
| chat prompt, Japanese + emoji | 57 | 638,968 | 167,859 |
| chat prompt, 879 chars | 197 | 659,117,948 | 4,711,791 |

At the rate assumed in section 7.2 (about 16,000 Udon instructions in 2 ms)
the 879-character encode took about 80 s before, far beyond the 10 s Udon VM
limit, and takes about 0.6 s now. The cost is now close to linear, about
5,000 instructions per input character. The remaining cost is dominated by
raw-heap accesses (four externs each) in the heap sift and in the
character-to-token binary search.

A long message still must not run in one Udon event, so all encode work can
be sliced. Progress lives in the caller-owned encoder state, the caller
passes the same strings on every call, and nothing is allocated:

- `udontokenizersEncoderAppendPartRaw` and
  `udontokenizersEncoderAppendChatMessagePartRaw` consume at most about
  `max_code_units` UTF-16 code units per call (a surrogate pair or an added
  token is never split) and report when the text is consumed.
- `udontokenizersEncoderStepRaw` does at most `max_work` merge work units per
  call: one unit links one symbol, resolves one queued pair, or emits one
  token. `udontokenizersEncoderFinishRaw` runs the remaining steps.
- The Chat Host exposes this as `BeginChatPrompt`, then
  `ContinueChatPrompt(systemText, userText, maxWork)` until
  `ChatPromptPending` returns 0. One work unit is one consumed code unit or
  one merge work unit. The prompt is committed by the call that finishes it;
  a failure abandons it and leaves no prompt.

Measured on the text layer in the emulator, a `ContinueChatPrompt` call with
a budget of 8 costs at most about 35,000 Udon instructions and with a budget
of 32 at most about 79,000 (roughly 2,500 to 4,400 per work unit, falling
with larger budgets because each call also pays a fixed setup). The
175-character chat prompt needs 92 calls at budget 8 and 23 at budget 32;
slicing adds about 12% and 3% to the total. The caller chooses the budget
for its frame time. In the real Chat Artifact, run in the same emulator, the
oracle prompt ("What is your name?", 22 tokens) takes five
`ContinueChatPrompt` calls at budget 32: 175,713 instructions in total and
at most 71,489 in one call.

### 6.10 Streamed reply text

U# polls synchronous exports; C never calls back. `StreamedTextLength`
decodes the tokens generated since the previous call and returns the number
of UTF-16 code units of the reply that are final. `ReadStreamedText` copies
any range of the final units. The contract:

- Final units never change and the length never decreases during one
  generation, so every polled text is a prefix of every later one.
- After the generation completed, the streamed text is exactly the text
  `ReadGeneratedText` returns: Hugging Face `decode(ids,
  skip_special_tokens=True)`.
- U# never sees a partial character. The UTF-8 state machine, U+FFFD
  replacement and the leading-space strip stay in C. An open byte-fallback
  run is withheld until a non-byte token or the end of the generation closes
  it, because Hugging Face decodes a run as a whole and replaces every byte
  of it when any part is invalid UTF-8. A character that is not in the
  vocabulary therefore appears when its last byte token has been followed by
  another token, not byte by byte.
- A poll costs time proportional to the new tokens only: about 4,000 Udon
  instructions for a poll that decodes one token and 1,400 for one that
  finds nothing new, against a whole re-decode per call before (about 32,000
  for a 20-token reply).
- A new generation or a reset starts an empty text; the Host reports this
  through a generation serial.

The library function behind it is `udontokenizersDecoderStepRaw`, the
counterpart of Hugging Face `DecodeStream.step`: it appends one token and
reports the chunk that became final. `DecodeStream` emits a chunk as soon as
the text does not end in U+FFFD and never flushes, so its chunks can differ
from `decode` for invalid or truncated UTF-8. Here the chunks of all steps
plus the units `udontokenizersDecoderFinishRaw` adds always equal `decode`.
Native tests check this after every token against Hugging Face, including
byte fallback split across tokens and by a skipped special token, invalid
runs, a run cut off by the end, and 60 pseudo-random token sequences.

Known limit: the generated binding currently returns `StatusBusy` from every
synchronous method while an asynchronous export such as `Generate` is
pending, so a poll made during a generation is refused before it reaches the
C Host. The C exports themselves are valid during a generation. Until the
binding admits them, the first successful poll is the one made after
completion.

The gate is the binding's, not the program's, as far as the emulator can
tell. Driving the real Chat Artifact with scripted reply tokens and running
`StreamedTextLength` and `ReadStreamedText` directly on the Udon program
between the events of the suspended `Generate` (790 polls, at most 3,408
instructions each) left the generation and its result intact. The polled
text grew with each token read back, 0, 2, 7, 10, 16 and 17 code units for
"My name is Sarah.", and every polled text was a prefix of the next. This
has not been tried in the Udon VM.

### 6.11 Multi-turn chat

The Chat Artifact keeps one conversation in the sequence state across turns.
The conversation state is example Host C (`conversation_host.c`,
`conversation_text.c`); the only library addition is the model-independent
`udontokenizersEncoderAppendChatTurn[Part]Raw`, and no compiler code knows
about turns.

- `BeginChatPrompt`/`ContinueChatPrompt` start a conversation as before.
  After a completed reply, `BeginChatTurn` then `ContinueChatTurn(userText,
  maxWork)` (bounded like `ContinueChatPrompt`, until `ChatPromptPending` is
  0) append the next user message, and `Generate` continues at the position
  where the reply ended. A new prompt after a reply stays `INVALID_STATE`;
  `ResetChat` (async, the Host Reset) discards the conversation and any text
  being built. A turn committed but not generated yet may be replaced by
  another `BeginChatTurn`.
- The reply enters the context as its generated token IDs, never decoded and
  encoded again: Hugging Face would encode the reply text differently (for
  example `My` instead of the generated `▁My` after `<|assistant|>\n`).
- Closing the reply: an EOS selected before the limit was already run by the
  speculative decode step, so nothing is added. A reply that reached the
  token limit never ran its last token; the next turn first runs it and then
  EOS (only the token when it is EOS itself). Each prompt step after the
  first of a conversation advances the pending sequence-state entry first,
  as the generated binding requires.
- The appended text is the rest of the previous role's suffix after its
  leading EOS text, the user message and the generation prompt, encoded
  without BOS as a new segment: `\n<|user|>\n{content}</s>\n<|assistant|>\n`
  for TinyLlama. `tinyllama-conversation-text` splits Hugging Face's
  `apply_chat_template` encoding of whole multi-turn conversations into the
  first prompt, the replies and the appended turns and requires each turn
  exactly.
- `ReadPromptTokens` after `ContinueChatTurn` returns the tokens the next
  generation feeds: the closing tokens and the appended turn. The streamed
  and generated text restarts with every turn; the history display belongs
  to U#.
- Overflow: a turn whose tokens do not fit fails `ContinueChatTurn`, and a
  `Generate` whose prompt plus token budget does not fit fails, both with
  `CAPACITY_EXCEEDED` and the conversation unchanged; the caller resets.
  There is no sliding window, and every run position stays below the
  capacity. Capacity 64 holds a short exchange: the oracle conversation
  below reaches 51 positions before its second reply.
- `tests/boundary/tinyllama_chat_oracle.py --next-user` continues the same
  ORT sequence state the same way and lists later turns under `turns`, with
  the token limit reduced to the remaining capacity. "What is your name?"
  then "Repeat your name." gives "My name is Sarah." and then
  `[29903, 279, 801, 29889, 2]`, "Sarah." (minimum margin 0.30); asked
  alone, "Repeat your name." does not produce the name. The Unity fixture
  `PublicTinyLlamaMultiTurnChatLifecycleTests.GeneratedUSharpBindingContinuesChatAcrossTurns`
  runs in the chat focused gate when the oracle has later turns. It has not
  been run in Unity yet.
- Udon emulator, real Chat Artifact, selected tokens scripted from the
  oracle (so the Host protocol, not numerics): both turns of the oracle above
  return `Generate` 0 (the generated binding rejects a run or advance out of
  protocol), the turn-2 prompt equals the oracle's 23 tokens, its tokens are
  staged at positions 28..50 and its steps run at 28..55, and both texts
  match. With `--max-new-tokens 4` the cut-off first reply is closed by
  `[19235, 2]` and turn 2 continues at position 25. `ContinueChatTurn` at
  budget 32 takes five calls, about 155,000 instructions in total.
  `ResetChat` returns 0, after which `BeginChatTurn` is `INVALID_STATE`.

## 7. Tests, performance, and caching

### 7.1 Numerical acceptance

For every `M=1` prompt and decode step, Native and Device debug paths compare
all 32,000 logits and all 22 layers of K/V with ORT. Unity runs the same public
generated-token sequence through the generated binding and compares exact
greedy token IDs. Tests also require finite values, argmax agreement, and
recorded absolute/relative error. Tolerances are fixed from the first
independently correct end-to-end run and are not loosened per backend for
convenience.

Greedy ArgMax selects the lowest token ID when finite values tie and treats a
non-finite comparison domain as execution failure. Generation stops at the
requested token count or the configured EOS token, whichever comes first. EOS
participates in the token/K/V sequence and terminates generation.

The Unity fixture supplies external prompt token IDs, generates one bounded
completion through the generated U# binding, and checks that K/V position
advances without recomputing prior prompt tokens. Console errors, callback
status, temporary asset cleanup, and lifecycle cleanup are mandatory.

### 7.2 Performance observations

Initial correctness is not gated on a guessed tokens/second value. Every run
records initialization, prefill, per-token decode time, dispatch
count, Device/workspace/KV memory, artifact size, and maximum continuous Udon
execution time. GPU waits must yield frames from the beginning. A measured
baseline establishes a later timeout that distinguishes hangs from slow runs;
until then tests emit phase heartbeat/progress and use a deliberately generous
upper bound.

#### Measurement basis

Performance is judged by wall-clock time per model step and by frames at a
fixed frame rate. A frame count from a `-batchmode` Editor is neither: that
Editor renders on D3D11 with no vsync and no frame-rate limit, so near-empty
frames spin while an asynchronous GPU readback waits for the GPU backlog. One
`M=1` step measured 109 batchmode frames against about 34 structural frames
(continuations plus readback round trips). Batchmode frame counts remain in
the logs as a hang/timeout diagnostic only; they are not compared between
runs, machines, or commits.

Every focused TinyLlama fixture (prefill, `M=1` step, conversation, chat)
therefore measures on the Editor side with `UdonclFixtureTiming`
(`Time.realtimeSinceStartup`, sampled once per polled frame, so every boundary
has one-frame resolution) and writes these Editor-log lines:

```text
udoncl <label> timing[ step=<name>] mode=<unbounded|fixed> target_fps=<n> vsync=<n> previous_target_fps=<n> previous_vsync=<n> batchmode=<true|false> graphics=<api>
udoncl <label> timing[ step=<name>] phase=<name> frames=<n> ms=<t> steps=<n>[ ms_per_step=<t> frames_per_step=<f>]
udoncl <label> readback[ step=<name>] index=<n> name=<tensor> frames=<n> ms=<t>
udoncl <label> timing[ step=<name>] total frames=<n> unity_frames=<n> ms=<t> fps=<f> mode=<m> target_fps=<n> paced_frames=<n> readbacks=<n>[ readback_frames_min=.. readback_frames_max=.. readback_frames_mean=.. readback_ms_min=.. readback_ms_max=.. readback_ms_mean=..]
```

`<label>` is `tinyllama_prefill`, `tinyllama_m1_step`, `tinyllama_conversation`
or `tinyllama_chat`. A phase line is written when the phase ends. `steps` is
the number of `M=1` model steps the phase runs: 1 for the prefill and step
`run` phase, and `prompt + generated - 1` for a `Generate` (the last generated
token is not fed back); phases that run no model step report `steps=0` and no
per-step fields. The pre-existing progress and result lines keep their fields
in order and gain trailing ones (`ms=`, `fps=`, `mode=`, `target_fps=`, and
`run_ms=`/`generate_ms=` style per-phase totals).

Setting `UDONCL_FIXTURE_TARGET_FRAME_RATE=90` in the environment of the
focused run selects the VRChat-like `fixed` mode: the fixture sets
`QualitySettings.vSyncCount = 0` and `Application.targetFrameRate = 90` for
the measured interval and restores both afterwards. Because a batchmode Editor
does not necessarily honor `targetFrameRate`, the fixture also holds every
polled frame until its period has elapsed; `paced_frames` counts the frames
that needed this hold, and `fps` reports the rate actually achieved. Unset,
empty or `0` leaves the Editor unbounded (`mode=unbounded`), which is the
default and the mode of the commit validation gate. The frame count to quote
for a step is `frames_per_step` of a `mode=fixed` run; the time to quote is
`ms_per_step`, with its mode.

In `fixed` mode the conversation and chat fixtures also write, from
`UdonclFixtureTiming.End()`, a `frame_timing` line (`FrameTimingManager`
GPU, main-thread and render-thread milliseconds per polled frame, mean and
maximum, `available=false` when the Editor reports none) and, through the
generated binding's opt-in pump trace (`TraceBinding`, see the device work
scheduling section of
`docs/architecture/refactor-baseline-and-mlir-pipeline.md`), these lines:

```text
udoncl <label> pump trace=<on|unavailable> continuations=<n> in_window=<n> first_of_frame=<n> unity_frames=<n> continuations_per_frame_mean=<f> continuations_per_frame_max=<n> event_us_total=<n> event_us_mean=<f> event_us_max=<n> frame_event_us_mean=<f> frame_event_us_max=<n> windows=<n> stop_limit=<n> stop_budget=<n> stop_estimate=<n> stop_idle=<n> window_us_mean=<f> window_us_max=<n>
udoncl <label> pump_continuations <frame>:<driven>:<us>,...
udoncl <label> pump_windows <frame>:<slices>:<us>:<reason>,...
```

They say per Unity frame how many continuations ran, how long each Udon
event took, and whether the pump stopped at the continuation limit (1), the
time budget (2), the adaptive estimate (4) or because nothing was ready (3,
a readback wait or the end of the export). `driven` is 0 for the first Host
slice charged to its frame's window (`first_of_frame`). The continuation
order matches the Udon emulator's event order for the same artifact, so
per-event instruction and Blit counts can be joined to the measured
microseconds.

`TraceBinding` also logs, in either mode, the binding's run-time frame budget
(`udoncl <label> pump_budget budget_us=<n> continuations=<n>
adaptive=<true|false> override=<true|false>`) after applying the optional
environment overrides `UDONCL_FIXTURE_FRAME_BUDGET_US`,
`UDONCL_FIXTURE_FRAME_CONTINUATIONS` (each 0 to 1,000,000) and
`UDONCL_FIXTURE_FRAME_ADAPTIVE` (0 or 1). Unset, they keep the manifest
defaults, which is what the commit validation gate runs; an invalid value
fails the fixture. They let one artifact be measured at several budgets
without recompiling it.

Readback latency (request to completion callback, in frames and ms) is
reported for the public-buffer readbacks of the prefill and step fixtures,
where the test controller issues each `StartRead*` from the previous
completion callback and the per-frame poll sees both ends. The readbacks the
generated binding issues on behalf of the C Host inside an asynchronous export
(the ArgMax token readback of every `Generate` step) are not observable from a
fixture, and neither are the individual token steps inside one `Generate`. To
expose them without Editor-only reflection the generated binding would need,
per instance: a counter and the `Time.frameCount`/`Time.realtimeSinceStartup`
pair taken at each `VRCAsyncGPUReadback.Request` of an async-command readback
and at its `OnAsyncGpuReadbackComplete`, published as public read-only
accessors (count, last/maximum/total latency in frames and ms), and a counter
of `_ContinueAsyncCommand` resumptions so slices per step can be counted. This
is a binding-generator change and is deliberately not emulated in the
fixtures.

The first measured baseline (gate 6, before `232d42f`) spent about 4,029
frames per `M=1` token step: every one of the 4,019 Device dispatches of a
step was followed by an await, and every await suspends the Udon Host for at
least one frame although a Blit is complete at issue. 3,096 of those
dispatches are the Q4 `MatMulNBits` K tiles of 128. `232d42f` makes Device
commands enqueue-only and yields by a per-slice work budget
(`vrc-schedule-udon-device-work`, described in
`docs/architecture/refactor-baseline-and-mlir-pipeline.md`). With the default
budget of 16 commands the Udon emulator, driving the generated binding
protocol over the same capacity-64 conversation, measures per token step:

| | before | after |
|---|---:|---:|
| Udon events | 4,021 | 254 |
| estimated frames | 4,029 | 262 |
| Udon instructions | 22.6 M | 3.6 M |
| Device Blits | 4,019 | 4,019 |

The instruction count falls because the per-dispatch suspend/resume round
trip, not the dispatch itself, dominated Host work. One continuation per step
still runs about 1.36 M instructions in a single event; it is the largest
remaining Udon slice. Frames are an emulator estimate (one frame per
continuation, two per readback): a structural lower bound, not a batchmode
frame count and not a time. Wall-clock time per step comes from the fixture
measurement described above.

Two generic follow-ups then cut the per-step Device work. `d7bde51` lets one
Q4 dispatch accumulate the whole K of every output word (up to 256 blocks of
32, bit-identical to the K tiles), so every TinyLlama projection is one
dispatch: 155 instead of 3,096 Q4 dispatches, about 1.2 G instead of about
3.1 G estimated Q4 texel loads, and 20 M instead of 406 M launched Q4
fragments per step. `7f79b20` and `121dd69` replace the command-count budget
by static Device (texel-load) and Host (estimated Udon instruction) costs per
command, with declared kernel fragment costs; the defaults close a slice at
about 2 ms of mid-range GPU work or about 16,000 Udon instructions. The same
emulator protocol measures per token step:

| | gate 7 | Q4 whole-K | + cost budget |
|---|---:|---:|---:|
| Udon events | 254 | 70 | 33 |
| estimated frames | 262 | 78 | 41 |
| Udon instructions | 3.6 M | 1.93 M | 1.79 M |
| Device Blits | 4,019 | 1,078 | 1,078 |
| median Blits per budget slice | 16 | 16 | 37 |

The Udon Host budget, not the GPU budget, closes most slices (median 12,230
Udon instructions per slice). The 1.35 M-instruction continuation after the
token upload is now three quarters of the step's Udon work and is the next
Host-lowering target.

Attributing that continuation to functions (from `host.udonasm.mlir`
function ranges) shows it is not transfer work: 1.22 M instructions (90%)
are `udononnxEnqueueRun`, whose `udononnxValidateRunBindings` re-validates
all 92 bindings and checks every input/output range pair (45 x 47 + 45 x 44
/ 2 = 3,105 overlap tests) with byte-addressed 64-bit metadata loads on every
run, and 117 K are the generated run prologue. Likewise 1.29 M of the 1.33 M
`Initialize` instructions are `tinyllama_initialize` with the inlined binding
creation, mostly byte-wise memsets of the 4,136- and 3,960-byte binding
tables. In the budget slices each dispatch costs about 210 Udon
instructions in the generated run, and the async resume dispatcher compares
the resume state against every registered state in turn, growing from about
1,000 to 6,200 instructions per slice through a step.

The generic transfer work of this plan's M2 then removed every Host transfer
but the token readback from a step and every transfer from Initialize and
Reset. `clEnqueueFillBuffer` (`f1c6302`, `d2a9293`), the sequence-state
reset (`9120f80`) and the attention-mask clear (`43d0d90`) issue Device
word fills, and uploads of up to 128 word-aligned bytes are written in place
by a word-fill Blit inside the issuing Udon event; the transfer shader
discards every fragment outside the written words, so no write reads the
surface back; a larger partial upload is staged by the binding and written
with a range-copy Blit instead of a readback-modify-upload (`7f911ba`).
Driving the gate 9 artifacts through the same emulator
protocol (frames: one per continuation, one per whole-buffer upload, three
per readback or partial upload):

| | gate 8 | M2 |
|---|---:|---:|
| `Initialize`: Udon events / frames / Blits | 45 / 44 / 0 | 1 / 0 / 45 |
| `Initialize`: binding uploads (words copied) | 44 (709,632) | 0 |
| `Reset`: Udon events / frames / Udon instructions | 45 / 44 / 45.8 K | 1 / 0 / 18.4 K |
| `Reset`: binding uploads (words copied) | 44 (709,632) | 0 |
| decode step: Udon events / frames | 33 / 37 | 30 / 32 |
| decode step: Udon instructions | 1.79 M | 1.76 M |
| decode step: Device Blits | 1,078 | 1,081 |
| decode step: binding commands | 2 uploads, 1 partial upload, 1 readback | 1 readback |

The partial attention-mask upload was a readback-modify-upload; in Unity
batchmode its readback waited for the GPU backlog of the step, so removing
it matters more than the frame estimate suggests. The largest event of a
step (1.38 M instructions) is unchanged in kind: it is the run validation
above, now also issuing the three input patches.

Two generic graph-level reductions then removed 159 of the 1,078 dispatches
of a step without a new kernel. `udoncl-simplify-onnx-views` (`59a5c88`)
turns a Transpose that only moves unit dimensions into a Reshape and removes
unit-step Slices and Expands that keep their input shape (88 Transposes, 23
Slices, 4 Expands). The Host planner (`e720a72`) folds a Device Mul read only
by one Add into that Add's OpTensor dispatch as `a * b + previous`, which is
bit-identical to the two dispatches (44 rotary-embedding products). The same
emulator protocol measures per decode step:

| | M2 | + views | + Mul/Add |
|---|---:|---:|---:|
| Device dispatches | 1,078 | 963 | 919 |
| Device Blits (with input patches) | 1,081 | | 922 |
| Udon events / frames | 30 / 32 | | 26 / 28 |
| Udon instructions | 1.76 M | | 1.70 M |

The 919 remaining dispatches are 244 OpTensor, 155 Q4 `MatMulNBits`, 132
Slice (44 of them the sequence-state advance), 89 Concat, 90 RMSNorm
passes, 46 unary, 44 Expand, 45 SGEMM, 22 Transpose, 22 Softmax, 22
activation and 8 others. Per layer 4 Slices, 2 Negs, 2 Concats and 5 OpTensor
dispatches are rotary embedding and 3 dispatches are SiLU gating; merging
them needs compiler-generated fused kernels, because every Device kernel is
still a hand-written library kernel wired by name through the runtime.

The generated kernel channel and generic Device fusion groups
(`docs/architecture/refactor-baseline-and-mlir-pipeline.md`) then provide
exactly that. Single-reader pointwise nodes and views fuse into their reader
under caps of 8 leaf buffers, 16 Loads per path and 64 expression nodes; each
group is one synthesized, bounds-proven kernel that is bit-identical to the
library dispatches it replaces. The capacity-64 step falls from 919 to 567
dispatches with four distinct synthesized kernels, each dispatched once per
layer (the key rotary embedding inside the key-cache Concat, the query rotary
embedding with its scale, the grouped-query key expansion with its transpose
and scale, and the SiLU gate), at most 4 Loads per fragment. The remaining 567
are 155 Q4 `MatMulNBits`, 90 RMSNorm passes, 88 synthesized, 68 OpTensor
(residual and mask Adds), 45 SGEMM, 44 sequence-state Slices, 23 Concat, 22
Expand, 22 Softmax and 10 others; the emulator counts 570 Blits per decode
step.

Three further generic rewrites cut that to 460 Blits. Sibling `MatMulNBits`
nodes that read one activation stack their weight rows into one node whose
result the readers Slice (`udoncl-merge-onnx-sibling-matmulnbits`; q/k/v and
gate/up: 155 to 89 Q4 dispatches). A MatMul whose right operand an Expand
repeats along the batch multiplies the unrepeated tensor with the repeated
left matrices stacked as rows (`udoncl-fold-onnx-repeated-matmul-operand`; the
22 value-cache Expands disappear). A Softmax is a reduction root of a fusion
group, so the attention mask Add it alone reads runs inside its kernel (22
OpTensor dispatches fewer). All three are bit-identical to the unfused
kernels. Emulator, capacity-64 decode step on the conversation artifact:

| | before | sibling Q4 merge | + repeated operand + Softmax root |
|---|---:|---:|---:|
| Device Blits | 570 | 504 | 460 |
| Udon events (frames) | 17 | 15 | 14 |
| Udon instructions | 241 K | 223 K | 217 K |

The remaining 460 are 89 Q4, 90 RMSNorm passes, 45 SGEMM, 45 OpTensor
(residual Adds), 44 sequence-state advance Slices, 132 synthesized kernels in
6 shapes per layer, and 15 others. The next large items are the advance
copies (double-buffered state would remove them, but changes the binding of
past/present tensors) and the second RMSNorm pass.

Those frame estimates charge one frame per continuation, which on a 90 Hz
client is the dominant cost of the remaining budget slices. The generated
binding now resumes ready continuations in the same Udon event under a
per-frame wall-clock budget (`device_work_budget.frame_microseconds`, default
8 ms, and `frame_continuations`, default 8; see the device work scheduling
section of `docs/architecture/refactor-baseline-and-mlir-pipeline.md`). A
frame therefore runs as many slices as fit the budget, and a synchronously
completed upload no longer costs a frame; only readbacks still wait. The
resulting frames per step depend on the real Udon time of a slice and need a
Unity measurement.

The first `fixed`-mode pump trace (90 Hz, the earlier 4 ms default) showed
that the pump stopped on the time budget in nearly every busy frame and
never at the continuation limit, and that the continuation
`_ContinueAsyncCommand` delivered at the start of a frame ran before the
window opened, uncounted and untimed. A busy frame therefore spent the first
continuation plus 4 ms plus one overrun: about 5 ms of Udon events on
average and up to 8.5 ms. A decode step needs about 16 to 17 ms of Udon time
(about 15 us per thousand Udon instructions and 30 us per Blit), so it
occupied about 2.9 busy frames, and the step's token readback waits added
about 3 to 4.6 frames with no continuation: 7.5 (conversation) and 5.9 (chat)
frames per step in total.

The binding now charges every Host slice of a frame to the window, the first
one included, defaults to 8 ms of the 11.1 ms frame, lets U# change both
limits at run time (`TrySetAsyncFrameBudget`) and offers an adaptive mode
that stops before a slice the running average predicts would overrun
(`SetAsyncFrameAdaptive`). Replaying the measured per-continuation times of
that trace under the new rule, keeping its readback-wait frames unchanged,
predicts these frames per step (busy frames per step in parentheses):

| budget | conversation | chat | multi-turn chat |
|---|---:|---:|---:|
| before, 4 ms with the uncharged first continuation | 7.50 (2.86) | 5.89 (2.89) | 5.54 (2.57) |
| 4 ms | 8.36 (3.71) | 6.56 (3.56) | 6.28 (3.31) |
| 6 ms | 7.50 (2.86) | 5.67 (2.67) | 5.33 (2.37) |
| 8 ms | 6.93 (2.29) | 5.07 (2.07) | 4.81 (1.85) |
| 8 ms adaptive | 7.21 (2.57) | 5.33 (2.33) | 5.02 (2.06) |

At 8 ms the predicted Udon time of a busy frame averages about 7 to 8 ms and
stays below 11.1 ms. These are predictions from one trace, not
measurements; the readback-wait frames, which the frame budget cannot
remove, are now the larger part of a step.

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
   table without placing bytes in Host globals. Historical 920/31 and
   1,033/35 static/mutable counts came from different compiler checkpoints and
   are not acceptance baselines. Replace them with one report from the next
   fresh accepted real-model artifact; do not encode either historical count
   as a compiler limit. Static resource IDs and dispatch uses remain fixed by
   AOT lowering and bypass the dynamic OpenCL Device-memory handle table.
4. Implement static shape/view canonicalization and Device workspace planning.
5. Replace `MatMulNBits` with the Device-resident implementation.
6. Implement and test every remaining generic operator required by the
   canonical model, including batched MatMul and generic ScatterND/cache update.
7. Extend generated model metadata with logical byte length, physical capacity,
   dynamic bounds, and stable Logical Tensor Binding identity. Zero-length
   tensors retain a valid capacity allocation.
8. Generate the complete 47/45 Prefill Validation Artifact ABI, unrolled
   physical allocation/bind MLIR, artifact-specific mutable-handle capacity,
   and dtype-correct `long[]`/`float[]` U# whole-buffer APIs. Remove the real
   target's dependence on the toy C/JSON tensor table without changing the
   synthetic fixture.
9. The ordinary prefill C Host lifecycle and model-independent generated
   physical helpers exist and have structural/small-fixture coverage. The
   remaining gate is full-model Native/Device and generated-U# prefill against
   every ORT logit and K/V output; structural artifact validation is not a
   numerical pass.
10. Identify semantic ONNX past/present pairs generically, generate an unrolled
    helper that materializes each
    `present[:, :, step:capacity, ...] -> past` with the generic Slice kernel,
    and prove the capacity-64, left-padded `M=1` profile against variable-length
    ORT execution. Add iterative decode, Device ArgMax, and EOS handling;
    compare every decode step with ORT.
11. `udon_string_t`, caller-owned `udon_int32_array_t`, the allocation-free
    UdonTokenizers encode core, the dedicated compiler lowering, package
    target, generated `string`/`int[]` U# binding, and a Unity VM value test are
    implemented. The four pinned tokenizer assets now have an atomic,
    content-addressed, receipt-verified cache. The generic build-time compiler
    validates the supported byte-fallback BPE schema and deterministically
    emits the real 32,000-entry vocabulary projection, 256 byte IDs, and 61,249
    merge records as project-local C tables. The generic runtime consumes the
    compiled normalizer/BOS metadata, uses binary scalar/pair lookup, and its
    real-table Native output matches the pinned Hugging Face `tokenizers`
    implementation for empty, ASCII, Japanese/BMP, emoji/surrogate, combining,
    whitespace, newline, and byte-fallback cases. The public CMake package now
    generates project-local flat tables, links the user's Host sources with
    them, and produces the real TinyLlama tokenizer Unity artifact. Its
    generated U# `TryTokenize(string, int[], out int)` binding is executed in
    Play Mode and matches the same real token IDs without direct heap/event
    manipulation. The fixed-C-workspace `udontokenizersEncodeRaw` path is also
    implemented and covered by both Native value tests and an installed CMake
    consumer that lowers a fixed token array through the Udon backend. Decode,
    special-token recognition and chat-template compilation followed and are
    described in section 6.6.
    The first fixed-four-token prefill input helper is also implemented: it
    widens tokenizer output to ONNX INT64, constructs attention-mask and
    position-ID arrays, resolves inputs 0--2 through the generated generic
    binding, and enqueues one ordinary OpenCL write per input. Native tests
    verify all three Device buffers and capacity failure, while the installed
    CMake consumer links the real generated TinyLlama tokenizer tables and
    lowers the helper through a separate small synthetic artifact. The Prefill
    Validation Artifact deliberately excludes tokenizer/conversation code and
    accepts already-tokenized low-level input. The previously generated real
    47-input/45-output artifact passed structural, static-weight, raw-heap, and
    generated-U# binding validation. The current fresh artifact is not
    accepted until the dedicated generated-literal path is exercised by a
    fresh focused build and numerical execution is green. The independent ORT
    oracle emits a model-hash and typed-shape manifest plus 45 little-endian
    FP32 raw outputs, so Unity does
    not need an NPZ parser or compiler-embedded expectations. A dedicated
    test-only U# controller is structurally covered: it writes the three
    fixed-prefill INT64 inputs, runs the ordinary generated lifecycle, and
    serially reads logits plus all 44 present K/V buffers through callbacks.
    The real-model target has the distinct artifact identity
    `tinyllama_real_prefill` and generated class
    `UdonclTinyLlamaPrefillBinding`; the bounded synthetic Transformer keeps
    `generic_onnx_transformer_tiny` and its existing binding class. Both may
    therefore be imported without duplicate C# types or generated-root
    collisions.
    The tokenizer remains independent of the Conversation Artifact. A
    model-independent Unity reader for the `udoncl.onnx-oracle.v1` sidecar now
    validates typed inputs, ordered FP32 outputs, shapes, byte lengths, model
    hashes, and numerical tolerances. A focused real-prefill Unity fixture now
    imports the ordinary artifact, drives the generated U# binding, and
    compares logits plus every present K/V tensor through that generic reader.
    The next gate is making this numerical fixture pass, then applying the same
    proof to generation state; numerical prefill and generation are not yet
    accepted until those focused executions are green.
    The first focused import exposed and fixed a generic Unity validation
    mismatch: zero-length logical buffers backed by positive physical capacity
    are valid OpenCL/ONNX resources. Unity artifact and fallback wrapper
    validation now accept the same `logical_bytes >= 0` contract already used
    by the compiler resource plan and Device buffer-view runtime.
    The subsequent fresh build exposed a generic static-resource join bug:
    reconciliation `buffer_id` values identify Host logical buffers, whereas
    canonical layout `buffer_id` values identify Unity Texture slots. Static
    logical-buffer materialization now joins both schemas through their stable
    `resource_id` and verifies the reconciled Texture identity instead of
    assuming the two buffer-ID spaces have matching numbers. A regression
    fixture deliberately uses different logical and physical IDs. The focused
    real-prefill numerical gate remains the acceptance point for this fix.
    Pool replacement must also carry that canonical identity into the tool
    adapter. When static lowering is both `applied` and
    `pool_replacement_ready`, `vrccl` now rebuilds the final Unity buffer
    configuration from the versioned canonical layout before binding assigned
    resources or emitting the manifest. Unready programs retain their legacy
    configured pool. This removes the remaining assumption that a mutable
    logical-handle ordinal is a physical Texture slot.
    The last recorded focused build before HEAD crossed this former
    Texture-identity barrier and reached Host backend-literal finalization.
    Commit `86437ad` then removed
    generated backend literals from generic source pooling/spill allocation and
    gave them their dedicated ownership path; its general regression is covered
    by `precommit=passed`. This is compiler-wide behavior, not an ONNX or
    TinyLlama exception. It still requires a fresh real artifact and generated-
    U# numerical run before the production path is accepted.
12. Finish and validate the existing example token-array generation Host,
    lifecycle/status exports, and generated async U# binding against the fresh
    production artifact. Do not add a second TinyLlama-only Host ABI.
13. Pass the remaining gates in order: generate the fresh focused prefill
    artifact; compare 128,000 logits and all 44 present K/V outputs through the
    generated U# binding; reuse K/V and match the agreed greedy conversation
    token sequence `[1701, 29916, 432, 17204]`; then add token-to-text decoding
    plus chat-template and special-token handling. Record the first performance
    and resource baseline, then integrate deterministic cached inference into
    the mandatory precheck.
14. After all consumers pass, remove the old ONNX execution-plan JSON,
    Material/arena scheduler, `static_inputs`, old runtime, and toy TinyLlama
    sample. Do not remove the only working comparison path earlier.

## 9. Completion checklist

The work is complete only when all of the following are true:

- generic C++ ONNX-to-MLIR lowering handles the pinned graph without a
  model-profile fallback;
- MNIST and TinyLlama use the same immutable AOT Device tensor facility;
- Q4 weights, activations, intermediates, and K/V remain Device resident;
- K/V is reused across every prompt/decode step by materializing each present
  suffix into distinct past storage with the generated generic Slice helper,
  without same-dispatch read/write aliasing;
- the C generation Host accepts and returns token ID arrays through
  compiler-generated U# bindings;
- Unity accepts an external prompt token array and returns the exact greedy
  token completion through EOS or the configured limit;
- every prefill/decode logit and K/V comparison passes against ORT;
- the real-model gate runs on every final precheck with cache reuse only for
  inputs and compilation;
- Unity reports zero console errors and cleans temporary assets;
- stale execution-plan and toy-runtime consumers are removed after migration.
