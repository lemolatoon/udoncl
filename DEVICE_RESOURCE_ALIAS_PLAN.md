# Device Resource Alias and Device Library Migration Plan

> Status: **Phase A is complete. Region sub-buffer ownership/transfers, queue
> tails, deferred resource release, stream association, UdonBLAS SGEMM Device
> pointers, and generated static-tensor `cl_mem` adoption are implemented.
> Remaining Phase B/C work is constant/dynamic sub-buffer Device addressing,
> the rest of UdonBLAS/UdonDNN Device-pointer migration, ONNX queue-tail library
> calls, and final generated-U# MNIST acceptance after those migrations.**
>
> Scope: UdonCL VRC fragment dispatch resource aliasing, Debug validation,
> standard `cl_mem`/sub-buffer semantics, and the UdonBLAS/UdonDNN migration
> from Host-array staging behavior to CUDA 13.0-compatible typed Device-pointer
> APIs backed by asynchronous UdonCL Device buffers.

## 1. Problem and platform fact

UdonCL exposes OpenCL buffer semantics, but its VRC backend executes a Device
kernel as a fragment draw. Shader-readable inputs become texture resources and
the single write resource becomes the render target of `VRCGraphics.Blit`.
Unity does not define sampling and rendering to the same RenderTexture in one
draw, and D3D11 removes conflicting shader-resource bindings.

ShaderGPT does not rely on that undefined behavior. It uses separate temporary
RenderTextures for ordinary operators, fixed-function blending for the small
set of updates exactly expressible as a blend equation, and partial draws for
cache updates. It does not use `RWTexture`, UAVs, framebuffer fetch, or general
same-resource fragment read/write.

Therefore the initial VRC target must reject a fragment dispatch whose read and
write roles resolve to the same physical GPU resource. This is a documented
UdonCL VRC backend capability restriction, not a claim about general OpenCL
buffer semantics.

## 2. Agreed semantic contract

1. A single VRC fragment dispatch must not bind the same physical resource or
   subresource in both a shader-read role and the render-target-write role.
2. Read/read aliasing is legal. `kernel(A, A, B)` is accepted when both uses of
   `A` are reads and `B` is the write.
3. The rule is based on physical resource identity, not only `cl_mem` handle or
   SSA identity. Distinct `cl_mem` wrappers that adopt the same physical token
   still alias.
4. With the current one-RenderTexture-per-allocation backend, sub-buffers of
   the same backing allocation also conflict even when their logical byte
   ranges do not overlap. A future backend may relax this only when it binds
   genuinely distinct GPU subresources.
5. Separate dispatches may read and write the same buffer in queue order.
6. Declaring or allocating a `CL_MEM_READ_WRITE` buffer remains legal.
7. The compiler initially performs no automatic alias legalization: no hidden
   ping-pong, scratch insertion, copy-back, or conversion to blending.
8. A library may explicitly implement its public contract with temporary
   buffers and copy-back. That is library code, not an implicit compiler
   transformation.
9. A future explicitly modeled blend pass may legalize only operations proven
   exactly equivalent to the selected fixed-function blend equation. It is not
   part of the initial implementation.

## 3. Resource roles and validation

Read/write roles have one source of truth in IR. Extend `vrc.dispatch` (or its
resource descriptor) with explicit per-resource access metadata. The current
single-output form may use an output-resource index internally, but manifest
generation, verifier diagnostics, and Debug runtime checks must all derive from
that same IR fact. Kernel names and handwritten argument-position tables are
not the semantic source of truth.

`distinct_resources` must not be used as the correctness rule because it also
rejects safe read/read reuse. Preserve it only for schema compatibility while
consumers migrate to access-aware validation, then remove it when no consumer
requires it.

Validation is layered:

- Always-on compile validation rejects a read and write that are the same SSA
  value.
- Unity buffer binding rejects read/write resources that statically resolve to
  the same known heap slot.
- Optional Debug runtime validation compares resolved physical tokens after
  argument resolution and before enqueueing the draw. A match returns
  `CL_INVALID_OPERATION`; no event or Blit is issued.
- The compiler generates pairwise scalar checks in each kernel-specific enqueue
  wrapper from access metadata. It does not build a runtime descriptor loop.
- UdonBLAS maps the restriction to `UDONBLAS_STATUS_NOT_SUPPORTED`, UdonDNN to
  `UDONDNN_STATUS_NOT_SUPPORTED`, and ONNX binding validation to
  `UDONONNX_STATUS_INVALID_VALUE` where the high-level call itself violates the
  supported contract.

Runtime validation is compiled out, not dynamically branched:

- CMake `RUNTIME_VALIDATION AUTO`: ON for Debug, OFF for Release-family configs.
- CMake `RUNTIME_VALIDATION ON|OFF`: explicit override per artifact target.
- Direct `udonclc`: explicit `--runtime-validation` or
  `--no-runtime-validation`, default OFF because the CLI has no configuration
  concept.
- Compile-time SSA/heap-slot validation remains enabled in every profile.
- The artifact manifest records `validation.read_write_alias` so Editor tests
  and users can audit whether runtime checks were emitted. Import validates the
  field but does not use it to change execution.

## 4. `cl_mem` and sub-buffer model

The UdonCL allocation, transfer, queue, and physical-resource layer uses
standard `cl_mem`. UdonBLAS/UdonDNN public declarations do not expose a
library-specific Device-view type and do not replace CUDA-compatible pointer
arguments with `cl_mem`; §5 defines the compiler-internal provenance mapping
between those public Device pointers and their originating `cl_mem` objects.

Implement `clCreateSubBuffer` using the standard region contract. A runtime
sub-buffer records its backing allocation identity, byte origin, and byte
length. Kernel argument recovery carries origin and length far enough to apply
addressing and bounds checks.

Whole-program optimization removes a sub-buffer object when its parent,
origin, length, uses, and lifetime are statically known. The compiler folds the
constant origin into resource addressing and leaves no runtime sub-buffer
handle or offset calculation. Dynamic, escaping, or exported sub-buffers keep
the runtime representation.

Regardless of constant folding, two views with the same current backing
RenderTexture cannot occupy read and write roles in one fragment dispatch.

## 5. UdonBLAS and UdonDNN Device API

The existing public implementations treat `float *` tensor operands as Host
arrays and perform blocking upload, readback, and `clFinish` internally. That
is a staging wrapper, not the desired CUDA-library-style Device API.

The public compatibility baselines are cuBLAS from NVIDIA CUDA 13.0 and the
cuDNN 9.16.0 Legacy API. NVIDIA identifies cuDNN 9.16.0 with CUDA 13.0 as its
recommended configuration. Public declarations retain the corresponding
argument types after library-prefix substitution: cuBLAS tensor operands remain
typed pointers such as `const float *` and `float *`; cuDNN tensor, filter,
output, workspace, alpha, and beta operands retain their `const void *`/`void *`
forms. Tensor/filter/output/workspace values are Device addresses, not Host
arrays. Alpha and beta follow each pinned API's Host scalar pointer contract.
Changing public tensor arguments to `cl_mem` would violate the repository API
compatibility contract.

UdonCL allocation and transfer APIs still use standard `cl_mem`. A typed Device
pointer passed to UdonBLAS/UdonDNN carries the identity and constant byte origin
of its originating `cl_mem`; compiler lowering resolves it back to that physical
buffer and folds statically known casts/origins away. This is an internal ABI
mapping, not a public UdonBLAS/UdonDNN extension or a Host mapping. Until general
Device pointer arithmetic is supported, an operand not provably derived from a
live compatible `cl_mem` returns the library's invalid-value status.

Migrate the baseline behavior destructively:

- Matrix, vector, tensor, filter, output, and Device workspace pointer values
  denote Device storage derived from caller-owned `cl_mem`; they never denote
  Host arrays in baseline APIs.
- Keep CUDA/cuDNN-shaped function names, descriptor parameters, dimensions,
  leading dimensions, increments, strides, and `alpha`/`beta` contracts.
- `alpha` and `beta` initially use Host scalar pointer mode. Device scalar
  pointer mode is a later capability and returns `NOT_SUPPORTED` if exposed.
- `udonblasSetStream` and `udondnnSetStream` associate a
  `cl_command_queue` with each handle, analogous to CUDA library streams.
- The platform type mapping is explicit: CUDA's `cudaStream_t` parameter maps
  to `cl_command_queue` on UdonCL. Function name, argument position, ownership,
  ordering, and asynchronous semantics remain aligned; this is the documented
  target-platform type substitution rather than an `Ext` API.
- Library handles stop creating private contexts and queues. Every operand must
  belong to the queue context.
- Calls enqueue Device work and return without internal Host transfers or
  `clFinish`. Users perform `clEnqueueWriteBuffer`, `clEnqueueReadBuffer`,
  marker/event creation, callbacks, and synchronization.
- UdonCL currently supports in-order queues only. A marker enqueued after a
  library call represents completion of all commands emitted by that call.
- Library APIs do not add event/callback parameters. C/U# callers use the
  existing OpenCL event transport.

Fast paths write directly to the caller output only when no read role aliases
it. Examples include GEMM/GEMV/DNN with `beta == 0` and distinct inputs.

CUDA contracts that inherently read and update an output remain supported by
explicit library implementations. SAXPY and `beta != 0` operations may read
the caller output, write an internal temporary, and copy the result back in a
later command. This preserves public semantics without allowing an illegal raw
fragment dispatch.

Temporary workspace policy follows CUDA libraries:

- Caller-provided cuDNN-style workspace is borrowed for the operation.
- UdonBLAS may accept optional caller workspace analogous to
  `cublasSetWorkspace`.
- If required workspace is absent, the library may allocate Device temporary
  storage.
- Enqueued commands retain internal temporary objects. The library drops its
  own reference after submission; the terminal command releases the final
  reference on completion. No blocking wait is introduced, and no temporary
  survives completed work unnecessarily.
- Allocation failure returns the library allocation status. No Host staging
  fallback is permitted in the baseline Device API.

Host-array convenience wrappers, if reintroduced, must have distinct names and
must be implemented only from public buffer allocation/transfer/Device APIs.
They are not acceptance paths and are not linked into ONNX generated Host code.

## 6. ONNX generated Host and MNIST

ONNX generated Host code must make every Device allocation and transfer
explicit. The compiler does not repair an illegal schedule.

For MNIST:

1. Adopt immutable weights as read-only `cl_mem` resources.
2. Allocate input, output, live intermediates, and any operator workspace in
   generated Host initialization.
3. Use liveness reuse only when the prior value is dead before the next
   dispatch. Never reuse an input slot as the output of the same dispatch.
4. Conv, bias Add, activation, pooling, GEMM, and final bias Add each bind
   distinct physical read/write resources.
5. Bias GEMM writes its product to an explicit temporary and the following Add
   writes to the public output. No in-place final Add is generated.
6. All generated UdonBLAS/UdonDNN calls receive CUDA-compatible typed Device
   pointers derived from `cl_mem` plus handles associated with the shared queue.
7. Input upload and output readback stay in user/generated Host orchestration,
   outside the libraries.
8. The generated U# binding remains callback-first and model-generic. The Unity
   test supplies the 784-element input through the public generated binding and
   reads the 10 logits through the same API; it does not access Udon VM heap
   slots directly.

The current conditional `beta != 0` reads in Device shaders may remain as an
optimization and NaN-suppression detail, but they are not the correctness
mechanism. Host resource separation and validation are authoritative.

## 7. Implementation phases

### Autonomous grilling closure (2026-11-07)

The user explicitly accepted every remaining recommended answer, so the
remaining design frontier is closed as follows:

1. Finish correctness infrastructure before changing library semantics. Phase
   A metadata-derived validation and Release code-elision proof precede the
   destructive UdonBLAS/UdonDNN migration.
2. Implement standard `cl_mem` region views before typed Device-pointer
   provenance. A runtime view owns a parent reference and records origin and
   length; a constant compiler-proven view is folded later without changing
   observable OpenCL lifetime or bounds behavior.
3. Never silently dispatch a dynamic sub-buffer as origin zero. Until Device
   addressing carries its origin, AOT kernel argument resolution rejects that
   path. Upload, readback, copy, alias identity, and deferred release may be
   implemented and accepted independently.
4. Keep the CUDA-compatible typed-pointer declarations. The compiler recovers
   their originating `cl_mem` and constant byte origin; no public cast helper,
   library-specific Device-view replacement, or Host-pointer dual meaning is
   introduced.
5. Make library calls asynchronous on the caller's queue. Upload/readback,
   event/callback handling, and final synchronization remain caller/generated
   Host responsibilities. Library-owned temporaries are command-retained and
   released by the terminal command.
6. Treat the currently green generated-U# MNIST test as a regression baseline,
   not Phase C acceptance. It must pass again after ONNX lowering and every
   UdonBLAS/UdonDNN operation use the new Device API without internal Host
   staging or `clFinish`.
7. Do not begin the real TinyLlama conversation gate until the same Device API
   contract passes MNIST. This preserves one generic implementation route
   instead of adding a model-specific escape hatch.

### Device-pointer and queue-tail grilling closure (2026-11-07)

The remaining Phase B representation questions are closed with the recommended
answers below. These are implementation requirements, not optional variants.

1. A CUDA-compatible typed tensor pointer has one meaning: it is an opaque
   Device address whose provenance resolves to a `cl_mem` and byte origin. At
   an ABI boundary its base value is the corresponding opaque `cl_mem`; Host
   arrays are never accepted by the same parameter.
2. The compiler must not execute ordinary integer pointer arithmetic on a
   `cl_mem`. It traces constant GEPs, validates the byte range, and either folds
   the origin into the generated dispatch or materializes a standard OpenCL
   sub-buffer. A dynamic origin remains rejected until the runtime Device
   address representation carries it.
3. `alpha` and `beta` remain Host scalar pointers, matching cuBLAS/cuDNN Host
   pointer mode. Tensor and workspace parameters are Device pointers. Argument
   position, not runtime guessing, determines which contract applies.
4. Generated static tensors are adopted during initialize as immutable
   `cl_mem` objects with `udonclAdoptStaticBuffer`, retained for the session,
   and released during dispose. Operators do not receive raw physical texture
   tokens as typed pointers.
5. Public UdonBLAS/UdonDNN calls enqueue through the existing
   `*_device_args` runtime helpers. Those helpers validate queue/context,
   retain command resources, and update the in-order queue tail. Library calls
   do not call `__vrc_opencl_await`, `clFinish`, upload, or readback.
6. Handle creation in the Device-pointer profile creates an unbound logical
   handle. `SetStream` performs the first caller-queue binding and retains that
   queue. It must not allocate a private context, queue, staging texture, or
   program first. The legacy Host-array profile is not linked into installed
   artifacts once an operation migrates.
7. Calls with `beta == 0` use a direct disjoint-output fast path. Calls that
   require reading the prior destination use a distinct temporary and enqueue
   copy-back; the terminal command owns the temporary until completion. They
   never rely on binding the same physical texture for read and write.
8. The ONNX Device graph is migrated at `createRuntimeDispatchers`, because
   that is the active MNIST/Transformer route. The unused non-Device
   `createRunExport` fallback is not modified as a substitute. Generated code
   must call public library APIs, while direct dispatch intrinsics remain
   implementation details of the libraries/runtime.
9. Once a public library operation uses Device-pointer semantics, an ONNX
   graph must not silently fall back to `createRunExport` and pass Host tensor
   addresses to that operation. The canonical synthetic Transformer uses the
   supported generic Q4 `block_size=32` contract and therefore remains a full
   Device graph. A fallback graph that would cross a migrated Device-pointer
   library boundary receives a lowering-time diagnostic until all of its
   operators have Device implementations; Host-only operators may retain
   their existing fallback meanwhile. Pointer guessing and model-specific
   runtime branches are forbidden.

### Phase A: access-aware dispatch and validation

- Add access metadata to dispatch IR and verifiers.
- Generate manifest access data from the same metadata.
- Allow read/read reuse; reject static read/write SSA and heap-slot alias.
- Add the build-time runtime-validation option and manifest audit field.
- Generate Debug physical-token checks before dispatch.
- Cover generic enqueue, direct intrinsic dispatch, and static-buffer adoption.

Completed (2026-11-07): `vrc.dispatch` carries access metadata;
same-SSA and known-heap-slot read/write conflicts fail statically; AOT direct
helpers reject same-token conflicts when compiled with validation; CMake
supports `RUNTIME_VALIDATION AUTO|ON|OFF`; direct `udonclc` supports the two
explicit validation flags; and manifests record `validation.read_write_alias`.
The Unity importer requires that audit field, its rejection path is covered,
and packaged examples have been regenerated. Udon Assembly lowering now emits
one physical-token comparison for every `read` named by dispatch metadata and
branches around the Blit with `CL_INVALID_OPERATION` on a match. The whole block
is absent when validation is disabled. Codegen tests cover both profiles, and
installed-package tests require checks in a validation-enabled ONNX artifact and
their absence from a Release UdonBLAS artifact. The AOT runtime,
installed-package consumer, and Native MNIST tests pass.

### Phase B: `cl_mem` and asynchronous library APIs

- Implement standard sub-buffer runtime objects and constant-sub-buffer
  elimination.
- Add conformance tests against pinned CUDA 13.0 declarations and keep public
  typed-pointer signatures while changing their semantics from Host arrays to
  Device addresses backed by `cl_mem`.
- Before library temporary allocation, implement the AOT in-order queue tail,
  marker-after-call dependency, `cl_mem` retain/release, command-held resource
  references, and completion-time deferred physical release. An event omitted
  by the caller must not discard pending queue state.
- Add `SetStream`, context validation, nonblocking enqueue, command retention,
  and terminal temporary release.
- Remove implicit Host packing/upload/readback/finish from baseline operations.
- Implement direct fast paths and explicit temporary/copy-back paths.
- Update installed CMake packages, examples, headers, and Native tests.

Implemented prerequisite checkpoint (2026-11-07): the AOT in-order queue keeps
an internal event even when the caller omits one, records a monotonically
ordered tail, and makes a marker inherit the outstanding tail. Commands retain
each distinct `cl_mem` they use. `clReleaseMemObject` invalidates the user handle
at reference count zero but defers physical release until every retaining
command has completed; waiting for a later in-order event completes and drops
the retains of preceding commands. The AOT runtime test covers deferred release
through `clFinish`. Each command can retain the five distinct resources needed
by the current generated kernels; exceeding that bound fails before dispatch
instead of silently dropping a retain. This keeps the generated MNIST raw heap
within its 32 KiB budget. Sub-buffers and the public UdonBLAS/UdonDNN Device-API
migration remain Phase B work.

Implemented region-view checkpoint (2026-11-07): Native and AOT
`clCreateSubBuffer` validate the OpenCL region form, access compatibility,
four-byte backend alignment, non-empty in-bounds ranges, and reject nested
sub-buffers. Views retain their parent, share physical identity, translate
upload/readback/copy offsets, detect same-backing overlap, and defer root
physical release until view commands and references are gone. Blocking AOT
transfers without a caller event now complete their internal event and release
its retained resources. AOT kernel argument recovery still rejects a runtime
sub-buffer until origin-aware Device dispatch is implemented; constant-view
elimination and dynamic Device addressing therefore remain Phase B work.

Stream-API migration checkpoint (2026-11-07): the CUDA-compatible
`udonblasSetStream`/`udonblasGetStream` and
`udondnnSetStream`/`udondnnGetStream` entry points are public. On the Native
runtime both libraries can switch to a caller queue from either the current or
a new context: they retain the replacement queue, borrow its queue-owned
context, destroy every old context-owned program, kernel, and staging buffer,
and reconstruct their resources on the replacement context. Native ownership
tests cover both
transitions, caller reference release, final teardown, and zero leaked OpenCL
objects. Null and invalid queues fail without changing association. The AOT
singleton accepts its existing queue and rejects replacement rather than
pretending that an arbitrary handle is usable. This is deliberately a
transition checkpoint. Installed UdonBLAS SGEMM now uses caller-owned Device
pointers, the associated queue, and the asynchronous `*_device_args` helper
without an internal transfer or finish. Other UdonBLAS operations and UdonDNN
still contain Host-staging paths, so Phase B remains incomplete until those
baseline tensor pointers are migrated too.

The current Device-graph path still lowers through `createRuntimeDispatchers`
to direct `__vrc_opencl_dispatch_*` intrinsics. Its generated initialize export
now adopts every root static tensor as an immutable `cl_mem`, stores that handle
for the session, and releases it during dispose. Static physical tokens use the
negative encoding `-(resource_id + 1)`, keeping them disjoint from positive
RenderTexture-pool ids after a C-runtime round trip. Udon lowering dynamically
selects Texture2DArray resources for negative readable tokens and shares the
selection scratch slots across all dispatches to stay within the 512-slot Udon
Heap limit. Phase C must next replace the actual Device-graph dispatcher route
with public queue-tail Device-library calls; the legacy fallback is not the
migration target.

UdonDNN Device-pointer checkpoint (2026-11-07): the installed Device profile
now creates a logical handle without a private context, queue, program, kernel,
or staging allocation. `udondnnSetStream` retains the caller queue. The
cuDNN-shaped ConvolutionForward, ActivationForward, PoolingForward, and
OpTensor entry points interpret tensor/filter/output pointers as caller-owned
`cl_mem`-backed Device pointers and enqueue only through the queue-tail
`*_device_args` helpers. `beta == 0` writes directly to a distinct caller
output; `beta != 0` dispatches into a temporary, enqueues copy-back, and drops
the library reference immediately so command completion performs final
release. Physical read/write alias rejection maps to `NOT_SUPPORTED`, while
temporary allocation failure maps to `ALLOC_FAILED`. The AOT runtime test
covers all four public calls, no internal wait, marker-after-call completion,
temporary deferred release, alias rejection, and stream lifetime. Softmax,
BatchNorm, and the shared private UdonATen backends return explicit
`NOT_SUPPORTED` in this profile instead of touching absent legacy staging
resources. They remain Phase B work and no Host-array fallback is permitted.

### Phase C: ONNX and MNIST schedule

- Update ONNX C++ lowering and generated Host calls to the Device APIs.
- Emit explicit allocations and legal liveness reuse.
- Keep GEMM product and final bias output physically distinct.
- Remove temporary hard-coded heap diagnostics from Unity tests.
- Regenerate artifacts and packaged examples.
- Run Native numerical tests, generated-U# Unity MNIST numerical acceptance,
  console inspection, reimport, and cleanup validation.

### Phase D: cleanup and documented limitations

- Remove obsolete Host-array baseline paths and stale plan text.
- Document the VRC same-dispatch alias restriction, Debug validation settings,
  error mapping, sub-buffer behavior, and library migration.
- Leave Blend legalization and genuinely read-write fragment storage as future
  target capabilities, not silent fallback behavior.

## 8. Required tests

- IR verifier: same SSA read/write rejected; duplicate read accepted.
- Static binding: same heap slot read/write rejected with kernel and argument
  diagnostics.
- AOT Debug: same handle rejected; distinct handles adopting the same physical
  token rejected; different tokens accepted; duplicate reads accepted.
- Release artifact: runtime comparison blocks are absent and manifest records
  validation disabled.
- Sub-buffer: constant region folds away; dynamic region preserves origin and
  length; invalid region fails; same-backing read/write rejects.
- Library API: CUDA 13.0 declaration conformance, caller-created buffers and
  queue, typed Device pointers resolved from `cl_mem`, no internal Host transfer,
  asynchronous completion through marker, context mismatch diagnostics,
  `beta == 0` fast path, SAXPY and `beta != 0` temporary/copy-back, terminal
  release, and allocation failure.
- ONNX lowering: every generated dispatch has disjoint read/write physical
  resources, including Conv bias, activation, pooling, GEMM product, and final
  bias.
- MNIST Native oracle: all 10 logits within the existing tolerance.
- MNIST Unity: generated U# only, callback completion, 784 input elements, 10
  logits, no same-resource warning, no timeout, console errors zero, temporary
  assets cleaned.

## 9. Commit and acceptance gates

Use coherent commits for Phase A, Phase B, and Phase C where possible. Before
every commit, stage only the intended changes and run the repository-required
`python scripts/precommit.py`; commit only when it prints `precommit=passed`.
Unity MCP operations are delegated to the required Luna/xhigh validation agent.

This plan is complete only when the Device APIs are the installed-package
baseline and the generated-U# MNIST path passes Native and Unity numerical
validation. Passing only the existing Host-array wrappers or direct heap tests
does not satisfy the plan.
