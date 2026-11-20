# Refactor Baseline and MLIR Pipeline

Date: 2026-11-17

This document records the current codebase structure, the actual MLIR
transformation paths, the tests that protect them, and a staged boundary for a
large refactor. It describes the repository as it exists today. It is not a
replacement for the feature plans in the repository.

## Executive summary

The intended production architecture is:

```text
C host / OpenCL device / ONNX model
             |
             v
       MLIR-based compiler
             |
      +------+------+
      |             |
      v             v
 Udon Assembly   fragment shaders
      |             |
      +------+------+
             v
   versioned Unity artifact
             |
             v
 generated U# binding and prefab
```

The main technical pieces exist, but their ownership is not yet clean:

1. `tools/vrccl/vrccl.cpp` is both the CLI and the effective pipeline
   specification. Several important transformations are ordinary helper
   functions rather than registered MLIR passes.
2. The C++ ONNX-dialect path and the Python execution-plan path coexist. They
   overlap in host/session generation, static resources, artifact creation,
   and Unity integration.
3. Static device resource lowering is already well decomposed into small
   passes, but the order and cross-pass attributes are mostly stringly typed.
4. `FinalizeONNXHost.cpp`, `LowerScalar.cpp`, and
   `PropagateDeviceResourceProvenance.cpp` each contain multiple internal
   subsystems.
5. The CMake facade, artifact schema, generated binding ABI, and Unity importer
   each duplicate part of the compiler's domain model.
6. UdonDNN currently packages UdonATen device kernels even though UdonATen
   depends on UdonDNN at the public API level.

The safest first refactor is therefore not a rewrite. First freeze the stage
contracts with characterization tests, extract named pipeline builders from
`vrccl.cpp`, and split test orchestration. Then repair library ownership and
artifact authority one boundary at a time.

## Repository structure and current ownership

### Compiler dialects and transformations

```text
include/VRC/Dialect/                 TableGen dialect/op/type declarations
lib/Dialect/                         dialect implementations
include/VRC/Conversion/Passes.td     public pass names and options
lib/Conversion/LLVMToHighLevel/      LLVM recovery
lib/Conversion/BoundaryFrontend/     C/OpenCL boundary recovery and linking
lib/Conversion/ONNXToUdonHost/       ONNX dialect lowering and host generation
lib/Conversion/HighLevelToVRCGPU/    device lowering
lib/Conversion/HighLevelToUdonHL/    generic host lowering
lib/Conversion/VRCToUdonHL/          resource, async, and runtime lowering
lib/Conversion/HostStoragePipeline/  final storage pipeline
lib/Conversion/UdonHLToUdonAsm/      Udon Assembly IR lowering
lib/Analysis/                        liveness and host storage allocation
lib/Target/HLSL/                     HLSL/ShaderLab emission
lib/Target/UdonAssembly/             textual Udon Assembly emission
lib/ABI/Manifest/                    artifact ABI and generated U# binding
```

`tools/vrc-opt/vrc-opt.cpp` registers the standard and project dialects,
passes, and pipelines. `tools/vrccl/vrccl.cpp` (`udonclc`) additionally owns
input-mode selection, pass ordering, linking, semantic helper calls, artifact
planning, and file emission.

### Runtime and CUDA-shaped libraries

```text
UdonCL
├── compiler and artifact production
├── DeviceMemory/OpenCLAOT runtime
└── native fragment/OpenCL test runtimes

UdonBLAS ───────────────> UdonCL
UdonDNN  ───────────────> UdonCL
UdonATen ───────────────> UdonDNN ──> UdonCL
UdonONNXOperators ──────> UdonCL
UdonONNXRuntime ────────> UdonCL
UdonTokenizers ─────────> UdonCL
```

The intended public direction is clear, but two implementation details violate
it:

- `cmake/UdonDNNConfig.cmake` packages both `libraries/udondnn/device` and
  `libraries/udonaten/device`. UdonATen needs a real device target of its own.
- ONNXOperators and ONNXRuntime privately rebuild or embed parts of the
  OpenCLAOT/DeviceMemory runtime. Runtime composition should become explicit.

Public UdonBLAS/UdonDNN signatures remain outside this refactor boundary. Their
CUDA compatibility contract must be preserved while internal ownership is
repaired.

### Build and packaging

`cmake/UdonCLConfig.cmake` currently owns all of the following:

- public source-library APIs;
- target graph traversal through custom `UDONCL_*` properties;
- host/device compilation and linking;
- ONNX host generation;
- host storage capacity planning;
- Unity artifact generation and validation.

Runtime roles are partly recognized by source basenames such as `OpenCLAOT.c`
and `DeviceMemory.c`. Those filenames currently act as an undocumented ABI.
The replacement should use explicit module roles while keeping the public CMake
functions as compatibility wrappers during migration.

### Unity package

`unity/Packages/com.lemolatoon.udoncl/Editor/UdonclArtifactImporter.cs` owns
generic import infrastructure and application-specific construction in one
file:

- transaction and cleanup;
- manifest interpretation;
- Udon program assembly;
- shader, material, texture, and render-texture creation;
- generated U# source installation and compilation;
- generic runtime prefab binding;
- MNIST, TinyLlama, BLAS, DNN, and ATen demo construction;
- TinyLlama chat UI.

The target decomposition is:

```text
manifest parser/validator
  -> import plan
  -> transactional resource handlers
  -> binding installer
  -> generic prefab assembler
  -> sample-owned recipes and UI
```

### Two ONNX architectures currently coexist

The intended path is:

```text
ONNX protobuf
  -> ONNX dialect
  -> C++ MLIR passes
  -> generic provider calls and C/OpenCL-style host
  -> normal UdonCL pipeline
  -> compiler-generated artifact and U# binding
```

The legacy path is still installed and tested:

```text
ONNX protobuf
  -> runtime/onnx Python planner
  -> execution-plan JSON and provider-specific materialization
  -> UdonOnnxModelAsset/UdonOnnxRuntimeBehaviour
```

The legacy implementation is mainly in:

- `runtime/onnx/udon_onnx.py`
- `runtime/onnx/udon_pages.py`
- `runtime/onnx/udon_weights.py`
- `unity/Packages/com.lemolatoon.udoncl/Runtime/UdonOnnxModelAsset.cs`
- `unity/Packages/com.lemolatoon.udoncl/Runtime/UdonOnnxRuntimeBehaviour.cs`

It must not be removed until MNIST and real TinyLlama pass their U# numerical
tests through the C++ ONNX-dialect/generated-binding path.

## Actual MLIR transformation paths

### 1. C and OpenCL source to LLVM dialect

`cmake/UdonCLConfig.cmake` compiles host and device sources separately.

```text
Host C:
  clang -target spir-unknown-unknown -x c -std=c11 -O1 -fno-inline

Device OpenCL C:
  clang -target spir-unknown-unknown -x cl -cl-std=CL1.2 -O0
```

The modules are linked by role before `udonclc` imports LLVM IR into MLIR.

The registered C boundary recovery pipeline in
`lib/Conversion/BoundaryFrontend/BoundaryFrontend.cpp` is:

```text
mem2reg
-> vrc-raise-llvm-boundary-device
-> vrc-raise-llvm-memory (raw fallback allowed)
-> vrc-raise-llvm-raw-memory
-> vrc-raise-llvm-to-scalar
-> canonicalize
-> vrc-link-c-boundary
```

The separately registered high-level recovery pipeline also lifts CFG to SCF:

```text
mem2reg
-> RaiseLLVMMemory
-> RaiseLLVMRawMemory
-> RaiseLLVMToScalar
-> LiftControlFlowToSCF
```

The Unity host path in `vrccl.cpp` reconstructs a similar sequence manually
and adds SROA, export/calling-convention normalization, optional OpenCLAOT
intrinsic lowering, and export restoration. This duplication is the first
pipeline extraction target.

### 2. ONNX protobuf to provider calls

`scripts/compile_onnx_mlir.py` currently owns the complete pass order:

```text
repo-private udoncl-onnx-frontend
  ONNX protobuf -> ONNX dialect
  udoncl-specialize-onnx-inputs
  udoncl-infer-onnx-shapes
  shape-inference
  simplify-shape-related-ops-onnx
  constprop-onnx
  standard-func-return
  canonicalize

  udoncl-canonicalize-onnx-library-patterns
  udoncl-infer-onnx-shapes
  udoncl-materialize-onnx-constants
  udoncl-externalize-onnx-initializers
  udoncl-externalize-onnx-graph-constants
  udoncl-identify-onnx-sequence-state
  udoncl-lower-onnx-to-host
  udoncl-validate-onnx-host-graph
  udoncl-plan-onnx-host-graph
  udoncl-materialize-onnx-host-globals
  udoncl-materialize-onnx-host-model-registry
  udoncl-attach-onnx-host-metadata
  udoncl-finalize-onnx-host
  udoncl-cleanup-onnx-host-source
  canonicalize
  udoncl-emit-onnx-host-metadata
  emit-bytecode
```

ONNX-MLIR is an external SDK dependency. Its installed headers and libraries
provide protobuf import, the ONNX dialect, and selected upstream passes to the
repository-built private frontend. The SDK installs no compiler tool and
contains no UdonCL-specific source. Dimension bindings and result-type repair
remain typed repo-owned passes before upstream shape inference, so ordinary
UdonCL pass changes do not rebuild the external SDK.

Python orchestrates tools; the semantic lowering is C++. However, keeping the
pass order only in Python makes it an implicit compiler contract. This order
should become a named C++ pipeline that both the script and tools call.

`ONNXToUdonHostCore.cpp` converts ONNX operations to private provider
declarations and ordinary `func.call` operations.

Before, from `tests/frontend/onnx_to_udon_host.mlir`:

```mlir
%sum = "onnx.Add"(%left, %right)
  : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
"onnx.Return"(%sum) : (tensor<1x4xf32>) -> ()
```

After, schematically:

```mlir
func.func private @__udon_onnx_add_0(...)
  attributes {
    udon.onnx.provider = "udonatenAdd",
    udon.onnx.operation = "Add"
  }

%sum = func.call @__udon_onnx_add_0(%left, %right) : (...) -> ...
func.return %sum : ...
```

Current ownership mapping includes:

| ONNX operation | Provider |
|---|---|
| Add/Mul/Sub/Relu/Sigmoid/Tanh | UdonATen |
| Conv/AveragePool/Softmax | UdonDNN |
| Gemm/MatMul | UdonBLAS |
| Microsoft MatMulNBits | UdonONNXOperators |
| custom RmsNorm | UdonATen |
| other supported ONNX operations | UdonONNXOperators |

`ExternalizeONNXInitializers.cpp` replaces dense constant payloads with private
static tensor symbols and `udoncl.static_tensors` metadata. The bytes are placed
in the artifact's static-tensor store. `IdentifyONNXSequenceState.cpp` records
KV-like state in `udoncl.onnx_sequence_states`.

### 3. Provider graph to executable host

`FinalizeONNXHost.cpp` currently performs all of these in one pass:

- graph reading and validation;
- static shape and ABI validation;
- tensor lifetime and layout planning;
- OpenCL context, queue, model, and session lifecycle generation;
- public input/output and static buffer generation;
- BLAS/DNN/ATen/ONNXOperators call generation;
- initialize/write/run/read/dispose exports;
- sequence/KV state API generation;
- binding and device graph metadata.

Graph validation and graph-level planning are now explicit prerequisites.
`udoncl-validate-onnx-host-graph` validates the portable provider graph without
rewriting it. `udoncl-plan-onnx-host-graph` then records backend selection,
device-storage aliases, operator scratch capacity, and mutable-buffer capacity
in `udoncl.onnx_host_plan`. Finalization consumes that plan before performing
any destructive materialization. Global storage, the model registry, module
metadata (`udoncl-attach-onnx-host-metadata`), and source-graph cleanup are
separate passes with their own prerequisite markers. Session/binding export
and runtime-dispatch materialization remain to be split from finalization.

#### Device fusion groups

A Device dispatch is one full-surface Blit plus a few hundred Udon
instructions, so the number of dispatches, not the work inside one, bounds a
graph step. Planning therefore groups nodes that only combine or re-index
elements, and finalization emits one synthesized kernel per group through the
generated kernel channel (section 7) instead of one library dispatch per node.
The rule is generic: it names node kinds, never a model, a node or a shape.

- **Fusible kinds** (`deviceFusionKind`, `ONNXHostFusion.h`): the f32
  per-element nodes Add, Mul, Relu, Sigmoid, Tanh, Cos, Sin and Neg, the
  re-indexing nodes Slice, two-input Concat, Expand and Transpose, and the
  metadata aliases Reshape, Flatten, Identity, Squeeze and Unsqueeze, each
  within the static-shape limits of its library kernel.
- **Groups.** A group has one root, the only node whose result has a buffer,
  and interior nodes whose results exist only inside the root's kernel. A
  fusible producer joins the group of its readers when its result is not a
  graph input or output (which covers sequence state) and every reader is in
  that group. Everything else a group reads is a leaf buffer; a known f32
  scalar next to a non-constant Add or Mul operand is an immediate. Groups
  grow greedily from the last node to the first, breadth-first from the root.
- **Caps** (`deviceFusionMax*`): at most 8 leaf buffers, 16 Loads on the path
  of one fragment (a Concat executes one side) and 64 expression nodes,
  counted on the expression as a tree. A producer that would exceed a cap
  stays materialized. A node on its own keeps its library kernel, and a group
  that would only absorb aliases is not formed.
- **Plan.** `udoncl.onnx_host_plan` carries `device_fusion_roots`, the node
  whose dispatch computes each node. Storage planning keeps every leaf live
  until the root's dispatch and gives interior results no buffer, so a fused
  kernel never reads the buffer it writes. Every later pass reads the groups
  from the plan. `--udoncl-plan-onnx-host-graph=fuse-device-nodes=false`
  plans the unfused graph.
- **Kernel.** `ONNXHostFusionKernel.cpp` emits the `gpu.func` into the carrier
  `gpu.module @udononnx_fused_kernels` and a dispatch declaration with
  `vrc.synthesized_dispatch`. Its arguments are the leaf buffers and the
  root's buffer; all shapes are static, so it takes no scalars. It launches
  over the 8192 wide rows that hold the root's elements and writes zero
  outside them. Views are index arithmetic in front of the Loads, a Concat is
  an `scf.if`, and the float operations repeat the library kernels' formulas
  in their order without fast-math flags. Groups with the same expression,
  shapes and immediates share one kernel.
- **Proof after canonicalization.** The frontend canonicalizes after
  finalization and `udonclc` proves the kernel later. Every index is
  therefore bounded by interval arithmetic over constants, `remui` and
  `minui`, never by a branch condition inside the expression, which
  canonicalization can separate from the value it guards.

Representative output:

```mlir
func.func @udon_onnx_binding_create(...)
func.func @udon_onnx_binding_bind_tensor(...)
func.func @udon_onnx_binding_bind_all(...)
func.func @udon_onnx_binding_enqueue_initialize(...)
func.func @udon_onnx_binding_enqueue_run(...)
func.func @udon_onnx_binding_dispose(...)

%input = func.call @clCreateBuffer(...)
  {vrc.static_public_resource_binding = {
    public_access = "input",
    public_index = 0,
    ...
  }}
```

The module carries contracts such as:

```text
udoncl.onnx_binding
udoncl.onnx_device_graph
udoncl.onnx_operator_counts
udoncl.mutable_handle_capacity
udoncl.mutable_buffer_bytes
udoncl.onnx_lowering = "library-calls-v2"
vrc.static_public_resource_binding
vrc.async_effect = "may_suspend"
udoncl.export
```

The pass should first be split internally into graph reader, validator, tensor
plan, provider lowering, session API builder, sequence-state builder, and
metadata builder. Creating several externally visible passes before those
interfaces are stable would make the pipeline harder to change.

### 4. Static device resource analysis

The current analysis sequence in `vrccl.cpp` is:

```text
attachDeviceResourceOperandContracts        (ordinary helper)
DiscoverStaticDeviceResources
PlanStaticDeviceResourceContextSpecialization
SpecializeStaticAllocationContexts
DiscoverStaticDeviceResources               (specialized graph)
PlanStaticDeviceResourceContextSpecialization
InferStaticDeviceResourceRequirements
PropagateDeviceResourceProvenance
ResolveStaticTransferBindings
ClassifyStaticPublicDeviceResources
AnalyzeStaticDeviceResourcePersistence
BuildStaticDeviceResourceInventory
AllocateUnityTextures
ResolveStaticPublicResourceAliases
```

Physicalization then runs:

```text
ResolveStaticKernelBindings
ReconcileStaticUnityBuffers
PlanStaticUnityBufferLayout
MaterializeStaticLogicalBuffers
MaterializeStaticDispatchResources
PlanStaticPhysicalAllocationSpecialization
SpecializeStaticPhysicalAllocationPaths
MaterializeStaticPhysicalAllocations
MaterializeStaticPhysicalResolves
LowerCheckedAssignedBufferResolves
MaterializeFixedPhysicalReleases
EliminateDeadPhysicalAllocationFallbacks
AssessStaticUnityResourceLowering
```

Most entries are already one pass per file. The refactor should preserve these
small pass boundaries and extract only their orchestration first.

#### Concrete resource identity example

Before, from `tests/frontend/static_device_resources.mlir`:

```mlir
%buffer = func.call @clCreateBuffer() : () -> i32
```

After discovery:

```mlir
%buffer = func.call @clCreateBuffer()
  {vrc.static_device_allocation_site = {
    function = @RunDirect,
    occurrence = 0,
    primitive = "clCreateBuffer",
    context_id = ...
  }}
```

Calls receive context-sensitive edge identity:

```mlir
func.call @allocate()
  {vrc.static_device_call_site = {
    root = @RunRepeat,
    caller = @RunRepeat,
    callee = @allocate,
    occurrence = 0,
    context_id = ...
  }}
```

Provenance is propagated through values, arguments, results, aggregates,
memrefs, pointer cells, and calls using attributes including:

```text
vrc.static_device_argument_provenance
vrc.static_device_function_result_provenance
vrc.static_device_operand_provenance
vrc.static_device_result_provenance
```

#### Concrete kernel binding example

Before, from `tests/frontend/static_kernel_bindings.mlir`:

```mlir
%event = func.call @enqueue(%input, %output, %scale) : (...) -> !vrc.event
```

After binding resolution:

```mlir
%event = func.call @enqueue(%input, %output, %scale)
  {vrc.static_kernel_binding_snapshot = {
    kernel_id = 7,
    resources = [
      {operand_index = 0, resource_id = 0, access = "read"},
      {operand_index = 1, resource_id = 1, access = "write"}
    ],
    scalars = [{operand_index = 2, type = i32}]
  }}
```

The module-level handoff is:

```text
vrc.static_device_resource_inventory
vrc.unity_texture_assignments
vrc.static_public_resource_aliases
vrc.static_unity_buffer_reconciliation
-> vrc.static_unity_buffer_layout_plan
```

The final plan records heap slot, Udon type, logical and physical capacity,
public access/index, and readiness.

### 5. Async conversion and state machine generation

The Unity host path first propagates `may_suspend`, optimizes, and clones the
module. The async candidate runs:

```text
LowerHostToAsync
MLIR AsyncFuncToAsyncRuntime
PrepareUdonAsyncRuntime
DiscoverHostRematerialization
AllocateHostStorage
AnalyzeHostStorage
LowerUdonAsyncRuntime
bindUnityBufferOperations                  (ordinary helper)
```

The synchronous candidate runs `LowerHostToUdonHL`, runtime intrinsic
substitution, and buffer binding. The async clone is selected only if upload,
readback, or await operations remain.

Before, from `tests/frontend/host_to_async.mlir`:

```mlir
func.func private @wait_child(%event: !vrc.event, %value: i32) -> i32
  attributes {vrc.async_effect = "may_suspend"} {
  %status = vrc.await %event : (!vrc.event) -> i32
  %result = arith.addi %value, %status : i32
  return %result : i32
}
```

After `LowerHostToAsync`, schematically:

```mlir
async.func private @wait_child(...) -> !async.value<i32> {
  %token = vrc.event_as_async %event
  ...
  async.await ...
}
```

After standard async-to-runtime and preparation, the IR contains coroutine
suspension plus compiler metadata:

```text
vrc.async_frame_id
vrc.async_resume_pc
vrc.async_live_object_ids
vrc.async_result_object_ids
vrc.async_resume_routes
vrc.async_object_state_encoding
vrc.async_frame_store / vrc.async_frame_load
vrc.async_object_payload_store / load
```

Before `LowerHostToAsync`, the Unity pipeline runs
`vrc-schedule-udon-device-work` once the async effects have fixed the export
ABI. On Udon a `vrc.dispatch`, `vrc.buffer_copy` or `vrc.buffer_fill` is a
Blit on the in-order graphics queue and is complete at issue, so an await of
its event is replaced
by a cooperative checkpoint that is taken only when a per-slice budget is
full. Remaining awaits reset the budgets. Device commands are therefore
enqueue-only and one Udon event issues a bounded slice of work instead of one
command per frame.

Every command is charged two static costs, kept in two saturating instance
globals:

- Device cost in texel-load units: `fragment-weight` (default 16) per
  launched fragment for raster, output write and Blit overhead, plus one per
  texel load. A kernel may declare `fragment_cost` next to its
  `argument_bounds` in the Unity library JSON: fixed `loads` per fragment,
  `per_argument` loads per unit of a scalar argument (a loop trip count such
  as the Q4 `block_count` or the SGEMM `k`), and `active` arguments whose
  product bounds the fragments doing that work (such as `m * n`). The values
  are the dispatch site's constant scalar arguments, or the declared maxima
  when an argument is not constant. Undeclared kernels read every read
  resource once per launched fragment. The default budget of 134,217,728
  units is about 2 ms of GPU work on a mid-range desktop GPU: on an RTX 4070
  Ti a trivial full-surface 8192 x 16 Blit (about 2.1 M units) measured about
  19 microseconds, and texel loads are cheaper than fragments.
- Host cost in estimated Udon instructions: 32 per bound kernel argument plus
  32 for the Blit, and 8 per Host operation the command's block runs since
  the previous command, fitted to the Udon VM emulator (about 430 Udon
  instructions per dispatch in the budget slices of the TinyLlama graph). The default budget of 16,384 is
  about 1.5 to 2 ms of Udon VM time.

A slice ends after the command that fills either budget. The budgets are pass
options (`device-cost-budget`, `host-cost-budget`, `fragment-weight`) and
per-library settings in the Unity library JSON:
`"device_work_budget": {"device_cost": N, "host_cost": N,
"fragment_weight": N}`.

Host writes of Device memory use the program's transfer material (the
generated `buffer-copy.shader`). Its default mode 0 copies a whole surface
for whole-buffer copies and binding uploads; mode 1 (range copy) and mode 2
(word fill) write only the destination words `[First, First + Count)` and
discard every other fragment, so the rest of the buffer keeps its contents
without a readback. `vrc.buffer_fill` (from `clEnqueueFillBuffer` through
`__vrc_opencl_buffer_fill`, and from the generated sequence-state reset) is
one mode-2 Blit with a 1- to 32-word pattern read from the raw heap at issue.
A `vrc.buffer_upload` of a mutable buffer of 1 to 128 word-aligned bytes is
written the same way inside the issuing Udon event, with the upload as the
pattern and a complete event; larger uploads stay binding commands, and the
binding writes a partial one with a mode-1 Blit of its staging surface. The
Device stores 32-bit words, so the AOT `clEnqueueFillBuffer` rejects byte
ranges that are not word-aligned with `CL_INVALID_VALUE`.

A `vrc.buffer_copy` (from `clEnqueueCopyBuffer`) of two whole, equally shaped,
statically bound surfaces stays one mode-0 Blit. Every other copy is marked
`vrc.unity_copy_word_range` and is one mode-1 Blit with
`Delta = source_offset / 4 - destination_offset / 4`: the surfaces may differ
in shape and may be bound at run time, and the destination is not read back.
The AOT `clEnqueueCopyBuffer` therefore also requires word-aligned offsets
and size, and rejects a copy between disjoint ranges of one physical buffer
with `CL_INVALID_OPERATION`, because a Blit cannot sample its own target.

A Host read of Device memory stays a binding command (`vrc.buffer_readback`),
but only a blocking read suspends the Host until it completes. After
publishing a non-blocking `clEnqueueReadBuffer` the AOT runtime calls
`__vrc_opencl_flush`, which `vrc-lower-opencl-aot-intrinsics` lowers to a
cooperative checkpoint (a `vrc.await` of a complete event marked
`vrc.cooperative_checkpoint`). Every checkpoint bind also sets the public
`__udoncl_async_pending_yield` slot (`pending_yield_slot` in the manifest's
`async_commands`). When the Udon event returns, the generated binding
submits the published readback, so it observes the buffer at its position in
the in-order queue, and for a checkpoint it resumes the Host on the next
frame without waiting for `VRCAsyncGPUReadback`. Later checkpoints (budget
yields included) keep resuming the Host while the readback is in flight; a
suspension that is not a checkpoint, such as the await behind
`clWaitForEvents` or the drain before the next transfer, waits for the
callback and receives its status. A readback that fails while nothing waits
for it fails the next suspension that does. The single command slot is
unchanged: one transfer is outstanding at a time.

A slice bounds one Udon event, not one frame. The generated U# binding
resumes a continuation in the same Udon event when it is ready at once: after
a Device work slice (command kind 0) and after a command that completes
synchronously (an upload, or a rejected command). Its `DriveAsyncCommands`
loop is the only caller of `PumpAsyncCommand` and the only place that
schedules `_ContinueAsyncCommand` for the next frame:

- The first entry in a frame (`Time.frameCount`) opens a window of
  `frame_microseconds` of wall-clock time
  (`Time.realtimeSinceStartupAsDouble`) and `frame_continuations`
  continuations. A ready continuation is delivered and pumped while both
  remain; otherwise it waits one frame. The limits are checked before every
  continuation, so a frame overruns the time budget by at most one slice, and
  the continuation limit bounds the Device work one frame enqueues to
  `frame_continuations + 1` slices even when the slices are cheap on the Host.
- A command that really waits (`VRCAsyncGPUReadback`) returns from the loop.
  Its callback re-enters the loop, which charges the same window.
- An export started from a completion event only requests a pump from the
  running loop, so pumps and Host completion events never nest however many
  exports complete in one frame.

Both limits are `device_work_budget` settings of the Unity library JSON
(`"frame_microseconds": N, "frame_continuations": N`, each 0 to 1,000,000,
defaults 4,000 and 8). They are recorded in the manifest as
`program.async_commands.frame_budget_microseconds` and
`frame_continuation_limit` and emitted as the binding constants
`AsyncFrameBudgetMicroseconds` and `AsyncFrameContinuationLimit`. Setting
either to zero, or a manifest without them, resumes one continuation per
frame.

While an async export is pending the binding rejects every synchronous
`Try*` export with `StatusBusy`. A library can declare a synchronous export
`"allowed_while_busy": true` in its export entry (recorded on the manifest
export, outside the public ABI fingerprint) when the export neither suspends
nor changes state the suspended export owns, for example a progress or
streamed-text getter. The generated method then runs between continuations
and restores the pending export's busy state; it is still refused while a
public buffer readback is in flight.

Persistent discovery creates requests such as:

```mlir
module attributes {
  vrc.async_persistent_request_version = 1,
  vrc.async_persistent_requests = [{
    member_value_ids = [...],
    priority = ...,
    legal_alternatives = ["direct"],
    ...
  }]
}
```

Host storage analysis turns these into placements with classification,
interference, and storage key. `LowerUdonAsyncRuntime` finally produces
`udonhl.async_enter`, `udonhl.async_resume_point`, event binding, and the
completion entry. Standard async/coroutine operations must no longer remain.

### 6. Host storage and UdonAsm

The named finalization pipeline in
`lib/Conversion/HostStoragePipeline/HostStoragePipeline.cpp` is:

```text
MaterializeFixedPhysicalReleases
ExposeHostBackendLiteralUses
DiscoverAsyncPersistentStorage
MaterializeAsyncPersistentStorage
DiscoverHostRematerialization
FitHostStorageCapacity
VerifyHostStorageCapacity
LowerUdonHLToUdonAsm
```

Backend literals are exposed first because persistent discovery records
positional value identities that later insertions would invalidate.

`FitHostStorageCapacity` transactionally tries literal allocation and host
storage materialization. Exact capacity is measured by lowering a clone of the
module.

Before, represented by `tests/codegen/udon.py` fixtures:

```mlir
func.func @add(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  return %sum : i32
}
```

After:

```mlir
module attributes {udonasm.stage = "scalar-v1", ...} {
  udonasm.push "__0_arg0__param"
  udonasm.push "__0_arg1__param"
  udonasm.extern
    "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"
  ...
}
```

Memref loads/stores similarly become Udon extern calls such as
`SystemSingleArray.__Get__...` and `SystemSingleArray.__Set__...`.

`lib/Target/UdonAssembly/Emitter.cpp` requires the final data/export/stage
metadata and emits `.data_start`, `.export`, `PUSH`, `POP`, `JUMP`, `EXTERN`,
and `COPY`.

### 7. Device lowering and shader emission

Normal device code uses `LowerDeviceSingleInvocation`; gather-profile code uses
`LowerGatherDevice`. Both produce the VRCGPU dialect:

```mlir
vrcgpu.kernel ... {
  %resource = vrcgpu.resource ...
  vrcgpu.pass ...
  %x = vrcgpu.uniform_load ...
  %packed = vrcgpu.raw_pack ...
  vrcgpu.output ...
}
```

`lib/Target/HLSL/Emitter.cpp` emits fragment HLSL and ShaderLab for each
`vrcgpu.kernel`.

#### Generated kernel channel

Library kernels are hand-written OpenCL C, configured by name and fixed id in
the bundle JSON and dispatched through intrinsics that
`LowerOpenCLAOTIntrinsics` knows by name. A kernel the compiler synthesizes
has none of these, so a Host MLIR module can carry it and dispatch it with
nothing keyed on its name:

```mlir
gpu.module @carrier {
  gpu.func @fused(%x: memref<256xf32>, %out: memref<131072xf32>, %n: i32)
      kernel attributes {vrc.synthesized_kernel, vrc.profile = "gather-f32",
                         known_grid_size = array<i32: 8192, 16, 1>,
                         known_block_size = array<i32: 1, 1, 1>,
                         vrc.words = 131072 : i32} { ... }
}
func.func private @__vrc_opencl_dispatch_fused(i32, i32, i32) -> i32
    attributes {vrc.synthesized_dispatch = @carrier::@fused}
```

- **Carrier.** A top-level `gpu.module` that holds only `gpu.func` kernels
  marked `vrc.synthesized_kernel` (the gather contract checked by
  `ProveSynthesizedGatherKernels`), without `vrc.kernel_id`. A dispatch is a
  call to an external declaration whose `vrc.synthesized_dispatch` names the
  kernel and whose signature is the kernel's argument list with an i32
  physical buffer token for every buffer, returning the i32 event token of
  the named dispatch intrinsics.
- **Id assignment.** `vrc-assign-synthesized-kernel-ids` gives every kernel
  with a called declaration the next free id in module order (`first-id`,
  skipping `reserved-ids` and ids already in the module), stamps
  `vrc.kernel_id`, isolates it as `gpu.module @device_gpu_<id>`, and erases
  uncalled declarations, undispatched kernels and the carriers.
- **Dispatch attribute.** The pass checks each declaration against its kernel
  and replaces the reference with
  `vrc.device_dispatch = {kernel_id, buffer_count, global_size}` plus the
  kernel's `vrc.fragment_cost` as `fragment_cost`. `LowerOpenCLAOTIntrinsics`
  lowers any declaration with that attribute to `vrc.dispatch_desc_create`
  and `vrc.dispatch` exactly as it lowers a named intrinsic: leading buffers
  read, the last buffer written, scalar kinds from the declared types, and
  `vrc.fragment_cost` on the dispatch for `ScheduleUdonDeviceWork`.
- **Proof.** `vrc-prove-synthesized-gather-kernels` must mark the kernel
  `vrc.gather_proven` before `LowerGatherDevice` accepts it; a failed proof
  fails the build.
- **Unity bundle.** `udonclc --emit-unity-library-bundle` runs id assignment
  and the proof right after it parses `--host-mlir`, with ids starting at
  4096 and the JSON `device_kernels` ids reserved. It moves each
  `@device_gpu_<id>` module out of the Host into the Device module next to
  the library kernels, so gather lowering, shader emission
  (`kernels/kernel_<id>.shader`), the manifest `device.kernels` entry, the
  material slot `__udoncl_kernel_material_<id>` and the Unity finalizer treat
  it like any other kernel. Its grid width and word count take part in the
  texture row-pitch and shape decisions, and the build fails if the texture
  shape the Unity buffers share cannot hold its output. A Host module that
  carries no synthesized kernel produces the same bundle as before.

### 8. Artifact and generated binding

The final artifact combines:

- `host.uasm`;
- fragment shaders and kernel metadata;
- device buffer layout;
- raw heap initializer;
- static tensor payloads;
- `manifest.json`;
- compiler-generated `binding.cs`;
- optional intermediate MLIR and diagnostics.

Artifact concepts are independently represented in C++, JSON construction,
CMake, Python, and C#. The authoritative model should become a versioned typed
artifact schema with shared golden fixtures. Compiler-generated bindings should
be authoritative; Unity-side regeneration should be a temporary,
schema-versioned compatibility path only.

## Test topology and what it protects

### Native and Python tests

`tests/CMakeLists.txt` is the current inventory and registers more than one
hundred tests. The major groups are:

| Group | Main locations | Contract |
|---|---|---|
| analysis | `tests/analysis` | liveness, allocation, interference |
| frontend | `tests/frontend` | individual passes, attributes, diagnostics |
| codegen | `tests/codegen` | UdonAsm and textual Udon output |
| runtime | `tests/runtime` | OpenCL/ONNX/stream/tokenizer host behavior |
| boundary | `tests/boundary` | compiler bundles and native GPU execution |
| build/package | `tests/build` | CMake consumers, ONNX tools, artifact schema |

Static resource lowering has positive and negative fixtures for almost every
pass. This is the strongest existing base for refactoring pass orchestration.

Several large tests mix orchestration and assertions:

- `tests/build/cmake_package.py`
- `tests/codegen/udon.py`
- `tests/boundary/onnx_runtime.py`
- `tests/build/onnx_cpp_lowering.py`

They should be split by stage contract, not merely by line count.

### Unity EditMode/U# tests

The package tests cover:

- manifest validation and transactional import;
- generated assets and bindings;
- SDK UdonVM execution;
- fragment dispatch and numerical comparison;
- upload/readback and async pumping;
- public U# controller APIs;
- MNIST/TinyLlama and BLAS/DNN/operator lifecycles;
- cleanup.

Repeated helpers should move to a test-support assembly:

```text
artifact locator
Udon VM harness
texture and assigned-buffer fixture
async command pump
readback/oracle comparison
generated asset cleanup tracker
```

### Commit validation service

`scripts/precheck_service.py` owns the SQLite-backed queue, exact commit/tree
identity, leases, worker heartbeats, isolated worktrees, logs, and result
fingerprints. `scripts/precommit.py` owns the build, CTest, fixture packaging,
isolated Unity MCP lifecycle, Unity tests, console validation, and cleanup.

Each job uses:

```text
w/<slot>/r   exact commit worktree
w/<slot>/b   build tree
w/<slot>/o   compiler/test output
w/<slot>/u   isolated Unity project
```

This external contract should remain stable while the service is internally
split into database, coordinator, lease, workspace, native gate, Unity gate,
and result persistence modules.

## Main refactor risks and seams

### P0: Pipeline order is hidden in `vrccl.cpp`

Extract named builders before changing pass behavior:

```text
buildUnityHostRecoveryPipeline
buildStaticDeviceResourceAnalysisPipeline
buildStaticUnityResourceMaterializationPipeline
buildUnityHostOptimizationPipeline
buildUnityAsyncPipeline
buildNativeDeviceRecoveryPipeline
```

`vrc-opt` and `udonclc` should call the same builders.

### P0: ONNX production ownership is ambiguous

Declare the C++ ONNX-dialect path authoritative. Restrict Python to model
download/cache, protobuf and weight inspection, fixture generation, and
packaging. Keep the legacy execution-plan implementation quarantined until
generic MNIST and real TinyLlama U# numerical coverage replaces it.

### P0: Library dependency direction is inverted

Create a real UdonATen device target and remove ATen kernels from UdonDNN's
device bundle. Centralize DeviceMemory/OpenCLAOT runtime ownership and make
ONNX provider dependencies explicit.

### P1: Important semantics live outside passes

The following helpers should become passes or explicit compiler services with
typed inputs/outputs:

- `attachDeviceResourceOperandContracts`;
- `propagateUnityHostAsyncEffects`;
- `bindUnityBufferOperations`;
- Unity OpenCL call normalization.

### P1: Cross-pass attributes are string protocols

Introduce shared typed parsers/verifiers, then dialect attributes or op
interfaces, for:

```text
resource/allocation context identity
resource provenance and inventory
kernel binding snapshot
Unity buffer layout
async frame/object/resume route
persistent storage request
host storage placement
```

### P1: Static tensors use a compiler side channel

Large payload attributes are removed before async cloning and retained in C++
state for artifact packing. Replace this with lightweight IR symbols/references
whose bytes live in an artifact store. A pipeline should be reproducible from
the MLIR module plus referenced artifact store.

### P1: Large implementation units

Approximate current sizes and internal seams:

| File | Size | Internal split |
|---|---:|---|
| `LowerScalar.cpp` | 6.1k lines | data layout, slots, scalar/cf/memref, externs, metadata |
| `FinalizeONNXHost.cpp` | 5.0k | graph, tensor plan, providers, session, sequence state, metadata |
| `PropagateDeviceResourceProvenance.cpp` | 4.1k | lattice, memory models, summaries, solver, receipt |
| `tools/vrccl/vrccl.cpp` | 4.1k | CLI, pipelines, semantics, artifact writing |
| `UdonclArtifactImporter.cs` | 3.7k | parser, transaction, resources, bindings, examples/UI |
| `runtime/onnx/udon_onnx.py` | 3.4k | legacy planning and model-specific generation |
| `OpenCLAOT.c` | 2.7k | runtime and operation-specific dispatch |

Split these internally first. Only create additional externally visible passes
after each new boundary has an invariant and a focused test.

## Staged refactor plan

### Gate 0: Freeze contracts

- Document and version the artifact schema.
- Add shared golden artifacts/bindings for Hello, one BLAS operation, one DNN
  operation, generic MNIST, and one TinyLlama component.
- Add pipeline-level characterization tests around the concrete stage
  boundaries described above.
- Record the authoritative ONNX path.
- Keep the exact-commit background validation gate unchanged.

Exit criterion: current output is reproducible and stage failures are local.

### Gate 1: Refactor tests without behavior changes

- Split `tests/CMakeLists.txt` into group files.
- Split mega Python tests by stage contract.
- Create Python tool/consumer/artifact assertion helpers.
- Create the Unity test-support assembly.

Exit criterion: same test inventory and semantics, with independently runnable
stage tests.

### Gate 2: Extract compiler pipeline libraries

- Move pipeline order from `vrccl.cpp` to named builders.
- Share builders between `vrc-opt`, `udoncl-opt`, and `udonclc` where valid.
- Separate CLI parsing, compiler request, host frontend, device frontend,
  static-resource pipeline, async pipeline, storage pipeline, and artifact
  writer.
- Move the ONNX pass order from Python into a named C++ pipeline.

Exit criterion: tools produce schema-equivalent artifacts through shared
pipeline builders.

### Gate 3: Repair build and library ownership

- Introduce explicit host/device/runtime/static-tensor/generated-MLIR module
  roles in CMake.
- Move ONNX import/lowering from configure-time side effects to explicit build
  outputs.
- Give UdonATen its own device target.
- Centralize OpenCLAOT/DeviceMemory composition.
- Preserve existing public CMake functions and BLAS/DNN C APIs as wrappers.

Exit criterion: dependency direction is explicit and all existing consumers
produce equivalent artifacts.

### Gate 4: Establish artifact and binding authority

- Introduce a versioned in-memory artifact model and serialization layer.
- Extract artifact construction/serialization from `vrccl.cpp`.
- Add C++ emitter/C# validator golden parity tests.
- Make compiler-generated U# bindings authoritative.

Exit criterion: Unity installs artifacts and bindings without reinterpreting
compiler ABI semantics.

### Gate 5: Decompose Unity import

- Split parse/validate, plan, transaction, program, shader/material,
  texture/static tensor, binding, and generic prefab services.
- Move MNIST/TinyLlama and operation demo construction into sample recipes.

Exit criterion: adding a generic artifact resource does not require editing a
central importer or model-specific UI code.

### Gate 6: Remove the legacy ONNX path

- Keep generic MNIST lifecycle, real TinyLlama components, conversation
  exports, and packaged examples on the C++ ONNX-dialect path.
- Delete execution-plan generation/materialization, model-specific Python
  providers, legacy Unity runtime/model assets, and MNIST-specific OpenCLAOT
  paths. The legacy path is no longer a supported compatibility surface.

Exit criterion: MNIST and real TinyLlama pass U# numerical tests without an
`execution-plan.json` runtime.

## Refactor safety rules

- Do not change MLIR pipeline behavior and artifact serialization in one step.
- Do not change compiler ABI and Unity binding interpretation in one step.
- Do not change allocator/storage lowering and Udon Assembly emission in one
  step.
- Do not add new behavior to the legacy ONNX path while it is being removed.
- Preserve CUDA-compatible UdonBLAS/UdonDNN public contracts.
- Preserve current public CMake functions during migration.
- Preserve Unity import transactions, domain-reload timing, and cleanup.
- Each gate and each intermediate commit must pass exact-commit background
  validation before the next behavior change.
- Keep one MLIR pass implementation per source file and give every pass focused
  lit/FileCheck coverage for its input, output, and diagnostics.
- Use `RewritePattern`/`OpRewritePattern` for local rewrites and dialect
  conversion infrastructure for legality-driven lowering. Reserve direct IR
  walking for analysis; do not mutate several unrelated operation kinds from a
  broad `walk()` callback.

## Recommended first implementation slice

The first implementation slice should be deliberately structural:

1. Add LLVM lit/FileCheck coverage for every MLIR pass boundary.
2. Split `FinalizeONNXHost` into graph validation, planning, global
   materialization, export materialization, runtime dispatch, metadata, and
   source-graph cleanup passes. Keep one pass implementation per file.
3. Move the resulting pass order into a named C++ pipeline.
4. Delete the legacy execution-plan pipeline and its tests.
5. Extract static-resource analysis/materialization and Unity host recovery
   into named pipeline builders without changing order.
6. Make `vrccl.cpp` call those builders.

This reduces hidden coupling and makes later ONNX, artifact, and Unity importer
work observable without immediately changing runtime behavior.
