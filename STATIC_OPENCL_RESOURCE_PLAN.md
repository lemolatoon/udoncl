# Static OpenCL Resource Specialization Plan

> Status: implementation in progress; Gates 1 and 2 are integrated at the
> runtime-free Host boundary. Repeated root-distinct allocation contexts now
> use an integrated call-path specialization pass; common-prefix invocation
> tries remain. Gate 3 has standalone slices for immutable
> call-by-value and typed `vrc.dispatch` snapshots. The typed path is integrated
> for production artifacts; generic `clSetKernelArg` recovery remains.
> Decision date: 2026-11-12
> Parent architecture: [PLAN.md](PLAN.md)
> Related execution plan:
> [TINYLLAMA_AOT_CONVERSATION_PLAN.md](TINYLLAMA_AOT_CONVERSATION_PLAN.md)

## 1. Objective

Compile the supported OpenCL Host AOT subset into direct Unity Texture
bindings. For whole-program-known Hosts, Udon Assembly must not allocate or
resolve `cl_mem` handles at runtime. It should retain only the real transfers,
Blits, dynamic scalar values, and asynchronous continuation state required by
the program.

The first milestone keeps one physical Texture per proven allocation context.
Texture lifetime coloring is a separate optimization and must not block removal
of runtime handle resolution.

## 2. Existing facts

- ONNX finalization already determines mutable buffer capacities and emits
  `udoncl.mutable_buffer_bytes` and `udoncl.mutable_handle_capacity`.
- CMake already creates a Host module before `OpenCLAOT.c` and
  `DeviceMemory.c` are linked.
- `HostBufferPlan` walks exported direct call graphs and finds constant-size
  `clCreateBuffer` calls outside loops, but records only a size sequence.
- The internal IR already has typed buffer allocation, static-buffer, upload,
  readback, copy, dispatch, release, and await operations.
- Constant logical resources can already receive direct Unity buffer IDs and
  heap slots. Scalar lowering uses direct bindings when available and falls
  back to runtime token-to-pool selection only when provenance is dynamic.
- Immutable static tensors are already fixed Texture resources.
- ONNX graph-local storage planning already performs capacity-aware last-use
  reuse. The missing scope is C Host allocations, public buffers, library
  workspaces, sub-buffer views, and asynchronous command lifetime.

## 3. Required semantics

The specialization must preserve C and OpenCL semantics.

- A copied `cl_mem` denotes the same resource, not a new buffer.
- `clSetKernelArg` copies the argument value. Every enqueue observes the
  reaching argument state at that enqueue, even when the same kernel is later
  reconfigured.
- Releasing a memory object does not end its physical lifetime while a queued
  command still uses it.
- A sub-buffer is a bounded view of its root allocation. It does not acquire an
  independent Texture.
- In-order queue dependencies may become direct continuation order, but actual
  upload, readback, dispatch, failure status, and suspend/resume work remain.
- Static resources are single-flight unless a distinct physical pool exists
  per concurrent instance. Busy and callback-generation state must prevent
  reentry and stale completion.
- Public buffers, immutable tensors, and persistent K/V state keep stable
  physical identity and are not opportunistically colored with private
  temporaries.
- A Texture used as both shader input and shader output in one dispatch is not
  silently aliased. The compiler must prove a supported transformation or
  reject the program.

## 4. Supported initial AOT subset

The first implementation accepts:

- direct, non-recursive call graphs rooted at `UDONCL_EXPORT` functions;
- one known in-order queue and its known context;
- positive compile-time allocation capacities and supported memory flags;
- null Host pointers for Device allocations;
- handles whose complete flow is visible to the compiler;
- constant, statically bounded sub-buffer views;
- statically known kernels, argument indices, argument sizes, and NDRange
  profiles;
- null output events or events used only for compiler-known ordering and await;
- one active invocation per generated binding instance.

The first implementation rejects with precise diagnostics:

- recursion or indirect calls in resource-producing paths;
- allocation in an unbounded or unproved loop;
- dynamic allocation capacity or unsupported Host-pointer initialization;
- handle escape through an unknown call, integer conversion, comparison,
  query API, public ABI, or unresolved memory alias;
- canonical `eq`/`ne` null checks are the sole comparison exception when the
  compiler proves a finite, non-null static allocation identity and folds the
  result; this does not permit dynamic handle comparison;
- multiple or out-of-order queues whose ordering cannot be proven;
- nested or unbounded dynamic sub-buffer views;
- incomplete or ambiguous kernel argument state at enqueue;
- unresolved same-dispatch read/write alias;
- explicit event identity, queries, retain/release, callbacks, or multiple
  independently live output events not reducible to compiler Async values;
- reentrant use without a separate instance pool or a proven busy guard.

Unsupported programs must fail compilation. They must not silently fall back
to a slower Udon runtime path in the static Unity artifact.

## 5. Analysis model

### 5.1 Reachable call graph and allocation contexts

Build a direct non-recursive call graph from exported roots. Identify an
allocation by a stable context such as:

```text
(export root, call context, allocation site)
```

The context distinguishes simultaneous objects created by repeated calls to
one helper. Synchronous helpers may use existing inlining and canonicalization.
May-suspend helpers remain separate and are summarized.

### 5.2 Resource provenance and escape

Use a finite OpenCL-specific abstract domain rather than a general C pointer
analysis:

```text
Unknown
Null
Resource(allocation identity, root Texture, byte offset, extent, flags)
Set<Resource>
```

Propagate it through SSA copies, arguments, results, PHIs, selects, supported
alloca/global loads and stores, static tensors, public bindings, and
sub-buffer creation. Unknown escape is an error in the initial subset.

### 5.3 Kernel argument snapshots

Track the copied state of every `(kernel identity, argument index)`. At each
enqueue, create an explicit immutable dispatch snapshot containing:

- kernel ID;
- resource/view operands;
- scalar operands at their copied values;
- access modes;
- NDRange;
- dependency and completion information.

This removes the runtime kernel argument byte table without changing
`clSetKernelArg` semantics.

### 5.4 Sub-buffer views

Normalize every supported sub-buffer to:

```text
root allocation + byte offset + logical extent + access flags
```

Constant offsets and extents fold completely. A later extension may retain a
bounded dynamic offset as a scalar while keeping the root Texture static.

### 5.5 Command and event dependencies

Build a command graph from in-order enqueue order, event wait lists,
`clFinish`, blocking transfers, and compiler awaits. Resource lifetime extends
through the completion of the last consuming command.

Compiler-known events become typed Async dependencies. Generic event tables are
not retained merely to sequence one known in-order command stream. Resume PC,
completion status, busy state, and callback generation remain where required.

### 5.6 Physical Texture allocation

Milestone one assigns a unique Texture to every proven allocation context. It
does not require lifetime coloring.

Milestone two builds interference from command lifetime, capacity, format,
root/view aliasing, public/persistent identity, and read/write access. Private
resources may share a Texture only when their extended lifetimes do not
interfere and all physical constraints match.

## 6. Pass and component boundaries

Keep each responsibility independently testable.

1. `DiscoverStaticDeviceResources`
   - builds reachable call contexts;
   - assigns stable allocation identities;
   - rejects recursion, indirect allocation paths, and unproved loop
     allocation.
2. `PropagateDeviceResourceProvenance`
   - resolves aliases, views, joins, and escapes;
   - annotates every supported resource use with a finite identity set.
3. `ResolveStaticKernelBindings`
   - computes reaching copied kernel arguments;
   - replaces mutable kernel state with immutable enqueue snapshots.
4. `BuildOpenCLCommandGraph`
   - represents queue order, waits, completion, and resource retention without
     scheduling policy.
5. `AnalyzeDeviceResourceLifetimes`
   - computes command-completion-extended liveness and interference;
   - does not allocate physical slots.
6. `AllocateUnityTextures`
   - first implements one allocation context per Texture;
   - later adds capacity-aware coloring without changing earlier analyses.
7. `LowerStaticDeviceResources`
   - converts allocation, release, kernel state, transfer, copy, and dispatch
     into direct typed resource operations.
8. `VerifyStaticDeviceResources`
   - rejects unresolved handles, pool selection, invalid bounds, alias hazards,
     and inconsistent public or async lifetime contracts.

Analysis data must use explicit typed structures or IR attributes owned by the
corresponding pass. Do not communicate through positional assumptions in a
JSON size list. Serialization is allowed for build-boundary artifacts only
when it carries stable allocation identities and a versioned schema.

## 7. Pipeline placement

Use the runtime-free whole Host module as the source of truth:

```text
C/LLVM Host and linked libraries, without OpenCLAOT runtime
  -> SROA / mem2reg / SCCP / canonicalize
  -> resource discovery and provenance
  -> kernel snapshot recovery
  -> command dependency recovery
  -> physical Texture plan
  -> static OpenCL lowering and verification
  -> existing Host-to-Async lowering
  -> existing Unity binding
  -> UdonHL
  -> UdonAsm
```

The resource passes run after whole-program simplification and before Async
call-graph conversion. Native execution may continue using the ordinary
OpenCL runtime. Unity static artifacts must link only the residual helpers
whose semantics could not and should not be erased, such as status and
continuation support.

## 8. Implementation and commit gates

Every item is a separate tested commit. Do not combine a failing later gate
with an already passing earlier component.

### Gate 1: discovery analysis

Status: complete and enforced for split runtime-free Unity Host builds.

- Add a standalone allocation-context analysis/pass.
- Test direct calls, repeated helper calls, branches, recursion diagnostics,
  indirect calls, and loop allocation diagnostics.
- Do not change generated artifacts or runtime behavior.

### Gate 2: provenance analysis

Status: complete and integrated. DeviceMemory adopt, view, resolve, local/global
cell, aggregate pointer, output-cell, call-context, and nullable-handle paths
have focused coverage. The production pipeline emits the versioned provenance
schema into real MNIST and TinyLlama artifacts. A separate metadata-only
context-specialization planner now groups discovery contexts by allocation
site and records which internal call paths require cloning before static
lowering. It validates direct paths and dense resource identities without
changing the call graph. A separate standalone transformation now consumes
that plan for root-distinct paths, clones only private defined callees, rewrites
the selected call chain, and removes inherited LLVM symbol metadata. Re-running
discovery and planning after the transformation proves the formerly shared
contexts are singleton. Common-prefix paths are diagnosed until the generic
invocation-trie transformation is implemented. Current production acceptance
records one MNIST clone followed by five singleton sites; TinyLlama and
ordinary UdonBLAS remain zero-clone singleton cases. The production order is
`Discover -> Plan -> Specialize -> Discover -> Plan` before requirements and
provenance analysis, so all downstream metadata uses the specialized graph.

- Track handles through arguments, results, PHIs/selects, and supported memory
  promotion.
- Test alias copies, branch joins, escaped handles, and ambiguous identities.
- Do not allocate Textures.

### Gate 3: kernel snapshots

Status: standalone slices implemented for structured call-by-value Device
dispatch contracts and surviving typed `vrc.dispatch` operations. They record
versioned immutable allocation/static-tensor identities, access modes, copied
scalar positions, and kernel identity. Typed dispatch recovery proves the
successful `udonclResolvePhysicalBuffer` path that produced each physical
token. The typed path is integrated after OpenCL intrinsic recovery and before
the existing Host-to-Async pipeline. Dispatches with structurally valid but
currently identity-less public/session resources remain on the existing
dynamic path as whole dispatches; the summary records candidate, resolved, and
unresolved counts. Existing typed `vrc.buffer_alloc` operations use a distinct
`logical_buffer` identity namespace, and physical-token helpers are traced
through direct caller provenance. Recovering persistent identities across
exports and generic `clSetKernelArg` copied byte-state recovery remain separate
work.

- Resolve kernel identity and copied argument state at every enqueue.
- Test reconfiguration between two enqueues, missing arguments, branches, and
  buffer/scalar mixtures.
- Do not change async scheduling.

### Gate 4: one-to-one Texture assignment

Status: standalone allocator implemented against a versioned keyed inventory.
It assigns deterministic one-based Texture IDs to allocation contexts while
preserving capacity, format, public access, and persistence. The producer that
derives this inventory is split by responsibility: a standalone requirements
pass now reconstructs keyed capacity, memory flags, format, and usage through
discovery call paths. A standalone ONNX public-resource pass now joins generic
allocation provenance with stable input/output bind call sites and typed ONNX
metadata; it does not classify by model name, capacity, or fixed resource ID.
A separate generic persistence pass now classifies identities that cross an
export root or escape through global storage, independently of ONNX metadata.
A standalone keyed merger now combines requirements, persistence, and optional
ONNX public bindings into the allocator inventory; ordinary UdonCL modules do
not require ONNX metadata. Removal of the current duplicate public-buffer
capacities and manifest integration remain separate work.

- Assign one physical Texture per allocation context.
- Emit stable resource IDs, capacities, formats, and public/persistent flags.
- Test deterministic output and manifest/importer compatibility.
- Do not perform lifetime coloring.

### Gate 5: static lowering

Status: the first two standalone slices are implemented. A mutable
`vrc.assigned_buffer` reference is distinct from immutable static tensors and
dynamic allocation. Resolved typed dispatch operands are joined by
`(resource_kind, resource_id)` to the keyed Texture assignment and replaced
without relying on logical-buffer ordinals. Snapshot-less and logical-buffer
operands remain explicitly counted. Production integration waits for the
manifest/pool to use the same assignment identities; this avoids binding
TinyLlama allocation contexts to the legacy public-buffer ordinals.

Runtime-free `clEnqueueWriteBuffer`, `clEnqueueReadBuffer`, and
`clEnqueueCopyBuffer` call sites now have a separate generic transfer-binding
analysis. It records read/write roles and allocation identities while
preserving `(export root, call path)` cases. A call whose resource identity
changes by call context is marked for later context specialization instead of
being flattened. Unknown and public logical resources remain explicitly
unresolved. The analysis is integrated immediately after runtime-free
provenance propagation. This slice remains metadata-only: it does not lower
transfers, alter queue ordering, or map legacy logical-buffer ordinals to
allocation IDs.

A public-endpoint alias slice now joins generic
`(public_access, public_index)` identities from the static inventory to the
same keyed Texture assignments. It emits only proven
`public endpoint -> allocation -> texture_id` mappings and never creates an
additional Texture. Private allocations are omitted. Public endpoints that
have not reached the allocation inventory remain outside this mapping and
continue to require the legacy pool until their provenance is resolved. The
slice is integrated immediately after one-to-one Texture assignment and is
metadata-only.

A standalone Unity-buffer reconciliation pass now exact-joins configured
public endpoints to those aliases by `(public_access, public_index)`. The tool
adapter serializes the final configured buffer list into a versioned MLIR
attribute after Host-buffer planning, then the pass classifies each entry as a
resolved static alias, unresolved legacy public buffer, or unresolved legacy
private buffer. It never infers identity from a name, capacity, numeric buffer
ID, or list position. Static-metadata-free UdonCL artifacts remain unchanged;
incomplete metadata and identity/capacity conflicts are diagnosed. Real
artifact acceptance currently records MNIST as 2 resolved plus 5 legacy
private buffers, TinyLlama as 6 unresolved public plus 4 legacy private
buffers, and ordinary UdonBLAS as 3 unresolved public buffers. This slice is
still metadata-only.

A standalone canonical Unity-buffer layout planner now consumes the keyed
inventory, Texture assignments, public aliases, and reconciliation result. It
does not assume that a Texture ID equals a legacy buffer ID. Instead, it emits
an independently versioned resource-to-Texture-to-heap-slot layout and a
readiness decision. Each public planned entry is self-contained and carries
its semantic name, data type, and logical byte count in addition to the
physical capacity and public endpoint identity. Real MNIST is ready with 5
planned buffers replacing 7
legacy entries, so 2 duplicate legacy entries are removable. TinyLlama is
blocked by 6 unresolved public endpoints and ordinary UdonBLAS by 3; neither
silently replaces its pool. Manifest and pool replacement remain a separate
tool-adapter commit gated by this plan and by direct static resource lowering;
the existing runtime handle ordinal is not treated as a Texture identity.

The Host validator, Unity dispatch binder, and Udon Assembly lowering now
accept `vrc.assigned_buffer` as a mutable compiler-assigned resource. The
binder exact-matches its Texture ID and capacity to the selected Unity buffer,
records the resolved heap slot, and lowering emits a constant physical pool
token. Runtime read/write alias validation still observes that token. This
downstream support is deliberately independent from production
materialization: the canonical pool is not selected until dispatch and
transfer completeness have both been proven.

- Replace supported allocation/release and buffer arguments with direct typed
  resources.
- Remove runtime handle lookup, buffer free-list use, and kernel buffer-argument
  tables from the reachable Unity Host.
- Keep actual transfers, Blits, scalar state, status, and continuation state.

### Gate 6: MNIST numerical acceptance

- Use five mutable Textures for the five discovered allocation contexts; the
  current seven-entry list duplicates the public input/output capacities and is
  not the assignment contract.
- Pass Native, Device, direct SDK VM, and generated-U# numerical tests.
- Assert generated IR/UASM has no reachable dynamic buffer table, free-list,
  `clCreateBuffer`, `clReleaseMemObject`, `udonclResolvePhysicalBuffer`, or
  dynamic buffer-ID switch.
- Record UASM size, instruction counts, first-callback time, and total runtime.

### Gate 7: TinyLlama focused prefill

- Run the real focused generated-U# prefill against the ORT oracle.
- Compare all logits and present K/V outputs.
- Record watchdog result and the same resource/performance metrics as MNIST.
- Do not introduce a TinyLlama-specific resource scheduler or runtime.

### Gate 8: command lifetime analysis

- Extend liveness through queued-command completion and callback resume.
- Test release-before-completion, branch-exclusive lifetimes, sequential reuse,
  sub-buffer root lifetime, and finite loop pipelines.
- Keep one-to-one allocation available as a diagnostic baseline.

### Gate 9: private Texture coloring

- Reuse compatible private Textures by interference coloring.
- Keep public, immutable, K/V, and otherwise persistent resources pinned.
- Test deterministic coloring, capacity choice, format separation, and
  same-dispatch read/write non-aliasing.
- Re-run MNIST and TinyLlama numerical and performance gates.

## 9. Verification requirements

Pass-level tests must prove both accepted and rejected programs. End-to-end
tests must verify numerical behavior, not only generated text or diagnostics.

For a successfully specialized Unity artifact, generated IR/UASM must contain
none of the following in its reachable Host path:

- runtime `_cl_mem` handle-table access;
- buffer allocation/free-list traversal;
- `udonclResolvePhysicalBuffer`;
- dynamic logical-buffer-to-pool selection;
- runtime kernel buffer-argument byte tables;
- reachable `clCreateBuffer` or `clReleaseMemObject` implementations;
- a generic event table used only to serialize the known in-order queue.

The artifact may and normally will retain:

- direct Texture array access at compile-time constant indices when required to
  conserve Udon Heap addresses;
- `VRCGraphics.Blit` and real upload/readback operations;
- dynamic scalar kernel arguments;
- resume PC, completion status, busy state, and callback generation;
- explicit checks that remain observable at the supported public boundary.

## 10. Non-goals

- Do not change public OpenCL, UdonBLAS, UdonDNN, or ONNX Runtime semantics.
- Do not require a runtime scheduler for the known in-order command stream.
- Do not make U# responsible for Device allocation or buffer-ID selection.
- Incremental analysis gates may leave explicitly counted unresolved operations
  on the existing path. Gate 5 verification must reject that fallback in a
  completed static Unity artifact rather than silently accepting programs
  outside the documented static subset.
- Do not couple resource provenance, command lifetime, physical coloring, and
  scalar Host storage allocation into one pass.
- Do not specialize the design around MNIST or TinyLlama. They are numerical
  acceptance workloads for the generic mechanism.
