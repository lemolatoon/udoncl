# C Host Async / Low-Level U# Binding Plan

Status: approved implementation plan (2026-11-04)

> **Removed (2026-11):** the Unity library config file (`unity-library.json`,
> schema 1 and 2, `export_overrides`, `internal_exports`, `device_kernel(s)`,
> `buffers`) no longer exists. The Host source declares the program,
> exports and public buffers (`UDONCL_PROGRAM`, `UDONCL_EXPORT`,
> `UDONCL_EXPORT_WHILE_BUSY`, `UDONCL_PUBLIC(type)`), and
> `udoncl_add_unity_library` arguments set the rest; see README.md.
> The config text below is history.

This document is the source of truth for suspendable C host exports and their
low-level U# bindings. It replaces the earlier plan that made a generated
high-level ONNX Session wrapper the first implementation target. `PLAN.md`
contains the repository-wide roadmap; conflicting stale text must be removed,
not combined with this design.

## 1. Goal

Compile an ordinary C host program to Native or Udon while preserving C and
OpenCL semantics across blocking operations:

```text
C host + UDONCL_EXPORT
  -> LLVM IR
  -> high-level MLIR + vrc OpenCL/event operations
  -> MLIR Async dialect
  -> upstream async-func-to-async-runtime
  -> Native: LLVM coroutine + MLIR async runtime
     Udon: static heap frames + event-driven resume trampoline
  -> physical Udon export adapter
  -> generated low-level typed U# binding
```

The user owns the public host API. A model may expose `Initialize`, `Write`,
`Run`, `Read`, one combined `Infer`, or another set of C exports. The compiler
does not impose an ONNX Session-shaped U# API.

MNIST is the first ONNX acceptance case. Its host is ordinary C using OpenCL,
UdonBLAS, UdonDNN, UdonATen, and UdonONNXOperators APIs. ONNX graph lowering
after import remains entirely in C++ MLIR passes.

## 2. Scope

### 2.1 Required now

- A public `UDONCL_EXPORT` C annotation retained in linked Host LLVM IR.
- Schema-v2 Unity-library configuration for package policy and optional export
  naming overrides.
- Fixed-point may-suspend call-graph analysis in a C++ MLIR pass.
- Conversion of source-specific event operations to MLIR Async dialect.
- Upstream `async-func-to-async-runtime` as the coroutine transformation.
- Native lowering through LLVM coroutine and the standard MLIR async runtime.
- Udon lowering through static frame slots and a generated resume trampoline.
- Physical Udon entry adapters for C exports.
- Compiler-emitted manifest and low-level typed `binding.cs`, both derived from
  the same MLIR export ABI.
- CMake artifact generation and Unity import of compiler-produced bindings.
- Native, compiler, and Unity tests, including MNIST through generated U#.

### 2.2 Deferred

- A generated high-level ONNX Session U# wrapper.
- Multiple simultaneously active root coroutines per UdonBehaviour instance.
- Multiple pending waiters for one async object.
- `async.execute`, CPU task scheduling, worker pools, time slicing, or polling.
- Recursive may-suspend functions.
- Dynamic coroutine-frame allocation.
- C++ overload/name-mangling export support.
- CMake preset UI.

## 3. C export contract

### 3.1 Annotation

The public header defines a Clang-backed macro usable from C11 and from an
`extern "C"` declaration:

```c
UDONCL_EXPORT
cl_int Run(cl_mem input, cl_mem output);
```

The annotation is the only export-discovery source. A definition must be
present in the final linked Host module. Device functions, declarations without
definitions, indirect exports, C++ overloads, and private helpers are rejected.

The compiler derives whether an export may suspend. Users never write a
`may_suspend` annotation.

### 3.2 Schema-v2 configuration

> **Removed (2026-11):** the Unity library config file (`unity-library.json`,
> schema 1 and 2, `export_overrides`, `internal_exports`, `device_kernel(s)`,
> `buffers`) no longer exists. The Host source declares the program,
> exports and public buffers (`UDONCL_PROGRAM`, `UDONCL_EXPORT`,
> `UDONCL_EXPORT_WHILE_BUSY`, `UDONCL_PUBLIC(type)`), and
> `udoncl_add_unity_library` arguments set the rest; see README.md.
> The config text below is history.

`unity-library.json` no longer duplicates the export list. It may override
metadata for an annotated export:

- public name;
- stable parameter display names;
- generated binding namespace and class name;
- callback contract explicitly designed by the C host;
- artifact/package policy that cannot be derived from code.

Without an override, the public name is the C symbol and parameters use stable
positional names such as `arg0`. Export IDs are generated deterministically
from the public signature. Hash collisions are compile errors.

Config entries naming an unannotated or missing function, duplicate public
names, incompatible signatures, invalid C# identifiers, or generated method
collisions are compile errors.

### 3.3 C/U# type boundary

- Directly representable scalar C ABI types map to concrete U# scalar types.
- Host `T *` parameters map to supported U# arrays at the physical boundary;
  the C function retains pointer semantics internally.
- Array length is an explicit C argument. It is not guessed from a pointer.
- `const` describes host-memory mutation, not a device-copy direction.
- `cl_mem` and other device handles remain opaque handles and never become U#
  arrays.
- Device allocation and host/device transfer remain explicit OpenCL operations.
- A nonblocking transfer does not copy merely to make lifetime management
  easier. The programmer preserves host memory until its event completes.
- Unsupported export types fail compilation with the export name and type.

The initial required U# surface is `int`, `float`, `int[]`, `float[]`, and the
types needed for UdonBehaviour/custom-event callbacks. Additional types become
supported only after their physical ABI lowering exists.

### 3.4 OpenCL ownership at the ONNX boundary

ONNX is a generic OpenCL client, not the owner of an application-wide device
lifecycle. The ordinary C Host creates and owns its `cl_context` and
`cl_command_queue`, then passes both to the generated physical ONNX binding.
The binding borrows these handles until its dispose operation returns: it does
not select a platform/device, retain or release the handles, or substitute an
ONNX-specific transfer API. The queue must belong to the supplied context.

Generated code may allocate and own model-derived default input/output buffers
on the borrowed context and enumerate/bind them from compiler-owned metadata.
The C Host performs transfers and Device-to-Device copies with normal OpenCL
operations. Model-independent generated resolve functions return a borrowed
`cl_mem` together with its byte offset and logical byte length; returning a
bare `cl_mem` without the view range is forbidden. `CL_MEM_READ_ONLY` and
`CL_MEM_WRITE_ONLY` describe kernel access;
they do not prohibit the corresponding Host transfer. Any externally allocated
buffer remains owned by the C Host. Disposal order is ONNX session/generated
buffers first, followed by the Host-owned queue and context.

The physical create/bind functions are internal cross-library exports. Opaque
OpenCL handles are not exposed as public U# parameters. A higher-level C Host
chooses its public exports and the compiler generates U# bindings for those
exports only.

## 4. Async semantics

### 4.1 Source behavior

OpenCL source semantics are unchanged:

- nonblocking enqueue returns an event and does not suspend;
- blocking read/write, `clWaitForEvents`, and `clFinish` may suspend;
- an ordinary C call waits for its callee to return;
- normal runtime failures remain `cl_int` values;
- only compiler/runtime infrastructure failure is an Async error state.

When an ordinary C call targets a may-suspend function, conversion emits an
`async.call` followed by the corresponding `async.await`. Optimization may move
or erase this only when it proves the C-visible behavior unchanged.

### 4.2 Shared MLIR pipeline

The required order is:

1. import linked Host LLVM IR;
2. recover `func`, `cf`, `scf`, memory, and scalar operations;
3. lower OpenCL AOT intrinsics to source-specific `vrc` event operations;
4. infer may-suspend functions to a call-graph fixed point;
5. convert affected functions, calls, returns, and waits to Async dialect;
6. canonicalize and lower SCF to CFG where required;
7. run upstream `async-func-to-async-runtime`;
8. split into Native and Udon backend legalization.

Functions are not forcibly inlined. The upstream pass keeps parent and child
coroutines distinct and registers the parent continuation on the child's async
result.

### 4.3 Native backend

Native uses the standard Async-to-LLVM/LLVM coroutine path. External OpenCL or
test-event completion makes the corresponding async value available and
resumes the LLVM coroutine. Tests must not replace asynchronous behavior with a
blocking implementation.

### 4.4 Udon backend

Udon implements no general scheduler. It has only an event-driven resume
trampoline:

1. save the current continuation and live values;
2. return from the Udon event while the external operation is pending;
3. completion marks one async object ready;
4. a generated switch resumes its one continuation;
5. completed children select their parent continuation until the root finishes
   or another external wait suspends.

Step 5 is a one-element trampoline, not nesting. Udon functions return through
static per-function return slots, so a completing child must not enter its
parent while it is itself still waiting to return: once the resumed parent
calls the same child again, the outer return slot is overwritten and the event
never returns. A completing coroutine records its single ready waiter
(`__udoncl_async_ready_frame/pc`) and returns; the completion event drains
that slot after every entry, so each continuation runs at dispatcher depth
one. The single-waiter contract guarantees at most one ready continuation
between drains. Udon scalar lowering treats `async_enter` as an activation
edge like a call when coloring return-slot banks and rejects any activation
cycle at compile time.

There is no worker pool, task fairness, polling scan, time slicing, or generic
ready queue. `clWaitForEvents` validates and awaits its event list in order;
the ordinary C caller does not return until the final dependency completes,
and the first failed event status is returned without changing C semantics.

Udon initially accepts only structured Async IR produced from ordinary C:

- one active root export per component instance;
- one pending waiter per async value/token;
- no recursive suspending call graph;
- no escaping coroutine handles/results;
- no parallel async function calls;
- no indirect async calls;
- no `async.execute`.

Unsupported shapes produce precise diagnostics. GPU command/event concurrency
is still allowed; these restrictions concern Host coroutines.

### 4.5 Frame storage and ownership

Suspension cannot retain the Udon VM call stack. Values live across a suspend
are therefore stored in compiler-private Udon Heap slots. This is not a public
heap ABI and is not a runtime stack allocator.

- Each may-suspend function receives a statically planned logical frame.
- Only live-across-suspend values are persisted.
- Constants and recomputable metadata are removed before spilling.
- Slot coloring reuses physical heap slots when lifetimes cannot overlap.
- Known caller continuations and function IDs are constant-folded.
- Runtime reference counting emitted by the generic Async lowering is resolved
  statically; unprovable ownership is rejected initially.
- Already-ready awaits continue synchronously without event re-entry.

Frame slots, resume PCs, async-object IDs, and internal resume events remain
compiler-private and may change whenever the public ABI fingerprint changes.

## 5. Export entry adapters

Each annotated C export has a generated physical Udon entry adapter. It:

1. rejects a new root call while the instance is busy;
2. copies scalar public argument slots into C argument storage;
3. binds supported host-array references;
4. initializes the root coroutine frame;
5. hot-starts the C export;
6. returns when suspended;
7. accepts compiler-private resume events;
8. writes the final C result and `cl_int` status to public slots;
9. publishes completion/error state and clears busy.

Internal resume entries are not public exports. Every suspending low-level U#
entry accepts a receiver and completion-event name because U# cannot block;
this callback-first transport is common UdonCL infrastructure, not an ONNX
request-ID, Session, or model-specific callback API.

## 6. Generated low-level U# binding

The compiler emits a typed low-level binding for the C exports. This is not a
high-level model wrapper.

For a suspending export it exposes an API shaped like:

```csharp
int StartRun(..., UdonSharpBehaviour receiver, string completionEvent);
int GetRunStatus();
int TryGetRunResult(out TResult result);
```

For a non-suspending export it may expose a synchronous `TryX` form. Generated
names are validated as a set, including primary methods, status accessors,
result accessors, parameters, and reserved names.

Internally the binding holds the raw `UdonBehaviour` and uses the supported
transport:

```text
SetProgramVariable -> SendCustomEvent -> GetProgramVariable
```

Physical event and slot names are private generated constants. Performance-
critical sequences can be combined into one C export to avoid repeated
cross-behaviour calls.

The same generated class owns the compiler-defined command pump. It translates
pending upload/readback objects to `VRCGraphics.Blit` and
`VRCAsyncGPUReadback`, writes completion object/status slots, sends the private
completion event, and continues the static coroutine trampoline.

## 7. Artifacts and ABI

`udonclc --emit-unity-library-bundle` emits a self-contained directory with:

- `host.uasm`;
- device shaders and resources;
- `manifest.json`;
- compiler-generated `binding.cs`;
- required raw heap/weight artifacts;
- diagnostics and optional IR dumps.

The Unity importer validates and installs these artifacts. It does not infer or
regenerate binding source. It may create Unity assets, compile/import the C#,
connect serialized references, and preserve GUIDs transactionally.

Schema v2 fingerprints include every public or physical binding input:

- export ID, public name, order, parameter names and types;
- result type and may-suspend effect;
- callback contract;
- each generated public Device buffer's ID, name, logical byte length, element
  type, and input/output access;
- binding namespace/class name;
- physical ABI and runtime ABI versions.

The importer reconstructs the canonical v2 public ABI from typed exports and
binding metadata, recomputes its SHA-256 fingerprint, and also validates the
exact supported physical ABI, runtime ABI, compatible SDK, binding path/class,
target, and artifact readiness. Schema v2 is the public source format; an
explicit schema-v1 `binding` block is temporarily accepted only to migrate
existing examples.

## 8. CMake and user workflow

Users write ordinary multi-file CMake projects:

```cmake
find_package(UdonCL CONFIG REQUIRED)

udoncl_add_host_library(my_host SOURCES host.c helpers.c)
udoncl_add_unity_library(my_artifact
  HOST_LIBRARIES my_host)
```

The generated artifact directory is sufficient for Unity import; the importer
does not need the original source or build tree. Preset support remains
deferred.

## 9. ONNX integration

ONNX import and graph lowering remain generic:

```text
ONNX model
  -> ONNX MLIR dialect
  -> C++ canonicalization and legality checking
  -> scf/cf/memref/arith + func.call to UdonONNXOperators/UdonBLAS/
     UdonDNN/UdonATen/OpenCL
  -> common C Host/Async/export artifact pipeline
```

Python may download/cache test models and invoke tools. It must not lower the
imported graph or generate a privileged model execution plan.

The first acceptance host exposes annotated C exports such as `Initialize`,
`WriteInput`, `Run`, and `ReadOutput`. A combined `Infer` export may be added to
measure reduced U# cross-behaviour overhead, but it is not a required runtime
interface. High-level ONNX Session U# generation is deferred.

## 10. Implementation order

1. Add `UDONCL_EXPORT` and schema-v2 parsing/validation.
2. Add C++ VRC/OpenCL-to-Async call-graph conversion.
3. Register upstream Async dialect and `async-func-to-async-runtime`.
4. Connect Native execution to LLVM coroutine/runtime.
5. Lower the supported Async Runtime/Coro subset to Udon frames/trampoline.
6. Generate physical export adapters from MLIR.
7. Generate manifest and low-level U# binding from the same export ABI.
8. Emit `binding.cs` in the CMake artifact and simplify the Unity importer.
9. Migrate the MNIST C host and generated-U# acceptance test.
10. Remove the superseded direct await lowering and Editor binding generator.

### 10.1 Implementation checkpoint (2026-11-05)

Completed and verified:

- steps 1-3: `UDONCL_EXPORT`, schema-v2 discovery/overrides, fixed-point
  source-to-Async conversion, and upstream `async-func-to-async-runtime`;
- step 4 now has an executable Native path. The
  `vrc-lower-host-async-to-native` pass maps the common VRC completion event to
  an `async.value<i32>` without changing suspension semantics. The C Host
  parent/child fixture is lowered through upstream Async Runtime, Async-to-LLVM,
  LLVM coroutine split/cleanup and `lli`; completing the external value with 7
  resumes the suspended call graph and produces 42. The test also requires the
  generated coroutine suspend/resume functions rather than accepting a
  blocking rewrite. The same executable now forces two sequential suspension
  cycles by waiting for an in-coroutine checkpoint before completing the
  second event, and separately verifies an event completed before the call;
- the first half of step 5: a Udon Async preparation pass verifies the exact
  supported Runtime/Coro operation set, rejects recursion, `async.execute`,
  indirect may-suspend calls, and unsupported spill types, and assigns stable
  frame IDs and resume PCs;
- static frame planning now computes live-across-suspend scalar/pointer/event
  values, omits recomputable constants/global addresses, colors
  non-overlapping values onto reusable typed slots, and separately plans
  async-object slots. The plan is materialized as compiler-private
  `udonhl.global` storage for each frame PC, typed spill slot, and async-object
  state/waiter/payload slot, with symbol references attached to the plan.
  Each suspend now stores its resume PC and spills its live values, and each
  resume block reloads those values; handle-shaped VRC values retain their
  logical type while using the physical 32-bit frame representation. Reload
  selection is dominance-based, so structured branch CFGs are not forced into
  a linear resume chain, and fixed-width integer spills include `i64`;
- direct call/return edges now participate in an interprocedural async-object
  identity analysis: the caller call result and the object returned by the
  callee share one static state/waiter/payload allocation. Pending reset,
  waiter frame/PC registration, payload store/load, and available/error state
  writes are materialized against that canonical allocation; upstream
  `async.runtime.store/load`, `set_available`, and `set_error` operations are
  removed. The state encoding is fixed in module metadata as pending `0`,
  available `1`, error `-1`;
- await lowering now loads canonical state and branches directly to the resume
  block when the object is already available or failed; only pending state
  enters `async.coro.suspend`. `async.runtime.await_and_resume` and
  `async.runtime.is_error` are eliminated in favor of UdonHL loads and integer
  comparisons;
- completion now loads and clears the canonical waiter frame/PC and emits one
  `udonhl.async_resume(frame_id, resume_pc)`. Frame zero is defined as a no-op;
  the operation is a direct trampoline request and explicitly carries no
  scheduler, queue, polling, or dynamic-allocation semantics;
- a final Udon Async pass now removes every remaining upstream
  `async.runtime.*` and `async.coro.*` operation, converts coroutine functions
  and calls to their void control ABI, lowers event objects to
  `udonhl.async_bind_event`, and emits a deterministic static
  `(frame_id, resume_pc)` route table. Unity artifacts retain the prepared
  `host.async-runtime.mlir` diagnostic and also emit the fully legalized
  `host.udon-async.mlir` diagnostic;
- final lowering collects every dynamic resume request into one private
  `__udoncl_async_resume_dispatch` function. Its finite route table ends in
  `udonhl.async_enter` operations, which scalar UdonASM lowering turns into
  direct entries to marked coroutine resume blocks. The existing per-function
  static return slot restores internal callers; a resumed root terminates the
  Udon event. Frame zero and unknown routes are no-ops, and the route table is
  not duplicated at each request site;
- `udonhl.async_bind_event` now writes the canonical object ID to the public
  pending-object slot. Final lowering separately exposes completion-object and
  completion-status slots and generates compiler-private
  `__udoncl_async_complete(object_id, status)`. That adapter stores the event
  payload/state, clears its waiter and invokes the shared resume dispatcher.
  The compiler-generated binding consumes the pending object, performs
  upload/readback or a delayed logical completion, writes the two completion
  slots, and sends this compiler-private event. The Editor-side generator keeps
  equivalent logic only as a temporary compatibility path;
- scalar UdonASM lowering now performs call-site-aware interprocedural slot
  coloring. Caller values interfere with a callee or transitive descendant only
  when they are live after that call, while mutually exclusive internal values
  share typed storage. Fixed-size array extent is part of the storage key, and
  recursive call graphs fail instead of aliasing slots. Return addresses are
  materialized into the callee's existing `SystemUInt32` return slot with a
  fixed-length instruction sequence rather than consuming one Udon Heap address
  per call site. Deterministic single-use `i32`/index constants are also
  materialized directly into colored slots instead of immutable Heap entries;
- the textual Udon assembly translator now registers the DLTI and LLVM
  dialects needed to read compiler-emitted physical MLIR diagnostics;
- steps 6-8 now cover synchronous and asynchronous exports. The compiler emits
  `Start`/status/result methods, the callback-first U# completion contract, and
  the upload/readback command pump in `binding.cs`. Both schema-v1 compatibility
  configs and schema-v2 annotated exports can request a compiler binding. The
  Device buffer manifest is finalized before binding generation, and the
  Importer copies compiler source instead of regenerating it;
- the installed-package CMake consumer, schema-v2 Hello binding artifact, and
  structured Async compiler tests pass. The real UdonBLAS SGEMM graph plans 10
  coroutine frames and 17 suspend points; its generated preview contains 10 PC
  globals, 55 spill globals, and 12 canonical async-object state globals, and
  all generated frame/object symbol references resolve. It contains 17 PC
  stores, 127 matched spill/reload pairs, 17 waiter registrations, 10 payload
  stores, 17 payload loads, 10 available plus 10 error transitions, and 20
  direct `udonhl.async_resume` requests. Its final `host.udon-async.mlir`
  contains 17 resume points, 22 calls to one 17-route dispatcher (including the
  two external completion branches), 17 static entries and 2 event bindings,
  with zero remaining dynamic `udonhl.async_resume`, Async Runtime/Coro or
  `vrc.event_as_async` operations. The
  full installed-package CMake consumer, including MNIST and the tiny
  transformer artifact, passes with this preview enabled. Production Async
  `host.uasm` now uses the finalized static-frame path. Current physical totals
  are generic MNIST 471 data + 41 extern signatures = 512 and the bounded
  Transformer 462 + 49 = 511. UdonBLAS SGEMM also remains within the
  512-address limit. UdonBLAS SGEMM, generic MNIST, and the hidden-size-8 generic
  Transformer now emit compiler-generated Async `binding.cs`, and the CMake
  regression test checks their public methods, command pump, callback/readback
  surface, and absence of Editor-generator dependencies. ONNX configs provide
  logical export names, so the generic prefix policy emits `StartRun` rather
  than duplicated `StartStart...` names. The Importer now
  rejects malformed typed export metadata, unsupported physical ABIs, and
  fingerprints that do not exactly match the canonical export/buffer/binding
  ABI. Recomputed fingerprints match the generated MNIST and UdonBLAS
  artifacts. The compiler now refreshes the fingerprint only after public
  whole-buffer metadata is finalized, and the Importer independently includes
  the same sorted buffer records. A CMake regression proves that renaming a
  public buffer changes the fingerprint; an Editor test rejects a changed
  logical byte length under the old fingerprint.
  The C++ ONNX Host finalizer now marks its five model-independent physical
  binding functions with `udoncl.export`. An external CMake consumer builds a
  schema-v2 artifact whose `export_overrides` only rename those discovered
  definitions; stable IDs are derived from their public signatures rather
  than repeated in JSON. Schema-v2 `internal_exports` can hide annotated
  physical definitions when an ordinary C Host wrapper owns the user-facing
  API; hidden and overridden symbols must be disjoint annotated definitions.
  MNIST and the bounded Transformer fixture now use such a C Host to
  create/adopt explicit OpenCL input/output buffers, form
  `udonclBufferView_t` tensors, bind a generic session, and export only
  `Initialize`, `Run`, and `Shutdown`. The Transformer uses distinct past and
  present key/value Device buffers so a fragment pass never reads its own
  render target. Queue-ordered copy/swap orchestration will reuse completed KV
  state without a Host round trip. Compiler-generated bindings
  now consume Manifest `element_type`/`public_access` and emit dtype-correct
  whole-buffer APIs such as `TryWriteInputBuffer(float[])` and callback-first
  `StartReadOutputBuffer(float[], receiver, event)`. User readback and the C
  Host async command pump share one GPU-readback callback with explicit mode
  dispatch, generation rejection, and `busy`/`readbackBusy` exclusion; the
  CMake regression verifies this surface on the public UdonBLAS artifact.
  The MNIST U# controller now performs one whole-array upload and callback
  readback through `TryWriteInput(float[])` and `StartReadOutput(float[])`; it
  no longer loops over element access exports or exposes the flattened
  16-scalar adapter. `udonclDeviceBufferAdopt` supplies the ownership-transfer
  bridge from an ordinary `clCreateBuffer` allocation to a generation-checked
  `udonclBufferHandle_t`, with native ownership and stale-handle coverage.
  Schema-v2 config regression tests reject unknown and duplicate
  `internal_exports`, conflicts with `export_overrides`, and attempts to hide
  every annotated export. Scalar lowering now prioritizes single-use i64
  constant materialization within a 64-value budget and shares four
  non-reentrant i64 scratch slots across constant construction and raw
  load/store, keeping the C Host Transformer artifact at 501 total addresses
  without weakening the 512 gate.
  The AOT OpenCL runtime regression now waits on two pending events, observes
  both completions, and proves that an error from the second dependency is
  returned. Compiler-generated Transformer binding checks also require the
  busy guard on suspending exports; executing re-entry while suspended remains
  part of the Unity gate because Native Async has no UdonBehaviour instance.
  Udon Async preparation now verifies the structured call discipline directly:
  every coroutine call result has exactly one same-block await, may otherwise
  be used only by its generated error check/load, and cannot have another
  coroutine call inserted before that await. Dedicated malformed-runtime tests
  reject both an escaping result and a parallel call.
  Compiler-generated `binding.cs` now starts with an exact identity header for
  its public ABI fingerprint, namespace, and class. The Importer verifies that
  header against the already-recomputed Manifest before copying the source, so
  a mismatched generated type fails during artifact validation instead of
  leaving prefab finalization waiting for a type that can never appear. The
  external CMake regression checks the identity on Hello, MNIST, the bounded
  Transformer, and UdonBLAS artifacts. The Importer now takes a separate early
  branch for validated compiler bindings: it copies that file and creates its
  UdonSharp program asset without parsing the Manifest again or invoking
  `VRCWrapperGenerator`. The old generator is confined to the no-`binding_source`
  compatibility branch. MNIST and Transformer Unity lifecycle tests require
  the installed source to remain byte-for-byte equal to the compiler artifact.
  The common CMake regression also proves that MNIST, the bounded Transformer,
  and UdonBLAS physical artifacts use the static Async backend: their final IR
  contains the shared resume dispatcher but no `vrc.await`, Async Runtime/Coro
  operation, or dynamic `udonhl.async_resume`, and their assembly exports the
  compiler-private completion entry without the legacy resume-state slot.
  The complete non-Unity CTest suite, including Native coroutine execution and
  all three compiler-generated Async bindings, passes 46/46.
  The required Luna/xhigh Unity MCP route imports the compiler-generated MNIST
  binding byte-for-byte without C# errors. Earlier Unity execution exposed SDK
  parser-unsafe literals and a 10-second Udon VM performance limit. Scalar lowering now forces
  large-exponent `f32` values and `INT32_MIN` through runtime bit
  materialization, uses direct aligned `SystemInt32Array` access for proven
  non-aliased zero-initialized internal `i32`/pointer globals, selects constant
  materialization and raw-address caches by an exact 512-address probe, builds
  remaining `i32` literals bytewise rather than bitwise, derives uncached raw
  addresses from the nearest cached base, and replaces exact zero `i32` raw
  stores with `System.Array.Clear`. Identical immutable FP32 tensor descriptor
  shapes are interned by the C++ ONNX finalizer. UdonDNN Host C is optimized at
  `-O1 -fno-inline`, and its public convolution API uses a typed internal enqueue
  on the Udon backend while retaining normal `cl_mem` allocation and blocking
  upload/readback boundaries. Boolean `i1` and/or/xor lower through Boolean
  control flow rather than Int32 externs. These changes reduced the MNIST
  assembly from 619,565 lines to 200,621 while retaining 512 total addresses;
  the bounded Transformer remains within the address gate. Static continuation
  placement subsequently removed the watchdog failure. The generated-U# MNIST
  controller now passes initialize, 784-element upload, callback-first run,
  10-logit readback, oracle comparison, and shutdown. The bounded Transformer
  numerical fixture also passes with separate present K/V outputs after fixing
  physical Texture row-pitch propagation; its remaining acceptance item is the
  user-facing generated-U# route rather than direct IUdonVM execution. Cleanup
  restores the Unity Assets tree and leaves the Editor console clear.

Still required before removing compatibility paths:

- migrate UdonBLAS/UdonDNN from Host staging to the asynchronous caller-owned
  Device API in `DEVICE_RESOURCE_ALIAS_PLAN.md`, then rerun the same generated-U#
  MNIST numerical acceptance path;
- validate the schema-v2 Transformer whole-buffer binding and Device-resident
  KV reuse through the required Luna/xhigh Unity MCP route. Its old schema-v1
  element access compatibility exports have been removed, and the flattened
  16-scalar adapter is hidden from the user-facing tensor API.

## 11. Tests and removal gate

### 11.1 Compiler tests

- annotation discovery and invalid annotation/config combinations;
- fixed-point may-suspend propagation;
- parent/child/grandchild coroutine calls without forced inlining;
- multiple suspend points and immediate-ready fast path;
- wait-list completion and error propagation;
- structured-Async verifier failures;
- frame liveness and slot coloring;
- generated entry adapter, manifest, fingerprint, C# naming collisions, and
  busy re-entry guards on every suspending export.

### 11.2 Native execution

Run the common Host Async contract through LLVM coroutine/event completion for:

- no suspend (implemented as an ordinary `func.call` retained beside the
  coroutine path);
- one suspend (implemented with external completion and result check);
- nested suspending calls (implemented for one parent/child edge);
- multiple sequential suspends (implemented with an observable checkpoint
  between the two completions);
- propagated GPU/runtime error (implemented for an ordinary negative `cl_int`
  event payload, without converting it to infrastructure Async failure);
- immediate completion (implemented with a value made available before the
  Host call).

Separately run the OpenCLAOT C runtime's actual `clWaitForEvents` loop with two
pending events and require both successful completion and propagation of a
failure from the second dependency. The common coroutine fixture already
proves two sequential external suspensions; together these cover the API loop
and its backend suspension behavior without replacing either by a blocking
test double.

### 11.3 Unity execution

- Keep direct SDK `IUdonVM` tests as lower-layer tests.
- Add user-path tests that call the generated low-level U# binding.
- Execute a generic C/OpenCL host and MNIST `Initialize`/`WriteInput`/`Run`/
  `ReadOutput` through generated binding source.
- Execute busy re-entry rejection through the generated binding while its
  first root export remains suspended.
- Require exact test counts, no skipped tests, no console errors, and cleanup of
  temporary assets.

### 11.4 Legacy removal gate

After compiler, Native, generic Unity, MNIST Unity, and repository precommit
tests pass, remove together:

- direct `vrc.await` state-machine generation in `LowerScalar`;
- forced helper inlining used only to make that lowering possible;
- manual transitive `vrc.async_effect` propagation;
- the limited single-await continuation-frame pass;
- Editor-side binding source generation;
- temporary compatibility flags selecting the old path.

Do not maintain two production async implementations.

## 12. Repository gate

Before each commit, stage only the intended files and run the single required
repository command:

```text
python scripts/precommit.py
```

Unity MCP diagnosis beyond that command is delegated to the required
Luna/xhigh Unity validation subagent. No Unity MCP call is made directly by the
primary agent.
