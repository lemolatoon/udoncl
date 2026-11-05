# Host Storage Allocator Plan

> Status: **approved design; implementation not started**
> Decision date: **2026-11-05**
> Parent architecture: [PLAN.md](PLAN.md)
> Final acceptance target: **a user-facing compiler-generated U# binding runs MNIST end to end in Unity and its numerical test passes without the Udon 10-second watchdog**

## 1. Problem and current evidence

The Host compiler currently maps C globals, fixed `alloca` objects, descriptors,
and pointer-visible tensor storage into an emulated linear heap. The current
implementation uses a `SystemInt32Array` with one array element per logical
byte. A general four-byte load reconstructs the value from four array reads,
pointer additions, shifts, and OR operations.

This is correct enough as a bootstrap representation but is not the final Host
storage model. Small C helpers such as `get_convolution` become large Udon
instruction sequences even though they contain only a few C expressions.

The last Unity-tested MNIST artifact imported and compiled its generated U#
binding, but its focused test finished with 0/1 passing. The SDK Udon VM hit its
10-second limit at PC 468,832 inside the synchronous helper
`get_convolution`; numerical comparison was not reached. Cooperative
checkpoint insertion also underestimates synchronous callees because a call is
currently counted as one operation without its transitive Udon expansion.

Static instruction evidence for the current backend is:

| Operation | Udon opcodes | Extern calls |
|---|---:|---:|
| Direct scalar slot `COPY` | 3 | 0 |
| Typed array `Get` or `Set` | 4 | 1 |
| Current byte-per-element raw `i32` load | 67 | 13 |
| Current byte-per-element raw `i32` store | 59 | 14 |

No defensible wall-clock comparison exists yet. Editor IUdonVM, ClientSim, and
VRChat client measurements must be reported separately.

## 2. Required compilation order

Storage planning must not preserve address-taking artifacts that whole-program
optimization can remove. The required Host order is:

```text
Clang Host/runtime/library LLVM IR
  → static link
  → MLIR import and high-level recovery
  → export-aware internalization
  → SROA / mem2reg / inlining / constant propagation
  → descriptor specialization / aggregate scalar replacement / DCE
  → ABI, alias, and address-observability analysis
  → MLIR Async lowering and Udon static-frame legalization
  → Host storage planning and register allocation
  → Udon extern legalization
  → Udon Assembly + allocator report
```

Only `UDONCL_EXPORT` functions and required physical boundaries remain externally
observable. Optimization must preserve C semantics and emit a remark when an
object cannot be promoted.

For the generated MNIST graph, descriptor creation/configuration/use, fixed
handles, and `alpha`/`beta` pointer wrappers should normally specialize to SSA
values. Model input/output storage and genuinely observable memory remain
memory objects.

## 3. Four physical storage classes

Every surviving Host value belongs to exactly one class.

### 3.1 Fixed named Udon Heap slots

These are outside register allocation because Unity or U# accesses them by
name:

- exported parameter, result, and status slots;
- `Material`, `RenderTexture`, and `UdonBehaviour` resource bindings;
- async command/completion ABI slots;
- other event-visible public state.

Entry code copies inputs from fixed slots into internal registers. Exit or
completion code copies results back.

### 3.2 Typed scalar register bank

A small number of typed scalar Udon Heap slots act as physical registers.
Address-unobservable SSA values, arguments, results, locals, globals, and
temporaries are candidates.

Allocation is interprocedural over the complete non-recursive call graph.
Caller values live across a call interfere with callee values. Ordinary calls
must not add runtime caller-save or callee-save traffic merely to simplify the
allocator.

### 3.3 Per-physical-type typed spill arrays

Values that do not fit in the scalar register bank spill into one fixed-length
array per physical Udon type, for example:

```text
SystemInt32Array
SystemInt64Array
SystemSingleArray
SystemBooleanArray
UnityEngineRenderTextureArray
VRCUdonUdonBehaviourArray
```

Do not unify spills into `object[]`; boxing, casts, and runtime type checks are
not part of the design. Logical types may share an array when they lower to the
same physical Udon type, but the compiler retains logical storage keys for
verification.

Array extents are the compile-time maximum simultaneous spill count after
liveness coloring. There is no runtime spill allocator in the initial
non-recursive backend.

Values live across a suspend are forced into a persistent spill-frame region.
Nested may-suspend caller/callee live ranges interfere. The initial runtime
continues to allow one active root and one waiter per pending async object.

### 3.4 Word-packed raw linear memory

Only objects whose address or byte layout remains observable after optimization
use raw memory. Reasons include:

- pointer identity or escaping pointers;
- dynamic indexing that cannot be scalarized;
- cross-type or byte aliasing;
- `ptrtoint`/`inttoptr`;
- volatile access;
- `memcpy`, `memmove`, or byte-oriented library semantics;
- unaligned loads and stores.

The final raw representation is a little-endian word-packed
`SystemInt32Array`, with four logical bytes per array element. The existing
one-element-per-byte representation is a migration implementation, not a
target ABI.

Required fast paths are:

- aligned `i32`, pointer, and `f32` load/store with one array access plus any
  required bitcast;
- aligned word bulk copy and clear;
- byte/unaligned head and tail using shifts, masks, and read-modify-write;
- correct cross-word unaligned access;
- exact overlap semantics for `memcpy` versus `memmove`.

Every raw object and the reason it could not be promoted appears in the
allocator report.

## 4. Spill-index materialization

Udon has no immediate-value `PUSH`. Giving every spill index its own constant
slot would reproduce the Heap-slot problem that typed arrays are intended to
solve.

The allocator therefore treats index materialization as part of the same cost
problem:

- hot indices may receive immutable scalar cache slots;
- other indices are materialized into shared scratch using the module-wide
  `0..15` nibble constants;
- scalar registers, index-cache slots, scratch slots, and array-access count
  compete within the same address budget;
- no policy allocates one permanent constant slot for every spill index.

The register count and hot-index cache size are selected from measurements, not
hard-coded as an unverified performance claim.

## 5. Recursion and concurrency boundary

The initial allocator accepts only a structured direct, non-recursive call
graph. Recursive SCCs, indirect may-suspend calls, concurrent roots, escaping
coroutine results, and multiple waiters are explicit compile errors.

Future recursion support requires a separate dynamic-frame design using typed
spill-array frame bases or UdonSharp-style save/restore. It must not be
simulated by corrupting the static non-recursive coloring model.

GPU commands may remain concurrently pending within the in-order queue even
though the C Host execution context admits one active root.

## 6. Capacity and diagnostics

The SDK address limit is checked with the exact final program representation.
The count includes:

- fixed public and resource slots;
- scalar registers and scratch;
- hot-index constants;
- one slot for each typed spill array;
- the raw heap and its control state;
- async runtime state;
- extern signatures and all other addressable data.

If the program exceeds 512 addresses, compilation fails with:

- category totals;
- maximum live set by physical type;
- register and spill counts;
- cached and synthesized index counts;
- raw objects and fallback reasons;
- the largest async frame;
- actionable slot-reduction candidates.

The compiler must not silently switch to a slower representation to hide a
capacity failure.

## 7. Implementation phases

1. Add a storage-analysis report without changing code generation. Classify
   fixed slots, promotable SSA values, async-persistent values, and raw objects.
2. Add export-aware internalization and the whole-program promotion pipeline.
   Verify that MNIST descriptors and scalar pointer wrappers disappear where
   legal.
3. Introduce physical-type register banks and non-recursive interprocedural
   interference coloring. Keep existing storage as a differential oracle.
4. Add typed spill arrays, fixed extents, persistent async frame regions, and
   spill-index materialization/caching.
5. Replace the byte-per-element raw heap with word-packed raw memory and add
   aligned/unaligned/memcpy/memmove tests.
6. Integrate the exact 512-address capacity check and allocator diagnostics.
7. Tune register and index-cache budgets using the benchmark matrix.
8. Regenerate MNIST and run native, compiler, Unity import, generated-U#, and
   numerical acceptance tests.
9. Remove superseded byte-raw optimizations, unnecessary cooperative-checkpoint
   experiments, legacy direct-await code, and the old Editor binding generator
   only after the acceptance gate passes.

## 8. Tests and benchmarks

Compiler tests must cover:

- scalar promotion and non-promotion with an explicit reason;
- fixed structs and arrays scalarized into SSA;
- address escape, alias, volatile, and pointer-int round trips;
- interprocedural interference across nested calls;
- typed spill coloring and array extent;
- values live across one and multiple suspend points;
- hot-index cache versus synthesized index paths;
- aligned and unaligned integer/float/pointer raw accesses;
- cross-word loads/stores and little-endian layout;
- `memcpy`, overlapping `memmove`, clear, and aggregate initialization;
- deterministic allocator output and detailed 512-address failure.

The SDK IUdonVM microbenchmark compares:

1. direct scalar slot `COPY`;
2. typed array fixed-index `Get/Set`;
3. typed array dynamic-index `Get/Set`;
4. existing byte-per-element raw `i32/f32` load/store;
5. word-packed aligned and unaligned raw `i32/f32` load/store.

Reports include opcode count, extern count, iteration count, elapsed time,
Unity version, Worlds SDK version, execution environment, and warmup policy.
Editor results do not substitute for ClientSim or VRChat client measurements.

## 9. MNIST acceptance gate

This plan does not finish when the allocator compiles an artifact. It finishes
only when the same public generated U# surface available to a package user runs
MNIST end to end. The Unity acceptance test must:

1. import the CMake-produced MNIST artifact and its compiler-generated
   `binding.cs` without an Editor-regenerated wrapper;
2. create and initialize the linked UdonCL/UdonONNX library instance through
   the generated U# API;
3. pass the complete 784-element FP32 MNIST input through the generated
   whole-buffer U# write API, not through direct IUdonVM heap manipulation or a
   test-only generated source initializer;
4. start inference through the generated callback-first U# `Run` export;
5. wait for exactly one completion callback and read status/result only through
   the generated U# API;
6. read all 10 FP32 logits through the generated whole-buffer U# readback API;
7. compare every logit and the predicted class with the native/ORT reference;
8. shut down or dispose the instance through the generated U# lifecycle API;
9. verify console errors, callback/busy state, temporary assets, and lifecycle
   cleanup.

Direct Udon Assembly entry invocation, direct program-variable injection,
Editor-only IUdonVM slot reads, model-specific handwritten U# wrappers, and
test-only changes to the generated Host or binding are not acceptable
substitutes for this gate.

The plan is complete only when all of the following also hold:

- general raw accesses for descriptors and temporary scalar pointer wrappers
  are absent from the generated MNIST Host path;
- the allocator report explains every remaining raw object;
- the exact final address count is at most 512;
- compiler and native differential tests pass;
- the compiler-generated U# binding imports, compiles, and is the only public
  Host API used by the Unity numerical test;
- the Unity Test Runner MNIST test passes through that generated U# path and
  matches the native/ORT oracle within the specified error;
- no Udon 10-second watchdog fires;
- console inspection and temporary-asset cleanup pass;
- Transformer and UdonBLAS capacity/correctness regressions remain green.

Only after this gate may the project remove the legacy direct-await path, the
old Editor binding generator, and storage-specific checkpoint experiments.
