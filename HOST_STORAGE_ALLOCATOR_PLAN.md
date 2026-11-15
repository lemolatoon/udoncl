# Unified Host Storage Allocator Plan

> Status: **approved design; implementation in progress**
> Decision dates: **2026-11-05 through 2026-11-11**
> Parent architecture: [PLAN.md](PLAN.md)
> Acceptance targets: generated-U# MNIST remains green, fresh real TinyLlama
> generated-U# prefill matches every ORT output, and the mandatory pre-commit
> gate passes.

Current implementation checkpoint (2026-11-15):

- deterministic async placement metadata is emitted by the independent
  `AnalyzeHostStorage` pass;
- the ordinary interprocedural liveness/interference construction has been
  extracted from `LowerScalar.cpp` into reusable `HostStorageLiveness` analysis;
- live-across-suspend scalar values use the existing physical-type typed spill
  arrays instead of one `__udoncl_async_frame_*_spill_*` raw global per value;
- declared persistent spills and allocator-created transient spills share one
  bank per physical type without index collision;
- async resume PCs and async object state/waiter/payload fields still use the
  legacy raw-global path and remain migration work;
- persistent spill indices are colored from preserved full SSA interference,
  so non-interfering values in one coroutine reuse typed-array indices;
- address-cache capacity is now selected from exact post-allocation data and
  total-address probes even when constant pooling is active, and natural-loop
  depth weights hot spill indices ahead of cold duplicated accesses;
- the OpenCL AOT buffer and event tables use O(1) next-unused/free-list slot
  allocation, so enqueue does not linearly scan handle capacity inside one
  Udon invocation;
- feeding the discovered minimum simultaneous register count back into a fresh
  allocation remains migration work: a naive fixed-point retry was rejected
  because the post-spill required width grows again on each fresh allocation
  and exceeds the 256-slot compiler limit for the real TinyLlama Host;
- large `cf.switch` operations, including the async resume dispatcher, lower
  to a signed-i32 balanced decision tree while preserving case/default edge
  copies; runtime selection is logarithmic rather than a linear equality scan;
- persistent direct-slot promotion and cross-coroutine reuse remain migration
  work.
- `DiscoverHostRematerialization` now emits a deterministic, versioned
  physical-key/use inventory, and `HostLiteralKey` is shared by discovery and
  final lowering. Unity logical IDs, zero-based pool indices, and one-based
  physical tokens are distinct use kinds even when they share one physical
  literal key;
- source constants and decimal `SystemInt32` literals requested by final
  lowering now pass through one compatibility request API and appear in the
  storage report. Placement is intentionally still `resident_shared`; the
  report is aggregated by origin/kind/key and is not yet a use-site
  interference model;
- `AllocateHostStorage` now consumes a discovered literal inventory and emits
  a deterministic, type-separated `resident_shared` compatibility baseline.
  Resident keys use symbolic simultaneous groups rather than quadratic edge
  metadata. This is not yet the final pressure-selected placement: discovery
  still lacks literals created only inside final lowering, and the legacy
  constant-pool/rematerialization probes still run later;
- discovery now records source literal operand sites separately from the
  definition/requirement inventory. Stable consumer IDs, per-operand site IDs,
  `(consumer, key)` scratch instances, and per-physical-type consumer groups
  establish immediate-consumer lifetimes without changing executable IR. Two
  operands using the same key share one instance; distinct keys in one
  consumer remain distinct. Backend literals created only during final
  lowering are not yet represented as operand sites;
- discovery is not yet in the production compiler pipeline.
  `AllocateHostStorage` currently has two disconnected consumers of the
  shared allocator: the resident literal compatibility baseline and legacy
  async persistent spill indices. It does not yet combine writable literal
  scratch with direct SSA values. `MaterializeHostStorage` and
  `VerifyHostStorageCapacity` have not yet been extracted from final scalar
  lowering;
- reusable `HostStorageLiveness` now lives in a common analysis library. A
  feature-neutral `HostStorageProblem`/`HostStorageSolution` graph owns stable
  storage-domain and physical-type coloring, interference edges, and
  simultaneous groups. Resident immutable storage, writable direct/scratch
  storage, and typed spill arrays have distinct color namespaces; equal
  numeric colors in different domains do not alias. Legacy async persistent
  spill coloring and resident literal compatibility allocation use this graph,
  with placement and generated code unchanged;
- the fresh real TinyLlama Host remains the pressure reproducer: its last
  measured lowering required 752 data slots against an effective 503-slot
  data-table capacity, with 515 constant slots, including 509 `SystemInt32`
  constants.

## 1. Why this work remains

The current compiler has two different allocation models:

- ordinary scalar SSA values use interprocedural liveness, physical-type
  coloring, a requested 16-slot scalar bank, and typed spill arrays;
- values live across `async.coro.suspend`, resume PCs, and async object fields
  are converted earlier into individually addressed raw globals by
  `PrepareUdonAsyncRuntime`.

The ordinary allocator already preserves branch parallel-copy semantics and
checks the exact 512-address Udon limit. The async path is correct enough to
run MNIST, but it bypasses that allocator. In the real TinyLlama artifact this
produces thousands of raw objects and avoidable address calculation,
load/store, IR, and Udon Assembly overhead.

This plan replaces the split policy with one allocator. Ordinary SSA,
live-across-suspend values, resume PCs, and internal async object fields are
all logical variables. Liveness and cost decide whether each live segment is
placed in a direct typed Udon slot or a typed spill array.

The change must preserve C/OpenCL semantics. It must not make scalar arguments
reference-like, expose allocator details in the public ABI, or introduce a
runtime frame allocator for the currently supported non-recursive execution
model.

## 2. Existing facts to preserve

The accepted implementation already provides:

- a non-recursive interprocedural liveness/interference analysis for ordinary
  SSA values;
- typed scalar banks and one typed spill array per physical Udon type;
- correct simultaneous-operand and branch parallel-copy interference;
- exact final Udon data-address accounting and a hard 512-address check;
- a little-endian word-packed raw heap for genuinely address-observable data;
- generated-U# MNIST execution through whole-buffer write, callback-first run,
  whole-buffer read, and shutdown;
- a previously generated structural TinyLlama artifact with 155 Q4 nodes and
  two Q4 dispatch call sites per node; fresh generation and numerical
  acceptance are reopened by the current scalar-slot overflow.

These are regression constraints, not reasons to retain the current raw async
frame representation.

## 3. Storage contract

### 3.1 Logical variables

All internal values enter the same allocation model:

- ordinary SSA results, arguments, locals, and temporaries;
- values live across one or more suspend points;
- resume PCs and other internal coroutine state;
- internal async object state, waiter frame/PC, and payload fields;
- scalarized fields of fixed-layout aggregates.

Logical types remain available for verification. Values with the same
physical Udon type may share physical storage when their live segments do not
interfere.

### 3.2 Fixed and externally observable storage

The unified allocator does not relocate storage whose identity is externally
observable:

- public U#/Unity ABI parameter, result, status, and callback slots;
- named `Material`, `RenderTexture`, and `UdonBehaviour` resource bindings;
- public async command/completion communication slots;
- objects whose address, pointer identity, byte representation, aliasing, or
  volatility remains observable after whole-program optimization.

ABI boundary pack/unpack operations remain until linking and canonicalization
prove that the boundary is internal and removable.

### 3.3 Aggregate scalarization

By-value, non-escaping, fixed-layout structs are split into field scalar
values. Nested fixed aggregates are recursively scalarized when every access
can be expressed as a fixed field access.

An aggregate remains in raw observable storage only when required by semantics,
including address escape, pointer identity, union or cross-type aliasing,
volatile access, observable byte layout, `memcpy`/`memmove`, or a dynamic
offset that cannot be reduced to field selection.

Public C and U# ABI shapes are preserved. Scalarization is an internal
representation decision.

### 3.4 Direct typed slots

A direct slot is a typed Udon Heap location used as a physical register.
Non-interfering live segments may share the same slot. Caller values live
across a call interfere with callee values, and values live across nested
may-suspend caller/callee execution interfere.

There is no fixed semantic boundary at 16 registers. Direct-slot count is a
capacity- and cost-based allocator decision. The allocator must still reserve
enough direct storage for any simultaneous operand clique that cannot be
spilled.

Writable direct and rematerialization-scratch values share one color namespace
per physical Udon type. Immutable resident literals use a separate namespace:
they may have the same numeric color as a writable value, but must never map to
the same Heap slot because materialization overwrites scratch.

### 3.5 Typed spill arrays

There is at most one compile-time-sized spill array per physical Udon type in
a module/behaviour instance. There is no runtime spill allocator in the
initial implementation.

Non-interfering live segments share spill indices regardless of whether they
originated as ordinary SSA, coroutine state, or async object state. Array
indices are compile-time constants except where the program genuinely selects
an async object by a dynamic ID.

For a dynamic object ID, fields of one physical type occupy a statically
strided region:

```text
base + object_id * stride + field_offset
```

Static IDs and constant fields are folded. A statically unique object is
returned to the ordinary scalar allocation model. The initial implementation
does not replace a dynamic array access with a large selector branch merely to
use direct slots.

### 3.6 Raw linear memory

The word-packed raw `SystemInt32Array` remains only for data whose address or
byte layout is observable after optimization. Supported internal scalar,
coroutine, and async object state must not silently fall back to raw globals.

Every remaining raw object records a reason in `storage-report.json`.

### 3.7 Constants, literals, and rematerialization recipes

Every constant-like value enters the unified allocator before Udon Assembly
emission. This includes source `arith.constant` values and literals requested
later by lowering for resource IDs, table indices, raw addresses, descriptor
fields, return addresses, and helper arithmetic. `immutableSlot()` is a
materialization mechanism, not an allocation policy.

For each logical value the allocator chooses one of:

- one shared resident typed slot;
- a typed constant-pool load;
- derivation from an already-live value;
- straight-line rematerialization into a reusable typed scratch slot;
- fixed storage when its address or public identity is observable.

A derivation recipe records its source values, physical result type, exact
scratch requirements, required extern signatures, instruction cost, and
whether it may cross calls or suspend points. A source-derived placement is
legal only when every source is available at the use, or extending its
liveness is cheaper than retaining the result.

Udon Assembly has no general value-immediate operand: `PUSH` names a Heap
address. Rematerialization therefore reuses bounded typed scratch slots; it
does not pretend that literals consume zero storage. Recipe results participate
in the normal simultaneous-operand interference graph. Two values consumed by
the same `EXTERN` or `COPY` cannot share a scratch slot because Udon's stack
holds their slot addresses, not copied values. Scratch may be reused only after
the consumer and never across a may-suspend, reentrant, or unresolved call
boundary. A value needed after suspend is regenerated in the resume segment or
stored through the ordinary typed-spill policy.

Examples include:

- decoding a static tensor index from its existing tagged buffer token;
- deriving a Unity buffer index from an existing physical token;
- synthesizing an i32 from a shared small-literal basis;
- loading AOT-initialized i32/f32 bits from the raw constant pool;
- composing i64/f32 values through verified bit-exact recipes;
- deriving a raw address from a known base and constant offset.

Strings, Unity object references, public ABI slots, and values without a legal
recipe remain shared or fixed resident storage. Late-bound code/return
addresses retain a separate verified recipe because their values depend on
final instruction layout.

## 4. Liveness and live-range splitting

One analysis framework covers ordinary and persistent values. Suspend-to-resume
edges are part of the control-flow model. Callback/resume values and nested
may-suspend caller/callee frames interfere whenever they can coexist.

Live ranges may be split at suspend boundaries and other profitable points. A
value can therefore use a direct slot before suspend, occupy a typed spill
index only while suspended, and be reloaded into a direct slot after resume.
The ready/fast path must not execute spill traffic needed only by the suspended
path.

Stores are placed on the suspend edge and loads at the corresponding resume
entry when that preserves SSA semantics. Fast, ready, and error merge paths
must receive the same C-observable value without converting copy semantics to
reference semantics.

Constants, lowering-required literals, and cheap pure expressions use the
recipe and placement model in section 3.7 rather than operation-specific slot
heuristics.

## 5. Small-pass pipeline

Each transformation pass has one narrow responsibility and normally one
implementation file. Passes exchange explicit UdonHL operations or attributes,
not symbol-name conventions.

```text
Async runtime and coroutine CFG legalization
  -> ScalarizeHostStorage
  -> MarkAsyncPersistence
  -> SplitHostLiveRanges
  -> DiscoverHostRematerialization
  -> AllocateHostStorage
  -> MaterializeHostStorage
  -> VerifyHostStorageCapacity
  -> UdonHL scalar/extern lowering
  -> Udon Assembly
```

### 5.1 `ScalarizeHostStorage`

Splits eligible fixed-layout aggregates into scalar logical values. It does not
choose physical storage.

### 5.2 `MarkAsyncPersistence`

Identifies live-across-suspend values, resume state, and async object fields.
It emits logical persistence and object-identity information only. It must not
create one raw global per value.

### 5.3 `SplitHostLiveRanges`

Creates live segments and explicit logical save/restore points. It does not
assign slots or spill indices.

### 5.4 `DiscoverHostRematerialization`

Builds deterministic typed recipes for source constants and
lowering-required literals. It records derivation sources, exact scratch and
extern requirements, instruction cost, loop weight, and call/suspend legality.
It does not select resident storage, spill indices, or scratch colors, and it
does not emit Udon instructions.

The current checkpoint implements only deterministic physical keys,
context-free recipe names, stable source/backend requirement identities, and
source-operand immediate-consumer groups. Concrete derivation sources,
ordinary SSA/result interference, scratch and extern costs, loop weight, and
backend use sites remain required before any rematerialization placement is
enabled.

### 5.5 `AllocateHostStorage`

Consumes common liveness/interference analyses and assigns each segment a
direct color, typed spill index, fixed ABI placement, or justified raw
placement. It emits placement IR/attributes and does not materialize Udon
loads/stores.

The allocator chooses fixed, resident, typed-spill, pooled, derived, or
rematerialized placement using the same liveness graph and the exact address
and code-cost model.

### 5.6 `MaterializeHostStorage`

Mechanically converts placement decisions into direct storage, existing
`udonhl.spill_store`/`udonhl.spill_load` operations, constant-pool reads,
derivation sequences, and typed scratch use. It creates fixed-size typed arrays
and must not invent a new placement decision.

### 5.7 `VerifyHostStorageCapacity`

Counts the exact final address representation. If materialization changes
index constants, scratch, extern signatures, or other address costs, the
driver may rerun allocation with the measured totals. Generated placement is
tagged so a retry replaces only compiler-generated materialization.

The process normally converges in one allocation. At the 512 boundary it may
iterate until stable. Failure reports category totals and actionable placement
candidates; it never silently selects a semantically different fallback.

## 6. Allocation policy

The allocator optimizes in this strict order:

1. preserve liveness, copy, call, and suspend semantics;
2. remain within the exact 512-address Udon limit;
3. reduce typed-array accesses on statically hot paths;
4. reduce generated code size;
5. reduce spill-array extent.

The initial deterministic heuristic begins with spill placement and promotes
the most profitable live segments to direct slots. Non-interfering segments
that share a color can be promoted together.

Static cost considers:

- direct address count and exact fixed/extern/resource/scratch costs;
- array index materialization and scratch requirements;
- static load/store frequency;
- loop nesting weight;
- dispatch and descriptor hot paths;
- suspend crossings;
- call-graph frequency estimates;
- rematerialization cost;
- the unspillable simultaneous-operand clique.

Weights live in one configuration definition and are reported for diagnostics.
No profile data is required initially.

Ties are resolved by physical type, stable function symbol order, block order,
operation order, and result number. Placement must not depend on pointer values,
hash-map enumeration, platform, or Debug/Release mode.

## 7. Placement representation and reporting

For every logical live segment, allocator output records:

- logical and physical type;
- direct color or typed spill index;
- live-segment identity;
- suspend save and resume restore positions;
- estimated cost and placement reason;
- any fixed ABI or raw-observability exclusion reason.

For every literal-like value it also records a stable typed key, use count,
loop weight, legal recipes, selected placement, scratch color, added extern
signatures, estimated instruction cost, call/suspend segments, and the reason
for the choice. Inventory and placement order are deterministic and are fixed
before scalar Udon Assembly emission.

`storage-report.json` reports at least:

- direct slots by physical type;
- direct-color reuse count;
- typed spill-array extent by physical type;
- spill-index reuse count;
- suspend-crossing segment count;
- rematerialized value count;
- literal use/value counts and per-type `resident_shared`, `raw_pool`,
  `rematerialized`, `derived`, and `fixed` placement counts;
- resident literal slots, typed scratch slots, and addresses saved;
- distinct extern signatures and extern signatures added by recipes;
- estimated emitted instruction bytes and weighted execution cost;
- stable per-candidate recipe and selection diagnostics;
- call-crossing and suspend-crossing rematerialized segment counts;
- raw objects and individual reasons;
- data-slot count, distinct-extern count, and final Udon address count out of
  512;
- estimated hot-path load/store count.

Allocator tests consume placement results directly rather than inferring all
decisions from final Udon Assembly text.

## 8. Supported execution boundary

The initial backend retains the current structured, direct, non-recursive call
graph, one active root, and one waiter per pending async object. These limits
are explicit analysis inputs and diagnostics, not undocumented assumptions.

Recursive SCCs, indirect may-suspend calls, concurrent roots, escaping
coroutine results, and multiple waiters require a separate dynamic-frame
design. GPU commands may remain concurrently pending in the in-order queue;
the single-root restriction concerns Host execution state.

## 9. Migration sequence

1. Introduce placement IR and update the storage report without changing code
   generation.
2. Add fixed-layout struct scalarization and extract common
   liveness/interference analyses used by normal and async values.
3. Add live-range splitting and suspend-edge save/resume-entry restore markers.
4. Introduce a versioned literal key/use/recipe inventory with deterministic
   output while retaining current resident code generation.
5. Route source `arith.constant` and backend-required i32 literals through the
   same logical request API; `immutableSlot()` stops making placement policy.
   Source constants, Unity ID/index/token uses, and decimal `SystemInt32`
   literals requested during final lowering are routed. Late-bound code
   addresses and direct `addSlot` state still require explicit fixed/recipe
   migration.
6. Model recipe-result scratch lifetimes in the normal simultaneous-operand,
   call, branch-copy, and suspend interference graph.
7. Move shared resident and raw-pool placement into `AllocateHostStorage`.
8. Add generic derived/rematerialized placement and make the non-ONNX
   520-static-resource pressure fixture fit the exact 512-address budget.
9. Migrate Unity buffer IDs, raw addresses, pool/spill indices, and descriptor
   literals to the same mechanism. Delete the old per-feature
   `unity_buffer_id_rematerialization`, raw-address-cache, spill-index-cache,
   and provisional `static_tensor_id_*` allocation policies once their tests
   use the general report.
10. Add bit-exact i1, f32, and i64 recipes. Strings and Unity references remain
    shared/fixed until a separately verified table representation exists.
11. Move live-across-suspend values, resume PCs, and internal async object
    fields into the unified model; delete superseded raw-frame generation and
    incidental symbol-name tests.
12. Run native/compiler integration and generated-U# MNIST numerical
    validation.
13. Generate the fresh real TinyLlama artifact, execute generated U# prefill,
    and compare all 128,000 logits and 44 present K/V outputs with the ORT
    oracle.
14. Run the mandatory pre-commit gate.

Each step must build and pass its relevant focused tests. Do not accumulate a
large knowingly broken pipeline between steps.

Before step 6, run discovery in the production pipeline and represent literal
use sites and simultaneous consumers as nodes in the common
`HostStorageProblem`/`HostStorageSolution` boundary. Do not add another model-
or resource-specific placement heuristic to final scalar lowering.

## 10. Tests

Focused compiler tests cover:

- non-interfering values sharing a direct slot;
- interfering values receiving different placements;
- direct capacity pressure spilling to a typed array;
- non-interfering spills sharing an index;
- simultaneous operands and branch parallel-copy semantics;
- caller/callee and nested may-suspend interference;
- suspend live-range splitting and save/restore placement;
- no suspended-path spill access on the ready path;
- one and multiple suspend points;
- dynamic async object IDs and statically unique objects;
- struct field scalarization and every raw-observability exclusion;
- logical types sharing a compatible physical type;
- hot resident and cold rematerialized constants under measured pressure;
- two literals consumed by one extern receiving distinct simultaneous scratch;
- scratch operands and the writable result of one consumer receiving pairwise
  distinct colors;
- resident and writable colors using distinct non-aliasing storage domains;
- scratch reuse only after the consumer;
- ordinary and lowering-required literals sharing one planner/report;
- derived values whose sources remain available at every selected use;
- recipe regeneration after calls and resume rather than scratch persistence;
- bit-exact i32 boundaries, i1, i64, f32 `-0`/NaN/Inf, and string sharing;
- deterministic output across repeated builds;
- exact 512-address success and detailed overflow failure.

Integration proceeds through:

1. native/Udon result equality for small C Hosts;
2. a multiple-kernel, multiple-suspend Host;
3. generated-U# MNIST numerical execution;
4. real TinyLlama ONNX artifact generation and generated-U# prefill numerical
   comparison of every public output;
5. `python scripts/precommit.py`.

Tests assert placement and behavior, not incidental diagnostic wording or raw
symbol names.

## 11. Acceptance gates

### 11.1 MNIST

The existing user-facing generated-U# test remains authoritative. It imports
the CMake-produced binding, writes all 784 FP32 inputs through the whole-buffer
API, starts callback-first inference, reads all 10 logits through the generated
API, compares against the native/ORT oracle, and shuts down without direct
IUdonVM heap manipulation, test-only source generation, console errors,
cleanup leaks, or the Udon watchdog.

### 11.2 TinyLlama

The allocator change must preserve:

- real ONNX-to-artifact generation;
- no more than 512 final Udon addresses;
- 155 Q4 nodes with two Q4 dispatch call sites per node;
- deterministic placement;
- no model-specific literal or static-resource allocation rule;
- no dependency on `__udoncl_async_frame_*_spill_*` raw globals;
- no major regression from the currently observed approximately 818 MB
  `host.uasm` without an explained architectural reason.

The allocator milestone ends only after the ordinary generated U# binding
compares all 128,000 logits and all 44 present K/V outputs with the ORT oracle.
Artifact generation alone is not completion. The later interactive conversation
path remains governed by the dedicated TinyLlama plan and depends on this
allocator rather than duplicating it.

### 11.3 Repository gate

After staging the intended final change, run exactly:

```text
python scripts/precommit.py
```

The change is not committable until the command prints `precommit=passed`.

## 12. Completion definition

This plan is complete only when:

- normal and async internal values use the unified allocator;
- fixed-layout structs are scalarized where legal;
- direct slots and typed spill indices are reused by liveness;
- supported async state no longer becomes individual raw globals;
- remaining raw objects have observable-semantics reasons;
- exact capacity and deterministic placement tests pass;
- generated-U# MNIST passes;
- the real TinyLlama generated-U# numerical gate passes;
- stale allocator paths and tests are removed;
- the mandatory pre-commit gate passes.
