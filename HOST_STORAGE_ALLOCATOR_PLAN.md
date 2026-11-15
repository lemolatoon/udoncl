# Unified Host Storage Allocator Plan

> Status: **approved design; implementation in progress**
> Decision dates: **2026-11-05 through 2026-11-11**
> Parent architecture: [PLAN.md](PLAN.md)
> Acceptance targets: generated-U# MNIST remains green, fresh real TinyLlama
> generated-U# prefill matches every ORT output, and the mandatory pre-commit
> gate passes.

Current implementation checkpoint (2026-11-16):

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
  metadata. An opt-in pressure request can now select complete explicit
  static-resource-index groups for i32 rematerialization. This is not yet the
  final capacity loop: production exposure/materialization and exact
  post-selection remeasurement remain disconnected, and the legacy
  constant-pool/rematerialization probes still run later;
- discovery now records source literal operand sites separately from the
  definition/requirement inventory. Stable consumer IDs, per-operand site IDs,
  `(consumer, key)` scratch instances, and per-physical-type consumer groups
  establish immediate-consumer lifetimes without changing executable IR. Two
  operands using the same key share one instance; distinct keys in one
  consumer remain distinct. Backend literals created only during final
  lowering are not yet represented as operand sites;
- the source scratch schema is now version 2. A scratch instance identity
  includes consumer, physical literal key, and logical MLIR type. Equal i8,
  i16, i32, and index values may share one `SystemInt32` physical key, but they
  remain distinct typed SSA materialization instances; same-type duplicate
  operands still share one instance. Consumer groups remain physical-type
  groups, so simultaneously consumed logical types receive non-aliasing
  writable colors. `AllocateHostStorage` revalidates the logical type against
  every actual constant operand and emits writable-allocation schema version 2;
- discovery now runs in the production Host pipeline immediately after
  `PrepareUdonAsyncRuntime` and before `AllocateHostStorage`. The CMake package
  integration fixture proves that the production async intermediate contains
  the inventory, use-site, allocation, and writable-allocation schema versions
  while containing no `udonhl.literal_materialize`; this integration is
  metadata-only and does not select or materialize a new placement.
  `AllocateHostStorage` currently has two disconnected consumers of the
  shared allocator: the resident literal compatibility baseline and legacy
  async persistent spill indices. A third, metadata-only candidate graph now
  combines source literal scratch with ordinary writable SSA values and emits
  deterministic colors without selecting that placement.
  `MaterializeHostStorage` and `VerifyHostStorageCapacity` have not yet been
  extracted from final scalar lowering;
- reusable `HostStorageLiveness` now lives in a common analysis library. A
  feature-neutral `HostStorageProblem`/`HostStorageSolution` graph owns stable
  storage-domain and physical-type coloring, interference edges, and
  simultaneous groups. Resident immutable storage, writable direct/scratch
  storage, and typed spill arrays have distinct color namespaces; equal
  numeric colors in different domains do not alias. Legacy async persistent
  spill coloring and resident literal compatibility allocation use this graph,
  with placement and generated code unchanged;
- `udonhl.literal_materialize` now defines the non-Pure explicit IR contract
  that a later `MaterializeHostStorage` pass will insert. Its verifier fixes
  the selected recipe, typed literal/key, writable color, exact resident seed
  bindings, required externs, helper scratch, and generated version. The first
  supported contract is `rematerialize_i32`;
- an opt-in `MaterializeHostStorage` pass now parses a separate versioned
  selected-placement schema. Candidate-only modules remain a strict no-op. For
  a selected typed instance, it cross-checks the exact instance/site list,
  actual constant operands, writable candidate color/extent, and referenced
  resident seed allocations, then inserts one `udonhl.literal_materialize`
  before the consumer and rewrites that instance's operand sites. It is not in
  the production pipeline; `AllocateHostStorage` emits selections only for an
  explicit, versioned pressure request, and `LowerScalar` lowers the selected
  operation through its allocated writable color. The pass and operation
  verifier now share the exact i32 recipe helper; all selected plans, resident
  inventory/allocation versions, seed bindings, required externs, and actual
  operands are validated before mutation. Every distinct constant definition
  represented by a selected instance is collected, and each one is erased only
  when all of its uses have been replaced. Focused tests cover separate
  same-value definitions, duplicate operands, simultaneous distinct colors,
  mixed logical types, exact call arguments, ineligible branch selection,
  stale operands, schema versions, recipe fields, and missing/incorrect/
  duplicate seed bindings. Candidate colors remain non-executable without an
  explicit selected placement;
- opt-in `LowerScalar` support now recognizes an explicit
  `udonhl.literal_materialize`, assigns its result the selected function-local
  writable-color slot, resolves only already resident seed keys, and emits the
  mechanical nibble shift/OR sequence for i8, i16, i32, and index logical
  literals. It neither calls `immutableSlot()` nor creates a missing seed, and
  a focused Udon Assembly interpreter checks boundary values and instruction
  counts. LowerScalar now reparses the post-materialization stable value IDs,
  consumes fixed direct colors and literal colors in one interprocedural
  coloring, rejects interfering fixed colors, and maps resident key/type/color
  allocations to physical resident slot names. MaterializeHostStorage rewrites
  direct allocation IDs after inserting literal operations and erasing dead
  constants, so downstream IDs describe the mutated IR. Fixed allocation
  consumption is gated on an actual `udonhl.literal_materialize`; candidate-only
  modules bypass the new schema even when deliberately malformed candidate
  attributes exist, preserving the legacy coloring and resident-slot path.
  Focused tests also prove that every seed uses its binding color's physical
  resident slot and that selected scratch/direct colors remain distinct;
- the next implementation boundary is deliberately model-independent: finish
  the focused pressure-selection contract and exact post-selection capacity
  test, then wire backend-use exposure, allocation, materialization, and
  remeasurement in production order. After that, move every remaining late
  backend `SystemInt32` request that can be rematerialized out of hidden
  `LowerScalar::immutableSlot()` policy and into explicit pre-lowering IR use
  sites. Source-operand rematerialization alone cannot solve the real-model
  pressure because most measured integer constants are created by backend
  lowering rather than present as source SSA operands;
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
source-operand immediate-consumer groups. The backend requirement inventory
records which values final lowering requests, but it does not yet identify the
exact pre-lowering consumer operand/phase where writable scratch must be live;
it therefore cannot be materialized safely. Concrete derivation sources,
scratch and extern costs, loop weight, and backend use sites remain required
before any rematerialization placement is enabled. The allocator now joins
those source scratch groups with stable ordinary SSA/result liveness, including
live-through values and direct-call entry destinations, and emits
candidate-only writable colors. Branch terminator literals remain explicitly
ineligible until discovery assigns edge-specific consumer phases; they do not
receive an unsafe or over-conservative scratch candidate. Recursive nested SSA
collection is opt-in for this candidate analysis, so the existing production
LowerScalar allocation input remains unchanged. This is an opt-in pass
checkpoint rather than a production placement decision.

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

It consumes allocator-selected placement decisions, never candidate metadata,
and emits explicit placement IR. Every literal recipe must already name its
domain-qualified writable color, resident seed bindings, required externs, and
helper scratch. `rematerialize_i32` is a physical `SystemInt32` recipe shared
by i8, i16, i32, and index logical literals. Materialization and final lowering
must not call `immutableSlot()` or otherwise add hidden seed storage, choose a
different recipe, or independently recolor direct SSA values.

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

Completed foundations include placement IR, common liveness/interference
analysis, typed persistent-spill coloring, deterministic literal
key/requirement discovery, source immediate-consumer scratch groups, a shared
candidate color graph, the verified `udonhl.literal_materialize` contract, and
production metadata-only rematerialization discovery.
Continue in the following independently testable commits:

1. **Completed (2026-11-16):** run `DiscoverHostRematerialization` in the
   production Host pipeline
   immediately before `AllocateHostStorage`. Prove in an integration fixture
   that the production intermediate IR contains the deterministic versioned
   inventory. This commit must not change generated Udon Assembly.

   **Completed schema prerequisite (2026-11-16):** version source literal use
   sites and writable allocations at version 2 so canonical scratch instance
   identity includes logical type as well as its shared physical key. Mixed
   i8/i32 operands with identical bits now remain separate typed instances and
   receive distinct simultaneous writable colors, while same-type duplicate
   operands continue to share one instance. This remains metadata-only; Step 2
   has not materialized executable IR yet.

2. Add `MaterializeHostStorage` as one pass in one implementation file. It
   consumes only an explicit selected-placement schema and rewrites selected
   source operand sites to `udonhl.literal_materialize`; candidate metadata
   alone is never executable. Test duplicate operands, simultaneous distinct
   operands, calls, branches, and rejection of stale/partial decisions.

   **Completed (2026-11-16):** the independent pass, selected-placement
   schema, candidate-only no-op, typed instance/site/color/extent checks,
   resident seed lookup, deterministic insertion, and duplicate-operand reuse
   are implemented. Exact recipe validation is shared with the operation
   verifier and runs transactionally before mutation; resident schema versions
   and actual operands are checked, every distinct replaced constant definition
   is erased when dead, and behavioral/negative tests cover exact call
   operands, ineligible branch-derived selections, stale operands, recipe
   fields, and resident seed bindings. The pass remains opt-in until the
   allocator emits selected placements; production activation belongs to later
   steps.

3. Teach `LowerScalar` to lower an explicit `rematerialize_i32` operation
   mechanically from its exact seed bindings, extern list, and writable color.
   Add focused Udon Assembly and native-equivalence tests. It must not call
   `immutableSlot()`, choose a recipe, recolor the result, or add hidden seeds.

   **Completed (2026-11-16):** explicit i32 recipe lowering, exact selected
   scratch naming, resident-key lookup, no-hidden-seed failure, deterministic
   shift/OR emission, and physical `SystemInt32` behavior for i8/i16/i32/index
   are implemented and covered by a focused interpreter. Post-mutation stable
   IDs are regenerated for retained direct values; LowerScalar consumes those
   fixed direct colors and the literal color in one interprocedural plan,
   rejects fixed-color interference, and gives resident binding colors physical
   slot identities. A selected color 7 scratch and simultaneous color 8 result
   execute with equal native-value semantics. Allocation consumption is gated
   on an actual materialization operation; a candidate-only module, including
   one with intentionally incomplete candidate attributes, retains the legacy
   code-generation path. Candidate discovery alone does not activate placement.

4. Represent rematerializable late backend integer consumers as explicit
   pre-lowering IR use sites. Start with Unity buffer IDs, pool indices/tokens,
   static tensor/resource indices, descriptor fields, and typed spill/pool
   indices, but define this by backend value role rather than ONNX model or
   operator. Keep late code addresses and ABI/state constants fixed until
   their distinct semantics are modeled. Prove that the inventory and
   executable IR name the same consumer phases.

   **Next implementation slice: static-resource dispatch indices.** Do not
   start with typed-spill indices: `AllocateHostStorage` currently creates the
   spill operations after discovery, so exposing their index attributes first
   would not yet produce allocator-visible use sites. Static-resource indices
   already have a stable dispatch consumer phase and account for material
   integer pressure without depending on a model or source operator.

   Extend `vrc.dispatch` backward-compatibly rather than adding a marker-only
   operation. Add a zero-or-more `i32` `static_resource_indices` operand
   segment and an optional dense `static_resource_positions` attribute. The
   operation has one fixed operand followed by two variadic segments, so its
   ODS traits must include `AttrSizedOperandSegments`; builders must provide
   `operandSegmentSizes = [1, resources.size(),
   static_resource_indices.size()]`. Preserve the existing textual form when
   the new segment is empty. When nonempty, print an optional explicit clause,
   for example `static_indices(%i0, %i1)`, before the attribute dictionary;
   `functional-type(operands, results)` continues to describe all three
   segments. The position array is sorted, unique, has the same length as the
   index operands, and contains only resource ordinals in range. Every operand
   is exactly `i32` and must agree with the static-resource provenance and
   binding snapshot for that resource. An absent position attribute is valid
   only with an empty operand segment, keeping existing textual fixtures and
   callers valid.

   Add one `ExposeHostBackendLiteralUses` pass whose first and only supported
   role is `static_resource_index`. It runs after async-runtime preparation and
   immediately before `DiscoverHostRematerialization`, while dispatch resource
   provenance and the static kernel binding snapshot are still available. It
   must not consume `vrc.unity_static_tensor_ids`: that compatibility attribute
   is currently attached later by Unity buffer binding, after host-storage
   discovery. Instead, derive each index from the dispatch resource's verified
   static provenance/snapshot. Later binding must cross-check its compatibility
   attribute against the explicit operands rather than silently replacing
   them.

   The pass first validates and plans every dispatch, then mutates any of them.
   For each static resource position it inserts one dedicated `arith.constant`
   immediately before the dispatch, tagged with
   `vrc.host_literal_role = "static_resource_index"` and a generated schema
   version, then rebuilds or updates the dispatch operand segments without
   changing result identity or unrelated attributes. A module-level
   `vrc.host_backend_literal_use_version = 1` records completion. Running the
   pass again on an exact complete representation is a no-op; absent version
   with pre-existing generated operands, partial coverage, duplicate or stale
   positions, a mismatched literal/provenance pair, or an unsupported version
   is a transactional failure.

   Update `DiscoverHostRematerialization` to classify the new exact operand
   sites by the tagged backend value role. Bump the versioned use-site schema
   rather than making the role silently optional: ordinary source operands use
   `source_operand`, while these sites use `static_resource_index`. The stable
   site still names the dispatch operation and real operand index, so the
   existing typed scratch instance, simultaneous consumer group, candidate
   coloring, selected placement, and materialization path applies unchanged.
   Remove the old attribute-only `static_tensor_index` inventory entry for an
   exposed resource; the explicit constant definition and operand site are the
   single source of truth and must not be counted twice. Discovery rejects a
   generated backend literal whose role, version, dispatch segment, or
   provenance no longer agrees.

   `LowerScalar` obtains a static texture-array index only from the explicit
   operand slot when the segment is present; it must not call
   `immutableSlot()` for that use. Candidate-only compilation remains resident
   and produces equivalent behavior, while an allocator-selected operand may
   already be a `udonhl.literal_materialize` result using its fixed writable
   color. The native-test dispatch adaptor deliberately excludes these
   lowering-only operands from the native runtime call. Structural and
   identity validation belongs to the shared exposure/discovery/materialization
   contract and the Udon consumer boundary; after that contract succeeds, the
   native adaptor preserves its runtime ABI by forwarding Device resources
   only.

   Audit every programmatic `vrc.dispatch` construction when adding the sized
   segments. The current manual `OperationState` builders in
   `OpenCLAOTIntrinsics.cpp`, `RuntimeIntrinsics.cpp`, and the two recovery
   builders in `vrccl.cpp` must emit the three segment sizes with a zero final
   segment. Generated builders, adaptors, `DispatchOp::verify`, native-test
   lowering, boundary/manifest readers, static binding/resource
   materialization, Unity binding, scalar lowering, and textual fixtures must
   continue to use `getResources()` only for Device buffers and the new
   accessor only for backend indices. `MaterializeStaticDispatchResources`
   mutates resource operands by their descriptor-relative positions and must
   remain inside the resource segment. Register the new pass in `Passes.td`,
   the VRC conversion library, and the focused frontend test target. Generic
   rewrites that preserve the whole operation must preserve the segment sizes
   and optional position attribute.

   Develop this slice test-first in small commits: (a) add the backward-
   compatible ODS segment, verifier checks, builder updates, and old-syntax
   parse/print tests with no behavior change; (b) add the transactional,
   idempotent exposure pass and focused generic MLIR tests; (c) connect role-
   aware discovery and remove attribute-only double counting; (d) make native
   and Udon lowering consume the explicit segment and add equivalence tests;
   (e) only then place the pass in production. A non-ONNX fixture with two
   static read resources and one dynamic/write resource must prove sorted exact
   positions, two role-bearing use sites at the same dispatch phase, distinct
   simultaneous writable candidates, deterministic repeated output, and no
   site for the dynamic resource. Negative tests cover missing or duplicate
   positions, out-of-range positions, non-i32 operands, stale provenance,
   partial/idempotence schemas, and version mismatches. A manually selected
   placement must reach Udon Assembly through the chosen writable color and
   avoid the old resident index slot; candidate-only output must retain native
   value semantics. This is Step 4 coverage only: automatic selection and the
   exact 512-address acceptance remain Step 5.

   Implementation status: slices (a), (b), and (c) are implemented; slice (a)
   is recorded by `2eb8fae`, and role-aware discovery by `bea0007`.
   `vrc.dispatch` now has the backward-compatible sized
   operand segment, and the standalone transactional
   `vrc-expose-host-backend-literal-uses` pass exposes verified static-tensor
   indices without ONNX or model-specific logic. Role-aware discovery is also
   implemented with use-site schema v3: generated indices carry the
   `static_resource_index` value role, participate in the existing simultaneous
   scratch coloring, and replace the legacy attribute-only inventory for an
   exposed module. The pass is intentionally not in the production pipeline
   yet. Slice (d) is split so each backend contract remains independently
   testable: (d1) makes Udon lowering consume the explicit allocated operand
   slot while retaining legacy fallback only for unexposed IR; (d2) adds the
   native ABI exclusion/equivalence regression without changing native runtime
   arguments. Slice (d1) is recorded by `330951e`, and slice (d2) by
   `d19375c`: the native-fragment regression proves that adding a
   lowering-only static-resource index yields byte-identical canonical native
   IR, preserves the fixed three-argument native dispatch ABI, and leaves the
   existing executed results unchanged. Slices (a)-(d) are complete;
   production wiring slice (e) remains.
5. Extend `AllocateHostStorage` with a deterministic pressure/cost selection
   between resident and `rematerialize_i32` placement. Emit complete selected
   recipe data: exact resident seed bindings, writable color, required externs,
   helper scratch, and generated version. Test the non-ONNX static-resource
   pressure fixture at the exact 512-address boundary; source-only fixtures are
   insufficient for this gate.

   The first Step 5 slice accepts a versioned
   `vrc.host_storage_pressure_request` containing exact i32
   `measured_data_slots`, `data_slot_capacity`, and
   `required_resident_slot_savings = measured - capacity`. Absence of the
   request preserves candidate-only output. The selector may initially choose
   only complete literal-key groups whose uses are all explicit
   `static_resource_index` sites, whose i32 recipe seeds already have resident
   allocations, and whose key is not itself retained as another candidate's
   recipe seed. Every instance for a selected key is emitted through the
   existing MaterializeHostStorage selection schema, while the reported
   guaranteed saving is one resident slot per selected key.

   This request expresses resident pressure, not proof that the final Udon
   address table now fits. Activating writable colors and generated operations
   can change other address categories. The driver must therefore re-run the
   exact final capacity probe after materialization; Step 6 owns stable,
   monotonic retry and final acceptance. The initial deterministic ranking is
   deliberately limited to unweighted straight-line recipe bytes: 20 bytes
   for the copy plus 32 bytes per nibble shift and nonzero lower-nibble OR,
   multiplied by the selected key's instance count, with stable instance ID
   and literal key tie-breaks. Shared extern/address marginal cost and loop
   weight remain required refinements before selection expands to general
   literal roles. A focused end-to-end test must cover request absence,
   deterministic selection, schema rejection, insufficient safe candidates,
   direct MaterializeHostStorage consumption, and final lowering. A generic
   two-dispatch fixture now proves the essential net effect: two distinct
   resident static indices reuse one non-overlapping writable color, reducing
   the final address count by one; with an identical final reserve the selected
   program fits exactly 512 addresses while candidate-only lowering fails at
   513. Fresh production generation and remeasurement remain acceptance gates
   before production activation.
6. Add `VerifyHostStorageCapacity` as an independent pass and connect
   discovery, allocation, materialization, verification, then scalar lowering
   in production order. It reports exact category totals and replacement
   candidates and must never select a fallback itself. If measured costs
   require a retry, replace only compiler-generated placement and require
   monotonic/stable convergence.
7. Remove the superseded per-feature integer literal policies only after the
   general backend-use path covers them. This includes
   `unity_buffer_id_rematerialization`, raw-address-cache, spill-index-cache,
   provisional `static_tensor_id_*` allocation, and hidden decimal
   `SystemInt32` allocation through `immutableSlot()`.
8. Add bit-exact i1, f32, and i64 recipes. Strings, Unity references,
   late-bound code addresses, and ABI/state constants remain shared/fixed until
   separately modeled and verified.
9. Continue unified allocation of live-across-suspend values, resume PCs, and
   internal async object fields; delete superseded raw-frame generation and
   incidental symbol-name tests.
10. Run native/compiler integration and generated-U# MNIST numerical
    validation.
11. Generate the fresh real TinyLlama artifact, verify the exact final address
    count, execute generated U# prefill, and compare all 128,000 logits and 44
    present K/V outputs with the ORT oracle.
12. Run the mandatory pre-commit gate.

Each step must build and pass its relevant focused tests. Do not accumulate a
large knowingly broken pipeline between steps.

At every step, literal use sites and simultaneous consumers cross the common
`HostStorageProblem`/`HostStorageSolution` boundary. Do not add another model-
or resource-specific placement heuristic to final scalar lowering. In
particular, passing source-only pressure tests is not evidence that the real
TinyLlama blocker is fixed: fresh production generation must show that late
backend integer requests have become explicit selected use sites and no longer
consume one resident slot per distinct value.

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
