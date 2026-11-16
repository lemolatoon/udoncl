# Unified Host Storage Allocator Plan

## 1. Purpose

UdonCL allocates every Host-side SSA value through one model-independent
storage allocator. Individual lowering passes describe semantic requirements;
they do not independently assign Udon heap slots, spill indices, constant
pools, or scratch storage.

The pinned Worlds SDK 3.10.5 contract is:

- heap addresses and `PUSH` operands are `uint32`;
- the default Udon Assembly path creates a 512-entry heap;
- the supported heap maximum is 1,048,576 entries;
- UdonSharp already assembles with an explicitly sized heap through a custom
  `IUdonHeapFactory`.

UdonCL therefore uses 1,048,576 as the physical Host-address limit and
assembles generated programs with their exact required capacity. This is an
SDK-version-specific limit that must be revalidated when the pinned SDK
changes. The allocator is general compiler infrastructure; MNIST and
TinyLlama are acceptance consumers only.

## 2. Decisions

### 2.1 One value-centric allocation problem

Every allocatable SSA value contributes one request containing:

- a stable value identity and physical Udon type;
- liveness and interference;
- a deterministic priority;
- whether its identity is externally observable or fixed;
- legal placement alternatives aggregated from every definition and use;
- shared resources required by each alternative.

All passes submit requests through this interface. Only the allocator chooses
placements and physical slots. Materialization mechanically implements the
selected plan.

### 2.2 Legal alternatives

A complete legal alternative can be:

- a reusable direct typed heap slot;
- a fixed ABI or externally visible slot;
- an immutable pooled constant slot;
- rematerialization through shared typed scratch and seed resources;
- raw memory required by C pointer or aggregate semantics.

Each alternative records its kind, cost, deterministic identity, shared
dependencies, physical-address cost, and the definitions and uses for which it
is legal. Legality is intersected across all definitions and uses of the
logical value. If one use cannot consume a placement, that placement is not
legal for the value. Partial rewrites are forbidden.

### 2.3 Required resources and optional placements

`required` applies to physical resources that semantics or a selected
placement truly requires. It does not mean every SSA value gets a permanent
dedicated slot.

- A public U# binding variable is a required fixed slot.
- A selected direct placement requires one typed color during its live range.
- Rematerialized constants may share required seed and scratch slots.
- Equal immutable values may share one typed constant-pool entry.
- A raw backing array is required only by addressable C memory semantics.

Shared dependencies are first-class allocator objects, so multiple values and
alternatives can use one physical resource.

### 2.4 Priority and objective

Priority chooses among legal solutions; it cannot override correctness.
Absolute requirements use fixed or required resources. The allocator
optimizes lexicographically:

1. preserve semantics and satisfy required resources;
2. remain within 1,048,576 physical addresses;
3. satisfy simultaneous-operand and interference constraints;
4. minimize estimated runtime work;
5. minimize generated code and constant-storage size;
6. use stable identities as deterministic tie-breakers.

### 2.5 No capacity-driven spill strategy

The old 512-address workaround put ordinary and live-across-suspend values in
typed spill arrays. That is no longer a placement strategy.

Both kinds of value use the same typed direct-slot coloring over complete live
ranges. Non-interfering values reuse a slot. Suspend liveness extends a live
range; it does not change C value semantics or turn a scalar into an array
element.

Arrays and raw storage remain only where source memory semantics, public ABI,
or an explicit runtime contract requires them. They are not a fallback for
address pressure.

### 2.6 Constant pooling remains

Equal immutable constants of the same physical type share a resident slot when
that alternative wins. Constants may instead be rematerialized when every use
permits it and shared seed/scratch resources are cheaper.

Source constants, generated literals, static-resource indices, and other
backend literals enter the same request model. Roles may change legality,
priority, and materialization recipes, but no role owns an independent slot
allocator or pressure selector.

`PUSH` pushes a heap address, not a scalar immediate. A literal therefore
needs a resident slot or a legal runtime materialization recipe unless a
specific opcode genuinely supports a value immediate.

## 3. MLIR contract

### 3.1 Discovery

Discovery emits versioned requests without mutating storage. A request records:

- value ID, physical type, and priority;
- fixed or external identity;
- liveness and interference identity;
- legal alternatives;
- per-alternative dependencies and cost;
- rejected alternatives and the rejecting use.

Static OpenCL analysis contributes resource identity and use semantics only.
ONNX lowering contributes ordinary Host values and fixed public ABI
requirements only. Neither assigns Host slots.

### 3.2 Selection

One pass normalizes every request into a module-wide problem, selects
alternatives, colors direct slots by type and interference, allocates fixed and
shared resources, and verifies the exact address count.

Its deterministic versioned report records priority, aggregated legality,
selected and rejected alternatives, dependencies, assigned slots, declared
data slots, distinct `EXTERN` strings, total addresses, and the 1,048,576
limit.

### 3.3 Materialization

Materialization consumes the selected plan once. It may attach direct or fixed
slots, merge equal constants, emit complete rematerialization recipes, create
selected shared resources, and implement suspend-crossing copies. It must not
rerun allocation, invent a fallback, partially rewrite a value, or recognize a
model.

### 3.4 Selected-plan authority

Once `vrc.host_writable_allocation_version` and
`vrc.host_direct_color_allocations` are present, they are the only authority
for ordinary SSA placement in final Udon lowering. Consumption is independent
of whether the module contains `udonhl.literal_materialize` operations.

The consumer reconstructs stable value identities only to validate the plan
and map selected colors to SSA values. It must reject a partial schema, an
unknown or duplicate value identity, a physical-type mismatch, an unsupported
placement, an interfering color, or an allocation that omits an ordinary
allocatable SSA value. It must not repair any of these cases by graph-coloring
the missing value again.

Literal-materialization results, fixed ABI fields, descriptor snapshots, and
other resources not yet represented by the ordinary-SSA schema remain
separate explicit categories during migration. Their handling must not provide
an implicit fallback for an ordinary SSA value covered by the schema.

Modules with no writable-allocation schema at all may temporarily use the
standalone-lowering compatibility path while direct pass tests and external
tools migrate. A malformed or half-present schema is never a legacy input.
Production compiler pipelines must always run discovery and allocation first.
The compatibility path is removed after standalone callers use that pipeline;
it is not part of the final contract.

### 3.5 Final lowering and Unity assembly

Udon Assembly lowering consumes only selected placements. Verification
recounts `.data` declarations and distinct anonymous `EXTERN` strings, proves
the result is at most 1,048,576, and emits its exact required capacity.

The Unity Package uses a package-owned custom `IUdonHeapFactory` and stores the
normal `IUdonProgram` in the SDK `SerializedUdonProgramAsset`. It must not
patch SDK files or require user-machine changes.

## 4. Pipeline and ownership

```text
source/ONNX/C Host lowering
  -> semantic and fixed-resource discovery
  -> liveness/interference analysis
  -> allocator request discovery
  -> unified Host storage selection
  -> selected-plan materialization
  -> selected-plan verification
  -> Udon Assembly lowering
  -> exact-capacity Unity assembly and serialization
```

Analysis does not rewrite. Selection does not emit Assembly. Materialization
does not choose placements. Final lowering does not invent storage.

## 5. Migration

Implement and commit in independently testable stages:

1. Extend `HostStorageProblem` with priority, legal alternatives, required and
   shared dependencies, and all-use legality. Keep a temporary compatibility
   adapter only while callers migrate.
2. Introduce the shared pinned maximum 1,048,576 and replace Host heap-only
   512 invariants. Do not change unrelated operator limits equal to 512.
3. Add exact-capacity Unity assembly and a regression that assembles,
   serializes, retrieves, and executes a program requiring more than 512
   addresses.
4. Route ordinary SSA through the unified allocator in three independently
   tested steps:
   1. remove the 16/256 direct-register-budget spill recursion;
   2. submit explicit prioritized requests with a legal direct alternative and
      report the selected alternative and color;
   3. consume that complete report independently of literal materialization,
      reject missing selections, and remove ordinary-SSA recoloring from final
      lowering.
5. Route live-across-suspend values through the same direct allocation and
   remove capacity-driven typed spill arrays and spill-index allocation.
6. Route constants, generated literals, static-resource indices, fixed ABI
   values, and shared scratch/seed resources through the same problem.
7. Remove old resident-savings retries, prefix selectors, spill reports, and
   stale tests after every production caller uses the unified plan.
8. Synchronize the root and consumer PLAN files without duplicating this
   contract.

Every stage starts with general MLIR fixtures and focused tests. No stage may
recognize MNIST, TinyLlama, tensor names, layer counts, or model shapes.

## 6. Tests

### 6.1 Allocator unit tests

- deterministic direct-slot coloring and liveness reuse;
- simultaneous same-type operands receiving distinct slots;
- fixed/external slots never moving;
- priority choosing among otherwise legal alternatives;
- one rejecting use removing an alternative for the whole value;
- shared dependencies allocated once;
- equal immutable constants sharing one pool entry;
- rematerialization seed and scratch sharing;
- live-across-suspend interference without typed spill arrays;
- detailed overflow at 1,048,577 required addresses.

### 6.2 Compiler integration

- C scalar copy semantics across calls and suspend points;
- public ABI slots and generated U# bindings;
- static-resource indices mixed with equal source constants;
- ordinary UdonCL programs with sequential kernels;
- no capacity-driven `udonhl.spill_load`/`spill_store` or spill banks;
- capacity reports matching emitted Assembly.

### 6.3 Unity and consumer acceptance

- greater-than-512 assembly through the package-owned asset path;
- serialization and retrieval preserving exact capacity;
- an exported event accessing addresses above 511;
- MNIST numerical validation through generated U#;
- real TinyLlama generated-U# prefill: 128,000 logits and 44 K/V outputs;
- K/V reuse and the agreed greedy token sequence;
- generation to EOS, token-to-text decoding, chat-template handling, and
  conversation from the public U# surface.

## 7. Completion criteria

- Every Host allocatable value uses the unified request/selection interface.
- No pass independently assigns capacity-driven spill indices or direct slots.
- Constant pooling and legal rematerialization are allocator alternatives.
- Final programs use exact capacity no greater than 1,048,576.
- No SDK file is patched.
- Mandatory precommit passes.
- Generated-U# MNIST remains numerically correct.
- Real TinyLlama completes the agreed conversation path from U#.

## 8. Implementation status

- Stages 1-3 are complete: the value-centric request model, shared
  1,048,576-address limit, and exact-capacity Unity assembly path are in use.
- Stage 4.1 is complete: ordinary SSA coloring has no 16/256 register-budget
  cutoff and does not synthesize typed-array spill loads or stores.
- Stages 4.2 and 4.3 are complete: ordinary SSA requests carry an explicit
  priority and direct alternative; the selected report covers every supported
  Host physical type and is authoritative even without literal
  materialization. A partial, stale, or incomplete report fails instead of
  invoking final-lowering recoloring. Only a module with no allocation schema
  may use the temporary standalone compatibility path.
- Stage 5 next migrates pre-existing live-across-suspend spill operations to
  the same direct-slot allocation, then remove their typed spill banks.
- Stages 6-8 and the TinyLlama generated-U# conversation acceptance path remain
  open.
