#ifndef VRC_BOUNDARY_PASSES
#define VRC_BOUNDARY_PASSES
include "mlir/Pass/PassBase.td"
def LowerHostToUdonHLPass : Pass<"vrc-lower-host-to-udonhl", "mlir::ModuleOp"> {
  let summary = "Lower recovered structured Host control flow and direct C calls to UdonHL";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "::vrc::vrc::VRCDialect", "mlir::cf::ControlFlowDialect", "mlir::arith::ArithDialect"];
}
def LowerHostObjectIntrinsicsPass : Pass<"vrc-lower-host-object-intrinsics", "mlir::ModuleOp"> {
  let summary = "Lower typed Udon string and array accessors to UdonHL";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect"];
}
def LowerONNXToUdonHostPass : Pass<"udoncl-lower-onnx-to-host", "mlir::ModuleOp"> {
  let summary = "Lower typed ONNX dialect operations directly to Udon library calls";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
def SpecializeONNXInputsPass : Pass<"udoncl-specialize-onnx-inputs", "mlir::ModuleOp"> {
  let summary = "Specialize public ONNX tensor inputs from explicit dimension bindings";
  let dependentDialects = ["mlir::func::FuncDialect"];
  let options = [
    Option<"dimensionBindings", "dimensions", "std::string", /*default=*/"",
           "Semicolon-separated symbolic dimension bindings name=value">,
    Option<"inputShapeBindings", "input-shapes", "std::string", /*default=*/"",
           "Semicolon-separated public input shapes name=d0,d1,...">
  ];
}
def ExternalizeONNXGraphConstantsPass : Pass<"udoncl-externalize-onnx-graph-constants", "mlir::ModuleOp"> {
  let summary = "Replace graph-local dense ONNX constants with immutable static tensor symbols";
  let description = [{
    Runs after udoncl-externalize-onnx-initializers and the last constant
    folding pass. Every remaining ranked, static-shape, non-empty dense
    tensor constant (onnx.Constant without an initializer origin and folded
    arith.constant tensors) becomes an immutable static tensor appended to
    udoncl.static_tensors, so a Device operand never relies on an
    uninitialized mutable buffer. The dense value stays inline in the record
    so Host finalization can still fold scalar and shape operands.
  }];
  let dependentDialects = ["mlir::func::FuncDialect"];
  let options = [
    Option<"staticDataDirectory", "static-data-directory", "std::string",
           /*default=*/"", "Directory receiving external static tensor payloads">
  ];
}
def SimplifyONNXEmptyConcatPass : Pass<"udoncl-simplify-onnx-empty-concat", "mlir::ModuleOp"> {
  let summary = "Remove statically empty operands from ONNX Concat operations";
}
def SimplifyONNXViewsPass : Pass<"udoncl-simplify-onnx-views", "mlir::ModuleOp"> {
  let summary = "Remove ONNX view operations that keep the row-major element order";
  let description = [{
    Runs on statically shaped ONNX graphs. A Transpose that only moves unit
    dimensions becomes a Reshape, and a unit-step Slice or an Expand whose
    result shape equals its input shape is replaced by its input. The Host
    planner aliases a Reshape to its operand, so none of these views costs a
    Device dispatch or a buffer.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
}
def FoldONNXRepeatedMatMulOperandPass : Pass<"udoncl-fold-onnx-repeated-matmul-operand", "mlir::ModuleOp"> {
  let summary = "Fold a MatMul right operand repeated by Expand into the left operand rows";
  let description = [{
    Runs on statically shaped ONNX graphs. When the right operand of a MatMul
    is Reshape(Expand(X)) with X[P..., G, 1, K, N] expanded to
    [P..., G, R, K, N] and reshaped to [P..., G*R, K, N], the R left matrices
    of each group multiply the same matrix. The MatMul becomes one over X
    whose left matrices are those R matrices stacked as rows,
    MatMul(A[P..., G, R*M, K], X[P..., G, K, N]), reshaped back to the
    original result. Every element is the same dot product, and the repeated
    operand is never materialized.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
}
def MergeONNXSiblingMatMulNBitsPass : Pass<"udoncl-merge-onnx-sibling-matmulnbits", "mlir::ModuleOp"> {
  let summary = "Merge MatMulNBits nodes that read the same activation into one node";
  let description = [{
    Runs on statically shaped ONNX graphs whose weights are constants. Every
    output column n of com.microsoft::MatMulNBits is computed from row n of
    its packed weights and scales alone, so readers of one activation with
    equal K, bits, block size and other attributes, constant weights and no
    zero points, group index or bias become one node over the concatenated
    rows, followed by one Slice of its last axis per original result. Each
    column is computed from the same operands in the same order as before.
    Groups form in program order and stay within `max-result-words` result
    elements, the largest tensor a Device Slice reads.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
  let options = [
    Option<"maxResultWords", "max-result-words", "int64_t",
           /*default=*/"16384",
           "Largest number of result elements of a merged node">
  ];
}
def ValidateONNXHostGraphPass : Pass<"udoncl-validate-onnx-host-graph", "mlir::ModuleOp"> {
  let summary = "Validate the portable ONNX Host graph before materialization";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
def PlanONNXHostGraphPass : Pass<"udoncl-plan-onnx-host-graph", "mlir::ModuleOp"> {
  let summary = "Plan ONNX Host storage and backend selection before materialization";
  let description = [{
    Records `udoncl.onnx_host_plan` on the module: whether the Device runs the
    whole graph, the fusion group of every node (`device_fusion_roots`, the
    node whose dispatch computes it), the buffer of every tensor, and the
    sequence states that alternate between two banks
    (`sequence_state_banks`, indices into `udoncl.onnx_sequence_states`). Every
    later pass reads these decisions from the plan, so they agree on which
    nodes have a dispatch and which results have a buffer.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
  let options = [
    Option<"fuseDeviceNodes", "fuse-device-nodes", "bool", /*default=*/"true",
           "Fuse trees of per-element and re-indexing Device nodes into one "
           "synthesized kernel each">,
    Option<"bankSequenceState", "bank-sequence-state", "bool",
           /*default=*/"true",
           "Alternate qualifying sequence states between two Device banks "
           "instead of copying the carried rows on every advance">
  ];
}
def MaterializeONNXHostGlobalsPass : Pass<"udoncl-materialize-onnx-host-globals", "mlir::ModuleOp"> {
  let summary = "Materialize planned ONNX Host global storage without consuming the source graph";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::LLVM::LLVMDialect"];
}
def MaterializeONNXHostModelRegistryPass : Pass<"udoncl-materialize-onnx-host-model-registry", "mlir::ModuleOp"> {
  let summary = "Materialize the planned ONNX Host model registry and descriptor accessors";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect"];
}
def AttachONNXHostMetadataPass : Pass<"udoncl-attach-onnx-host-metadata", "mlir::ModuleOp"> {
  let summary = "Attach planned ONNX Host binding, Device graph, and capacity metadata";
}
def FinalizeONNXHostPass : Pass<"udoncl-finalize-onnx-host", "mlir::ModuleOp"> {
  let summary = "Lower portable ONNX library calls to the Udon Host ABI and session exports";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect", "mlir::gpu::GPUDialect", "mlir::LLVM::LLVMDialect", "mlir::math::MathDialect", "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"];
}
def CleanupONNXHostSourcePass : Pass<"udoncl-cleanup-onnx-host-source", "mlir::ModuleOp"> {
  let summary = "Erase finalized ONNX Host source symbols while preserving unrelated symbols";
  let dependentDialects = ["mlir::func::FuncDialect"];
}
def EmitONNXHostMetadataPass : Pass<"udoncl-emit-onnx-host-metadata", "mlir::ModuleOp"> {
  let summary = "Write compact compiler-owned ONNX Host artifact metadata";
  let options = [
    Option<"outputPath", "output", "std::string", /*default=*/"",
           "Metadata JSON output path">
  ];
}
def SubstituteRuntimeIntrinsicsPass : Pass<"vrc-substitute-runtime-intrinsics", "mlir::ModuleOp"> {
  let summary = "Replace typed __vrc_* calls after UdonHL construction; preserve C runtime ownership";
  let dependentDialects = ["::vrc::vrc::VRCDialect"];
}
def LowerOpenCLAOTIntrinsicsPass : Pass<"vrc-lower-opencl-aot-intrinsics", "mlir::ModuleOp"> {
  let summary = "Recover the linked OpenCL AOT runtime backend ABI as typed VRC operations";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect", "mlir::arith::ArithDialect"];
}
def AttachHostAnnotationsPass : Pass<"vrc-attach-host-annotations", "mlir::ModuleOp"> {
  let summary = "Attach the UdonCL source annotations of a recovered Host to its symbols";
  let description = [{
    Consumes the `udoncl.host_annotations` module attribute, which udonclc
    builds from the annotations of the linked Host LLVM module before it is
    imported (<udoncl/export.h>), and marks the recovered symbols:
    `exports` -> `udoncl.export`, `while_busy` -> `udoncl.export_while_busy`,
    `constructors` -> `udoncl.static_constructor`, `destructors` ->
    `udoncl.static_destructor`, and `parameter_names` ->
    `udoncl.parameter_names` on a function whose argument count still
    matches. A named function that recovery lost is an error.
  }];
  let dependentDialects = ["mlir::func::FuncDialect"];
}
def DiscoverStaticDeviceResourcesPass : Pass<"vrc-discover-static-device-resources", "mlir::ModuleOp"> {
  let summary = "Discover deterministic OpenCL allocation contexts without lowering them";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::cf::ControlFlowDialect", "mlir::scf::SCFDialect"];
}
def PlanStaticDeviceResourceContextSpecializationPass : Pass<"vrc-plan-static-device-resource-context-specialization", "mlir::ModuleOp"> {
  let summary = "Plan context specialization for shared static allocation sites";
}
def SpecializeStaticAllocationContextsPass : Pass<"vrc-specialize-static-allocation-contexts", "mlir::ModuleOp"> {
  let summary = "Clone root-distinct internal allocation paths into singleton contexts";
  let dependentDialects = ["mlir::func::FuncDialect"];
}
def InferStaticDeviceResourceRequirementsPass : Pass<"vrc-infer-static-device-resource-requirements", "mlir::ModuleOp"> {
  let summary = "Infer keyed capacity and memory requirements for static allocation contexts";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
def PropagateDeviceResourceProvenancePass : Pass<"vrc-propagate-device-resource-provenance", "mlir::ModuleOp"> {
  let summary = "Propagate context-sensitive static device resource identities through Host SSA";
  let description = [{
    Also records, as `vrc.static_device_global_cell_provenance`, the
    allocation identities every tracked Host global cell may hold: the union
    over all writes and call contexts, whether it may be null and whether it
    may hold an unknown handle.
  }];
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect", "mlir::cf::ControlFlowDialect", "mlir::arith::ArithDialect", "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"];
}
def ClassifyONNXPublicDeviceResourcesPass : Pass<"vrc-classify-onnx-public-device-resources", "mlir::ModuleOp"> {
  let summary = "Classify static allocation identities bound to public ONNX inputs and outputs";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
def ClassifyStaticPublicDeviceResourcesPass : Pass<"vrc-classify-static-public-device-resources", "mlir::ModuleOp"> {
  let summary = "Classify frontend-declared public resources from generic allocation provenance";
}
def AnalyzeStaticDeviceResourcePersistencePass : Pass<"vrc-analyze-static-device-resource-persistence", "mlir::ModuleOp"> {
  let summary = "Classify static allocation identities that must survive an exported Host call";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect"];
}
def BuildStaticDeviceResourceInventoryPass : Pass<"vrc-build-static-device-resource-inventory", "mlir::ModuleOp"> {
  let summary = "Merge keyed allocation requirements, persistence and optional public bindings";
}
def ResolveStaticKernelBindingsPass : Pass<"vrc-resolve-static-kernel-bindings", "mlir::ModuleOp"> {
  let summary = "Capture immutable call-by-value Device dispatch bindings";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "::vrc::vrc::VRCDialect", "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect"];
}
def ResolveStaticTransferBindingsPass : Pass<"vrc-resolve-static-transfer-bindings", "mlir::ModuleOp"> {
  let summary = "Capture context-sensitive static resources at OpenCL transfer call sites";
  let dependentDialects = ["mlir::func::FuncDialect"];
}
def ResolveStaticPublicResourceAliasesPass : Pass<"vrc-resolve-static-public-resource-aliases", "mlir::ModuleOp"> {
  let summary = "Resolve public Device endpoints to compiler-assigned Unity Textures";
}
def ReconcileStaticUnityBuffersPass : Pass<"vrc-reconcile-static-unity-buffers", "mlir::ModuleOp"> {
  let summary = "Reconcile configured Unity buffers with resolved static public resources";
}
def PlanStaticUnityBufferLayoutPass : Pass<"vrc-plan-static-unity-buffer-layout", "mlir::ModuleOp"> {
  let summary = "Plan a canonical Unity buffer layout from keyed Texture assignments";
}
def AllocateUnityTexturesPass : Pass<"vrc-allocate-unity-textures", "mlir::ModuleOp"> {
  let summary = "Assign one deterministic Unity Texture to each keyed allocation resource";
}
def MaterializeStaticDispatchResourcesPass : Pass<"vrc-materialize-static-dispatch-resources", "mlir::ModuleOp"> {
  let summary = "Replace proven dispatch allocation operands with assigned mutable Texture resources";
  let dependentDialects = ["::vrc::vrc::VRCDialect"];
}
def MaterializeStaticLogicalBuffersPass : Pass<"vrc-materialize-static-logical-buffers", "mlir::ModuleOp"> {
  let summary = "Replace constant logical buffers with their reconciled mutable Texture resources";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::arith::ArithDialect"];
}
def PlanStaticPhysicalAllocationSpecializationPass : Pass<"vrc-plan-static-physical-allocation-specialization", "mlir::ModuleOp"> {
  let summary = "Plan private call-path specialization to compiler-assigned physical buffers";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::func::FuncDialect"];
}
def SpecializeStaticPhysicalAllocationPathsPass : Pass<"vrc-specialize-static-physical-allocation-paths", "mlir::ModuleOp"> {
  let summary = "Clone planned physical allocation paths into resource-owned private functions";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::func::FuncDialect"];
}
def MaterializeStaticPhysicalAllocationsPass : Pass<"vrc-materialize-static-physical-allocations", "mlir::ModuleOp"> {
  let summary = "Replace proven resource-owned physical allocations with assigned Unity buffers";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::arith::ArithDialect"];
}
def MaterializeStaticPhysicalResolvesPass : Pass<"vrc-materialize-static-physical-resolves", "mlir::ModuleOp"> {
  let summary = "Replace proven physical-buffer resolves with assigned Unity buffer tokens";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "::vrc::udonhl::UdonHLDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
def LowerCheckedAssignedBufferResolvesPass : Pass<"vrc-lower-checked-assigned-buffer-resolves", "mlir::ModuleOp"> {
  let summary = "Lower checked assigned-buffer resolves to explicit mutable cl_mem state checks";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect"];
}
def MaterializeFixedPhysicalReleasesPass : Pass<"vrc-materialize-fixed-physical-releases", "mlir::ModuleOp"> {
  let summary = "Materialize fixed Unity pool physical releases as successful no-ops";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::arith::ArithDialect"];
}
def EliminateDeadPhysicalAllocationFallbacksPass : Pass<"vrc-eliminate-dead-physical-allocation-fallbacks", "mlir::ModuleOp"> {
  let summary = "Erase unreachable private physical-allocation fallback call chains";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::func::FuncDialect"];
}
def AssessStaticUnityResourceLoweringPass : Pass<"vrc-assess-static-unity-resource-lowering", "mlir::ModuleOp"> {
  let summary = "Assess static resource operands separately from canonical Unity pool replacement";
}
def ReportStaticUnityResourceLoweringPass : Pass<"vrc-report-static-unity-resource-lowering", "mlir::ModuleOp"> {
  let summary = "Report why static Unity resource lowering is or is not applied";
  let description = [{
    Emits a remark with the state of the static Unity resource lowering
    (`vrc.static_unity_resource_lowering_readiness`) and one remark per
    blocker: a public allocation without a public alias, a configured public
    buffer no static public resource resolves, a static public resource no
    configured buffer uses, and every remaining dynamic allocation, release,
    resolve, transfer or unresolved Device operand. The module is unchanged.
  }];
}
def LowerAwaitToContinuationsPass : Pass<"vrc-lower-await-to-continuations", "mlir::ModuleOp"> {
  let summary = "Split a single typed await into explicit UdonHL frame, suspend and resume blocks";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "::vrc::vrc::VRCDialect"];
}
def LowerHostToAsyncPass : Pass<"vrc-lower-host-to-async", "mlir::ModuleOp"> {
  let summary = "Convert the may-suspend Host call graph to MLIR Async without forced inlining";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::async::AsyncDialect", "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect"];
}
def InsertUdonCooperativeCheckpointsPass : Pass<"vrc-insert-udon-cooperative-checkpoints", "mlir::ModuleOp"> {
  let summary = "Split long Udon-only synchronous Host regions with static cooperative checkpoints";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
def ConfigureUnityExportsPass : Pass<"vrc-configure-unity-exports", "mlir::ModuleOp"> {
  let summary = "Select and name the U# exports of a Unity library Host";
  let description = [{
    Consumes the Unity library configuration that udonclc attaches as the
    `udoncl.unity_library_config` module dictionary (the JSON configuration
    file converted key by key). Schema 1 lists every export with its id,
    public name and parameter names. Schema 2 publishes every function marked
    `udoncl.export` under the PascalCase of its symbol, with parameters named
    after `udoncl.parameter_names` (arg<i> without one), unless an
    `export_overrides` entry names it. A function also marked
    `udoncl.internal_export` (the generated ONNX binding) is a root of the
    Host but never a U# export.

    The C `__attribute__((constructor))` function (`udoncl.static_constructor`,
    at most one) becomes the static constructor; one that is not also an
    export is configured as one marked `vrc.internal_static_constructor`, so
    the analyses see it as a root. An export that is a C
    `__attribute__((destructor))` function (`udoncl.static_destructor`) is
    omitted. The `static_initialization` configuration key names a
    constructor and omitted exports instead; the two do not combine.

    Each U# export receives `vrc.export_id`, `vrc.public_name`,
    `vrc.parameter_names` and, when allowed while busy,
    `vrc.allowed_while_busy`; the module receives `vrc.binding_namespace`,
    `vrc.binding_class` and `vrc.static_initialization`. With
    `finalize-reachability`, functions no export reaches are erased, reached
    helpers become private and the module is marked `vrc.target = "vrc-pc"`.
  }];
  let options = [
    Option<"finalizeReachability", "finalize-reachability", "bool", "true",
           "Erase unreachable functions and privatize reached helpers">
  ];
  let dependentDialects = ["mlir::func::FuncDialect"];
}
def EvaluateHostConstructorsPass : Pass<"vrc-evaluate-host-constructors", "mlir::ModuleOp"> {
  let summary = "Evaluate static Host constructors at build time and commit their memory as initializers";
  let description = [{
    A function marked `vrc.static_constructor` has run exactly once before any
    other export of a closed-world artifact, so the Host memory it leaves is a
    function of the build. The pass interprets it over the Host IR (the MLIR
    counterpart of GlobalOpt's `llvm.global_ctors` evaluation), rewrites the
    initializer of every global it wrote, with relocations for words that
    hold an address or a Unity buffer token, and reduces its body to its
    result.

    A constructor that cannot be evaluated, or that does not return success,
    is an error at the blocking operation; nothing is committed.

    A constructor marked `vrc.internal_static_constructor` (one that is not
    also an export) leaves the exports once it has run and becomes private;
    a Device command it would leave behind for U# to run is an error.
  }];
  let options = [
    Option<"stepBudget", "step-budget", "uint64_t", "268435456",
           "Operations the evaluation of one constructor may execute">
  ];
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "::vrc::vrc::VRCDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
}
def FoldImmutableHostMemoryPass : Pass<"vrc-fold-immutable-host-memory", "mlir::ModuleOp"> {
  let summary = "Fold loads from raw Host memory that no reachable operation can store to";
  let description = [{
    A raw global of a linked Host program is closed-world state: every store
    that can reach it is an operation of the module, or the completion of a
    readback the module issues. A load from a byte range no store can touch
    reads the initializer under every interleaving of export events and
    continuations, so it is replaced by a constant, by the address a
    relocated word names, or by the Unity buffer a token word names. A global
    nothing stores to becomes `constant`.

    The analysis is flow-insensitive and field-sensitive, in the style of
    LLVM GlobalOpt: addresses are (object, offset) sets with strided offsets,
    memory keeps what stored values may point to, and a store through an
    address the analysis lost clobbers exactly the objects whose address
    escaped it. Run it before async conversion, alternating with scalar
    simplification: a folded load can make the code holding a store
    unreachable. `vrc.immutable_host_memory` on the module reports what one
    run folded.
  }];
  let options = [
    Option<"explain", "explain", "bool", "false",
           "Print how many loads were not folded, and why, to stderr">
  ];
  let statistics = [
    Statistic<"foldedLoads", "folded-loads", "Loads replaced by their initializer">,
    Statistic<"constantGlobals", "constant-globals", "Globals proven never stored to">
  ];
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "::vrc::vrc::VRCDialect", "mlir::arith::ArithDialect"];
}
def PromoteHostGlobalsToUdonVariablesPass : Pass<"vrc-promote-host-globals-to-udon-variables", "mlir::ModuleOp"> {
  let summary = "Hold scalar Host globals whose address is unobservable in named Udon variables";
  let description = [{
    A `udonhl.global` of one i32 or pointer whose address is used only as the
    pointer of whole-object loads and stores of its own type is marked
    `vrc.udon_variable`. Its accesses become `udonhl.variable_load` and
    `udonhl.variable_store`, which name the symbol instead of an address, and
    the `udonhl.address_of` operations are erased. Scalar Udon lowering gives
    such a global a named heap slot initialized with the global's initializer
    instead of a raw heap cell.

    Globals with any other address use (pointer arithmetic with a run-time
    offset, bulk memory operations, calls, comparisons, a stored address),
    with a partial or retyped access, or with an initializer the Udon
    assembler cannot spell are left in the raw heap. With
    `vrc.debug_compiler_phases` on the module the decision for every global is
    printed to stderr.
  }];
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect"];
  let statistics = [
    Statistic<"numPromoted", "num-promoted", "Globals held in Udon variables">,
    Statistic<"numKeptInRawHeap", "num-kept-in-raw-heap", "Globals left in the raw heap">,
    Statistic<"numAccessesRewritten", "num-accesses-rewritten", "Loads and stores rewritten to variable accesses">
  ];
}
def AssignUnityDispatchMaterialsPass : Pass<"vrc-assign-unity-dispatch-materials", "mlir::ModuleOp"> {
  let summary = "Give every Unity dispatch site its own kernel material";
  let description = [{
    A Unity dispatch binds its kernel arguments as properties of the material
    it Blits with, and a material keeps its properties between Blits. With
    one material per kernel, every dispatch rebinds whatever differs from the
    previous dispatch of that kernel elsewhere in the program. This pass gives
    every `vrc.dispatch` carrying `vrc.unity_dispatch_material_slot` its own
    public material slot `__udoncl_dispatch_material_<n>`, numbered in module
    order, so the scalar Udon lowering can bind fixed arguments once per site
    and remember the values that change.

    The module attribute `vrc.unity_dispatch_materials` lists every material
    slot the dispatches use with the kernel id whose shader it renders:
    `[{slot = "...", kernel_id = k : i32}, ...]`. The Unity importer creates
    one material per entry. With `max-materials` set, dispatches beyond that
    count keep their kernel's shared slot, which is listed once.
  }];
  let options = [
    Option<"maxMaterials", "max-materials", "int64_t", /*default=*/"0",
           "Upper bound of per-site materials (0: one material per site)">
  ];
  let statistics = [
    Statistic<"numSiteMaterials", "num-site-materials", "Dispatch sites given their own material">,
    Statistic<"numSharedMaterials", "num-shared-materials", "Kernel materials still shared by several sites">
  ];
}
def ThreadConstantBranchesPass : Pass<"vrc-thread-constant-branches", "mlir::ModuleOp"> {
  let summary = "Retarget edges whose constant block arguments decide the conditional branch of their successor";
  let description = [{
    A block that only compares its own arguments with constants and branches
    on the result is the recovered shape of a C short-circuit or conditional
    expression. An edge that passes constants for the compared arguments
    already knows the taken successor: it is retargeted to that successor,
    with the block's forwarded arguments replaced by the values the edge
    passes. Edges with a run-time argument keep the block.

    The block must contain nothing but the comparison and the branch, and
    none of its values may be used outside it, so a threaded edge skips no
    effect and breaks no dominance.
  }];
  let dependentDialects = ["mlir::cf::ControlFlowDialect"];
}
def ScheduleUdonDeviceWorkPass : Pass<"vrc-schedule-udon-device-work", "mlir::ModuleOp"> {
  let summary = "Make Udon Device commands enqueue-only and yield by a per-slice work budget";
  let description = [{
    On the Udon target vrc.dispatch, vrc.buffer_copy and vrc.buffer_fill are
    Blits recorded on Unity's in-order graphics queue and complete at issue. A vrc.await of such
    an event is folded to CL_SUCCESS. After every Device command in a
    may-suspend function its static cost is charged to two slice budgets held
    in instance globals; at the former await the Host yields through a
    cooperative checkpoint when either budget is full, which starts a new
    slice. Transfers, readbacks and other events keep their awaits. A command
    whose former await directly follows it is charged at that yield point:
    the check compares the sums it just stored, which need not saturate
    because a full slice yields at once and the yield resets both budgets.

    The Device cost is counted in texel-load units: `fragment-weight` units
    per launched fragment plus one unit per texel load, with loads per
    fragment taken from an optional `vrc.fragment_cost` dispatch attribute
    (evaluated with the site's constant scalar arguments or the declared
    maxima) and otherwise one load per read resource. The Host cost estimates
    the Udon instructions spent binding the command's arguments and running
    the Host operations its block issues since the previous command.
  }];
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "::vrc::vrc::VRCDialect", "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect"];
  let options = [
    Option<"deviceCostBudget", "device-cost-budget", "int64_t",
           /*default=*/"134217728",
           "Device cost units (texel loads) one Udon event may issue before "
           "yielding">,
    Option<"hostCostBudget", "host-cost-budget", "int64_t",
           /*default=*/"16384",
           "Estimated Udon instructions one Udon event may spend issuing "
           "Device commands before yielding">,
    Option<"fragmentWeight", "fragment-weight", "int64_t", /*default=*/"16",
           "Device cost units charged per launched fragment">
  ];
}
def LowerHostAsyncToNativePass : Pass<"vrc-lower-host-async-to-native", "mlir::ModuleOp"> {
  let summary = "Map VRC completion events in the common Host Async contract to Native async values";
  let dependentDialects = ["mlir::async::AsyncDialect", "mlir::func::FuncDialect"];
}
def PrepareUdonAsyncRuntimePass : Pass<"vrc-prepare-udon-async-runtime", "mlir::ModuleOp"> {
  let summary = "Verify the structured Async Runtime subset and assign static Udon frame/resume ids";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::arith::ArithDialect", "mlir::async::AsyncDialect", "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect"];
}
def DiscoverAsyncPersistentStoragePass : Pass<"vrc-discover-async-persistent-storage", "mlir::ModuleOp"> {
  let summary = "Describe live-across-suspend SSA equivalence classes without choosing storage";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect"];
}
def MaterializeAsyncPersistentStoragePass : Pass<"vrc-materialize-async-persistent-storage", "mlir::ModuleOp"> {
  let summary = "Normalize validated async frame aliases to one SSA storage identity before allocation";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect"];
}
def ExposeHostBackendLiteralUsesPass : Pass<"vrc-expose-host-backend-literal-uses", "mlir::ModuleOp"> {
  let summary = "Expose verified backend-only Host literals as typed SSA operands";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::arith::ArithDialect"];
}
def AnalyzeHostStoragePass : Pass<"vrc-analyze-host-storage", "mlir::ModuleOp"> {
  let summary = "Record deterministic logical Host storage placements before physical allocation";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect"];
}
def DiscoverHostRematerializationPass : Pass<"vrc-discover-host-rematerialization", "mlir::ModuleOp"> {
  let summary = "Inventory typed Host literal uses and legal rematerialization recipes";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
def AllocateHostStoragePass : Pass<"vrc-allocate-host-storage", "mlir::ModuleOp"> {
  let summary = "Assign deterministic physical storage to planned Host live ranges";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect"];
}
def AllocateHostLiteralStoragePass : Pass<"vrc-allocate-host-literal-storage", "mlir::ModuleOp"> {
  let summary = "Assign literal storage without changing async spill indices";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect"];
}
def MaterializeHostStoragePass : Pass<"vrc-materialize-host-storage", "mlir::ModuleOp"> {
  let summary = "Materialize allocator-selected Host storage placements";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
def MeasureHostStorageCapacityPass : Pass<"vrc-measure-host-storage-capacity", "mlir::ModuleOp"> {
  let summary = "Measure exact final Udon address pressure without lowering the input module";
}
def FitHostStorageCapacityPass : Pass<"vrc-fit-host-storage-capacity", "mlir::ModuleOp"> {
  let summary = "Transactionally select and materialize Host literals until exact Udon address pressure fits";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
def VerifyHostStorageCapacityPass : Pass<"vrc-verify-host-storage-capacity", "mlir::ModuleOp"> {
  let summary = "Verify fitted Host storage against the exact final Udon address plan";
}
def LowerUdonAsyncRuntimePass : Pass<"vrc-lower-udon-async-runtime", "mlir::ModuleOp"> {
  let summary = "Remove upstream Async/Coro scaffolding after static Udon routing is materialized";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "::vrc::vrc::VRCDialect", "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect"];
}
def LowerDeviceSingleInvocationPass : Pass<"vrc-lower-device-single-invocation", "mlir::ModuleOp"> {
  let summary = "Lower a proven 1-invocation scalar-output GPU kernel to fragment-safe vrcgpu";
  let dependentDialects = ["::vrc::vrcgpu::VRCGPUDialect", "mlir::arith::ArithDialect"];
}
def VerifyHostDeviceBoundaryPass : Pass<"vrc-verify-host-device", "mlir::ModuleOp"> {
  let summary = "Verify kernel ids, descriptors, resource order and launch ABI across host/device";
}
def RaiseLLVMBoundaryDevicePass : Pass<"vrc-raise-llvm-boundary-device", "mlir::ModuleOp"> {
  let summary = "Recover the selected single-invocation OpenCL output pointer as a bounded memref";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::arith::ArithDialect", "mlir::memref::MemRefDialect"];
}
def LinkCBoundaryPass : Pass<"vrc-link-c-boundary", "mlir::ModuleOp"> {
  let summary = "Recover explicit uint32 Host handle roles and link recovered C/OpenCL modules";
  let dependentDialects = ["::vrc::vrc::VRCDialect", "mlir::gpu::GPUDialect"];
}
def LowerHostToNativeTestPass : Pass<"vrc-lower-host-to-native-test", "mlir::ModuleOp"> {
  let summary = "Lower the verified UdonHL Host boundary to native func calls for testing only";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::arith::ArithDialect"];
}
def LowerUdonHLToUdonAsmPass : Pass<"vrc-lower-udonhl-to-udonasm", "mlir::ModuleOp"> {
  let summary = "Allocate typed Udon heap slots and lower scalar Host SSA to the nine real VM opcodes";
  let dependentDialects = ["::vrc::udonasm::UdonAsmDialect", "mlir::memref::MemRefDialect"];
}
def RaiseNativeHostCallsPass : Pass<"vrc-raise-native-host-calls", "mlir::ModuleOp"> {
  let summary = "Expose native C calls as UdonHL while retaining native LLVM memory and ABI semantics";
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::func::FuncDialect"];
}
def LowerNativeHostCallsPass : Pass<"vrc-lower-native-host-calls", "mlir::ModuleOp"> {
  let summary = "Restore native LLVM function/call ABI from the native-only UdonHL adapter";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}
def PrepareNativeDevicePass : Pass<"vrc-prepare-native-device", "mlir::ModuleOp"> {
  let summary = "Wrap the single recovered Device kernel for a native-only shader bundle";
  let dependentDialects = ["mlir::gpu::GPUDialect"];
}
def RaiseGatherDevicePass : Pass<"vrc-raise-gather-device", "mlir::ModuleOp"> {
  let summary = "Prove fixed global-buffer gather bounds, loop bounds and exclusive scalar output ownership";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::memref::MemRefDialect", "mlir::arith::ArithDialect", "mlir::math::MathDialect", "mlir::gpu::GPUDialect"];
}
def ProveSynthesizedGatherKernelsPass : Pass<"vrc-prove-synthesized-gather-kernels", "mlir::ModuleOp"> {
  let summary = "Prove the buffer bounds and output ownership of compiler-synthesized gather kernels";
  let description = [{
    A `gpu.func` kernel marked `vrc.synthesized_kernel` was built directly as
    MLIR, so no LLVM-level gather proof covers it. This pass checks its
    argument, attribute and operation contract, proves by 32-bit integer range
    analysis that every `memref.load` stays inside its input buffer, checks
    that the single top-level `memref.store` writes the output word of the
    current fragment and that every `scf.while` is a bounded counted loop, and
    then marks the kernel `vrc.gather_proven`. Kernels without the marker are
    left untouched.
  }];
}
def AssignSynthesizedKernelIdsPass : Pass<"vrc-assign-synthesized-kernel-ids", "mlir::ModuleOp"> {
  let summary = "Assign kernel ids to the synthesized kernels a Host module carries and make their dispatch declarations self-describing";
  let description = [{
    A Host module may carry `gpu.module`s that hold only `gpu.func` kernels
    marked `vrc.synthesized_kernel`, and dispatch one through an external
    declaration with `vrc.synthesized_dispatch = @gpu_module::@kernel`, whose
    signature is the kernel's argument list with every buffer replaced by an
    i32 token, returning i32.

    Every kernel with a called declaration gets the next free id, starting at
    `first-id` and skipping `reserved-ids` and the ids already present in the
    module, in module order. It is stamped `vrc.kernel_id` and moved into its
    own `gpu.module @device_gpu_<id>`. Each declaration is checked against its
    kernel and its reference is replaced by
    `vrc.device_dispatch = {kernel_id, buffer_count, global_size}` (plus
    `fragment_cost` when the kernel declares `vrc.fragment_cost`), the form
    `vrc-lower-opencl-aot-intrinsics` lowers without a name table. Uncalled
    declarations, kernels nothing dispatches and the emptied carriers are
    erased; a reference that names no synthesized kernel is an error.
  }];
  let dependentDialects = ["mlir::gpu::GPUDialect"];
  let options = [
    Option<"firstId", "first-id", "int64_t", /*default=*/"1",
           "Smallest kernel id to assign">,
    ListOption<"reservedIds", "reserved-ids", "int64_t",
               "Kernel ids that belong to other kernels and are skipped">
  ];
}
def LowerGatherDevicePass : Pass<"vrc-lower-gather-device", "mlir::ModuleOp"> {
  let summary = "Lower proven gather kernels with scalar loops to fragment-safe resources and passes";
  let dependentDialects = ["::vrc::vrcgpu::VRCGPUDialect", "mlir::arith::ArithDialect"];
}
#endif
