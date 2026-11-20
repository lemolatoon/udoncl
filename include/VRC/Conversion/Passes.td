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
def ValidateONNXHostGraphPass : Pass<"udoncl-validate-onnx-host-graph", "mlir::ModuleOp"> {
  let summary = "Validate the portable ONNX Host graph before materialization";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
def PlanONNXHostGraphPass : Pass<"udoncl-plan-onnx-host-graph", "mlir::ModuleOp"> {
  let summary = "Plan ONNX Host storage and backend selection before materialization";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
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
  let dependentDialects = ["::vrc::udonhl::UdonHLDialect", "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
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
def LowerGatherDevicePass : Pass<"vrc-lower-gather-device", "mlir::ModuleOp"> {
  let summary = "Lower proven gather kernels with scalar loops to fragment-safe resources and passes";
  let dependentDialects = ["::vrc::vrcgpu::VRCGPUDialect", "mlir::arith::ArithDialect"];
}
#endif
