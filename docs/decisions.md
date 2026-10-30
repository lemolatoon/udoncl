# Design decisions

Source: ../PLAN.md. User amendments: LLVM/MLIR **23** baseline; Windows host
build support. Exact starting pin: LLVM/MLIR/Clang 23.1.4, recorded with a commit.

- UdonSharp is the application entry; annotated C exports form a library. No automatic main().
- Host ABI is ILP32 little-endian with a genuine target definition pending. Existing host triples must not masquerade as the final target.
- Host pointers, device byte addresses and opaque handles are distinct 32-bit domains.
- Comprehensive LLVM → high-level MLIR recovery is required, in the independent `VRCLLVMToHighLevelMLIR` CMake library, shared by host/device. Selective-only recovery is not the completed design.
- Use standard MLIR operations where possible. Custom catalog is closed: udonhl 22, udonasm 9, vrc 13, vrcgpu 13. No pseudo CALL or extra arithmetic ops.
- UdonAsm matches VM opcodes 0,1,2,4,5,6,7,8,9. Directives, heap declarations, labels and exports are metadata, not new ops.
- C runtime owns OpenCL contracts and handle/queue logic; __vrc_* intrinsics express platform boundaries.
- One execution context and in-order queue per library instance; busy calls are rejected. Async continuation state outlives the VM event stack. Reset/Dispose generations reject stale callbacks; completion clears busy before notification.
- GPU memory is logically raw bytes, physically multiple textures, with allocation-wide typed optimization only when proven safe. Partial writes preserve other bytes; source and destination textures differ.
- Fragment passes have one render target. Reject unproven scatter, atomics and divergent barriers. Local memory/barriers need dependence proofs and cost-aware multipass conversion.
- MVP requires SAXPY and full multi-work-group float32 sum reduction including __local/barrier, final readback and resumed C/U# completion.
- CLI emits uasm/shader/manifest/debug data. Unity importer generates assets and typed wrappers, preserves GUIDs and retains valid assets on failure. No silent private SDK reflection dependency.
- Picolibc / full ISO C11 libc is a later coverage goal, not a claim of MVP completeness. Unsupported features produce explicit diagnostics.
- SDK/client execution is required for runtime guarantees. Documentation lookup and host parser tests cannot pass G0/G1 VM semantics.
- P0 gates dependent work. SDK/client unavailable here: retain reproducible host work and mark runtime gates pending, without inventing results.
- Linux/macOS/Windows use the same CMake targets and portable Python tests. Windows compiler host support is separate from the ILP32 emitted C target ABI.

The full normative operation and ABI contracts remain in PLAN.md §§3–12.

Build conventions: CMake 3.30 minimum (Windows Ninja depfile fixes), tested with
4.2.3. Target usage requirements and explicit PUBLIC/PRIVATE scopes carry
headers, definitions and C++17; TableGen gets explicit EXTRA_INCLUDES.
Presets keep machine-specific compiler choices out of the project; local overrides
belong in ignored CMakeUserPresets.json. Compile databases are generated for clangd.
See https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html and
https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html.

SDK discovery: explicit VRC_LLVM_ROOT overrides auto-discovery; otherwise prefer
.deps/llvm23 over system packages and stale LLVM_DIR/MLIR_DIR cache entries.
Both packages must come from the chosen prefix. An incomplete local prefix
fails configuration instead of mixing local LLVM with system MLIR.

Raising is implemented as MLIR Passes following upstream conventions:
TableGen Passes.td, generated declarations/base/registration, conversion patterns,
ConversionTarget and applyFullConversion. The first scalar Pass rejects unsupported
semantics. A registered pipeline composes upstream mem2reg, scalar raising and
upstream CFG-to-SCF. Broader memory, affine, GPU and barrier recovery remain part
of the agreed design. The importer captures frontend metadata before translation;
its schema v1 sidecar includes canonical LLVM IR and a SHA256 source fingerprint.

Fixed local-memory recovery uses a separate TableGen MLIR module Pass. It traces
alloca/GEP provenance, checks LLVM/DataLayout stride against homogeneous memref
storage and proves bounds, alignment, nonescape and definite initialization.
Nested arrays flatten to one dimension so LLVM cross-subarray GEP semantics are
preserved. Only structurally proven constant/masked/counted-loop offsets are
accepted. Unsupported byte aliasing, pointer API/escape and volatile/atomic
semantics remain diagnostics, not a change to the agreed C memory model.
The original LLVM layout is retained as vrc.llvm_data_layout; LLVM layout queries
and consistency checks supplement DLTI. This does not claim the final host ABI.

The initial paired boundary lowering keeps the planned Host ordering:
func.call → udonhl.call → substitution of a typed internal dispatch intrinsic
with vrc.dispatch. C/OpenCL API names are not treated as MLIR operations.
The Device proof is deliberately restricted to one invocation, one complete
f32 output store and pure straight-line scalar arithmetic. It lowers to
vrcgpu resource/pass/version operations and direct ShaderLab/HLSL emission.
A single manifest connects kernel identity, argument order, scalar payload,
resource bindings and launch dimensions. Its request checker proves structural
consistency only; it is not a live runtime handle table or GPU executor.
Assets remain marked ready_for_unity=false until assembly/SDK/runtime gates pass.

The source boundary example compiles real C/OpenCL C to LLVM and runs registered
MLIR recovery Passes. A versioned AOT config supplies symbol roles and launch
bounds; it does not supply a replacement kernel body. OpenCL metadata and a
single direct f32 store prove the bounded output memref. Host uint32 handles
become opaque IR types only after proving a direct descriptor/resource forwarding
function with no arithmetic or additional handle callers. The temporary SPIR
Host layout is explicitly a compiler probe, not the final ILP32 TargetInfo.

Normal CI installs pinned binary archives instead of building LLVM on cache
misses. Linux retains the official 23.1.4 distribution already used locally.
Windows uses Portable MLIR Toolchain's 2026.10.28 Release/noassert LLVM+MLIR SDK;
only the executable and resource files of official Clang are added. The matching
SDK headers/libraries/CMake packages are kept together. URLs and SHA256 are
locked locally; upstream setup scripts or moving manifests are not executed.
Receipts and required SDK components are validated on cache restore. Source
builds remain an explicit local option, with no CI source-build fallback.

Build parallelism follows the CPUs available to the process rather than a fixed
two-job limit. CI and Make use the portable Python CMake build entry point; the
optional LLVM source build uses the same CPU detection by default. Ninja's
native automatic parallelism applies when invoking the build preset directly.
