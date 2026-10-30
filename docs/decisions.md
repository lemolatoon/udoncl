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
