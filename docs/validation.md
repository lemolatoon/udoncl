# Validation status

No SDK/client gate has passed. Compiler host and generated runtime are separate.
All pending target tests below are `TODO: verify on target SDK/client`.

| ID | Test | State / acceptance evidence needed |
|---|---|---|
| HOST-01 | LLVM/MLIR 23.1.4 TableGen + four independent dialect targets | PASS on Linux: validation/2026-10-30-host.txt |
| HOST-02 | Nine opcode parser/printer and invalid attributes/unknown ops | PASS on Linux: round-trip + 16 rejection cases; not VM semantics |
| HOST-03 | LLVM import through independent recovery library | PASS scalar LLVM dialect import + verification; metadata sidecar and initial scalar/CFG/SCF raising implemented |
| HOST-04 | OpenCL Clang → LLVM → MLIR metadata matrix | PASS: 6 cases (SAXPY/Reduction × O0/no-optnone/mem2reg) |
| HOST-05 | MLIR raising Pass and SCF pipeline | PASS: native differential oracle, source locations, 14 rejections + embedded expected diagnostics |
| HOST-06 | Fixed local memory → memref Pass | PASS: array loops/aliases, native oracle, 29 rejection cases and expected diagnostics |
| HOST-07 | Paired Host/Device lowering and boundary bundle | PASS on Linux: 56 rejection cases, deterministic assets, two standalone fragment shaders validated; see validation/2026-10-30-boundary-report.json |
| HOST-08 | Real C/OpenCL C → paired boundary bundle | PASS on Linux: make example; native original/recovered Host oracle; source mutation; 12 rejection cases; validation/2026-10-30-c-boundary-report.json |
| HOST-WIN-01 | Windows x64 MSVC configure/build/CTest | CI workflow provided, not executed here |
| P0-SDK-01 | Public Program Asset generation and behaviour binding | Pending: Unity + installed Worlds SDK unavailable |
| P0-VM-01 | Nine actual VM opcodes and COPY stack order | Pending SDK VM golden output |
| P0-ABI-01 | Two behaviours, exported slots, CustomEvent sync/callback/reentrancy | Pending SDK VM + PC client |
| P0-GPU-01 | Fragment/Blit/fence/readback ordering, src != dst | Pending PC Build & Test stress run |
| P0-UPLOAD-01 | Typed array → Texture upload, Q18 latency/throughput | Pending sizes/format/bit-pattern measurements |
| P0-FORMAT-01 | RGBA8 byte fidelity, sRGB disabled, RFloat roundtrip | Pending PC graphics API / device results |
| P0-REQUEST-01 | Multiple request identity; stale generation callbacks | Pending; serialize requests unless proven identifiable |
| P0-ABI-02 | Genuine C11 ILP32 Clang TargetInfo/DataLayout | Pending; no replacement host target claimed |
| P0-LIBC-01 | Picolibc Clang cross-build | Pending |

Environment: Linux x86_64 host. No Unity/VRChat Worlds SDK installation located
under the examined user/opt directories. No PC VRChat client tests executed.
Windows CI is configured but has no execution result in this workspace.

P0/G0 must be validated before claiming GPU/SDK functionality. P1 remains
partial: 17/57 operations have ODS definitions. The boundary prototype includes
callee/resource symbol checks and a restricted lowering. Complete code bounds,
stack dataflow, Extern DB, effects, other types, the remaining 40 operations and
general lowering are pending. No pseudo VM or interpreter was added.

Reference lookup (2026-10-30, evidence of documentation only):

- https://creators.vrchat.com/worlds/udon/vm-and-assembly/
- https://mlir.llvm.org/docs/DefiningDialects/Operations/
- https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.4

Reproduce host verification with README.md commands. Logs under
`docs/validation/` record actual executed commands and results when available.

Actual host execution: [2026-10-30-host.txt](validation/2026-10-30-host.txt).
CMake 4.2.3, GCC 15.2.0; LLVM/MLIR 23.1.4 official Linux binary SDK.
CMake now emits ten compilation database entries. clangd 23.1.4 reports zero errors
for UdonAsmDialect.cpp when using `--query-driver=/usr/bin/c++`.
The importer requires DLTI registered alongside LLVM; this is covered by HOST-03.

2026-10-30 recovery update: [recovery.md](recovery.md) describes the TableGen
Pass, full conversion legality and preserved metadata. Results:
[2026-10-30-recovery.txt](validation/2026-10-30-recovery.txt) and
[OpenCL report](validation/2026-10-30-opencl-report.json).
CTest has five passing tests on this Linux host. Metadata import is validated;
OpenCL kernel raising, general memory/alias recovery, GPU execution and SDK gates
remain incomplete. Initial scalar recovery does not close R6 or complete P1.

Fixed local memory recovery update: `vrc-raise-llvm-memory` now precedes scalar
conversion in the composed pipeline. Evidence is in
[2026-10-30-memory.txt](validation/2026-10-30-memory.txt) and
[memory report](validation/2026-10-30-memory-report.json).
All seven CTest tests pass on Linux. General alias/byte-memory/struct/global and
OpenCL GPU-memory recovery remain pending; this does not complete P1 or G0/G2.

Boundary compiler evidence is recorded in
[validation/2026-10-30-boundary-report.json](validation/2026-10-30-boundary-report.json).
The complete CTest suite now passes 10/10 on Linux, including the real C/OpenCL C example. Optional `glslangValidator` and
`spirv-val` validate two standalone generated HLSL fragment shaders; SPIR-V is
only a test artifact, not the compiler output route. Linux CI installs these
validators. ShaderLab import, Unity compilation, byte fidelity on render targets,
actual buffer handles, queue/fence/readback and Udon execution remain unverified.

The source example is invoked with `make example`; `make test` configures,
builds and runs CTest. Its LLVM input uses a temporary SPIR32 layout and explicit
Host C calling convention. The final Host C11 TargetInfo remains a pending gate.
[Source example evidence](validation/2026-10-30-c-boundary-report.json) includes
native Host differential execution, replay through the registered MLIR pipeline,
changed C producing changed HLSL, and 12 source/config rejection cases. Two
source-generated fragment shaders were also compiled with glslang and validated
with spirv-val. This is compiler validation, not SDK/client execution.
