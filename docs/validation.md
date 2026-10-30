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
| HOST-09 | Prebuilt SDK install and CPU parallelism | PASS on Linux: fresh official archive → configure/build; 9 installer tests and 5 CPU/build-command tests; validation/2026-10-30-prebuilt-sdk.json |
| HOST-10 | UdonHL → LLVM → native Host + fragment execution | PASS on Linux: 16 bitwise readbacks, 8 runtime and 4 native-lowering rejections; validation/2026-10-30-native-fragment.json |
| HOST-11 | OpenCL Host C + linked C runtime + native fragment execution | PASS on Linux and Windows CI: square/x+x, original/lowered Host differential, API contracts and six adapter rejections; validation/2026-10-30-opencl-host.json |
| HOST-WIN-01 | Windows x64 MSVC configure/build/CTest | PASS: VS 2026/windows-2025 configure/build and 14/14 CTests, including OpenCL Host API and D3D11/WARP fragment draw/readbacks. See validation/2026-10-30-windows-ci.json |
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
Windows CI configure/build and all 14 CTests passed in run 41064059775,
including native UdonHL Host binaries linked to the D3D11/WARP fragment runtime.
The earlier VS 2022 link failure and its fix are recorded below.

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
The complete CTest suite now passes 14/14 on Linux with native fragment tests enabled, including the real C/OpenCL C example and SDK/build helpers. Optional `glslangValidator` and
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

Prebuilt SDK validation used a fresh extraction of the locked official Linux
archive into a separate prefix, then a full project build and all 12 CTests.
Windows archives were downloaded, SHA256-verified and extracted on Linux;
LLVM/MLIR headers, static libraries, CMake packages, tools and Clang resources
were checked. PE/library metadata confirmed the selected SDK's Release `/MD`
runtime and assertion-free ABI; Windows binaries were not executed here.
Windows native configure/build/CTest subsequently passed in run 41059537479.
Both matrix jobs install archives on cache misses, without compiling LLVM from source.

Windows CI run [41055911752](https://github.com/lemolatoon/udoncl/actions/runs/41055911752/job/124063257866)
failed linking `vrc-opt`: `__std_rotate`, `__std_min_element_8i` and related MSVC
STL helpers required by the prebuilt LLVM/MLIR libraries were absent from the
runner's MSVC 14.44/VS 2022 standard library. The SDK's pinned
[release workflow](https://github.com/munich-quantum-software/portable-mlir-toolchain/blob/2026.10.28/.github/workflows/build-portable-mlir-toolchain.yml)
builds and tests Windows x64 on `windows-2025`. That image uses
[Visual Studio 2026](https://github.blog/changelog/2026-05-14-github-actions-upcoming-image-migrations/).
The project now uses the same runner and its latest installed VS toolset. No
compatibility stubs or changes to LLVM libraries were introduced. Native Windows
validation of this runner change subsequently passed in run 41059537479.

A follow-up [Windows job](https://github.com/lemolatoon/udoncl/actions/runs/41057266482/job/124067836214)
confirmed the runner contains VS 2026 version 18.11, but the explicit `18.0`
selection failed: `msvc-dev-cmd@v1` constructs an upper version bound of `18.9`.
Omitting that input uses the action's `vswhere -latest` discovery and avoids
excluding newer VS 2026 minor releases.

Native fragment validation executes the generated Host object after UdonHL →
func/arith → LLVM dialect → LLVM IR lowering and links it to the independent
`VRCNativeFragmentRuntime` static library. Linux Lavapipe renders the generated
Device fragment shader into 1x1 RGBA8_UNorm, then explicit image/buffer barriers,
a fence and host readback recover the f32 bits. Square and x+x source variants
match an independent native C oracle for 16 draws, including signed-zero checks.
Runtime and native-lowering rejection cases are exercised separately. An additional
square run with Khronos validation and synchronization validation enabled emitted
no VUID, synchronization-hazard or validation-error diagnostics. The Windows
D3D11/WARP runtime also passed 16 fragment draw/readbacks in CI, including the
source variant and bitwise native C oracle checks.
This does not pass Unity/VRChat SDK gates or implement asynchronous suspension.

The successful [Windows job](https://github.com/lemolatoon/udoncl/actions/runs/41059537479/job/124075510957)
uses MSVC 14.51.36231 and passed all 13 CTests, including the C example and
native-fragment-execution. The latter exercised 16 real D3D11/WARP draws and
readbacks, 8 invalid runtime request cases and 4 native-lowering rejection cases.
The final test summaries use ASCII so Windows' default cp1252 console encoding
cannot turn a completed test into a UnicodeEncodeError. Linux passed in the
same workflow run. Neither result claims Unity/VRChat execution.

The same successful workflow reported a separate Linux SDK cache-save warning:
`zstd` could not write the cache archive because the runner ran out of disk
space. The Linux build/tests still passed and the Windows SDK cache restored
successfully. Reducing Linux cache-pack disk usage remains follow-up work.

The OpenCL Host API example uses vendored Khronos declarations and a linked C
runtime, with no vendor OpenCL implementation. It validates 16 lowered-Host
executions plus two runtime contract draws, repeats these with the original
Host objects, and compares f32 results bitwise after the Host's +1 computation.
The runtime contract checks cover source/argument copies, context/event isolation,
NDRange validation, errors, partial transfers, allocation generations, parent
lifetimes and zero live objects after cleanup. A targeted C runtime test using
a CPU backend stub also passed AddressSanitizer/UndefinedBehaviorSanitizer and
leak detection; actual rendering is validated separately by the integration test.

The new native call adapter preserves native LLVM pointer/memory/CFG and ABI
attributes while exposing calls as udonhl.call. It is not production high-level
memory recovery or proof of the final Udon ILP32 ABI. The standalone Device
bundle stage is native-device-prototype and Unity readiness remains false.
The new OpenCL API flow passed Windows D3D11/WARP and Linux Lavapipe in
[CI run 41064059775](https://github.com/lemolatoon/udoncl/actions/runs/41064059775).
Both jobs passed all 14 CTests. The earlier native fragment-only WARP success
remains recorded separately.
