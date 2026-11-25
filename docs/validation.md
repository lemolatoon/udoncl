# Validation status

This is a dated log; entries describe the state on their date. Since then the
Python ONNX execution-plan path (its planner, examples, Unity runtime,
`UdonOnnxRuntimeBehaviour` and the Arena Add / Tiny Llama samples) was removed
in 343bd31a, ce4c1ad7, edb10b4e, 450e4305 and 43eabad6, and the legacy OpenCL
recovery of Hosts without a linked runtime, with the Hello Dispatch libraries
and `<udoncl/unity_buffer.h>`, in 2f8e2bc1, 80d3bd30 and 38d03881. The
example Hosts later moved to the `initialize`/`run`/`dispose` lifecycle: the
SAXPY entry below calls the former `run_saxpy` export (`RunSaxpy`), which
`examples/saxpy` replaced with `initialize`, `run(n, alpha)` and `dispose`.
Current Unity validation runs through the commit validation service described
in [validation/unity-ci.md](validation/unity-ci.md); the real-TinyLlama gates
run through `scripts/tinyllama_gates.py` (see the 2026-11-25 entry).

## TinyLlama chat example gate (2026-11-25)

`scripts/tinyllama_gates.py` builds an exact commit's compiler, SDK and real
TinyLlama artifacts in an isolated tree, generates their ONNX Runtime oracles,
and runs the focused Unity fixtures (`run --gates ...`) in a copy of the Unity
template. The gates are `prefill`, `step`, `conv`, `chat`, `multiturn`,
`conv-4k`, `chat-4k`, `multiturn-4k` (the advanced `examples/onnx-transformer-tiny`
artifacts) and `tinychat` (build target `tinyllama-chat`, the
`examples/tinyllama-chat` artifact at 2048 tokens, checked by
`scripts/validate_tinyllama_chat_artifact.py`). The `tinychat` oracle
(`chat-2k`) is `tests/boundary/tinyllama_chat_oracle.py --capacity 2048` with
the user messages "What is your name?" and "Repeat your name.".

The `tinychat` fixture runs both turns through `UdonclTinyllamaChatBinding`
(`BeginMessage`, `ContinueMessage` per frame, `Generate` while
`ReadStreamedText` polls every frame), compares each reply's tokens
(`ReadGeneratedTokens`) and text with the oracle, then disposes, initializes
again and disposes once more. On 4f33feb0 it passed in Unity 2022.3 with the
Worlds SDK (D3D11): both replies matched the oracle token for token
(`[3421, 1024, 338, 19235, 29889, 2]` "My name is Sarah." and
`[29903, 279, 801, 29889, 2]` "Sarah."), the streamed text only grew and
never changed, no poll was refused as busy, and the second Initialize started
an empty conversation. The advanced `chat` gate passed in the same run.

## Public ONNX Package Sample lifecycle (2026-11-03)

Unity 2022.3.22f1 with Worlds SDK 3.10.5 passed all 85 Editor tests. The
non-Pooling Arena Add and standard-operator Tiny Llama Package Samples were
compiled to Udon programs and executed in Play Mode against their imported
runtime prefabs. Arena Add completed float32 upload, one Device pass, async
readback, and returned `5`. Tiny Llama completed int32 token upload, all 18
Device passes, async 16-logit readback, and returned the reference argmax token
`2`. Lifecycle cleanup and the final Unity error-console check passed. This is
Editor/D3D11 validation, not a VRChat client claim, and the disabled general
ONNX Pooling path was not exercised.

## UdonSharp assembly discovery in batch Editor tests (2026-11-03)

An asmdef containing `UdonSharpBehaviour` subclasses is not a U# compilation
unit merely because it references `UdonSharp.Runtime`. Worlds SDK 3.10.5 also
requires a `UdonSharpAssemblyDefinition` asset whose `sourceAssembly` points at
that asmdef. `Udoncl.Runtime.asset` supplies this registration for the package
runtime.

The SDK lazily discovers those assets through `AssetDatabase` and Unity Package
Manager APIs. If its first `CompilerUdonInterface` assembly-cache access occurs
inside the background work started by `UdonSharpCompilerV1.CompileSync`, Unity
throws `GetAllRegisteredPackages can only be called from the main thread`; the
failed cache then makes even registered SDK scripts appear not to belong to a
U# assembly. Importers that explicitly compile package U# assets must therefore
materialize `UdonSharpAssemblyDefinitions` on the Editor main thread before
entering `CompileSync`. This is an SDK 3.10.5 integration constraint, not proof
that arbitrary Unity or worker-thread U# compilation is safe.

No VRChat client gate has passed. Targeted Unity Editor, installed Worlds SDK
assembler/IUdonVM, and GPU readback gates have passed for the explicitly listed
subsets; compiler host and the incomplete generated runtime remain separate.

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
| P0-SDK-01 | Public Program Asset generation and behaviour binding | PARTIAL: scalar, fixed CopyBuffer, fixed f32/i32 dispatch, and two-buffer SAXPY import/reimport pass in Unity 2022.3.22f1 + Worlds SDK 3.10.5; general runtime pending |
| P0-VM-01 | Nine actual VM opcodes and COPY stack order | Pending SDK VM golden output |
| P0-ABI-01 | Two behaviours, exported slots, CustomEvent sync/callback/reentrancy | Pending SDK VM + PC client |
| P0-GPU-01 | Fragment/Blit/fence/readback ordering, src != dst | PARTIAL: Editor IUdonVM + synchronous readback passes fixed CopyBuffer and SAXPY; async fence/readback and PC Build & Test stress remain pending |
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
partial: 20/57 operations have ODS definitions. The boundary prototype includes
callee/resource symbol checks and a restricted lowering. Complete code bounds,
stack dataflow, Extern DB, effects, other types, the remaining 37 operations and
general lowering are pending. No pseudo VM or interpreter was added.

Reference lookup (2026-10-30, evidence of documentation only):

- https://creators.vrchat.com/worlds/udon/vm-and-assembly/
- https://mlir.llvm.org/docs/DefiningDialects/Operations/
- https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.4

## Windows VS 2026 scalar Udon backend bootstrap (2026-10-31)

Local Windows x64 configure and build completed with Visual Studio 2026
Insiders 18.12 / MSVC 19.52.37025 and the locked LLVM/MLIR 23.1.4 SDK. All
13 CTests passed. The new `vrc-lower-udonhl-to-udonasm` pass allocates typed
heap slots and lowers scalar `i32`/`f32` arithmetic, signed integer comparison,
block arguments, loops and conditional branches to the existing nine-opcode
UdonAsm dialect. `udoncl-translate --emit-udon-assembly` emits data/export
directives and textual opcode assembly with byte-address relocation.

This remains a compiler bootstrap, not a complete G0/G1 runtime claim. The
installed SDK assembler and `IUdonVM` execute all four scalar golden tests:
`17 + 25 = 42`, `3.5 * -2 = -7`, `sum_to(5) = 10`, and the float-comparison
phase branch (`0.02 -> 0`, `0.30 -> 2`). Unity Test Framework job
`f9d6edf33b524a49bd35407962455a8e` passed 4/4. The preceding VM run caught and
corrected the SDK's float extern spelling (`Multiplication`, not `Multiply`). Extern
signatures are matched to installed SDK sample assembly, but generated Extern
DB validation remains pending. Unsigned C provenance, normal/recursive calls,
memory, VRC GPU intrinsics and continuation lowering are deliberately rejected
until their general contracts are implemented. The LJ Host remains on the
native backend at this checkpoint. See `validation/2026-10-31-udon-vm.json`.

## Public CMake SAXPY Unity dispatch (2026-11-02)

The standalone `examples/saxpy` consumer passed the public
`find_package(UdonCL CONFIG)` path, artifact import, and reimport in Unity
2022.3.22f1 with Worlds SDK 3.10.5. The artifact contains two 128-byte buffers,
the gather kernel `saxpy_demo`, mixed copied i32/f32 scalars, and the typed
`RunSaxpy(int n, float alpha)` binding.

Both initial import and reimport ran `RunSaxpy(16, 2.0f)` through the installed
SDK `IUdonVM` with Interpret/status 0. Input words were `x = 1..16` followed by
`y = 100..115`. All 32 output words matched exactly: the first 16 were
`2*x+y = 102, 105, ..., 147`, and the remaining 16 were the kernel's
`input[gid]` passthrough `100..115`. Input roundtrip and output each had zero
mismatches and maximum absolute error 0.

The generated shader was supported with one pass. The prefab/Udon heap bound
the exact input/output RenderTextures and material; `_VRC_Input`,
`_VRC_Arg2=16`, and `_VRC_Arg3=2.0` matched the dispatched values. Final Unity
console counts were 0 errors and 0 warnings. Cleanup removed the generated
asset root and serialized Udon program, leaving no scene instance or temporary
import backup directory. This proves the fixed synchronous AOT subset, not
public upload/readback, asynchronous continuation, or a VRChat client run.

## LJ ice / liquid / refreeze native oracle (2026-10-31)

The MD example now adds a zero-target-temperature Langevin drag term while
retaining the old NVE path when its cooling rate is zero. A sixfold
bond-orientational order measurement (`|Psi6|`) distinguishes loss and recovery
of the triangular crystal structure. The deterministic Windows D3D11 WARP run
executed 23,200 real fragment draws and retained exact CPU/GPU trajectory
agreement (`max_cpu_gpu_error=0`). Its phase cycle measured
`Psi6 = 1.000 -> 0.173 -> 0.990` and
`T = 0.000200 -> 8.66776 -> 0.036085`, passing
`ice -> liquid -> ice`. See
`validation/2026-10-31-md-phase.json`. This is the native oracle; it does not by
itself prove Unity/Udon execution.

Reproduce host verification with the commands in [development.md](development.md). Logs under
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

## MD crystal host-native (stage 1)

`make example-md` compiles ordinary OpenCL Host C through the native UdonHL
call adapter back to LLVM and links the C API/runtime. Two OpenCL C Device
kernels recover bounded gathers and counted loops, structure control flow,
lower through `vrcgpu`, and execute as Fragment Shaders. The kernel state is
256 float words in a 256×1 byte-preserving RGBA8 texture. Input/output must be
distinct, initialized where read, and match the selected AOT extent.

Linux `ctest --preset udoncl`: **15/15 passed**, including the MD test:

- Independent double-precision potential derivative checks the kernel force.
- 100 heated steps / 200 draws agree with the CPU-compiled OpenCL source
  (maximum observed error 0 on Lavapipe; tolerance 0.002).
- 1000 unheated steps: maximum relative energy drift 1.62851e-8 and momentum
  drift 2.21713e-7.
- After 1000 heated steps, local mean kinetic energy is 0.0125993, compared
  with 0.0000900968 in the unheated control.
- After release and a further 1000 steps, the distant region's mean kinetic
  energy is 0.00214197 versus the same-time control 0.0000298581; release energy drift is
  2.32987e-7. This is a finite reduced-unit test, not a material calibration.
- Nine malformed kernels/IR layouts reject bounds violations, scatter/input writes,
  barriers, wrong global-id dimension, an unbounded loop, padded f32 storage
  and a 64-bit global-pointer layout without publishing
  a bundle. Six runtime cases reject aliasing, uninitialized input/readback,
  incorrect NDRange, undersized and stale allocations.
- Native interactive protocol steps and reset pass. Chromium interaction
  heats to step 1372, pauses, and resets to step 0, with no JavaScript errors.
  Two simultaneous touch points and their release pass too. `make md-view`
  serves the page and returns updated native state through `/step`.

[Recorded results](validation/2026-10-30-md-crystal.json) and
[interactive screenshot](validation/2026-10-30-md-crystal.png).
`make md-view` serves only on localhost and displays states read from the
lowered native binary. UdonAssembly, U#, Editor import and actual avatar hands
remain stage 2; no Unity/VRChat execution is claimed by this example.
