# Validation status

No SDK/client gate has passed. Compiler host and generated runtime are separate.
All pending target tests below are `TODO: verify on target SDK/client`.

| ID | Test | State / acceptance evidence needed |
|---|---|---|
| HOST-01 | LLVM/MLIR 23.1.4 TableGen + four independent dialect targets | PASS on Linux: validation/2026-10-30-host.txt |
| HOST-02 | Nine opcode parser/printer and invalid attributes/unknown ops | PASS on Linux: round-trip + 16 rejection cases; not VM semantics |
| HOST-03 | LLVM import through independent recovery library | PASS scalar LLVM dialect import + verification; raising and metadata sidecar pending |
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
partial: 9/57 operations have ODS definitions and local attribute checks; full
symbol resolution, code bounds, stack dataflow, Extern DB, effects, other types,
48 operations and lowering are pending. No pseudo VM or interpreter was added.

Reference lookup (2026-10-30, evidence of documentation only):

- https://creators.vrchat.com/worlds/udon/vm-and-assembly/
- https://mlir.llvm.org/docs/DefiningDialects/Operations/
- https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.4

Reproduce host verification with README.md commands. Logs under
`docs/validation/` record actual executed commands and results when available.

Actual host execution: [2026-10-30-host.txt](validation/2026-10-30-host.txt).
CMake 4.2.3, GCC 15.2.0; LLVM/MLIR 23.1.4 official Linux binary SDK.
CMake emits seven compilation database entries. clangd 23.1.4 reports zero errors
for UdonAsmDialect.cpp when using `--query-driver=/usr/bin/c++`.
The importer requires DLTI registered alongside LLVM; this is covered by HOST-03.
