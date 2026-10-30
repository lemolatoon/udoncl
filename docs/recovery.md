# LLVM import and high-level recovery

The independent `VRCLLVMToHighLevelMLIR` library owns import metadata capture
and raising. The design remains comprehensive high-level recovery as specified
in PLAN.md §4. This implementation is the first supported stage, with explicit
errors for unimplemented semantics; it is not the completed recovery pipeline.

## P0 OpenCL findings (Clang/LLVM/MLIR 23.1.4)

SAXPY and one work-group reduction stage are compiled as OpenCL C 1.2 to
`spir-unknown-unknown` (32-bit SPIR). This is a device import experiment, not the
final host C target definition. No native OpenCL device, Unity or VRChat client
was used. The reduction example still needs iterative inter-group dispatches
and final GPU readback to satisfy the MVP.

The test matrix compares `-O0 -g`, `-O0 -g -Xclang -disable-O0-optnone`, and the
latter followed by LLVM `-passes=mem2reg`. All six inputs import and verify.

- Native import retains LLVM calling conventions, pointer address spaces,
  builtin/barrier calls, CFG and debug locations.
- Native import does not retain `kernel_arg_*` attachments. The importer adapter
  captures metadata before translation, rather than inferring element types
  from opaque pointers later.
- In these fixtures Clang emits a kernel entry plus
  `__clang_ocl_kern_imp_<name>`. The entry may call the helper or contain a body
  copy depending on optnone. Calls/barriers must be analysed across functions;
  counting every definition as independent dispatch work would be wrong.
- Reduction's pointer arguments include SPIR AS1 (global) and AS3 (local).
  Local-memory fence flags and all call-site source locations survive capture.
- LLVM mem2reg exposes SSA/PHI form. The LLVM opt executable warns that it cannot
  construct a SPIR target machine on this SDK; the target-independent promotion
  still succeeds. Diagnostics are retained rather than hidden.

Reproduce with `ctest --preset udoncl -R opencl-frontend-metadata`.
Generated `.ll`, `.mlir`, `.json`, `.stderr`, `commands.json` and `report.json`
are under `build/udoncl/tests/opencl`. Checked-in results are in
`docs/validation/2026-10-30-opencl-report.json`.

## Frontend sidecar v1

`vrc-translate --import-llvm input.ll --metadata-output sidecar.json -o input.mlir`
captures metadata in a JSON document with:

- Schema version, LLVM version, target triple, data layout, pointer bit width.
- Named module metadata, per-function calling convention/kernel entry status,
  signature, attributes, arguments/address spaces and attached kernel metadata.
- Direct/indirect call sites, argument operands, source locations and allocas.
- `source_llvm_ir`: canonical printed LLVM module including full metadata/debug
  information; `source_sha256` hashes its UTF-8 bytes, not the original file bytes.

A compact JSON form without `source_llvm_ir` is embedded in the MLIR module's
`vrc.frontend` string attribute, whether or not an external sidecar is requested.
Metadata references use canonical module slot IDs, and repeated imports are tested
for identical output. The source hash ties summaries to the captured module. Generic metadata
operands outside strings/integers are retained textually; the complete canonical
IR is the fallback source. No inference of array bounds, aliasing, buffer sizes,
barrier uniformity or helper ownership is claimed from these records alone.
The current Pass does not consume the sidecar as proof of memory legality.

## MLIR Pass conventions and scope

`include/VRC/Conversion/LLVMToHighLevel/Passes.td` defines
`vrc-raise-llvm-to-scalar`; TableGen generates declarations, base class and
registration. `registerVRCRecoveryPasses()` exposes the Pass to `vrc-opt`.
Conversion uses `ConversionTarget`, rewrite patterns and `applyFullConversion`.
LLVM operations are illegal after this Pass; unknown operations do not become
implicitly legal. Validation precedes rewriting, diagnoses the original location
and fails the Pass when it cannot prove a supported mapping.

Supported operations:

- Nonvariadic scalar C-convention functions, integer/f32/f64 signatures.
- Integer/float constants; integer add/sub/mul, bitwise operations, shifts,
  zero/sign extension, narrowing, integer comparisons and select.
- Float add/sub/mul/neg with no fast-math flags.
- Return and conditional/unconditional branches with SSA block arguments.
- `nsw`/`nuw` are translated explicitly to arith overflow flags, including trunc.

Source function attributes are archived under `vrc.llvm_function`, while only
validated function contracts are accepted. Archiving an attribute is not treated
as a substitute for implementing its semantics. Source locations are preserved.
Exact/nneg/disjoint flags, target attributes, other ABI attributes, vectors,
pointers, calls, integer division, explicit poison and unsupported operations
produce diagnostics. SCF structuring may create UB poison placeholders for values
that are unused on the relevant paths; these are introduced by the upstream
structuring transform, not silently accepted from unsupported LLVM input.

`vrc-recover-high-level` is a registered Pass pipeline:

1. Upstream `mem2reg` promotes local storage proven promotable by memory-slot interfaces.
2. `vrc-raise-llvm-to-scalar` converts to `func` / `arith` / `cf`.
3. Upstream `lift-cf-to-scf` reconstructs structured control flow.

General globals/GEP/alias/byte-memory/aggregate recovery to memref, induction and
affine analysis, OpenCL builtins to GPU, local-size specialization and barrier
legality remain required future stages. Unpromotable LLVM memory is rejected.

## Verification

`llvm-high-level-recovery` checks PassManager invocation, CFG/PHI edges,
`scf.while`/`scf.if`, overflow flags, promotion of initialized nonescaping local
storage, source locations and 14 rejection cases. Split-input `.mlir` tests also use the
upstream `--verify-diagnostics` / `expected-error` convention. It lowers recovered IR back to
LLVM with upstream conversions, then builds both the original and recovered
versions against the same independent C oracle. Each executable checks dynamic
loop bounds, unsigned wrap/comparisons, integer casts, exact binary f32 values,
valid overflow-flag inputs and promoted local storage. Native execution does not
prove Udon VM, GPU or the final ILP32 host ABI.

Upstream references used for the implementation:

- https://mlir.llvm.org/docs/TargetLLVMIR/
- https://mlir.llvm.org/docs/DialectConversion/
- https://mlir.llvm.org/docs/Dialects/ArithOps/
- LLVM 23.1.4 installed `ControlFlowToSCF.h`, `Passes.td`, LLVM/Arith ODS and interfaces.
