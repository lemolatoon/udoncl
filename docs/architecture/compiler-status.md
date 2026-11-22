# Compiler status and internal tools

This page records which compiler components exist today, what remains
pending, and how to drive the internal `udoncl-translate`, `udoncl-opt`, and
`udonclc` tools directly. It is aimed at compiler developers; users who only
compile their own sources into Unity artifacts should start from the
[README](../../README.md). Run the commands below from the repository root after
building the compiler as described in [development.md](../development.md).
Gates and evidence are recorded in [validation.md](../validation.md).

## Current components

- Four separate TableGen dialect libraries; 27 of the closed catalog of 57 operations are defined.
- A C OpenCL 1.2 AOT API subset and native Host Runtime linked to the fragment renderer.
- Explicit Device buffer allocation with bounded Write/Read/Copy commands;
  host-backed buffers, SubBuffer, Map/Unmap, and rectangular copies fail
  explicitly instead of selecting an unimplemented fallback.
- The Udon backend can bind typed `vrc.buffer_alloc` operations to the ordered
  fixed buffer pool and lower an initial full-buffer `vrc.buffer_copy` to
  `VRCGraphics.Blit`. This first Unity subset requires constant exact sizes,
  zero offsets, equal texture shapes, and distinct source/destination buffers;
  partial, dynamic, or self copies are compile errors rather than approximated.
  `udonclc --host-mlir` remains available for typed compiler validation.
- Normal Host C can use a restricted straight-line OpenCL 1.2 setup sequence:
  one platform and GPU device, one context and in-order queue, constant-sized
  `clCreateBuffer` allocations, full `clEnqueueCopyBuffer` commands, and their
  release calls. The compiler validates context/queue/opaque-handle flow and
  recovers typed buffer IR. A second fixed subset accepts ordered read-only
  inputs plus one write-only output, copied f32/i32 scalar arguments, and a
  statically configured `clEnqueueNDRangeKernel`, lowering it to generated
  texture/scalar material bindings and `VRCGraphics.Blit`. The standalone
  `examples/saxpy` consumer exercises two buffers and mixed i32/f32 scalars.
  Status-dependent control flow, uploads/readbacks, dynamic NDRange sizes,
  other argument layouts, partial copies, and general OpenCL lifetime semantics
  remain explicit compile errors in the C Host lowering. Generated U# bindings
  separately expose byte-exact whole-buffer upload and asynchronous readback;
  direct lowering of blocking C read/write calls to Host continuations remains
  pending. `<udoncl/unity_buffer.h>` remains a temporary
  compatibility input, not the preferred public example.
- `udoncl_add_unity_library` accepts ordinary `DEVICE_LIBRARIES` targets and
  packages the selected AOT kernel's ShaderLab/HLSL plus manifest metadata in
  the Unity artifact. The `hello-dispatch-unity-library` example connects its
  restricted standard OpenCL Host sequence to the generated kernel material.
- The `mnist-dense` ONNX profile lowers its Host graph to `func`/`cf`/`arith`
  plus the raw `!udonhl.ptr<0>` UdonBLAS ABI. `udoncl_add_unity_library` links
  that Host MLIR against the recovered UdonBLAS/OpenCL AOT library module and
  emits the final Unity artifact without using the generated Host C as input.
  Host C remains a numerical oracle and temporary compatibility output.
- The full MNIST profile also emits a verified candidate Host MLIR call graph
  using the public UdonDNN convolution, op-tensor, activation, and pooling APIs
  followed by UdonBLAS SGEMM. It is not selected as the executable artifact yet:
  public `udondnnConvolutionForward` and `udondnnPoolingForward` now have
  generic 16,384-word Device backends covering the model's 5x5 convolution,
  channel-bias Add, ReLU, and 8x8 AveragePool. The full graph is not selected
  as the executable artifact until its multi-kernel Host continuation is
  numerically validated. The compiler does not silently substitute an
  ONNX-specific arena scheduler.
- The typed Unity backend can lower a fixed AOT `vrc.dispatch` with ordered
  input/output buffers and a copied f32/i32 descriptor to typed material
  property updates and one `VRCGraphics.Blit`; the Importer binds the generated
  kernel material through its manifest dispatch site. The standard OpenCL Host
  bridge validates the straight-line object and argument provenance before
  recovering that dispatch. Multiple kernels, dynamic layouts/NDRange, C-side
  upload/readback lowering, and asynchronous Host continuation remain pending.
  Generated bindings already provide explicit fixed-buffer transfers.
- `udoncl-opt`: MLIR parser/printer with operation verification, rejects unknown ops by default.
- `VRCLLVMToHighLevelMLIR`: independent import/metadata, local-memory and scalar/CFG recovery library for host/device development.
- `udoncl-translate --import-llvm input.ll`: LLVM dialect import with preserved frontend metadata and optional JSON sidecar.
- `vrc-lower-udonhl-to-udonasm` and `udoncl-translate --emit-udon-assembly`:
  an incremental scalar/CFG backend for real Udon opcodes and textual `.uasm`.
- `vrc-raise-llvm-memory` / `vrc-raise-llvm-to-scalar` MLIR Passes and the `vrc-recover-high-level` pipeline: proven fixed local arrays, scalar/CFG conversion and upstream SCF structuring.

The other 32 operations, complete verifiers/lowerings, the full executable Udon
host ABI, complete OpenCL runtime coverage and production Unity integration
remain pending. The scalar backend's integer, float and CFG golden programs have
passed the installed SDK assembler and actual `IUdonVM`; memory, calls,
continuations and GPU dispatch are still incremental work. A small paired
Host/Device lowering prototype emits ShaderLab/HLSL and a shared manifest.
See [validation.md](../validation.md) for gates and evidence.

## Import and high-level recovery

Import an LLVM IR loop and recover high-level MLIR (add `.exe` to tool names
on Windows):

```text
build/udoncl/tools/vrc-translate/udoncl-translate --import-llvm tests/recovery/loop.ll --metadata-output build/loop.frontend.json -o build/loop.llvm.mlir
build/udoncl/tools/vrc-opt/udoncl-opt --pass-pipeline="builtin.module(vrc-recover-high-level)" --verify-each build/loop.llvm.mlir -o build/loop.high.mlir
```

The raising Pass is defined in TableGen and uses MLIR dialect conversion;
`--vrc-raise-llvm-to-scalar` runs just scalar/CFG conversion. The composed pipeline
first runs upstream mem2reg, then proven local-memory recovery, scalar raising,
then upstream CFG-to-SCF.
Unsupported memory, pointers, calling conventions, instructions and semantic
flags fail explicitly. General memory/alias, affine, OpenCL builtin, GPU local
memory and barrier/dependence recovery remain pending.
See [recovery.md](../recovery.md) for scope, sidecar schema and P0 findings.

Try fixed-array recovery (initialization, update and sum):

```text
build/udoncl/tools/vrc-translate/udoncl-translate --import-llvm tests/recovery/array-loop.ll -o build/array.llvm.mlir
build/udoncl/tools/vrc-opt/udoncl-opt --pass-pipeline="builtin.module(vrc-recover-high-level)" --verify-each build/array.llvm.mlir -o build/array.high.mlir
```

`build/array.high.mlir` contains `memref.alloca`, `memref.load/store` and
`scf.while`. For just the memory conversion, use `--vrc-raise-llvm-memory`
instead of the pipeline. That Pass leaves scalar LLVM operations for the next
stage. Memory recovery currently requires fixed-size, homogeneous, nonescaping
entry-block allocations with proven bounds, alignment and initialization.
Unsupported pointer/alias, volatile/atomic and byte reinterpretation cases fail.
The IR and reports retained by the corresponding CTests are listed under
[Test artifacts](../development.md#test-artifacts).

## Host/Device boundary bundle

Host/Device boundary example (prepared high-level MLIR input):

```text
build/udoncl/tools/vrccl/udonclc --emit-boundary-bundle examples/host-device-boundary/probe.mlir -o build/boundary-example
build/udoncl/tools/vrccl/udonclc --check-boundary-request build/boundary-example/manifest.json --request examples/host-device-boundary/request.json
```

The Host path lowers `func.call` through `udonhl.call`, then substitutes the
internal typed dispatch intrinsic with `vrc.dispatch`. The Device path lowers a
single-invocation f32 store through `vrcgpu` to a fragment shader. Both sides must
agree on kernel ID, argument/resource order and NDRange before assets are emitted.
The bundle includes `host-device.mlir`, `manifest.json` and `kernels/*.shader` /
`*.hlsl`. Reusing identical output is allowed; differing existing output is refused.
This prepared boundary example does not yet execute its GPU request from Udon.
Scalar/CFG `.uasm` emission is tested separately. See
[boundary.md](../boundary.md) for the supported pattern and next steps.
