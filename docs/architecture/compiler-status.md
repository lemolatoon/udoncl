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
- Host C calls the OpenCL 1.2 API through the linked OpenCL AOT runtime
  (`UdonCL::OpenCLRuntime`): buffers, writes, reads, copies, and the launches
  of library kernels and of the OpenCL C kernels of its own Device libraries,
  with ordinary status checks. udonclc derives each user kernel launch from the
  Host C; buffer arguments precede `int`/`uint`/`float` scalars, and the
  global work size is one constant unless a `device_kernels` entry gives it.
  The standalone `examples/saxpy` consumer launches a three-buffer kernel with
  mixed i32/f32 scalars. A Host without the runtime may not call the OpenCL
  API: the build names the calls and asks for `UdonCL::OpenCLRuntime`.
  Generated U# bindings separately expose byte-exact whole-buffer upload and
  asynchronous readback.
- `udoncl_add_unity_library` accepts ordinary `DEVICE_LIBRARIES` targets and
  packages the ShaderLab/HLSL of every kernel the Host launches, with its
  manifest metadata, in the Unity artifact. The config does not declare those
  kernels: udonclc derives each `device_kernels` entry from the Host's dispatch
  sites, from the library kernel contract for UdonBLAS, UdonDNN, UdonATen and
  UdonONNXOperators kernels, and from the `clEnqueueNDRangeKernel` launches
  for the user OpenCL C kernels the linked runtime numbers from 1024. A
  hand-written entry replaces the derived one of its `kernel_id`. The
  `examples/saxpy` Host C launches its OpenCL C kernel this way.
- ONNX models compile through the C++ ONNX-dialect path.
  `udoncl_add_onnx_host_library` runs `scripts/compile_onnx_mlir.py` with the
  repository ONNX frontend on the pinned ONNX-MLIR SDK, specializes the model
  to its `DIMENSION_BOUNDS` and `INPUT_SHAPES`, and lowers it to Host MLIR that
  calls UdonBLAS, UdonDNN, UdonATen and UdonONNXOperators. A user C Host
  drives that model through `UdonONNXRuntime` (sessions of
  `<udononnxruntime.h>` or the generated `udon_onnx_binding_*` functions) and
  is linked into an ordinary Unity library with a generated U# binding.
  `examples/onnx-mnist` and `examples/onnx-transformer-tiny`, up to the real
  TinyLlama-1.1B with 512- and 4096-token contexts, use this path. The Python
  execution-plan planner and its Unity runtime were removed (43eabad6,
  450e4305); no execution-plan JSON is generated or interpreted.
- The typed Unity backend lowers each fixed AOT `vrc.dispatch` with ordered
  input/output buffers and copied f32/i32 arguments to typed material property
  updates and one `VRCGraphics.Blit`; the Importer binds the generated kernel
  material through its manifest dispatch site. Hosts that launch many kernels
  and exports whose Device work spans frames use the same lowering, and
  generated bindings provide explicit whole-buffer transfers. Dynamic buffer
  layouts and NDRange sizes remain pending.
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

Host/Device boundary example (C and OpenCL C input; compile `host.ll` and
`device.ll` as in [examples/c-boundary](../../examples/c-boundary/README.md)):

```text
build/udoncl/tools/vrccl/udonclc --emit-llvm-boundary-bundle examples/c-boundary/boundary.json --host-llvm build/c-boundary/host.ll --device-llvm build/c-boundary/device.ll -o build/boundary-example
build/udoncl/tools/vrccl/udonclc --check-boundary-request build/boundary-example/manifest.json --request examples/c-boundary/request.json
```

The Host path lowers `func.call` through `udonhl.call`, then substitutes the
internal typed dispatch intrinsic with `vrc.dispatch`. The Device path lowers a
single-invocation f32 store through `vrcgpu` to a fragment shader. Both sides must
agree on kernel ID, argument/resource order and NDRange before assets are emitted.
The bundle includes `host-device.mlir`, `manifest.json` and `kernels/*.shader` /
`*.hlsl`. Reusing identical output is allowed; differing existing output is refused.
This boundary example does not yet execute its GPU request from Udon.
Scalar/CFG `.uasm` emission is tested separately. See
[boundary.md](../boundary.md) for the supported pattern and next steps.
