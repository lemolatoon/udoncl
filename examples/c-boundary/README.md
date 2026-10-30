# C to Host/Device boundary bundle

From the repository root, after installing `.deps/llvm23`:

```text
make example
```

This configures/builds with the `udoncl` CMake preset and runs `run.py`. The
Makefile delegates build settings to CMake; it does not maintain a second build
system. `make test` runs the complete CTest suite. `make example-c-boundary` is
an alias. Output can be changed with `EXAMPLE_OUTPUT=build/another-example`.
Differing existing bundle content is refused, so use a fresh output directory
after changing the source. Identical output can be reused.

Without Make, run `cmake --preset udoncl`, `cmake --build --preset udoncl`, then
`python examples/c-boundary/run.py`. On Windows use Developer PowerShell; the
script handles `.exe`. With Make, `PYTHON=python` selects the Python command.
For a custom build, pass `--build-dir path/to/build` to the script. It locates
Clang from that build's `LLVM_DIR`, keeping it consistent with linked LLVM/MLIR.

Inputs:

- `host.c`: public scalar `add` calls `add_impl`; internal `request_square`
  forwards a uint32 buffer handle and descriptor to `__vrc_dispatch`.
- `device.cl`: OpenCL C 1.2 kernel `square(__global float*, float)`, writing x*x
  to output[0].
- `boundary.json`: versioned AOT contract selecting export/request/intrinsic/
  kernel names, IDs and fixed launch dimensions.
- `request.json`: example descriptor for 2.5f, one four-byte buffer, one invocation.

The actual compilation route is:

```text
host.c / device.cl → pinned Clang → host.ll / device.ll
  → LLVM dialect import with frontend metadata
  → upstream mem2reg
  → vrc-raise-llvm-boundary-device (prove single output pointer and recover memref)
  → vrc-raise-llvm-to-scalar (including direct C calls)
  → upstream canonicalize
  → vrc-link-c-boundary (prove Host handle forwarding and link both modules)
  → vrc-lower-host-to-udonhl
  → vrc-substitute-runtime-intrinsics
  → vrc-lower-device-single-invocation
  → vrc-verify-host-device
  → manifest.json + ShaderLab/HLSL
```

`vrc-recover-c-boundary` registers the preparation/recovery/link pipeline with
PassManager. The bundle retains `frontend.imported.mlir` (the real imported
nested Host/Device modules), `frontend.recovered.mlir` (func/arith/memref + gpu),
and `host-device.mlir` (udonhl/vrc/vrcgpu). Frontend metadata includes original
LLVM fingerprints and OpenCL pointee/address-space metadata. The script keeps
`commands.json` with commands and diagnostics.

Inspect the output:

```text
build/c-boundary/bundle/manifest.json
build/c-boundary/bundle/host-device.mlir
build/c-boundary/bundle/kernels/kernel_7.shader
build/c-boundary/bundle/kernels/kernel_7.hlsl
```

You can also replay recovery independently:

```text
build/udoncl/tools/vrc-opt/vrc-opt --pass-pipeline="builtin.module(vrc-recover-c-boundary)" build/c-boundary/bundle/frontend.imported.mlir
```

The script invokes this compiler mode after Clang (add `.exe` on Windows):

```text
build/udoncl/tools/vrccl/vrccl --emit-llvm-boundary-bundle examples/c-boundary/boundary.json --host-llvm build/c-boundary/host.ll --device-llvm build/c-boundary/device.ll -o build/c-boundary/manual-bundle
```

Scope: one configured kernel, (__global float*, float), one direct f32 store,
straight-line scalar math and a [1,1,1] global/local launch with zero offset.
Pointer arithmetic, buffer reads, barriers and other address spaces are rejected.
The request must directly forward the descriptor and resource in the declared
order; arithmetic or additional callers of the opaque handles are rejected.
Names are configuration, not hard-coded replacements for the source calculation.
The tests change x*x to x+x and check that the emitted shader changes.

Both frontend invocations use the 32-bit little-endian SPIR layout for this
compiler probe. Host explicitly selects the C calling convention. This is not
the final C11 ILP32 Host TargetInfo or libc implementation. `request_square`
exercises the internal compiler boundary; it does not implement clSetKernelArg,
queue ownership, upload or readback. Add remains a separate public scalar probe.

Successful request validation prints `Descriptor contract valid; no GPU dispatch
executed`. Udon assembly, Unity import and actual GPU execution remain pending;
`ready_for_unity` stays false. Tests execute original/recovered Host scalar code
against a native C stub, validate source/config rejection cases and optionally
compile the emitted standalone HLSL with glslangValidator/spirv-val.
