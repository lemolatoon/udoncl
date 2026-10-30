# Native Host and Fragment Shader execution

This flow compiles the real C/OpenCL C boundary example, lowers its UdonHL Host
back through LLVM IR into a native object, links a graphics runtime library,
then calls the generated Host code to render and read back the generated Device
fragment shader. It does not execute a replacement CPU kernel in the renderer.

On Ubuntu install the graphics test dependencies:

```text
sudo apt-get install libvulkan-dev mesa-vulkan-drivers glslang-tools
make example-native-fragment
```

Windows uses the Windows SDK's Direct3D 11, D3DCompile and WARP software device.
Run from the Visual Studio Developer PowerShell used to build the compiler:

```text
python examples/native-fragment/run.py
```

On either platform, `--build-dir path/to/compiler/build` selects a different
compiler build and its configured LLVM SDK. `--output path/to/output` selects a
fresh output location. With Make use `NATIVE_OUTPUT=build/another-native-run`.
Identical compiler bundles can be reused; a different source requires a new
output directory. The compiler does not overwrite differing existing bundles.

The generated native executable prints eight comparisons, starting with:

```text
renderer=Vulkan llvmpipe (...) (software)
input=2.5 expected=0x40c80000 actual=0x40c80000
...
passed=8 native UdonHL dispatches and fragment readbacks
```

On Windows the renderer line identifies Direct3D11 WARP. The eight values include
negative values, signed zero and exactly representable noninteger f32 values.
The same Device source is independently compiled as ordinary native C (only
`__kernel` and `__global` are removed) to supply the comparison oracle. The
renderer always uses the generated shader and actual graphics draw/readback.

## Lowering and linking

```text
host.c → LLVM → high-level recovery → udonhl.call / vrc.dispatch
  → vrc-lower-host-to-native-test (MLIR dialect conversion)
  → func / arith
  → upstream func/arith-to-LLVM conversion → MLIR LLVM dialect
  → LLVM IR → Clang → host.o / host.obj
  → link with VRCNativeFragmentRuntime + harness → native executable

device.cl → LLVM → high-level recovery → vrcgpu → HLSL
  → Linux: glslang → SPIR-V → Vulkan graphics pipeline on Lavapipe
  → Windows: D3DCompile ps_5_0 → D3D11 graphics pipeline on WARP
```

The native test Pass requires a verified paired Host/Device boundary. Normal
`udonhl.call` becomes `func.call`; recursive/suspending call policies are rejected.
The runtime wrappers `!vrc.buffer`, `!vrc.dispatch_desc`, `!vrc.event` become i32,
while `vrc.dispatch` becomes:

```c
uint32_t __vrc_native_test_dispatch(uint32_t kernel_id,
                                  uint32_t descriptor_handle,
                                  uint32_t buffer_handle);
```

The C symbol is implemented by `VRCNativeFragmentRuntime`, an independent static
library under `lib/Testing/NativeFragment`. It has no LLVM/MLIR dependency. The
library owns the renderer, bindings, descriptor copy, handle/generation checks
and completion identity. The executable only registers test inputs, calls the
generated Host functions, and checks results. No exception unwinds through the
generated Host frames; runtime errors produce the reserved zero event.

`Bindings.h` is generated from Manifest argument bindings/texture settings and
the configured C symbol roles. The supported contract is checked before rendering:
`_VRC_Arg1`, four output bytes, one 1x1 RGBA8_UNorm render target, no sRGB/mipmaps,
no blending. Linux binds the HLSL globals uniform block at descriptor set 0,
binding 0; Windows binds its constant buffer at slot b0. These graphics API slots
belong to this adapter; Manifest resource argument order is preserved separately.
There is no sampling in this kernel, so texture filtering has no effect here.

The internal request function is exposed only in the native test module so the
harness can call it. The saved compiler bundle retains its original private
visibility and typed opaque handles. The Host native target uses its actual
machine layout while handles remain explicit uint32 values. The temporary SPIR32
source layout is not claimed to be the final C11 Host TargetInfo.

## Artifacts and tests

The default output is `build/native-fragment`:

- `compile/bundle/host-device.mlir`: UdonHL/VRC/VRCGPU boundary before native lowering.
- `host.native.mlir`: Host func/arith after native test lowering.
- `host.native.llvm.mlir`, `host.native.ll`: LLVM dialect and actual LLVM IR.
- `host.o` / `host.obj`, `runtime-build/native-fragment[.exe]`: generated Host object and linked executable.
- `runtime-build/native-runtime/`: the independently built runtime library.
- `commands.json`, `report.json`: commands, diagnostics and execution evidence.

Enable the CTest graphics flow explicitly:

```text
cmake --preset udoncl -DVRC_ENABLE_NATIVE_FRAGMENT_TESTS=ON
python scripts/build.py
ctest --preset udoncl
```

CI enables this option. CTest checks square and a source mutation to x+x: 16
actual draws/readbacks, eight invalid handle/kernel/generation requests, and
four native-lowering rejection cases. Linux uses the Mesa CPU device explicitly,
so a physical GPU or display server is not needed. Windows uses WARP explicitly.

This is a synchronous, serialized native validation adapter with one buffer and
one descriptor. It does not implement the OpenCL queue API, suspension, Udon
assembly, Unity ShaderLab import, VRChat readback behavior, or the full target
runtime. Those gates remain separate, and `ready_for_unity` remains false.

For ordinary OpenCL Host API usage, run `make example` and see
[the OpenCL Host example](../opencl-host/README.md). This native-fragment example
remains a lower-level internal dispatch probe.
