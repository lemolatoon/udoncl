# Hello kernel: build an artifact by hand

For a package-only first run, choose **Tools → udoncl → Examples → Import C
and OpenCL Hello Dispatch**. Unity imports the same artifact shape described
below and generates `UdonclHelloDispatchBinding.prefab` after script reload.
The source workflow remains the reproducible way to edit the Host C and OpenCL
C inputs yourself.

The Device kernel computes
`x * x + 1` and writes the result to its one-element output buffer. `host.c`
contains both a matching scalar Host export and the restricted Host-to-Device
dispatch request, so both halves of the current compiler boundary are visible.

## Build through a user CMake project

`CMakeLists.txt` is a standalone consumer of the public UdonCL package. It uses
three Host C files and three Device OpenCL C files across two targets of each
kind, so both the multi-translation-unit and multi-target paths are exercised.
After building the repository compiler, configure and build the example from
the repository root:

```powershell
cmake -S examples/hello-kernel -B build/hello-kernel-cmake -G Ninja `
  -DUdonCL_DIR="$PWD/cmake"
cmake --build build/hello-kernel-cmake --target hello-artifact
cmake --build build/hello-kernel-cmake --target hello-unity-library
cmake --build build/hello-kernel-cmake --target hello-dispatch-unity-library
cmake --build build/hello-kernel-cmake --target hello-int-dispatch-unity-library
```

The bundle is generated under
`build/hello-kernel-cmake/udoncl/hello-artifact/artifact`. That boundary bundle
contains the Device shader but remains a compiler/debug artifact while GPU Host
continuation lowering is pending. The separately built importable scalar Host
library is under
`build/hello-kernel-cmake/udoncl/hello-unity-library/artifact`; select that
directory in **Tools → udoncl → Import Artifacts...**. Unity generates
`UdonclHelloKernelBinding`, whose `SquarePlusOne(float value)` method calls the
compiler-emitted Udon Assembly.

The `hello-dispatch-unity-library` target compiles `dispatch.c` through the
public Host C path and the normal multi-file `hello_device` targets named in
`DEVICE_LIBRARIES`. It packages `square_plus_one` as
`kernels/kernel_1.shader` and `.hlsl`, and exports `DispatchSquare(float)`.
Import
`build/hello-kernel-cmake/udoncl/hello-dispatch-unity-library/artifact` to call
the generated Device kernel from the generated U# binding. The repository
validation calls `DispatchSquare(3.0f)` through SDK IUdonVM and expects the
one-float GPU output to be exactly `10.0f` after both import and reimport.

This is deliberately a small AOT OpenCL 1.2 subset: one platform, one GPU
device, one context and in-order queue, one fixed four-byte output buffer, a
statically known program/kernel, buffer argument 0, copied f32 argument 1, and
a one-work-item 1D enqueue. The calls must be direct and straight-line and
must match `dispatch-library.json`. Runtime source compilation, dynamic
NDRange sizes, other argument layouts, readback, and general handle lifetime
remain compile errors rather than implicit fallbacks.

`hello-int-dispatch-unity-library` exercises the same public path with an i32
scalar and `square_plus_one_i32`. Its generated wrapper exports
`DispatchSquareInt(int)`, the Udon program uses `Material.SetInteger`, and the
HLSL/ShaderLab uniform is emitted as `int`/`Integer`. The output is still raw
four-byte RGBA8 storage; i32 support does not imply a broader buffer or NDRange
contract.

A packaged compiler installation finds its tools beside the CMake package. The source-tree package
also discovers the built `udonclc` and pinned `.deps/llvm23` tools. Explicit
`UDONCLC_EXECUTABLE`, `UDONCL_CLANG_EXECUTABLE`, and
`UDONCL_LLVM_LINK_EXECUTABLE` cache paths override discovery.

`unity-library.json` is the stable public ABI description for the scalar Host
library. Its `buffers` array reserves explicit linear RGBA8 Device backing;
the compiler exports the corresponding Udon heap slots and the Importer creates
and binds Point-filtered RenderTextures in the generated prefab. Helper calls must currently be inlined; unsupported pointer,
continuation, or GPU-dispatch exports fail compilation instead of publishing a
partly runnable Unity artifact.

The second Unity target exercises the initial explicit Device-memory subset
from C source:

```powershell
cmake --build build/hello-kernel-cmake --target hello-buffer-copy-unity-library
```

`buffer_copy.c` includes the installed `<CL/cl.h>`, creates one OpenCL 1.2
platform/device/context/in-order queue sequence, allocates three fixed
32,772-byte (8192×2 texel) buffers with `clCreateBuffer`, and copies source to
intermediate and then intermediate to destination with
`clEnqueueCopyBuffer`. The generated
artifact is under
`build/hello-kernel-cmake/udoncl/hello-buffer-copy-unity-library/artifact`.
This restricted AOT bridge accepts direct context/queue/buffer flow in a
straight-line setup function. Status-dependent cleanup/control flow, general
`cl_mem` lifetime, release/reuse, upload/readback, and partial copy remain
pending and are rejected rather than approximated.

The compiler package installs the pinned Khronos OpenCL headers, so Host
targets can include `<CL/cl.h>` without adding an include directory. The Unity
lowering accepts the restricted sequence above; this is not general OpenCL Host
API coverage.

## Build by invoking each CLI

The remainder deliberately avoids a wrapper script. Run each public CLI
yourself from a PowerShell opened at the repository root.

Configure and build the pinned toolchain first:

```powershell
cmake --preset udoncl
python scripts/build.py
```

Name the tools and create a work directory:

```powershell
$clang = ".deps/llvm23/bin/clang.exe"
$translate = "build/udoncl/tools/vrc-translate/udoncl-translate.exe"
$opt = "build/udoncl/tools/vrc-opt/udoncl-opt.exe"
$compile = "build/udoncl/tools/vrccl/udonclc.exe"
$out = "build/hello-kernel"
New-Item -ItemType Directory -Force $out | Out-Null
```

## 1. Compile Host C and OpenCL C to LLVM IR

Compile the Host with the same temporary 32-bit SPIR layout used by the current
boundary prototype:

```powershell
& $clang -target spir-unknown-unknown -x c -std=c11 `
  -Xclang -fdefault-calling-conv=cdecl -O0 `
  -Xclang -disable-O0-optnone -fomit-frame-pointer -ffp-contract=off `
  -S -emit-llvm examples/hello-kernel/host.c -o "$out/host.ll"
```

Compile the Device kernel:

```powershell
& $clang -target spir-unknown-unknown -x cl '-cl-std=CL1.2' `
  -cl-kernel-arg-info -O0 -Xclang -disable-O0-optnone `
  -fomit-frame-pointer -ffp-contract=off -S -emit-llvm `
  examples/hello-kernel/kernel.cl -o "$out/kernel.ll"
```

`host.ll` and `kernel.ll` are the portable 32-bit SPIR LLVM inputs consumed by
the current Host/Device boundary compiler.

## 2. Inspect the LLVM-to-MLIR import

```powershell
& $translate --import-llvm "$out/host.ll" `
  --metadata-output "$out/host.frontend.json" `
  -o "$out/host.imported.mlir"

& $translate --import-llvm "$out/kernel.ll" `
  --metadata-output "$out/kernel.frontend.json" `
  -o "$out/kernel.imported.mlir"
```

This standalone import is useful for inspecting the frontend result and its
metadata. The artifact compiler performs the same import with `kernel.json`
attached as the AOT kernel contract.

## 3. Generate the immutable Device-only artifact

Use a new artifact directory when changing the source or configuration. The
compiler intentionally refuses to overwrite different existing content.

```powershell
& $compile --emit-native-device-bundle examples/hello-kernel/kernel.json `
  --device-llvm "$out/kernel.ll" -o "$out/artifact"
```

The important outputs are:

```text
build/hello-kernel/artifact/manifest.json
build/hello-kernel/artifact/frontend.imported.mlir
build/hello-kernel/artifact/frontend.recovered.mlir
build/hello-kernel/artifact/host-device.mlir
build/hello-kernel/artifact/kernels/kernel_1.hlsl
build/hello-kernel/artifact/kernels/kernel_1.shader
```

## 4. Replay the Device recovery passes with the pass driver

`udoncl-opt` lets you inspect the same proof-producing recovery stage outside
the artifact driver:

```powershell
$pipeline = 'builtin.module(builtin.module(mem2reg,vrc-raise-llvm-boundary-device,vrc-raise-llvm-to-scalar,canonicalize),vrc-prepare-native-device)'
& $opt "--pass-pipeline=$pipeline" --verify-each `
  "$out/artifact/frontend.imported.mlir" `
  -o "$out/kernel.recovered-by-hand.mlir"
```

Compare `kernel.recovered-by-hand.mlir` with
`artifact/frontend.recovered.mlir`. Formatting or location details may differ,
but both must contain the recovered `gpu.func @square_plus_one` kernel.

## 5. Link the Host and Device boundary

Generate a second artifact from both LLVM modules. This recovers the scalar Host
export, proves that the request only forwards opaque handles, links it to kernel
ID 1, and emits the Device shader:

```powershell
& $compile --emit-llvm-boundary-bundle examples/hello-kernel/kernel.json `
  --host-llvm "$out/host.ll" --device-llvm "$out/kernel.ll" `
  -o "$out/host-device-artifact"
```

Inspect the linked boundary:

```powershell
Select-String -Path "$out/host-device-artifact/host-device.mlir" `
  -Pattern 'host_square_plus_one|request_square_plus_one|vrc.dispatch|vrcgpu.kernel'
```

The Host export has `vrc.export_id = 1`; the internal request uses typed
`!vrc.buffer`, `!vrc.dispatch_desc`, and `!vrc.event` values rather than treating
runtime handles as ordinary integers.

## 6. Validate a concrete Host dispatch request

`request.json` represents `x = 2.0f` as a copied little-endian scalar payload
and a one-word output buffer. Validate it against the generated manifest:

```powershell
& $compile --check-boundary-request `
  "$out/host-device-artifact/manifest.json" `
  --request examples/hello-kernel/request.json
```

The command prints `Descriptor contract valid; no GPU dispatch executed`.

## Unity boundary

This lower-level procedure produces Device and linked Host/Device boundary
artifacts whose manifests have `ready_for_unity: false`; they remain useful for
inspecting recovery. Use the CMake `hello-dispatch-unity-library` target above
to run the same kernel through the restricted executable Unity path.
