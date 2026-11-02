# Hello kernel: build an artifact by hand

This example deliberately has no wrapper script. Run each public CLI yourself
from a PowerShell opened at the repository root. The kernel computes `x * x + 1`
and writes the result to its one-element output buffer.

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

## 1. Compile OpenCL C to LLVM IR

```powershell
& $clang -target spir-unknown-unknown -x cl '-cl-std=CL1.2' `
  -cl-kernel-arg-info -O0 -Xclang -disable-O0-optnone `
  -fomit-frame-pointer -ffp-contract=off -S -emit-llvm `
  examples/hello-kernel/kernel.cl -o "$out/kernel.ll"
```

`kernel.ll` is the portable 32-bit SPIR LLVM input consumed by UdonCL.

## 2. Inspect the LLVM-to-MLIR import

```powershell
& $translate --import-llvm "$out/kernel.ll" `
  --metadata-output "$out/frontend.json" `
  -o "$out/kernel.imported.mlir"
```

This standalone import is useful for inspecting the frontend result and its
metadata. The artifact compiler performs the same import with `kernel.json`
attached as the AOT kernel contract.

## 3. Generate the immutable Device artifact

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

## 4. Replay the recovery passes with the pass driver

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

## Unity boundary

This procedure produces a real compiler Device artifact. Its manifest has
`artifact_stage: native-device-prototype` and `ready_for_unity: false`, so the
Unity package correctly refuses to import it as a runnable prefab. A runnable
Unity artifact additionally needs a Host Udon Assembly program, display shader,
runtime resources, and a supported Unity-stage manifest. Those are packaging
and runtime inputs, not something the Device compiler should silently invent.
