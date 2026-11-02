# UdonCL examples

The examples are ordered from the smallest supported user workflow to the
largest ONNX plan. They use public CMake, compiler, artifact, VPM, and Unity
importer interfaces; test-only fixture commands are not required.

## Common setup

Build the compiler once:

```powershell
python scripts/install_llvm.py
cmake --preset udoncl
python scripts/build.py
```

Install `com.lemolatoon.udoncl` in a VRChat Worlds project through the VPM
repository `https://lemolatoon.github.io/udoncl/index.json`. Unity artifacts
are imported with **Tools → udoncl → Import Artifacts...**.

## Choose a workflow

| Example | User path | Result | Current boundary |
| --- | --- | --- | --- |
| [`hello-kernel`](hello-kernel/README.md) | User CMake Host C + OpenCL C | Generated U# binding and importable scalar/Device artifacts | Fixed AOT dispatch and buffer-copy subsets |
| [`saxpy`](saxpy/README.md) | Standard OpenCL Host API + kernel | Two input vectors, copied i32/f32 arguments, GPU output, public Write/Read API | Fixed buffers and NDRange |
| [`onnx-arena-add`](onnx-arena-add/README.md) | Standard ONNX `Add` | General ONNX runtime prefab plus importable U# Package Sample | One non-Pooling pointwise pass |
| [`tiny-llama`](tiny-llama/README.md) | Standard ONNX small decoder | Importable 18-pass general-runtime artifact and U# Package Sample | CI-sized model, not pretrained 8B weights |
| [`llama-8b-onnx`](llama-8b-onnx/README.md) | Licensed checkpoint + external Q4 weights | 546-operator plan and optional multi-gigabyte Unity artifact | Full materialization requires user-owned weights |
| [`mnist-cnn`](mnist-cnn/README.md) | Trained CNN | Native and Device reference execution | General Unity ONNX Pooling path remains disabled |

## Fastest importable example

After installing the VPM package, the shortest user path needs no compiler
checkout: choose **Tools → udoncl → Examples → Import ONNX Arena Add**, then
choose **Add to Scene**. Import the **ONNX Arena Add** Package Sample for its U#
caller component.

To regenerate the artifact yourself instead, use the source workflow below.

Generate the non-Pooling ONNX artifact:

```powershell
python examples/onnx-arena-add/build.py `
  --output build/onnx-arena-add
```

Import `build/onnx-arena-add/unity-artifact` in Unity. In Package Manager,
select **UdonCL Runtime and ONNX Importer**, open **Samples**, and import
**ONNX Arena Add**. Choose **Add to Scene** in the import completion dialog;
the sample component calls that runtime instance through typed
Write/Run/asynchronous Read methods.

## C and OpenCL C consumer example

Configure `hello-kernel` as an independent user project and build its
importable Device dispatch artifact:

```powershell
cmake -S examples/hello-kernel -B build/hello-kernel-cmake -G Ninja `
  -DUdonCL_DIR="$PWD/cmake"
cmake --build build/hello-kernel-cmake `
  --target hello-dispatch-unity-library
```

Import
`build/hello-kernel-cmake/udoncl/hello-dispatch-unity-library/artifact`.
The importer creates the Udon program, shader, material, fixed Device buffer,
prefab, and typed U# binding. See the example README for scalar, integer,
buffer-copy, and individual CLI variants.

## Verification

Before committing repository changes, run only:

```powershell
python scripts/precommit.py
```

It rebuilds all examples used as fixtures, runs native execution checks,
starts the configured Unity project through Unity MCP, runs every Editor test,
checks console errors, compiles the package Sample, and verifies cleanup.
