# udoncl

udoncl is an ahead-of-time compiler, under development, that turns C, OpenCL C,
and ONNX models into VRChat Worlds artifacts. Host code becomes Udon Assembly,
and OpenCL C kernels and ONNX operators become fragment-shader kernels that run
on the GPU. A Unity package imports the artifacts and generates typed
UdonSharp bindings.

The application entry is UdonSharp; C is a library. Your UdonSharpBehaviour
calls the exported C functions through a generated binding, uploads and reads
back Device buffers, and owns the world logic. [PLAN.md](PLAN.md) is the design
specification. The implemented fixed AOT subsets compile Host C and OpenCL C
into importable VRChat Worlds artifacts; the full PLAN and general OpenCL/C11
coverage remain in progress.

## Try it without building

The Unity runtime and artifact importer are distributed as the prerelease VPM
package `com.lemolatoon.udoncl`. You can try prebuilt examples without building
the compiler:

1. In the VRChat Creator Companion, add the community repository
   `https://lemolatoon.github.io/udoncl/index.json`.
2. Enable prerelease packages in VCC (**Show Pre-Release Packages**) to install
   beta versions, then add **UdonCL Runtime and ONNX Importer** to a Worlds
   project.
3. In Unity, choose **Tools → udoncl → Examples** and one of:
   - **Import ONNX Arena Add**, then **Add to Scene**. Import the **ONNX Arena
     Add** Package Sample for its U# caller component.
   - **Import C and OpenCL Hello Dispatch** or **Import C and OpenCL SAXPY**.
     These generate typed U# binding prefabs after Unity reloads the imported
     wrapper; choose **Add after Compile** to place one in the active scene.

See [examples/README.md](examples/README.md#fastest-importable-example) for
details.

## What works today

- **Host C and OpenCL C:** fixed AOT subsets. Scalar Host C exports become typed
  U# methods, and a restricted straight-line OpenCL 1.2 Host sequence
  (constant-sized buffers, copied f32/i32 arguments, one statically configured
  NDRange) dispatches an OpenCL C kernel as a fragment shader.
- **Buffers:** fixed-size Device buffers with whole-buffer upload and
  asynchronous readback through the generated U# binding.
- **ONNX:** standard ONNX models compile to Unity artifacts, from a single
  `Add` up to the real TinyLlama-1.1B model with 512- and 4096-token
  conversation artifacts.
- **Libraries:** CUDA-compatible subsets of UdonBLAS (cuBLAS) and UdonDNN
  (cuDNN), plus UdonTokenizers for Hugging Face tokenizers.
- **Validation:** artifacts are validated in the Unity Editor (D3D11 on
  Windows) with the VRChat Worlds SDK. This is not a VRChat client claim.
- **Not supported yet:** dynamic NDRange sizes, general OpenCL Host semantics,
  and general C11. Unsupported code fails to compile instead of being
  approximated.

The detailed component list and pending work are in
[docs/architecture/compiler-status.md](docs/architecture/compiler-status.md).

## Prerequisites

- CMake 3.30+, Ninja, Python 3.12+, and a C++17 compiler (C++20 for the native
  renderer examples). Git is required when building LLVM from source.
- LLVM, MLIR and Clang are pinned together to **23.1.4** (commit recorded in
  [cmake/llvm-lock.json](cmake/llvm-lock.json)) and installed by the scripts
  below; no system-wide LLVM installation is required. Compiler development
  targets Linux, macOS and Windows x64; the initial generated world targets
  VRChat PC (Windows).
- Unity 2022.3 with the VRChat Worlds SDK and the `com.lemolatoon.udoncl` VPM
  package to import artifacts.

**Windows:** for the prebuilt Windows SDK, install **Visual Studio 2026** with
Desktop development with C++. In an ordinary PowerShell, activate its CMake,
Ninja, compiler, Windows SDK include paths and libraries once per terminal:

```powershell
. .\scripts\activate_msvc.ps1
```

The locked SDK uses newer MSVC STL symbols unavailable in Visual Studio 2022;
a VS 2022 source build must use an SDK built with that toolset instead.

**Ubuntu:** install the SDK's system libraries and the Vulkan/glslang packages
used by native fragment execution before configuring:

```text
sudo apt-get install zlib1g-dev libzstd-dev libxml2-dev glslang-tools libvulkan-dev mesa-vulkan-drivers
```

## Install and build

Install the pinned toolchains, then configure, build, and test the compiler
from the repository root:

```text
python -m pip install -r scripts/requirements-llvm.txt
python -m pip install -r scripts/requirements-onnx.txt
python scripts/install_llvm.py
python scripts/install_llvm.py --check
python scripts/install_onnx_mlir.py --jobs 2
python scripts/install_onnx_mlir.py --check
cmake --preset udoncl
python scripts/build.py
ctest --preset udoncl
```

Once installed, subsequent builds only need configure, build, and test.
Everything else, including the formatter and test requirements, the TinyLlama
fixtures, the `.deps` / `build` layout, LLVM discovery, building without
presets, clangd, and the commit validation service, is in
[docs/development.md](docs/development.md).

## Using UdonCL from your CMake project

User CMake projects can consume the compiler with `find_package(UdonCL CONFIG)`.
Host C and Device OpenCL C are ordinary source-library targets; the artifact
target compiles every translation unit, links one Host and one Device LLVM
module, then invokes `udonclc`:

```cmake
find_package(UdonCL 0.1 REQUIRED CONFIG)
udoncl_add_host_library(my_host host.c host_helpers.c)
udoncl_add_device_library(my_device kernel.cl device_helpers.cl)
udoncl_add_artifact(my_artifact
  CONFIG kernel.json
  HOST_LIBRARIES my_host
  DEVICE_LIBRARIES my_device)

# A scalar Host library is directly importable by the Unity package.
udoncl_add_unity_library(my_unity_library
  CONFIG unity-library.json
  HOST_LIBRARIES my_host)
```

When configuring against a source checkout, point CMake at the compiler's
package with `-DUdonCL_DIR=<path-to-udoncl>/cmake`, as the examples do. The
package also honors `target_include_directories`, `target_compile_definitions`,
and `target_compile_options` usage requirements on those source targets. Build
`my_artifact` to generate its immutable bundle.

`udoncl_add_unity_library` also accepts `DEVICE_LIBRARIES` targets and packages
the selected AOT kernel in the Unity artifact. The Unity-library config assigns
stable export IDs, public U# names and parameter names. Optional explicit
`buffers` entries become canonical linear RGBA8 RenderTexture assets bound to
compiler-emitted Udon heap slots. The config does not list the UdonCL Device
library kernels a Host dispatches (an ONNX Host library, or C through
UdonBLAS, UdonDNN, UdonATen or UdonONNXOperators): `udonclc` derives each
kernel's `device_kernels` entry from its dispatch sites and the library
kernel contract, and a hand-written entry replaces the derived one of its
`kernel_id` (see "Library kernel entries" in
[docs/architecture/refactor-baseline-and-mlir-pipeline.md](docs/architecture/refactor-baseline-and-mlir-pipeline.md)).
The same holds for OpenCL C kernels of your own `DEVICE_LIBRARIES` that a
Host C launches with `clCreateKernel`, `clSetKernelArg` and a constant-size
`clEnqueueNDRangeKernel` through the linked runtime
(`HOST_LIBRARIES ... UdonCL::OpenCLRuntime`); see "User OpenCL C kernels"
in the same document.
Its artifact contains `manifest.json`,
compiler-emitted `host.uasm`, and a valid display shader; import it with
**Tools → udoncl → Import Artifacts...**. CMake preset support and the future
Unity directory-picker UI are deliberately separate.

### Tokenizers

Projects generate and link Hugging Face tokenizer tables through the
UdonTokenizers CMake API:

```cmake
find_package(UdonTokenizers 0.1 REQUIRED CONFIG)
udontokenizers_add_huggingface_model(my_tokenizer
  TOKENIZER_DIRECTORY "${tokenizer_directory}"
  SYMBOL my_tokenizer
  SOURCES tokenizer_host.c)
udoncl_add_unity_library(my-tokenizer-unity-library
  CONFIG tokenizer-library.json
  HOST_LIBRARIES my_tokenizer UdonTokenizers::UdonTokenizers)
```

`TOKENIZER_DIRECTORY` contains the four verified Hugging Face assets. Generated
`.c`, `.h`, and manifest files remain in the consumer build tree. For TinyLlama,
`python scripts/fetch_tinyllama_tokenizer.py` fetches and verifies the
tokenizer assets (and `python scripts/fetch_tinyllama_onnx.py` fetches the
model).

## Tutorials

### Write your own Host C and OpenCL C kernel (SAXPY)

This tutorial builds the [`examples/saxpy`](examples/saxpy/README.md) project
from source, imports it into Unity, and calls it from your own
UdonSharpBehaviour. The kernel computes `z[i] = alpha * x[i] + y[i]` for 16
elements on the GPU. Build the compiler first as described in
[Install and build](#install-and-build).

#### 1. The files you write

A UdonCL project has four inputs: an OpenCL C kernel, Host C that launches it,
a `CMakeLists.txt`, and a `unity-library.json` that describes the public U#
API.

**The OpenCL C kernel** ([`saxpy.cl`](examples/saxpy/saxpy.cl)). The current
Device profile, `gather-f32`, runs one work item per word of the output
buffer, and every work item writes exactly its own word `result[i]`. All three
buffers hold 32 words; the words at or beyond `n` keep `y`:

```c
__kernel void saxpy(__global const float *x, __global const float *y,
                    __global float *result, int n, float alpha) {
  uint i = (uint)get_global_id(0);
  float value = y[i];
  if (i < (uint)n)
    value = alpha * x[i] + value;
  result[i] = value;
}
```

**The Host C** ([`host.c`](examples/saxpy/host.c)). Host code uses the
ordinary Khronos OpenCL 1.2 API, and `UDONCL_EXPORT` makes `run_saxpy` a U#
method. It creates the buffers and the kernel, sets the kernel arguments,
launches it and waits for it, and returns the OpenCL status. The essential
part is:

```c
static cl_mem x;
static cl_mem y;
static cl_mem result;

UDONCL_EXPORT int run_saxpy(int n, float alpha) {
  /* ... clGetPlatformIDs, clGetDeviceIDs, clCreateContext,
     clCreateCommandQueue, each followed by a status check ... */
  x = clCreateBuffer(context, CL_MEM_READ_ONLY, SAXPY_WORDS * sizeof(float), 0,
                     &status);
  /* ... y (CL_MEM_READ_ONLY) and result (CL_MEM_WRITE_ONLY) likewise ... */
  program = clCreateProgramWithSource(context, 1, &source, 0, &status);
  status = clBuildProgram(program, 0, 0, 0, 0, 0);
  kernel = clCreateKernel(program, "saxpy", &status);
  if ((status = clSetKernelArg(kernel, 0, sizeof(x), &x)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 1, sizeof(y), &y)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 2, sizeof(result), &result)) !=
          CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 3, sizeof(n), &n)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 4, sizeof(alpha), &alpha)) != CL_SUCCESS)
    goto cleanup;
  status =
      clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global, &local, 0, 0, 0);
  if (status != CL_SUCCESS)
    goto cleanup;
  status = clFinish(queue);
cleanup:
  /* ... release the kernel, program, buffers, queue, and context ... */
  return status;
}
```

Kernels are not compiled at run time: the kernel body comes from the Device
library in `CMakeLists.txt`, and the build derives the launch from the Host
C. That works within limits: every buffer argument precedes the
`int`/`uint`/`float` scalars, each argument has a dominating
`clSetKernelArg`, the kernel object comes from one `clCreateKernel` of a
constant name, and the global work size is one constant. A buffer whose
address is passed to the runtime lives in a Host global. Unsupported patterns
are compile errors rather than approximations.

**The CMake project** ([`CMakeLists.txt`](examples/saxpy/CMakeLists.txt)) is
an ordinary consumer of the UdonCL package. `UdonCL::OpenCLRuntime` is the
OpenCL implementation the Host C runs on, and `OMIT_INTERMEDIATE_IR` leaves
the intermediate MLIR, which Unity does not import, out of the artifact:

```cmake
cmake_minimum_required(VERSION 3.30...4.2)
project(udoncl_saxpy LANGUAGES NONE)

find_package(UdonCL 0.1 REQUIRED CONFIG)

udoncl_add_host_library(saxpy_host host.c)
udoncl_add_device_library(saxpy_device saxpy.cl)

udoncl_add_unity_library(saxpy-unity-library
  CONFIG unity-library.json
  OMIT_INTERMEDIATE_IR
  HOST_LIBRARIES saxpy_host UdonCL::OpenCLRuntime
  DEVICE_LIBRARIES saxpy_device)
```

**The public API description**
([`unity-library.json`](examples/saxpy/unity-library.json)). Each key you
write:

- `schema_version`: `2`.
- `program`: the program name, `saxpy_demo`. Unity imports the artifact under
  `Assets/udoncl-generated/saxpy_demo` and names the generated U# class
  `UdonclSaxpyDemoBinding` (`Udoncl` + the PascalCase program name +
  `Binding`) in the namespace `Lemolatoon.Udoncl.Generated`. An optional
  `binding` object with `namespace` and `class_name` overrides both.
- `export_overrides`: optional names for the `UDONCL_EXPORT` functions.
  `symbol` is the C name (`run_saxpy`), `public_name` the U# name
  (`RunSaxpy`), and `parameters` the U# parameter names (`n`, `alpha`). The
  parameter and result types come from the C signature.
- `buffers`: one entry per OpenCL buffer, in `clCreateBuffer` order. `id` is
  the buffer ID, `name` produces the U# transfer method names
  (`x` → `TryWriteX`, `result` → `StartReadResult`), `byte_size` is the
  `clCreateBuffer` size (128), `element_type` is `"float32"`, and
  `public_access` is `"input"` for a buffer U# writes or `"output"` for one it
  reads.

#### 2. Build

From the repository root, build the compiler and then the consumer project:

```powershell
cmake --preset udoncl
python scripts/build.py
cmake -S examples/saxpy -B build/saxpy-cmake -G Ninja `
  -DUdonCL_DIR="$PWD/cmake"
cmake --build build/saxpy-cmake --target saxpy-unity-library
```

The importable artifact is written to:

```text
build/saxpy-cmake/udoncl/saxpy-unity-library/artifact
```

#### 3. Import into Unity

1. Open a Unity 2022.3 VRChat Worlds project and add the
   **UdonCL Runtime and ONNX Importer** package (`com.lemolatoon.udoncl`) from
   the VPM repository, as in [Try it without building](#try-it-without-building).

   > **Screenshot (pending):** VCC project view with UdonCL Runtime and ONNX Importer added — docs/images/getting-started/saxpy-vcc-package.png

2. Choose **Tools → udoncl → Import Artifacts...** and select the
   `build/saxpy-cmake/udoncl/saxpy-unity-library/artifact` directory.

   > **Screenshot (pending):** Tools → udoncl → Import Artifacts… with the SAXPY artifact folder selected — docs/images/getting-started/saxpy-import-artifacts.png

3. In the **udoncl Library Import** dialog, choose **Add after Compile**.
   After the generated U# wrapper compiles, Unity creates
   `Assets/udoncl-generated/saxpy_demo/Prefabs/UdonclSaxpyDemoBinding.prefab`
   and places an instance in the active scene. With **Import Only**, drag that
   prefab into the scene yourself. The importer also creates the Udon program,
   the kernel material, and three 128-byte linear RGBA8 RenderTextures for the
   buffers; the generated binding source is
   `Assets/udoncl-generated/saxpy_demo/Bindings/UdonclSaxpyDemoBinding.cs`.

   > **Screenshot (pending):** udoncl Library Import dialog with Add after Compile highlighted — docs/images/getting-started/saxpy-add-after-compile.png

   > **Screenshot (pending):** Project window showing Assets/udoncl-generated/saxpy_demo with its Bindings and Prefabs folders — docs/images/getting-started/saxpy-generated-assets.png

4. In the Project window, choose **Create → U# Script**, name it
   `SaxpyCaller`, and replace its contents with the script in the next step.
   This creates both the C# file and its UdonSharp program asset.
5. Create an empty GameObject, add the **Saxpy Caller** component, and drag
   the `UdonclSaxpyDemoBinding` scene instance into its **Binding** field.

   > **Screenshot (pending):** Inspector of the SaxpyCaller GameObject with Binding set to the UdonclSaxpyDemoBinding scene instance — docs/images/getting-started/saxpy-inspector-binding.png

6. Enter Play Mode and watch the Console.

#### 4. Your UdonSharpBehaviour

The generated binding exposes:

- `int TryWriteX(float[] source)` and `int TryWriteY(float[] source)`: upload
  the whole `x` or `y` buffer. The array length must equal the buffer size in
  words (32).
- `int StartRunSaxpy(int n, float alpha, UdonSharpBehaviour receiver, string completionEvent)`:
  starts the `RunSaxpy` export. An export that launches Device work runs
  asynchronously: when it finishes, the binding calls
  `receiver.SendCustomEvent(completionEvent)`, so the event method must be
  `public`.
- `int GetRunSaxpyStatus()`: the status of the finished export, and
  `int TryGetRunSaxpyResult(out int result)`: the value `run_saxpy` returned,
  here the OpenCL status (`0` is `CL_SUCCESS`).
- `int StartReadResult(float[] destination, UdonSharpBehaviour receiver, string completionEvent)`:
  starts an asynchronous GPU readback of the whole `result` buffer into
  `destination` (again exactly 32 words) and calls the completion event like
  `StartRunSaxpy`.
- `int GetLastReadbackStatus()`: the status of the completed readback.
- Status constants `StatusSuccess` (0), `StatusNotInitialized` (-1),
  `StatusBusy` (-2), `StatusInvalidArgument` (-3, for example a wrong array
  length), `StatusAbiMismatch` (-4), `StatusExecutionFailure` (-5), and
  `StatusDisposed` (-6).

`SaxpyCaller.cs`:

```csharp
using Lemolatoon.Udoncl.Generated;
using UdonSharp;
using UnityEngine;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class SaxpyCaller : UdonSharpBehaviour
{
    [SerializeField] private UdonclSaxpyDemoBinding binding;

    private float[] x;
    private float[] y;
    private float[] result;

    private void Start()
    {
        // Run one frame after start-up so every behaviour is initialized.
        SendCustomEventDelayedFrames("RunSaxpyOnce", 1);
    }

    public void RunSaxpyOnce()
    {
        if (binding == null)
        {
            Debug.LogError("SAXPY: assign the UdonclSaxpyDemoBinding instance.");
            return;
        }

        x = new float[32];
        y = new float[32];
        result = new float[32];
        for (int i = 0; i < 32; i++)
        {
            x[i] = i + 1;
            y[i] = 100 + i;
        }

        int status = binding.TryWriteX(x);
        if (status == UdonclSaxpyDemoBinding.StatusSuccess)
            status = binding.TryWriteY(y);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
        {
            Debug.LogError("SAXPY: writing the inputs failed with status " + status);
            return;
        }

        status = binding.StartRunSaxpy(16, 2.0f, this, "OnSaxpyComplete");
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
            Debug.LogError("SAXPY: StartRunSaxpy failed with status " + status);
    }

    public void OnSaxpyComplete()
    {
        int openclStatus = 0;
        int status = binding.GetRunSaxpyStatus();
        if (status == UdonclSaxpyDemoBinding.StatusSuccess)
            status = binding.TryGetRunSaxpyResult(out openclStatus);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess || openclStatus != 0)
        {
            Debug.LogError("SAXPY: RunSaxpy failed with status " + status
                + " and OpenCL status " + openclStatus);
            return;
        }

        status = binding.StartReadResult(result, this, "OnSaxpyReadback");
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
            Debug.LogError("SAXPY: StartReadResult failed with status " + status);
    }

    public void OnSaxpyReadback()
    {
        int status = binding.GetLastReadbackStatus();
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
        {
            Debug.LogError("SAXPY: readback failed with status " + status);
            return;
        }

        for (int i = 0; i < 16; i++)
        {
            float expected = 2.0f * x[i] + y[i];
            Debug.Log("SAXPY: result[" + i + "] = " + result[i] + " (expected " + expected + ")");
        }
    }
}
```

#### 5. Expected result

Each Console line shows `result[i] = 2 * x[i] + y[i]`. With the inputs above,
`x[i] = i + 1` and `y[i] = 100 + i`, so `result[i] = 3 * i + 102`:
`result[0] = 102`, `result[1] = 105`, ..., `result[15] = 147`, each equal to
its `expected` value. Result words 16–31 (not logged) hold a copy of `y`, as
do the words at or beyond a smaller `n`.

> **Screenshot (pending):** Unity Console in Play Mode showing the sixteen SAXPY result lines — docs/images/getting-started/saxpy-console-output.png

The generic buffer-ID methods `TryWriteFloatBuffer` / `TryWriteIntBuffer` and
`StartReadFloatBuffer` / `StartReadIntBuffer` remain available as a low-level
alternative to the named transfers.

### More tutorials

- ONNX + C tutorials (MNIST, and TinyLlama with a 4096-token context) are
  coming once the compiler derives Device kernel declarations automatically.
- Until then, [examples/README.md](examples/README.md#choose-a-workflow) lists
  the current ONNX workflows.

## More examples

[examples/README.md](examples/README.md) catalogues every source-tree example
and its support boundary, from the smallest C and OpenCL C consumer to ONNX
models. It also contains walkthroughs of the
[native and compiler-development examples](examples/README.md#native-and-compiler-development-examples):
the C/OpenCL C boundary probe, native Host + Fragment Shader execution, the
ordinary OpenCL Host API, UdonBLAS, the MD crystal, and the MNIST CNN
reference.

## Documentation

- [PLAN.md](PLAN.md): design specification and overall roadmap.
- [CONTEXT.md](CONTEXT.md): glossary of the project's domain language.
- [docs/development.md](docs/development.md): developer setup, toolchain
  details, and the commit validation service.
- [docs/architecture/compiler-status.md](docs/architecture/compiler-status.md):
  current compiler components, pending work, and internal tool walkthroughs.
- [docs/architecture/refactor-baseline-and-mlir-pipeline.md](docs/architecture/refactor-baseline-and-mlir-pipeline.md):
  codebase structure and the actual MLIR transformation paths.
- [docs/architecture/operator-ownership.md](docs/architecture/operator-ownership.md):
  which public library owns each ONNX/PyTorch operator.
- [docs/architecture/udon-vrchat-shader-constraints.md](docs/architecture/udon-vrchat-shader-constraints.md):
  confirmed U#, Udon, and VRChat shader platform constraints.
- [docs/decisions.md](docs/decisions.md): summary of design decisions.
- [docs/adr/](docs/adr/): architecture decision records.
- [docs/recovery.md](docs/recovery.md): LLVM import and high-level MLIR recovery.
- [docs/boundary.md](docs/boundary.md): the Host/Device boundary prototype.
- [docs/validation.md](docs/validation.md): validation status, gates, and
  evidence.
- [docs/validation/unity-ci.md](docs/validation/unity-ci.md): the local Unity
  Editor test gate.
- [docs/vpm-release.md](docs/vpm-release.md): VPM package setup and publication
  steps for maintainers.
- [docs/dependencies.md](docs/dependencies.md): third-party dependencies and
  redistribution notes.
- [docs/investigations/](docs/investigations/): dated investigation notes.
- [libraries/udonblas/README.md](libraries/udonblas/README.md): the UdonBLAS
  API and its profile limits.
- [unity/Packages/com.lemolatoon.udoncl/README.md](unity/Packages/com.lemolatoon.udoncl/README.md):
  the Unity package, artifact importer, and ONNX runtime.
- [examples/README.md](examples/README.md): example catalogue and walkthroughs.

## License

See [LICENSE](LICENSE).
