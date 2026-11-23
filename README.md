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
3. In Unity, choose **Tools → udoncl → Examples** and one of the two bundled
   artifacts:
   - **Import C and OpenCL SAXPY**: the Host C and OpenCL C kernel of
     `examples/saxpy`, computing `alpha * x + y` on the GPU.
   - **Import ONNX MNIST**: the MNIST CNN compiled from `examples/onnx-mnist`,
     classifying a 28x28 digit.

   Each import generates a typed U# binding prefab after Unity reloads the
   imported wrapper and attaches a generated example controller to it. Choose
   **Add after Compile** to place the prefab in the active scene, enter Play
   Mode, and interact with it to run the example.
4. For a U# component of your own, open Package Manager, select **UdonCL
   Runtime and ONNX Importer**, and import one of its **Samples**:
   - **C and OpenCL SAXPY** (`SaxpyCaller`) calls the SAXPY binding.
   - **ONNX MNIST** (`MnistClassifier`) classifies a hand-drawn seven with the
     MNIST binding.
   - **TinyLlama Chat** (`TinyLlamaChat`) chats with TinyLlama-1.1B. The model
     is too large to bundle, so this sample needs the artifact you build in
     the [TinyLlama tutorial](#chat-with-tinyllama-4096-token-context).

The [tutorials](#tutorials) below show how to build these artifacts
yourself and call them from U#. See
[examples/README.md](examples/README.md#fastest-importable-example) for the
other examples.

## What works today

- **Host C and OpenCL C:** fixed AOT subsets. Scalar Host C exports become typed
  U# methods. Host C on the OpenCL 1.2 API, linked with
  `UdonCL::OpenCLRuntime`, launches your own OpenCL C kernels and the library
  kernels as fragment shaders. The compiler derives each kernel's launch from
  the Host C (constant-size buffers, `int`/`uint`/`float` arguments after the
  buffers, a constant NDRange); you do not declare kernels by hand.
- **Buffers and calls:** fixed-size Device buffers with whole-buffer upload and
  asynchronous readback through the generated U# binding. An export that
  launches GPU work runs asynchronously and reports completion through a U#
  event.
- **ONNX:** standard ONNX models compile through the ONNX dialect into a Host
  library that your own Host C drives through `UdonONNXRuntime`, from the
  MNIST CNN up to the real TinyLlama-1.1B model with 512- and 4096-token chat
  artifacts. The model runs as compiled Udon Assembly and fragment shaders,
  not through an execution-plan interpreter.
- **Libraries:** CUDA-compatible subsets of UdonBLAS (cuBLAS) and UdonDNN
  (cuDNN), plus UdonTokenizers for Hugging Face tokenizers.
- **Validation:** artifacts are validated in the Unity Editor (D3D11 on
  Windows) with the VRChat Worlds SDK. This is not a VRChat client claim.
- **Not supported yet:** dynamic NDRange sizes, general OpenCL Host semantics,
  general C11, and ONNX dimensions without a build-time bound. Unsupported
  code fails to compile instead of being approximated.

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

User CMake projects consume the compiler with `find_package(UdonCL CONFIG)`.
Host C and Device OpenCL C are ordinary source-library targets.
`udoncl_add_unity_library` compiles every translation unit, links one Host and
one Device LLVM module, and invokes `udonclc` to produce an artifact the Unity
package imports:

```cmake
find_package(UdonCL 0.1 REQUIRED CONFIG)
udoncl_add_host_library(my_host host.c host_helpers.c)
udoncl_add_device_library(my_device kernel.cl device_helpers.cl)

udoncl_add_unity_library(my_unity_library
  CONFIG unity-library.json
  OMIT_INTERMEDIATE_IR
  HOST_LIBRARIES my_host UdonCL::OpenCLRuntime
  DEVICE_LIBRARIES my_device)
```

- When configuring against a source checkout, point CMake at the compiler's
  package with `-DUdonCL_DIR=<path-to-udoncl>/cmake`, as the examples do. The
  source targets honor `target_include_directories`,
  `target_compile_definitions`, and `target_compile_options` usage
  requirements.
- `UdonCL::OpenCLRuntime` is the OpenCL 1.2 implementation that Host C calls
  through `<CL/cl.h>`. It launches the OpenCL C kernels of your own
  `DEVICE_LIBRARIES` and the kernels of the UdonCL libraries. A Host that calls
  the OpenCL API without it is rejected with a message naming the calls. A
  scalar Host library that calls no OpenCL API needs neither the runtime nor
  `DEVICE_LIBRARIES`.
- `LIBRARIES` adds a library to both the Host and the Device side, for
  example `UdonBLAS::UdonBLAS` after `find_package(UdonBLAS 0.1 REQUIRED
  CONFIG)`. `OMIT_INTERMEDIATE_IR` leaves the intermediate MLIR, which Unity
  does not import, out of the artifact.
- The Unity-library config (`unity-library.json`) names the program, the
  public U# methods and their parameters, and the public Device buffers; the
  tutorials below explain each key. It does not list the kernels the Host
  launches: `udonclc` derives each kernel's `device_kernels` entry from the
  Host's dispatch sites, both for UdonCL library kernels and for your own
  OpenCL C kernels launched with `clCreateKernel`, `clSetKernelArg`, and a
  constant-size `clEnqueueNDRangeKernel`. A hand-written entry replaces the
  derived one of its `kernel_id` (see "Library kernel entries" and "User
  OpenCL C kernels" in
  [docs/architecture/refactor-baseline-and-mlir-pipeline.md](docs/architecture/refactor-baseline-and-mlir-pipeline.md)).
- An ONNX model becomes a Host library with
  `udoncl_add_onnx_host_library(<target> MODEL <model.onnx> DIMENSION_BOUNDS
  <name>=<value>...)`, which your own Host C drives through
  `UdonONNXRuntime`; see the
  [MNIST tutorial](#compile-your-own-onnx-model-with-host-c-mnist).

Build the target to write the importable artifact to
`<build>/udoncl/<target>/artifact`. It contains `manifest.json`, the
compiler-emitted `host.uasm`, and the generated shaders; import it with
**Tools → udoncl → Import Artifacts...**. CMake preset support and a Unity
directory-picker UI are not provided yet. For the compiler-development
boundary bundles of `udoncl_add_artifact`, which Unity cannot import, see
[docs/development.md](docs/development.md#hostdevice-boundary-bundles).

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

Each tutorial lists the files you write, builds them, imports the artifact
into Unity, and calls it from a complete UdonSharpBehaviour that is also
available as a Package Sample:

1. [SAXPY](#write-your-own-host-c-and-opencl-c-kernel-saxpy): Host C that
   launches an OpenCL C kernel of your own.
2. [MNIST](#compile-your-own-onnx-model-with-host-c-mnist): an ONNX model
   driven by Host C.
3. [TinyLlama](#chat-with-tinyllama-4096-token-context): a 1.1B-parameter
   chat model with a 4096-token context.

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
([`unity-library.json`](examples/saxpy/unity-library.json)):

```json
{
  "schema_version": 2,
  "program": "saxpy_demo",
  "export_overrides": [
    {
      "symbol": "run_saxpy",
      "public_name": "RunSaxpy",
      "parameters": ["n", "alpha"]
    }
  ],
  "buffers": [
    {
      "id": 1,
      "name": "x",
      "byte_size": 128,
      "element_type": "float32",
      "public_access": "input"
    },
    {
      "id": 2,
      "name": "y",
      "byte_size": 128,
      "element_type": "float32",
      "public_access": "input"
    },
    {
      "id": 3,
      "name": "result",
      "byte_size": 128,
      "element_type": "float32",
      "public_access": "output"
    }
  ]
}
```

Each key you write:

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

There is no kernel entry: the build derives the `saxpy` kernel's launch (its
buffers, its `int` and `float` arguments, and the global size of 32) from the
Host C.

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
   Alternatively, import the same script as the **C and OpenCL SAXPY** sample:
   in Package Manager select **UdonCL Runtime and ONNX Importer**, open
   **Samples**, and choose **Import**.
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

`SaxpyCaller.cs` (identical to the **C and OpenCL SAXPY** Package Sample):

```csharp
using Lemolatoon.Udoncl.Generated;
using UdonSharp;
using UnityEngine;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class SaxpyCaller : UdonSharpBehaviour
{
    [SerializeField]
    private UdonclSaxpyDemoBinding binding;

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
            Debug.LogError(
                "SAXPY: RunSaxpy failed with status "
                    + status
                    + " and OpenCL status "
                    + openclStatus
            );
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

### Compile your own ONNX model with Host C (MNIST)

This tutorial builds the [`examples/onnx-mnist`](examples/onnx-mnist/) project:
a small trained CNN, given as a standard ONNX model, that classifies a 28x28
handwritten digit on the GPU. The model's operators compile to UdonBLAS,
UdonDNN and UdonATen kernels; a Host C file of your own creates the model
session and becomes the U# API. Build the compiler first as described in
[Install and build](#install-and-build), including `install_onnx_mlir.py`.

> **Shortcut:** **Tools → udoncl → Examples → Import ONNX MNIST** imports a
> prebuilt copy of the same artifact, with the same program and binding names,
> so the U# component below works with it unchanged. Build it yourself to
> change the model or the Host C.

#### 1. The files you write

An ONNX project has four inputs: the `.onnx` model, Host C that drives it, a
`CMakeLists.txt`, and a `unity-library.json`.

**The model.** [`examples/mnist-cnn/generate_onnx.py`](examples/mnist-cnn/generate_onnx.py)
writes the trained network of the [MNIST CNN reference](examples/mnist-cnn/README.md)
as a standard ONNX model, `Conv → Add → Relu → AveragePool → Flatten → Gemm →
Add`, with the input `image` (`1x1x28x28` float) and the output `logits`
(`1x10` float). The CMake project below runs it at configure time; any other
ONNX file works the same way as long as its operators are supported.

**The Host C** ([`host.c`](examples/onnx-mnist/host.c)). It uses the OpenCL
1.2 API and the `UdonONNXRuntime` session API of `<udononnxruntime.h>`, and
exports three functions. Every export returns a `udononnxStatus_t`
(`UDONONNX_STATUS_SUCCESS` is `0`). The essential parts are:

```c
static cl_context context;
static cl_command_queue queue;
static udonclBufferHandle_t input_buffer;
static udonclBufferHandle_t output_buffer;
static udononnxSession_t session;

/* clCreateBuffer, then wrap the cl_mem as an ONNX tensor of the model's
   shape (udonclDeviceBufferAdopt, udonclDeviceBufferView,
   udononnxCreateTensor). */
static udononnxStatus_t create_buffer(cl_mem_flags flags, size_t elements,
                                      udonclBufferHandle_t *handle,
                                      udononnxTensor_t *tensor, uint32_t rank,
                                      const int64_t *dimensions);

/* Waits for an enqueued session command and reports its completion. */
static udononnxStatus_t complete(cl_event event,
                                 udononnxStatus_t enqueue_status);

UDONCL_EXPORT udononnxStatus_t initialize(void) {
  /* ... udononnxGetModel(0, &model), clGetPlatformIDs, clGetDeviceIDs,
     clCreateContext, clCreateCommandQueue ... */
  status = create_buffer(CL_MEM_READ_ONLY, MNIST_INPUT_ELEMENTS, &input_buffer,
                         &input_tensor, model->inputs[0].rank,
                         model->inputs[0].dimensions);
  /* ... the output buffer (CL_MEM_WRITE_ONLY, 10 elements) likewise ... */
  status = udononnxCreateSession(&session, model, context, input_bindings,
                                 output_bindings);
  /* Uploads the model weights. */
  status = udononnxEnqueueInitialize(&session, queue, 0, 0, &completion);
  status = complete(completion, status);
  status = udononnxSessionBindInput(&session, 0, input_tensor);
  status = udononnxSessionBindOutput(&session, 0, output_tensor);
  /* ... each step checks its status and releases everything on failure ... */
}

UDONCL_EXPORT udononnxStatus_t run(void) {
  cl_event completion = 0;
  udononnxStatus_t status;
  if (!session_created)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udononnxEnqueueRun(&session, queue, 0, 0, &completion);
  return complete(completion, status);
}

UDONCL_EXPORT udononnxStatus_t shutdown(void) {
  /* ... udononnxSessionDispose, udonclDeviceBufferDestroy,
     clReleaseCommandQueue, clReleaseContext ... */
}
```

`initialize` and `run` launch GPU work, so they become
asynchronous U# methods. U# writes the image into the buffer bound as the
model input and reads the buffer bound as the model output; the Host C never
copies tensor data itself.

**The CMake project** ([`CMakeLists.txt`](examples/onnx-mnist/CMakeLists.txt)):

```cmake
cmake_minimum_required(VERSION 3.24)
project(udoncl_onnx_mnist LANGUAGES NONE)

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 CONFIG REQUIRED)
find_package(UdonBLAS 0.1 CONFIG REQUIRED)
find_package(UdonDNN 0.1 CONFIG REQUIRED)
find_package(UdonATen 0.1 CONFIG REQUIRED)
find_package(UdonONNXOperators 0.1 CONFIG REQUIRED)
find_package(UdonONNXRuntime 0.1 CONFIG REQUIRED)

set(model "${CMAKE_CURRENT_BINARY_DIR}/mnist.onnx")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/../mnist-cnn/generate_onnx.py"
          -o "${model}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "MNIST ONNX generation failed:\n${stdout}${stderr}")
endif()

udoncl_add_onnx_host_library(mnist_host MODEL "${model}"
  DIMENSION_BOUNDS batch=1)
target_include_directories(mnist_host INTERFACE
  "${UdonBLAS_INCLUDE_DIRS}"
  "${UdonDNN_INCLUDE_DIRS}"
  "${UdonATen_INCLUDE_DIRS}")
udoncl_add_host_library(mnist_user_host host.c)
target_include_directories(mnist_user_host INTERFACE
  "${UdonONNXRuntime_INCLUDE_DIRS}")
udoncl_add_unity_library(onnx-mnist
  CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/unity-library.json"
  HOST_LIBRARIES
    mnist_host
    mnist_user_host
    UdonONNXRuntime::UdonONNXRuntime
    UdonDNN::OpenCLRuntime
  DEVICE_LIBRARIES
    UdonATen::Device
    UdonBLAS::Device)
```

- `udoncl_add_onnx_host_library` compiles the model, at configure time,
  into the Host library `mnist_host`. `DIMENSION_BOUNDS batch=1` fixes the
  model's symbolic `batch` dimension; every symbolic dimension needs a
  build-time bound.
- `mnist_user_host` is your Host C. `UdonONNXRuntime::UdonONNXRuntime` is the
  session API it calls, and `UdonDNN::OpenCLRuntime` is the linked OpenCL
  runtime with the UdonDNN kernels enabled.
- `UdonATen::Device` and `UdonBLAS::Device` provide the OpenCL C kernels the
  model's operators launch. You do not list the kernels: the build derives
  them from the Host.

**The public API description**
([`unity-library.json`](examples/onnx-mnist/unity-library.json)), without
its `buffers` array (see below):

```json
{
  "schema_version": 2,
  "program": "generic_onnx_mnist",
  "generate_onnx_public_buffers": true,
  "binding": {
    "namespace": "Lemolatoon.Udoncl.Generated",
    "class_name": "UdonclGenericOnnxMnistBinding"
  }
}
```

- `schema_version`: `2`.
- `program`: `generic_onnx_mnist`; Unity imports the artifact under
  `Assets/udoncl-generated/generic_onnx_mnist`.
- `generate_onnx_public_buffers`: `true` makes the compiler create the public
  U# buffers from the model's inputs and outputs, named after the ONNX
  tensors: `image` (784 floats, `TryWriteImage`) and `logits` (10 floats,
  `StartReadLogits`).
- `binding`: the namespace and class name of the generated U# binding,
  `UdonclGenericOnnxMnistBinding`.
- The three `UDONCL_EXPORT` functions are U# methods named after the
  PascalCase of their C names: `initialize` → `Initialize`, `run` → `Run`,
  and `shutdown` → `Shutdown`, none with parameters.
- The generated ONNX Host library also defines helper functions
  (`udon_onnx_binding_*`) that your Host calls. They are never part of your
  U# API, and you do not list them.

You do not list buffers: the compiler derives every Device buffer of an ONNX
Host from the model. The example's config keeps a `buffers` array with the
two public buffers, `image` and `logits`, as a checked record of its U# API.
If you list entries in your own config, each must match the derived buffer,
or the build fails and names the derived value.

#### 2. Build

From the repository root, build the compiler and then the consumer project.
`CMAKE_PREFIX_PATH` lets `find_package` find UdonCL and the library packages
in the checkout's `cmake` directory; configuring generates the model and
compiles it:

```powershell
cmake --preset udoncl
python scripts/build.py
cmake -S examples/onnx-mnist -B build/onnx-mnist-cmake -G Ninja `
  -DCMAKE_PREFIX_PATH="$PWD"
cmake --build build/onnx-mnist-cmake --target onnx-mnist
```

The importable artifact is written to:

```text
build/onnx-mnist-cmake/udoncl/onnx-mnist/artifact
```

#### 3. Import into Unity

1. In a Unity 2022.3 VRChat Worlds project with the
   **UdonCL Runtime and ONNX Importer** package, choose
   **Tools → udoncl → Import Artifacts...** and select the
   `build/onnx-mnist-cmake/udoncl/onnx-mnist/artifact` directory.

   > **Screenshot (pending):** Tools → udoncl → Import Artifacts… with the onnx-mnist artifact folder selected — docs/images/getting-started/mnist-import-artifacts.png

2. In the **udoncl Library Import** dialog, choose **Add after Compile**.
   After the generated U# wrapper compiles, Unity creates
   `Assets/udoncl-generated/generic_onnx_mnist/Prefabs/UdonclGenericOnnxMnistBinding.prefab`
   and places an instance in the active scene.

   > **Screenshot (pending):** Project window showing Assets/udoncl-generated/generic_onnx_mnist with its Bindings and Prefabs folders — docs/images/getting-started/mnist-generated-assets.png

3. Choose **Create → U# Script**, name it `MnistClassifier`, and replace its
   contents with the script in the next step, or import the same script as
   the **ONNX MNIST** sample from Package Manager (**UdonCL Runtime and ONNX
   Importer** → **Samples** → **Import**).

   > **Screenshot (pending):** Package Manager with the ONNX MNIST sample of UdonCL Runtime and ONNX Importer and its Import button — docs/images/getting-started/mnist-package-sample.png

4. Create a GameObject (for example a Cube, whose Collider lets you interact
   with it), add the **Mnist Classifier** component, and drag the
   `UdonclGenericOnnxMnistBinding` scene instance into its **Binding** field.

   > **Screenshot (pending):** Inspector of the MnistClassifier GameObject with Binding set to the UdonclGenericOnnxMnistBinding scene instance — docs/images/getting-started/mnist-inspector-binding.png

5. Enter Play Mode and watch the Console.

#### 4. Your UdonSharpBehaviour

The generated binding follows the same rules as in the SAXPY tutorial:

- `int StartInitialize(UdonSharpBehaviour receiver, string completionEvent)`,
  `int GetInitializeStatus()`, and `int TryGetInitializeResult(out int result)`:
  run `initialize` asynchronously and read its `udononnxStatus_t`.
- `int StartRun(...)`, `int GetRunStatus()`, and
  `int TryGetRunResult(out int result)`: the same for `run`.
- `int TryShutdown(out int result)`: `shutdown`, which launches no GPU
  work and therefore runs synchronously.
- `int TryWriteImage(float[] source)`: upload the 784 pixels of the `image`
  input, row by row.
- `int StartReadLogits(float[] destination, UdonSharpBehaviour receiver, string completionEvent)`
  and `int GetLastReadbackStatus()`: read the 10 `logits` asynchronously.

`MnistClassifier.cs` (identical to the **ONNX MNIST** Package Sample) draws a
seven as text, initializes the model once, classifies the image, and logs the
digit with the largest logit. Interacting with the object classifies the
image again:

```csharp
using Lemolatoon.Udoncl.Generated;
using UdonSharp;
using UnityEngine;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class MnistClassifier : UdonSharpBehaviour
{
    // A hand-drawn 28x28 digit, row by row. The model reads pixel
    // intensities in [0, 1]: '#' is ink (1.0) and '.' is background (0.0).
    private const string Digit =
        "............................"
        + "............................"
        + "............................"
        + "............................"
        + "............................"
        + ".......##############......."
        + ".......##############......."
        + "...................###......"
        + "..................###......."
        + "..................###......."
        + ".................###........"
        + ".................###........"
        + "................###........."
        + "................###........."
        + "...............###.........."
        + "...............###.........."
        + "..............###..........."
        + "..............###..........."
        + ".............###............"
        + ".............###............"
        + "............###............."
        + "............###............."
        + "............###............."
        + "............................"
        + "............................"
        + "............................"
        + "............................"
        + "............................";

    [SerializeField]
    private UdonclGenericOnnxMnistBinding binding;

    private float[] image;
    private float[] logits;
    private bool initialized;

    private void Start()
    {
        // Initialize one frame after start-up so every behaviour is ready.
        SendCustomEventDelayedFrames("InitializeModel", 1);
    }

    public void InitializeModel()
    {
        if (binding == null)
        {
            Debug.LogError("MNIST: assign the UdonclGenericOnnxMnistBinding instance.");
            return;
        }

        image = new float[28 * 28];
        logits = new float[10];
        for (int i = 0; i < image.Length; i++)
            image[i] = Digit[i] == '#' ? 1.0f : 0.0f;

        // Initialize creates the OpenCL context, the ONNX session and its
        // input and output buffers in the C Host, and uploads the weights.
        int status = binding.StartInitialize(this, "OnModelInitialized");
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess)
            Debug.LogError("MNIST: StartInitialize failed with status " + status);
    }

    public void OnModelInitialized()
    {
        int libraryStatus = 0;
        int status = binding.GetInitializeStatus();
        if (status == UdonclGenericOnnxMnistBinding.StatusSuccess)
            status = binding.TryGetInitializeResult(out libraryStatus);
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess || libraryStatus != 0)
        {
            Debug.LogError(
                "MNIST: Initialize failed with status "
                    + status
                    + " and ONNX status "
                    + libraryStatus
            );
            return;
        }

        initialized = true;
        Classify();
    }

    public override void Interact()
    {
        Classify();
    }

    public void Classify()
    {
        if (!initialized)
            return;

        int status = binding.TryWriteImage(image);
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess)
        {
            Debug.LogError("MNIST: writing the image failed with status " + status);
            return;
        }

        status = binding.StartRun(this, "OnRunComplete");
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess)
            Debug.LogError("MNIST: StartRun failed with status " + status);
    }

    public void OnRunComplete()
    {
        int libraryStatus = 0;
        int status = binding.GetRunStatus();
        if (status == UdonclGenericOnnxMnistBinding.StatusSuccess)
            status = binding.TryGetRunResult(out libraryStatus);
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess || libraryStatus != 0)
        {
            Debug.LogError(
                "MNIST: Run failed with status " + status + " and ONNX status " + libraryStatus
            );
            return;
        }

        status = binding.StartReadLogits(logits, this, "OnLogitsRead");
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess)
            Debug.LogError("MNIST: StartReadLogits failed with status " + status);
    }

    public void OnLogitsRead()
    {
        int status = binding.GetLastReadbackStatus();
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess)
        {
            Debug.LogError("MNIST: readback failed with status " + status);
            return;
        }

        int predicted = 0;
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[predicted])
                predicted = i;
        }
        Debug.Log("MNIST: predicted " + predicted + " (logit " + logits[predicted] + ")");
    }
}
```

#### 5. Expected result

After the model is initialized, the Console shows
`MNIST: predicted 7 (logit ...)` with the largest logit. Interact with the
object to classify the image again; edit `Digit` to draw another digit.

> **Screenshot (pending):** Unity Console in Play Mode showing "MNIST: predicted 7" — docs/images/getting-started/mnist-console-output.png

### Chat with TinyLlama (4096-token context)

This tutorial builds the 4096-token Chat Artifact of
[`examples/onnx-transformer-tiny`](examples/onnx-transformer-tiny/README.md)
(target `onnx-transformer-tiny-chat-4k`) from the real TinyLlama-1.1B-Chat
model and its tokenizer, and chats with it from U#. The C Host applies the
chat template, tokenizes your message, runs the model on the GPU one position
at a time, and decodes the reply, which U# streams while it is generated.
Build the compiler first as described in [Install and build](#install-and-build),
including `install_onnx_mlir.py`, and work through the
[MNIST tutorial](#compile-your-own-onnx-model-with-host-c-mnist) first: this
one uses the same ONNX workflow at a much larger scale.

The model is too large to ship with the Unity package, so there is no bundled
import; the **TinyLlama Chat** Package Sample needs the artifact you build
here. Keep these limits in mind:

- The model download is the pinned 4-bit quantized
  `onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX` `model_q4.onnx` (about
  910 MB). In the measurement recorded in
  [TINYLLAMA_AOT_CONVERSATION_PLAN.md](TINYLLAMA_AOT_CONVERSATION_PLAN.md#4096-token-artifacts),
  the 4096-token Conversation Artifact compiled to 37 kernels and 107 Device
  buffers totalling 358 MB of GPU memory.
- Per decode step, attention reads and writes all 4096 positions whatever the
  conversation length.
- TinyLlama-1.1B was trained on 2048 positions (`max_position_embeddings`).
  Longer conversations run, because its rotary embedding is computed from the
  position, but replies past position 2048 may degrade.

#### 1. The files you write

The example keeps the model-independent parts in C files of its own and lets
CMake derive the chat configuration:

**The token-array Conversation Host**
([`conversation_host.c`](examples/onnx-transformer-tiny/conversation_host.c),
with [`conversation_host.h`](examples/onnx-transformer-tiny/conversation_host.h)).
It owns the OpenCL context and queue, drives the model through the generated
ONNX binding of `<udoncl/onnx_generated_binding.h>` (`udon_onnx_binding_*`),
keeps the key/value cache on the GPU between positions, and selects each next
token with an argmax on the GPU (greedy decoding). Its exports are
`tinyllama_initialize`, `tinyllama_reset`, `tinyllama_set_prompt_tokens`,
`tinyllama_generate`, `tinyllama_generated_token_count`,
`tinyllama_read_generated_tokens`, and `tinyllama_dispose`:

```c
UDONCL_EXPORT udononnxStatus_t tinyllama_initialize(void) {
  /* ... clGetPlatformIDs, clGetDeviceIDs, clCreateContext,
     clCreateCommandQueue ... */
  status = udon_onnx_binding_create(context, queue);
  if (status == UDONONNX_STATUS_SUCCESS)
    status = udon_onnx_binding_enqueue_initialize();
  if (status == UDONONNX_STATUS_SUCCESS)
    status = udon_onnx_binding_bind_all();
  /* ... resolve the input_ids, attention_mask, position_ids and logits
     buffers, create the argmax buffers, clear the sequence state ... */
}

UDONCL_EXPORT udononnxStatus_t tinyllama_generate(uint32_t max_new_tokens) {
  /* ... check the state and the Context Capacity ... */
  /* Run the prompt, then decode one token per model run until EOS or
     max_new_tokens. */
  for (index = 0; index < prompt_tokens; index += rows) {
    /* ... advance the sequence state past the previous run ... */
    status = enqueue_chunk(prompt + index, rows, base + index);
  }
  for (output_index = 0; output_index < max_new_tokens; ++output_index) {
    status = decode_token(base + prompt_tokens + output_index,
                          output_index + 1u == max_new_tokens, &token);
    generated[output_index] = (int32_t)token;
    generated_count = output_index + 1u;
    if (token == TINYLLAMA_EOS_TOKEN)
      break;
  }
  /* ... */
}
```

**The 4096-token Host file**
([`conversation_host_4k.c`](examples/onnx-transformer-tiny/conversation_host_4k.c))
pins the Context Capacity and includes the shared Host, so the graph, the Host
and the text layer cannot drift apart:

```c
#ifndef TINYLLAMA_CONTEXT_CAPACITY
#define TINYLLAMA_CONTEXT_CAPACITY 4096u
#endif
#if TINYLLAMA_CONTEXT_CAPACITY != 4096u
#error "conversation_host_4k.c is the Host of a 4096-token Context Capacity"
#endif

#include "conversation_host.c"
```

**The text layer**
([`conversation_text.c`](examples/onnx-transformer-tiny/conversation_text.c)).
It composes the Conversation Host with UdonTokenizers and the tokenizer tables
compiled at build time (the chat template is compiled too, not interpreted).
A message is tokenized in bounded slices so that a long message never has to
fit in one Udon event, and decoded text is returned as UTF-16 code units, one
per `int`:

```c
/* Starts a conversation; tinyllama_begin_chat_turn appends the next user
   message after a completed reply. */
UDONCL_EXPORT udononnxStatus_t tinyllama_begin_chat_prompt(void);

/* Does at most max_work units of prompt work (about one UTF-16 code unit
   each); the prompt is committed by the call that leaves nothing pending. */
UDONCL_EXPORT udononnxStatus_t tinyllama_continue_chat_prompt(
    udon_string_t system_text, udon_string_t user_text, uint32_t max_work);

/* 1 while a begun chat prompt or turn still has work pending, else 0. */
UDONCL_EXPORT uint32_t tinyllama_chat_prompt_pending(void);

/* The number of UTF-16 code units of the reply that are final so far, valid
   while the generation is still running. */
UDONCL_EXPORT uint32_t tinyllama_streamed_text_length(void);

/* Copies the final code units [offset, offset + count) of the reply to the
   start of destination. */
UDONCL_EXPORT udononnxStatus_t tinyllama_read_streamed_text(
    udon_int32_array_t destination, uint32_t offset, uint32_t count);
```

`udon_string_t` parameters become U# `string` parameters, and
`udon_int32_array_t` becomes `int[]`.

**The CMake project**
([`CMakeLists.txt`](examples/onnx-transformer-tiny/CMakeLists.txt)). Its
helper functions build the 512-token and 4096-token variants from the same
files. For `onnx-transformer-tiny-chat-4k` they amount to:

```cmake
# The model, specialized to one position per run on a past of 4095 positions.
udoncl_add_onnx_host_library(transformer_conversation_model_host_4k
  MODEL "${UDONCL_TINYLLAMA_ONNX_MODEL}"
  DIMENSION_BOUNDS
    batch_size=1
    sequence_length=1
    past_sequence_length=4095
  INPUT_SHAPES
    attention_mask=1,4096)

# The Conversation Host and the text layer, compiled with the same bounds.
udoncl_add_host_library(transformer_conversation_user_host_4k
  conversation_host_4k.c)
target_compile_definitions(transformer_conversation_user_host_4k INTERFACE
  TINYLLAMA_CONTEXT_CAPACITY=4096u
  TINYLLAMA_GENERATED_CAPACITY=32u
  TINYLLAMA_SEQUENCE_ROWS=1u)
udontokenizers_add_huggingface_model(transformer_chat_tokenizer_4k
  TOKENIZER_DIRECTORY "${UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY}"
  SYMBOL tinyllama_tokenizer
  SOURCES conversation_text.c)

udoncl_add_unity_library(onnx-transformer-tiny-chat-4k
  CONFIG "${chat_unity_config}"
  OMIT_INTERMEDIATE_IR
  HOST_LIBRARIES
    transformer_conversation_model_host_4k
    transformer_conversation_user_host_4k
    transformer_chat_tokenizer_4k
    UdonONNXRuntime::UdonONNXRuntime
    UdonTokenizers::UdonTokenizers
  LIBRARIES
    UdonONNXOperators::UdonONNXOperators
    UdonATen::UdonATen
    UdonBLAS::UdonBLAS)
```

The real CMake file also sets the include directories and the same
definitions on the tokenizer target. These cache variables control the build:

- `UDONCL_TINYLLAMA_ONNX_MODEL`: the downloaded ONNX model. Without it the
  project builds only the small synthetic transformer.
- `UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY`: the downloaded tokenizer files. The
  Chat Artifacts need it; without it only the token-array Conversation
  Artifacts are configured.
- `UDONCL_TINYLLAMA_CONTEXT_4K` (default `ON`): also configure the 4096-token
  `-4k` targets next to the 512-token ones
  (`UDONCL_TINYLLAMA_CONTEXT_CAPACITY`, default 512). Both variants have the
  same program, binding and U# API, so a project imports one of the two.
- `UDONCL_TINYLLAMA_MAX_NEW_TOKENS` (default 32): the most tokens one
  generation may request.

**The public API description.** CMake writes the chat config,
`unity-library-chat.json` in the build directory, from the example's
[`unity-library.json`](examples/onnx-transformer-tiny/unity-library.json)
with `string(JSON ...)`. Its keys are:

- `schema_version`: `2`.
- `program`: `tinyllama_real_chat`; Unity imports the artifact under
  `Assets/udoncl-generated/tinyllama_real_chat`.
- `binding`: the class `UdonclTinyLlamaChatBinding` in
  `Lemolatoon.Udoncl.Generated`.
- `device_texel_format`: `"rgba32_float"`, which stores four words per
  RGBA32 float texel in the real model's Device buffers and weights (cache
  variable `UDONCL_TINYLLAMA_DEVICE_TEXEL_FORMAT`).
- `generate_onnx_public_buffers`: `false`. No tensor is public; text goes in
  as U# strings and comes out through `int[]` arrays of the exports.
- `export_overrides`: the U# names of the 21 Conversation and text-layer
  exports, for example
  `{"symbol": "tinyllama_continue_chat_prompt", "public_name": "ContinueChatPrompt", "parameters": ["systemText", "userText", "maxWork"]}`.
  `StreamedTextLength` and `ReadStreamedText` also set
  `"allowed_while_busy": true`, so U# can poll them while `Generate` runs.

The base config's `buffers` array lists the public buffers of the synthetic
fixture; CMake drops it for the real model, whose buffers the compiler
derives.

#### 2. Build

From the repository root, fetch and verify the pinned model and tokenizer.
Each script downloads into `.deps/models` once (set `UDONCL_MODEL_CACHE` to
use another directory) and prints the cached path, so the output can be
captured directly. Then configure the example with both paths and build the
4096-token Chat Artifact:

```powershell
cmake --preset udoncl
python scripts/build.py
$model = python scripts/fetch_tinyllama_onnx.py
$tokenizer = python scripts/fetch_tinyllama_tokenizer.py
cmake -S examples/onnx-transformer-tiny -B build/tinyllama-cmake -G Ninja `
  -DCMAKE_PREFIX_PATH="$PWD" `
  -DUDONCL_TINYLLAMA_ONNX_MODEL="$model" `
  -DUDONCL_TINYLLAMA_TOKENIZER_DIRECTORY="$tokenizer"
cmake --build build/tinyllama-cmake --target onnx-transformer-tiny-chat-4k
```

Configuring compiles the real model once for every configured variant, so it
takes much longer than the MNIST project. The importable artifact is written
to:

```text
build/tinyllama-cmake/udoncl/onnx-transformer-tiny-chat-4k/artifact
```

#### 3. Import into Unity

1. In a Unity 2022.3 VRChat Worlds project with the
   **UdonCL Runtime and ONNX Importer** package, choose
   **Tools → udoncl → Import Artifacts...** and select the
   `build/tinyllama-cmake/udoncl/onnx-transformer-tiny-chat-4k/artifact`
   directory.

   > **Screenshot (pending):** Tools → udoncl → Import Artifacts… with the onnx-transformer-tiny-chat-4k artifact folder selected — docs/images/getting-started/tinyllama-import-artifacts.png

2. In the **udoncl Library Import** dialog, choose **Add after Compile**.
   After the generated U# wrapper compiles, Unity creates
   `Assets/udoncl-generated/tinyllama_real_chat/Prefabs/UdonclTinyLlamaChatBinding.prefab`
   and places an instance in the active scene.

   > **Screenshot (pending):** Project window showing Assets/udoncl-generated/tinyllama_real_chat with its Bindings and Prefabs folders — docs/images/getting-started/tinyllama-generated-assets.png

3. Choose **Create → U# Script**, name it `TinyLlamaChat`, and replace its
   contents with the script in the next step, or import the same script as
   the **TinyLlama Chat** sample from Package Manager (**UdonCL Runtime and
   ONNX Importer** → **Samples** → **Import**).
4. Create a GameObject (for example a Cube, whose Collider lets you interact
   with it), add the **Tiny Llama Chat** component, and drag the
   `UdonclTinyLlamaChatBinding` scene instance into its **Binding** field.
   Edit **System Text** and **User Text** as you like. **Max New Tokens** must
   not exceed `UDONCL_TINYLLAMA_MAX_NEW_TOKENS` (32 by default).

   > **Screenshot (pending):** Inspector of the TinyLlamaChat GameObject with Binding, System Text, User Text, Max New Tokens and Prompt Work Per Frame — docs/images/getting-started/tinyllama-inspector.png

5. Enter Play Mode and watch the Console.

#### 4. Your UdonSharpBehaviour

The generated binding follows the same rules as before: an export that
launches GPU work becomes `Start<Name>(..., receiver, completionEvent)` with
`Get<Name>Status()` and `TryGet<Name>Result(out ...)`, and any other export
becomes `Try<Name>(..., out result)`. `TinyLlamaChat` uses:

- `StartInitialize` (spans many frames while the weights are uploaded) and
  `StartGenerate(int maxNewTokens, ...)`.
- `TryBeginChatPrompt` and `TryContinueChatPrompt(systemText, userText,
  maxWork, out status)` for the first message, `TryBeginChatTurn` and
  `TryContinueChatTurn(userText, maxWork, out status)` for every later one,
  and `TryChatPromptPending(out pending)`: the message is encoded with a
  bounded amount of work per frame.
- `TryStreamedTextLength(out length)` and
  `TryReadStreamedText(int[] destination, int offset, int count, out status)`:
  the final part of the reply, read every frame while `Generate` runs.

The exports that return a `udononnxStatus_t` report `0` on success; the
sample checks it next to the binding status.

`TinyLlamaChat.cs` (identical to the **TinyLlama Chat** Package Sample):

```csharp
using Lemolatoon.Udoncl.Generated;
using UdonSharp;
using UnityEngine;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class TinyLlamaChat : UdonSharpBehaviour
{
    // The status the C Host returns from every export (udononnxStatus_t).
    private const int OnnxSuccess = 0;

    [SerializeField]
    private UdonclTinyLlamaChatBinding binding;

    [SerializeField]
    private string systemText = "You are a friendly assistant in a VRChat world.";

    [SerializeField]
    private string userText = "What is your name?";

    // At most the generated-token capacity the artifact was built with
    // (UDONCL_TINYLLAMA_MAX_NEW_TOKENS, 32 by default).
    [SerializeField]
    private int maxNewTokens = 32;

    // Tokenizer work per frame while a message is encoded, about one UTF-16
    // code unit each; a long message is spread over several frames.
    [SerializeField]
    private int promptWorkPerFrame = 32;

    private bool initialized;
    private bool busy;
    private bool conversationStarted;
    private bool encodingTurn;
    private bool generating;
    private int[] textChunk;
    private int replyLength;
    private string reply;

    private void Start()
    {
        // Initialize one frame after start-up so every behaviour is ready.
        SendCustomEventDelayedFrames("InitializeChat", 1);
    }

    public void InitializeChat()
    {
        if (binding == null)
        {
            Debug.LogError("TinyLlama: assign the UdonclTinyLlamaChatBinding instance.");
            return;
        }

        textChunk = new int[256];
        busy = true;
        // Initialize creates the OpenCL context and the ONNX session in the
        // C Host and uploads the model weights; it spans many frames.
        int status = binding.StartInitialize(this, "OnChatInitialized");
        if (status != UdonclTinyLlamaChatBinding.StatusSuccess)
            Fail("StartInitialize", status, OnnxSuccess);
    }

    public void OnChatInitialized()
    {
        int onnxStatus = OnnxSuccess;
        int status = binding.GetInitializeStatus();
        if (status == UdonclTinyLlamaChatBinding.StatusSuccess)
            status = binding.TryGetInitializeResult(out onnxStatus);
        if (!Succeeded("Initialize", status, onnxStatus))
            return;

        initialized = true;
        busy = false;
        Send();
    }

    public override void Interact()
    {
        Send();
    }

    // Sends userText: the first message starts the conversation with the
    // system text, every later one is appended to it as the next turn.
    public void Send()
    {
        if (!initialized || busy)
            return;

        busy = true;
        encodingTurn = conversationStarted;
        int onnxStatus;
        int status;
        if (encodingTurn)
            status = binding.TryBeginChatTurn(out onnxStatus);
        else
            status = binding.TryBeginChatPrompt(out onnxStatus);
        if (!Succeeded("BeginChatPrompt", status, onnxStatus))
            return;
        Debug.Log("TinyLlama: you: " + userText);
        EncodeMessage();
    }

    // The C Host applies the chat template and tokenizes the message, a
    // bounded amount of work per frame, until nothing is pending.
    public void EncodeMessage()
    {
        int onnxStatus;
        int status;
        if (encodingTurn)
            status = binding.TryContinueChatTurn(userText, promptWorkPerFrame, out onnxStatus);
        else
            status = binding.TryContinueChatPrompt(
                systemText,
                userText,
                promptWorkPerFrame,
                out onnxStatus
            );
        if (!Succeeded("ContinueChatPrompt", status, onnxStatus))
            return;

        int pending;
        status = binding.TryChatPromptPending(out pending);
        if (!Succeeded("ChatPromptPending", status, OnnxSuccess))
            return;
        if (pending != 0)
        {
            SendCustomEventDelayedFrames("EncodeMessage", 1);
            return;
        }

        conversationStarted = true;
        reply = "";
        replyLength = 0;
        status = binding.StartGenerate(maxNewTokens, this, "OnReplyGenerated");
        if (!Succeeded("StartGenerate", status, OnnxSuccess))
            return;
        generating = true;
        SendCustomEventDelayedFrames("StreamReply", 1);
    }

    // While the reply is generated, read the part of it that is final.
    public void StreamReply()
    {
        if (!generating)
            return;
        ReadReplyText();
        SendCustomEventDelayedFrames("StreamReply", 1);
    }

    public void OnReplyGenerated()
    {
        generating = false;
        int onnxStatus = OnnxSuccess;
        int status = binding.GetGenerateStatus();
        if (status == UdonclTinyLlamaChatBinding.StatusSuccess)
            status = binding.TryGetGenerateResult(out onnxStatus);
        if (!Succeeded("Generate", status, onnxStatus))
            return;

        // After the generation completed the streamed text is the whole reply.
        ReadReplyText();
        Debug.Log("TinyLlama: assistant: " + reply);
        busy = false;
    }

    // Appends the reply text that became final since the last call. The
    // binding returns UTF-16 code units, one per int.
    private void ReadReplyText()
    {
        int length;
        // StatusBusy while the model runs a step: try again next frame.
        if (binding.TryStreamedTextLength(out length) != UdonclTinyLlamaChatBinding.StatusSuccess)
            return;
        while (replyLength < length)
        {
            int count = Mathf.Min(length - replyLength, textChunk.Length);
            int onnxStatus;
            int status = binding.TryReadStreamedText(textChunk, replyLength, count, out onnxStatus);
            if (status != UdonclTinyLlamaChatBinding.StatusSuccess || onnxStatus != OnnxSuccess)
                return;
            for (int i = 0; i < count; i++)
                reply += (char)textChunk[i];
            replyLength += count;
        }
    }

    private bool Succeeded(string step, int status, int onnxStatus)
    {
        if (status == UdonclTinyLlamaChatBinding.StatusSuccess && onnxStatus == OnnxSuccess)
            return true;
        Fail(step, status, onnxStatus);
        return false;
    }

    private void Fail(string step, int status, int onnxStatus)
    {
        generating = false;
        busy = false;
        Debug.LogError(
            "TinyLlama: "
                + step
                + " failed with status "
                + status
                + " and ONNX status "
                + onnxStatus
        );
    }
}
```

#### 5. Expected result

After the model is initialized, the sample sends **User Text** and logs
`TinyLlama: you: What is your name?`, then streams the reply and logs it as
`TinyLlama: assistant: ...` once the generation completes. Decoding is greedy,
so the same conversation gives the same reply. Interact with the object to
send **User Text** again as the next turn of the same conversation.

> **Screenshot (pending):** Unity Console in Play Mode showing the "TinyLlama: you:" and "TinyLlama: assistant:" lines — docs/images/getting-started/tinyllama-console-output.png

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
- [AGENTS.md](AGENTS.md): repository rules for contributors and agents.
- Implementation plans:
  [TINYLLAMA_AOT_CONVERSATION_PLAN.md](TINYLLAMA_AOT_CONVERSATION_PLAN.md)
  (TinyLlama conversation and chat),
  [TINYLLAMA_DEVICE_PREFILL_PLAN.md](TINYLLAMA_DEVICE_PREFILL_PLAN.md)
  (TinyLlama Device prefill),
  [ONNX_USHARP_BINDING_PLAN.md](ONNX_USHARP_BINDING_PLAN.md) (suspendable C
  Host exports and U# bindings),
  [STATIC_OPENCL_RESOURCE_PLAN.md](STATIC_OPENCL_RESOURCE_PLAN.md) (static
  OpenCL resource specialization),
  [DEVICE_RESOURCE_ALIAS_PLAN.md](DEVICE_RESOURCE_ALIAS_PLAN.md) (Device
  resource aliases), and
  [HOST_STORAGE_ALLOCATOR_PLAN.md](HOST_STORAGE_ALLOCATOR_PLAN.md) (Host
  storage allocation).
- [libraries/README.md](libraries/README.md): the Udon compute libraries and
  their layering.
- [libraries/udonblas/README.md](libraries/udonblas/README.md): the UdonBLAS
  API and its profile limits.
- [libraries/udondnn/README.md](libraries/udondnn/README.md) and
  [libraries/udonaten/README.md](libraries/udonaten/README.md): the UdonDNN and
  UdonATen APIs.
- [runtime/onnx/README.md](runtime/onnx/README.md): the ONNX model builders
  used by the example model generators.
- [unity/Packages/com.lemolatoon.udoncl/README.md](unity/Packages/com.lemolatoon.udoncl/README.md):
  the Unity package, artifact importer, bundled examples, and Package
  Samples.
- [examples/README.md](examples/README.md): example catalogue and walkthroughs.

## License

See [LICENSE](LICENSE).
