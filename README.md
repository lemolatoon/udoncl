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
   - **TinyLlama Chat** (`TinyLlamaChat`) chats with TinyLlama-1.1B, one
     conversation shared by the players of an instance. The model is too
     large to bundle, so this sample needs the artifact you build in the
     [TinyLlama tutorial](#chat-with-tinyllama).

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
  library that your own Host C drives through the `UdonONNXRuntime` session
  API (`udononnxCreateSession`, `udononnxBindInput`/`udononnxBindOutput`,
  `udononnxRun`, `udononnxReleaseSession`, after ONNX Runtime's C API), from
  the MNIST CNN up to the real TinyLlama-1.1B model: a 2048-token chat in one
  Host C file, and 512- and 4096-token advanced chat artifacts. The model
  runs as compiled Udon Assembly and fragment shaders, not through an
  execution-plan interpreter.
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
- The Host C source describes the library with the macros of
  `<udoncl/export.h>`: `UDONCL_PROGRAM("name")` names the program,
  `UDONCL_EXPORT` (or `UDONCL_EXPORT_WHILE_BUSY`) marks each public U# method,
  and `UDONCL_PUBLIC(float|int|long) cl_mem name;` declares each public
  Device buffer; the tutorials below explain each one. There is no config
  file. `BINDING_NAMESPACE`, `BINDING_CLASS`, `DEVICE_TEXEL_FORMAT` and
  `DEVICE_WORK_BUDGET` arguments set the rest. Every Device buffer must
  resolve at compile time to one Unity Texture; `udonclc` reports what
  prevents it otherwise. The kernels the Host launches are derived from its
  dispatch sites, both for UdonCL library kernels and for your own OpenCL C
  kernels launched with `clCreateKernel`, `clSetKernelArg`, and a
  constant-size `clEnqueueNDRangeKernel` (see "Library kernel entries" and
  "User OpenCL C kernels" in
  [docs/architecture/refactor-baseline-and-mlir-pipeline.md](docs/architecture/refactor-baseline-and-mlir-pipeline.md)).
- An ONNX model becomes a Host library with
  `udoncl_add_onnx_host_library(<target> MODEL <model.onnx> DIMENSIONS
  <name>=<value>... DIMENSION_BOUNDS <name>=<max>...)`, which your own Host
  C drives through `UdonONNXRuntime`. `DIMENSIONS` fixes a symbolic
  dimension of the model. `DIMENSION_BOUNDS` keeps it dynamic with extents
  from 0 to `max`, and the Host gives the extent of each run when it binds
  the tensors. Every symbolic dimension needs one of the two; see the
  [MNIST tutorial](#compile-your-own-onnx-model-with-host-c-mnist) for a
  fixed dimension and the
  [TinyLlama tutorial](#chat-with-tinyllama) for a
  bounded one.

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
3. [TinyLlama](#chat-with-tinyllama): a 1.1B-parameter
   chat model in one Host C file and one OpenCL C kernel.

### Write your own Host C and OpenCL C kernel (SAXPY)

This tutorial builds the [`examples/saxpy`](examples/saxpy/README.md) project
from source, imports it into Unity, and calls it from your own
UdonSharpBehaviour. The kernel computes `result[i] = alpha * x[i] + y[i]` on
the GPU, first for 16 elements and then, with new inputs, for all 32. Build the compiler first as described in
[Install and build](#install-and-build).

#### 1. The files you write

A UdonCL project has three inputs: an OpenCL C kernel, Host C that launches
it and describes the public U# API, and a `CMakeLists.txt`.

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
ordinary Khronos OpenCL 1.2 API and follows the UdonCL lifecycle Initialize →
Write → Run → Read → Dispose: `initialize` creates the buffers and the kernel
once, `run` sets the kernel arguments, launches the kernel and waits for it,
and `dispose` releases everything. Each returns the OpenCL status. Three
macros of `<udoncl/export.h>` describe the public U# API. The essential part
is:

```c
UDONCL_PROGRAM("saxpy_demo");

UDONCL_PUBLIC(float) cl_mem x, y, result;

static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel;

UDONCL_EXPORT int initialize(void) {
  /* ... clGetPlatformIDs, clGetDeviceIDs, clCreateContext,
     clCreateCommandQueue, each followed by a status check ... */
  x = clCreateBuffer(context, CL_MEM_READ_ONLY, SAXPY_WORDS * sizeof(float), 0,
                     &status);
  /* ... y (CL_MEM_READ_ONLY) and result (CL_MEM_WRITE_ONLY) likewise ... */
  program = clCreateProgramWithSource(context, 1, &source, 0, &status);
  status = clBuildProgram(program, 0, 0, 0, 0, 0);
  kernel = clCreateKernel(program, "saxpy", &status);
  return status;
}

UDONCL_EXPORT int run(int n, float alpha) {
  size_t global = SAXPY_WORDS;
  size_t local = 1;
  cl_int status;
  if ((status = clSetKernelArg(kernel, 0, sizeof(x), &x)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 1, sizeof(y), &y)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 2, sizeof(result), &result)) !=
          CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 3, sizeof(n), &n)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 4, sizeof(alpha), &alpha)) != CL_SUCCESS)
    return status;
  status =
      clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global, &local, 0, 0, 0);
  if (status != CL_SUCCESS)
    return status;
  return clFinish(queue);
}

UDONCL_EXPORT int dispose(void) {
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  /* ... release the buffers, the queue and the context ... */
  return CL_SUCCESS;
}
```

Kernels are not compiled at run time: the kernel body comes from the Device
library in `CMakeLists.txt`, and the build derives the launch from the Host
C. That works within limits: every buffer argument precedes the
`int`/`uint`/`float` scalars, each argument has a dominating
`clSetKernelArg`, the kernel object comes from one `clCreateKernel` of a
constant name, and the global work size is one constant. A buffer whose
address is passed to the runtime lives in a Host global. The kernel object may
live in a global created by one export and launched by another, as here; do
not store `0` into it in `dispose`, since a launch must name one kernel.
Unsupported patterns are compile errors rather than approximations.

The three macros replace a separate API description:

- `UDONCL_PROGRAM("saxpy_demo")` names the program. Unity imports the
  artifact under `Assets/udoncl-generated/saxpy_demo` and names the generated
  U# class `UdonclSaxpyDemoBinding` (`Udoncl` + the PascalCase program name +
  `Binding`) in the namespace `Lemolatoon.Udoncl.Generated`. The optional
  `BINDING_CLASS` and `BINDING_NAMESPACE` arguments of
  `udoncl_add_unity_library` override both.
- `UDONCL_EXPORT` makes a function a U# method named after the PascalCase of
  its C name (`initialize` → `Initialize`, `run` → `Run`, `dispose` →
  `Dispose`), with the C parameter names (`n`, `alpha`). The parameter and
  result types come from the C signature. An export that launches no Device
  work (`initialize`, `dispose`) is a synchronous `Try*` method; one that does
  (`run`) is an asynchronous `Start*` method.
- `UDONCL_PUBLIC(float)` makes a file-scope `cl_mem` variable a public Device
  buffer of `float` elements, named after the variable: `x` → `TryWriteX`,
  `result` → `StartReadResult`. Its `clCreateBuffer` flags give its
  direction (`CL_MEM_READ_ONLY` for a buffer U# writes, `CL_MEM_WRITE_ONLY`
  for one it reads, `CL_MEM_READ_WRITE` for one it does both with, which
  gets both methods) and size (128 bytes). The variable must hold the buffer
  of exactly one `clCreateBuffer`; the build resolves it to its Unity
  texture. Each buffer method carries an XML doc comment that names the C
  variable, the element type and count, and the direction, so the IDE shows
  where a method comes from.

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
  OMIT_INTERMEDIATE_IR
  HOST_LIBRARIES saxpy_host UdonCL::OpenCLRuntime
  DEVICE_LIBRARIES saxpy_device)
```

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
5. Create a GameObject with a collider (for example **3D Object → Cube**), so
   that it can be interacted with, add the **Saxpy Caller** component, and
   drag the `UdonclSaxpyDemoBinding` scene instance into its **Binding**
   field.

   > **Screenshot (pending):** Inspector of the SaxpyCaller GameObject with Binding set to the UdonclSaxpyDemoBinding scene instance — docs/images/getting-started/saxpy-inspector-binding.png

6. Enter Play Mode, interact with the cube, and watch the Console.

#### 4. Your UdonSharpBehaviour

The generated binding exposes:

- `int TryWriteX(float[] source)` and `int TryWriteY(float[] source)`: upload
  the whole `x` or `y` buffer. The array length must equal the buffer size in
  words (32).
- `int TryInitialize(out int result)` and `int TryDispose(out int result)`:
  run `initialize` and `dispose` at once and return the value they returned
  in `result`, here the OpenCL status (`0` is `CL_SUCCESS`).
- `int StartRun(int n, float alpha, UdonSharpBehaviour receiver, string completionEvent)`:
  starts the `Run` export. An export that launches Device work runs
  asynchronously: when it finishes, the binding calls
  `receiver.SendCustomEvent(completionEvent)`, so the event method must be
  `public`.
- `int GetRunStatus()`: the status of the finished export, and
  `int TryGetRunResult(out int result)`: the value `run` returned, the
  OpenCL status.
- `int StartReadResult(float[] destination, UdonSharpBehaviour receiver, string completionEvent)`:
  starts an asynchronous GPU readback of the whole `result` buffer into
  `destination` (again exactly 32 words) and calls the completion event like
  `StartRun`.
- `int GetLastReadbackStatus()`: the status of the completed readback.
- Status constants `StatusSuccess` (0), `StatusNotInitialized` (-1),
  `StatusBusy` (-2), `StatusInvalidArgument` (-3, for example a wrong array
  length), `StatusAbiMismatch` (-4), `StatusExecutionFailure` (-5), and
  `StatusDisposed` (-6).
- `int TryBindProgram(UdonBehaviour program)`, `bool IsInitialized()` and
  `int TryUnbindProgram()`: attach the binding to its raw Udon program (the
  binding prefab does this for you), ask whether it is attached, and detach
  it. `TryUnbindProgram` runs no Host code; the Host `dispose` export is its
  own method, `TryDispose`.

`SaxpyCaller` follows the pattern of every UdonCL example: `OnEnable`
initializes the library one frame later, once the binding is ready; `Interact`
writes the inputs, runs the kernel and reads the result; `OnDisable` disposes
the library. A second run reuses the same kernel with new inputs.

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

    private bool initialized;
    private bool secondRun;
    private float[] x;
    private float[] y;
    private float[] result;

    private void OnEnable()
    {
        // One frame later every behaviour, the binding included, is ready.
        SendCustomEventDelayedFrames(nameof(InitializeSaxpy), 1);
    }

    private void OnDisable()
    {
        if (!initialized)
            return;
        initialized = false;
        int openclStatus;
        int status = binding.TryDispose(out openclStatus);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
            Debug.LogError("SAXPY: Dispose failed with status " + status);
    }

    public void InitializeSaxpy()
    {
        if (initialized)
            return;
        if (binding == null)
        {
            Debug.LogError("SAXPY: assign the UdonclSaxpyDemoBinding instance.");
            return;
        }
        // Initialize creates the buffers and the kernel and enqueues no Device
        // work, so it completes at once.
        int openclStatus;
        int status = binding.TryInitialize(out openclStatus);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess || openclStatus != 0)
        {
            Debug.LogError(
                "SAXPY: Initialize failed with status "
                    + status
                    + " and OpenCL status "
                    + openclStatus
            );
            return;
        }
        initialized = true;
    }

    public override void Interact()
    {
        if (!initialized)
            return;
        x = new float[32];
        y = new float[32];
        result = new float[32];
        for (int i = 0; i < 32; i++)
        {
            x[i] = i + 1;
            y[i] = 100 + i;
        }
        secondRun = false;
        StartSaxpy(16, 2.0f);
    }

    private void StartSaxpy(int n, float alpha)
    {
        int status = binding.TryWriteX(x);
        if (status == UdonclSaxpyDemoBinding.StatusSuccess)
            status = binding.TryWriteY(y);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
        {
            Debug.LogError("SAXPY: writing the inputs failed with status " + status);
            return;
        }

        status = binding.StartRun(n, alpha, this, nameof(OnSaxpyComplete));
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
            Debug.LogError("SAXPY: StartRun failed with status " + status);
    }

    public void OnSaxpyComplete()
    {
        int openclStatus = 0;
        int status = binding.GetRunStatus();
        if (status == UdonclSaxpyDemoBinding.StatusSuccess)
            status = binding.TryGetRunResult(out openclStatus);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess || openclStatus != 0)
        {
            Debug.LogError(
                "SAXPY: Run failed with status " + status + " and OpenCL status " + openclStatus
            );
            return;
        }

        status = binding.StartReadResult(result, this, nameof(OnSaxpyReadback));
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

        if (!secondRun)
        {
            // The first run: alpha 2 over the first 16 words.
            for (int i = 0; i < 16; i++)
            {
                float expected = 2.0f * x[i] + y[i];
                Debug.Log(
                    "SAXPY: result[" + i + "] = " + result[i] + " (expected " + expected + ")"
                );
            }

            // The second run reuses the kernel with new inputs: alpha 0.5 over
            // all 32 words.
            for (int i = 0; i < 32; i++)
            {
                x[i] = 10 * (i + 1);
                y[i] = -i;
            }
            secondRun = true;
            StartSaxpy(32, 0.5f);
            return;
        }

        for (int i = 0; i < 32; i++)
        {
            float expected = 0.5f * x[i] + y[i];
            Debug.Log(
                "SAXPY: second result[" + i + "] = " + result[i] + " (expected " + expected + ")"
            );
        }
    }
}
```

#### 5. Expected result

Enter Play Mode and interact with the GameObject (or call `Interact` from
another behaviour). The first sixteen Console lines show
`result[i] = 2 * x[i] + y[i]`. With the inputs above, `x[i] = i + 1` and
`y[i] = 100 + i`, so `result[i] = 3 * i + 102`: `result[0] = 102`,
`result[1] = 105`, ..., `result[15] = 147`, each equal to its `expected`
value. Result words 16–31 (not logged) hold a copy of `y`, as do the words at
or beyond a smaller `n`.

The second run writes `x[i] = 10 * (i + 1)` and `y[i] = -i` and runs with
`n = 32` and `alpha = 0.5`, so the next 32 lines show
`second result[i] = 4 * i + 5`, from `5` to `129`. Leaving Play Mode disables
the behaviour, which disposes the library.

> **Screenshot (pending):** Unity Console in Play Mode showing the sixteen SAXPY result lines — docs/images/getting-started/saxpy-console-output.png

The named transfers are the only way U# moves a public buffer: the binding has
no buffer-ID methods, and each named method addresses its buffer directly.

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

An ONNX project has three inputs: the `.onnx` model, Host C that drives it
and describes the U# API, and a `CMakeLists.txt`.

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
UDONCL_PROGRAM("generic_onnx_mnist");

UDONCL_PUBLIC(float) cl_mem image, logits;

static const int64_t image_dims[4] = {1, 1, 28, 28};
static const int64_t logits_dims[2] = {1, 10};

static cl_context context;
static cl_command_queue queue;
static udononnxSession_t *session;

UDONCL_EXPORT udononnxStatus_t initialize(void) {
  /* ... clGetPlatformIDs, clGetDeviceIDs, clCreateContext,
     clCreateCommandQueue ... */
  image = clCreateBuffer(context, CL_MEM_READ_ONLY, 28 * 28 * sizeof(float), 0,
                         &error);
  logits =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, 10 * sizeof(float), 0, &error);
  status = udononnxCreateSession(context, queue, &session);
  status = udononnxBindInput(session, "image", image, 4, image_dims, 0);
  status = udononnxBindOutput(session, "logits", logits, 2, logits_dims, 0);
  /* ... each step checks its status and releases everything on failure ... */
}

UDONCL_EXPORT udononnxStatus_t run(void) { return udononnxRun(session); }

UDONCL_EXPORT udononnxStatus_t dispose(void) {
  /* ... udononnxReleaseSession, clReleaseMemObject, clReleaseCommandQueue,
     clReleaseContext ... */
}
```

`run` launches GPU work, so it becomes an asynchronous U# method;
`initialize` only creates the buffers and the session (the weights are
compiled into the artifact), so it is synchronous, as is `dispose`, which
releases them; `initialize` may run again after it. `udononnxRun` returns once
the graph is enqueued on the session's in-order queue. U# writes the image
into the buffer bound as the model input and reads the buffer bound as the
model output; the Host C never copies tensor data itself.

- `UDONCL_PROGRAM("generic_onnx_mnist")` names the program: Unity imports the
  artifact under `Assets/udoncl-generated/generic_onnx_mnist` and names the
  binding `UdonclGenericOnnxMnistBinding`.
- `UDONCL_PUBLIC(float) cl_mem image, logits;` makes the two buffers public,
  named after the variables: `image` (784 floats, `CL_MEM_READ_ONLY`,
  `TryWriteImage`) and `logits` (10 floats, `CL_MEM_WRITE_ONLY`,
  `StartReadLogits`). `udononnxBindInput` and `udononnxBindOutput` bind
  each to the model tensor of that name; the build resolves each variable
  and each bind to the one buffer it holds, and rejects a tensor left
  unbound.
- The three `UDONCL_EXPORT` functions are U# methods named after the
  PascalCase of their C names: `initialize` → `Initialize`, `run` → `Run`,
  and `dispose` → `Dispose`, none with parameters.
- The other Device buffers of the model belong to the session and stay
  private; the compiler derives them from the model.

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
  DIMENSIONS batch=1)
target_include_directories(mnist_host INTERFACE
  "${UdonBLAS_INCLUDE_DIRS}"
  "${UdonDNN_INCLUDE_DIRS}"
  "${UdonATen_INCLUDE_DIRS}")
udoncl_add_host_library(mnist_user_host host.c)
target_include_directories(mnist_user_host INTERFACE
  "${UdonONNXRuntime_INCLUDE_DIRS}")
udoncl_add_unity_library(onnx-mnist
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
  into the Host library `mnist_host`. `DIMENSIONS batch=1` fixes the
  model's symbolic `batch` dimension; every symbolic dimension needs a
  build-time value or, with `DIMENSION_BOUNDS`, a build-time bound (see the
  [TinyLlama tutorial](#chat-with-tinyllama)).
- `mnist_user_host` is your Host C. `UdonONNXRuntime::UdonONNXRuntime` is the
  session API it calls, and `UdonDNN::OpenCLRuntime` is the linked OpenCL
  runtime with the UdonDNN kernels enabled.
- `UdonATen::Device` and `UdonBLAS::Device` provide the OpenCL C kernels the
  model's operators launch. You do not list the kernels: the build derives
  them from the Host.

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

5. Enter Play Mode, interact with the cube, and watch the Console.

#### 4. Your UdonSharpBehaviour

The generated binding follows the same rules as in the SAXPY tutorial:

- `int TryInitialize(out int result)`: `initialize`, which launches no GPU
  work and therefore runs synchronously; `result` is its `udononnxStatus_t`.
- `int StartRun(UdonSharpBehaviour receiver, string completionEvent)`,
  `int GetRunStatus()`, and `int TryGetRunResult(out int result)`: run `run`
  asynchronously and read its `udononnxStatus_t`.
- `int TryDispose(out int result)`: `dispose`, which launches no GPU work
  and therefore runs synchronously.
- `int TryWriteImage(float[] source)`: upload the 784 pixels of the `image`
  input, row by row.
- `int StartReadLogits(float[] destination, UdonSharpBehaviour receiver, string completionEvent)`
  and `int GetLastReadbackStatus()`: read the 10 `logits` asynchronously.

`MnistClassifier.cs` (identical to the **ONNX MNIST** Package Sample) follows
the pattern of the SAXPY tutorial: `OnEnable` initializes the model one frame
later, `Interact` writes a seven drawn as text, runs the model, reads the
logits and logs the digit with the largest logit, and `OnDisable` disposes the
model:

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

    private void OnEnable()
    {
        // One frame later every behaviour, the binding included, is ready.
        SendCustomEventDelayedFrames(nameof(InitializeModel), 1);
    }

    private void OnDisable()
    {
        if (!initialized)
            return;
        initialized = false;
        // Dispose releases the session and the buffers of the C Host; the
        // next OnEnable initializes them again.
        int libraryStatus;
        int status = binding.TryDispose(out libraryStatus);
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess || libraryStatus != 0)
            Debug.LogError(
                "MNIST: Dispose failed with status " + status + " and ONNX status " + libraryStatus
            );
    }

    public void InitializeModel()
    {
        if (initialized)
            return;
        if (binding == null)
        {
            Debug.LogError("MNIST: assign the UdonclGenericOnnxMnistBinding instance.");
            return;
        }

        image = new float[28 * 28];
        logits = new float[10];
        for (int i = 0; i < image.Length; i++)
            image[i] = Digit[i] == '#' ? 1.0f : 0.0f;

        // Initialize creates the OpenCL context, the image and logits
        // buffers and the ONNX session in the C Host and binds the buffers to
        // the model input and output. The weights are compiled into the
        // artifact, so it enqueues no Device work and completes at once.
        int libraryStatus = 0;
        int status = binding.TryInitialize(out libraryStatus);
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

        status = binding.StartRun(this, nameof(OnRunComplete));
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

        status = binding.StartReadLogits(logits, this, nameof(OnLogitsRead));
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

Each interaction with the object logs `MNIST: predicted 7 (logit ...)` with
the largest logit; edit `Digit` to draw another digit. Leaving Play Mode
disables the behaviour, which disposes the model.

> **Screenshot (pending):** Unity Console in Play Mode showing "MNIST: predicted 7" — docs/images/getting-started/mnist-console-output.png

### Chat with TinyLlama

This tutorial builds [`examples/tinyllama-chat`](examples/tinyllama-chat/README.md)
from the real TinyLlama-1.1B-Chat model and its tokenizer, and chats with it
from U#. The whole chat is one Host C file and one OpenCL C kernel: the Host
applies the chat template, tokenizes your message, runs the model on the GPU
one position at a time, selects each next token with the kernel, and decodes
the reply, which U# streams while it is generated. A conversation holds 2048
tokens. Build the compiler first as described in
[Install and build](#install-and-build), including `install_onnx_mlir.py`,
and work through the [MNIST tutorial](#compile-your-own-onnx-model-with-host-c-mnist)
first: this one uses the same ONNX workflow at a much larger scale.

The model is too large to ship with the Unity package, so there is no bundled
import; the **TinyLlama Chat** Package Sample needs the artifact you build
here. Keep these limits in mind:

- The model download is the pinned 4-bit quantized
  `onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX` `model_q4.onnx` (about
  910 MB). Its key/value cache takes 88 MB of GPU memory at 2048 tokens.
- A conversation grows. Every message continues after the previous reply
  until the 2048 tokens are used up; then `ContinueMessage` fails with
  `CAPACITY_EXCEEDED`. `Dispose` followed by `Initialize` starts a new one,
  and `RestoreConversation` continues one from its token history.
- The message runs one token per model run (there is no batched prefill), and
  a reply has at most 32 tokens.

[`examples/onnx-transformer-tiny`](examples/onnx-transformer-tiny/README.md)
is the advanced version of the same chat: a 4096-token context, a reset
export, a token-array Host separated from its text layer, and a decode loop
that overlaps the next model run with the token read-back.

#### 1. The files you write

**The Host C file**
([`tinyllama_chat.c`](examples/tinyllama-chat/tinyllama_chat.c)). It owns
the OpenCL context and queue and every graph input and output, drives the
model through the UdonONNX session API of the MNIST tutorial, and is the
library's whole U# API:

| C export | U# method | What it does |
| --- | --- | --- |
| `initialize()` | `StartInitialize` | Creates the OpenCL objects, the session and the ArgMax kernel |
| `begin_message(text)` | `TryBeginMessage` | Starts tokenizing the next user message |
| `continue_message()` | `TryContinueMessage` | Tokenizes a bounded slice: 1 while work is pending, 0 once the message is ready, -status on failure |
| `generate()` | `StartGenerate` | Runs the message, then replies greedily until EOS or 32 tokens |
| `read_streamed_text(destination, offset)` | `TryReadStreamedText` | Copies the reply text that is final so far; also while `Generate` runs |
| `read_generated_tokens(destination)` | `TryReadGeneratedTokens` | Copies the token IDs of the last reply |
| `read_conversation_tokens(destination)` | `TryReadConversationTokens` | Copies the committed history: every token so far, the last reply closed |
| `restore_conversation(tokens, count)` | `StartRestoreConversation` | Continues the conversation from a committed history; `count` 0 starts a new one |
| `dispose()` | `TryDispose` | Releases everything; `Initialize` may run again |

The model library bounds `past_sequence_length` instead of fixing it (see
the CMake project below), so one run is one position `t` on a past of `t`
rows and costs as many rows as the conversation holds. Each of the 22 layers
keeps its K and its V in one buffer of the whole capacity, `[1, 4, 2048, 64]`
floats, in the arrays `keys[22]` and `values[22]` that `initialize` creates in
a loop and `dispose` releases in a loop. Every run binds the attention mask
at `t + 1`, and each layer's past at `t` rows and present at `t + 1` rows,
both in the same buffer with `buffer_dims` of the capacity. The past is then
the leading block of the present, so the run writes only the new row (an
in-place Concat; see
[operator-ownership.md](docs/architecture/operator-ownership.md#in-place-concat)):

```c
static udononnxStatus_t bind_position(uint32_t t) {
  int64_t mask_dims[2] = {1, (int64_t)t + 1};
  int64_t past_dims[4] = {1, 4, (int64_t)t, 64};
  int64_t present_dims[4] = {1, 4, (int64_t)t + 1, 64};
  udononnxStatus_t status;
  status = udononnxBindInput(session, "attention_mask", attention_mask, 2,
                             mask_dims, mask_buffer_dims);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  for (int layer = 0; layer < LAYERS; ++layer) {
    status = udononnxBindInput(session, past_key_names[layer], keys[layer], 4,
                               past_dims, cache_buffer_dims);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
    /* ... the past value, then the present key and value ... */
  }
  return UDONONNX_STATUS_SUCCESS;
}
```

The token, its position and the logits keep their shape, so `initialize`
binds them once; `generate` binds the rest at every run. Each tensor is bound
from one export only. `generate` runs the message one token per run, then
lets the ArgMax kernel pick each next token from the logits and reads that
one word back:

```c
  for (index = 0; index < count; ++index) {
    status = run_token((uint32_t)message_words[index]);
    if (status != UDONONNX_STATUS_SUCCESS)
      return fail(status);
  }
  for (index = 0; index < limit; ++index) {
    /* The token selected last runs before the next one is selected. */
    if (index > 0) {
      status = run_token(selected);
      if (status != UDONONNX_STATUS_SUCCESS)
        return fail(status);
    }
    status = select_token(); /* the ArgMax kernel, then a 4-byte read */
    /* ... record the token and decode it into the streamed text ... */
    if (selected == (uint32_t)model.eos_token_id)
      break;
  }
```

The reply's last token is never run; the next message runs it first,
followed by EOS when the reply was cut off. The message itself is kept in a
file-scope `udon_string_t message` between `begin_message` and the
`continue_message` calls, which tokenize it with the chat template compiled
from the tokenizer files (UdonTokenizers) a bounded amount of work at a time.
`udon_string_t` parameters become U# `string` parameters, and
`udon_int32_array_t` becomes `int[]`.

**The ArgMax kernel**
([`argmax.cl`](examples/tinyllama-chat/argmax.cl)), launched with
`clCreateKernel`, `clSetKernelArg` and a one-item `clEnqueueNDRangeKernel`:

```c
/* Greedy decoding: the index of the largest of the 32000 logits of the last
 * run, ties to the lowest index. One work item reads every logit. */
#define VOCABULARY 32000u

__kernel void tinyllama_argmax(__global const float *logits,
                               __global uint *token) {
  uint best = 0u;
  float value = 0.0f;
  for (uint index = 0u; index < VOCABULARY; ++index) {
    float candidate = logits[index];
    if (index == 0u || candidate > value) {
      value = candidate;
      best = index;
    }
  }
  token[get_global_id(0)] = best;
}
```

**The CMake project**
([`CMakeLists.txt`](examples/tinyllama-chat/CMakeLists.txt)):

```cmake
math(EXPR past_length "${capacity} - 1")
udoncl_add_onnx_host_library(tinyllama_model
  MODEL "${UDONCL_TINYLLAMA_ONNX_MODEL}"
  DIMENSIONS batch_size=1 sequence_length=1
  DIMENSION_BOUNDS past_sequence_length=${past_length})

# The Host C file, compiled next to the tokenizer tables.
udontokenizers_add_huggingface_model(tinyllama_chat_host
  TOKENIZER_DIRECTORY "${UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY}"
  SYMBOL tinyllama_tokenizer
  SOURCES tinyllama_chat.c)
target_compile_definitions(tinyllama_chat_host INTERFACE
  TINYLLAMA_CAPACITY=${capacity}u
  TINYLLAMA_MAX_NEW_TOKENS=${max_new_tokens}u)

udoncl_add_device_library(tinyllama_argmax argmax.cl)

udoncl_add_unity_library(tinyllama-chat
  DEVICE_TEXEL_FORMAT rgba32_float
  OMIT_INTERMEDIATE_IR
  HOST_LIBRARIES
    tinyllama_model
    tinyllama_chat_host
    UdonONNXRuntime::UdonONNXRuntime
    UdonTokenizers::UdonTokenizers
  DEVICE_LIBRARIES
    tinyllama_argmax
  LIBRARIES
    UdonONNXOperators::UdonONNXOperators
    UdonATen::UdonATen
    UdonBLAS::UdonBLAS)
```

The real file also finds the packages, sets the include directories and
checks these cache variables:

- `UDONCL_TINYLLAMA_ONNX_MODEL` and `UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY`
  (required): the downloaded model and tokenizer files.
- `UDONCL_TINYLLAMA_CONTEXT_CAPACITY` (default 2048): the tokens one
  conversation holds. TinyLlama-1.1B was trained on 2048 positions.
- `UDONCL_TINYLLAMA_MAX_NEW_TOKENS` (default 32): the most tokens of one
  reply.

**The public API description.** No config file describes the artifact.
`UDONCL_PROGRAM("tinyllama_chat")` in the Host names the program, so Unity
imports it under `Assets/udoncl-generated/tinyllama_chat` and the binding is
the default `UdonclTinyllamaChatBinding` in `Lemolatoon.Udoncl.Generated`.
The exports are U# methods named after the PascalCase of their C names
(`continue_message` -> `ContinueMessage`), with the C parameter names.
`read_streamed_text` is `UDONCL_EXPORT_WHILE_BUSY`, so U# can poll it while
`Generate` runs. No tensor is public: text goes in as a U# string and comes
out through `int[]` arrays, and the compiler derives the Device buffers from
the ONNX and C Hosts. `DEVICE_TEXEL_FORMAT rgba32_float` stores four words per
RGBA32 float texel in the real model's Device buffers and weights.

#### 2. Build

From the repository root, fetch and verify the pinned model and tokenizer.
Each script downloads into `.deps/models` once (set `UDONCL_MODEL_CACHE` to
use another directory) and prints the cached path, so the output can be
captured directly. Then configure the example with both paths and build it:

```powershell
cmake --preset udoncl
python scripts/build.py
$model = python scripts/fetch_tinyllama_onnx.py
$tokenizer = python scripts/fetch_tinyllama_tokenizer.py
cmake -S examples/tinyllama-chat -B build/tinyllama-chat -G Ninja `
  -DCMAKE_PREFIX_PATH="$PWD" `
  -DUDONCL_TINYLLAMA_ONNX_MODEL="$model" `
  -DUDONCL_TINYLLAMA_TOKENIZER_DIRECTORY="$tokenizer"
cmake --build build/tinyllama-chat --target tinyllama-chat
```

Configuring compiles the real model once, so it takes much longer than the
MNIST project. The importable artifact is written to:

```text
build/tinyllama-chat/udoncl/tinyllama-chat/artifact
```

#### 3. Import into Unity

1. In a Unity 2022.3 VRChat Worlds project with the
   **UdonCL Runtime and ONNX Importer** package, choose
   **Tools → udoncl → Import Artifacts...** and select the
   `build/tinyllama-chat/udoncl/tinyllama-chat/artifact` directory.

   > **Screenshot (pending):** Tools → udoncl → Import Artifacts… with the tinyllama-chat artifact folder selected — docs/images/getting-started/tinyllama-import-artifacts.png

2. In the **udoncl Library Import** dialog, choose **Add after Compile**.
   After the generated U# wrapper compiles, Unity creates
   `Assets/udoncl-generated/tinyllama_chat/Prefabs/UdonclTinyllamaChatBinding.prefab`
   and places an instance in the active scene.

   > **Screenshot (pending):** Project window showing Assets/udoncl-generated/tinyllama_chat with its Bindings and Prefabs folders — docs/images/getting-started/tinyllama-generated-assets.png

3. Choose **Create → U# Script**, name it `TinyLlamaChat`, and replace its
   contents with the script in the next step, or import the same script as
   the **TinyLlama Chat** sample from Package Manager (**UdonCL Runtime and
   ONNX Importer** → **Samples** → **Import**).
4. Create a GameObject (for example a Cube, whose Collider lets you interact
   with it), add the **Tiny Llama Chat** component, and drag the
   `UdonclTinyllamaChatBinding` scene instance into its **Binding** field.
   Edit **User Text** as you like.

   > **Screenshot (pending):** Inspector of the TinyLlamaChat GameObject with Binding and User Text — docs/images/getting-started/tinyllama-inspector.png

5. Enter Play Mode and watch the Console.

#### 4. Your UdonSharpBehaviour

The generated binding follows the same rules as before: an export that
launches GPU work becomes `Start<Name>(..., receiver, completionEvent)` with
`Get<Name>Status()` and `TryGet<Name>Result(out ...)`, and any other export
becomes `Try<Name>(..., out result)`. `TinyLlamaChat` follows the lifecycle of
the other tutorials:

- `OnEnable` starts `Initialize` one frame later, and `OnDisable` calls
  `TryDispose`, which releases the model.
- `Interact` sends **User Text**: `StartRestoreConversation` with the synced
  history, then `TryBeginMessage(message, out status)`,
  `TryContinueMessage(out pending)` once per frame until it returns 0, and
  `StartGenerate`.
- While `Generate` runs, `TryReadStreamedText(textChunk, replyLength, out
  count)` reads the part of the reply that is final, every frame.

The players of an instance share one conversation
(`BehaviourSyncMode.Manual`). It syncs the conversation's token history
(`TryReadConversationTokens`), its text log and a `busy` flag, never the KV
cache. The player who sends a message takes ownership, restores the history
into their own Host (only the tokens it does not hold yet run), generates the
reply and calls `RequestSerialization` once with the new history and log;
the others only show the log. A sender that leaves drops the reply, and
`Clear` syncs an empty conversation. Alone in an instance it is a plain
chat. The sample's README describes the flow; multi-player sync is checked
by hand only.

The exports that return a `udononnxStatus_t` report `0` on success; the
sample checks it next to the binding status.

`TinyLlamaChat.cs` (identical to the **TinyLlama Chat** Package Sample):

```csharp
using Lemolatoon.Udoncl.Generated;
using UdonSharp;
using UnityEngine;
using UnityEngine.UI;
using VRC.SDKBase;

// One conversation shared by every player of the instance. The player who
// sends the next message generates its reply on their own GPU: they take
// ownership, restore the synced history into their Host, generate, and sync
// the new history and text log once. Everyone else only shows the log.
[UdonBehaviourSyncMode(BehaviourSyncMode.Manual)]
public class TinyLlamaChat : UdonSharpBehaviour
{
    // The status the C Host returns from its exports (udononnxStatus_t).
    private const int OnnxSuccess = 0;
    private const int OnnxCapacityExceeded = 13;

    // UDONCL_TINYLLAMA_CONTEXT_CAPACITY of the artifact: the longest history.
    private const int Capacity = 2048;

    [SerializeField]
    private UdonclTinyllamaChatBinding binding;

    [SerializeField]
    private string userText = "What is your name?";

    // Optional: shows the conversation log.
    [SerializeField]
    private Text logText;

    // The committed history of the conversation (ReadConversationTokens),
    // its text, and whether a player is generating the next reply.
    [UdonSynced]
    private int[] conversationTokens = new int[0];

    [UdonSynced]
    private string conversationLog = "";

    [UdonSynced]
    private bool busy;

    private bool initialized;

    // This player's Host is initializing, restoring or replying.
    private bool working;
    private string message;
    private int[] textChunk;
    private int[] historyChunk;
    private int replyLength;
    private string reply;
    private int shownLength;

    private void OnEnable()
    {
        // One frame later every behaviour, the binding included, is ready.
        SendCustomEventDelayedFrames(nameof(InitializeChat), 1);
    }

    private void OnDisable()
    {
        if (!initialized)
            return;
        initialized = false;
        // Dispose releases the model and this player's copy of the
        // conversation; the synced history stays.
        int onnxStatus;
        int status = binding.TryDispose(out onnxStatus);
        if (status != UdonclTinyllamaChatBinding.StatusSuccess || onnxStatus != OnnxSuccess)
            Debug.LogError(
                "TinyLlama: Dispose failed with status " + status + " and ONNX status " + onnxStatus
            );
    }

    public void InitializeChat()
    {
        if (initialized || working)
            return;
        if (binding == null)
        {
            Debug.LogError("TinyLlama: assign the UdonclTinyllamaChatBinding instance.");
            return;
        }

        textChunk = new int[256];
        historyChunk = new int[Capacity];
        working = true;
        // Initialize creates the OpenCL objects, the model session and the
        // ArgMax kernel in the C Host; it spans a few frames.
        int status = binding.StartInitialize(this, nameof(OnChatInitialized));
        if (status != UdonclTinyllamaChatBinding.StatusSuccess)
            Fail("StartInitialize", status, OnnxSuccess);
    }

    public void OnChatInitialized()
    {
        int onnxStatus = OnnxSuccess;
        int status = binding.GetInitializeStatus();
        if (status == UdonclTinyllamaChatBinding.StatusSuccess)
            status = binding.TryGetInitializeResult(out onnxStatus);
        if (!Succeeded("Initialize", status, onnxStatus))
            return;
        initialized = true;
        working = false;
        ShowLog();
    }

    public override void Interact()
    {
        Send();
    }

    // Sends userText as the next message of the shared conversation.
    public void Send()
    {
        if (!initialized || working)
            return;
        if (busy)
        {
            Debug.Log("TinyLlama: another player's reply is being generated.");
            return;
        }

        // Take the conversation: the busy flag keeps the other players from
        // sending until this reply is synced.
        working = true;
        Networking.SetOwner(Networking.LocalPlayer, gameObject);
        busy = true;
        RequestSerialization();
        message = userText;

        // Rebuild this player's KV cache from the synced history. Only what
        // follows the tokens the Host already holds runs, so the player who
        // sent the last message restores almost nothing.
        int status = binding.StartRestoreConversation(
            conversationTokens,
            conversationTokens.Length,
            this,
            nameof(OnConversationRestored)
        );
        Succeeded("StartRestoreConversation", status, OnnxSuccess);
    }

    public void OnConversationRestored()
    {
        int onnxStatus = OnnxSuccess;
        int status = binding.GetRestoreConversationStatus();
        if (status == UdonclTinyllamaChatBinding.StatusSuccess)
            status = binding.TryGetRestoreConversationResult(out onnxStatus);
        if (!Succeeded("RestoreConversation", status, onnxStatus))
            return;

        status = binding.TryBeginMessage(message, out onnxStatus);
        if (!Succeeded("BeginMessage", status, onnxStatus))
            return;
        EncodeMessage();
    }

    // The C Host applies the chat template and tokenizes the message, a
    // bounded amount of work per frame: ContinueMessage returns 1 while work
    // is pending and 0 once the message is ready.
    public void EncodeMessage()
    {
        int pending;
        int status = binding.TryContinueMessage(out pending);
        if (!Succeeded("ContinueMessage", status, pending < 0 ? -pending : OnnxSuccess))
            return;
        if (pending != 0)
        {
            SendCustomEventDelayedFrames(nameof(EncodeMessage), 1);
            return;
        }

        reply = "";
        replyLength = 0;
        status = binding.StartGenerate(this, nameof(OnReplyGenerated));
        if (!Succeeded("StartGenerate", status, OnnxSuccess))
            return;
        SendCustomEventDelayedFrames(nameof(StreamReply), 1);
    }

    // While the reply is generated, read the part of it that is final.
    public void StreamReply()
    {
        if (!working)
            return;
        ReadReplyText();
        if (logText != null)
            logText.text = conversationLog + Turn();
        SendCustomEventDelayedFrames(nameof(StreamReply), 1);
    }

    public void OnReplyGenerated()
    {
        int onnxStatus = OnnxSuccess;
        int status = binding.GetGenerateStatus();
        if (status == UdonclTinyllamaChatBinding.StatusSuccess)
            status = binding.TryGetGenerateResult(out onnxStatus);
        if (!Succeeded("Generate", status, onnxStatus))
            return;

        // After the generation completed the streamed text is the whole reply.
        ReadReplyText();

        // The new history: the old one, the message and the reply.
        int count;
        status = binding.TryReadConversationTokens(historyChunk, out count);
        if (status == UdonclTinyllamaChatBinding.StatusSuccess && count == -OnnxCapacityExceeded)
        {
            // The reply filled the conversation: no message can follow it,
            // so it is not synced.
            Debug.LogWarning(
                "TinyLlama: the conversation is full; Clear starts a new one." + Turn()
            );
            Release();
            return;
        }
        if (!Succeeded("ReadConversationTokens", status, count < 0 ? -count : OnnxSuccess))
            return;
        if (!Networking.IsOwner(gameObject))
        {
            // Another player sent a message at the same time and took the
            // conversation; theirs is synced.
            Debug.Log("TinyLlama: another player took the conversation." + Turn());
            working = false;
            return;
        }

        int[] tokens = new int[count];
        for (int i = 0; i < count; i++)
            tokens[i] = historyChunk[i];
        conversationTokens = tokens;
        conversationLog = conversationLog + Turn();
        busy = false;
        working = false;
        RequestSerialization();
        ShowLog();
    }

    // Starts a new conversation for everyone: an empty history and log.
    public void Clear()
    {
        if (busy || working)
            return;
        Networking.SetOwner(Networking.LocalPlayer, gameObject);
        conversationTokens = new int[0];
        conversationLog = "";
        RequestSerialization();
        ShowLog();
    }

    public override void OnDeserialization()
    {
        ShowLog();
    }

    // When the player generating a reply leaves, another player becomes the
    // owner with the busy flag still set: the reply is dropped and the last
    // synced history stays.
    public override void OnOwnershipTransferred(VRCPlayerApi player)
    {
        if (!Utilities.IsValid(player) || !player.isLocal || working || !busy)
            return;
        busy = false;
        RequestSerialization();
    }

    // The log lines of this player's message and the reply so far.
    private string Turn()
    {
        VRCPlayerApi player = Networking.LocalPlayer;
        string name = Utilities.IsValid(player) ? player.displayName : "you";
        return (conversationLog.Length == 0 ? "" : "\n")
            + name
            + ": "
            + message
            + "\nTinyLlama: "
            + reply;
    }

    // Logs the part of the synced log this player has not seen yet.
    private void ShowLog()
    {
        if (conversationLog.Length < shownLength)
            Debug.Log("TinyLlama: the conversation was cleared.");
        else if (conversationLog.Length > shownLength)
            Debug.Log("TinyLlama: " + conversationLog.Substring(shownLength));
        shownLength = conversationLog.Length;
        if (logText != null)
            logText.text = conversationLog;
    }

    // Appends the reply text that became final since the last call. The
    // binding returns UTF-16 code units, one per int, at most textChunk's
    // length per call.
    private void ReadReplyText()
    {
        int count = textChunk.Length;
        while (count == textChunk.Length)
        {
            int status = binding.TryReadStreamedText(textChunk, replyLength, out count);
            if (status != UdonclTinyllamaChatBinding.StatusSuccess || count < 0)
                return;
            for (int i = 0; i < count; i++)
                reply += (char)textChunk[i];
            replyLength += count;
        }
    }

    private bool Succeeded(string step, int status, int onnxStatus)
    {
        if (status == UdonclTinyllamaChatBinding.StatusSuccess && onnxStatus == OnnxSuccess)
            return true;
        Fail(step, status, onnxStatus);
        return false;
    }

    // A failed step drops the reply: the synced history stays as it was.
    private void Fail(string step, int status, int onnxStatus)
    {
        Release();
        Debug.LogError(
            "TinyLlama: "
                + step
                + " failed with status "
                + status
                + " and ONNX status "
                + onnxStatus
        );
    }

    // Lets the other players send again.
    private void Release()
    {
        working = false;
        if (busy && Networking.IsOwner(gameObject))
        {
            busy = false;
            RequestSerialization();
        }
    }
}
```

#### 5. Expected result

Interact with the object: the sample streams the reply and, once the
generation completes, logs the new lines of the conversation, `<your name>:
What is your name?` and `TinyLlama: ...`. Decoding is greedy, so the same
conversation gives the same reply. Interact again to send **User Text** as
the next turn of the same conversation. Leaving Play Mode disables the
behaviour, which disposes the model.

> **Screenshot (pending):** Unity Console in Play Mode showing the conversation lines — docs/images/getting-started/tinyllama-console-output.png

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
  which public library owns each ONNX/PyTorch operator, and what the UdonONNX
  session API mirrors from ONNX Runtime and where it deviates.
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
