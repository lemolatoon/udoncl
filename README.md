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
compiler-emitted Udon heap slots. Its artifact contains `manifest.json`,
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

**The OpenCL C kernel** ([`unity_saxpy.cl`](examples/saxpy/unity_saxpy.cl)).
The current Device profile, `gather-f32`, runs one work item per word of the
output buffer, and every work item writes exactly its own word
`output[gid]`. The input buffer holds `x` in words 0–15 and `y` in words
16–31; the output buffer has the same 32-word size, so its upper half simply
copies `y`:

```c
#define SAXPY_WORDS 32u
#define SAXPY_VECTOR_WORDS 16u

__kernel void saxpy_demo(__global const float *input, __global float *output,
                         int n, float alpha) {
  uint gid = (uint)get_global_id(0);
  float value = input[gid];
  if (gid < (uint)n && gid < SAXPY_VECTOR_WORDS)
    value = alpha * input[gid] +
            input[SAXPY_VECTOR_WORDS + (gid & (SAXPY_VECTOR_WORDS - 1u))];
  output[gid] = value;
}
```

**The Host C** ([`host.c`](examples/saxpy/host.c)). Host code uses the
ordinary Khronos OpenCL 1.2 API. The exported function `run_saxpy` performs
one complete, straight-line setup, dispatch, and release sequence; the
compiler recovers it into a typed Unity dispatch. The essential part is:

```c
void run_saxpy(int n, float alpha) {
  /* ... clGetPlatformIDs, clGetDeviceIDs, clCreateContext,
     clCreateCommandQueue ... */
  input = clCreateBuffer(context, CL_MEM_READ_ONLY, 32 * sizeof(float), 0, 0);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 32 * sizeof(float), 0, 0);
  program = clCreateProgramWithSource(context, 1, &source, 0, 0);
  (void)clBuildProgram(program, 0, 0, 0, 0, 0);
  kernel = clCreateKernel(program, "saxpy_demo", 0);
  (void)clSetKernelArg(kernel, 0, sizeof(input), &input);
  (void)clSetKernelArg(kernel, 1, sizeof(output), &output);
  (void)clSetKernelArg(kernel, 2, sizeof(n), &n);
  (void)clSetKernelArg(kernel, 3, sizeof(alpha), &alpha);
  (void)clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global, &local, 0, 0, 0);
  /* ... release the kernel, program, buffers, queue, and context ... */
}
```

This is a deliberately fixed AOT subset. The calls must appear in exactly this
order with no status-dependent control flow: one platform, one GPU device, one
context and in-order queue, read-only input buffers and exactly one write-only
output buffer, `clSetKernelArg` calls in argument order (buffers before
f32/i32 scalars), one 1D enqueue with a constant global size and a local size
of 1, and the matching release calls. Kernels are not compiled at run time:
the kernel body comes from the Device library in `CMakeLists.txt`, and the
name passed to `clCreateKernel` must match `kernel` in `unity-library.json`.
Unsupported patterns are compile errors rather than approximations.

**The CMake project** ([`CMakeLists.txt`](examples/saxpy/CMakeLists.txt)) is
an ordinary consumer of the UdonCL package:

```cmake
cmake_minimum_required(VERSION 3.30...4.2)
project(udoncl_saxpy LANGUAGES NONE)

find_package(UdonCL 0.1 REQUIRED CONFIG)

udoncl_add_host_library(saxpy_host host.c)
udoncl_add_device_library(saxpy_device unity_saxpy.cl)

udoncl_add_unity_library(saxpy-unity-library
  CONFIG unity-library.json
  HOST_LIBRARIES saxpy_host
  DEVICE_LIBRARIES saxpy_device)
```

**The public API description**
([`unity-library.json`](examples/saxpy/unity-library.json)). Each key you
write:

- `schema_version`: `1`.
- `program`: the program name, `saxpy_demo`. Unity imports the artifact under
  `Assets/udoncl-generated/saxpy_demo` and names the generated U# class
  `UdonclSaxpyDemoBinding` (`Udoncl` + the PascalCase program name +
  `Binding`) in the namespace `Lemolatoon.Udoncl.Generated`. An optional
  `binding` object with `namespace` and `class_name` overrides both.
- `device_kernel`: the AOT launch contract of the kernel.
  - `schema_version`: `1`; `profile`: `"gather-f32"`.
  - `words`: the number of 32-bit words in each buffer (32), at most 32,768.
  - `kernel`: the OpenCL C kernel name (`saxpy_demo`); `kernel_id`: a positive
    integer that identifies it in the artifact manifest (10).
  - `argument_kinds`: the kernel arguments in order, `"buffer"` entries first,
    then `"i32"` / `"f32"` scalars: `["buffer", "buffer", "i32", "f32"]`.
  - `global_size`: the NDRange; its element product must equal `words`
    (`[32, 1, 1]`). `local_size` must be `[1, 1, 1]` and `global_offset`
    `[0, 0, 0]`.
- `exports`: the C functions that become U# methods. `symbol` is the C name
  (`run_saxpy`), `public_name` the U# name (`RunSaxpy`), `id` a stable export
  ID, and `parameters` the U# parameter names (`n`, `alpha`). The parameter
  types come from the C signature and must match the scalar `argument_kinds`.
- `buffers`: one entry per OpenCL buffer, in `clCreateBuffer` order. `id` is
  the buffer ID, `name` produces the U# transfer method names
  (`input` → `TryWriteInput`, `output` → `StartReadOutput`), `byte_size` must
  equal both `words * 4` and the `clCreateBuffer` size (128),
  `element_type` is `"float32"`, and `public_access` is `"input"` for a
  `CL_MEM_READ_ONLY` buffer or `"output"` for the `CL_MEM_WRITE_ONLY` buffer.

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
   the kernel material, and two 128-byte linear RGBA8 RenderTextures for the
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

- `int TryWriteInput(float[] source)`: uploads the whole `input` buffer. The
  array length must equal the buffer size in words (32).
- `int TryRunSaxpy(int n, float alpha)`: runs the `RunSaxpy` export.
  Synchronous exports get a `Try` prefix and return a status code.
- `int StartReadOutput(float[] destination, UdonSharpBehaviour receiver, string completionEvent)`:
  starts an asynchronous GPU readback of the whole `output` buffer into
  `destination` (again exactly 32 words). When it finishes, the binding calls
  `receiver.SendCustomEvent(completionEvent)`, so the event method must be
  `public`.
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

    private float[] input;
    private float[] output;

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

        // The input buffer holds 32 words: x in [0, 16) and y in [16, 32).
        input = new float[32];
        output = new float[32];
        for (int i = 0; i < 16; i++)
        {
            input[i] = i + 1;        // x[i]
            input[16 + i] = 100 + i; // y[i]
        }

        int status = binding.TryWriteInput(input);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
        {
            Debug.LogError("SAXPY: TryWriteInput failed with status " + status);
            return;
        }

        status = binding.TryRunSaxpy(16, 2.0f);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
        {
            Debug.LogError("SAXPY: TryRunSaxpy failed with status " + status);
            return;
        }

        status = binding.StartReadOutput(output, this, "OnSaxpyReadback");
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
            Debug.LogError("SAXPY: StartReadOutput failed with status " + status);
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
            float expected = 2.0f * input[i] + input[16 + i];
            Debug.Log("SAXPY: z[" + i + "] = " + output[i] + " (expected " + expected + ")");
        }
    }
}
```

#### 5. Expected result

Each Console line shows `z[i] = 2 * x[i] + y[i]`. With the inputs above,
`x[i] = i + 1` and `y[i] = 100 + i`, so `z[i] = 3 * i + 102`: `z[0] = 102`,
`z[1] = 105`, ..., `z[15] = 147`, each equal to its `expected` value. Output
words 16–31 (not logged) hold a copy of `y`. With a smaller `n`, elements at
or beyond `n` contain `x[i]` instead.

> **Screenshot (pending):** Unity Console in Play Mode showing the sixteen SAXPY result lines — docs/images/getting-started/saxpy-console-output.png

The generic buffer-ID methods `TryWriteFloatBuffer` / `TryWriteIntBuffer` and
`StartReadFloatBuffer` / `StartReadIntBuffer` remain available as a low-level
alternative to the named transfers; see
[examples/saxpy/README.md](examples/saxpy/README.md).

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
