# UdonCL examples

The examples are ordered from the smallest supported user workflow to the
largest ONNX model. They use public CMake, compiler, artifact, VPM, and Unity
importer interfaces; test-only fixture commands are not required. The
[README tutorials](../README.md#tutorials) walk through `saxpy`, `onnx-mnist`,
and the TinyLlama chat of `onnx-transformer-tiny` step by step, including the
U# code that calls them.

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
| [`hello-kernel`](hello-kernel/README.md) | User CMake Host C + OpenCL C | Generated U# bindings for the importable scalar and tokenizer libraries; a Host/Device boundary artifact for inspection | Scalar Host exports; the boundary artifact is not importable |
| [`saxpy`](saxpy/README.md) | Standard OpenCL Host API + OpenCL C kernel on `UdonCL::OpenCLRuntime` | Asynchronous `UDONCL_EXPORT` method, buffers `x`, `y` and `result`, public Write/Read API; **Import C and OpenCL SAXPY** and the **C and OpenCL SAXPY** Package Sample | Fixed buffers; constant NDRange derived from the Host C |
| [`udonblas-sgemm`](udonblas-sgemm/README.md) | Host C calling the cuBLAS-shaped UdonBLAS API | Asynchronous `RunUdonBlasSgemm` export and per-column result getter | MNIST Dense dimensions (`m=1, n=10, k=144`) |
| [`onnx-mnist`](../README.md#compile-your-own-onnx-model-with-host-c-mnist) | Standard ONNX MNIST CNN + C Host on `UdonONNXRuntime` | Generated U# binding; **Import ONNX MNIST** and the **ONNX MNIST** Package Sample | Fixed batch of one image |
| [`onnx-transformer-tiny`](onnx-transformer-tiny/README.md) | Synthetic transformer, or the real TinyLlama-1.1B with its tokenizer | Generated U# bindings; the 4096-token Chat Artifact backs the **TinyLlama Chat** Package Sample | The real model is built from user-downloaded files |
| [`mnist-cnn`](mnist-cnn/README.md) | Trained CNN | Native C reference and the ONNX model of `onnx-mnist` | Reference only |

`onnx-q4-component`, `onnx-rms-component`, and `onnx-transpose-component` are
not tutorials: each compiles one TinyLlama-sized ONNX operator (Q4
`MatMulNBits`, `RMSNormalization`, `Transpose`) into a Unity library for the
focused operator tests under `tests/build`.

## Fastest importable example

After installing the VPM package, the shortest user path needs no compiler
checkout. The package bundles two prebuilt artifacts under
**Tools → udoncl → Examples**:

- **Import C and OpenCL SAXPY**: the `saxpy` artifact.
- **Import ONNX MNIST**: the `onnx-mnist` artifact.

Each import generates a typed U# binding prefab after Unity reloads the
imported wrapper, and attaches a generated example controller to it. Choose
**Add after Compile** in the import dialog to place the prefab in the active
scene, then interact with it in Play Mode to run the example.

The package also ships three Package Samples (Package Manager → **UdonCL
Runtime and ONNX Importer** → **Samples**), each a U# component of your own
that calls a generated binding:

- **C and OpenCL SAXPY** (`SaxpyCaller`) and **ONNX MNIST**
  (`MnistClassifier`) work with the two bundled imports above.
- **TinyLlama Chat** (`TinyLlamaChat`) needs a 4096-token Chat Artifact that
  you build from the downloaded model; the model is too large to bundle.

To build the artifacts yourself instead, follow the
[README tutorials](../README.md#tutorials): configure the example as a CMake
project and import its `artifact` directory with
**Tools → udoncl → Import Artifacts...**.

## C and OpenCL C consumer example

Configure `hello-kernel` as an independent user project and build its
importable scalar Host library:

```powershell
cmake -S examples/hello-kernel -B build/hello-kernel-cmake -G Ninja `
  -DUdonCL_DIR="$PWD/cmake"
cmake --build build/hello-kernel-cmake --target hello-unity-library
```

Import `build/hello-kernel-cmake/udoncl/hello-unity-library/artifact`. The
importer creates the Udon program and the typed U# binding. See the example
README for the individual CLI steps, and [`saxpy`](saxpy/README.md) for a Host
C that launches an OpenCL C kernel.

## Native and compiler-development examples

These examples exercise the compiler and its native runtimes outside Unity.
They are mainly useful when developing UdonCL itself. Run the commands from the
repository root; `make` targets configure and build the compiler first. See
[../docs/development.md](../docs/development.md) for the developer toolchain
setup.

### C/OpenCL C compiler-boundary probe

Original compiler-boundary probe (after installing the pinned LLVM SDK):

```text
make example-c-boundary
```

The Makefile configures and builds the project, then runs
`examples/c-boundary/run.py`. It compiles `host.c` and `device.cl` with the Clang
from the configured LLVM SDK, imports the actual LLVM IR, runs recovery Passes,
and emits a paired boundary bundle. No hand-written MLIR is substituted.
To choose another output directory, use
`make example-c-boundary BOUNDARY_OUTPUT=build/my-c-example`.

Without Make (including Developer PowerShell on Windows), run:

```text
cmake --preset udoncl
python scripts/build.py
python examples/c-boundary/run.py
```

The script handles `.exe` automatically. With Make on Windows, set
`PYTHON=python` if `python3` is unavailable. The LLVM IR files are written to
`build/c-boundary/host.ll` and `device.ll`; the bundle is under
`build/c-boundary/bundle/`. See [c-boundary/README.md](c-boundary/README.md)
for the source contract, intermediate IR and current limitations.

### Native Host + Fragment Shader execution

```text
# Ubuntu graphics dependencies (Windows uses the Windows SDK / WARP):
sudo apt-get install libvulkan-dev mesa-vulkan-drivers glslang-tools
make example-native-fragment
```

This compiles the Host through `udonhl → func/arith → LLVM dialect → LLVM IR →
native object`, then links it with the independent `VRCNativeFragmentRuntime`
library. Generated Host dispatch calls execute the generated Device fragment
shader on Lavapipe (Linux) or WARP (Windows), read back RGBA8 bytes and compare
f32 bit patterns against a C oracle. See
[native-fragment/README.md](native-fragment/README.md) for
bindings, intermediate artifacts, Windows commands and scope. Enable the same
CTest flow with `-DVRC_ENABLE_NATIVE_FRAGMENT_TESTS=ON`; CI enables it explicitly.

### Ordinary OpenCL Host API

```text
make example
# equivalent: make example-opencl
```

`opencl-host/host.c` uses the Khronos OpenCL 1.2 declarations to create a
context/queue/program/kernel/buffer, copy kernel arguments, enqueue an NDRange,
read the result and release objects. The generated Host computes `result + 1`
after readback. Native linking includes `VRCNativeOpenCLRuntime`: a C OpenCL API
implementation plus the binding to the fragment renderer. No system OpenCL
loader or vendor OpenCL driver is used. Linux requires the Vulkan/Lavapipe test
dependencies listed above; Windows uses WARP.

Outputs are under `build/opencl-host`; use `EXAMPLE_OUTPUT=build/another-opencl`
to choose a fresh directory. See [opencl-host/README.md](opencl-host/README.md)
for supported APIs, native ABI preservation and the AOT subset's limits.

### UdonBLAS on the native fragment backend

UdonBLAS is the reusable CUDA-family library layer on top of that runtime. Its
public CUDA-13.0-compatible subset includes fragment-parallel SAXPY, SGEMV,
SGEMM, and the two-kernel balanced-tree SDOT reduction:

```text
python libraries/udonblas/run_device.py --build-dir build/udoncl
```

The gather profile supports one- and two-dimensional NDRange ownership. SGEMM
uses a 32×32 fragment target, assigns one active fragment to every output
element, and validates rectangular/transposed/padded matrices against an
independent CPU reference on the real fragment backend. See
[libraries/udonblas/README.md](../libraries/udonblas/README.md) for the stable API
and the initial 16×16 profile limits.

### MD crystal

A host-native example of a 64-particle MD crystal that you can heat by touching it:

```sh
make example-md  # checks CPU/GPU agreement, conservation laws, local heating, and ice -> liquid -> refreeze
make md-view     # touch and control it in the browser
```

See the [MD crystal README](md-crystal/README.md) for the computational model,
the Host/Device paths, dependencies, and controls.

### MNIST CNN reference

The [MNIST CNN reference](mnist-cnn/README.md) lives in `examples/mnist-cnn`. Its compact
conv/ReLU/pooling/dense model reaches 95.49% on the MNIST test set. The C oracle
is executed directly and after a native UdonHL round trip by `mnist-cnn-native`.
Its `generate_onnx.py` writes the same trained model as ONNX for
`examples/onnx-mnist`.

## Verification

Repository changes, including changes to these examples, are validated by the
background commit validation service rather than by running the full gate by
hand. See the
[commit validation service](../docs/development.md#commit-validation-service)
section of the developer guide.
