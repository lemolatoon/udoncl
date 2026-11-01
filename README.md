# udoncl

C / OpenCL C → Udon / Fragment Shader compiler under development.
[PLAN.md](PLAN.md) is the design specification. The application entry is UdonSharp;
C is a library. This repository does not yet compile C into a working VRChat world.

LLVM, MLIR and Clang are pinned together to **23.1.4** (commit recorded in
[cmake/llvm-lock.json](cmake/llvm-lock.json)). Compiler development targets Linux,
macOS and Windows x64; the initial generated world targets VRChat PC (Windows).

Prerequisites: CMake 3.30+, Ninja, Python 3.12+, and a C++17 compiler (C++20 for the native renderer examples).
Git is required when building LLVM from source.
For the prebuilt Windows SDK, install **Visual Studio 2026** with Desktop
development with C++, and run these commands from its **x64 Native Tools /
Developer PowerShell**. The locked SDK uses newer MSVC STL symbols unavailable
in Visual Studio 2022; a VS 2022 source build must use an SDK built with that
toolset instead.

```text
python -m pip install -r scripts/requirements-llvm.txt
python scripts/install_llvm.py
cmake --preset udoncl
python scripts/build.py
ctest --preset udoncl
```

The prebuilt installer verifies pinned SHA256 checksums and installs into
`.deps/llvm23`; it never builds LLVM. Linux uses the official LLVM SDK archive.
Windows uses the community [Portable MLIR Toolchain](https://github.com/munich-quantum-software/portable-mlir-toolchain/releases/tag/2026.10.28)
Release/noassert SDK and supplements it with only `clang.exe` and Clang resource
files from the official LLVM archive. All LLVM/MLIR headers, libraries and CMake
packages come from the same SDK; official LLVM link libraries are not overlaid.
Download URLs and checksums are recorded in `cmake/llvm-lock.json`.

On Ubuntu, install the SDK's system libraries before configuring:

```text
sudo apt-get install zlib1g-dev libzstd-dev libxml2-dev
```

Once installed, subsequent builds only need the last three commands above.
CMake automatically prefers this repository-local SDK, including when
configuring without presets:

```text
cmake -S . -B build/udoncl -G Ninja -DCMAKE_BUILD_TYPE=Release
python scripts/build.py --build-dir build/udoncl
ctest --test-dir build/udoncl --output-on-failure
```

Discovery order is an explicit `VRC_LLVM_ROOT`, then `.deps/llvm23`, then system
LLVM (only if the local directory does not exist). LLVM and MLIR are selected
from the same prefix. A local SDK overrides stale `LLVM_DIR` / `MLIR_DIR` cache
entries; an incomplete local SDK fails with a diagnostic instead of falling
back to system packages. Configure output prints both selected package paths.

To use another complete 23.1.4 installation explicitly:

```text
cmake --preset udoncl -DVRC_LLVM_ROOT="path/to/llvm"
python scripts/build.py
ctest --preset udoncl
```

To return to automatic repository-local discovery, configure with
`-DVRC_LLVM_ROOT:PATH=`. Use `cmake --fresh --preset udoncl` to clear all cached
settings when changing compilers or toolchains (then supply any desired root).
`CMAKE_PREFIX_PATH` alone does not override an available repository-local SDK.
LLVM and MLIR must come from the same build, architecture and C++ runtime.
No system-wide LLVM installation is required. On Windows, use the
DLL MSVC runtime (`/MD`) for both LLVM and this project. Paths with spaces are
supported. Windows native execution is covered by CI and the local Visual
Studio 2026 validation recorded in `docs/validation.md`.

CMake emits `build/udoncl/compile_commands.json` for clangd. The checked-in
`.clangd` points to that directory on all hosts. Run configure and build once
so TableGen-generated headers also exist. With a custom build directory, set
clangd's `--compile-commands-dir` to that directory. If clangd selects a different
system C++ installation than the build compiler, allow querying the compiler
with `--query-driver=/absolute/path/to/compiler` in your editor's clangd arguments.
For example, this Linux host needs `--query-driver=/usr/bin/c++` to discover GCC
15 headers instead of an incomplete GCC 16 installation. On Windows, launch the
editor from the Developer PowerShell so clangd inherits the MSVC include paths.

`scripts/build.py` selects the number of CPUs available to the process and passes
it to CMake's `--parallel`, including on Windows. CI and Makefile targets use the
same entry point. The LLVM source builder also defaults to that CPU count. The
CMake build preset has no fixed job limit; direct Ninja builds use its native
automatic parallelism.

Before every commit, format C/C++/OpenCL C with LLVM style and C# with the
repository-pinned CSharpier tool:

```text
python -m pip install -r scripts/requirements-format.txt
python scripts/format.py
python scripts/format.py --check
```

The check mode is non-mutating and runs as a separate CI job. Formatter
versions are pinned in `scripts/requirements-format.txt` and
`.config/dotnet-tools.json`; a different `clang-format` version is rejected.

The MNIST CNN reference lives in `examples/mnist-cnn`. Its compact
conv/ReLU/pooling/dense model reaches 95.49% on the MNIST test set. The C oracle
is executed directly and after a native UdonHL round trip by `mnist-cnn-native`;
`mnist-cnn-device-execution` compiles the same model from OpenCL C through the
Device fragment backend and compares real GPU logits and predictions.

CI caches the verified, installed LLVM/MLIR SDK (`.deps/llvm23`) separately for
each OS image and architecture. The key includes the archive lock, installer and
extraction dependency. A cache miss downloads and extracts the locked archives;
CI never falls back to compiling LLVM. Verified SDKs are saved before project
configure/build/tests, so a later project failure does not discard them. Downloaded
archives are removed after installation to save runner disk space. Cache hits
validate required SDK components and the install receipt before configuring.

Existing SDK directories are never overwritten. A manually installed SDK such as
an older checkout can be checked with `python scripts/install_llvm.py --check`;
to install separately, use `--prefix path/to/new-sdk` and pass that path as
`VRC_LLVM_ROOT`. Repeating an installer-managed installation reuses its receipt.
For macOS, unsupported prebuilt hosts, or a source SDK with assertions, run
`python scripts/build_llvm.py` instead of the two installer commands.
That source build checks out the commit recorded in the lock; it is an explicit
local alternative and is not run by CI.

Current components:

- Four separate TableGen dialect libraries; 20 of the closed catalog of 57 operations are defined.
- A C OpenCL 1.2 AOT API subset and native Host Runtime linked to the fragment renderer.
- `vrc-opt`: MLIR parser/printer with operation verification, rejects unknown ops by default.
- `VRCLLVMToHighLevelMLIR`: independent import/metadata, local-memory and scalar/CFG recovery library for host/device development.
- `vrc-translate --import-llvm input.ll`: LLVM dialect import with preserved frontend metadata and optional JSON sidecar.
- `vrc-lower-udonhl-to-udonasm` and `vrc-translate --emit-udon-assembly`:
  an incremental scalar/CFG backend for real Udon opcodes and textual `.uasm`.
- `vrc-raise-llvm-memory` / `vrc-raise-llvm-to-scalar` MLIR Passes and the `vrc-recover-high-level` pipeline: proven fixed local arrays, scalar/CFG conversion and upstream SCF structuring.

The other 37 operations, complete verifiers/lowerings, the full executable Udon
host ABI, complete OpenCL runtime coverage and production Unity integration
remain pending. The scalar backend's integer, float and CFG golden programs have
passed the installed SDK assembler and actual `IUdonVM`; memory, calls,
continuations and GPU dispatch are still incremental work. A small paired
Host/Device lowering prototype emits ShaderLab/HLSL and a shared manifest.
See [docs/validation.md](docs/validation.md) for gates and evidence.

Import and high-level recovery examples (add `.exe` to tool names on Windows):

```text
build/udoncl/tools/vrc-translate/vrc-translate --import-llvm tests/recovery/loop.ll --metadata-output build/loop.frontend.json -o build/loop.llvm.mlir
build/udoncl/tools/vrc-opt/vrc-opt --pass-pipeline="builtin.module(vrc-recover-high-level)" --verify-each build/loop.llvm.mlir -o build/loop.high.mlir
```

The raising Pass is defined in TableGen and uses MLIR dialect conversion;
`--vrc-raise-llvm-to-scalar` runs just scalar/CFG conversion. The composed pipeline
first runs upstream mem2reg, then proven local-memory recovery, scalar raising,
then upstream CFG-to-SCF.
Unsupported memory, pointers, calling conventions, instructions and semantic
flags fail explicitly. General memory/alias, affine, OpenCL builtin, GPU local
memory and barrier/dependence recovery remain pending.
See [docs/recovery.md](docs/recovery.md) for scope, sidecar schema and P0 findings.

`ctest --preset udoncl` also runs the Clang OpenCL import matrix and native
recovery differential tests. It uses tools from the selected SDK, never system
Clang/opt from PATH. IR, sidecars, command logs and the OpenCL report are retained
in `build/udoncl/tests/opencl`; recovery artifacts are in `build/udoncl/tests/recovery`.

Try fixed-array recovery (initialization, update and sum):

```text
build/udoncl/tools/vrc-translate/vrc-translate --import-llvm tests/recovery/array-loop.ll -o build/array.llvm.mlir
build/udoncl/tools/vrc-opt/vrc-opt --pass-pipeline="builtin.module(vrc-recover-high-level)" --verify-each build/array.llvm.mlir -o build/array.high.mlir
```

`build/array.high.mlir` contains `memref.alloca`, `memref.load/store` and
`scf.while`. For just the memory conversion, use `--vrc-raise-llvm-memory`
instead of the pipeline. That Pass leaves scalar LLVM operations for the next
stage. Memory recovery currently requires fixed-size, homogeneous, nonescaping
entry-block allocations with proven bounds, alignment and initialization.
Unsupported pointer/alias, volatile/atomic and byte reinterpretation cases fail.
Memory test IR, native executables, diagnostics and reports are saved under
`build/udoncl/tests/memory`.

Host/Device boundary example (prepared high-level MLIR input):

```text
build/udoncl/tools/vrccl/vrccl --emit-boundary-bundle examples/host-device-boundary/probe.mlir -o build/boundary-example
build/udoncl/tools/vrccl/vrccl --check-boundary-request build/boundary-example/manifest.json --request examples/host-device-boundary/request.json
```

The Host path lowers `func.call` through `udonhl.call`, then substitutes the
internal typed dispatch intrinsic with `vrc.dispatch`. The Device path lowers a
single-invocation f32 store through `vrcgpu` to a fragment shader. Both sides must
agree on kernel ID, argument/resource order and NDRange before assets are emitted.
The bundle includes `host-device.mlir`, `manifest.json` and `kernels/*.shader` /
`*.hlsl`. Reusing identical output is allowed; differing existing output is refused.
This prepared boundary example does not yet execute its GPU request from Udon.
Scalar/CFG `.uasm` emission is tested separately. See
[docs/boundary.md](docs/boundary.md) for the supported pattern and next steps.

Original compiler-boundary probe (after installing the pinned LLVM SDK above):

```text
make example-c-boundary
```

The Makefile configures and builds the project, then runs
`examples/c-boundary/run.py`. It compiles `host.c` and `device.cl` with the Clang
from the configured LLVM SDK, imports the actual LLVM IR, runs recovery Passes,
and emits a paired boundary bundle. No hand-written MLIR is substituted.
Use `make test` to build and run all CTests; `make` alone builds the project.
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
`build/c-boundary/bundle/`. See [examples/c-boundary/README.md](examples/c-boundary/README.md)
for the source contract, intermediate IR and current limitations.

Native Host + Fragment Shader execution:

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
[examples/native-fragment/README.md](examples/native-fragment/README.md) for
bindings, intermediate artifacts, Windows commands and scope. Enable the same
CTest flow with `-DVRC_ENABLE_NATIVE_FRAGMENT_TESTS=ON`; CI enables it explicitly.

Ordinary OpenCL Host API example:

```text
make example
# equivalent: make example-opencl
```

`examples/opencl-host/host.c` uses the Khronos OpenCL 1.2 declarations to create a
context/queue/program/kernel/buffer, copy kernel arguments, enqueue an NDRange,
read the result and release objects. The generated Host computes `result + 1`
after readback. Native linking includes `VRCNativeOpenCLRuntime`: a C OpenCL API
implementation plus the binding to the fragment renderer. No system OpenCL
loader or vendor OpenCL driver is used. Linux requires the Vulkan/Lavapipe test
dependencies listed above; Windows uses WARP.

Outputs are under `build/opencl-host`; use `EXAMPLE_OUTPUT=build/another-opencl`
to choose a fresh directory. See [examples/opencl-host/README.md](examples/opencl-host/README.md)
for supported APIs, native ABI preservation and the AOT subset's limits.

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
[libraries/udonblas/README.md](libraries/udonblas/README.md) for the stable API
and the initial 16×16 profile limits.

触れて加熱する64粒子MD結晶のhost-native例:

```sh
make example-md  # CPU/GPU比較・保存則・局所加熱・氷→液体→再凍結を検証
make md-view     # ブラウザで触って操作
```

計算モデル、Host/Device経路、依存と操作は[MD結晶のREADME](examples/md-crystal/README.md)を参照してください。
