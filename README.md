# udoncl

C / OpenCL C → Udon / Fragment Shader compiler under development.
[PLAN.md](PLAN.md) is the design specification. The application entry is UdonSharp;
C is a library. This repository does not yet compile C into a working VRChat world.

LLVM, MLIR and Clang are pinned together to **23.1.4** (commit recorded in
[cmake/llvm-lock.json](cmake/llvm-lock.json)). Compiler development targets Linux,
macOS and Windows x64; the initial generated world targets VRChat PC (Windows).

Prerequisites: CMake 3.30+, Ninja, Python 3, Git, and a C++17 compiler.
For Windows, install Visual Studio 2022 with Desktop development with C++,
and run these commands from **x64 Native Tools / Developer PowerShell**.

```text
python scripts/build_llvm.py --jobs 2
cmake --preset udoncl
cmake --build --preset udoncl
ctest --preset udoncl
```

The dependency build installs into `.deps/llvm23`. Once installed, subsequent
builds only need the last three commands above. CMake automatically prefers this
repository-local SDK, including when configuring without presets:

```text
cmake -S . -B build/udoncl -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/udoncl --parallel 2
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
cmake --build --preset udoncl
ctest --preset udoncl
```

To return to automatic repository-local discovery, configure with
`-DVRC_LLVM_ROOT:PATH=`. Use `cmake --fresh --preset udoncl` to clear all cached
settings when changing compilers or toolchains (then supply any desired root).
`CMAKE_PREFIX_PATH` alone does not override an available repository-local SDK.
LLVM and MLIR must come from the same build, architecture and C++ runtime.
No system-wide installation is required by the build script. On Windows, use the
DLL MSVC runtime (`/MD`) for both LLVM and this project. Paths with spaces are
supported. Windows execution remains unverified until Windows CI logs exist.

CMake emits `build/udoncl/compile_commands.json` for clangd. The checked-in
`.clangd` points to that directory on all hosts. Run configure and build once
so TableGen-generated headers also exist. With a custom build directory, set
clangd's `--compile-commands-dir` to that directory. If clangd selects a different
system C++ installation than the build compiler, allow querying the compiler
with `--query-driver=/absolute/path/to/compiler` in your editor's clangd arguments.
For example, this Linux host needs `--query-driver=/usr/bin/c++` to discover GCC
15 headers instead of an incomplete GCC 16 installation. On Windows, launch the
editor from the Developer PowerShell so clangd inherits the MSVC include paths.

CI caches the installed LLVM/MLIR SDK (`.deps/llvm23`) separately for each OS
image and architecture. The cache key includes `cmake/llvm-lock.json` and
`scripts/build_llvm.py`; ordinary source changes reuse it and skip the LLVM
build. A cache miss builds the SDK once and saves it immediately, before this
project's configure/build/tests, so a later test failure does not discard it.
The first run, a dependency/key change, or cache eviction requires rebuilding.
The LLVM source tree and temporary build tree are not cached.

Current components:

- Four separate TableGen dialect libraries; only the nine `udonasm` operations are registered.
- `vrc-opt`: MLIR parser/printer with operation verification, rejects unknown ops by default.
- `VRCLLVMToHighLevelMLIR`: independent import/metadata and scalar/CFG recovery library for host/device development.
- `vrc-translate --import-llvm input.ll`: LLVM dialect import with preserved frontend metadata and optional JSON sidecar.
- `vrc-raise-llvm-to-scalar` MLIR Pass and `vrc-recover-high-level` pipeline: scalar/CFG conversion and upstream memory promotion / SCF structuring.

The other 48 operations, full verifiers/lowerings, assembly emitter, host ABI,
OpenCL runtime, generated shaders and Unity integration are not implemented.
See [docs/validation.md](docs/validation.md) for gates and evidence.

Import and high-level recovery examples (add `.exe` to tool names on Windows):

```text
build/udoncl/tools/vrc-translate/vrc-translate --import-llvm tests/recovery/loop.ll --metadata-output build/loop.frontend.json -o build/loop.llvm.mlir
build/udoncl/tools/vrc-opt/vrc-opt --pass-pipeline="builtin.module(vrc-recover-high-level)" --verify-each build/loop.llvm.mlir -o build/loop.high.mlir
```

The raising Pass is defined in TableGen and uses MLIR dialect conversion;
`--vrc-raise-llvm-to-scalar` runs just scalar/CFG conversion. The composed pipeline
first runs upstream mem2reg, then scalar raising, then upstream CFG-to-SCF.
Unsupported memory, pointers, calling conventions, instructions and semantic
flags fail explicitly. This is the first recovery stage; general memref, affine,
OpenCL builtin, local-memory and barrier/dependence recovery remain pending.
See [docs/recovery.md](docs/recovery.md) for scope, sidecar schema and P0 findings.

`ctest --preset udoncl` also runs the Clang OpenCL import matrix and native
recovery differential tests. It uses tools from the selected SDK, never system
Clang/opt from PATH. IR, sidecars, command logs and the OpenCL report are retained
in `build/udoncl/tests/opencl`; recovery artifacts are in `build/udoncl/tests/recovery`.
