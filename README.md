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
cmake --preset llvm23
cmake --build --preset llvm23
ctest --preset llvm23
```

The dependency build installs into `.deps/llvm23`. Once installed, subsequent
builds only need the last three commands above. CMake automatically prefers this
repository-local SDK, including when configuring without presets:

```text
cmake -S . -B build/llvm23 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/llvm23 --parallel 2
ctest --test-dir build/llvm23 --output-on-failure
```

Discovery order is an explicit `VRC_LLVM_ROOT`, then `.deps/llvm23`, then system
LLVM (only if the local directory does not exist). LLVM and MLIR are selected
from the same prefix. A local SDK overrides stale `LLVM_DIR` / `MLIR_DIR` cache
entries; an incomplete local SDK fails with a diagnostic instead of falling
back to system packages. Configure output prints both selected package paths.

To use another complete 23.1.4 installation explicitly:

```text
cmake --preset llvm23 -DVRC_LLVM_ROOT="path/to/llvm"
cmake --build --preset llvm23
ctest --preset llvm23
```

To return to automatic repository-local discovery, configure with
`-DVRC_LLVM_ROOT:PATH=`. Use `cmake --fresh --preset llvm23` to clear all cached
settings when changing compilers or toolchains (then supply any desired root).
`CMAKE_PREFIX_PATH` alone does not override an available repository-local SDK.
LLVM and MLIR must come from the same build, architecture and C++ runtime.
No system-wide installation is required by the build script. On Windows, use the
DLL MSVC runtime (`/MD`) for both LLVM and this project. Paths with spaces are
supported. Windows execution remains unverified until Windows CI logs exist.

CMake emits `build/llvm23/compile_commands.json` for clangd. The checked-in
`.clangd` points to that directory on all hosts. Run configure and build once
so TableGen-generated headers also exist. With a custom build directory, set
clangd's `--compile-commands-dir` to that directory. If clangd selects a different
system C++ installation than the build compiler, allow querying the compiler
with `--query-driver=/absolute/path/to/compiler` in your editor's clangd arguments.
For example, this Linux host needs `--query-driver=/usr/bin/c++` to discover GCC
15 headers instead of an incomplete GCC 16 installation. On Windows, launch the
editor from the Developer PowerShell so clangd inherits the MSVC include paths.

Current components:

- Four separate TableGen dialect libraries; only the nine `udonasm` operations are registered.
- `vrc-opt`: MLIR parser/printer with operation verification, rejects unknown ops by default.
- `VRCLLVMToHighLevelMLIR`: independent LLVM import library shared by future host/device recovery.
- `vrc-translate --import-llvm input.ll`: LLVM dialect import only; recovery and metadata sidecar are pending.

The other 48 operations, full verifiers/lowerings, assembly emitter, host ABI,
OpenCL runtime, generated shaders and Unity integration are not implemented.
See [docs/validation.md](docs/validation.md) for gates and evidence.
