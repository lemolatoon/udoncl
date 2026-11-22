# Developing UdonCL

This guide is for people who change UdonCL itself: the compiler, runtime
libraries, Unity package, examples, or tests. The [README](../README.md) covers
the minimal compiler build for users; this file collects the full developer
setup, dependency layout, toolchain details, and the commit validation service.
Repository rules for agents and contributors are in [AGENTS.md](../AGENTS.md).

## Python requirements

Install every pinned Python requirement set before configuring:

```text
python -m pip install -r scripts/requirements-llvm.txt
python -m pip install -r scripts/requirements-onnx.txt
python -m pip install -r scripts/requirements-format.txt
python -m pip install -r scripts/requirements-test.txt
```

## Full toolchain and fixture setup

Install the pinned toolchains and model fixtures before configuring UdonCL:

```text
python scripts/install_llvm.py
python scripts/install_llvm.py --check
python scripts/install_onnx_mlir.py --jobs 2
python scripts/install_onnx_mlir.py --check
python scripts/fetch_tinyllama_onnx.py
python scripts/fetch_tinyllama_tokenizer.py
cmake --preset udoncl
python scripts/build.py
ctest --preset udoncl
```

Once installed, subsequent builds only need configure, build, and test.
`make` alone configures and builds the project; `make test` builds and runs all
CTests.

### TinyLlama model and tokenizer fixtures

`python scripts/fetch_tinyllama_onnx.py` downloads the pinned TinyLlama ONNX
model fixture. The TinyLlama tokenizer command
(`python scripts/fetch_tinyllama_tokenizer.py`) verifies and atomically caches
the pinned `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, and
`special_tokens_map.json` in `.deps/models`. To inspect the generated C tables
directly:

```text
python scripts/compile_tokenizer_tables.py <printed-tokenizer-directory> --output build/tinyllama-tokenizer-tables
```

Projects normally generate and link those tables through the public
`udontokenizers_add_huggingface_model` CMake API described in the README.

## Dependency and build directory layout

The repository keeps installed dependencies separate from build products:

- `.deps/llvm23` is the verified LLVM/MLIR/Clang SDK.
- `.deps/onnx-mlir` is the single supported installed ONNX-MLIR SDK prefix.
  It contains headers and libraries for protobuf import, the ONNX dialect, and
  the required upstream ONNX transforms. It contains no UdonCL-specific pass or
  compiler tool. The repository-built internal ONNX frontend owns import,
  shape specialization, upstream transforms, and every UdonCL semantic pass.
- `.deps/onnx-mlir-src` is a disposable checkout managed by the installer.
- `.deps/models` contains checksum-addressed downloaded model fixtures.
- `build/onnx-mlir` contains ONNX-MLIR intermediate build products.
- `build/udoncl` contains UdonCL build and test products.

## ONNX-MLIR SDK prefix

Do not combine tools from older prefixes. In particular,
`.deps/onnx-mlir-complete` is not a supported discovery location. The ONNX-MLIR
installer validates an immutable receipt and does not overwrite an existing
prefix. Rebuild this dependency only when the pinned ONNX-MLIR revision,
upstream compatibility patches, or SDK packaging schema changes. Changes to
repo-owned UdonCL passes require only rebuilding UdonCL. To replace an obsolete
tool-only prefix with the dialect SDK prefix, remove only that installed prefix:

```powershell
Remove-Item -LiteralPath .\.deps\onnx-mlir -Recurse -Force
python scripts/install_onnx_mlir.py --jobs 2
python scripts/install_onnx_mlir.py --check
cmake --fresh --preset udoncl
```

The installer resets its disposable source checkout to the pinned ONNX-MLIR
commit, reapplies the complete locked patch stack, and publishes the three
verified executables together. `--jobs 2` is recommended on developer machines
to bound peak memory use. Use `cmake --fresh --preset udoncl` after replacing a
toolchain so an older CMake program path cannot survive in the cache.

## Prebuilt LLVM SDK

LLVM, MLIR and Clang are pinned together to **23.1.4** (commit recorded in
[cmake/llvm-lock.json](../cmake/llvm-lock.json)).

The prebuilt installer verifies pinned SHA256 checksums and installs into
`.deps/llvm23`; it never builds LLVM. Linux uses the official LLVM SDK archive.
Windows uses the community [Portable MLIR Toolchain](https://github.com/munich-quantum-software/portable-mlir-toolchain/releases/tag/2026.10.28)
Release/noassert SDK and supplements it with only `clang.exe` and Clang resource
files from the official LLVM archive. All LLVM/MLIR headers, libraries and CMake
packages come from the same SDK; official LLVM link libraries are not overlaid.
Download URLs and checksums are recorded in `cmake/llvm-lock.json`.

On Ubuntu, install the SDK's system libraries (the `apt-get` command in the
README prerequisites) before configuring.

### Manual SDKs, separate prefixes, and source builds

Existing SDK directories are never overwritten. A manually installed SDK such as
an older checkout can be checked with `python scripts/install_llvm.py --check`;
to install separately, use `--prefix path/to/new-sdk` and pass that path as
`VRC_LLVM_ROOT`. Repeating an installer-managed installation reuses its receipt.
For macOS, unsupported prebuilt hosts, or a source SDK with assertions, run
`python scripts/build_llvm.py` instead of the two installer commands.
That source build checks out the commit recorded in the lock; it is an explicit
local alternative and is not run by CI.

### CI LLVM SDK cache

CI caches the verified, installed LLVM/MLIR SDK (`.deps/llvm23`) separately for
each OS image and architecture. The key includes the archive lock, installer and
extraction dependency. A cache miss downloads and extracts the locked archives;
CI never falls back to compiling LLVM. Verified SDKs are saved before project
configure/build/tests, so a later project failure does not discard them. Downloaded
archives are removed after installation to save runner disk space. Cache hits
validate required SDK components and the install receipt before configuring.

## LLVM discovery and configuring without presets

CMake automatically prefers the repository-local SDK, including when
configuring without presets:

```text
cmake -S . -B build/udoncl -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/udoncl
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
No system-wide LLVM installation is required. On Windows, use the
DLL MSVC runtime (`/MD`) for both LLVM and this project. Paths with spaces are
supported. Windows native execution is covered by CI and the local Visual
Studio 2026 validation recorded in [validation.md](validation.md).

## Editor setup (clangd)

CMake emits `build/udoncl/compile_commands.json` for clangd. The checked-in
`.clangd` points to that directory on all hosts. Run configure and build once
so TableGen-generated headers also exist. With a custom build directory, set
clangd's `--compile-commands-dir` to that directory. If clangd selects a different
system C++ installation than the build compiler, allow querying the compiler
with `--query-driver=/absolute/path/to/compiler` in your editor's clangd arguments.
For example, this Linux host needs `--query-driver=/usr/bin/c++` to discover GCC
15 headers instead of an incomplete GCC 16 installation. On Windows, launch the
editor from the Developer PowerShell so clangd inherits the MSVC include paths.

## Native fragment execution and build parallelism

The standard preset builds and registers native fragment execution by default.
On Linux this requires Vulkan, Mesa's software renderer and glslang as listed
in the README prerequisites; Windows uses D3D11. `scripts/build.py` remains
available for automation that needs an explicit process-aware parallel job
count. The CMake build preset has no fixed job limit, so direct Ninja builds use
its native parallelism.

## Test artifacts

`ctest --preset udoncl` also runs the Clang OpenCL import matrix and native
recovery differential tests. It uses tools from the selected SDK, never system
Clang/opt from PATH. IR, sidecars, command logs and the OpenCL report are retained
in `build/udoncl/tests/opencl`; recovery artifacts are in `build/udoncl/tests/recovery`.
Memory test IR, native executables, diagnostics and reports are saved under
`build/udoncl/tests/memory`.

## Commit validation service

Commit validation runs in the background against every exact commit SHA. The
binding procedure for creating, checking, and pushing commits is in
[AGENTS.md](../AGENTS.md); this section explains the setup. The Unity Editor
gate itself is described in [validation/unity-ci.md](validation/unity-ci.md).

First create a dedicated Unity 2022.3 template project containing the VRChat
Worlds SDK and Unity for MCP. It must be a real Unity project with
`Packages/manifest.json`; the validation service treats it as read-only and
opens only isolated worker copies. Record the editor and template paths in the
repository-local `.env`:

```dotenv
UNITY_EDITOR_PATH=C:\Program Files\Unity\Hub\Editor\2022.3.22f1\Editor\Unity.exe
UDONCL_UNITY_TEST_PROJECT=C:\path\to\dedicated\udoncl-test
```

Install the tracked Git hooks and per-user coordinator once. The coordinator is
registered through Task Scheduler on Windows and a systemd user service on
Linux. If Task Scheduler registration is denied, Windows automatically installs
a hidden per-user Startup supervisor instead. `--no-system-service` keeps the
manual background process only:

```text
python scripts/precheck_service.py install --jobs 2
python scripts/precheck_service.py status
```

If a worker has no LLVM or ONNX-MLIR dependency prefix, it runs the locked
repository installers before validation. Each worker slot has its own
persistent Git worktree, Unity project and build/output directories, which are
reset for every commit, so multiple commits can be tested concurrently without
sharing mutable build or Unity state. On Linux the Unity gate uses Vulkan;
Windows uses D3D11.

The pre-commit hook requires a healthy coordinator and performs only the fast
format and whitespace checks. The post-commit hook writes the new SHA to the
SQLite queue and returns immediately:

```text
git add <paths>
git commit -m "describe the change"
python scripts/precheck_status.py --commit HEAD --require-enqueued
```

Idle workers claim the most recently enqueued commit first, so the commit a
developer just created is validated ahead of an older backlog; equal enqueue
times are ordered by descending commit SHA. `retry` counts as a new enqueue and
moves the commit to the front, while jobs requeued after a worker or
coordinator crash keep their original position.

Successful background jobs are silent. Failed jobs produce a desktop
notification and remain visible in the database:

```text
python scripts/precheck_status.py --failures --unreported
python scripts/precheck_service.py retry HEAD
```

Every outgoing SHA must pass independently. A later successful commit does not
cover a failed ancestor. Before pushing, wait for the complete outgoing range;
the pre-push hook rejects missing, pending and failed results too:

```text
python scripts/precheck_status.py --outgoing --wait --require-passed
git push
```

Service administration is cross-platform:

```text
python scripts/precheck_service.py start --jobs 2
python scripts/precheck_service.py set-workers 3
python scripts/precheck_service.py stop
python scripts/precheck_service.py status
python scripts/precheck_service.py uninstall
```

`--jobs` (default `UDONCL_PRECHECK_JOBS`, else 2) is the number of concurrent
validation workers the coordinator starts with; build parallelism inside a job
is chosen by `scripts/precommit.py`. `set-workers N` stores a worker count in
the queue database that overrides `--jobs`, survives restarts, and is applied
by a running coordinator within a second. Lowering it never interrupts running
jobs: the coordinator stops claiming until fewer than `N` workers remain.
`set-workers 0` pauses claiming, which lets running jobs drain before a
`stop`. `status` reports the effective count as `workers=N` with
`workers_source=set-workers` or `startup`. Raising the count adds worker slots
that are isolated exactly like the existing ones.

`UDONCL_PRECHECK_CACHE_ROOT` overrides the per-user state directory. Queue
state and structured results live in SQLite; large Unity/MCP/native logs remain
content-hashed files referenced by the database. `scripts/precommit.py` remains
the authoritative full native and Unity gate used internally by commit-snapshot
workers; do not run it synchronously for routine commits.

## Formatting

Formatter versions are pinned in `scripts/requirements-format.txt` and
`.config/dotnet-tools.json`; a different `clang-format` version is rejected.
