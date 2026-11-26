# Dependencies and redistribution

- LLVM / MLIR / Clang: 23.1.4, Apache-2.0 WITH LLVM-exception.
  https://github.com/llvm/llvm-project/blob/llvmorg-23.1.4/LICENSE.TXT
  Source/commit are pinned in cmake/llvm-lock.json. The local `.deps` directory
  is ignored by Git. Retain upstream notices when redistributing binaries.
- LLVM FileCheck driver: `third_party/llvm-project` vendors
  `llvm/utils/FileCheck/FileCheck.cpp` and `LICENSE.TXT` from the commit in
  cmake/llvm-lock.json (file digests in its `SOURCE.json`). The official Linux
  release archive ships the FileCheck library but not the tool, so the test
  build compiles this driver when the SDK exports no `FileCheck` target.
- CMake, Ninja, Python and Git are build tools installed separately.
- Unity, VRChat Worlds SDK, UdonSharp and Picolibc are not vendored here.
  SDK/package versions and their license terms must be recorded when P0 imports
  actual dependencies. No verified SDK version is currently asserted.
- CI uses actions/checkout, actions/setup-python, actions/cache and
  ilammy/msvc-dev-cmd. Review their upstream licenses when maintaining CI.

The repository's own distribution license has not been selected. This file does
not relicense PLAN.md or claim rights to redistribute the Unity/VRChat SDK.
