# Repository Agent Rules

These rules apply to the entire repository.

## Unity MCP

- Route every MCP for Unity operation through a subagent using the
  `gpt-6-luna` model with `xhigh` reasoning effort. The primary agent must not
  operate Unity MCP directly.
- When creating that subagent, use `fork_turns="none"` so the Luna/xhigh model
  override can be selected, and provide all required repository, Unity project,
  test, and cleanup context explicitly. Reuse an already-running Luna/xhigh
  Unity validation agent when practical.
- The primary agent owns implementation and repository edits. The Luna/xhigh
  agent owns Unity refresh/import, compilation checks, SDK IUdonVM execution,
  Unity Test Runner execution, console inspection, and lifecycle cleanup
  verification. It must not edit repository files unless the user explicitly
  changes this division of responsibility.
- The Luna/xhigh agent must read and follow the `unity-mcp-orchestrator` skill,
  use its resource-first workflow, wait for compilation/domain reload to
  finish, report exact test counts and failures, check console errors, and
  confirm that temporary test assets and directories were cleaned up.

## Pre-Commit Gates

- Before every commit, run the complete Unity Editor test gate through the
  local Python driver against a normally closed Unity 2022.3 project with the
  VRChat Worlds SDK and Unity for MCP installed:

  ```text
  python scripts/unity_ci.py --unity <Unity.exe> --project <UnityProject>
  ```

- The command must execute every test registered in the
  `Udoncl.Editor.Tests` assembly. Do not use `--testFilter`, run only a subset,
  accept skipped/inconclusive tests, or commit when any test fails. Confirm the
  reported test count, zero failures, clean Unity console, and lifecycle cleanup
  before committing.
- The driver must fail before generating fixtures or launching Unity when the
  project does not declare or embed `com.coplaydev.unity-mcp`. Do not bypass
  that check. The Python command above is the authoritative Unity test gate;
  do not substitute a GitHub Actions, GameCI, class-filtered, or method-filtered
  run for it.
- Unity execution for this gate must follow the Unity MCP subagent rules above.

- Immediately before every commit, run the repository formatter in write mode:

  ```text
  python scripts/format.py
  ```

- Then run the non-mutating formatter check and Git whitespace check:

  ```text
  python scripts/format.py --check
  git diff --cached --check
  ```

- Do not commit unless all three commands succeed. Stage the intended changes
  before `git diff --cached --check`.
- If any file is edited after the write-mode formatter runs, rerun the complete
  format/check sequence before committing.
- C, C++, and OpenCL C use the repository-pinned LLVM `clang-format`; C# uses
  the repository-pinned CSharpier tool. Do not bypass or substitute formatter
  versions locally.

## CUDA Library API Compatibility

- Keep public UdonBLAS and UdonDNN C APIs aligned with the corresponding
  cuBLAS and cuDNN APIs after the `cublas`/`cudnn` to `udonblas`/`udondnn`
  prefix substitution. Match function names, argument order and types, enum
  numeric values, status semantics, and matrix/tensor layout contracts.
- Pin and document the NVIDIA API version being mirrored. Treat the existing
  fixed-function UdonDNN surface as a cuDNN Legacy API compatibility layer;
  add Graph API support separately rather than silently changing that contract.
- Preserve the compatible public signature when Udon cannot implement an
  option yet. Declare the compatible value and return `NOT_SUPPORTED` instead
  of removing parameters, changing types, or imposing a different contract.
- Put Udon-specific operations such as quantized Q4 kernels in an explicitly
  named extension API; they must not redefine or overload a compatible entry
  point with non-cuBLAS/cuDNN semantics.
- Every public API change must include conformance tests against the pinned
  NVIDIA declarations, plus behavioral tests for supported paths and explicit
  `NOT_SUPPORTED` tests for unsupported paths.
