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

## Formatting Before Commits

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
