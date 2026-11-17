# Repository Agent Rules

These rules apply to the entire repository.

## Unity MCP

- Route every direct MCP for Unity operation through a subagent using the
  `gpt-6-luna` model with `xhigh` reasoning effort. The subagent must follow the
  `unity-mcp-orchestrator` skill and report readiness, tests, console, and
  cleanup results. The primary agent owns repository edits.
- The background commit validation service is the sole routine exception. The
  primary agent may start, inspect, and enqueue work for the service because
  each worker executes the fixed repository-owned Unity MCP workflow through
  `scripts/precommit.py` in an isolated Unity project.
- Do not replace the service gate with direct Unity MCP calls or ad-hoc Unity
  test commands. Additional interactive Unity diagnosis still requires the
  Luna/xhigh subagent.
- A pre-commit failure may be inspected without Unity MCP. Route any additional
  Unity MCP diagnosis or validation through the Luna/xhigh subagent.

## Commit Validation Service

- Every newly created commit is enqueued independently in the repository
  background validation service. A passing descendant does not change an
  ancestor's recorded result.
- Before creating a commit, run:

  ```text
  python scripts/precheck_service.py require-ready
  ```

  Do not commit if it fails. Do not bypass or disable the repository hooks.
- Stage only the intended change and commit normally. The `pre-commit` hook
  performs synchronous formatting and whitespace checks. The `post-commit`
  hook durably enqueues the exact new commit SHA and returns without waiting for
  the native or Unity gate.
- After committing, confirm that the exact SHA was enqueued:

  ```text
  python scripts/precheck_status.py --commit HEAD --require-enqueued
  ```

  Continue working while pending validation runs in the background.
- Before another commit and before reporting completed work, run:

  ```text
  python scripts/precheck_status.py --outgoing --failures --unreported
  ```

  This reports only push-eligible commits: outgoing descendants strictly after
  the tracked `.precheck-baseline` bootstrap anchor. The anchor is already
  accepted, and older database failures do not block completion or pushing.
  Successful or pending jobs require no user-facing report. On an eligible
  failure, immediately report the commit and concise failure, then amend/rebase
  it or remove it from eligible history.
- Before pushing, wait for every push-eligible commit to pass:

  ```text
  python scripts/precheck_status.py --outgoing --wait --require-passed
  ```

  The `pre-push` hook enforces the same exclusive bootstrap anchor. Every
  pushed tip must descend from the full SHA in `.precheck-baseline`; a tip equal
  to the anchor has no eligible commits, and unrelated histories fail
  explicitly. Never push an eligible pending, failed, stale, or unvalidated
  commit. Change the anchor only as an explicit reviewed policy update after
  its commit has passed the full validation gate.
- Results are authoritative only in the service SQLite database and must match
  the exact commit SHA and validation fingerprint. Never edit the database or
  synthesize a pass. Amend, rebase, cherry-pick, and merge create new SHAs that
  require independent validation.
- `scripts/precommit.py` is the internal worker gate. Do not run it
  synchronously for routine commits. Manual use is limited to focused diagnosis
  of a failed background job.

## Validation Service Safety

- The lightweight coordinator must remain running while commits are created;
  heavy executors run only while jobs are queued. Use the repository service
  commands on Windows and Linux and do not launch worker commands manually.
- Never share a mutable Git worktree, build directory, Unity project, Unity
  Library, MCP status directory, or output directory between concurrent jobs.
- Treat `UDONCL_UNITY_TEST_PROJECT` as a read-only template. Workers open only
  their isolated copies. Preserve failure logs and artifacts. A crashed or
  cleanup-failed worker slot is contaminated and must be rebuilt before reuse.

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
- Do not invent public `Ext` APIs for operations outside cuBLAS/cuDNN. First
  identify the owning PyTorch/TorchAO CUDA component and follow that interface;
  keep any truly Udon-only mechanism private until it has a separate documented
  contract.
- Every public API change must include conformance tests against the pinned
  NVIDIA declarations, plus behavioral tests for supported paths and explicit
  `NOT_SUPPORTED` tests for unsupported paths.

## Operator Ownership

- Before adding an API for an ONNX/PyTorch operator, inspect official
  PyTorch/TorchAO sources, trace its CUDA dispatch, and identify whether the
  implementation is owned by cuBLAS, cuDNN, ATen, or TorchAO. If PyTorch calls
  a CUDA library, mirror that library's interface; otherwise mirror the owning
  ATen/TorchAO operator contract. Do not place an operation in UdonBLAS/UdonDNN
  merely because it runs on the GPU.
- cuBLAS/cuDNN counterparts belong in UdonBLAS/UdonDNN. ATen-native operators
  belong in UdonATen. TorchAO/ATen quantized operators follow their packed
  weight, group-size, scale, and zero-point contracts. Record the source and
  ownership decision in `docs/architecture/operator-ownership.md`.
- Do not introduce `udondnnExt*` or `udonblasExt*` as a shortcut around this
  ownership check. Existing provisional extensions are backend details to
  migrate, not precedents for new public APIs.
