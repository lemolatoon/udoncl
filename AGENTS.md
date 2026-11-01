# Repository Agent Rules

These rules apply to the entire repository.

## Unity MCP

- Route every direct MCP for Unity operation through a subagent using the
  `gpt-6-luna` model with `xhigh` reasoning effort. The subagent must follow the
  `unity-mcp-orchestrator` skill and report readiness, tests, console, and
  cleanup results. The primary agent owns repository edits.
- The sole routine exception is `scripts/precommit.py`: the primary agent is
  explicitly allowed to run this Python command directly because the script
  performs the fixed Unity MCP validation workflow internally. Do not replace
  it with direct Unity MCP calls or separate gate commands.
- A pre-commit failure may be inspected without Unity MCP. Route any additional
  Unity MCP diagnosis or validation through the Luna/xhigh subagent.

## Pre-Commit Gates

- Stage the intended change, then run exactly one gate command against a closed
  Unity 2022.3 project with the VRChat Worlds SDK and Unity for MCP installed:

  ```text
  python scripts/precommit.py
  ```

- Set `UNITY_EDITOR_PATH` and `UDONCL_UNITY_TEST_PROJECT` in the repository
  `.env`. Explicit `--unity` and `--project` arguments override these defaults.

- The command is authoritative. It generates fixtures, uses Python to run the
  whole `Udoncl.Editor.Tests` assembly through Unity MCP, rejects skipped tests,
  console errors, or cleanup leaks, then formats and re-stages only the paths
  that were already staged and runs format and Git whitespace checks.
- Do not commit unless it prints `precommit=passed`. If files change afterward,
  stage the intended change and rerun the same command.

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

## Operator Ownership

- Before adding an API for an ONNX/PyTorch operator, identify its actual
  PyTorch CUDA owner in official PyTorch/TorchAO sources. Match that component's
  operator name, argument order, and semantics; do not place an operation in
  UdonBLAS/UdonDNN merely because its implementation runs on the GPU.
- cuBLAS/cuDNN counterparts belong in UdonBLAS/UdonDNN. ATen-native operators
  belong in UdonATen. TorchAO/ATen quantized operators follow their packed
  weight, group-size, scale, and zero-point contracts. Record the source and
  ownership decision in `docs/architecture/operator-ownership.md`.
- Do not introduce `udondnnExt*` or `udonblasExt*` as a shortcut around this
  ownership check. Existing provisional extensions are backend details to
  migrate, not precedents for new public APIs.
