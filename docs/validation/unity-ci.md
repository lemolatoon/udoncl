# Local Unity Editor test gate

All commit checks are run locally through `scripts/precommit.py`. The repository
does not run Unity through GitHub Actions or GameCI.

Use a normally closed Unity 2022.3 project containing both the VRChat Worlds
SDK and Unity for MCP. Put the local paths in the repository `.env`:

```text
UNITY_EDITOR_PATH=C:\Program Files\Unity\Hub\Editor\2022.3.22f1\Editor\Unity.exe
UDONCL_UNITY_TEST_PROJECT=C:\path\to\test-project
```

The process environment overrides `.env`, and explicit CLI arguments override
both. Build the compiler tools first, then run:

```text
git add <intended-files>
python scripts/precommit.py
```

For example on Windows with Unity 2022.3.22f1:

```text
git add <intended-files>
python scripts/precommit.py --unity "C:\Program Files\Unity\Hub\Editor\2022.3.22f1\Editor\Unity.exe" --project "C:\path\to\test-project"
```

The driver fails before generating fixtures or launching Unity unless the test
project declares or embeds `com.coplaydev.unity-mcp`. After Unity starts, it
also verifies that Unity for MCP was resolved into `Packages/packages-lock.json`.
It starts an isolated batchmode Editor and a pinned MCP server, reads the Editor
state resource until compilation and domain reload are complete, and then uses
MCP tools for the test and console checks.

The driver generates fresh lowering output and UdonBLAS, UdonDNN, MNIST, and
Tiny Llama artifacts. It runs the complete `Udoncl.Editor.Tests` EditMode
assembly without a class or method filter. Failed, skipped, or inconclusive
tests fail the command. Newly registered tests therefore join the gate
automatically.

The result verifier separately requires all six current SDK `IUdonVM` lowering
cases: two integer cases, one floating-point case, two control-flow cases, and
one float-array load/store case. It also rejects a leaked lowering setup
directory or any path or content change under `Assets/udoncl-generated` and
`Assets/SerializedUdonPrograms` relative to the pre-test baseline.

The command writes evidence under `build/precommit/`, including the MCP test
result JSON, console JSON, Unity/MCP logs, fresh UdonAsm MLIR, and `scalar.uasm`.
It then runs the repository formatter, re-stages only paths that were staged at
startup, and checks formatting and staged Git whitespace. A successful complete
run ends with `precommit=passed`.
