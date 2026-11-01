# Local Unity Editor test gate

Unity Editor tests are run locally through `scripts/unity_ci.py`. The
repository does not run Unity through GitHub Actions or GameCI.

Use a normally closed Unity 2022.3 project containing both the VRChat Worlds
SDK and Unity for MCP. Build the compiler tools first, then run:

```text
python scripts/unity_ci.py --unity <Unity.exe> --project <UnityProject>
```

For example on Windows with Unity 2022.3.22f1:

```text
python scripts/unity_ci.py --unity "C:\Program Files\Unity\Hub\Editor\2022.3.22f1\Editor\Unity.exe" --project "C:\path\to\test-project"
```

The driver fails before generating fixtures or launching Unity unless the test
project declares or embeds `com.coplaydev.unity-mcp`. After Unity starts, it
also verifies that Unity for MCP was resolved into `Packages/packages-lock.json`.

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

The command writes evidence under `build/unity-ci/`, including the NUnit XML,
Unity log, fresh UdonAsm MLIR, and emitted `scalar.uasm`.
