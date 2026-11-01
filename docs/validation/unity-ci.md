# Unity Editor test CI

The `Unity Editor tests` workflow is the authoritative CI gate for every test in
the package's `Udoncl.Editor.Tests` assembly. `scripts/unity_ci.py` generates
fresh UdonBLAS, UdonDNN, MNIST, and Tiny Llama Unity bundles, then runs all
registered EditMode tests without a class filter. The gate rejects failed,
skipped, or inconclusive tests and requires the XML's total and passed counts
to match, so newly registered tests automatically join the gate.

The lowering fixture does not execute a permanently checked-in assembly as
its source of truth. `scripts/unity_ci.py` first runs the just-built
`vrc-opt --vrc-lower-udonhl-to-udonasm` and `vrc-translate
--emit-udon-assembly` pipeline, then passes that fresh `scalar.uasm` to the
Unity fixture through `UDONCL_LOWERED_UASM`.

That EditMode fixture assembles the output with the installed VRChat SDK,
loads it into SDK `IUdonVM`, and verifies integer arithmetic, a loop, float
arithmetic, control flow, and float-array load/store. Its one-time setup stages
the assembly under the Unity project's `Library` directory; teardown removes
the directory and asserts that cleanup succeeded.

The job uses a dedicated Windows self-hosted runner labelled
`unity-2022.3` and `vrchat-worlds`. Configure these repository variables:

- `UNITY_EDITOR_PATH`: absolute path to the Unity 2022.3 editor executable on
  the runner, for example
  `C:\Program Files\Unity\Hub\Editor\2022.3.22f1\Editor\Unity.exe`.
- `UDONCL_UNITY_TEST_PROJECT`: absolute path on the runner to a dedicated,
  normally closed Unity project with the VRChat Worlds SDK installed, for
  example `C:\actions\udoncl-test`.

Set them under **Settings > Secrets and variables > Actions > Variables**, or
with GitHub CLI:

```text
gh variable set UNITY_EDITOR_PATH --body "C:\Program Files\Unity\Hub\Editor\2022.3.22f1\Editor\Unity.exe"
gh variable set UDONCL_UNITY_TEST_PROJECT --body "C:\actions\udoncl-test"
```

Register a Windows x64 self-hosted runner under **Settings > Actions >
Runners**, and add the custom labels `unity-2022.3` and `vrchat-worlds` during
`config.cmd` setup. The runner account must be able to launch Unity, access the
test project, Visual Studio C++ tools, CMake, Ninja, and a valid Unity license.
Run the agent in an interactive Windows session with D3D11 available; the
Editor lifecycle tests create and read RenderTextures, so `-nographics` and a
Windows service session without a graphics device are unsupported. The driver
selects D3D11 explicitly with `-force-d3d11`.
No Unity-related GitHub Actions secrets are required by this self-hosted setup.
The lightweight `configuration` job runs on GitHub-hosted Ubuntu and reports
missing variables as explicit errors instead of silently skipping the Unity
job. A job that remains queued after that check usually means no online runner
matches all five labels: `self-hosted`, `Windows`, `X64`, `unity-2022.3`, and
`vrchat-worlds`.

The runner temporarily points that project's package manifest at the checkout's
UPM package and restores both package manifest and lock file after Unity exits.
Fork pull requests never execute on the self-hosted runner. The complete Editor
test XML, Unity log, fresh UdonAsm MLIR, and emitted assembly are uploaded for
every run. Every lifecycle fixture must remove its generated assets and
serialized Udon program; the CI driver also rejects leaked lowering or
any path or content change under `Assets/udoncl-generated` and
`Assets/SerializedUdonPrograms` relative to the pre-test baseline.

For MCP-for-Unity validation in a repository checkout, the same fixture first
uses `build/udoncl/tests/udon-scalar/scalar.uasm` when present and only falls
back to the checked-in package fixture for an installed-package consumer. This
keeps local MCP and batchmode CI on the same SDK `IUdonVM` assertions.
