# Unity SDK UdonVM CI

The `Unity SDK UdonVM` workflow is the authoritative CI gate for UdonAsm
lowering behavior. It does not execute a permanently checked-in assembly as
its source of truth. `scripts/unity_ci.py` first runs the just-built
`vrc-opt --vrc-lower-udonhl-to-udonasm` and `vrc-translate
--emit-udon-assembly` pipeline, then passes that fresh `scalar.uasm` to the
Unity fixture through `UDONCL_LOWERED_UASM`.

The EditMode fixture assembles the output with the installed VRChat SDK,
loads it into SDK `IUdonVM`, and verifies integer arithmetic, a loop, float
arithmetic, control flow, and float-array load/store. Its one-time setup stages
the assembly under the Unity project's `Library` directory; teardown removes
the directory and asserts that cleanup succeeded.

The job uses a dedicated Windows self-hosted runner labelled
`unity-2022.3` and `vrchat-worlds`. Configure these repository variables:

- `UNITY_EDITOR_PATH`: absolute path to the Unity 2022.3 editor executable.
- `UDONCL_UNITY_TEST_PROJECT`: absolute path to a dedicated Unity project with
  the VRChat Worlds SDK installed.

The runner temporarily points that project's package manifest at the checkout's
UPM package and restores both package manifest and lock file after Unity exits.
Fork pull requests never execute on the self-hosted runner. Test XML, Unity log,
fresh UdonAsm MLIR, and emitted assembly are uploaded for every run.

For MCP-for-Unity validation in a repository checkout, the same fixture first
uses `build/udoncl/tests/udon-scalar/scalar.uasm` when present and only falls
back to the checked-in package fixture for an installed-package consumer. This
keeps local MCP and batchmode CI on the same SDK `IUdonVM` assertions.
