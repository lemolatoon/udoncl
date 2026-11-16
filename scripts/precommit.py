"""Run the complete repository pre-commit gate with one command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from msvc_environment import ensure_msvc_environment

from unity_ci import (
    LOWERING_TEST_CASES,
    TEST_ASSEMBLY,
    prepare_cmake_unity_library,
    prepare_fixtures,
    require_resolved_unity_mcp,
    require_unity_mcp,
    tree_snapshot,
)


MCP_REVISION = "81542a7e86292b5c4c21b1782c5af23b567f45ce"
MCP_SOURCE = (
    "git+https://github.com/CoplayDev/unity-mcp.git@"
    f"{MCP_REVISION}#subdirectory=Server"
)


def load_dotenv_defaults(path: Path):
    """Load simple KEY=VALUE entries without overriding the process environment."""
    if not path.is_file():
        return
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise RuntimeError(f"invalid .env entry at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise RuntimeError(f"empty .env key at {path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def ensure_cmake_toolchain_on_path():
    if shutil.which("cmake") and shutil.which("ninja"):
        return
    candidates = []
    if sys.platform == "win32":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates.extend(
            program_files.glob(
                "Microsoft Visual Studio/*/*/Common7/IDE/CommonExtensions/"
                "Microsoft/CMake/CMake/bin"
            )
        )
    for cmake_dir in candidates:
        ninja_dir = cmake_dir.parent.parent / "Ninja"
        if (cmake_dir / "cmake.exe").is_file() and (ninja_dir / "ninja.exe").is_file():
            os.environ["PATH"] = os.pathsep.join(
                [str(cmake_dir), str(ninja_dir), os.environ.get("PATH", "")]
            )
            break
    if not shutil.which("cmake") or not shutil.which("ninja"):
        raise RuntimeError("CMake and Ninja are required for fixture generation")


def run(command, **kwargs):
    print("+", subprocess.list2cmdline([str(item) for item in command]), flush=True)
    return subprocess.run([str(item) for item in command], check=True, **kwargs)


def prepare_tinyllama_focused_case(root: Path, build: Path, case: str):
    entry = root / "tests/focused" / f"tinyllama_{case}.py"
    with tempfile.TemporaryDirectory(prefix=f"udoncl-{case}-", dir=build) as temporary:
        result_path = Path(temporary) / "result.json"
        run(
            [
                sys.executable,
                entry,
                "--build-dir",
                build,
                "--prepare-only",
                "--result-json",
                result_path,
            ],
            cwd=root,
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or result.get("case") != case:
        raise RuntimeError(f"invalid TinyLlama {case} focused result")
    artifact = Path(result["artifact"]).resolve()
    oracle = Path(result["oracle"]).resolve()
    if not (artifact / "manifest.json").is_file() or not oracle.is_file():
        raise RuntimeError(f"TinyLlama {case} focused result is incomplete")
    return artifact, oracle


def extract_payload(result):
    structured = result.get("structuredContent")
    if structured is not None:
        return structured.get("result", structured)
    for item in result.get("content", []):
        text = item.get("text")
        if text:
            try:
                payload = json.loads(text)
                return payload.get("result", payload)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"MCP response has no structured JSON payload: {result}")


class McpClient:
    def __init__(self, command, environment, stderr_path: Path):
        self._stderr = stderr_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            [str(item) for item in command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
        )
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._request_id = 0

    def _read_stdout(self):
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _send(self, message):
        if self._process.poll() is not None:
            raise RuntimeError(f"MCP server exited with {self._process.returncode}")
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def request(self, method, params=None, timeout=120):
        self._request_id += 1
        request_id = self._request_id
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"MCP request timed out: {method}")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError(f"MCP request timed out: {method}") from error
            if line is None:
                raise RuntimeError(
                    f"MCP server closed stdout with exit code {self._process.poll()}"
                )
            response = json.loads(line)
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise RuntimeError(f"MCP {method} failed: {response['error']}")
            return response["result"]

    def notify(self, method, params=None):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def initialize(self):
        self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "udoncl-precommit", "version": "1"},
            },
        )
        self.notify("notifications/initialized")

    def resource(self, uri):
        result = self.request("resources/read", {"uri": uri})
        for content in result.get("contents", []):
            if content.get("text"):
                return json.loads(content["text"])
        raise RuntimeError(f"MCP resource returned no JSON: {uri}")

    def tool(self, name, arguments, timeout=120):
        result = self.request(
            "tools/call", {"name": name, "arguments": arguments}, timeout=timeout
        )
        if result.get("isError"):
            raise RuntimeError(f"MCP tool {name} failed: {result}")
        payload = extract_payload(result)
        if isinstance(payload, dict) and payload.get("success") is False:
            raise RuntimeError(f"MCP tool {name} failed: {payload}")
        return payload

    def close(self):
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        self._stderr.close()


def wait_for_status(status_dir: Path, unity_process, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if unity_process.poll() is not None:
            raise RuntimeError(
                f"Unity exited before MCP became ready: {unity_process.returncode}"
            )
        files = list(status_dir.glob("unity-mcp-status-*.json"))
        if len(files) == 1:
            try:
                status = json.loads(files[0].read_text(encoding="utf-8"))
                if status.get("unity_port") and not status.get("reloading"):
                    return status
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(1)
    raise TimeoutError(f"Unity MCP status did not become ready: {status_dir}")


def wait_for_editor(client: McpClient, timeout=300):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = client.resource("mcpforunity://editor/state")
        data = last.get("data", last)
        if data.get("advice", {}).get("ready_for_tools") is True:
            return data
        # MCP for Unity 9.6.6 does not advance observed_at while an otherwise
        # idle batchmode Editor is waiting for commands. Its server therefore
        # adds stale_status after two seconds even though the live resource
        # explicitly reports every operational blocker as false.
        unity = data.get("unity", {})
        compilation = data.get("compilation", {})
        refresh = data.get("assets", {}).get("refresh", {})
        tests = data.get("tests", {})
        reasons = data.get("advice", {}).get("blocking_reasons", [])
        if (
            unity.get("is_batch_mode") is True
            and reasons == ["stale_status"]
            and compilation.get("is_compiling") is False
            and compilation.get("is_domain_reload_pending") is False
            and refresh.get("is_refresh_in_progress") is False
            and tests.get("is_running") is False
        ):
            return data
        time.sleep(1)
    raise TimeoutError(f"Unity Editor never became ready: {last}")


def wait_for_instance_reconnect(client: McpClient, timeout=600):
    """Wait for the MCP bridge to remain usable after a script-import reload."""
    deadline = time.monotonic() + timeout
    connected_since = None
    last = None
    while time.monotonic() < deadline:
        try:
            last = client.resource("mcpforunity://instances")
            instances = last.get("instances", [])
        except RuntimeError as error:
            last = str(error)
            instances = []
        if not instances:
            connected_since = None
            time.sleep(1)
            continue
        if connected_since is None:
            connected_since = time.monotonic()
        # Depending on transport timing, the server either reports no instance
        # during the reload or replaces the connection under the same instance
        # id. Requiring a continuously available bridge avoids accepting the
        # short stale-ready window in both cases.
        if time.monotonic() - connected_since < 10:
            time.sleep(1)
            continue
        if len(instances) != 1:
            raise RuntimeError(
                f"Expected one isolated Unity MCP instance after reload: {last}"
            )
        instance = instances[0].get("id") or instances[0].get("instance_id")
        if not instance:
            raise RuntimeError(f"Reloaded Unity MCP instance has no id: {last}")
        try:
            client.tool("set_active_instance", {"instance": instance})
            wait_for_editor(client, timeout=max(1, int(deadline - time.monotonic())))
            return instance
        except RuntimeError as error:
            last = str(error)
            connected_since = None
            time.sleep(1)
    raise TimeoutError(
        "Unity MCP bridge did not stabilize after script import: " f"{last}"
    )


def execute_menu_after_reload(
    client: McpClient, menu_path: str, *, timeout=300, reconnect_timeout=600
):
    for attempt in range(3):
        try:
            client.tool(
                "execute_menu_item",
                {"menu_path": menu_path},
                timeout=timeout,
            )
            return wait_for_instance_reconnect(client, timeout=reconnect_timeout)
        except RuntimeError as error:
            transient = (
                "Could not connect to Unity" in str(error)
                or "No Unity Editor instances found" in str(error)
                or "Timeout receiving Unity response" in str(error)
            )
            if not transient or attempt == 2:
                raise
            print(
                f"unity_mcp_menu=reconnecting menu={menu_path!r}",
                flush=True,
            )
            wait_for_instance_reconnect(client, timeout=reconnect_timeout)
    raise AssertionError("unreachable")


def start_unity_tests(client: McpClient, test_names=(), assembly_names=(TEST_ASSEMBLY,)):
    for reconnect_attempt in range(3):
        try:
            arguments = {
                "mode": "EditMode",
                "assembly_names": list(assembly_names),
                "include_failed_tests": True,
                "include_details": True,
            }
            if test_names:
                arguments["test_names"] = list(test_names)
            return client.tool(
                "run_tests",
                arguments,
                timeout=300,
            )
        except RuntimeError as error:
            if (
                "No Unity Editor instances found" not in str(error)
                or reconnect_attempt == 2
            ):
                raise
            print("unity_mcp=reconnecting_after_domain_reload", flush=True)
            wait_for_instance_reconnect(client, timeout=600)
    raise AssertionError("unreachable")


def validate_test_payload(payload, require_lowering=True):
    data = payload.get("data", payload)
    if data.get("status") != "succeeded":
        raise RuntimeError(f"Unity tests did not succeed: {payload}")
    result = data.get("result") or {}
    summary = result.get("summary") or {}
    total = int(summary.get("total", -1))
    passed = int(summary.get("passed", -1))
    failed = int(summary.get("failed", -1))
    skipped = int(summary.get("skipped", -1))
    if total <= 0 or passed != total or failed != 0 or skipped != 0:
        raise RuntimeError(f"Unity test summary is not clean: {summary}")
    if not require_lowering:
        return total
    results = result.get("results") or []
    observed = {
        method: sum(
            method in (case.get("name", "") + case.get("fullName", ""))
            for case in results
            if "UdonVmLoweringLifecycleTests" in case.get("fullName", "")
        )
        for method in LOWERING_TEST_CASES
    }
    if observed != LOWERING_TEST_CASES:
        raise RuntimeError(
            "Unity lowering tests were incomplete: "
            f"expected={LOWERING_TEST_CASES} observed={observed}"
        )
    return total


def is_transient_unity_import_failure(payload):
    data = payload.get("data", payload)
    result = data.get("result") or {}
    progress = data.get("progress") or {}
    failed = [
        case
        for case in (
            result.get("results") or progress.get("failures_so_far") or []
        )
        if case.get("state") not in ("Passed", "Success")
    ]
    if not failed:
        return False
    messages = [case.get("message") or "" for case in failed]
    runtime_compile_failures = [
        "Unity could not compile the Udon ONNX runtime U# program asset"
        in message
        for message in messages
    ]
    project_locks = [
        "Sharing violation on path" in message and ".csproj" in message
        for message in messages
    ]
    if not any(project_locks):
        return False
    return all(
        project_lock or runtime_compile_failure
        for project_lock, runtime_compile_failure in zip(
            project_locks, runtime_compile_failures
        )
    )


def is_transient_unity_test_start_failure(payload):
    data = payload.get("data", payload)
    progress = data.get("progress") or {}
    return (
        data.get("status") == "failed"
        and data.get("result") is None
        and int(progress.get("completed") or 0) == 0
        and "tests did not start within timeout" in (data.get("error") or "")
    )


def unity_console_messages(payload):
    data = payload.get("data", payload)
    if isinstance(data, dict):
        return data.get("messages", [])
    if isinstance(data, list):
        return data
    return []


def is_transient_unity_project_file_lock_messages(messages):
    if not messages:
        return False
    for message in messages:
        text = message.get("message") or ""
        file = message.get("file") or ""
        project_lock = "Sharing violation on path" in text and ".csproj" in text
        udonsharp_skin_startup = (
            "Unable to find style 'FloatFieldLinkButton' in skin 'GameSkin'"
            in text
            and "UdonSharpEditorManager.cs" in file
        )
        if not project_lock and not udonsharp_skin_startup:
            return False
    return True


def wait_for_test_job(client: McpClient, job_id: str, timeout=1260):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            payload = client.tool(
                "get_test_job",
                {
                    "job_id": job_id,
                    "include_failed_tests": True,
                    "include_details": True,
                    "wait_timeout": 60,
                },
                timeout=90,
            )
            status = (payload.get("data") or payload).get("status")
            if status in ("succeeded", "failed", "cancelled"):
                return payload
            last_error = f"test job remained in status {status!r}"
        except (RuntimeError, TimeoutError) as error:
            # The Unity bridge cannot answer discovery probes while a long
            # main-thread test/import is running. Keep the job and reconnect
            # after the Editor becomes responsive again.
            transient = (
                "No Unity Editor instances found" in str(error)
                or "Timeout receiving Unity response" in str(error)
                or "Could not connect to Unity" in str(error)
                or "MCP request timed out: tools/call" in str(error)
            )
            if not transient:
                raise
            last_error = str(error)
            print("unity_mcp_tests=reconnecting_after_disconnect", flush=True)
            wait_for_instance_reconnect(
                client, timeout=max(1, min(600, int(deadline - time.monotonic())))
            )
        time.sleep(2)
    raise TimeoutError(f"Unity test job did not finish: {last_error}")


def run_unity_mcp_gate(
    root,
    unity,
    project,
    build,
    output,
    test_names=(),
    test_assemblies=(TEST_ASSEMBLY,),
    mnist_artifact=None,
    tinyllama_prefill_artifact=None,
    tinyllama_prefill_oracle=None,
    tinyllama_conversation_artifact=None,
    tinyllama_conversation_oracle=None,
    prepare_standard_fixtures=True,
):
    require_unity_mcp(project)
    output.mkdir(parents=True, exist_ok=True)
    status_dir = output / "mcp-status"
    if status_dir.exists():
        shutil.rmtree(status_dir)
    status_dir.mkdir()
    manifest_path = project / "Packages/manifest.json"
    lock_path = project / "Packages/packages-lock.json"
    generated = project / "Assets/udoncl-generated"
    serialized = project / "Assets/SerializedUdonPrograms"
    sample_validation = project / "Assets/UdonclSampleValidation"
    sample_validation_meta = sample_validation.with_suffix(".meta")
    validation_scene = project / "Assets/UdonclValidationScene.unity"
    validation_scene_meta = validation_scene.with_suffix(".unity.meta")
    if sample_validation.exists() or sample_validation_meta.exists():
        raise RuntimeError(f"Unity sample validation path already exists: {sample_validation}")
    if validation_scene.exists() or validation_scene_meta.exists():
        raise RuntimeError(f"Unity validation scene path already exists: {validation_scene}")
    generated_before = tree_snapshot(generated)
    serialized_before = tree_snapshot(serialized)
    if prepare_standard_fixtures:
        unity_library = prepare_cmake_unity_library(build)
        assembly = prepare_fixtures(root, build, output)
        run(
            [
            sys.executable,
            root / "scripts/package_example_artifacts.py",
            "--arena",
            output / "onnx-aot/unity-arena-add-model-bundle",
            "--tiny",
            output / "onnx-aot/unity-tiny-llama-standard-model-bundle",
            "--hello",
            build
            / "tests/cmake-package/consumer/udoncl/hello-dispatch-unity-library/artifact",
            "--saxpy",
            build
            / "tests/cmake-package/saxpy-consumer/udoncl/saxpy-unity-library/artifact",
            "--pooling",
            build
            / "tests/cmake-package/average-pooling-consumer/udoncl/average-pooling-unity-library/artifact",
            "--check",
            ],
            cwd=root,
        )
    manifest_original = manifest_path.read_bytes()
    lock_original = lock_path.read_bytes() if lock_path.is_file() else None
    manifest = json.loads(manifest_original.decode("utf-8"))
    manifest.setdefault("dependencies", {})["com.lemolatoon.udoncl"] = (
        "file:" + (root / "unity/Packages/com.lemolatoon.udoncl").resolve().as_posix()
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    environment = dict(os.environ)
    if prepare_standard_fixtures:
        environment.update(
            {
            "UDONCL_LOWERED_UASM": str(assembly),
            "UDONCL_CONTINUATION_UASM": str(
                (output / "lowering/continuation.uasm").resolve()
            ),
            "UDONCL_UNITY_LIBRARY_ARTIFACT": str(unity_library),
            "UDONCL_PUBLIC_MNIST_UNITY_BUNDLE": str(
                (
                    mnist_artifact
                    if mnist_artifact is not None
                    else build
                    / "tests/cmake-package/onnx-mnist-consumer/udoncl/onnx-mnist/artifact"
                ).resolve()
            ),
            "UDONCL_PUBLIC_TRANSFORMER_UNITY_BUNDLE": str(
                (
                    build
                    / "tests/cmake-package/onnx-transformer-consumer/udoncl/onnx-transformer-tiny/artifact"
                ).resolve()
            ),
            "UDONCL_PUBLIC_TOKENIZER_UNITY_BUNDLE": str(
                (
                    build
                    / "tests/cmake-package/consumer/udoncl/tinyllama-tokenizer-unity-library/artifact"
                ).resolve()
            ),
            "UDONCL_ONNX_AOT_UASM": str(
                (output / "onnx-aot/generic-host.uasm").resolve()
            ),
            "UDONCL_ONNX_TILED_AOT_UASM": str(
                (output / "onnx-aot/tiled-gemm-host.uasm").resolve()
            ),
            "UDONCL_ONNX_RL_OPERATOR_AOT_UASM": str(
                (output / "onnx-aot/rl-policy-host.uasm").resolve()
            ),
            "UDONCL_ONNX_TINY_LLAMA_OPERATOR_AOT_UASM": str(
                (output / "onnx-aot/tiny-llama-standard-host.uasm").resolve()
            ),
            "UDONCL_ONNX_LLAMA_8B_MODEL_UASM": str(
                (output / "llama-8b-onnx/model-host.uasm").resolve()
            ),
            "UDONCL_ONNX_TINY_STANDARD_UNITY_BUNDLE": str(
                (
                    output
                    / "onnx-aot/unity-tiny-llama-standard-model-bundle"
                ).resolve()
            ),
            "UDONCL_ONNX_TINY_STANDARD_REFERENCE": str(
                (
                    output
                    / "onnx-aot/unity-tiny-llama-standard-reference.bytes"
                ).resolve()
            ),
            "UDONCL_PACKAGED_EXAMPLE_VALIDATION": "1",
            }
        )
    environment.update(
        {
            "UDONCL_ASSIGNED_Q4_COMPONENT_UNITY_BUNDLE": str(
                (
                    build
                    / "tests/onnx-q4-component/consumer/udoncl/onnx-q4-component/artifact"
                ).resolve()
            ),
            "UDONCL_ASSIGNED_RMS_COMPONENT_UNITY_BUNDLE": str(
                (
                    build
                    / "tests/onnx-rms-component/consumer/udoncl/onnx-rms-component/artifact"
                ).resolve()
            ),
            "UDONCL_ASSIGNED_TRANSPOSE_COMPONENT_UNITY_BUNDLE": str(
                (
                    build
                    / "tests/onnx-transpose-component/consumer/udoncl/onnx-transpose-component/artifact"
                ).resolve()
            ),
            "UNITY_MCP_STATUS_DIR": str(status_dir),
            "UNITY_MCP_TRANSPORT": "stdio",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.longpaths",
            "GIT_CONFIG_VALUE_0": "true",
        }
    )
    if tinyllama_conversation_artifact is not None:
        environment["UDONCL_PUBLIC_TINYLLAMA_CONVERSATION_UNITY_BUNDLE"] = str(
            tinyllama_conversation_artifact
        )
        environment["UDONCL_PUBLIC_TINYLLAMA_CONVERSATION_ORACLE"] = str(
            tinyllama_conversation_oracle
        )
    if tinyllama_prefill_artifact is not None:
        environment["UDONCL_PUBLIC_TINYLLAMA_PREFILL_UNITY_BUNDLE"] = str(
            tinyllama_prefill_artifact
        )
        environment["UDONCL_PUBLIC_TINYLLAMA_PREFILL_ORACLE"] = str(
            tinyllama_prefill_oracle
        )
    lifecycle_backup = output / "unity-lifecycle-backup"
    shutil.rmtree(lifecycle_backup, ignore_errors=True)
    lifecycle_backup.mkdir()
    lifecycle_state = []
    for lifecycle_root in (generated, serialized):
        backup_root = lifecycle_backup / lifecycle_root.name
        root_existed = lifecycle_root.is_dir()
        if root_existed:
            shutil.copytree(lifecycle_root, backup_root)
        root_meta = lifecycle_root.with_suffix(".meta")
        meta_original = root_meta.read_bytes() if root_meta.is_file() else None
        lifecycle_state.append(
            (lifecycle_root, backup_root, root_existed, root_meta, meta_original)
        )
    unity_log = output / "unity.log"
    unity_process = None
    client = None
    passed = 0
    failure = None
    try:
        if prepare_standard_fixtures:
            sample_validation.mkdir()
            sample_sources = sorted(
                (
                    root / "unity/Packages/com.lemolatoon.udoncl/Samples~"
                ).rglob("*.cs")
            )
            if not sample_sources:
                raise RuntimeError("Unity package contains no sample scripts")
            for source in sample_sources:
                destination = sample_validation / source.name
                if destination.exists():
                    raise RuntimeError(f"duplicate Unity sample script name: {source.name}")
                shutil.copy2(source, destination)
        unity_command = [
            unity,
            "-batchmode",
            "-force-vulkan" if platform.system() == "Linux" else "-force-d3d11",
            "-projectPath",
            project,
            "-executeMethod",
            "MCPForUnity.Editor.McpCiBoot.StartStdioForCi",
            "-logFile",
            unity_log,
        ]
        print("+", subprocess.list2cmdline([str(item) for item in unity_command]), flush=True)
        unity_process = subprocess.Popen([str(item) for item in unity_command], env=environment)
        status = wait_for_status(status_dir, unity_process)
        require_resolved_unity_mcp(project)
        uvx = shutil.which("uvx")
        if not uvx:
            raise RuntimeError("uvx is required to start the pinned Unity MCP server")
        client = McpClient(
            [uvx, "--from", MCP_SOURCE, "mcp-for-unity", "--transport", "stdio"],
            environment,
            output / "mcp-server.log",
        )
        client.initialize()
        instances = client.resource("mcpforunity://instances")
        instance_list = instances.get("instances", [])
        if len(instance_list) != 1:
            raise RuntimeError(f"Expected one isolated Unity MCP instance: {instances}")
        instance = instance_list[0].get("id") or instance_list[0].get("instance_id")
        if not instance:
            instance = status.get("project_name")
        client.tool("set_active_instance", {"instance": instance})
        wait_for_editor(client)
        client.tool("read_console", {"action": "clear"})
        if prepare_standard_fixtures:
            execute_menu_after_reload(
                client,
                "Tools/udoncl/Internal/Prepare Public MNIST Validation",
            )
        if tinyllama_prefill_artifact is not None:
            execute_menu_after_reload(
                client,
                "Tools/udoncl/Internal/Prepare Public TinyLlama Prefill Validation",
                timeout=900,
                reconnect_timeout=900,
            )
            focused_test = (
                Path(project)
                / "Assets/UdonclSampleValidation/Editor/"
                "PublicTinyLlamaPrefillLifecycleTests.cs"
            )
            imported_program = (
                Path(project)
                / "Assets/udoncl-generated/tinyllama_real_prefill/Programs/"
                "host_program.asset"
            )
            missing = [
                str(path)
                for path in (focused_test, imported_program)
                if not path.is_file()
            ]
            if missing:
                raise RuntimeError(
                    "TinyLlama prefill preparation did not complete: "
                    + ", ".join(missing)
                )
        if tinyllama_conversation_artifact is not None:
            execute_menu_after_reload(
                client,
                "Tools/udoncl/Internal/Prepare Public TinyLlama Conversation Validation",
                timeout=900,
                reconnect_timeout=900,
            )
            focused_test = (
                Path(project)
                / "Assets/UdonclSampleValidation/Editor/"
                "PublicTinyLlamaConversationLifecycleTests.cs"
            )
            imported_program = (
                Path(project)
                / "Assets/udoncl-generated/tinyllama_real_conversation/Programs/"
                "host_program.asset"
            )
            missing = [
                str(path)
                for path in (focused_test, imported_program)
                if not path.is_file()
            ]
            if missing:
                raise RuntimeError(
                    "TinyLlama conversation preparation did not complete: "
                    + ", ".join(missing)
                )
        if prepare_standard_fixtures:
            execute_menu_after_reload(
                client,
                "Tools/udoncl/Internal/Prepare Public Transformer Validation",
            )
            execute_menu_after_reload(
                client,
                "Tools/udoncl/Internal/Prepare Public Tokenizer Validation",
            )
            execute_menu_after_reload(
                client,
                "Tools/udoncl/Internal/Prepare Public MNIST Controller Validation",
            )
            execute_menu_after_reload(
                client,
                "Tools/udoncl/Internal/Prepare Packaged Example Validation",
            )
            execute_menu_after_reload(
                client,
                "Tools/udoncl/Internal/Ensure Validation Scene Saved",
            )
        wait_for_editor(client)
        for attempt in range(4):
            started = start_unity_tests(client, test_names, test_assemblies)
            job_id = (started.get("data") or started).get("job_id")
            if not job_id:
                raise RuntimeError(f"Unity MCP did not return a test job id: {started}")
            finished = wait_for_test_job(client, job_id)
            (output / f"unity-mcp-results-attempt-{attempt + 1}.json").write_text(
                json.dumps(finished, indent=2) + "\n", encoding="utf-8"
            )
            import_failure = is_transient_unity_import_failure(finished)
            start_failure = is_transient_unity_test_start_failure(finished)
            if (not import_failure and not start_failure) or attempt == 3:
                break
            reason = (
                "transient_unity_import_compile"
                if import_failure
                else "transient_test_start_timeout"
            )
            print(f"unity_mcp_tests=retry reason={reason}", flush=True)
            wait_for_instance_reconnect(client, timeout=600)
            time.sleep(5)
            client.tool("read_console", {"action": "clear"})
        (output / "unity-mcp-results.json").write_text(
            json.dumps(finished, indent=2) + "\n", encoding="utf-8"
        )
        wait_for_editor(client)
        console = client.tool(
            "read_console",
            {"action": "get", "types": ["error"], "count": 200, "format": "detailed"},
        )
        (output / "unity-console.json").write_text(
            json.dumps(console, indent=2) + "\n", encoding="utf-8"
        )
        messages = unity_console_messages(console)
        if is_transient_unity_project_file_lock_messages(messages):
            print(
                "unity_mcp_console=retry reason=transient_editor_noise",
                flush=True,
            )
            client.tool("read_console", {"action": "clear"})
            wait_for_editor(client)
            time.sleep(2)
            console = client.tool(
                "read_console",
                {"action": "get", "types": ["error"], "count": 200, "format": "detailed"},
            )
            (output / "unity-console.json").write_text(
                json.dumps(console, indent=2) + "\n", encoding="utf-8"
            )
            messages = unity_console_messages(console)
        if messages:
            raise RuntimeError(f"Unity console contains {len(messages)} error(s): {messages}")
        passed = validate_test_payload(finished, require_lowering=not test_names)
    except Exception as error:
        failure = error
    finally:
        if client is not None:
            client.close()
        if unity_process is not None and unity_process.poll() is None:
            unity_process.terminate()
            try:
                unity_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                unity_process.kill()
                unity_process.wait(timeout=30)
        for (
            lifecycle_root,
            backup_root,
            root_existed,
            root_meta,
            meta_original,
        ) in lifecycle_state:
            shutil.rmtree(lifecycle_root, ignore_errors=True)
            if root_existed:
                shutil.copytree(backup_root, lifecycle_root)
            if meta_original is None:
                root_meta.unlink(missing_ok=True)
            else:
                root_meta.write_bytes(meta_original)
        shutil.rmtree(lifecycle_backup, ignore_errors=True)
        manifest_path.write_bytes(manifest_original)
        if lock_original is None:
            lock_path.unlink(missing_ok=True)
        else:
            lock_path.write_bytes(lock_original)
        shutil.rmtree(sample_validation, ignore_errors=True)
        sample_validation_meta.unlink(missing_ok=True)
        validation_scene.unlink(missing_ok=True)
        validation_scene_meta.unlink(missing_ok=True)
    leaked = project / "Library/UdonclLoweringIntegration"
    if leaked.exists():
        raise RuntimeError(f"Unity lowering fixture leaked its setup directory: {leaked}")
    if sample_validation.exists() or sample_validation_meta.exists():
        raise RuntimeError("Unity sample validation source was not cleaned up")
    if validation_scene.exists() or validation_scene_meta.exists():
        raise RuntimeError("Unity validation scene was not cleaned up")
    if tree_snapshot(generated) != generated_before or tree_snapshot(serialized) != serialized_before:
        raise RuntimeError("Unity tests changed lifecycle-managed assets")
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    print(f"unity_mcp_tests=passed tests={passed} cleanup=passed")


def restage_original_paths(root: Path, staged_paths):
    existing = [path for path in staged_paths if (root / path).exists()]
    if existing:
        run(["git", "add", "--", *existing], cwd=root)


def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv_defaults(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unity", type=Path, default=os.environ.get("UNITY_EDITOR_PATH")
    )
    parser.add_argument(
        "--project", type=Path, default=os.environ.get("UDONCL_UNITY_TEST_PROJECT")
    )
    parser.add_argument("--build-dir", type=Path, default=root / "build/udoncl")
    parser.add_argument("--output", type=Path, default=root / "build/precommit")
    parser.add_argument(
        "--unity-only",
        action="store_true",
        help="run only the MCP-driven Unity gate without the native pre-commit steps",
    )
    parser.add_argument(
        "--minimal-unity-fixtures",
        action="store_true",
        help="skip unrelated standard fixture generation for a named package test",
    )
    parser.add_argument(
        "--commit-snapshot",
        metavar="COMMIT",
        help=(
            "validate this exact clean checked-out commit without requiring or "
            "modifying a staged change"
        ),
    )
    parser.add_argument(
        "--unity-test",
        action="append",
        default=[],
        help="fully qualified Unity test name; may be repeated",
    )
    parser.add_argument(
        "--mnist-artifact",
        type=Path,
        help="override the public C++ ONNX MNIST Unity artifact",
    )
    parser.add_argument(
        "--tinyllama-prefill",
        action="store_true",
        help="build and run the focused real TinyLlama prefill Unity test",
    )
    parser.add_argument(
        "--reuse-tinyllama-prefill-artifact",
        type=Path,
        help=(
            "focused diagnosis only: reuse an already generated TinyLlama "
            "prefill artifact instead of rebuilding it"
        ),
    )
    parser.add_argument(
        "--tinyllama-conversation",
        action="store_true",
        help="build and run the focused real TinyLlama conversation Unity test",
    )
    parser.add_argument(
        "--reuse-tinyllama-conversation-artifact",
        type=Path,
        help=(
            "focused diagnosis only: reuse an already generated TinyLlama "
            "conversation artifact instead of rebuilding it"
        ),
    )
    args = parser.parse_args()
    snapshot_commit = None
    if args.commit_snapshot:
        snapshot_commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{args.commit_snapshot}^{{commit}}"],
            cwd=root,
            text=True,
        ).strip()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        if head != snapshot_commit:
            parser.error(
                f"checked-out HEAD is {head}, expected commit snapshot {snapshot_commit}"
            )
    if args.tinyllama_prefill and args.tinyllama_conversation:
        parser.error("run one focused TinyLlama Unity gate at a time")
    if args.minimal_unity_fixtures and (not args.unity_only or not args.unity_test):
        parser.error("--minimal-unity-fixtures requires --unity-only and --unity-test")
    if args.unity is None:
        parser.error("set UNITY_EDITOR_PATH in .env or pass --unity")
    if args.project is None:
        parser.error("set UDONCL_UNITY_TEST_PROJECT in .env or pass --project")
    unity = args.unity.resolve()
    project = args.project.resolve()
    if not unity.is_file():
        parser.error(f"Unity executable does not exist: {unity}")
    staged = []
    ensure_cmake_toolchain_on_path()
    ensure_msvc_environment(args.build_dir.resolve())
    if not args.unity_only:
        if args.commit_snapshot:
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                text=True,
            )
            if dirty:
                parser.error("commit snapshot validation requires a clean worktree")
            run([sys.executable, root / "scripts/format.py", "--check"], cwd=root)
        else:
            staged = subprocess.check_output(
                ["git", "diff", "--cached", "--name-only", "-z"], cwd=root
            ).decode("utf-8").rstrip("\0").split("\0")
            staged = [path for path in staged if path]
            if not staged:
                parser.error(
                    "stage the intended commit before running the pre-commit gate"
                )
            run([sys.executable, root / "scripts/format.py"], cwd=root)
            restage_original_paths(root, staged)
        run(
            [
                sys.executable,
                root / "scripts/install_onnx_mlir.py",
                "--check",
            ],
            cwd=root,
        )
        run([sys.executable, root / "scripts/validate_vpm_package.py"], cwd=root)
        if args.commit_snapshot:
            run(
                [
                    "cmake",
                    "-S",
                    root,
                    "-B",
                    args.build_dir.resolve(),
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_PREFIX_PATH={root / '.deps/llvm23'}",
                    "-DBUILD_TESTING=ON",
                    "-DVRC_ENABLE_NATIVE_FRAGMENT_TESTS=ON",
                ],
                cwd=root,
            )
        run(
            [
                sys.executable,
                root / "scripts/build.py",
                "--build-dir",
                args.build_dir.resolve(),
            ],
            cwd=root,
        )
        run(
            [
                "ctest",
                "--test-dir",
                args.build_dir.resolve(),
                "--output-on-failure",
                "-j",
                "1",
            ],
            cwd=root,
        )
        run(
            [
                sys.executable,
                root / "tests/boundary/md_crystal.py",
                args.build_dir.resolve(),
                args.output.resolve() / "md-crystal",
            ],
            cwd=root,
            timeout=1200,
        )
    tinyllama_prefill_artifact = None
    tinyllama_prefill_oracle = None
    tinyllama_conversation_artifact = None
    tinyllama_conversation_oracle = None
    unity_tests = list(args.unity_test)
    unity_test_assemblies = [TEST_ASSEMBLY]
    if args.tinyllama_prefill:
        if args.reuse_tinyllama_prefill_artifact:
            tinyllama_prefill_artifact = args.reuse_tinyllama_prefill_artifact.resolve()
            tinyllama_prefill_oracle = (
                args.build_dir.resolve() / "tests/tinyllama-real-onnx"
            )
            if not (tinyllama_prefill_artifact / "manifest.json").is_file():
                parser.error(
                    "reused TinyLlama prefill artifact has no manifest: "
                    f"{tinyllama_prefill_artifact}"
                )
            if not (tinyllama_prefill_oracle / "oracle.json").is_file():
                parser.error(
                    "reused TinyLlama prefill oracle is missing: "
                    f"{tinyllama_prefill_oracle / 'oracle.json'}"
                )
        else:
            tinyllama_prefill_artifact, oracle_file = prepare_tinyllama_focused_case(
                root, args.build_dir.resolve(), "prefill"
            )
            tinyllama_prefill_oracle = oracle_file.parent
        if not unity_tests:
            unity_tests.append(
                "Lemolatoon.Udoncl.Editor.Tests."
                "PublicTinyLlamaPrefillLifecycleTests."
                "GeneratedUSharpBindingMatchesAllRealPrefillOutputs"
            )
            unity_test_assemblies = ["Udoncl.TinyLlama.Prefill.Focused.Tests"]
    if args.tinyllama_conversation:
        if args.reuse_tinyllama_conversation_artifact:
            tinyllama_conversation_artifact = (
                args.reuse_tinyllama_conversation_artifact.resolve()
            )
            tinyllama_conversation_oracle = (
                args.build_dir.resolve()
                / "tests/tinyllama-real-onnx/m1-oracle.json"
            )
            if not (tinyllama_conversation_artifact / "manifest.json").is_file():
                parser.error(
                    "reused TinyLlama conversation artifact has no manifest: "
                    f"{tinyllama_conversation_artifact}"
                )
            if not tinyllama_conversation_oracle.is_file():
                parser.error(
                    "reused TinyLlama conversation oracle is missing: "
                    f"{tinyllama_conversation_oracle}"
                )
        else:
            tinyllama_conversation_artifact, tinyllama_conversation_oracle = (
                prepare_tinyllama_focused_case(
                    root, args.build_dir.resolve(), "conversation"
                )
            )
        if not unity_tests:
            unity_tests.append(
                "Lemolatoon.Udoncl.Editor.Tests."
                "PublicTinyLlamaConversationLifecycleTests."
                "GeneratedUSharpBindingGeneratesFirstRealToken"
            )
            unity_test_assemblies = ["Udoncl.TinyLlama.Focused.Tests"]
    run_unity_mcp_gate(
        root,
        unity,
        project,
        args.build_dir.resolve(),
        args.output.resolve(),
        test_names=unity_tests,
        test_assemblies=unity_test_assemblies,
        mnist_artifact=(
            args.mnist_artifact.resolve() if args.mnist_artifact else None
        ),
        tinyllama_prefill_artifact=tinyllama_prefill_artifact,
        tinyllama_prefill_oracle=tinyllama_prefill_oracle,
        tinyllama_conversation_artifact=tinyllama_conversation_artifact,
        tinyllama_conversation_oracle=tinyllama_conversation_oracle,
        prepare_standard_fixtures=not args.minimal_unity_fixtures,
    )
    if args.unity_only:
        print("unity_validation=passed")
        return
    run([sys.executable, root / "scripts/format.py", "--check"], cwd=root)
    if args.commit_snapshot:
        parent_count = len(
            subprocess.check_output(
                ["git", "rev-list", "--parents", "-n", "1", "HEAD"],
                cwd=root,
                text=True,
            ).split()
        ) - 1
        if parent_count:
            run(["git", "diff", "--check", "HEAD^", "HEAD"], cwd=root)
        else:
            run(["git", "diff-tree", "--check", "--root", "HEAD"], cwd=root)
    else:
        run(["git", "diff", "--cached", "--check"], cwd=root)
    if snapshot_commit:
        final_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        if final_head != snapshot_commit:
            parser.error(
                f"HEAD changed during validation: expected {snapshot_commit}, got {final_head}"
            )
    print("precommit=passed")


if __name__ == "__main__":
    main()
