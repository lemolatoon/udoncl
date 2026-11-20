#!/usr/bin/env python3
"""Fixed, repository-owned workflow for real-TinyLlama measurements.

Subcommands (every output lands below --out):
  build      export one exact commit, build the compiler/SDK and the requested
             real TinyLlama artifacts, and regenerate their ONNX Runtime oracles
  run        run focused TinyLlama Unity gates (scripts/precommit.py, Unity
             MCP) in an isolated copy of the read-only Unity template and
             write summary.json/summary.txt
  probe      GPU time per kernel configuration of one decode run (batchmode
             EditMode test from scripts/tinyllama_gpu_probe) in an isolated
             project copy, one Editor session per plan
  summarize  rewrite the summary of a run directory from its logs
  clean      delete the heavy trees a previous subcommand recorded, keep logs

The workflow never opens the UDONCL_UNITY_TEST_PROJECT template, never shares
a mutable tree between concurrent invocations, and never touches the commit
validation service, its cache root or its Unity processes.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import precheck_service as service


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "udoncl-gates.json"
LOCK = ".udoncl-gates.lock"
SCHEMA = 1
PARALLEL = 4

# Build target name -> examples/onnx-transformer-tiny CMake target.
TARGETS = {
    "conversation": "onnx-transformer-tiny-conversation",
    "chat": "onnx-transformer-tiny-chat",
    "m1-step": "onnx-transformer-tiny-m1-step",
    "prefill": "onnx-transformer-tiny",
}
# tests/boundary/tinyllama_real_onnx.py writes the M=1 step sidecars for the
# example's default Context Capacity only.
ORACLE_CAPACITY = 512
# The pinned two-turn chat oracle (TINYLLAMA_AOT_CONVERSATION_PLAN.md 6.11).
CHAT_USER = "What is your name?"
CHAT_NEXT_USER = "Repeat your name."
# Free space needed before a build: the compiler tree plus about 8 GB per
# real artifact (observed 2026-11-20).
BUILD_BASE_GB = 10
BUILD_TARGET_GB = 10
BUILD_STEP_TIMEOUT = 4 * 3600
ORACLE_TIMEOUT = 1800


class GateError(RuntimeError):
    """A refused or failed step, reported without a traceback."""


def log(message: str):
    print(f"tinyllama_gates {message}", flush=True)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def lexically_within(path: Path, parent: Path) -> bool:
    """Containment by absolute path text, without resolving links."""
    path = os.path.normcase(os.path.abspath(path))
    parent = os.path.normcase(os.path.abspath(parent))
    try:
        return os.path.commonpath([path, parent]) == parent
    except ValueError:  # different drives
        return False


def environment_value(name: str, root: Path = ROOT) -> str | None:
    return os.environ.get(name) or service.parse_dotenv(root / ".env").get(name)


def free_gb(path: Path) -> float:
    probe = Path(path).resolve()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


def require_free_space(path: Path, needed_gb: float):
    available = free_gb(path)
    if available < needed_gb:
        raise GateError(
            f"only {available:.1f} GiB free below {path}; {needed_gb:.0f} GiB "
            "required (free space or pass a smaller --min-free-gb)"
        )


def check_out_directory(
    out: Path,
    *,
    root: Path,
    template: Path | None,
    service_root: Path,
):
    """Refuse output locations that would share state with something else."""
    out = Path(out).resolve()
    if is_within(out, service_root) or is_within(service_root, out):
        raise GateError(
            f"{out} overlaps the commit validation service cache {service_root}"
        )
    if template is not None:
        template = Path(template).resolve()
        if is_within(out, template) or is_within(template, out):
            raise GateError(f"{out} overlaps the Unity template {template}")
    if is_within(root, out):
        raise GateError(f"{out} contains the repository checkout {root}")
    if is_within(out, root) and not is_within(out, root / "build"):
        raise GateError(f"{out} is inside the repository but not below build/")


def parse_targets(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise GateError("no build targets given")
    unknown = [name for name in names if name not in TARGETS]
    if unknown:
        raise GateError(
            f"unknown build target(s) {', '.join(unknown)}; "
            f"choose from {', '.join(TARGETS)}"
        )
    # Deterministic order regardless of the command line.
    return [name for name in TARGETS if name in names]


def validate_build(targets: list[str], capacity: int):
    if capacity < 16:
        raise GateError(f"--capacity must be at least 16: {capacity}")
    if "m1-step" in targets and capacity != ORACLE_CAPACITY:
        raise GateError(
            f"the M=1 step oracle sidecars exist only for capacity {ORACLE_CAPACITY}"
        )


def required_build_gb(targets: list[str]) -> int:
    return BUILD_BASE_GB + BUILD_TARGET_GB * len(targets)


def read_manifest(directory: Path) -> dict:
    path = Path(directory) / MANIFEST
    if not path.is_file():
        raise GateError(f"{directory} has no {MANIFEST}; not a tinyllama_gates directory")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise GateError(f"unsupported {MANIFEST} schema in {directory}")
    return data


def write_manifest(directory: Path, data: dict):
    data = {"schema": SCHEMA, **data}
    path = Path(directory) / MANIFEST
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class DirectoryLock:
    """An exclusive, pid-stamped lock file; a dead owner's lock is stale."""

    def __init__(self, directory: Path, purpose: str):
        self.path = Path(directory) / LOCK
        self.purpose = purpose
        self.held = False

    def owner(self) -> int | None:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return None
        for line in text.splitlines():
            if line.startswith("pid="):
                with contextlib.suppress(ValueError):
                    return int(line[4:])
        return None

    def busy(self) -> bool:
        pid = self.owner()
        return pid is not None and pid != os.getpid() and service.process_alive(pid)

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self.busy():
                    raise GateError(
                        f"{self.path.parent} is in use by pid {self.owner()} "
                        f"({self.path.read_text(encoding='utf-8').strip()!r})"
                    )
                log(f"stale_lock removed={self.path}")
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(
                    f"pid={os.getpid()}\nhost={platform.node()}\n"
                    f"purpose={self.purpose}\nstarted={now()}\n"
                )
            self.held = True
            return self
        raise GateError(f"could not lock {self.path.parent}")

    def release(self):
        if self.held and self.owner() == os.getpid():
            self.path.unlink(missing_ok=True)
        self.held = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()


def kill_tree(pid: int):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)


def popen_group_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def tail(path: Path, lines: int = 20) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(content.splitlines()[-lines:])


def run_logged(command, log_path: Path, *, cwd=None, env=None, timeout=None):
    """Run one step with its output in log_path; kill its process tree on timeout."""
    command = [str(item) for item in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"step log={log_path} command={subprocess.list2cmdline(command)}")
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8", errors="replace") as stream:
        stream.write(f"+ {subprocess.list2cmdline(command)}\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            **popen_group_options(),
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            kill_tree(process.pid)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=30)
            raise
    if code != 0:
        raise GateError(
            f"step failed with exit code {code} after "
            f"{time.monotonic() - started:.0f} s; log {log_path}:\n{tail(log_path)}"
        )


def make_directory_link(link: Path, target: Path):
    if link.exists() or os.path.lexists(link):
        return
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(Path(target).resolve()), str(link))
    else:
        os.symlink(Path(target).resolve(), link, target_is_directory=True)


def remove_directory_link(link: Path):
    if os.path.isjunction(link):
        os.rmdir(link)
    elif os.path.islink(link):
        if os.name == "nt" and os.path.isdir(link):
            os.rmdir(link)
        else:
            os.unlink(link)


def remove_heavy(path: Path, owner: Path):
    """Remove one recorded heavy tree below owner without following links."""
    path = Path(owner) / path if not Path(path).is_absolute() else Path(path)
    if not lexically_within(path, owner) or lexically_within(owner, path):
        raise GateError(f"refusing to delete {path} outside {owner}")
    if os.path.isjunction(path) or os.path.islink(path):
        remove_directory_link(path)
        return
    if not path.exists():
        return
    # The exported source links the shared repository .deps; drop the link
    # before the tree so nothing below it is ever traversed.
    remove_directory_link(path / ".deps")
    service.remove_owned(path, owner)
    log(f"removed {path}")


def git_hooks_path(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "config", "--local", "--get", "core.hooksPath"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def restore_git_hooks_path(root: Path, original: str | None):
    """Undo a core.hooksPath rewrite by anything this workflow ran."""
    current = git_hooks_path(root)
    if current == original:
        return
    expected = original
    if original and Path(original).as_posix().rstrip("/").endswith(".githooks"):
        expected = ".githooks"
    if expected is None:
        subprocess.run(["git", "config", "--local", "--unset", "core.hooksPath"], cwd=root)
    else:
        subprocess.run(["git", "config", "--local", "core.hooksPath", expected], cwd=root)
    log(f"restored core.hooksPath {current!r} -> {expected!r}")


def resolve_commit(root: Path, commit: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GateError(f"unknown commit {commit!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def export_commit(root: Path, commit: str, destination: Path):
    """Extract the exact commit (LF text, no Git metadata) into destination."""
    marker = destination / ".udoncl-gates-commit"
    if destination.exists():
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == commit:
            return
        raise GateError(f"{destination} exists and is not an export of {commit}")
    archive = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "archive", "--format=tar", commit],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        service.remove_owned(temporary, destination.parent)
    temporary.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(temporary, filter="data")
    (temporary / ".udoncl-gates-commit").write_text(commit + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def dependency_root(root: Path) -> Path:
    """The pinned LLVM/onnx-mlir/model prefix: this checkout's .deps or,
    in a linked worktree, the main checkout's."""
    candidates = [root / ".deps"]
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if common:
        candidates.append(Path(common).parent / ".deps")
    for candidate in candidates:
        if (candidate / "llvm23").is_dir():
            return candidate.resolve()
    raise GateError(
        "no .deps/llvm23 in " + " or ".join(str(item) for item in candidates)
        + "; run scripts/install_llvm.py in the main checkout"
    )


def cmake_cache_value(build: Path, name: str) -> str:
    prefix = f"{name}:"
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix) and "=" in line:
            return line.split("=", 1)[1]
    raise GateError(f"{build} has no {name} in CMakeCache.txt")


def example_configure_command(
    source: Path,
    example: Path,
    sdk: Path,
    llvm: Path,
    make_program: str,
    model: Path,
    tokenizer: Path,
    capacity: int,
) -> list[str]:
    suffix = ".exe" if os.name == "nt" else ""
    return [
        "cmake",
        "-S",
        str(source / "examples/onnx-transformer-tiny"),
        "-B",
        str(example),
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={make_program}",
        f"-DCMAKE_PREFIX_PATH={sdk}",
        f"-DUDONCLC_EXECUTABLE={sdk / 'bin' / ('udonclc' + suffix)}",
        f"-DUDONCL_CLANG_EXECUTABLE={llvm / 'bin' / ('clang' + suffix)}",
        f"-DUDONCL_LLVM_LINK_EXECUTABLE={llvm / 'bin' / ('llvm-link' + suffix)}",
        f"-DUDONCL_ONNX_FRONTEND_EXECUTABLE={sdk / 'bin' / ('udoncl-onnx-frontend' + suffix)}",
        f"-DUDONCL_TINYLLAMA_ONNX_MODEL={model}",
        f"-DUDONCL_TINYLLAMA_TOKENIZER_DIRECTORY={tokenizer}",
        f"-DUDONCL_TINYLLAMA_CONTEXT_CAPACITY={capacity}",
    ]


def last_line(command, *, cwd, env) -> str:
    output = subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not output:
        raise GateError(f"{command} printed nothing")
    return output.splitlines()[-1]


def build_main(args) -> int:
    root = ROOT
    out = args.out.resolve()
    targets = parse_targets(args.targets)
    validate_build(targets, args.capacity)
    template = environment_value("UDONCL_UNITY_TEST_PROJECT")
    check_out_directory(
        out,
        root=root,
        template=Path(template) if template else None,
        service_root=service.default_cache_root(root),
    )
    commit = resolve_commit(root, args.commit)
    needed = args.min_free_gb if args.min_free_gb is not None else required_build_gb(targets)
    require_free_space(out, needed)
    out.mkdir(parents=True, exist_ok=True)
    source, compiler, sdk, example = out / "src", out / "cb", out / "sdk", out / "ex"
    oracles, logs = out / "oracle", out / "logs"
    previous = None
    if (out / MANIFEST).is_file():
        previous = read_manifest(out)
        if previous.get("kind") != "build" or previous.get("commit") != commit:
            raise GateError(f"{out} already holds {previous.get('kind')} output for another commit")
        if previous.get("capacity") != args.capacity:
            raise GateError(f"{out} was built at capacity {previous.get('capacity')}")
    hooks = git_hooks_path(root)
    manifest = {
        "kind": "build",
        "commit": commit,
        "capacity": args.capacity,
        "source": "src",
        "compiler": "cb",
        "sdk": "sdk",
        "example": "ex",
        "targets": dict((previous or {}).get("targets", {})),
        "oracles": dict((previous or {}).get("oracles", {})),
        "heavy": ["src", "cb", "sdk", "ex"],
        "started": now(),
    }
    with DirectoryLock(out, "build"):
        try:
            write_manifest(out, {**manifest, "state": "building"})
            log(f"build commit={commit} out={out} targets={','.join(targets)}")
            export_commit(root, commit, source)
            deps = dependency_root(root)
            make_directory_link(source / ".deps", deps)
            llvm = deps / "llvm23"
            # Toolchain discovery only; the exported tree builds itself.
            from msvc_environment import ensure_msvc_environment
            from precommit import ensure_cmake_toolchain_on_path

            ensure_cmake_toolchain_on_path()
            ensure_msvc_environment(compiler)
            environment = dict(os.environ)
            environment["UDONCL_MODEL_CACHE"] = str(deps / "models")
            environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(PARALLEL)
            if not (compiler / "CMakeCache.txt").is_file():
                run_logged(
                    [
                        "cmake", "-S", source, "-B", compiler, "-G", "Ninja",
                        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={llvm}",
                        "-DBUILD_TESTING=OFF",
                    ],
                    logs / "compiler-configure.log", cwd=source, env=environment,
                    timeout=BUILD_STEP_TIMEOUT,
                )
            run_logged(
                ["cmake", "--build", compiler, "--parallel", str(PARALLEL)],
                logs / "compiler-build.log", cwd=source, env=environment,
                timeout=BUILD_STEP_TIMEOUT,
            )
            if sdk.exists():
                service.remove_owned(sdk, out)
            run_logged(
                ["cmake", "--install", compiler, "--prefix", sdk, "--component", "UdonCLCompiler"],
                logs / "sdk-install.log", cwd=source, env=environment,
                timeout=BUILD_STEP_TIMEOUT,
            )
            model = last_line(
                [sys.executable, source / "scripts/fetch_tinyllama_onnx.py"],
                cwd=source, env=environment,
            )
            tokenizer = last_line(
                [
                    sys.executable, "-c",
                    "import importlib.util as u;"
                    "s=u.spec_from_file_location('f','scripts/fetch_tinyllama_tokenizer.py');"
                    "m=u.module_from_spec(s);s.loader.exec_module(m);"
                    "print(m.ensure(m.default_cache()))",
                ],
                cwd=source, env=environment,
            )
            if not (example / "CMakeCache.txt").is_file():
                run_logged(
                    example_configure_command(
                        source, example, sdk, llvm,
                        cmake_cache_value(compiler, "CMAKE_MAKE_PROGRAM"),
                        Path(model), Path(tokenizer), args.capacity,
                    ),
                    logs / "example-configure.log", cwd=source, env=environment,
                    timeout=BUILD_STEP_TIMEOUT,
                )
            for name in targets:
                target = TARGETS[name]
                started = time.monotonic()
                run_logged(
                    ["cmake", "--build", example, "--target", target, "--parallel", str(PARALLEL), "-v"],
                    logs / f"build-{name}.log", cwd=source, env=environment,
                    timeout=BUILD_STEP_TIMEOUT,
                )
                artifact = example / "udoncl" / target / "artifact"
                if not (artifact / "manifest.json").is_file():
                    raise GateError(f"{target} produced no artifact manifest: {artifact}")
                manifest["targets"][name] = {
                    "artifact": artifact.relative_to(out).as_posix(),
                    "seconds": round(time.monotonic() - started, 1),
                }
                write_manifest(out, {**manifest, "state": "building"})
                log(f"artifact target={name} path={artifact}")
            if {"conversation", "m1-step", "prefill"} & set(targets):
                real = oracles / "real"
                if not (real / "m1-oracle.json").is_file() or not (real / "oracle.json").is_file():
                    run_logged(
                        [sys.executable, source / "tests/boundary/tinyllama_real_onnx.py", source, real],
                        logs / "oracle-real.log", cwd=source, env=environment,
                        timeout=ORACLE_TIMEOUT,
                    )
                manifest["oracles"]["real"] = "oracle/real"
            if "chat" in targets:
                chat = oracles / "chat"
                if not (chat / "chat-oracle.json").is_file():
                    run_logged(
                        [
                            sys.executable, source / "tests/boundary/tinyllama_chat_oracle.py",
                            source, chat, "--user", CHAT_USER, "--next-user", CHAT_NEXT_USER,
                            "--capacity", str(args.capacity),
                        ],
                        logs / "oracle-chat.log", cwd=source, env=environment,
                        timeout=ORACLE_TIMEOUT,
                    )
                manifest["oracles"]["chat"] = "oracle/chat/chat-oracle.json"
            write_manifest(out, {**manifest, "state": "built", "finished": now()})
        except BaseException:
            write_manifest(out, {**manifest, "state": "failed", "finished": now()})
            raise
        finally:
            restore_git_hooks_path(root, hooks)
    log(f"build=passed out={out}")
    return 0


def prune_except(directory: Path, keep: list[Path], owner: Path):
    """Delete everything below directory except the kept trees."""
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if any(lexically_within(child, item) for item in keep):
            continue
        linked = os.path.isjunction(child) or os.path.islink(child)
        if not linked and any(lexically_within(item, child) for item in keep):
            prune_except(child, keep, owner)
        elif child.is_dir() or os.path.isjunction(child) or os.path.islink(child):
            remove_heavy(child, owner)
        else:
            child.unlink()


def clean_main(args) -> int:
    out = args.out.resolve()
    manifest = read_manifest(out)
    lock = DirectoryLock(out, "clean")
    if lock.busy():
        raise GateError(f"{out} is in use by pid {lock.owner()}")
    with lock:
        heavy = list(manifest.get("heavy", []))
        if args.keep_artifacts and manifest.get("kind") == "build":
            # `run` still needs the exported source (its precommit gate and
            # Unity package), the artifacts and the oracles.
            heavy = [item for item in heavy if item not in (manifest.get("example"), manifest.get("source"))]
            keep = [out / target["artifact"] for target in manifest.get("targets", {}).values()]
            prune_except(out / manifest["example"], keep, out)
        for item in heavy:
            remove_heavy(Path(item), out)
        manifest["cleaned"] = now()
        manifest.pop("schema", None)
        write_manifest(out, manifest)
    log(f"clean=passed out={out}")
    return 0


# ---------------------------------------------------------------------------
# run: focused TinyLlama Unity gates through scripts/precommit.py


FOCUSED_TEST_NAMESPACE = "Lemolatoon.Udoncl.Editor.Tests."
# Gate -> build target, fixture log label, test assembly and test.
GATES = {
    "prefill": (
        "prefill",
        "tinyllama_prefill",
        "Udoncl.TinyLlama.Prefill.Focused.Tests",
        "PublicTinyLlamaPrefillLifecycleTests.GeneratedUSharpBindingMatchesAllRealPrefillOutputs",
    ),
    "step": (
        "m1-step",
        "tinyllama_m1_step",
        "Udoncl.TinyLlama.Step.Focused.Tests",
        "PublicTinyLlamaStepValidationTests.GeneratedUSharpBindingMatchesM1StepOutputs",
    ),
    "conv": (
        "conversation",
        "tinyllama_conversation",
        "Udoncl.TinyLlama.Focused.Tests",
        "PublicTinyLlamaConversationLifecycleTests.GeneratedUSharpBindingRunsGreedyConversationLifecycle",
    ),
    "chat": (
        "chat",
        "tinyllama_chat",
        "Udoncl.TinyLlama.Chat.Focused.Tests",
        "PublicTinyLlamaChatLifecycleTests.GeneratedUSharpBindingAnswersChatPromptWithText",
    ),
    "multiturn": (
        "chat",
        "tinyllama_multi_turn_chat",
        "Udoncl.TinyLlama.Chat.Focused.Tests",
        "PublicTinyLlamaMultiTurnChatLifecycleTests.GeneratedUSharpBindingContinuesChatAcrossTurns",
    ),
}
# Gates whose fixture drives a binding with asynchronous exports, the only
# ones UdonclFixtureTiming applies the UDONCL_FIXTURE_FRAME_* overrides to.
PACED_GATES = ("conv", "chat", "multiturn")
FIXTURE_VARIABLES = (
    "UDONCL_FIXTURE_TARGET_FRAME_RATE",
    "UDONCL_FIXTURE_FRAME_BUDGET_US",
    "UDONCL_FIXTURE_FRAME_CONTINUATIONS",
    "UDONCL_FIXTURE_FRAME_ADAPTIVE",
)
FRAME_SETTING_MAXIMUM = 1000000
# The template parts a project copy is made of (as the validation service).
PROJECT_HEAVY = ["Assets", "Packages", "ProjectSettings", "Library", "Temp", "Logs", "obj", "UserSettings"]
RUN_FREE_GB = 15
# Failures of the Editor or the MCP bridge, not of the measured code. The
# precommit gate already retries in-session; a whole session is retried once.
TRANSIENT_FAILURES = (
    "Unity MCP status did not become ready",
    "Unity exited before MCP became ready",
    "Unity Editor never became ready",
    "Unity MCP bridge did not stabilize",
    "Expected one isolated Unity MCP instance",
    "Could not connect to Unity",
    "No Unity Editor instances found",
    "Timeout receiving Unity response",
    "MCP request timed out",
    "tests did not start within timeout",
    "Sharing violation on path",
)
PERMANENT_FAILURES = (
    "Unity tests did not succeed",
    "Unity test summary is not clean",
    "Unity console contains",
    "changed lifecycle-managed assets",
    "preparation did not complete",
)
PROGRESS_MARKERS = ("exceeded max VM", "OPENCL_ERROR", "CL_INVALID")


def parse_gates(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise GateError("no gates given")
    unknown = [name for name in names if name not in GATES]
    if unknown:
        raise GateError(f"unknown gate(s) {', '.join(unknown)}; choose from {', '.join(GATES)}")
    return [name for name in GATES if name in names]


def validate_run(
    gates: list[str],
    *,
    fps: int,
    budget_us: int | None,
    continuations: int | None,
    adaptive: int | None,
    trace: bool,
):
    if not 0 <= fps <= 1000:
        raise GateError(f"--fps must be 0 (unbounded) to 1000: {fps}")
    for flag, value, maximum in (
        ("--budget-us", budget_us, FRAME_SETTING_MAXIMUM),
        ("--continuations", continuations, FRAME_SETTING_MAXIMUM),
        ("--adaptive", adaptive, 1),
    ):
        if value is not None and not 0 <= value <= maximum:
            raise GateError(f"{flag} must be 0 to {maximum}: {value}")
    if any(value is not None for value in (budget_us, continuations, adaptive)):
        unpaced = [gate for gate in gates if gate not in PACED_GATES]
        if unpaced:
            raise GateError(
                f"frame budget overrides apply only to {', '.join(PACED_GATES)}; "
                f"run {', '.join(unpaced)} separately"
            )
    if trace and fps <= 0:
        raise GateError("--trace needs --fps > 0: the pump trace exists only in fixed-rate mode")


def fixture_environment(base: dict, *, fps: int, budget_us, continuations, adaptive) -> dict:
    """The child environment: only the requested fixture overrides survive."""
    environment = {key: value for key, value in base.items() if key not in FIXTURE_VARIABLES}
    for name, value in zip(FIXTURE_VARIABLES, (fps or None, budget_us, continuations, adaptive)):
        if value is not None:
            environment[name] = str(value)
    return environment


def plan_sessions(gates: list[str]) -> list[list[str]]:
    """One Unity session per artifact; chat and multiturn share the Chat Artifact."""
    sessions: list[list[str]] = []
    for gate in gates:
        target = GATES[gate][0]
        for session in sessions:
            if GATES[session[0]][0] == target:
                session.append(gate)
                break
        else:
            sessions.append([gate])
    return sessions


def session_artifacts(build: Path, manifest: dict, gates: list[str]) -> dict[str, str]:
    """run_unity_mcp_gate keyword arguments of one session's artifact and oracle."""
    target = GATES[gates[0]][0]
    built = manifest.get("targets", {}).get(target)
    if not built:
        raise GateError(f"{build} has no {target} artifact; build it with --targets {target}")
    artifact = build / built["artifact"]
    oracles = manifest.get("oracles", {})
    oracle_name = "chat" if target == "chat" else "real"
    if oracle_name not in oracles:
        raise GateError(f"{build} has no {oracle_name} oracle")
    oracle = build / oracles[oracle_name]
    if target == "prefill":
        result = {"tinyllama_prefill_artifact": artifact, "tinyllama_prefill_oracle": oracle}
        required = oracle / "oracle.json"
    elif target == "m1-step":
        result = {"tinyllama_step_artifact": artifact, "tinyllama_step_oracles": oracle / "m1-steps"}
        required = oracle / "m1-steps"
    elif target == "conversation":
        result = {
            "tinyllama_conversation_artifact": artifact,
            "tinyllama_conversation_oracle": oracle / "m1-oracle.json",
        }
        required = oracle / "m1-oracle.json"
    else:
        result = {"tinyllama_chat_artifact": artifact, "tinyllama_chat_oracle": oracle}
        required = oracle
        if "multiturn" in gates and os.path.isfile(oracle):
            turns = json.loads(Path(oracle).read_text(encoding="utf-8")).get("turns") or []
            if not turns:
                raise GateError(f"{oracle} has no later turns for the multiturn gate")
    for path in (artifact / "manifest.json", required):
        if not path.exists():
            raise GateError(f"missing {path}")
    return {key: str(value) for key, value in result.items()}


def is_progress_line(text: str) -> bool:
    return text.startswith("udoncl ") or any(marker in text for marker in PROGRESS_MARKERS)


class ProgressTail:
    """Copy fixture lines of a growing Unity log to progress-times.log with
    the seconds since the session started."""

    def __init__(self, log_path: Path, progress_path: Path):
        import threading

        self.log_path, self.progress_path = log_path, progress_path
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.position, self.buffer = 0, b""

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        self.poll()

    def loop(self):
        while not self.stop_event.wait(0.5):
            self.poll()

    def poll(self):
        try:
            if not self.log_path.exists():
                return
            if self.log_path.stat().st_size < self.position:
                self.position, self.buffer = 0, b""
            with self.log_path.open("rb") as stream:
                stream.seek(self.position)
                data = stream.read()
            self.position += len(data)
            *lines, self.buffer = (self.buffer + data).split(b"\n")
            elapsed = time.monotonic() - self.started
            selected = [
                text
                for text in (line.decode("utf-8", "replace").rstrip() for line in lines)
                if is_progress_line(text)
            ]
            if selected:
                with self.progress_path.open("a", encoding="utf-8") as stream:
                    for text in selected:
                        stream.write(f"{elapsed:10.1f}s {text}\n")
        except OSError:
            pass


def gate_child_main(args) -> int:
    """One Unity session in its own process (killed as a tree on timeout)."""
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    source = Path(spec["source"])
    # The exported commit's own gate, Unity package and fixtures.
    sys.path.insert(0, str(source / "scripts"))
    sys.dont_write_bytecode = True
    import precommit

    startup, tests = spec["startup_timeout"], spec["test_timeout"]
    wait_for_status = precommit.wait_for_status
    precommit.wait_for_status = lambda status_dir, process, timeout=600: wait_for_status(
        status_dir, process, timeout=max(timeout, startup)
    )
    wait_for_test_job = precommit.wait_for_test_job
    precommit.wait_for_test_job = lambda client, job_id, timeout=1260: wait_for_test_job(
        client, job_id, timeout=max(timeout, tests)
    )
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    progress = ProgressTail(output / "unity.log", output / "progress-times.log")
    progress.start()
    try:
        precommit.run_unity_mcp_gate(
            source,
            Path(spec["unity"]),
            Path(spec["project"]),
            Path(spec["build"]),
            output,
            test_names=spec["tests"],
            test_assemblies=spec["assemblies"],
            prepare_standard_fixtures=False,
            **{key: Path(value) for key, value in spec["artifacts"].items()},
        )
    finally:
        progress.stop()
    print("tinyllama_gate=passed", flush=True)
    return 0


def last_error(text: str) -> str | None:
    """The final exception line of a Python traceback, if any."""
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith("Traceback (most recent call last)"):
            for line in reversed(lines[index:]):
                if line and not line.startswith((" ", "\t")):
                    if not line.startswith("Traceback"):
                        return line
            return None
    return None


def is_transient_failure(text: str) -> bool:
    error = last_error(text)
    if error is None:
        return False
    if any(marker in error for marker in PERMANENT_FAILURES):
        return False
    return any(marker in error for marker in TRANSIENT_FAILURES)


def command_mentions_path(command_line: str, path: Path) -> bool:
    """Whether a process command line names exactly this directory."""
    haystack = command_line.replace("\\", "/").casefold()
    for form in {str(Path(path)), str(Path(path).resolve())}:
        needle = form.replace("\\", "/").rstrip("/").casefold()
        start = 0
        while (index := haystack.find(needle, start)) >= 0:
            end = index + len(needle)
            if end == len(haystack) or haystack[end] in "\"' /":
                return True
            start = index + 1
    return False


def unity_processes(project: Path) -> list[int]:
    """Unity Editors started on this project copy, and nothing else."""
    rows: list[tuple[int, str]] = []
    if os.name == "nt":
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='Unity.exe'\" | "
                "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
        )
        text = result.stdout.strip()
        if text:
            data = json.loads(text)
            for item in data if isinstance(data, list) else [data]:
                rows.append((int(item["ProcessId"]), item.get("CommandLine") or ""))
    else:
        for entry in Path("/proc").glob("[0-9]*"):
            with contextlib.suppress(OSError):
                rows.append((int(entry.name), (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")))
    return [pid for pid, command in rows if "-projectpath" in command.casefold() and command_mentions_path(command, project)]


def stop_project_editors(project: Path):
    for pid in unity_processes(project):
        log(f"killing leftover Unity pid={pid} project={project}")
        kill_tree(pid)


def prepare_project(template: Path, project: Path):
    """Create or refresh an isolated copy of the read-only template."""
    marker = project / MANIFEST
    if project.exists() and not marker.is_file() and any(
        child.name != LOCK for child in project.iterdir()
    ):
        raise GateError(f"{project} is not a tinyllama_gates project copy; refusing to use it")
    if unity_processes(project):
        raise GateError(f"a Unity Editor already has {project} open")
    project.mkdir(parents=True, exist_ok=True)
    if not marker.is_file():
        write_manifest(project, {"kind": "project", "template": str(template), "heavy": PROJECT_HEAVY})
    try:
        # Copies only Assets, Packages and ProjectSettings and keeps this
        # copy's own Library; the template is read, never opened.
        service.sync_unity_project(template, project, project)
    except OSError as error:
        raise GateError(f"could not refresh {project} (is an Editor still using it?): {error}")


def discard_project(project: Path):
    """A failed or killed session leaves the copy contaminated: rebuild it."""
    for name in PROJECT_HEAVY:
        remove_heavy(Path(name), project)


UDONCL_PACKAGE = "com.lemolatoon.udoncl"


def project_needs_warmup(project: Path) -> bool:
    """A copy whose Library holds no compiled scripts was never opened."""
    assemblies = project / "Library" / "ScriptAssemblies"
    return not assemblies.is_dir() or not any(assemblies.glob("*.dll"))


def manifest_with_package(text: str, package: Path) -> str:
    """The template manifest with the UdonCL package at an absolute path, as
    scripts/precommit.py writes it (the template's relative file: path does
    not resolve from a copy elsewhere)."""
    manifest = json.loads(text)
    manifest.setdefault("dependencies", {})[UDONCL_PACKAGE] = "file:" + Path(package).resolve().as_posix()
    return json.dumps(manifest, indent=2) + "\n"


def warmup_command(unity: Path, project: Path, log_path: Path) -> list[str]:
    return [
        str(unity),
        "-batchmode",
        "-force-vulkan" if platform.system() == "Linux" else "-force-d3d11",
        "-projectPath",
        str(project),
        "-quit",
        "-logFile",
        str(log_path),
    ]


def warm_up_project(
    unity: Path,
    project: Path,
    package: Path,
    log_path: Path,
    *,
    timeout: float,
    restore_manifest: bool,
):
    """Open a never-opened copy once in a plain batchmode Editor and quit.

    The first launch of a copy with an empty Library runs the initial import,
    the API updater and extra domain reloads; an MCP session started on such
    a copy never compiled the focused test assembly its Prepare menu wrote, so
    the Unity test run found 0 tests (2026-11-21, C:/t/gs-u0). The scratch
    drivers that worked always reused a copy with a built Library. This
    launch builds that Library before any gate or probe session.
    """
    manifest = project / "Packages" / "manifest.json"
    lock = project / "Packages" / "packages-lock.json"
    manifest_original = manifest.read_bytes()
    lock_original = lock.read_bytes() if lock.is_file() else None
    manifest.write_text(
        manifest_with_package(manifest_original.decode("utf-8"), package), encoding="utf-8"
    )
    environment = dict(os.environ)
    environment["UNITY_MCP_STATUS_DIR"] = str(log_path.parent / "warmup-mcp-status")
    command = warmup_command(unity, project, log_path)
    log(f"warm_up project={project} log={log_path}")
    code, started = None, time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            **popen_group_options(),
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            kill_tree(process.pid)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=60)
            if not isinstance(sys.exc_info()[1], subprocess.TimeoutExpired):
                raise
    finally:
        stop_project_editors(project)
        if restore_manifest:
            manifest.write_bytes(manifest_original)
            if lock_original is None:
                lock.unlink(missing_ok=True)
            else:
                lock.write_bytes(lock_original)
    if code != 0 or project_needs_warmup(project):
        raise GateError(
            f"Unity warm-up of {project} failed (exit code {code}, "
            f"{time.monotonic() - started:.0f} s); see {log_path}"
        )
    log(f"warm_up=passed seconds={time.monotonic() - started:.0f}")


def parse_value(text: str):
    if text in ("true", "false"):
        return text == "true"
    if text.startswith("[") and text.endswith("]"):
        items = [item for item in text[1:-1].split(",") if item]
        try:
            return [int(item) for item in items]
        except ValueError:
            return text
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def parse_fields(text: str) -> list[tuple[str, object]]:
    """key=value pairs of a fixture line; "quoted" values may hold spaces."""
    fields: list[tuple[str, object]] = []
    index, length = 0, len(text)
    while index < length:
        while index < length and text[index] == " ":
            index += 1
        start = index
        while index < length and text[index] not in " =":
            index += 1
        if index < length and text[index] == "=" and index > start:
            key = text[start:index]
            index += 1
            if index < length and text[index] == '"':
                end = text.find('" ', index + 1)
                end = length - 1 if end < 0 else end
                fields.append((key, text[index + 1 : end]))
                index = end + 1
            else:
                end = text.find(" ", index)
                end = length if end < 0 else end
                fields.append((key, parse_value(text[index:end])))
                index = end
        else:
            while index < length and text[index] != " ":
                index += 1
    return fields


def fixture_lines(text: str, label: str) -> list[tuple[str, list[tuple[str, object]], str]]:
    """(kind, fields, rest) of every `udoncl <label> ...` line."""
    prefix = f"udoncl {label} "
    result = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        rest = line[len(prefix) :].rstrip()
        kind, _, tail = rest.partition(" ")
        if "=" in kind:  # no kind word: the prefill result, phase changes
            kind, tail = "", rest
        result.append((kind, parse_fields(tail), tail))
    return result


def stop_reasons(windows: str) -> dict[str, int]:
    """Counts of the pump window exit reasons (UdonclFixtureTiming)."""
    names = {1: "limit", 2: "budget", 3: "idle", 4: "estimate"}
    counts = {name: 0 for name in names.values()}
    for item in windows.split(","):
        parts = item.split(":")
        if len(parts) == 4 and parts[3].isdigit():
            counts[names.get(int(parts[3]), "idle")] += 1
    return counts


def token_summary(fields: list[tuple[str, object]]) -> dict:
    """Generated tokens of a result line against the oracle's.

    conversation: first/second against expected; chat: tokens against
    expectedTokens; multi-turn chat: one tokens/expectedTokens pair per turn.
    """
    tokens = [value for key, value in fields if key in ("first", "second", "tokens")]
    expected = [value for key, value in fields if key in ("expected", "expectedTokens")]
    if not tokens or not all(isinstance(item, list) for item in tokens):
        return {}
    result: dict = {"generated_tokens": sum(len(item) for item in tokens)}
    if len(expected) == len(tokens):
        result["tokens_match"] = tokens == expected
    elif len(expected) == 1:
        result["tokens_match"] = all(item == expected[0] for item in tokens)
    texts = [value for key, value in fields if key == "text"]
    if texts:
        result["text"] = texts
        expected_texts = [value for key, value in fields if key == "expectedText"]
        if len(expected_texts) == len(texts):
            result["text_match"] = texts == expected_texts
    return result


def summarize_label(text: str, label: str) -> dict:
    lines = fixture_lines(text, label)
    summary: dict = {"label": label, "found": bool(lines)}
    if not lines:
        return summary
    frames = ms = 0.0
    totals = 0
    step_frames = steps = 0
    generate_frames = 0
    max_error = None
    mismatches = [0, 0]
    for kind, fields, tail in lines:
        values = dict(fields)
        if kind == "timing" and "total" in tail.split():
            totals += 1
            frames += values.get("frames", 0)
            ms += values.get("ms", 0.0)
            for key in ("mode", "target_fps", "paced_frames", "unity_frames"):
                if key in values:
                    summary[key] = values[key]
        elif kind == "timing" and "phase" in values:
            if values.get("steps", 0) > 0:
                step_frames += values.get("frames", 0)
                steps += values["steps"]
            if "generate" in str(values["phase"]):
                generate_frames += values.get("frames", 0)
        elif kind == "frame_timing":
            for key in ("main_ms_mean", "main_ms_max", "gpu_ms_mean", "gpu_ms_max"):
                if key in values:
                    summary[key] = values[key]
        elif kind == "pump":
            summary["pump"] = {key: value for key, value in fields}
        elif kind == "pump_budget":
            summary["pump_budget"] = values
        elif kind == "pump_windows":
            summary["stops_from_trace"] = stop_reasons(tail.split(" ")[-1])
        elif kind in ("result", "passed"):
            summary[kind] = True
            for key in ("status", "libraryStatus", "generate_steps", "fps"):
                if key in values:
                    summary[key] = values[key]
            summary.update(token_summary(fields))
        elif kind == "" and "outputs" in values:
            # The prefill fixture logs its one result line after every
            # output matched ORT.
            summary["result"] = summary["passed"] = True
            summary["outputs"] = values["outputs"]
        elif "output" in values and "max_abs_error" in values:
            error = values["max_abs_error"]
            if isinstance(error, (int, float)):
                max_error = error if max_error is None else max(max_error, error)
            parts = str(values.get("mismatches", "")).split("/")
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                mismatches[0] += int(parts[0])
                mismatches[1] += int(parts[1])
    if totals:
        summary["frames"] = int(frames)
        summary["ms"] = round(ms, 3)
    if steps:
        summary["steps"] = steps
        summary["frames_per_step"] = round(step_frames / steps, 3)
    if generate_frames and summary.get("generated_tokens"):
        summary["frames_per_token"] = round(generate_frames / summary["generated_tokens"], 3)
    if max_error is not None:
        summary["max_abs_error"] = max_error
        summary["mismatches"] = f"{mismatches[0]}/{mismatches[1]}"
    pump = summary.get("pump")
    if pump and "unity_frames" in pump:
        summary["busy_frames"] = pump["unity_frames"]
        if "frames" in summary:
            summary["idle_frames"] = summary["frames"] - pump["unity_frames"]
        summary["stops"] = {
            name: pump.get(f"stop_{name}", 0) for name in ("limit", "budget", "estimate", "idle")
        }
    return summary


def unity_test_counts(results: dict) -> dict:
    data = results.get("data", results)
    result = data.get("result") or {}
    counts = dict(result.get("summary") or {})
    failures = [
        f"{case.get('fullName') or case.get('name')}: {(case.get('message') or '')[:300]}"
        for case in result.get("results") or []
        if case.get("state") not in ("Passed", "Success")
    ]
    if failures:
        counts["failures"] = failures
    counts["status"] = data.get("status")
    return counts


def summarize_attempt(attempt: Path, gates: list[str]) -> dict:
    output = attempt / "output"
    unity_log = output / "unity.log"
    text = unity_log.read_text(encoding="utf-8", errors="replace") if unity_log.is_file() else ""
    driver = attempt / "driver.log"
    driver_text = driver.read_text(encoding="utf-8", errors="replace") if driver.is_file() else ""
    result = {
        "attempt": attempt.name,
        "passed": "tinyllama_gate=passed" in driver_text,
        "error": last_error(driver_text),
        "labels": {gate: summarize_label(text, GATES[gate][1]) for gate in gates},
    }
    results = output / "unity-mcp-results.json"
    if results.is_file():
        with contextlib.suppress(ValueError):
            result["tests"] = unity_test_counts(json.loads(results.read_text(encoding="utf-8")))
    console = output / "unity-console.json"
    if console.is_file():
        with contextlib.suppress(ValueError):
            payload = json.loads(console.read_text(encoding="utf-8"))
            data = payload.get("data", payload)
            messages = data.get("messages", []) if isinstance(data, dict) else data
            result["console_errors"] = len(messages) if isinstance(messages, list) else None
    timing = attempt / "attempt.json"
    if timing.is_file():
        result.update(json.loads(timing.read_text(encoding="utf-8")))
    if not result["passed"] and result.get("tests", {}).get("total") == 0:
        # The Editor ran but its test runner found nothing to run: the focused
        # assembly was not compiled into the domain. Retrying the same session
        # cannot help, so the run stops with this message.
        assemblies = []
        spec = attempt / "spec.json"
        if spec.is_file():
            with contextlib.suppress(ValueError):
                assemblies = json.loads(spec.read_text(encoding="utf-8")).get("assemblies", [])
        result["zero_tests"] = True
        result["error"] = (
            f"Unity ran 0 tests: the focused test assembly {', '.join(assemblies) or '?'} "
            f"was not compiled or loaded; see {unity_log}"
        )
    return result


def format_cell(value, digits=1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_table(summary: dict) -> str:
    header = (
        "gate", "result", "tests", "wall_s", "frames", "ms", "frames/step",
        "frames/token", "busy", "idle", "stops_l/b/e/i", "main_ms", "max_err", "tokens",
    )
    rows = [header]
    for session in summary.get("sessions", []):
        final = session["attempts"][-1] if session.get("attempts") else {}
        tests = final.get("tests", {})
        for gate in session["gates"]:
            label = final.get("labels", {}).get(gate, {})
            stops = label.get("stops")
            rows.append((
                gate,
                "pass" if final.get("passed") else "FAIL",
                f"{tests.get('passed', '-')}/{tests.get('total', '-')}",
                format_cell(final.get("wall_s"), 0),
                format_cell(label.get("frames")),
                format_cell(label.get("ms"), 0),
                format_cell(label.get("frames_per_step"), 2),
                format_cell(label.get("frames_per_token"), 2),
                format_cell(label.get("busy_frames")),
                format_cell(label.get("idle_frames")),
                "/".join(str(stops[name]) for name in ("limit", "budget", "estimate", "idle")) if stops else "-",
                format_cell(label.get("main_ms_mean"), 2),
                format_cell(label.get("max_abs_error"), 6) if label.get("max_abs_error") is not None else "-",
                "match" if label.get("tokens_match") else ("-" if "tokens_match" not in label else "DIFF"),
            ))
    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]
    for session in summary.get("sessions", []):
        final = session["attempts"][-1] if session.get("attempts") else {}
        if final and not final.get("passed"):
            lines.append(f"{'+'.join(session['gates'])}: {final.get('error') or 'failed'}")
    return "\n".join(lines) + "\n"


def write_summary(out: Path, summary: dict):
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = format_table(summary)
    (out / "summary.txt").write_text(table, encoding="utf-8")
    print(table, end="", flush=True)


def run_session(
    *,
    out: Path,
    gates: list[str],
    spec: dict,
    environment: dict,
    template: Path,
    project: Path,
    timeout: float,
) -> dict:
    name = "-".join(gates)
    attempts = []
    for number in (1, 2):
        attempt = out / name / f"attempt-{number}"
        attempt.mkdir(parents=True, exist_ok=True)
        prepare_project(template, project)
        spec_path = attempt / "spec.json"
        spec_path.write_text(
            json.dumps({**spec, "output": str(attempt / "output")}, indent=2) + "\n", encoding="utf-8"
        )
        started, timed_out = time.monotonic(), False
        if project_needs_warmup(project):
            try:
                warm_up_project(
                    Path(spec["unity"]),
                    project,
                    Path(spec["source"]) / "unity" / "Packages" / UDONCL_PACKAGE,
                    attempt / "warmup-unity.log",
                    timeout=spec["startup_timeout"],
                    restore_manifest=True,
                )
            except GateError as error:
                (attempt / "attempt.json").write_text(
                    json.dumps({"exit_code": None, "error": str(error),
                                "wall_s": round(time.monotonic() - started, 1)}) + "\n",
                    encoding="utf-8",
                )
                attempts.append(summarize_attempt(attempt, gates))
                log(f"session={name} attempt={number} error={error}")
                discard_project(project)
                break
        log(f"session={name} attempt={number} log={attempt / 'driver.log'}")
        with (attempt / "driver.log").open("w", encoding="utf-8", errors="replace") as stream:
            process = subprocess.Popen(
                [sys.executable, Path(__file__).resolve(), "_gate", spec_path],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
                **popen_group_options(),
            )
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out, code = True, None
                kill_tree(process.pid)
                process.wait(timeout=60)
            except BaseException:
                kill_tree(process.pid)
                stop_project_editors(project)
                raise
            finally:
                stop_project_editors(project)
        wall = time.monotonic() - started
        (attempt / "attempt.json").write_text(
            json.dumps({"exit_code": code, "timed_out": timed_out, "wall_s": round(wall, 1)}) + "\n",
            encoding="utf-8",
        )
        result = summarize_attempt(attempt, gates)
        attempts.append(result)
        log(f"session={name} attempt={number} passed={result['passed']} wall_s={wall:.0f}")
        if result["passed"]:
            break
        discard_project(project)
        if result.get("zero_tests"):
            log(f"session={name} error={result['error']}")
            break
        driver_text = (attempt / "driver.log").read_text(encoding="utf-8", errors="replace")
        if number == 1 and not timed_out and is_transient_failure(driver_text):
            log(f"session={name} retry reason={last_error(driver_text)!r}")
            continue
        break
    return {"gates": gates, "attempts": attempts}


def run_main(args) -> int:
    root = ROOT
    gates = parse_gates(args.gates)
    validate_run(
        gates,
        fps=args.fps,
        budget_us=args.budget_us,
        continuations=args.continuations,
        adaptive=args.adaptive,
        trace=args.trace,
    )
    build = args.artifacts.resolve()
    manifest = read_manifest(build)
    if manifest.get("kind") != "build" or manifest.get("state") != "built":
        raise GateError(f"{build} is not a finished build ({manifest.get('kind')}, {manifest.get('state')})")
    sessions = plan_sessions(gates)
    artifacts = {tuple(session): session_artifacts(build, manifest, session) for session in sessions}
    unity_value = environment_value("UNITY_EDITOR_PATH")
    template_value = environment_value("UDONCL_UNITY_TEST_PROJECT")
    if not unity_value or not Path(unity_value).is_file():
        raise GateError("UNITY_EDITOR_PATH is not set (.env) or does not exist")
    if not template_value or not (Path(template_value) / "Packages/manifest.json").is_file():
        raise GateError("UDONCL_UNITY_TEST_PROJECT is not set (.env) or is not a Unity project")
    unity, template = Path(unity_value).resolve(), Path(template_value).resolve()
    service_root = service.default_cache_root(root)
    out = args.out.resolve()
    project = (args.project or out / "project").resolve()
    for path in (out, project):
        check_out_directory(path, root=root, template=template, service_root=service_root)
    if (out / MANIFEST).is_file():
        raise GateError(f"{out} already holds a previous invocation; choose a new --out")
    if shutil.which("uvx") is None:
        raise GateError("uvx is required to start the pinned Unity MCP server")
    require_free_space(out, args.min_free_gb)
    if os.name == "nt" and len(str(project)) > 40:
        log(f"warning: long project path {project}; Unity Library paths may hit MAX_PATH")
    out.mkdir(parents=True, exist_ok=True)
    environment = fixture_environment(
        {**os.environ, "PYTHONUNBUFFERED": "1"},
        fps=args.fps,
        budget_us=args.budget_us,
        continuations=args.continuations,
        adaptive=args.adaptive,
    )
    settings = {
        "gates": gates,
        "fps": args.fps,
        "budget_us": args.budget_us,
        "continuations": args.continuations,
        "adaptive": args.adaptive,
        "trace": args.trace,
    }
    record = {
        "kind": "run",
        "artifacts": str(build),
        "commit": manifest.get("commit"),
        "capacity": manifest.get("capacity"),
        "settings": settings,
        "project": str(project),
        "heavy": ["project"] if project == out / "project" else [],
        "started": now(),
    }
    summary = {**record, "sessions": []}
    with DirectoryLock(out, "run"), DirectoryLock(project, "run"):
        write_manifest(out, {**record, "state": "running"})
        try:
            for session in sessions:
                spec = {
                    "source": str(build / manifest["source"]),
                    "unity": str(unity),
                    "project": str(project),
                    "build": str(build / manifest["compiler"]),
                    "tests": [FOCUSED_TEST_NAMESPACE + GATES[gate][3] for gate in session],
                    "assemblies": [GATES[session[0]][2]],
                    "artifacts": artifacts[tuple(session)],
                    "startup_timeout": args.startup_timeout_min * 60,
                    "test_timeout": args.timeout_min * 60,
                }
                summary["sessions"].append(
                    run_session(
                        out=out,
                        gates=session,
                        spec=spec,
                        environment=environment,
                        template=template,
                        project=project,
                        timeout=args.timeout_min * 60 + args.startup_timeout_min * 60 + 1800,
                    )
                )
                write_summary(out, summary)
                final = summary["sessions"][-1]["attempts"][-1:]
                if final and (
                    final[0].get("zero_tests") or "Unity warm-up" in str(final[0].get("error"))
                ):
                    # The project copy, not one gate, is broken: stop here.
                    log(f"run stopped: {final[0]['error']}")
                    break
        finally:
            summary["finished"] = now()
            write_summary(out, summary)
            write_manifest(out, {**record, "state": "finished", "finished": summary["finished"]})
    if args.trace:
        export_traces(out, summary)
    passed = all(session["attempts"] and session["attempts"][-1]["passed"] for session in summary["sessions"])
    log(f"run={'passed' if passed else 'failed'} summary={out / 'summary.json'}")
    return 0 if passed else 1


def export_traces(out: Path, summary: dict):
    """Write the raw pump trace of every paced gate as JSON next to the summary."""
    for session in summary["sessions"]:
        if not session["attempts"]:
            continue
        attempt = out / "-".join(session["gates"]) / session["attempts"][-1]["attempt"]
        log_path = attempt / "output" / "unity.log"
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for gate in session["gates"]:
            label = GATES[gate][1]
            trace = {}
            for kind, _, tail in fixture_lines(text, label):
                if kind == "pump_continuations":
                    trace["continuations"] = [
                        [int(part) for part in item.split(":")]
                        for item in tail.split(" ")[-1].split(",") if item.count(":") == 2
                    ]
                elif kind == "pump_windows":
                    trace["windows"] = [
                        [int(part) for part in item.split(":")]
                        for item in tail.split(" ")[-1].split(",") if item.count(":") == 3
                    ]
            if trace:
                (out / f"trace-{gate}.json").write_text(json.dumps(trace) + "\n", encoding="utf-8")


def summarize_main(args) -> int:
    """Re-read a run directory (for example after an interrupted run)."""
    out = args.out.resolve()
    manifest = read_manifest(out)
    if manifest.get("kind") != "run":
        raise GateError(f"{out} is not a run directory")
    summary = {key: value for key, value in manifest.items() if key != "schema"}
    summary["sessions"] = []
    for session in plan_sessions(manifest["settings"]["gates"]):
        directory = out / "-".join(session)
        attempts = sorted(directory.glob("attempt-*"), key=lambda path: path.name)
        summary["sessions"].append(
            {"gates": session, "attempts": [summarize_attempt(path, session) for path in attempts]}
        )
    write_summary(out, summary)
    return 0


# ---------------------------------------------------------------------------
# probe: GPU time per kernel of one decode run (batchmode Editor test)


PROBE_SOURCES = ROOT / "scripts" / "tinyllama_gpu_probe"
# Repository file -> file name inside <project>/Assets/UdonclGpuProbe/Tests.
PROBE_FILES = {
    "UdonclGpuKernelTimingProbe.cs.txt": "UdonclGpuKernelTimingProbe.cs",
    "UdonclGpuProbe.Tests.asmdef": "UdonclGpuProbe.Tests.asmdef",
}
PROBE_FOLDER = "Assets/UdonclGpuProbe"
PROBE_ASSEMBLY = "UdonclGpuProbe.Tests"
PROBE_TEST = "UdonclGpuKernelTimingProbe.MeasureDecodeRunKernels"
PROBE_PLAN_KEYS = ("configs", "sequence", "sequenceStatic", "staticIds", "staticWidth", "staticHeight")
SHADER_NAME = re.compile(r'^\s*Shader\s+"([^"]+)"', re.MULTILINE)


def probe_artifact(path: Path, target: str) -> Path:
    """An artifact directory, or the `target` artifact of a build directory."""
    path = Path(path).resolve()
    if (path / "manifest.json").is_file() and (path / "kernels").is_dir():
        return path
    manifest = read_manifest(path)
    built = manifest.get("targets", {}).get(target) if manifest.get("kind") == "build" else None
    if not built:
        raise GateError(f"{path} has no {target} artifact")
    return path / built["artifact"]


def shader_names(directory: Path) -> dict[str, Path]:
    names: dict[str, Path] = {}
    for path in sorted(Path(directory).rglob("*.shader")):
        with contextlib.suppress(OSError):
            match = SHADER_NAME.search(path.read_text(encoding="utf-8", errors="replace"))
            if match:
                names.setdefault(match.group(1), path)
    return names


def artifact_shader_names(artifact: Path) -> set[str]:
    """The shaders the probe copies: kernels/*.shader and buffer-copy.shader."""
    names = set(shader_names(artifact / "kernels"))
    copy = artifact / "buffer-copy.shader"
    if copy.is_file():
        match = SHADER_NAME.search(copy.read_text(encoding="utf-8", errors="replace"))
        if match:
            names.add(match.group(1))
    return names


def shader_conflicts(project: Path, artifact: Path) -> list[str]:
    """Artifact shader names a project copy already defines: Shader.Find
    would be ambiguous (for example an imported TinyLlama artifact)."""
    ours = artifact_shader_names(artifact)
    existing: dict[str, Path] = {}
    for folder in ("Assets", "Packages"):
        for name, path in shader_names(project / folder).items():
            if not lexically_within(path, project / PROBE_FOLDER):
                existing.setdefault(name, path)
    return sorted(name for name in ours if name in existing)


def validate_plan(plan: Path, artifact: Path) -> dict:
    try:
        data = json.loads(Path(plan).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GateError(f"cannot read probe plan {plan}: {error}")
    missing = [key for key in PROBE_PLAN_KEYS if key not in data]
    if missing:
        raise GateError(
            f"{plan} is not a Unity probe plan (missing {', '.join(missing)}); "
            "pass the *.unity.json plan"
        )
    available = artifact_shader_names(artifact)
    unknown = sorted({config.get("shader") for config in data["configs"]} - available)
    if unknown:
        raise GateError(f"{plan} names shaders {artifact} does not define: {', '.join(unknown[:5])}")
    if len(data["sequenceStatic"]) != 8 * len(data["sequence"]):
        raise GateError(f"{plan} sequenceStatic is not 8 entries per Blit")
    # Shader names repeat across builds; the device buffer and static tensor
    # shapes tie a plan to the artifact (capacity, layout) it was made from.
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    buffers = {
        item.get("id"): (item.get("width"), item.get("height"))
        for item in (manifest.get("device") or {}).get("buffers", [])
    }
    statics = {
        item.get("id"): ((item.get("texture") or {}).get("width"), (item.get("texture") or {}).get("height"))
        for item in manifest.get("static_tensors", [])
    }
    shapes = [
        (buffers, config.get("dstBuffer"), (config.get("dstWidth"), config.get("dstHeight")))
        for config in data["configs"]
    ] + [
        (statics, index, (width, height))
        for index, width, height in zip(data["staticIds"], data["staticWidth"], data["staticHeight"])
    ]
    for table, key, shape in shapes:
        if table and table.get(key) != shape:
            raise GateError(
                f"{plan} does not match {artifact}: tensor {key} is {table.get(key)}, plan {shape}"
            )
    return data


def plan_name(plan: Path) -> str:
    name = Path(plan).name
    for suffix in (".unity.json", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in name)


def parse_probe_log(text: str) -> dict:
    """The `udoncl gpu_probe` lines of one Editor log."""
    result: dict = {"configs": []}
    for line in text.splitlines():
        if not line.startswith("udoncl gpu_probe "):
            continue
        rest = line[len("udoncl gpu_probe ") :].rstrip()
        if rest.startswith("device="):
            # The device name holds spaces; keep the line as written.
            result["device"] = rest
        elif rest.startswith("config "):
            result["configs"].append(dict(parse_fields(rest[len("config ") :])))
        elif rest.startswith("replay "):
            result["replay"] = dict(parse_fields(rest[len("replay ") :]))
    result["configs"].sort(key=lambda config: -float(config.get("ms_per_run", 0.0)))
    return result


def parse_test_results(text: str) -> dict:
    """The test-run attributes of a Unity (NUnit 3) results file."""
    import xml.etree.ElementTree as ElementTree

    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        return {"result": "unreadable", "error": str(error)}
    if root.tag != "test-run":
        found = root.find(".//test-run")
        root = found if found is not None else root
    result = {"result": root.get("result")}
    for key in ("total", "passed", "failed", "skipped", "inconclusive"):
        if root.get(key) is not None:
            result[key] = int(root.get(key))
    messages = [item.text.strip() for item in root.iter("message") if item.text and item.text.strip()]
    if messages:
        result["messages"] = [message[:500] for message in messages[:5]]
    return result


def probe_command(unity: Path, project: Path, results: Path, log_path: Path) -> list[str]:
    return [
        str(unity),
        "-batchmode",
        "-force-vulkan" if platform.system() == "Linux" else "-force-d3d11",
        "-projectPath",
        str(project),
        "-runTests",
        "-testPlatform",
        "EditMode",
        "-assemblyNames",
        PROBE_ASSEMBLY,
        "-testFilter",
        PROBE_TEST,
        "-testResults",
        str(results),
        "-logFile",
        str(log_path),
    ]


def install_probe(project: Path):
    tests = project / PROBE_FOLDER / "Tests"
    tests.mkdir(parents=True, exist_ok=True)
    for source, destination in PROBE_FILES.items():
        shutil.copyfile(PROBE_SOURCES / source, tests / destination)


def format_probe_table(summary: dict, top: int) -> str:
    lines = []
    for item in summary["plans"]:
        replay = item.get("replay", {})
        lines.append(
            f"{item['name']}: {'pass' if item.get('passed') else 'FAIL'} "
            f"ms_per_run={replay.get('ms_per_run', '-')} "
            f"single_run_with_fence_ms={replay.get('single_run_with_fence_ms', '-')} "
            f"sum_of_configs_ms={replay.get('sum_of_configs_ms', '-')} "
            f"blits={replay.get('blits', '-')} wall_s={item.get('wall_s', '-')}"
        )
        if item.get("device"):
            lines.append(f"  {item['device']}")
        for config in item.get("configs", [])[:top]:
            lines.append(
                f"  kernel={config.get('kernel')} count={config.get('count')} "
                f"ms_per_blit={config.get('ms_per_blit')} ms_per_run={config.get('ms_per_run')} "
                f"dst={config.get('dst')} ints={config.get('ints')}"
            )
        if not item.get("passed"):
            lines.append(f"  error: {item.get('error') or item.get('tests')}")
    return "\n".join(lines) + "\n"


def probe_main(args) -> int:
    root = ROOT
    artifact = probe_artifact(args.artifacts, args.target)
    plans = [Path(plan).resolve() for plan in args.plan]
    names = [plan_name(plan) for plan in plans]
    if len(set(names)) != len(names):
        raise GateError(f"probe plan names collide: {', '.join(names)}")
    for plan in plans:
        validate_plan(plan, artifact)
    unity_value = environment_value("UNITY_EDITOR_PATH")
    template_value = environment_value("UDONCL_UNITY_TEST_PROJECT")
    if not unity_value or not Path(unity_value).is_file():
        raise GateError("UNITY_EDITOR_PATH is not set (.env) or does not exist")
    if not template_value or not (Path(template_value) / "Packages/manifest.json").is_file():
        raise GateError("UDONCL_UNITY_TEST_PROJECT is not set (.env) or is not a Unity project")
    unity, template = Path(unity_value).resolve(), Path(template_value).resolve()
    service_root = service.default_cache_root(root)
    out = args.out.resolve()
    project = (args.project or out / "project").resolve()
    for path in (out, project):
        check_out_directory(path, root=root, template=template, service_root=service_root)
    if (out / MANIFEST).is_file():
        raise GateError(f"{out} already holds a previous invocation; choose a new --out")
    require_free_space(out, args.min_free_gb)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": "probe",
        "artifact": str(artifact),
        "plans": [str(plan) for plan in plans],
        "project": str(project),
        "heavy": ["project"] if project == out / "project" else [],
        "started": now(),
    }
    summary = {**record, "plans": []}
    with DirectoryLock(out, "probe"), DirectoryLock(project, "probe"):
        write_manifest(out, {**record, "state": "running"})
        try:
            for plan, name in zip(plans, names):
                directory = out / name
                directory.mkdir(parents=True, exist_ok=True)
                # One plan per Editor session in a freshly synchronized
                # copy: every artifact defines the same shader names.
                prepare_project(template, project)
                # The template's relative UdonCL package path does not
                # resolve from the copy; use this checkout's package.
                manifest_path = project / "Packages" / "manifest.json"
                manifest_path.write_text(
                    manifest_with_package(
                        manifest_path.read_text(encoding="utf-8"),
                        root / "unity" / "Packages" / UDONCL_PACKAGE,
                    ),
                    encoding="utf-8",
                )
                if project_needs_warmup(project):
                    warm_up_project(
                        unity,
                        project,
                        root / "unity" / "Packages" / UDONCL_PACKAGE,
                        directory / "warmup-unity.log",
                        timeout=args.timeout_min * 60,
                        restore_manifest=False,
                    )
                install_probe(project)
                conflicts = shader_conflicts(project, artifact)
                if conflicts:
                    raise GateError(
                        f"the project copy already defines {len(conflicts)} artifact shader(s), "
                        f"e.g. {conflicts[0]}; Shader.Find would be ambiguous"
                    )
                environment = dict(os.environ)
                environment.update(
                    {
                        "UDONCL_GPU_PROBE_PLAN": str(plan),
                        "UDONCL_GPU_PROBE_ARTIFACT": str(artifact),
                        "UDONCL_GPU_PROBE_OUT": str(directory / "probe-result.json"),
                        "UNITY_MCP_STATUS_DIR": str(directory / "mcp-status"),
                    }
                )
                command = probe_command(unity, project, directory / "results.xml", directory / "unity.log")
                log(f"probe plan={name} log={directory / 'unity.log'}")
                started, timed_out, code = time.monotonic(), False, None
                with (directory / "driver.log").open("w", encoding="utf-8") as stream:
                    stream.write(f"+ {subprocess.list2cmdline(command)}\n")
                    stream.flush()
                    process = subprocess.Popen(
                        command, stdout=stream, stderr=subprocess.STDOUT, env=environment,
                        **popen_group_options(),
                    )
                    try:
                        code = process.wait(timeout=args.timeout_min * 60)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        kill_tree(process.pid)
                        process.wait(timeout=60)
                    except BaseException:
                        kill_tree(process.pid)
                        raise
                    finally:
                        stop_project_editors(project)
                text = ""
                if (directory / "unity.log").is_file():
                    text = (directory / "unity.log").read_text(encoding="utf-8", errors="replace")
                tests = {}
                if (directory / "results.xml").is_file():
                    tests = parse_test_results((directory / "results.xml").read_text(encoding="utf-8", errors="replace"))
                item = {
                    "name": name,
                    "plan": str(plan),
                    "exit_code": code,
                    "timed_out": timed_out,
                    "wall_s": round(time.monotonic() - started, 1),
                    "tests": tests,
                    **parse_probe_log(text),
                }
                item["passed"] = (
                    code == 0 and tests.get("result", "").startswith("Passed") and "replay" in item
                )
                if not item["passed"]:
                    item["error"] = (
                        "timed out" if timed_out else f"exit code {code}, tests {tests.get('result')}"
                    )
                    discard_project(project)
                summary["plans"].append(item)
                (directory / "probe-summary.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
        finally:
            summary["finished"] = now()
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            table = format_probe_table(summary, args.top)
            (out / "summary.txt").write_text(table, encoding="utf-8")
            print(table, end="", flush=True)
            write_manifest(out, {**record, "state": "finished", "finished": summary["finished"]})
    passed = all(item["passed"] for item in summary["plans"])
    log(f"probe={'passed' if passed else 'failed'} summary={out / 'summary.json'}")
    return 0 if passed else 1



def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(
        dest="command", required=True, metavar="{build,run,probe,summarize,clean}"
    )

    build = commands.add_parser("build", help="build artifacts and oracles of one commit")
    build.add_argument("--commit", required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument(
        "--targets",
        default=",".join(TARGETS),
        help=f"comma-separated subset of {','.join(TARGETS)} (default: all)",
    )
    build.add_argument("--capacity", type=int, default=ORACLE_CAPACITY)
    build.add_argument("--min-free-gb", type=float)
    build.set_defaults(main=build_main)

    clean = commands.add_parser("clean", help="delete recorded heavy trees, keep logs")
    clean.add_argument("--out", type=Path, required=True)
    clean.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="for a build directory, keep what `run` needs: the exported source and the artifacts",
    )
    clean.set_defaults(main=clean_main)

    run = commands.add_parser("run", help="run focused TinyLlama Unity gates on a build")
    run.add_argument("--artifacts", type=Path, required=True, help="a finished `build` --out")
    run.add_argument(
        "--gates",
        required=True,
        help=f"comma-separated subset of {','.join(GATES)}",
    )
    run.add_argument("--out", type=Path, required=True)
    run.add_argument(
        "--fps",
        type=int,
        default=0,
        help="UDONCL_FIXTURE_TARGET_FRAME_RATE; 0 keeps the unbounded batchmode rate",
    )
    run.add_argument("--budget-us", type=int, help="UDONCL_FIXTURE_FRAME_BUDGET_US")
    run.add_argument("--continuations", type=int, help="UDONCL_FIXTURE_FRAME_CONTINUATIONS")
    run.add_argument("--adaptive", type=int, choices=(0, 1), help="UDONCL_FIXTURE_FRAME_ADAPTIVE")
    run.add_argument(
        "--trace",
        action="store_true",
        help="also write the raw pump trace of each paced gate (needs --fps)",
    )
    run.add_argument(
        "--project",
        type=Path,
        help="reusable isolated project copy (default <out>/project); keeps its Library",
    )
    run.add_argument("--timeout-min", type=int, default=360, help="Unity test job timeout")
    run.add_argument("--startup-timeout-min", type=int, default=60, help="Editor start/import timeout")
    run.add_argument("--min-free-gb", type=float, default=RUN_FREE_GB)
    run.set_defaults(main=run_main)

    probe = commands.add_parser("probe", help="measure GPU time per kernel of one decode run")
    probe.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="an artifact directory, or a `build` --out (see --target)",
    )
    probe.add_argument(
        "--target",
        default="conversation",
        choices=tuple(TARGETS),
        help="artifact of a build directory to probe (default: conversation)",
    )
    probe.add_argument(
        "--plan",
        type=Path,
        action="append",
        required=True,
        help="a gpu-probe-plan-*.unity.json of that artifact; repeatable, one Editor session each",
    )
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--project", type=Path, help="reusable isolated project copy (default <out>/project)")
    probe.add_argument("--top", type=int, default=10, help="kernel configurations listed per plan")
    probe.add_argument("--timeout-min", type=int, default=60, help="per-plan Editor timeout")
    probe.add_argument("--min-free-gb", type=float, default=RUN_FREE_GB)
    probe.set_defaults(main=probe_main)

    summarize = commands.add_parser("summarize", help="rewrite the summary of a run directory")
    summarize.add_argument("--out", type=Path, required=True)
    summarize.set_defaults(main=summarize_main)

    gate = commands.add_parser("_gate", help=argparse.SUPPRESS)
    gate.add_argument("spec", type=Path)
    gate.set_defaults(main=gate_child_main)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.main(args)
    except GateError as error:
        print(f"tinyllama_gates: error: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
