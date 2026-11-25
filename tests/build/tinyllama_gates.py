"""Pure parts of scripts/tinyllama_gates.py; never launches Unity or a build."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "scripts"))

import tinyllama_gates as gates  # noqa: E402


def refused(function, *args, **kwargs) -> str:
    try:
        function(*args, **kwargs)
    except gates.GateError as error:
        return str(error)
    raise AssertionError(f"{function.__name__} did not refuse {args} {kwargs}")


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


# Build arguments.
assert gates.parse_targets("prefill, conversation") == ["conversation", "prefill"]
assert gates.parse_targets("chat,m1-step,chat") == ["chat", "m1-step"]
assert "unknown build target" in refused(gates.parse_targets, "conversation,decode")
assert "no build targets" in refused(gates.parse_targets, " , ")
gates.validate_build(["conversation", "chat"], 64)
assert "capacity 512" in refused(gates.validate_build, ["m1-step"], 64)
assert "at least 16" in refused(gates.validate_build, ["chat"], 8)
assert gates.required_build_gb(["conversation", "chat"]) == 30
command = gates.example_configure_command(
    Path("s"), Path("e"), Path("k"), Path("l"), "ninja", Path("m.onnx"), Path("t"), 64
)
assert "-DUDONCL_TINYLLAMA_CONTEXT_CAPACITY=64" in command, command
assert "-DUDONCL_TINYLLAMA_ONNX_MODEL=m.onnx" in command, command
assert "-DUDONCL_TINYLLAMA_CONTEXT_4K=OFF" in command, command
assert not any("SEQUENCE_ROWS" in item for item in command), command
command = gates.example_configure_command(
    Path("s"), Path("e"), Path("k"), Path("l"), "ninja", Path("m.onnx"), Path("t"), 512, True
)
assert "-DUDONCL_TINYLLAMA_CONTEXT_4K=ON" in command, command
assert not gates.example_context_4k(Path("no-such-build"))

# The 4096-token targets: not built by default, fixed capacity, own oracles.
assert gates.parse_targets("chat-4k,conversation,conversation-4k") == [
    "conversation", "conversation-4k", "chat-4k",
]
assert "conversation-4k" not in gates.DEFAULT_TARGETS
assert gates.target_capacity("chat-4k", 64) == 4096
assert gates.target_capacity("chat", 64) == 64
assert gates.target_family("conversation-4k") == "conversation"
assert gates.TARGET_ORACLES["conversation-4k"] == "real-4k"
gates.validate_build(["conversation-4k", "chat-4k"], 64)
assert gates.GATES["conv-4k"][0] == "conversation-4k"
assert gates.GATES["multiturn-4k"][1:] == gates.GATES["multiturn"][1:]
assert "step-4k" not in gates.GATES and "prefill-4k" not in gates.GATES
assert gates.parse_gates("multiturn-4k,conv,chat-4k") == ["conv", "chat-4k", "multiturn-4k"]
# The minimal chat example: a project of its own at its own capacity, gated
# against the chat oracle at that capacity.
assert gates.parse_targets("tinyllama-chat,chat") == ["chat", "tinyllama-chat"]
assert "tinyllama-chat" not in gates.DEFAULT_TARGETS
assert gates.target_capacity("tinyllama-chat", 512) == gates.CHAT_EXAMPLE_CAPACITY == 2048
assert gates.target_family("tinyllama-chat") == "tinyllama-chat"
assert gates.TARGET_ORACLES["tinyllama-chat"] == "chat-2k"
assert gates.GATES["tinychat"][0] == "tinyllama-chat"
assert "tinychat" not in gates.PACED_GATES and "tinychat-4k" not in gates.GATES
assert gates.parse_gates("tinychat,conv") == ["conv", "tinychat"]
command = gates.chat_example_configure_command(
    Path("s"), Path("e"), Path("k"), Path("l"), "ninja", Path("m.onnx"), Path("t")
)
assert command[1:3] == ["-S", str(Path("s") / "examples/tinyllama-chat")], command
assert "-DUDONCL_TINYLLAMA_CONTEXT_CAPACITY=2048" in command, command
assert "-DUDONCL_TINYLLAMA_TOKENIZER_DIRECTORY=t" in command, command
assert not any("CONTEXT_4K" in item for item in command), command
with contextlib.redirect_stderr(io.StringIO()) as stderr:
    code = gates.main(["build", "--commit", "HEAD", "--out", "x", "--targets", "decode"])
assert code == 2 and "unknown build target" in stderr.getvalue(), stderr.getvalue()

with tempfile.TemporaryDirectory(prefix="udoncl gates ") as temporary:
    temporary = Path(temporary)

    # Output isolation.
    service_root = temporary / "ucl" / "0123456789abcdef"
    template = temporary / "template"
    repository = temporary / "repository"
    for path in (service_root, template, repository / "build"):
        path.mkdir(parents=True)
    isolation = dict(root=repository, template=template, service_root=service_root)
    gates.check_out_directory(temporary / "out", **isolation)
    gates.check_out_directory(repository / "build" / "gates", **isolation)
    assert "validation service" in refused(
        gates.check_out_directory, service_root / "w" / "0", **isolation
    )
    assert "validation service" in refused(
        gates.check_out_directory, temporary / "ucl", **isolation
    )
    assert "Unity template" in refused(
        gates.check_out_directory, template / "copy", **isolation
    )
    assert "repository" in refused(
        gates.check_out_directory, repository / "scripts" / "out", **isolation
    )
    assert "contains the repository" in refused(
        gates.check_out_directory,
        temporary,
        root=repository,
        template=None,
        service_root=Path(tempfile.gettempdir()) / "udoncl-gates-no-service",
    )
    assert "GiB" in refused(gates.require_free_space, temporary, 1e12)

    # Locks: a live owner refuses, a dead owner's lock is stale.
    locked = temporary / "locked"
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        locked.mkdir()
        (locked / gates.LOCK).write_text(f"pid={sleeper.pid}\n", encoding="utf-8")
        assert gates.DirectoryLock(locked, "test").busy()
        assert f"in use by pid {sleeper.pid}" in refused(
            gates.DirectoryLock(locked, "test").acquire
        )
    finally:
        sleeper.kill()
        sleeper.wait()
    assert not gates.DirectoryLock(locked, "test").busy()
    with gates.DirectoryLock(locked, "test") as lock:
        assert lock.owner() == os.getpid()
    assert not (locked / gates.LOCK).exists()

    # Heavy trees are removed without following the shared .deps link.
    shared = temporary / "shared-deps"
    (shared / "llvm23").mkdir(parents=True)
    (shared / "llvm23" / "keep.txt").write_text("keep", encoding="utf-8")
    build = temporary / "build-out"
    source = build / "src"
    (source / "scripts").mkdir(parents=True)
    gates.make_directory_link(source / ".deps", shared)
    assert (source / ".deps" / "llvm23" / "keep.txt").is_file()
    for name in ("cb", "sdk"):
        (build / name / "deep").mkdir(parents=True)
        (build / name / "deep" / "object.o").write_bytes(b"0")
    keep = build / "ex" / "udoncl" / "onnx-transformer-tiny-chat" / "artifact"
    keep.mkdir(parents=True)
    (keep / "manifest.json").write_text("{}", encoding="utf-8")
    (keep.parent / "CMakeFiles").mkdir()
    (build / "ex" / "build.ninja").write_text("", encoding="utf-8")
    keep_chat = build / "ex-chat" / "udoncl" / "tinyllama-chat" / "artifact"
    keep_chat.mkdir(parents=True)
    (keep_chat / "manifest.json").write_text("{}", encoding="utf-8")
    (build / "ex-chat" / "build.ninja").write_text("", encoding="utf-8")
    (build / "logs").mkdir()
    (build / "logs" / "build-chat.log").write_text("log", encoding="utf-8")
    gates.write_manifest(
        build,
        {
            "kind": "build",
            "source": "src",
            "example": "ex",
            "example_chat": "ex-chat",
            "targets": {
                "chat": {"artifact": keep.relative_to(build).as_posix()},
                "tinyllama-chat": {"artifact": keep_chat.relative_to(build).as_posix()},
            },
            "heavy": ["src", "cb", "sdk", "ex", "ex-chat"],
        },
    )
    assert "outside" in refused(gates.remove_heavy, Path(".."), build)
    assert "outside" in refused(gates.remove_heavy, build, build)
    assert gates.main(["clean", "--out", str(build), "--keep-artifacts"]) == 0
    assert (source / "scripts").is_dir() and (source / ".deps" / "llvm23").is_dir()
    assert not (build / "cb").exists() and not (build / "sdk").exists()
    assert (keep / "manifest.json").is_file()
    assert not (keep.parent / "CMakeFiles").exists()
    assert not (build / "ex" / "build.ninja").exists()
    assert (keep_chat / "manifest.json").is_file()
    assert not (build / "ex-chat" / "build.ninja").exists()
    assert (build / "logs" / "build-chat.log").is_file()
    assert "cleaned" in gates.read_manifest(build)
    assert gates.main(["clean", "--out", str(build)]) == 0
    assert not (build / "ex").exists() and not source.exists()
    assert not (build / "ex-chat").exists()
    assert (shared / "llvm23" / "keep.txt").is_file()
    with contextlib.redirect_stderr(io.StringIO()) as stderr:
        assert gates.main(["clean", "--out", str(temporary / "template")]) == 2
    assert gates.MANIFEST in stderr.getvalue()

    # Exact-commit export with LF text and the hooks path guard.
    project = temporary / "git"
    project.mkdir()
    git(project, "init", "-q")
    git(project, "config", "user.email", "gates@example.invalid")
    git(project, "config", "user.name", "Gates Test")
    git(project, "config", "core.autocrlf", "true")
    (project / ".gitattributes").write_text("*.py text eol=lf\n", encoding="utf-8")
    (project / "a.py").write_bytes(b"print(1)\nprint(2)\n")
    git(project, "add", ".")
    git(project, "commit", "-q", "-m", "fixture")
    commit = gates.resolve_commit(project, "HEAD")
    assert "unknown commit" in refused(gates.resolve_commit, project, "no-such-ref")
    exported = temporary / "export" / "src"
    gates.export_commit(project, commit, exported)
    assert (exported / "a.py").read_bytes() == b"print(1)\nprint(2)\n"
    assert not (exported / ".git").exists()
    gates.export_commit(project, commit, exported)
    assert "not an export" in refused(gates.export_commit, project, "0" * 40, exported)
    git(project, "config", "--local", "core.hooksPath", ".githooks")
    git(project, "config", "--local", "core.hooksPath", str(project / ".githooks"))
    with contextlib.redirect_stdout(io.StringIO()):
        gates.restore_git_hooks_path(project, ".githooks")
    assert git(project, "config", "--local", "--get", "core.hooksPath") == ".githooks"

    manifest = json.loads((build / gates.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["schema"] == gates.SCHEMA

# Run arguments.
assert gates.parse_gates("chat,step,multiturn") == ["step", "chat", "multiturn"]
assert "unknown gate" in refused(gates.parse_gates, "conv,decode")
paced = dict(fps=90, budget_us=8000, continuations=8, adaptive=1, trace=True)
gates.validate_run(["conv", "chat", "multiturn"], **paced)
gates.validate_run(["prefill", "step"], fps=0, budget_us=None, continuations=None, adaptive=None, trace=False)
assert "apply only to" in refused(gates.validate_run, ["conv", "step"], **paced)
assert "needs --fps" in refused(gates.validate_run, ["conv"], **{**paced, "fps": 0})
assert "--fps" in refused(gates.validate_run, ["conv"], **{**paced, "fps": 1001})
assert "--budget-us" in refused(gates.validate_run, ["conv"], **{**paced, "budget_us": -1})
assert "--adaptive" in refused(gates.validate_run, ["conv"], **{**paced, "adaptive": 2})
ambient = {"PATH": "p", "UDONCL_FIXTURE_FRAME_BUDGET_US": "1", "UDONCL_FIXTURE_TARGET_FRAME_RATE": "60"}
assert gates.fixture_environment(
    ambient, fps=0, budget_us=None, continuations=None, adaptive=None
) == {"PATH": "p"}
assert gates.fixture_environment(ambient, fps=90, budget_us=4000, continuations=None, adaptive=0) == {
    "PATH": "p",
    "UDONCL_FIXTURE_TARGET_FRAME_RATE": "90",
    "UDONCL_FIXTURE_FRAME_BUDGET_US": "4000",
    "UDONCL_FIXTURE_FRAME_ADAPTIVE": "0",
}
assert gates.plan_sessions(["prefill", "conv", "chat", "multiturn"]) == [
    ["prefill"], ["conv"], ["chat", "multiturn"],
]
assert gates.plan_sessions(["multiturn"]) == [["multiturn"]]
assert gates.plan_sessions(["chat", "multiturn", "tinychat"]) == [
    ["chat", "multiturn"], ["tinychat"],
]
assert gates.plan_sessions(["chat", "conv-4k", "chat-4k", "multiturn-4k"]) == [
    ["chat"], ["conv-4k"], ["chat-4k", "multiturn-4k"],
]
gates.validate_run(
    ["conv-4k", "chat-4k"], fps=90, budget_us=8000, continuations=None, adaptive=1, trace=True
)

# Fixture log parsing (lines as the Editor fixtures write them).
fields = gates.parse_fields(
    'phase=12 (done) status=0 first=[1701,29916] text="My name is Sarah." '
    'expectedText="My name is Sarah." ms=1359.781 err=4.6E-05 mode=fixed ok=true'
)
assert fields == [
    ("phase", 12), ("status", 0), ("first", [1701, 29916]), ("text", "My name is Sarah."),
    ("expectedText", "My name is Sarah."), ("ms", 1359.781), ("err", 4.6e-05),
    ("mode", "fixed"), ("ok", True),
], fields
assert gates.parse_fields("ints=[_VRC_Arg3=2048,_VRC_Arg4=8]") == [("ints", "[_VRC_Arg3=2048,_VRC_Arg4=8]")]
assert gates.stop_reasons("11:0:17877:2,12:4:4964:2,23:5:3711:3,45:4:2863:4,50:2:100:1") == {
    "limit": 1, "budget": 2, "idle": 1, "estimate": 1,
}
assert gates.is_progress_line("udoncl tinyllama_chat pump trace=on")
assert gates.is_progress_line("Udon VM exceeded max VM time")
assert not gates.is_progress_line("UnityEngine.Debug:Log (object)")

CONVERSATION = """\
udoncl tinyllama_conversation pump_budget budget_us=8000 continuations=8 adaptive=true override=true
udoncl tinyllama_conversation timing mode=fixed target_fps=90 vsync=0 batchmode=true graphics=Direct3D11
udoncl tinyllama_conversation timing phase=initialize frames=1 ms=100.544 steps=0
udoncl tinyllama_conversation phase=3 (first-generate) frame=1 ms=102.337
udoncl tinyllama_conversation timing phase=first-generate frames=55 ms=621.261 steps=7 ms_per_step=88.752 frames_per_step=7.857
UnityEngine.Debug:Log (object)
udoncl tinyllama_conversation timing phase=second-generate frames=57 ms=632.650 steps=7 ms_per_step=90.379 frames_per_step=8.143
udoncl tinyllama_conversation timing total frames=113 unity_frames=105 ms=1359.781 fps=83.102 mode=fixed target_fps=90 paced_frames=111 readbacks=0\r
udoncl tinyllama_conversation frame_timing available=true feature_enabled=true samples=113 gpu_ms_mean=0.000 gpu_ms_max=0.000 main_ms_mean=2.783 main_ms_max=22.983 render_ms_mean=0.014 render_ms_max=0.040
udoncl tinyllama_conversation pump trace=on continuations=190 in_window=153 first_of_frame=37 unity_frames=40 windows=46 stop_limit=0 stop_budget=37 stop_estimate=0 stop_idle=9 window_us_mean=4688.435 window_us_max=17877
udoncl tinyllama_conversation pump_continuations 12:0:2929,12:1:1048,13:0:1203
udoncl tinyllama_conversation pump_windows 11:0:17877:2,12:4:4964:2,23:5:3711:3
udoncl tinyllama_conversation result phase=12 (done) status=0 libraryStatus=0 asyncDiagnostic=0 frames=113 firstCount=4 first=[1701,29916,432,17204] secondCount=4 second=[1701,29916,432,17204] expected=[1701,29916,432,17204] rejectedPromptStatus=12 disposedPromptStatus=8 ms=1359.781 fps=83.102 mode=fixed target_fps=90 generate_steps=7 first_generate_ms=621.261 second_generate_ms=632.65
udoncl tinyllama_conversation passed phase=12 (done) status=0 libraryStatus=0 frames=113 first=[1701,29916,432,17204] second=[1701,29916,432,17204] expected=[1701,29916,432,17204]
"""
conversation = gates.summarize_label(CONVERSATION, "tinyllama_conversation")
expected_conversation = {
    "frames": 113, "ms": 1359.781, "mode": "fixed", "target_fps": 90, "paced_frames": 111,
    "steps": 14, "frames_per_step": 8.0, "generated_tokens": 8, "tokens_match": True,
    "frames_per_token": 14.0, "busy_frames": 40, "idle_frames": 73, "main_ms_mean": 2.783,
    "main_ms_max": 22.983, "stops": {"limit": 0, "budget": 37, "estimate": 0, "idle": 9},
    "stops_from_trace": {"limit": 0, "budget": 2, "idle": 1, "estimate": 0},
    "result": True, "passed": True, "status": 0, "libraryStatus": 0, "generate_steps": 7,
}
for key, value in expected_conversation.items():
    assert conversation.get(key) == value, (key, conversation.get(key), value)
assert conversation["pump_budget"] == {
    "budget_us": 8000, "continuations": 8, "adaptive": True, "override": True,
}
assert gates.summarize_label(CONVERSATION, "tinyllama_chat") == {
    "label": "tinyllama_chat", "found": False,
}
diverged = gates.summarize_label(
    CONVERSATION.replace("second=[1701,29916,432,17204] expected", "second=[1701,29916,432,1] expected"),
    "tinyllama_conversation",
)
assert diverged["tokens_match"] is False

MULTI_TURN = """\
udoncl tinyllama_multi_turn_chat timing phase=generate-0 frames=160 ms=2399.012 steps=27
udoncl tinyllama_multi_turn_chat timing phase=generate-1 frames=155 ms=2413.156 steps=27
udoncl tinyllama_multi_turn_chat timing total frames=323 unity_frames=302 ms=4968.648 fps=65.008 mode=fixed target_fps=90 paced_frames=267 readbacks=0
udoncl tinyllama_multi_turn_chat result phase=10 (done) status=0 turn=0 tokens=[3421,1024,338,19235,29889,2] expectedTokens=[3421,1024,338,19235,29889,2] text="My name is Sarah." expectedText="My name is Sarah." turn=1 tokens=[29903,279,801,29889,2] expectedTokens=[29903,279,801,29889,2] text="Sarah." expectedText="Sarah." ms=4968.648 generate_steps=54
"""
multi_turn = gates.summarize_label(MULTI_TURN, "tinyllama_multi_turn_chat")
assert multi_turn["generated_tokens"] == 11 and multi_turn["tokens_match"] is True, multi_turn
assert multi_turn["text"] == ["My name is Sarah.", "Sarah."] and multi_turn["text_match"]
assert multi_turn["frames_per_token"] == round(315 / 11, 3), multi_turn
assert "busy_frames" not in multi_turn and "stops" not in multi_turn

STEP = """\
udoncl tinyllama_m1_step timing step=step-003 phase=run frames=2 ms=9.348 steps=1 ms_per_step=9.348 frames_per_step=2.000
udoncl tinyllama_m1_step timing step=step-003 total frames=192 unity_frames=190 ms=1847.473 fps=103.926 mode=unbounded target_fps=0 paced_frames=0 readbacks=45
udoncl tinyllama_m1_step step-003 phase=6 status=0 libraryStatus=0 frames=192 ms=1847.473 run_frames=2 run_ms=9.348
udoncl tinyllama_m1_step step-003 output=logits max_abs_error=4.673004150390625E-05 mismatches=0/32000 argmax_expected=1701 argmax_actual=1701
udoncl tinyllama_m1_step step-003 output=present.0.key max_abs_error=9.5367431640625E-06 mismatches=0/16384
udoncl tinyllama_m1_step timing step=step-004 total frames=100 unity_frames=99 ms=1000.0 fps=100.0 mode=unbounded target_fps=0 paced_frames=0 readbacks=45
udoncl tinyllama_m1_step step-004 output=logits max_abs_error=1.5E-04 mismatches=2/32000 first=7 expected=1.0 actual=1.1
"""
step = gates.summarize_label(STEP, "tinyllama_m1_step")
assert step["frames"] == 292 and step["ms"] == 2847.473, step
assert step["max_abs_error"] == 1.5e-04 and step["mismatches"] == "2/80384", step
assert step["frames_per_step"] == 2.0 and "result" not in step, step
PREFILL = (
    "udoncl tinyllama_prefill timing total frames=120 unity_frames=118 ms=900.5 fps=133.3 "
    "mode=unbounded target_fps=0 paced_frames=0 readbacks=45\n"
    "udoncl tinyllama_prefill frames=120 outputs=45 ms=900.5 fps=133.3 mode=unbounded "
    "target_fps=0 run_frames=3 run_ms=15.159\n"
)
prefill = gates.summarize_label(PREFILL, "tinyllama_prefill")
assert prefill["passed"] and prefill["outputs"] == 45 and prefill["frames"] == 120, prefill

# Failure classification from the child's traceback.
TRANSIENT = """\
+ Unity.exe -batchmode
Traceback (most recent call last):
  File "precommit.py", line 959, in run_unity_mcp_gate
    status = wait_for_status(status_dir, unity_process)
TimeoutError: Unity MCP status did not become ready: C:\\t\\run\\chat\\output\\mcp-status
"""
PERMANENT = """\
Traceback (most recent call last):
  File "precommit.py", line 1153, in run_unity_mcp_gate
RuntimeError: Unity tests did not succeed: {'status': 'failed', 'error': 'Could not connect to Unity'}
"""
assert gates.last_error(TRANSIENT).startswith("TimeoutError: Unity MCP status")
assert gates.is_transient_failure(TRANSIENT)
assert not gates.is_transient_failure(PERMANENT)
assert not gates.is_transient_failure("tinyllama_gate=passed\n")
assert gates.last_error("no traceback") is None

# Leftover Editors are matched by the exact project path only.
project_path = Path(tempfile.gettempdir()) / "gates-project"
unity_command = f'"C:/Unity.exe" -batchmode -projectPath {project_path} -logFile x.log'
assert gates.command_mentions_path(unity_command, project_path)
assert gates.command_mentions_path(f'Unity.exe -projectPath "{project_path}"', project_path)
assert not gates.command_mentions_path(unity_command.replace("gates-project", "gates-project2"), project_path)
assert not gates.command_mentions_path("Unity.exe -projectPath C:/ucl/w/0/u", project_path)

with tempfile.TemporaryDirectory(prefix="udoncl gates run ") as temporary:
    temporary = Path(temporary)

    # Session inputs come only from a finished build directory.
    build = temporary / "build"
    artifact = build / "ex" / "udoncl" / "onnx-transformer-tiny-chat" / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "manifest.json").write_text("{}", encoding="utf-8")
    (build / "oracle" / "chat").mkdir(parents=True)
    oracle = build / "oracle" / "chat" / "chat-oracle.json"
    oracle.write_text(json.dumps({"turns": []}), encoding="utf-8")
    build_manifest = {
        "targets": {"chat": {"artifact": artifact.relative_to(build).as_posix()}},
        "oracles": {"chat": "oracle/chat/chat-oracle.json"},
    }
    assert gates.session_artifacts(build, build_manifest, ["chat"]) == {
        "tinyllama_chat_artifact": str(artifact),
        "tinyllama_chat_oracle": str(oracle),
    }
    assert "no later turns" in refused(gates.session_artifacts, build, build_manifest, ["chat", "multiturn"])
    assert "no conversation artifact" in refused(gates.session_artifacts, build, build_manifest, ["conv"])
    oracle.write_text(json.dumps({"turns": [{}]}), encoding="utf-8")
    gates.session_artifacts(build, build_manifest, ["chat", "multiturn"])
    # tinychat runs the minimal chat fixture on the chat example artifact
    # and needs an oracle with a later turn.
    artifact_chat = build / "ex-chat" / "udoncl" / "tinyllama-chat" / "artifact"
    artifact_chat.mkdir(parents=True)
    (artifact_chat / "manifest.json").write_text("{}", encoding="utf-8")
    (build / "oracle" / "chat-2k").mkdir(parents=True)
    oracle_2k = build / "oracle" / "chat-2k" / "chat-oracle.json"
    oracle_2k.write_text(json.dumps({"turns": []}), encoding="utf-8")
    build_manifest_chat = {
        "targets": {
            "tinyllama-chat": {"artifact": artifact_chat.relative_to(build).as_posix()}
        },
        "oracles": {"chat-2k": "oracle/chat-2k/chat-oracle.json"},
    }
    assert "no later turn" in refused(
        gates.session_artifacts, build, build_manifest_chat, ["tinychat"]
    )
    oracle_2k.write_text(json.dumps({"turns": [{}]}), encoding="utf-8")
    assert gates.session_artifacts(build, build_manifest_chat, ["tinychat"]) == {
        "tinyllama_minimal_chat_artifact": str(artifact_chat),
        "tinyllama_minimal_chat_oracle": str(oracle_2k),
    }
    # A -4k gate runs its family's fixture on the -4k artifact and oracle.
    artifact_4k = build / "ex" / "udoncl" / "onnx-transformer-tiny-conversation-4k" / "artifact"
    artifact_4k.mkdir(parents=True)
    (artifact_4k / "manifest.json").write_text("{}", encoding="utf-8")
    (build / "oracle" / "real-4k").mkdir(parents=True)
    (build / "oracle" / "real-4k" / "m1-oracle.json").write_text("{}", encoding="utf-8")
    build_manifest_4k = {
        "targets": {
            "conversation-4k": {"artifact": artifact_4k.relative_to(build).as_posix()}
        },
        "oracles": {"real-4k": "oracle/real-4k"},
    }
    assert gates.session_artifacts(build, build_manifest_4k, ["conv-4k"]) == {
        "tinyllama_conversation_artifact": str(artifact_4k),
        "tinyllama_conversation_oracle": str(build / "oracle" / "real-4k" / "m1-oracle.json"),
    }
    assert "no conversation artifact" in refused(
        gates.session_artifacts, build, build_manifest_4k, ["conv"]
    )
    assert "no chat-4k oracle" in refused(
        gates.session_artifacts,
        build,
        {**build_manifest_4k, "targets": {"chat-4k": {"artifact": "ex"}}},
        ["chat-4k"],
    )
    # run refuses before any Unity work: unfinished builds, unpaced overrides.
    gates.write_manifest(build, {"kind": "build", "state": "building", **build_manifest})
    for arguments, message in (
        (["--gates", "chat"], "not a finished build"),
        (["--gates", "step", "--budget-us", "4000"], "apply only to"),
        (["--gates", "chat", "--trace"], "needs --fps"),
    ):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = gates.main(
                ["run", "--artifacts", str(build), "--out", str(temporary / "never"), *arguments]
            )
        assert code == 2 and message in stderr.getvalue(), (arguments, stderr.getvalue())
    assert not (temporary / "never").exists()

    # Progress lines are copied with the elapsed time.
    output = temporary / "output"
    output.mkdir()
    progress = gates.ProgressTail(output / "unity.log", output / "progress-times.log")
    (output / "unity.log").write_bytes(
        b"noise\nudoncl tinyllama_chat timing phase=generate frames=1\nUdon VM exceeded max VM time\npartial"
    )
    progress.poll()
    copied = (output / "progress-times.log").read_text(encoding="utf-8").splitlines()
    assert len(copied) == 2 and copied[0].rstrip().endswith("udoncl tinyllama_chat timing phase=generate frames=1")
    assert copied[0].lstrip().split("s ", 1)[0].replace(".", "").isdigit(), copied

    # Isolated project copies: refresh keeps the Library, foreign trees are refused.
    template = temporary / "template"
    for name in ("Assets", "Packages", "ProjectSettings"):
        (template / name).mkdir(parents=True)
    (template / "Packages" / "manifest.json").write_text("{}", encoding="utf-8")
    (template / "Library").mkdir()
    (template / "Library" / "template-only").write_text("x", encoding="utf-8")
    copy = temporary / "copy"
    with gates.DirectoryLock(copy, "test"):
        gates.prepare_project(template, copy)
        assert (copy / "Packages" / "manifest.json").is_file()
        assert not (copy / "Library").exists()
        (copy / "Library").mkdir()
        (copy / "Library" / "warm").write_text("x", encoding="utf-8")
        (copy / "Assets" / "leftover.cs").write_text("x", encoding="utf-8")
        gates.prepare_project(template, copy)
        assert (copy / "Library" / "warm").is_file()
        assert not (copy / "Assets" / "leftover.cs").exists()
        gates.discard_project(copy)
        assert not (copy / "Library").exists() and (copy / gates.LOCK).is_file()
    assert gates.read_manifest(copy)["kind"] == "project"
    foreign = temporary / "foreign"
    (foreign / "Assets").mkdir(parents=True)
    assert "not a tinyllama_gates project" in refused(gates.prepare_project, template, foreign)

    # A run directory is summarized from its logs alone.
    run = temporary / "run"
    run.mkdir()
    gates.write_manifest(run, {"kind": "run", "settings": {"gates": ["conv", "chat"]}, "heavy": []})
    attempt = run / "conv" / "attempt-1"
    (attempt / "output").mkdir(parents=True)
    (attempt / "output" / "unity.log").write_text(CONVERSATION, encoding="utf-8")
    (attempt / "driver.log").write_text("tinyllama_gate=passed\n", encoding="utf-8")
    (attempt / "attempt.json").write_text(json.dumps({"exit_code": 0, "wall_s": 812.4}), encoding="utf-8")
    (attempt / "output" / "unity-mcp-results.json").write_text(
        json.dumps({"data": {"status": "succeeded", "result": {
            "summary": {"total": 1, "passed": 1, "failed": 0, "skipped": 0},
            "results": [{"fullName": "T.Conv", "state": "Passed"}],
        }}}),
        encoding="utf-8",
    )
    (attempt / "output" / "unity-console.json").write_text(json.dumps({"data": {"messages": []}}), encoding="utf-8")
    failed = run / "chat" / "attempt-1"
    failed.mkdir(parents=True)
    (failed / "driver.log").write_text(PERMANENT, encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        assert gates.main(["summarize", "--out", str(run)]) == 0
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    first, second = summary["sessions"]
    assert first["gates"] == ["conv"] and first["attempts"][0]["passed"]
    assert first["attempts"][0]["wall_s"] == 812.4
    assert first["attempts"][0]["tests"]["passed"] == 1
    assert first["attempts"][0]["console_errors"] == 0
    assert first["attempts"][0]["labels"]["conv"]["busy_frames"] == 40
    assert second["gates"] == ["chat"] and not second["attempts"][0]["passed"]
    table = (run / "summary.txt").read_text(encoding="utf-8")
    assert table == stdout.getvalue()
    lines = table.splitlines()
    assert lines[0].split()[:3] == ["gate", "result", "tests"], lines
    assert lines[1].split()[:13] == [
        "conv", "pass", "1/1", "812", "113", "1360", "8.00", "14.00", "40", "73", "0/37/0/9", "2.78", "-",
    ], lines[1]
    assert lines[1].split()[-1] == "match"
    assert lines[2].split()[:2] == ["chat", "FAIL"], lines[2]
    assert lines[-1].startswith("chat: RuntimeError: Unity tests did not succeed"), lines[-1]

# A never-opened project copy is warmed up first: an MCP session started on
# an empty Library never compiled the focused test assembly and ran 0 tests.
with tempfile.TemporaryDirectory(prefix="udoncl gates warm ") as temporary:
    temporary = Path(temporary)
    project = temporary / "project"
    (project / "Packages").mkdir(parents=True)
    template_manifest = (
        '{\n  "testables": ["com.lemolatoon.udoncl"],\n  "dependencies": {\n'
        '    "com.coplaydev.unity-mcp": "https://example.invalid/mcp.git#main",\n'
        '    "com.lemolatoon.udoncl": "file:../../../../vrchat/udoncl/unity/Packages/com.lemolatoon.udoncl"\n'
        "  }\n}\n"
    )
    (project / "Packages" / "manifest.json").write_text(template_manifest, encoding="utf-8")
    (project / "Packages" / "packages-lock.json").write_text('{"dependencies": {}}\n', encoding="utf-8")
    package = temporary / "src" / "unity" / "Packages" / "com.lemolatoon.udoncl"
    patched = json.loads(gates.manifest_with_package(template_manifest, package))
    assert patched["dependencies"]["com.lemolatoon.udoncl"] == "file:" + package.resolve().as_posix()
    assert patched["dependencies"]["com.coplaydev.unity-mcp"].endswith("#main")
    assert patched["testables"] == ["com.lemolatoon.udoncl"]
    command = gates.warmup_command(Path("U.exe"), project, Path("w.log"))
    assert "-quit" in command and "-batchmode" in command and "-projectPath" in command
    assert "-executeMethod" not in command and "-runTests" not in command and "-nographics" not in command

    assert gates.project_needs_warmup(project)
    (project / "Library" / "ScriptAssemblies").mkdir(parents=True)
    assert gates.project_needs_warmup(project)
    fake_unity = temporary / "fake_unity.py"
    fake_unity.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "project, package, build = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'build'\n"
        "manifest = json.loads((project / 'Packages/manifest.json').read_text(encoding='utf-8'))\n"
        "assert manifest['dependencies']['com.lemolatoon.udoncl'] == 'file:' + package, manifest\n"
        "(project / 'Packages/packages-lock.json').write_text('rewritten', encoding='utf-8')\n"
        "if build:\n"
        "    (project / 'Library/ScriptAssemblies/Assembly-CSharp.dll').write_bytes(b'0')\n",
        encoding="utf-8",
    )
    original_command = gates.warmup_command
    try:
        for mode in ("empty", "build"):
            gates.warmup_command = lambda unity, project, log_path, mode=mode: [
                sys.executable, str(fake_unity), str(project), package.resolve().as_posix(), mode,
            ]
            arguments = (Path("U.exe"), project, package, temporary / f"warmup-{mode}.log")
            with contextlib.redirect_stdout(io.StringIO()):
                if mode == "empty":
                    assert "Unity warm-up" in refused(
                        gates.warm_up_project, *arguments, timeout=60, restore_manifest=True
                    )
                else:
                    gates.warm_up_project(*arguments, timeout=60, restore_manifest=True)
            # The copy keeps the template manifest and lock for the session.
            assert (project / "Packages" / "manifest.json").read_text(encoding="utf-8") == template_manifest
            assert (project / "Packages" / "packages-lock.json").read_text(encoding="utf-8") == (
                '{"dependencies": {}}\n'
            )
    finally:
        gates.warmup_command = original_command
    assert not gates.project_needs_warmup(project)

    # A session whose Unity run finds 0 tests fails at once with a clear
    # message and is not retried; a transient Editor failure is retried once.
    ZERO = (
        "Traceback (most recent call last):\n"
        '  File "precommit.py", line 1153, in run_unity_mcp_gate\n'
        "RuntimeError: Unity test summary is not clean: {'total': 0, 'passed': 0, 'failed': 0, "
        "'skipped': 0, 'durationSeconds': 0.0105962, 'resultState': 'Passed'}\n"
    )
    launches = []

    class FakeGate:
        def __init__(self, command, stdout=None, **_):
            spec = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            output = Path(spec["output"])
            output.mkdir(parents=True, exist_ok=True)
            launches.append(output)
            script = scripts.pop(0)
            stdout.write(script)
            total = 0 if "not clean" in script else 1
            (output / "unity-mcp-results.json").write_text(
                json.dumps({"data": {"status": "succeeded", "result": {"summary": {
                    "total": total, "passed": total, "failed": 0, "skipped": 0,
                }}}}),
                encoding="utf-8",
            )
            self.pid = 0

        def wait(self, timeout=None):
            return 0 if "tinyllama_gate=passed" in stdout_scripts[len(launches) - 1] else 1

    patched_names = ("prepare_project", "project_needs_warmup", "stop_project_editors", "discard_project")
    saved = {name: getattr(gates, name) for name in patched_names}
    saved_popen = gates.subprocess.Popen
    gates.prepare_project = lambda template, project: None
    gates.project_needs_warmup = lambda project: False
    gates.stop_project_editors = lambda project: None
    gates.discard_project = lambda project: None
    gates.subprocess.Popen = FakeGate
    try:
        session_spec = {"assemblies": ["Udoncl.TinyLlama.Focused.Tests"], "unity": "U.exe", "source": "src"}
        for stdout_scripts, expected_attempts, zero in (
            ([ZERO, "tinyllama_gate=passed\n"], 1, True),
            ([TRANSIENT, "tinyllama_gate=passed\n"], 2, False),
        ):
            scripts = list(stdout_scripts)
            launches.clear()
            out = temporary / f"session-{expected_attempts}"
            with contextlib.redirect_stdout(io.StringIO()):
                session = gates.run_session(
                    out=out, gates=["conv"], spec=session_spec, environment={},
                    template=temporary, project=project, timeout=60,
                )
            assert len(session["attempts"]) == expected_attempts, session
            final = session["attempts"][-1]
            if zero:
                assert final["zero_tests"] and not final["passed"]
                assert final["error"].startswith(
                    "Unity ran 0 tests: the focused test assembly Udoncl.TinyLlama.Focused.Tests"
                ), final["error"]
                assert "Unity ran 0 tests" in gates.format_table({"sessions": [session]})
            else:
                assert final["passed"] and "zero_tests" not in final
    finally:
        for name, value in saved.items():
            setattr(gates, name, value)
        gates.subprocess.Popen = saved_popen

# GPU kernel timing probe.
PROBE_LOG = """\
udoncl gpu_probe device=NVIDIA GeForce RTX 4070 api=Direct3D11 gpu_recorder=True batchmode=True
udoncl gpu_probe config id=3 kernel=217 count=45 dst=8192x2 ms_per_blit=0.1000 ms_per_run=4.500 model_loads=1 model_ops=2 model_max_loads=3 model_max_ops=4 ints=[_VRC_Arg3=2048,_VRC_Arg4=8]
udoncl gpu_probe config id=0 kernel=242 count=1 dst=8192x2 ms_per_blit=0.0123 ms_per_run=0.012 model_loads=0 model_ops=0 model_max_loads=0 model_max_ops=0 ints=[]
udoncl gpu_probe config id=4 kernel=4096 count=22 dst=64x64 ms_per_blit=2.0000 ms_per_run=44.000 model_loads=0 model_ops=0 model_max_loads=0 model_max_ops=0 ints=[_VRC_Arg9=4]
udoncl gpu_probe replay blits=415 ms_per_run=48.688 single_run_with_fence_ms=51.643 sum_of_configs_ms=48.512
"""
probe = gates.parse_probe_log(PROBE_LOG)
assert probe["device"].startswith("device=NVIDIA GeForce RTX 4070 api=Direct3D11"), probe
assert [config["kernel"] for config in probe["configs"]] == [4096, 217, 242], probe["configs"]
assert probe["configs"][0]["ms_per_run"] == 44.0 and probe["configs"][0]["dst"] == "64x64"
assert probe["configs"][1]["ints"] == "[_VRC_Arg3=2048,_VRC_Arg4=8]"
assert probe["replay"] == {
    "blits": 415, "ms_per_run": 48.688, "single_run_with_fence_ms": 51.643, "sum_of_configs_ms": 48.512,
}
assert gates.parse_probe_log("nothing") == {"configs": []}
assert gates.parse_test_results(
    '<?xml version="1.0"?><test-run id="2" testcasecount="1" result="Passed" total="1" '
    'passed="1" failed="0" inconclusive="0" skipped="0"><test-suite/></test-run>'
) == {"result": "Passed", "total": 1, "passed": 1, "failed": 0, "skipped": 0, "inconclusive": 0}
failed_results = gates.parse_test_results(
    '<test-run result="Failed(Child)" total="1" passed="0" failed="1" skipped="0">'
    "<test-case><failure><message><![CDATA[shader not found: UdonCL/x]]></message></failure>"
    "</test-case></test-run>"
)
assert failed_results["result"] == "Failed(Child)" and failed_results["messages"] == ["shader not found: UdonCL/x"]
assert gates.parse_test_results("<broken")["result"] == "unreadable"
assert gates.plan_name(Path("G/gpu-probe-plan-splitk64.unity.json")) == "gpu-probe-plan-splitk64"
assert gates.plan_name(Path("my plan.json")) == "my_plan"
command = gates.probe_command(Path("U.exe"), Path("P"), Path("r.xml"), Path("u.log"))
for flag in ("-batchmode", "-runTests", "UdonclGpuProbe.Tests", gates.PROBE_TEST, "EditMode"):
    assert flag in command, (flag, command)
assert "-nographics" not in command and "-quit" not in command

with tempfile.TemporaryDirectory(prefix="udoncl gates probe ") as temporary:
    temporary = Path(temporary)
    artifact = temporary / "artifact"
    (artifact / "kernels").mkdir(parents=True)
    (artifact / "kernels" / "kernel_242.shader").write_text(
        'Shader "UdonCL/device_gpu_242__gather"\n{\n}\n', encoding="utf-8"
    )
    (artifact / "buffer-copy.shader").write_text('Shader "Hidden/UdonCL/tiny/BufferCopy" {}\n', encoding="utf-8")
    (artifact / "manifest.json").write_text(
        json.dumps({
            "device": {"buffers": [{"id": 93, "width": 8192, "height": 2}]},
            "static_tensors": [{"id": 0, "texture": {"width": 8192, "height": 8000}}],
        }),
        encoding="utf-8",
    )
    plan = {
        "label": "gen1", "step": 0, "sequence": [0, 1], "sequenceStatic": [0] + [-1] * 15,
        "staticIds": [0], "staticWidth": [8192], "staticHeight": [8000],
        "configs": [
            {"id": 0, "kernel": "242", "shader": "UdonCL/device_gpu_242__gather", "dstBuffer": 93,
             "dstWidth": 8192, "dstHeight": 2},
            {"id": 1, "kernel": "buffer_copy", "shader": "Hidden/UdonCL/tiny/BufferCopy", "dstBuffer": 93,
             "dstWidth": 8192, "dstHeight": 2},
        ],
    }
    plan_path = temporary / "gpu-probe-plan-conv64.unity.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    assert gates.validate_plan(plan_path, artifact)["label"] == "gen1"
    for change, message in (
        ({"staticHeight": [16]}, "does not match"),
        ({"configs": [{**plan["configs"][0], "dstHeight": 16}]}, "does not match"),
        ({"configs": [{**plan["configs"][0], "shader": "UdonCL/other"}]}, "does not define"),
        ({"sequenceStatic": [0]}, "8 entries"),
    ):
        broken = temporary / "broken.unity.json"
        broken.write_text(json.dumps({**plan, **change}), encoding="utf-8")
        assert message in refused(gates.validate_plan, broken, artifact), change
    flat = temporary / "flat.json"
    flat.write_text(json.dumps({"configs": [], "sequence": []}), encoding="utf-8")
    assert "*.unity.json" in refused(gates.validate_plan, flat, artifact)
    assert gates.artifact_shader_names(artifact) == {
        "UdonCL/device_gpu_242__gather", "Hidden/UdonCL/tiny/BufferCopy",
    }

    # The probe installs its two files; an imported artifact's shaders conflict.
    project = temporary / "project"
    (project / "Assets" / "udoncl-generated" / "tiny").mkdir(parents=True)
    (project / "Packages").mkdir()
    gates.install_probe(project)
    installed = project / gates.PROBE_FOLDER / "Tests"
    assert (installed / "UdonclGpuKernelTimingProbe.cs").read_bytes() == (
        ROOT / "scripts/tinyllama_gpu_probe/UdonclGpuKernelTimingProbe.cs.txt"
    ).read_bytes()
    assert json.loads((installed / "UdonclGpuProbe.Tests.asmdef").read_text(encoding="utf-8"))["name"] == (
        gates.PROBE_ASSEMBLY
    )
    (project / gates.PROBE_FOLDER / "Shaders").mkdir()
    (project / gates.PROBE_FOLDER / "Shaders" / "kernel_242.shader").write_text(
        'Shader "UdonCL/device_gpu_242__gather" {}', encoding="utf-8"
    )
    assert gates.shader_conflicts(project, artifact) == []
    (project / "Assets" / "udoncl-generated" / "tiny" / "k.shader").write_text(
        'Shader "UdonCL/device_gpu_242__gather" {}', encoding="utf-8"
    )
    assert gates.shader_conflicts(project, artifact) == ["UdonCL/device_gpu_242__gather"]

    # Artifacts come from an artifact directory or a build's target.
    assert gates.probe_artifact(artifact, "conversation") == artifact.resolve()
    probe_build = temporary / "probe-build"
    probe_build.mkdir()
    gates.write_manifest(
        probe_build, {"kind": "build", "targets": {"conversation": {"artifact": "ex/a"}}}
    )
    assert gates.probe_artifact(probe_build, "conversation") == probe_build.resolve() / "ex/a"
    assert "no chat artifact" in refused(gates.probe_artifact, probe_build, "chat")
    with contextlib.redirect_stderr(io.StringIO()) as stderr:
        code = gates.main([
            "probe", "--artifacts", str(artifact), "--plan", str(flat), "--out", str(temporary / "never"),
        ])
    assert code == 2 and "*.unity.json" in stderr.getvalue(), stderr.getvalue()
    assert not (temporary / "never").exists()

# The probe source keeps culture-invariant JSON numbers (an interpolated
# {ms:R}}} once wrote a literal R into the result file).
probe_source = (ROOT / "scripts/tinyllama_gpu_probe/UdonclGpuKernelTimingProbe.cs.txt").read_text(encoding="utf-8")
assert ":R}" not in probe_source and "CultureInfo.InvariantCulture" in probe_source
assert "public void MeasureDecodeRunKernels()" in probe_source

print("tinyllama_gates=passed")
