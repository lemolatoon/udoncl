import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


root = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location(
    "tinyllama_focused", root / "scripts/tinyllama_focused.py"
)
focused = importlib.util.module_from_spec(spec)
spec.loader.exec_module(focused)


def exercise(case):
    commands = []
    with tempfile.TemporaryDirectory(prefix=f"tinyllama-{case}-") as temporary:
        fake_root = Path(temporary)
        build = fake_root / "build/udoncl"
        build.mkdir(parents=True)
        (build / "CMakeCache.txt").write_text(
            "CMAKE_MAKE_PROGRAM:FILEPATH=/tools/ninja\n", encoding="utf-8"
        )
        oracle = build / "tests/tinyllama-real-onnx" / (
            "oracle.json" if case == "prefill" else "m1-oracle.json"
        )
        oracle.parent.mkdir(parents=True)
        oracle.write_text("{}", encoding="utf-8")

        def fake_run(command, **kwargs):
            command = [str(item) for item in command]
            commands.append(command)
            if command[1:2] == [str(fake_root / "scripts/fetch_tinyllama_onnx.py")]:
                return SimpleNamespace(stdout=str(fake_root / "model.onnx") + "\n")
            if "--target" in command:
                target = command[command.index("--target") + 1]
                artifact = fake_root / "build" / f"tinyllama-focused-{case}"
                artifact = artifact / "udoncl" / target / "artifact"
                artifact.mkdir(parents=True)
                (artifact / "manifest.json").write_text("{}", encoding="utf-8")
            return SimpleNamespace(stdout="")

        original = focused.run
        focused.run = fake_run
        try:
            result = focused.prepare(case, fake_root, build)
        finally:
            focused.run = original

    assert Path(commands[0][1]).parts[-2:] == ("scripts", "build.py"), commands[0]
    assert commands[1][0] == "cmake" and commands[1][1] == "--install"
    configure = next(command for command in commands if "-S" in command)
    assert any(item.startswith("-DUDONCLC_EXECUTABLE=") for item in configure)
    target = "onnx-transformer-tiny" + (
        "" if case == "prefill" else "-conversation"
    )
    assert any("--target" in command and target in command for command in commands)
    assert result["schema_version"] == 1 and result["case"] == case


exercise("prefill")
exercise("conversation")

calls = []


def fail_first(command, **kwargs):
    calls.append(command)
    raise subprocess.CalledProcessError(1, command)


original = focused.run
focused.run = fail_first
try:
    try:
        focused.prepare("prefill", root, root / "build/udoncl")
        raise AssertionError("build failure was not propagated")
    except subprocess.CalledProcessError:
        pass
finally:
    focused.run = original
assert len(calls) == 1
print("tinyllama_focused_driver=passed cases=2")
