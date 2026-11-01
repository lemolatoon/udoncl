"""Generate fresh UdonAsm lowering and execute it with Unity's SDK IUdonVM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


TEST_FILTER = "Lemolatoon.Udoncl.Editor.Tests.UdonVmLoweringLifecycleTests"


def executable(path: Path) -> Path:
    candidate = path.with_suffix(".exe") if sys.platform == "win32" else path
    if not candidate.is_file():
        raise SystemExit(f"required compiler tool does not exist: {candidate}")
    return candidate


def run(command, **kwargs):
    print("+", subprocess.list2cmdline([str(item) for item in command]), flush=True)
    return subprocess.run([str(item) for item in command], check=True, **kwargs)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=root / "build/udoncl")
    parser.add_argument("--output", type=Path, default=root / "build/unity-ci")
    args = parser.parse_args()
    unity = args.unity.resolve()
    project = args.project.resolve()
    build = args.build_dir.resolve()
    output = args.output.resolve()
    manifest_path = project / "Packages/manifest.json"
    lock_path = project / "Packages/packages-lock.json"
    if not unity.is_file():
        parser.error(f"Unity executable does not exist: {unity}")
    if not manifest_path.is_file():
        parser.error(f"Unity project has no Packages/manifest.json: {project}")
    output.mkdir(parents=True, exist_ok=True)
    lowering = output / "lowering"
    run(
        [
            sys.executable,
            root / "tests/codegen/udon.py",
            executable(build / "tools/vrc-opt/vrc-opt"),
            executable(build / "tools/vrc-translate/vrc-translate"),
            lowering,
        ]
    )
    assembly = (lowering / "scalar.uasm").resolve()
    results = (output / "udonvm-lowering-results.xml").resolve()
    log = (output / "unity.log").resolve()
    manifest_original = manifest_path.read_bytes()
    lock_original = lock_path.read_bytes() if lock_path.is_file() else None
    manifest = json.loads(manifest_original.decode("utf-8"))
    dependencies = manifest.setdefault("dependencies", {})
    dependencies["com.lemolatoon.udoncl"] = (
        "file:" + (root / "unity/Packages/com.lemolatoon.udoncl").resolve().as_posix()
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["UDONCL_LOWERED_UASM"] = str(assembly)
    try:
        run(
            [
                unity,
                "-batchmode",
                "-nographics",
                "-projectPath",
                project,
                "-runTests",
                "-testPlatform",
                "EditMode",
                "-testFilter",
                TEST_FILTER,
                "-testResults",
                results,
                "-logFile",
                log,
            ],
            env=environment,
            timeout=1200,
        )
    finally:
        manifest_path.write_bytes(manifest_original)
        if lock_original is None:
            lock_path.unlink(missing_ok=True)
        else:
            lock_path.write_bytes(lock_original)
    if not results.is_file():
        raise SystemExit(f"Unity did not write test results; inspect {log}")
    test_run = ET.parse(results).getroot()
    failed = int(test_run.attrib.get("failed", "-1"))
    passed = int(test_run.attrib.get("passed", "0"))
    result = test_run.attrib.get("result", "")
    if failed != 0 or passed != 6 or result not in ("Passed", "Success"):
        raise SystemExit(
            f"Unity lowering fixture result={result} passed={passed} failed={failed}; inspect {results}"
        )
    leaked = project / "Library/UdonclLoweringIntegration"
    if leaked.exists():
        raise SystemExit(f"Unity lowering fixture leaked its setup directory: {leaked}")
    print(f"unity_udonvm_lowering=passed tests={passed} cleanup=passed results={results}")


if __name__ == "__main__":
    main()
