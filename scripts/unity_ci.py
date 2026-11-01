"""Build Unity fixtures and run every UdonCL package Editor test in CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


TEST_ASSEMBLY = "Udoncl.Editor.Tests"


def tree_snapshot(root: Path):
    if not root.exists():
        return {}
    snapshot = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = (
            "directory"
            if path.is_dir()
            else hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return snapshot


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
    generated = project / "Assets/udoncl-generated"
    serialized = project / "Assets/SerializedUdonPrograms"
    generated_before = tree_snapshot(generated)
    serialized_before = tree_snapshot(serialized)
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
    artifact_commands = [
        (
            root / "libraries/udonblas/run_device.py",
            root / "build/udonblas-device-run",
        ),
        (
            root / "libraries/udondnn/run_device.py",
            root / "build/udondnn-device-run",
        ),
        (
            root / "examples/mnist-cnn/run_device.py",
            root / "build/mnist-cnn-device-run",
        ),
        (
            root / "examples/tiny-llama/run_device.py",
            root / "build/tiny-llama-device-run",
        ),
    ]
    for generator, artifact_output in artifact_commands:
        run(
            [
                sys.executable,
                generator,
                "--build-dir",
                build,
                "--output",
                artifact_output,
            ],
            timeout=1200,
        )
    assembly = (lowering / "scalar.uasm").resolve()
    results = (output / "editor-test-results.xml").resolve()
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
                "-force-d3d11",
                "-projectPath",
                project,
                "-runTests",
                "-testPlatform",
                "EditMode",
                "-assemblyNames",
                TEST_ASSEMBLY,
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
    total = int(test_run.attrib.get("total", "-1"))
    failed = int(test_run.attrib.get("failed", "-1"))
    passed = int(test_run.attrib.get("passed", "0"))
    skipped = int(test_run.attrib.get("skipped", "0"))
    inconclusive = int(test_run.attrib.get("inconclusive", "0"))
    result = test_run.attrib.get("result", "")
    if (
        total <= 0
        or failed != 0
        or skipped != 0
        or inconclusive != 0
        or passed != total
        or result not in ("Passed", "Success")
    ):
        raise SystemExit(
            "Unity Editor tests "
            f"result={result} total={total} passed={passed} failed={failed} "
            f"skipped={skipped} inconclusive={inconclusive}; inspect {results}"
        )
    leaked = project / "Library/UdonclLoweringIntegration"
    if leaked.exists():
        raise SystemExit(f"Unity lowering fixture leaked its setup directory: {leaked}")
    generated_after = tree_snapshot(generated)
    serialized_after = tree_snapshot(serialized)
    if generated_after != generated_before or serialized_after != serialized_before:
        raise SystemExit(
            "Unity Editor tests changed lifecycle-managed assets; "
            f"generated entries {len(generated_before)} -> {len(generated_after)}, "
            f"serialized entries {len(serialized_before)} -> {len(serialized_after)}"
        )
    print(f"unity_editor_tests=passed tests={passed} cleanup=passed results={results}")


if __name__ == "__main__":
    main()
