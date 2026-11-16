"""Shared orchestration for independently runnable real TinyLlama tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


BUILD_TIMEOUT_SECONDS = 3600


def run(command, **kwargs):
    print("+", subprocess.list2cmdline([str(item) for item in command]), flush=True)
    return subprocess.run([str(item) for item in command], check=True, **kwargs)


def prepare(case: str, root: Path, build: Path) -> dict[str, str]:
    # This must remain the first subprocess: a focused consumer may never use a
    # compiler left over from an earlier main-tree build.
    run([sys.executable, root / "scripts/build.py", "--build-dir", build], cwd=root)

    sdk = root / "build/tinyllama-focused-sdk"
    run(
        [
            "cmake",
            "--install",
            build,
            "--prefix",
            sdk,
            "--component",
            "UdonCLCompiler",
        ],
        cwd=root,
    )
    fetched = run(
        [sys.executable, root / "scripts/fetch_tinyllama_onnx.py"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    model = Path(fetched.stdout.strip().splitlines()[-1]).resolve()
    consumer = root / f"build/tinyllama-focused-{case}"
    suffix = ".exe" if sys.platform == "win32" else ""
    llvm = root / ".deps/llvm23/bin"
    onnx_mlir = root / ".deps/onnx-mlir/bin"
    run(
        [
            "cmake",
            "-S",
            root / "examples/onnx-transformer-tiny",
            "-B",
            consumer,
            "-G",
            "Ninja",
            f"-DCMAKE_PREFIX_PATH={sdk.resolve()}",
            f"-DUDONCLC_EXECUTABLE={sdk / 'bin' / ('udonclc' + suffix)}",
            f"-DUDONCL_CLANG_EXECUTABLE={llvm / ('clang' + suffix)}",
            f"-DUDONCL_LLVM_LINK_EXECUTABLE={llvm / ('llvm-link' + suffix)}",
            f"-DUDONCL_ONNX_IMPORTER_EXECUTABLE={onnx_mlir / ('udoncl-onnx' + suffix)}",
            f"-DUDONCL_ONNX_OPT_EXECUTABLE={onnx_mlir / ('udoncl-onnx-opt' + suffix)}",
            f"-DUDONCL_TINYLLAMA_ONNX_MODEL={model}",
        ],
        cwd=root,
    )
    target = (
        "onnx-transformer-tiny"
        if case == "prefill"
        else "onnx-transformer-tiny-conversation"
    )
    run(
        [
            "cmake",
            "--build",
            consumer,
            "--target",
            target,
            "--parallel",
            str(os.cpu_count() or 1),
        ],
        cwd=root,
        timeout=BUILD_TIMEOUT_SECONDS,
    )
    artifact = (consumer / "udoncl" / target / "artifact").resolve()
    oracle_root = (build / "tests/tinyllama-real-onnx").resolve()
    oracle = oracle_root / ("oracle.json" if case == "prefill" else "m1-oracle.json")
    if not (artifact / "manifest.json").is_file():
        raise RuntimeError(f"TinyLlama {case} artifact is missing: {artifact}")
    if not oracle.is_file():
        oracle_root.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                root / "tests/boundary/tinyllama_real_onnx.py",
                root,
                oracle_root,
            ],
            cwd=root,
            timeout=1200,
        )
    return {
        "schema_version": 1,
        "case": case,
        "artifact": str(artifact),
        "oracle": str(oracle),
    }


def main(case: str) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=f"Build and run the real TinyLlama {case} focused test"
    )
    parser.add_argument("--build-dir", type=Path, default=root / "build/udoncl")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    result = prepare(case, root, args.build_dir.resolve())
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result) + "\n", encoding="utf-8")
    if args.prepare_only:
        return
    reuse = (
        "--reuse-tinyllama-prefill-artifact"
        if case == "prefill"
        else "--reuse-tinyllama-conversation-artifact"
    )
    run(
        [
            sys.executable,
            root / "scripts/precommit.py",
            "--unity-only",
            f"--tinyllama-{case}",
            reuse,
            result["artifact"],
        ],
        cwd=root,
    )
