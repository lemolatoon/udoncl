"""Build deterministic package archives for the importable user examples."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_EXAMPLES = ROOT / "unity/Packages/com.lemolatoon.udoncl/Editor/Examples"
DEFAULT_ARENA = ROOT / "build/precommit/onnx-aot/unity-arena-add-model-bundle"
DEFAULT_TINY = (
    ROOT / "build/precommit/onnx-aot/unity-tiny-llama-standard-model-bundle"
)
DEFAULT_HELLO = (
    ROOT
    / "build/udoncl/tests/cmake-package/consumer/udoncl/hello-dispatch-unity-library/artifact"
)
DEFAULT_SAXPY = (
    ROOT
    / "build/udoncl/tests/cmake-package/saxpy-consumer/udoncl/saxpy-unity-library/artifact"
)
DEFAULT_POOLING = (
    ROOT
    / "build/udoncl/tests/cmake-package/average-pooling-consumer/udoncl/average-pooling-unity-library/artifact"
)


def require_artifact(source: Path) -> None:
    if not (source / "manifest.json").is_file():
        raise SystemExit(
            f"example artifact is missing: {source}; run python scripts/precommit.py "
            "once to generate the source fixtures"
        )


def archive_bytes(source: Path, program_name: str) -> bytes:
    require_artifact(source)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            if path.is_symlink():
                raise SystemExit(f"example artifact contains a symlink: {path}")
            relative = path.relative_to(source).as_posix()
            data = path.read_bytes()
            if relative == "manifest.json":
                manifest = json.loads(data.decode("utf-8"))
                manifest["program"]["name"] = program_name
                data = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_STORED)
    return output.getvalue()


def publish(source: Path, destination: Path, program_name: str, check: bool) -> None:
    expected = archive_bytes(source.resolve(), program_name)
    if check:
        if not destination.is_file() or destination.read_bytes() != expected:
            raise SystemExit(
                f"packaged example is stale: {destination}; run "
                "python scripts/package_example_artifacts.py"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", type=Path, default=DEFAULT_ARENA)
    parser.add_argument("--tiny", type=Path, default=DEFAULT_TINY)
    parser.add_argument("--hello", type=Path, default=DEFAULT_HELLO)
    parser.add_argument("--saxpy", type=Path, default=DEFAULT_SAXPY)
    parser.add_argument("--pooling", type=Path, default=DEFAULT_POOLING)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    publish(
        args.arena,
        PACKAGE_EXAMPLES / "onnx-arena-add.zip.bytes",
        "onnx_arena_add_example",
        args.check,
    )
    publish(
        args.tiny,
        PACKAGE_EXAMPLES / "onnx-tiny-llama.zip.bytes",
        "onnx_tiny_llama_example",
        args.check,
    )
    publish(
        args.hello,
        PACKAGE_EXAMPLES / "hello-dispatch.zip.bytes",
        "hello_dispatch",
        args.check,
    )
    publish(
        args.saxpy,
        PACKAGE_EXAMPLES / "saxpy.zip.bytes",
        "saxpy_demo",
        args.check,
    )
    publish(
        args.pooling,
        PACKAGE_EXAMPLES / "average-pooling.zip.bytes",
        "average_pooling",
        args.check,
    )
    print("package_examples=valid" if args.check else "package_examples=updated")


if __name__ == "__main__":
    main()
