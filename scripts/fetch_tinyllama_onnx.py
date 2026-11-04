"""Fetch the pinned TinyLlama Q4 ONNX acceptance model into a verified cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


REPOSITORY = "onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX"
REVISION = "cb463df0db21d713728366575b5a71f80de7adb2"
MODEL_FILE = "onnx/model_q4.onnx"
MODEL_SIZE = 909_750_422
MODEL_SHA256 = "399360b3b9174b505eedfc71a305f48b98207c51fff4c4643c722b7e37f137eb"
OPERATOR_SCHEMA_VERSION = 1
MODEL_URL = (
    f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{MODEL_FILE}"
    "?download=true"
)


def default_cache() -> Path:
    override = os.environ.get("UDONCL_MODEL_CACHE")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1] / ".deps" / "models"


def cached_model(cache: Path) -> Path:
    return cache / "sha256" / MODEL_SHA256 / "model_q4.onnx"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"TinyLlama model is missing: {path}")
    size = path.stat().st_size
    if size != MODEL_SIZE:
        raise RuntimeError(
            f"TinyLlama model size mismatch: expected {MODEL_SIZE}, got {size}: {path}"
        )
    actual = _digest(path)
    if actual != MODEL_SHA256:
        raise RuntimeError(
            f"TinyLlama model SHA-256 mismatch: expected {MODEL_SHA256}, got {actual}: {path}"
        )


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "udoncl-tinyllama-acceptance/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)


def ensure(cache: Path, *, offline: bool = False, url: str = MODEL_URL) -> Path:
    destination = cached_model(cache.resolve())
    if destination.is_file():
        verify(destination)
        return destination
    if offline:
        raise RuntimeError(f"TinyLlama model is not cached: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="model_q4.onnx.", suffix=".part", dir=destination.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        _download(url, temporary)
        verify(temporary)
        os.replace(temporary, destination)
        receipt = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "file": MODEL_FILE,
            "size": MODEL_SIZE,
            "sha256": MODEL_SHA256,
            "operator_schema_version": OPERATOR_SCHEMA_VERSION,
        }
        destination.with_suffix(".json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=default_cache())
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        model = ensure(args.cache, offline=args.offline)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(model)


if __name__ == "__main__":
    main()
