"""Fetch the pinned TinyLlama tokenizer assets into a verified atomic cache."""

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
ASSETS = (
    {
        "file": "special_tokens_map.json",
        "size": 551,
        "sha256": "82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db",
    },
    {
        "file": "tokenizer.json",
        "size": 1_842_795,
        "sha256": "bf467c9e0f536bda271283c6ef85eb1a943e3196b621c8a912d64953b205df83",
    },
    {
        "file": "tokenizer.model",
        "size": 499_723,
        "sha256": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    },
    {
        "file": "tokenizer_config.json",
        "size": 1_368,
        "sha256": "c1cda9389292eb526c0064edc3b68287fb1096ac161871375dd07df1f904c337",
    },
)
DEFAULT_BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}"


def default_cache() -> Path:
    override = os.environ.get("UDONCL_MODEL_CACHE")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1] / ".deps" / "models"


def _canonical_assets() -> list[dict[str, object]]:
    return sorted((dict(asset) for asset in ASSETS), key=lambda asset: str(asset["file"]))


def bundle_sha256() -> str:
    description = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "assets": _canonical_assets(),
    }
    payload = json.dumps(
        description, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cached_bundle(cache: Path) -> Path:
    return cache / "tokenizers" / "sha256" / bundle_sha256()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_asset(root: Path, asset: dict[str, object]) -> None:
    path = root / str(asset["file"])
    if not path.is_file():
        raise RuntimeError(f"TinyLlama tokenizer asset is missing: {path}")
    expected_size = int(asset["size"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            "TinyLlama tokenizer asset size mismatch: "
            f"expected {expected_size}, got {actual_size}: {path}"
        )
    expected_digest = str(asset["sha256"])
    actual_digest = _digest(path)
    if actual_digest != expected_digest:
        raise RuntimeError(
            "TinyLlama tokenizer asset SHA-256 mismatch: "
            f"expected {expected_digest}, got {actual_digest}: {path}"
        )


def verify(root: Path, *, require_receipt: bool = True) -> None:
    for asset in _canonical_assets():
        _verify_asset(root, asset)
    if require_receipt:
        receipt_path = root / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"TinyLlama tokenizer receipt is missing: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "bundle_sha256": bundle_sha256(),
            "assets": _canonical_assets(),
        }
        if receipt != expected:
            raise RuntimeError(f"TinyLlama tokenizer receipt mismatch: {receipt_path}")


def _download(asset: dict[str, object], destination: Path) -> None:
    request = urllib.request.Request(
        str(asset["url"]),
        headers={"User-Agent": "udoncl-tinyllama-tokenizer/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)


def ensure(
    cache: Path,
    *,
    offline: bool = False,
    base_url: str = DEFAULT_BASE_URL,
) -> Path:
    destination = cached_bundle(cache.resolve())
    if destination.exists():
        verify(destination)
        return destination
    if offline:
        raise RuntimeError(f"TinyLlama tokenizer is not cached: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="tokenizer.", dir=destination.parent))
    try:
        for expected in _canonical_assets():
            downloadable = dict(expected)
            downloadable["url"] = f"{base_url.rstrip('/')}/{expected['file']}?download=true"
            _download(downloadable, temporary / str(expected["file"]))
            _verify_asset(temporary, expected)
        receipt = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "bundle_sha256": bundle_sha256(),
            "assets": _canonical_assets(),
        }
        (temporary / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=default_cache())
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        bundle = ensure(args.cache, offline=args.offline)
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        parser.exit(1, f"{error}\n")
    print(bundle)


if __name__ == "__main__":
    main()
