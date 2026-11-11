import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fetch_tinyllama_tokenizer",
    ROOT / "scripts" / "fetch_tinyllama_tokenizer.py",
)
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


class TinyLlamaTokenizerCacheTests(unittest.TestCase):
    def assets(self, payloads):
        return tuple(
            {
                "file": name,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in payloads.items()
        )

    def test_verified_bundle_download_is_atomic_and_reused(self):
        payloads = {
            "tokenizer.json": b'{"model":{"type":"BPE"}}',
            "tokenizer.model": b"sentencepiece-model",
            "tokenizer_config.json": b'{"add_bos_token":true}',
            "special_tokens_map.json": b'{"bos_token":"<s>"}',
        }
        assets = self.assets(payloads)
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            with mock.patch.object(FETCH, "ASSETS", assets), mock.patch.object(
                FETCH, "_download"
            ) as download:
                download.side_effect = (
                    lambda asset, path: path.write_bytes(payloads[asset["file"]])
                )
                first = FETCH.ensure(cache, base_url="https://invalid.test")
                second = FETCH.ensure(cache, offline=True)

            self.assertEqual(first, second)
            self.assertEqual(download.call_count, len(payloads))
            receipt = json.loads((first / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["repository"], FETCH.REPOSITORY)
            self.assertEqual(receipt["revision"], FETCH.REVISION)
            self.assertEqual(
                [item["file"] for item in receipt["assets"]], sorted(payloads)
            )

    def test_corrupt_asset_never_establishes_bundle(self):
        payloads = {
            "tokenizer.json": b"good-json",
            "tokenizer.model": b"good-model",
        }
        assets = self.assets(payloads)
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            with mock.patch.object(FETCH, "ASSETS", assets), mock.patch.object(
                FETCH, "_download"
            ) as download:
                download.side_effect = lambda _asset, path: path.write_bytes(b"bad")
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    FETCH.ensure(cache, base_url="https://invalid.test")
            self.assertFalse(FETCH.cached_bundle(cache.resolve()).exists())

    def test_offline_cache_miss_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "is not cached"):
                FETCH.ensure(Path(temporary), offline=True)


if __name__ == "__main__":
    unittest.main()
