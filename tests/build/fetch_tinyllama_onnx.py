import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fetch_tinyllama_onnx", ROOT / "scripts" / "fetch_tinyllama_onnx.py"
)
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


class TinyLlamaModelCacheTests(unittest.TestCase):
    def test_verified_download_is_atomic_and_reused(self):
        payload = b"pinned-tinyllama-model"
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            with mock.patch.object(FETCH, "MODEL_SIZE", len(payload)), mock.patch.object(
                FETCH, "MODEL_SHA256", hashlib.sha256(payload).hexdigest()
            ), mock.patch.object(FETCH, "_download") as download:
                download.side_effect = lambda _url, path: path.write_bytes(payload)
                first = FETCH.ensure(cache, url="https://invalid.test/model.onnx")
                second = FETCH.ensure(cache, offline=True)
            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), payload)
            self.assertEqual(download.call_count, 1)
            self.assertTrue(first.with_suffix(".json").is_file())

    def test_invalid_download_is_rejected_without_cache_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            with mock.patch.object(FETCH, "MODEL_SIZE", 4), mock.patch.object(
                FETCH, "MODEL_SHA256", hashlib.sha256(b"good").hexdigest()
            ), mock.patch.object(FETCH, "_download") as download:
                download.side_effect = lambda _url, path: path.write_bytes(b"evil")
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    FETCH.ensure(cache, url="https://invalid.test/model.onnx")
            self.assertFalse(FETCH.cached_model(cache.resolve()).exists())

    def test_offline_cache_miss_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "is not cached"):
                FETCH.ensure(Path(temporary), offline=True)


if __name__ == "__main__":
    unittest.main()
