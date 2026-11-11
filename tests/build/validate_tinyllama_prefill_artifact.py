from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(sys.argv[1]).resolve()
SPEC = importlib.util.spec_from_file_location(
    "validate_tinyllama_prefill_artifact",
    ROOT / "scripts/validate_tinyllama_prefill_artifact.py",
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


def tensor(index, name, data_type, dimensions, logical_bytes, capacity_bytes):
    return {
        "index": index, "name": name, "data_type": data_type,
        "dimensions": dimensions, "logical_bytes": logical_bytes,
        "capacity_bytes": capacity_bytes,
    }


class TinyLlamaPrefillArtifactValidatorTests(unittest.TestCase):
    def make_artifact(self, root: Path) -> Path:
        inputs = [
            tensor(0, "input_ids", "int64", [1, 4], 32, 32),
            tensor(1, "attention_mask", "int64", [1, 4], 32, 32),
            tensor(2, "position_ids", "int64", [1, 4], 32, 32),
        ]
        for layer in range(22):
            for kind in ("key", "value"):
                inputs.append(tensor(len(inputs),
                                     f"past_key_values.{layer}.{kind}",
                                     "float32", [1, 4, 0, 64], 0, 4))
        outputs = [tensor(0, "logits", "float32", [1, 4, 32000],
                          512000, 524288)]
        for layer in range(22):
            for kind in ("key", "value"):
                outputs.append(tensor(len(outputs), f"present.{layer}.{kind}",
                                      "float32", [1, 4, 4, 64], 4096, 4096))
        buffers = [
            {"public_access": "input", "name": item["name"]}
            for item in inputs
        ] + [
            {"public_access": "output", "name": item["name"]}
            for item in outputs
        ] + [{"name": "private_93"}]
        manifest = {
            "onnx_binding": {"inputs": inputs, "outputs": outputs},
            "device": {"buffers": buffers,
                       "mutable_handle_capacity": len(buffers)},
            "static_tensors": [{"name": "weight",
                                "byte_length": 900_000_001}],
            "program": {"binding_source": "binding.cs",
                        "storage_report": "storage-report.json"},
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "storage-report.json").write_text(json.dumps({
            "raw_heap_bytes": 128,
            "raw_objects": [{"name": "host_state"}],
        }), encoding="utf-8")
        (root / "binding.cs").write_text("\n".join([
            "TryWriteInputIds(long[] source)",
            "TryWriteAttentionMask(long[] source)",
            "TryWritePositionIds(long[] source)",
            "StartReadLogits(float[] destination",
            "StartReadPresent0Key(float[] destination",
            "StartReadPresent21Value(float[] destination",
        ]), encoding="utf-8")
        return root

    def test_accepts_complete_prefill_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = VALIDATOR.validate_artifact(
                self.make_artifact(Path(directory)))
            self.assertEqual(summary["inputs"], 47)
            self.assertEqual(summary["outputs"], 45)

    def test_rejects_missing_past_input(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.make_artifact(Path(directory))
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["onnx_binding"]["inputs"].pop()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 47 inputs"):
                VALIDATOR.validate_artifact(artifact)

    def test_rejects_static_weight_in_raw_heap(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.make_artifact(Path(directory))
            (artifact / "storage-report.json").write_text(json.dumps({
                "raw_heap_bytes": 128,
                "raw_objects": [{"name": "__udon_onnx_static_0"}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "static ONNX tensor"):
                VALIDATOR.validate_artifact(artifact)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
