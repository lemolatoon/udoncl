from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(sys.argv[1]).resolve()
SPEC = importlib.util.spec_from_file_location(
    "validate_tinyllama_conversation_artifact",
    ROOT / "scripts/validate_tinyllama_conversation_artifact.py",
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


def tensor(index, name, data_type, dimensions, logical_bytes, capacity_bytes=None):
    return {
        "index": index, "name": name, "data_type": data_type,
        "dimensions": dimensions, "logical_bytes": logical_bytes,
        "capacity_bytes": logical_bytes if capacity_bytes is None else capacity_bytes,
    }


def export(public_name, symbol, parameter_kinds=(), is_async=False):
    return {
        "async": is_async,
        "c_symbol": symbol,
        "parameters": [
            {"index": index, "kind": kind}
            for index, kind in enumerate(parameter_kinds)
        ],
        "public_name": public_name,
        "result": {"kind": "i32"},
    }


class TinyLlamaConversationArtifactValidatorTests(unittest.TestCase):
    def make_artifact(self, root: Path) -> Path:
        inputs = [
            tensor(0, "input_ids", "int64", [1, 1], 8),
            tensor(1, "attention_mask", "int64", [1, 64], 512),
            tensor(2, "position_ids", "int64", [1, 1], 8),
        ]
        for layer in range(22):
            for kind in ("key", "value"):
                inputs.append(tensor(len(inputs),
                                     f"past_key_values.{layer}.{kind}",
                                     "float32", [1, 4, 63, 64], 64512))
        outputs = [tensor(0, "logits", "float32", [1, 1, 32000],
                          128000, 131072)]
        for layer in range(22):
            for kind in ("key", "value"):
                outputs.append(tensor(len(outputs), f"present.{layer}.{kind}",
                                      "float32", [1, 4, 64, 64], 65536))
        manifest = {
            "exports": [
                export("Initialize", "tinyllama_initialize", is_async=True),
                export("Reset", "tinyllama_reset", is_async=True),
                export("SetPromptTokens", "tinyllama_set_prompt_tokens",
                       ("i32_array", "i32")),
                export("Generate", "tinyllama_generate", ("i32",), True),
                export("GeneratedTokenCount",
                       "tinyllama_generated_token_count"),
                export("ReadGeneratedTokens",
                       "tinyllama_read_generated_tokens", ("i32_array",)),
                export("Shutdown", "tinyllama_dispose"),
            ],
            "onnx_binding": {
                "inputs": inputs,
                "outputs": outputs,
                "physical": {
                    "enqueue_advance_sequence_state_symbol":
                        "udon_onnx_binding_enqueue_advance_sequence_state",
                    "enqueue_reset_sequence_state_symbol":
                        "udon_onnx_binding_enqueue_reset_sequence_state",
                },
            },
            "device": {
                "buffers": [
                    {"id": index, "name": f"private_{index}"}
                    for index in range(1, 127)
                ],
                "kernels": [{"id": 247}, {"id": 248}],
                "mutable_handle_capacity": 126,
            },
            "program": {
                "name": "tinyllama_real_conversation",
                "binding_class": "UdonclTinyLlamaConversationBinding",
                "binding_source": "binding.cs",
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "binding.cs").write_text(
            "\n".join((
                "StartInitialize(", "GetInitializeStatus(",
                "TryGetInitializeResult(", "StartReset(", "GetResetStatus(",
                "TryGetResetResult(",
                "TrySetPromptTokens(int[] tokenIds, int tokenCount, out int result)",
                "StartGenerate(int maxNewTokens, UdonSharpBehaviour receiver,",
                "GetGenerateStatus(", "TryGetGenerateResult(",
                "TryGeneratedTokenCount(out int result)",
                "TryReadGeneratedTokens(int[] destination, out int result)",
                "TryShutdown(out int result)",
            )) + "\n", encoding="utf-8")
        return root

    def test_accepts_private_m1_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = VALIDATOR.validate_artifact(
                self.make_artifact(Path(directory)))
            self.assertEqual(summary["inputs"], 47)
            self.assertEqual(summary["outputs"], 45)

    def test_rejects_fixed_prefill_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.make_artifact(Path(directory))
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["onnx_binding"]["inputs"][0]["dimensions"] = [1, 4]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input_ids: invalid dimensions"):
                VALIDATOR.validate_artifact(artifact)

    def test_rejects_public_tensor_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.make_artifact(Path(directory))
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["device"]["buffers"][0]["public_access"] = "input"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must remain private"):
                VALIDATOR.validate_artifact(artifact)

    def test_rejects_missing_sequence_advance_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.make_artifact(Path(directory))
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["onnx_binding"]["physical"][
                "enqueue_advance_sequence_state_symbol"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sequence-state advance helper"):
                VALIDATOR.validate_artifact(artifact)

    def test_rejects_missing_sequence_reset_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.make_artifact(Path(directory))
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["onnx_binding"]["physical"][
                "enqueue_reset_sequence_state_symbol"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sequence-state reset helper"):
                VALIDATOR.validate_artifact(artifact)

    def test_rejects_old_three_export_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.make_artifact(Path(directory))
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["exports"] = [
                export("Initialize", "tinyllama_conversation_initialize",
                       is_async=True),
                export("Run", "tinyllama_conversation_run", is_async=True),
                export("Shutdown", "tinyllama_conversation_dispose"),
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected public API"):
                VALIDATOR.validate_artifact(artifact)

    def test_rejects_missing_device_argmax(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.make_artifact(Path(directory))
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["device"]["kernels"] = [{"id": 247}]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ArgMax kernels"):
                VALIDATOR.validate_artifact(artifact)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
