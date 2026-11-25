"""Unit tests of scripts/validate_tinyllama_chat_artifact.py on synthetic
manifests of the examples/tinyllama-chat artifact, and of the Unity fixture
that drives it."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(sys.argv[1]).resolve()
SPEC = importlib.util.spec_from_file_location(
    "validate_tinyllama_chat_artifact",
    ROOT / "scripts/validate_tinyllama_chat_artifact.py",
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)

CAPACITY = 2048


def tensor(index, name, data_type, dimensions, logical_bytes, bounds=None):
    entry = {
        "index": index, "name": name, "data_type": data_type,
        "dimensions": dimensions, "logical_bytes": logical_bytes,
        "capacity_bytes": logical_bytes,
    }
    if bounds is not None:
        entry["dimension_bounds"] = bounds
        entry["dimension_names"] = [
            "past_sequence_length" if bound else "" for bound in bounds
        ]
    return entry


def manifest(capacity=CAPACITY):
    inputs = [
        tensor(0, "input_ids", "int64", [1, 1], 8),
        tensor(1, "attention_mask", "int64", [1, -1], 8 * capacity, [0, capacity]),
        tensor(2, "position_ids", "int64", [1, 1], 8),
    ]
    outputs = [tensor(0, "logits", "float32", [1, 1, 32000], 128000)]
    for layer in range(22):
        for offset, kind in enumerate(("key", "value")):
            inputs.append(tensor(
                3 + layer * 2 + offset, f"past_key_values.{layer}.{kind}", "float32",
                [1, 4, -1, 64], 16 * (capacity - 1) * 64, [0, 0, capacity - 1, 0]))
            outputs.append(tensor(
                1 + layer * 2 + offset, f"present.{layer}.{kind}", "float32",
                [1, 4, -1, 64], 16 * capacity * 64, [0, 0, capacity, 0]))
    exports = []
    for public_name, (symbol, kinds, is_async) in VALIDATOR.EXPORTS.items():
        entry = {
            "public_name": public_name, "c_symbol": symbol, "async": is_async,
            "parameters": [{"index": index, "kind": kind}
                           for index, kind in enumerate(kinds)],
            "result": {"kind": "i32"},
        }
        if public_name in VALIDATOR.ALLOWED_WHILE_BUSY:
            entry["allowed_while_busy"] = True
        exports.append(entry)
    return {
        "program": {"name": "tinyllama_chat",
                    "binding_class": "UdonclTinyllamaChatBinding",
                    "binding_source": "binding.cs"},
        "exports": exports,
        "onnx_binding": {
            "inputs": inputs, "outputs": outputs,
            "in_place_concats": [{"input": 3 + offset, "output": 1 + offset, "axis": 2}
                                 for offset in range(44)],
        },
        "device": {
            "buffers": [{"name": f"buffer{index}"} for index in range(60)],
            "kernels": [
                {"id": 247, "name": "udonaten_argmax_reduce"},
                {"id": 1024, "name": "tinyllama_argmax", "global_size": [1, 1, 1],
                 "args": [{"kind": "buffer", "required_bytes": 128000},
                          {"kind": "buffer", "required_bytes": 4}]},
            ],
        },
    }


BINDING = "\n".join(
    f"public int {method}" for method in VALIDATOR.BINDING_METHODS
) + "\n"


class TinyLlamaChatArtifactValidatorTests(unittest.TestCase):
    def validate(self, data, binding=BINDING, capacity=CAPACITY):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            (artifact / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
            (artifact / "binding.cs").write_text(binding, encoding="utf-8")
            return VALIDATOR.validate_artifact(artifact, capacity)

    def rejects(self, mutate, message, binding=BINDING):
        data = copy.deepcopy(manifest())
        mutate(data)
        with self.assertRaisesRegex(ValueError, message):
            self.validate(data, binding)

    def test_accepts_the_example_artifact(self):
        summary = self.validate(manifest())
        self.assertEqual(summary["argmax_id"], 1024)
        self.assertEqual(summary["buffers"], 60)

    def test_capacity_is_the_bound_of_the_past(self):
        self.validate(manifest(512), capacity=512)
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            self.validate(manifest(512))

    def test_rejects_another_program(self):
        self.rejects(lambda data: data["program"].update(name="tinyllama_real_chat"),
                     "unexpected program")

    def test_rejects_an_extra_or_missing_export(self):
        self.rejects(lambda data: data["exports"].pop(), "unexpected public API")
        self.rejects(lambda data: data["exports"].append(
            dict(data["exports"][0], public_name="Reset")), "unexpected public API")

    def test_only_read_streamed_text_runs_while_busy(self):
        def busy(data):
            for item in data["exports"]:
                item["allowed_while_busy"] = True
        self.rejects(busy, "invalid busy contract")

        def not_busy(data):
            for item in data["exports"]:
                item.pop("allowed_while_busy", None)
        self.rejects(not_busy, "ReadStreamedText: invalid busy contract")

    def test_generate_and_initialize_are_async(self):
        def sync(data):
            for item in data["exports"]:
                if item["public_name"] == "Generate":
                    item["async"] = False
        self.rejects(sync, "Generate: invalid async contract")

    def test_rejects_public_buffers(self):
        self.rejects(lambda data: data["device"]["buffers"][0].update(public_access="read"),
                     "no public buffers")

    def test_requires_the_user_argmax_kernel(self):
        self.rejects(lambda data: data["device"]["kernels"].pop(), "tinyllama_argmax")
        self.rejects(lambda data: data["device"]["kernels"][1].update(id=248),
                     "not a user kernel id")
        self.rejects(lambda data: data["device"]["kernels"][1].update(global_size=[32, 1, 1]),
                     "invalid NDRange")

    def test_requires_in_place_concat(self):
        self.rejects(lambda data: data["onnx_binding"].update(in_place_concats=[]),
                     "in-place Concat")

    def test_requires_the_binding_api(self):
        with self.assertRaisesRegex(ValueError, "TryBeginMessage"):
            self.validate(manifest(), BINDING.replace("TryBeginMessage", "TryBegin"))
        with self.assertRaisesRegex(ValueError, "StartRun"):
            self.validate(manifest(), BINDING + "public int StartRun()\n")

    def test_controller_uses_only_the_generated_chat_api(self):
        source = (
            ROOT / "unity/Packages/com.lemolatoon.udoncl/Editor/Examples"
            / "tinyllama-minimal-chat-test-controller.cs.txt"
        ).read_text(encoding="utf-8")
        for required in (
            "UdonclTinyllamaChatBinding binding",
            "StartInitialize(", "TryBeginMessage(", "TryContinueMessage(",
            "StartGenerate(", "TryReadStreamedText(", "TryReadGeneratedTokens(",
            "TryDispose(", "GetLastAsyncDiagnostic(",
        ):
            self.assertIn(required, source)
        test = (
            ROOT / "unity/Packages/com.lemolatoon.udoncl/Editor/Examples"
            / "tinyllama-minimal-chat-lifecycle-test.cs.txt"
        ).read_text(encoding="utf-8")
        self.assertIn('"Assets/udoncl-generated/tinyllama_chat"', test)
        self.assertIn("UDONCL_PUBLIC_TINYLLAMA_MINIMAL_CHAT_ORACLE", test)
        # The focused test assembly reads Editor internals, and the gate
        # prepares and runs it.
        editor = ROOT / "unity/Packages/com.lemolatoon.udoncl/Editor"
        self.assertIn('InternalsVisibleTo("Udoncl.TinyLlama.MinimalChat.Focused.Tests")',
                      (editor / "AssemblyInfo.cs").read_text(encoding="utf-8"))
        importer = (editor / "UdonclPackagedExampleImporter.cs").read_text(encoding="utf-8")
        precommit = (ROOT / "scripts/precommit.py").read_text(encoding="utf-8")
        for source in (importer, precommit):
            self.assertIn("Prepare Public TinyLlama Minimal Chat Validation", source)
        self.assertIn('"Udoncl.TinyLlama.MinimalChat.Focused.Tests"', precommit)
        self.assertIn("GeneratedUSharpBindingChatsAndMatchesOracleTokens", precommit)
        self.assertIn("IEnumerator GeneratedUSharpBindingChatsAndMatchesOracleTokens()", test)
        for name in ("test-controller", "lifecycle-test"):
            self.assertTrue((editor / f"Examples/tinyllama-minimal-chat-{name}.cs.txt.meta").is_file())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
