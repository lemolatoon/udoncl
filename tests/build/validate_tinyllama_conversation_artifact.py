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
    def test_controller_uses_only_generated_conversation_api(self):
        source = (
            ROOT
            / "unity/Packages/com.lemolatoon.udoncl/Editor/Examples"
            / "tinyllama-conversation-test-controller.cs.txt"
        ).read_text(encoding="utf-8")
        for required in (
            "StartInitialize(",
            "GetLastAsyncDiagnostic(",
            "TrySetPromptTokens(",
            "StartGenerate(",
            "TryGeneratedTokenCount(",
            "TryReadGeneratedTokens(",
            "StartReset(",
            "GetResetStatus(",
            "TryGetResetResult(",
            "TryShutdown(",
            "LibraryStatusInvalidState = 12",
            "LibraryStatusDisposed = 8",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "IUdonVM",
            "IUdonHeap",
            "GetProgramVariable",
            "SetProgramVariable",
            "RenderTexture",
            "__vrc_",
            "__udoncl_",
            "udon_onnx_binding_",
        ):
            self.assertNotIn(forbidden, source)

    def test_lifecycle_test_checks_whole_greedy_sequence(self):
        examples = ROOT / "unity/Packages/com.lemolatoon.udoncl/Editor/Examples"
        source = (
            examples / "tinyllama-conversation-lifecycle-test.cs.txt"
        ).read_text(encoding="utf-8")
        controller = (
            examples / "tinyllama-conversation-test-controller.cs.txt"
        ).read_text(encoding="utf-8")
        precommit = (ROOT / "scripts/precommit.py").read_text(encoding="utf-8")
        name = "GeneratedUSharpBindingRunsGreedyConversationLifecycle"
        self.assertIn(f"IEnumerator {name}()", source)
        self.assertIn(f'"{name}"', precommit)
        self.assertNotIn("GeneratesFirstRealToken", source + precommit)
        for required in (
            # The request limit is the whole oracle sequence, and both the
            # first and the post-Reset generation must match all of it.
            "int maxNewTokens = expectedTokens.Length;",
            "CollectionAssert.AreEqual(expectedTokens, Prefix(first, firstCount)",
            "Prefix(second, secondCount)",
            "rejectedPromptStatus",
            "disposedPromptStatus",
            # A VM halt is reported with the controller phase.
            "Application.logMessageReceived += CaptureVmError;",
            "Udon VM halted during ",
            # Phase/frame progress and the final state reach the Editor log.
            'Debug.Log("udoncl tinyllama_conversation result " + state);',
            "udoncl tinyllama_conversation phase={0} ({1}) frame={2}",
            # Every prerequisite fails with a message, not a bare NRE.
            "were not restored after the play-mode domain reload",
            "The conversation test controller program is not initialized.",
            "is unset.",
        ):
            self.assertIn(required, source)
        # EnterPlayMode reloads the domain and the Unity Test Framework
        # restores only the iterator state number. A closure's display object
        # is allocated at iterator entry and would be null after the reload.
        iterator = source.split(f"IEnumerator {name}()", 1)[1].split(
            "private void CaptureVmError", 1)[0]
        self.assertNotIn("=>", iterator)
        self.assertNotIn("delegate", iterator)
        self.assertNotIn("(int)UdonclArtifactImporter.GetUdonProgramVariable", source)
        self.assertNotIn("generated_tokens[0]", source)
        self.assertNotIn("TestContext.Progress", source)
        # The Editor test mirrors the controller's phase numbering.
        self.assertIn("private const int PhaseDone = 12;", source)
        self.assertIn("public const int PhaseDone = 12;", controller)

    def test_m1_step_validation_is_model_independent_and_wired(self):
        package = ROOT / "unity/Packages/com.lemolatoon.udoncl/Editor"
        source = (
            package / "Examples/tinyllama-step-lifecycle-test.cs.txt"
        ).read_text(encoding="utf-8")
        controller = (
            package / "Examples/tinyllama-step-test-controller.cs.txt"
        ).read_text(encoding="utf-8")
        importer = (package / "UdonclPackagedExampleImporter.cs").read_text(
            encoding="utf-8")
        assembly = (package / "AssemblyInfo.cs").read_text(encoding="utf-8")
        precommit = (ROOT / "scripts/precommit.py").read_text(encoding="utf-8")
        oracle = (ROOT / "tests/boundary/tinyllama_real_onnx.py").read_text(
            encoding="utf-8")
        cmake = (ROOT / "examples/onnx-transformer-tiny/CMakeLists.txt").read_text(
            encoding="utf-8")
        name = "GeneratedUSharpBindingMatchesM1StepOutputs"
        self.assertIn(f"IEnumerator {name}()", source)
        self.assertIn(f'"{name}"', precommit)
        self.assertIn("Prepare Public TinyLlama M1 Step Validation", importer)
        self.assertIn("Prepare Public TinyLlama M1 Step Validation", precommit)
        self.assertIn("Udoncl.TinyLlama.Step.Focused.Tests", assembly)
        self.assertIn("onnx-transformer-tiny-m1-step", cmake)
        self.assertIn('"m1-steps"', oracle)
        # Every sidecar output is compared and reported by name with its
        # maximum error; inputs and outputs follow manifest order generically.
        for required in (
            "UDONCL_PUBLIC_TINYLLAMA_STEP_UNITY_BUNDLE",
            "UDONCL_PUBLIC_TINYLLAMA_STEP_ORACLES",
            "for (int index = 0; index < oracle.Outputs.Count; ++index)",
            " max_abs_error=",
            'Debug.Log("udoncl tinyllama_m1_step " + line);',
        ):
            self.assertIn(required, source)
        iterator = source.split(f"IEnumerator {name}()", 1)[1].split(
            "private static string CompareOutput", 1)[0]
        self.assertNotIn("=>", iterator)
        self.assertNotIn("delegate", iterator)
        for required in (
            "TryWriteLongBuffer(",
            "TryWriteFloatBuffer(",
            "StartReadFloatBuffer(",
            "SendCustomEventDelayedFrames(nameof(_WriteNextInput), 1);",
        ):
            self.assertIn(required, controller)
        for forbidden in ("GetProgramVariable", "SetProgramVariable", "__vrc_",
                          "__udoncl_", "32000", "1701"):
            self.assertNotIn(forbidden, controller)

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
