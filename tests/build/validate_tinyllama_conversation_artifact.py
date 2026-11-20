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

    def make_chat_artifact(self, root: Path, capacity: int) -> Path:
        artifact = self.make_artifact(root)
        manifest_path = artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["program"]["name"] = "tinyllama_real_chat"
        manifest["program"]["binding_class"] = "UdonclTinyLlamaChatBinding"
        for symbol, (export_symbol, kinds, is_async) in VALIDATOR.CHAT_EXPORTS.items():
            manifest["exports"].append(export(symbol, export_symbol, kinds, is_async))
            if symbol in VALIDATOR.POLLED_WHILE_GENERATING:
                manifest["exports"][-1]["allowed_while_busy"] = True
        inputs = manifest["onnx_binding"]["inputs"]
        outputs = manifest["onnx_binding"]["outputs"]
        inputs[1].update(dimensions=[1, capacity], logical_bytes=8 * capacity,
                         capacity_bytes=8 * capacity)
        for item in inputs[3:]:
            size = 4 * 4 * (capacity - 1) * 64
            item.update(dimensions=[1, 4, capacity - 1, 64], logical_bytes=size,
                        capacity_bytes=size)
        for item in outputs[1:]:
            size = 4 * 4 * capacity * 64
            item.update(dimensions=[1, 4, capacity, 64], logical_bytes=size,
                        capacity_bytes=size)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        binding = artifact / "binding.cs"
        binding.write_text(binding.read_text(encoding="utf-8") + "\n".join((
            "TrySetChatPrompt(string systemText, string userText, out int result)",
            "TryBeginChatPrompt(out int result)",
            "TryContinueChatPrompt(string systemText, string userText, "
            "int maxWork, out int result)",
            "TryChatPromptPending(out int result)",
            "TryBeginChatTurn(out int result)",
            "TryContinueChatTurn(string userText, int maxWork, out int result)",
            "StartResetChat(", "GetResetChatStatus(", "TryGetResetChatResult(",
            "TryPromptTokenCount(out int result)",
            "TryReadPromptTokens(int[] destination, out int result)",
            "TryGeneratedTextLength(out int result)",
            "TryReadGeneratedText(int[] destination, out int result)",
            "TryStreamedTextLength(out int result)",
            "TryReadStreamedText(int[] destination, int offset, int count, "
            "out int result)",
            "TryLastTokenizerStatus(out int result)",
        )) + "\n", encoding="utf-8")
        return artifact

    def test_accepts_chat_profile_at_a_build_time_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.make_chat_artifact(Path(directory), 128)
            summary = VALIDATOR.validate_artifact(artifact, "chat", 128)
            self.assertEqual(summary["inputs"], 47)
            with self.assertRaisesRegex(ValueError, "attention_mask"):
                VALIDATOR.validate_artifact(artifact, "chat", 64)
            with self.assertRaisesRegex(ValueError, "program identity"):
                VALIDATOR.validate_artifact(artifact, "conversation", 128)

    def test_chat_profile_requires_the_multi_turn_api(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.make_chat_artifact(Path(directory), 64)
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            exports = manifest["exports"]
            manifest["exports"] = [
                item for item in exports if item["public_name"] != "ResetChat"
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected public API"):
                VALIDATOR.validate_artifact(artifact, "chat", 64)
            # ResetChat restarts the sequence state on the Device: async.
            manifest["exports"] = exports
            for item in exports:
                if item["public_name"] == "ResetChat":
                    item["async"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ResetChat: invalid async"):
                VALIDATOR.validate_artifact(artifact, "chat", 64)
            for item in exports:
                if item["public_name"] == "ResetChat":
                    item["async"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            binding = artifact / "binding.cs"
            binding.write_text(binding.read_text(encoding="utf-8").replace(
                "TryContinueChatTurn(", "TryContinueChat("), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "TryContinueChatTurn"):
                VALIDATOR.validate_artifact(artifact, "chat", 64)

    def test_chat_fixture_uses_only_the_generated_text_api(self):
        package = ROOT / "unity/Packages/com.lemolatoon.udoncl/Editor"
        controller = (
            package / "Examples/tinyllama-chat-test-controller.cs.txt"
        ).read_text(encoding="utf-8")
        source = (
            package / "Examples/tinyllama-chat-lifecycle-test.cs.txt"
        ).read_text(encoding="utf-8")
        importer = (package / "UdonclPackagedExampleImporter.cs").read_text(
            encoding="utf-8")
        assembly = (package / "AssemblyInfo.cs").read_text(encoding="utf-8")
        precommit = (ROOT / "scripts/precommit.py").read_text(encoding="utf-8")
        cmake = (ROOT / "examples/onnx-transformer-tiny/CMakeLists.txt").read_text(
            encoding="utf-8")
        name = "GeneratedUSharpBindingAnswersChatPromptWithText"
        self.assertIn(f"IEnumerator {name}()", source)
        self.assertIn(f'"{name}"', precommit)
        self.assertIn("Prepare Public TinyLlama Chat Validation", importer)
        self.assertIn("Prepare Public TinyLlama Chat Validation", precommit)
        self.assertIn("Udoncl.TinyLlama.Chat.Focused.Tests", assembly)
        self.assertIn("onnx-transformer-tiny-chat", cmake)
        self.assertIn("UDONCL_TINYLLAMA_CONTEXT_CAPACITY", cmake)
        for required in (
            "TryBeginChatPrompt(out libraryStatus)",
            "TryContinueChatPrompt(",
            "TryChatPromptPending(",
            "TryReadPromptTokens(",
            "StartGenerate(",
            "TryStreamedTextLength(",
            "TryReadStreamedText(",
            "TryReadGeneratedTokens(",
            "TryGeneratedTextLength(",
            "TryReadGeneratedText(",
            "TryShutdown(",
        ):
            self.assertIn(required, controller)
        for forbidden in ("GetProgramVariable", "SetProgramVariable", "__vrc_",
                          "__udoncl_", "udon_onnx_binding_", "1701", "29871"):
            self.assertNotIn(forbidden, controller)
        for required in (
            "UDONCL_PUBLIC_TINYLLAMA_CHAT_UNITY_BUNDLE",
            "UDONCL_PUBLIC_TINYLLAMA_CHAT_ORACLE",
            "udoncl.tinyllama-chat-oracle.v1",
            "CollectionAssert.AreEqual(expectedPrompt, Prefix(prompt, promptCount), state);",
            "CollectionAssert.AreEqual(expectedTokens, Prefix(tokens, generatedCount), state);",
            "CollectionAssert.AreEqual(expectedTextUnits, Prefix(units, textLength), state);",
            "CollectionAssert.AreEqual(expectedTextUnits, Prefix(streamed, streamedLength), state);",
            "Assert.IsTrue(streamMonotone,",
            "Assert.IsTrue(streamPrefixStable,",
            "Udon VM halted during ",
        ):
            self.assertIn(required, source)
        iterator = source.split(f"IEnumerator {name}()", 1)[1].split(
            "private void CaptureVmError", 1)[0]
        self.assertNotIn("=>", iterator)
        self.assertNotIn("delegate", iterator)
        self.assertIn("private const int PhaseDone = 9;", source)
        self.assertIn("public const int PhaseDone = 9;", controller)

    def test_multi_turn_chat_fixture_is_wired_to_the_chat_gate(self):
        package = ROOT / "unity/Packages/com.lemolatoon.udoncl/Editor"
        controller = (
            package / "Examples/tinyllama-multi-turn-chat-test-controller.cs.txt"
        ).read_text(encoding="utf-8")
        source = (
            package / "Examples/tinyllama-multi-turn-chat-lifecycle-test.cs.txt"
        ).read_text(encoding="utf-8")
        importer = (package / "UdonclPackagedExampleImporter.cs").read_text(
            encoding="utf-8")
        precommit = (ROOT / "scripts/precommit.py").read_text(encoding="utf-8")
        oracle = (ROOT / "tests/boundary/tinyllama_chat_oracle.py").read_text(
            encoding="utf-8")
        for name in ("test-controller", "lifecycle-test"):
            self.assertTrue((
                package / f"Examples/tinyllama-multi-turn-chat-{name}.cs.txt.meta"
            ).is_file())
        name = "GeneratedUSharpBindingContinuesChatAcrossTurns"
        self.assertIn(f"IEnumerator {name}()", source)
        self.assertIn(f'"{name}"', precommit)
        self.assertIn("tinyllama_chat_oracle_turns(tinyllama_chat_oracle)", precommit)
        self.assertIn("PublicTinyLlamaMultiTurnChatLifecycleTests.cs", importer)
        self.assertIn("UdonclTinyLlamaMultiTurnChatTestController.cs", importer)
        self.assertIn('"--next-user"', oracle)
        # The controller drives every turn through the high-level text API.
        for required in (
            "TryBeginChatPrompt(out libraryStatus)",
            "TryContinueChatPrompt(",
            "TryBeginChatTurn(out libraryStatus)",
            "TryContinueChatTurn(",
            "TryChatPromptPending(",
            "TryReadPromptTokens(",
            "StartGenerate(",
            "TryStreamedTextLength(",
            "TryReadStreamedText(",
            "TryReadGeneratedTokens(",
            "TryReadGeneratedText(",
            "StartResetChat(",
            "TryGetResetChatResult(",
            "TryShutdown(",
        ):
            self.assertIn(required, controller)
        for forbidden in ("GetProgramVariable", "SetProgramVariable", "__vrc_",
                          "__udoncl_", "udon_onnx_binding_", "TrySetPromptTokens",
                          "StartReset(", "29871"):
            self.assertNotIn(forbidden, controller)
        for required in (
            "UDONCL_PUBLIC_TINYLLAMA_CHAT_UNITY_BUNDLE",
            "UDONCL_PUBLIC_TINYLLAMA_CHAT_ORACLE",
            "udoncl.tinyllama-chat-oracle.v1",
            "public TinyLlamaChatTurn[] turns;",
            '"turn " + index + " prompt: " + state',
            '"turn " + index + " tokens: " + state',
            '"turn " + index + " text: " + state',
            '"turn " + index + " streamed text: " + state',
            "newPromptAfterReplyStatus",
            "turnAfterResetStatus",
            "Udon VM halted during ",
            "timing.Begin(",
            "timing.End();",
        ):
            self.assertIn(required, source)
        iterator = source.split(f"IEnumerator {name}()", 1)[1].split(
            "private void CaptureVmError", 1)[0]
        self.assertNotIn("=>", iterator)
        self.assertNotIn("delegate", iterator)
        # Jagged arrays do not survive the play-mode domain reload.
        fields = source.split("[OneTimeSetUp]", 1)[0]
        self.assertNotIn("int[][]", fields)
        self.assertIn("private const int PhaseDone = 10;", source)
        self.assertIn("public const int PhaseDone = 10;", controller)

    def test_fixtures_report_wall_clock_and_frame_rate_mode(self):
        # Batchmode frames are not a proxy for time, so every focused
        # TinyLlama fixture reports wall-clock time per phase and per step,
        # and can hold a fixed frame rate; see plan section 7.2.
        package = ROOT / "unity/Packages/com.lemolatoon.udoncl/Editor"
        timing = (package / "UdonclFixtureTiming.cs").read_text(encoding="utf-8")
        self.assertTrue((package / "UdonclFixtureTiming.cs.meta").is_file())
        for required in (
            'FrameRateVariable = "UDONCL_FIXTURE_TARGET_FRAME_RATE";',
            "QualitySettings.vSyncCount = 0;",
            "Application.targetFrameRate = targetFrameRate;",
            "QualitySettings.vSyncCount = savedVSyncCount;",
            "Application.targetFrameRate = savedTargetFrameRate;",
            "Time.realtimeSinceStartupAsDouble",
            '" mode="',
            '" target_fps="',
            '" phase="',
            '" ms_per_step="',
            '" frames_per_step="',
            '" readback"',
            '" readback_ms_max="',
            '" paced_frames="',
        ):
            self.assertIn(required, timing)
        plan = (ROOT / "TINYLLAMA_AOT_CONVERSATION_PLAN.md").read_text(
            encoding="utf-8")
        self.assertIn("UDONCL_FIXTURE_TARGET_FRAME_RATE", plan)
        fixtures = {
            "conversation": (
                'new UdonclFixtureTiming("tinyllama_conversation")',
                # Existing progress fields keep their order; ms is appended.
                "udoncl tinyllama_conversation phase={0} ({1}) frame={2} ms={3}",
                "IsGeneratePhase(currentPhase) ? generateSteps : 0",
                "int generateSteps = promptTokens.Length + maxNewTokens - 1;",
                '.Append(" first_generate_ms=")',
                '.Append(" second_generate_ms=")',
            ),
            "chat": (
                'new UdonclFixtureTiming("tinyllama_chat")',
                "udoncl tinyllama_chat phase={0} ({1}) frame={2} ms={3}",
                "currentPhase == PhaseGenerate ? generateSteps : 0",
                '.Append(" generate_ms=")',
            ),
            "prefill": (
                'new UdonclFixtureTiming("tinyllama_prefill")',
                '"tinyllama_prefill frames={0} outputs={1}{2} run_frames={3} run_ms={4}"',
                "timing.Readbacks(",
                'timing.Phase("run", 1);',
            ),
            "step": (
                '"tinyllama_m1_step",',
                '"step=" + stepName',
                "timing.Readbacks(",
                "currentPhase == PhaseRun ? 1 : 0",
                '+ " run_ms="',
            ),
        }
        for name, required_lines in fixtures.items():
            source = (
                package / f"Examples/tinyllama-{name}-lifecycle-test.cs.txt"
            ).read_text(encoding="utf-8")
            for required in required_lines + (
                "timing.Begin(",
                "timing.Frame();",
                "timing.End();",
                "timing.Summary",
            ):
                self.assertIn(required, source, name)
            # The settings are restored even when the wait loop throws.
            self.assertLess(
                source.index("timing.Begin("), source.index("finally"), name)
            self.assertLess(
                source.index("finally"), source.index("timing.End();"), name)
            controller = (
                package / f"Examples/tinyllama-{name}-test-controller.cs.txt"
            ).read_text(encoding="utf-8")
            # Measurement is Editor-side only: the U# controllers are unchanged.
            for forbidden in ("realtimeSinceStartup", "targetFrameRate",
                              "UdonclFixtureTiming"):
                self.assertNotIn(forbidden, controller, name)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
