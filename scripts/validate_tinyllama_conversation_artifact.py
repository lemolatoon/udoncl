"""Validate the pinned TinyLlama M=1 Conversation/Chat Artifact contract.

The Conversation Artifact exports only the token-array lifecycle. The Chat
Artifact is the same Host plus the optional text layer, including multi-turn
chat (BeginChatTurn/ContinueChatTurn append a user message after a reply,
ResetChat starts a new conversation). Both are specialized to a build-time
Context Capacity (default 512). With ``--sequence-rows k`` (the example's
UDONCL_TINYLLAMA_SEQUENCE_ROWS) one run covers k positions: k input, position
and logits rows, and an attention window of capacity + k - 1 on the same past.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_tensor(
    tensor: dict,
    *,
    index: int,
    name: str,
    data_type: str,
    dimensions: list[int],
    logical_bytes: int,
) -> None:
    require(tensor.get("index") == index, f"{name}: invalid index")
    require(tensor.get("name") == name, f"input/output {index}: expected {name}")
    require(tensor.get("data_type") == data_type, f"{name}: invalid data type")
    require(tensor.get("dimensions") == dimensions, f"{name}: invalid dimensions")
    require(tensor.get("logical_bytes") == logical_bytes, f"{name}: invalid logical bytes")
    capacity = tensor.get("capacity_bytes")
    require(isinstance(capacity, int), f"{name}: missing capacity")
    require(capacity >= logical_bytes, f"{name}: capacity is smaller than its view")


CHAT_EXPORTS = {
    "SetChatPrompt": ("tinyllama_set_chat_prompt", ["string", "string"], False),
    "BeginChatPrompt": ("tinyllama_begin_chat_prompt", [], False),
    "ContinueChatPrompt": (
        "tinyllama_continue_chat_prompt", ["string", "string", "i32"], False),
    "ChatPromptPending": ("tinyllama_chat_prompt_pending", [], False),
    "BeginChatTurn": ("tinyllama_begin_chat_turn", [], False),
    "ContinueChatTurn": (
        "tinyllama_continue_chat_turn", ["string", "i32"], False),
    "ResetChat": ("tinyllama_reset_chat", [], True),
    "PromptTokenCount": ("tinyllama_prompt_token_count", [], False),
    "ReadPromptTokens": ("tinyllama_read_prompt_tokens", ["i32_array"], False),
    "GeneratedTextLength": ("tinyllama_generated_text_length", [], False),
    "ReadGeneratedText": ("tinyllama_read_generated_text", ["i32_array"], False),
    "StreamedTextLength": ("tinyllama_streamed_text_length", [], False),
    "ReadStreamedText": (
        "tinyllama_read_streamed_text", ["i32_array", "i32", "i32"], False),
    "LastTokenizerStatus": ("tinyllama_last_tokenizer_status", [], False),
}
POLLED_WHILE_GENERATING = ("StreamedTextLength", "ReadStreamedText")
PROFILES = {
    "conversation": ("tinyllama_real_conversation", "UdonclTinyLlamaConversationBinding"),
    "chat": ("tinyllama_real_chat", "UdonclTinyLlamaChatBinding"),
}


def validate_exports(manifest: dict, profile: str = "conversation") -> None:
    expected = {
        "Initialize": ("tinyllama_initialize", [], True),
        "Reset": ("tinyllama_reset", [], True),
        "SetPromptTokens": (
            "tinyllama_set_prompt_tokens", ["i32_array", "i32"], False,
        ),
        "Generate": ("tinyllama_generate", ["i32"], True),
        "GeneratedTokenCount": ("tinyllama_generated_token_count", [], False),
        "ReadGeneratedTokens": (
            "tinyllama_read_generated_tokens", ["i32_array"], False,
        ),
        "Shutdown": ("tinyllama_dispose", [], False),
    }
    if profile == "chat":
        expected.update(CHAT_EXPORTS)
    exports = manifest.get("exports")
    require(isinstance(exports, list), "artifact has no public exports")
    by_name = {item.get("public_name"): item for item in exports}
    require(set(by_name) == set(expected), "artifact has an unexpected public API")
    for public_name, (symbol, parameter_kinds, is_async) in expected.items():
        export = by_name[public_name]
        require(export.get("c_symbol") == symbol,
                f"{public_name}: invalid C symbol")
        parameters = export.get("parameters")
        require(isinstance(parameters, list),
                f"{public_name}: missing parameter schema")
        require([item.get("kind") for item in parameters] == parameter_kinds,
                f"{public_name}: invalid parameter schema")
        require(export.get("async") is is_async,
                f"{public_name}: invalid async contract")
        require(export.get("result", {}).get("kind") == "i32",
                f"{public_name}: invalid result schema")
        # Only the streamed-text exports may be polled while Generate is
        # pending; every other export stays rejected as busy.
        require(export.get("allowed_while_busy", False)
                is (public_name in POLLED_WHILE_GENERATING),
                f"{public_name}: invalid busy contract")


def validate_artifact(
    artifact: Path, profile: str = "conversation", capacity: int = 512,
    rows: int = 1,
) -> dict[str, int]:
    require(profile in PROFILES, f"unknown artifact profile: {profile}")
    require(capacity >= 2, "Context Capacity must be at least 2")
    require(1 <= rows <= capacity, "sequence rows must be within the capacity")
    window = capacity + rows - 1
    program_name, binding_class = PROFILES[profile]
    manifest_path = artifact / "manifest.json"
    require(manifest_path.is_file(), f"artifact has no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    program = manifest.get("program", {})
    require(program.get("name") == program_name,
            "artifact has an unexpected program identity")
    require(program.get("binding_class") == binding_class,
            "artifact has an unexpected generated U# binding class")
    validate_exports(manifest, profile)

    binding = manifest.get("onnx_binding")
    require(isinstance(binding, dict), "artifact has no generated ONNX binding")
    inputs = binding.get("inputs")
    outputs = binding.get("outputs")
    require(isinstance(inputs, list) and len(inputs) == 47, "expected 47 inputs")
    require(isinstance(outputs, list) and len(outputs) == 45, "expected 45 outputs")

    validate_tensor(inputs[0], index=0, name="input_ids", data_type="int64",
                    dimensions=[1, rows], logical_bytes=8 * rows)
    validate_tensor(inputs[1], index=1, name="attention_mask", data_type="int64",
                    dimensions=[1, window], logical_bytes=8 * window)
    validate_tensor(inputs[2], index=2, name="position_ids", data_type="int64",
                    dimensions=[1, rows], logical_bytes=8 * rows)
    for layer in range(22):
        for kind_offset, kind in enumerate(("key", "value")):
            index = 3 + layer * 2 + kind_offset
            validate_tensor(
                inputs[index], index=index,
                name=f"past_key_values.{layer}.{kind}", data_type="float32",
                dimensions=[1, 4, capacity - 1, 64],
                logical_bytes=4 * 4 * (capacity - 1) * 64,
            )

    validate_tensor(outputs[0], index=0, name="logits", data_type="float32",
                    dimensions=[1, rows, 32000], logical_bytes=128000 * rows)
    for layer in range(22):
        for kind_offset, kind in enumerate(("key", "value")):
            index = 1 + layer * 2 + kind_offset
            validate_tensor(
                outputs[index], index=index, name=f"present.{layer}.{kind}",
                data_type="float32", dimensions=[1, 4, window, 64],
                logical_bytes=4 * 4 * window * 64,
            )

    physical = binding.get("physical", {})
    require(physical.get("enqueue_advance_sequence_state_symbol") ==
            "udon_onnx_binding_enqueue_advance_sequence_state",
            "artifact has no generated sequence-state advance helper")
    require(physical.get("enqueue_reset_sequence_state_symbol") ==
            "udon_onnx_binding_enqueue_reset_sequence_state",
            "artifact has no generated sequence-state reset helper")
    buffers = manifest.get("device", {}).get("buffers")
    mutable_capacity = manifest.get("device", {}).get("mutable_handle_capacity")
    require(isinstance(buffers, list), "artifact has no Device buffers")
    require(mutable_capacity == len(buffers),
            "conversation Device buffer pool does not match compiler metadata")
    require(not any(item.get("public_access") for item in buffers),
            "conversation tensors must remain private")

    binding_source = artifact / program.get("binding_source", "binding.cs")
    require(binding_source.is_file(), "artifact has no generated U# binding")
    source = binding_source.read_text(encoding="utf-8")
    required_methods = (
        "StartInitialize(", "GetInitializeStatus(", "TryGetInitializeResult(",
        "StartReset(", "GetResetStatus(", "TryGetResetResult(",
        "TrySetPromptTokens(int[] tokenIds, int tokenCount, out int result)",
        "StartGenerate(int maxNewTokens, UdonSharpBehaviour receiver,",
        "GetGenerateStatus(", "TryGetGenerateResult(",
        "TryGeneratedTokenCount(out int result)",
        "TryReadGeneratedTokens(int[] destination, out int result)",
        "TryShutdown(out int result)",
    )
    if profile == "chat":
        required_methods += (
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
        )
    for method in required_methods:
        require(method in source, f"conversation binding is missing API: {method}")
    forbidden_api = ("StartRun", "TryWriteInput", "StartRead", "udon_string_t")
    if profile == "conversation":
        forbidden_api += ("Tokenize",)
    for forbidden in forbidden_api:
        require(forbidden not in source,
                f"conversation binding exposes internal model/tokenizer API: {forbidden}")

    kernel_ids = {
        item.get("id") for item in manifest.get("device", {}).get("kernels", [])
    }
    require({247, 248}.issubset(kernel_ids),
            "artifact has no Device ArgMax kernels")

    return {"inputs": len(inputs), "outputs": len(outputs),
            "buffers": len(buffers)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="conversation")
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--sequence-rows", type=int, default=1)
    args = parser.parse_args()
    summary = validate_artifact(
        args.artifact.resolve(), args.profile, args.capacity, args.sequence_rows
    )
    print(f"tinyllama_m1_{args.profile}_profile=passed capacity={args.capacity} "
          f"sequence_rows={args.sequence_rows} " + " ".join(
        f"{name}={value}" for name, value in summary.items()
    ))


if __name__ == "__main__":
    main()
