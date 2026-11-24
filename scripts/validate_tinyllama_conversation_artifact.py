"""Validate the pinned TinyLlama M=1 Conversation/Chat Artifact contract.

The Conversation Artifact exports only the token-array lifecycle. The Chat
Artifact is the same Host plus the optional text layer, including multi-turn
chat (BeginChatTurn/ContinueChatTurn append a user message after a reply,
ResetChat starts a new conversation). Both run one position per model run
on a past that is a bounded dynamic dimension of at most capacity - 1 rows
(the build-time Context Capacity, default 512): the past and present K/V and
the attention mask are -1 on that axis, and every present is an in-place
Concat of its past. The Host keeps every graph tensor in a private buffer.
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
    bounds: list[int] | None = None,
) -> None:
    require(tensor.get("index") == index, f"{name}: invalid index")
    require(tensor.get("name") == name, f"input/output {index}: expected {name}")
    require(tensor.get("data_type") == data_type, f"{name}: invalid data type")
    require(tensor.get("dimensions") == dimensions, f"{name}: invalid dimensions")
    # A -1 axis is the bounded past_sequence_length, with its largest extent.
    if bounds is None:
        require("dimension_bounds" not in tensor, f"{name}: unexpected bounds")
    else:
        require(tensor.get("dimension_bounds") == bounds, f"{name}: invalid bounds")
        require(tensor.get("dimension_names") == [
            "past_sequence_length" if bound else "" for bound in bounds
        ], f"{name}: invalid dimension names")
    require(tensor.get("logical_bytes") == logical_bytes, f"{name}: invalid logical bytes")
    capacity = tensor.get("capacity_bytes")
    require(isinstance(capacity, int), f"{name}: missing capacity")
    require(capacity >= logical_bytes, f"{name}: capacity is smaller than its view")


CHAT_EXPORTS = {
    "SetChatPrompt": ("set_chat_prompt", ["string", "string"], False),
    "BeginChatPrompt": ("begin_chat_prompt", [], False),
    "ContinueChatPrompt": (
        "continue_chat_prompt", ["string", "string", "i32"], False),
    "ChatPromptPending": ("chat_prompt_pending", [], False),
    "BeginChatTurn": ("begin_chat_turn", [], False),
    "ContinueChatTurn": (
        "continue_chat_turn", ["string", "i32"], False),
    "ResetChat": ("reset_chat", [], True),
    "PromptTokenCount": ("prompt_token_count", [], False),
    "ReadPromptTokens": ("read_prompt_tokens", ["i32_array"], False),
    "GeneratedTextLength": ("generated_text_length", [], False),
    "ReadGeneratedText": ("read_generated_text", ["i32_array"], False),
    "StreamedTextLength": ("streamed_text_length", [], False),
    "ReadStreamedText": (
        "read_streamed_text", ["i32_array", "i32", "i32"], False),
    "LastTokenizerStatus": ("last_tokenizer_status", [], False),
}
POLLED_WHILE_GENERATING = ("StreamedTextLength", "ReadStreamedText")
PROFILES = {
    "conversation": ("tinyllama_real_conversation", "UdonclTinyLlamaConversationBinding"),
    "chat": ("tinyllama_real_chat", "UdonclTinyLlamaChatBinding"),
}


def validate_exports(manifest: dict, profile: str = "conversation") -> None:
    expected = {
        "Initialize": ("initialize", [], True),
        "Reset": ("reset", [], True),
        "SetPromptTokens": (
            "set_prompt_tokens", ["i32_array", "i32"], False,
        ),
        "Generate": ("generate", ["i32"], True),
        "GeneratedTokenCount": ("generated_token_count", [], False),
        "ReadGeneratedTokens": (
            "read_generated_tokens", ["i32_array"], False,
        ),
        "Shutdown": ("shutdown", [], False),
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
) -> dict[str, int]:
    require(profile in PROFILES, f"unknown artifact profile: {profile}")
    require(capacity >= 2, "Context Capacity must be at least 2")
    program_name, binding_class = PROFILES[profile]
    manifest_path = artifact / "manifest.json"
    require(manifest_path.is_file(), f"artifact has no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    program = manifest.get("program", {})
    # The 4096-token artifacts (-4k targets) have programs of their own.
    require(program.get("name") in (program_name, program_name + "_4k"),
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
                    dimensions=[1, 1], logical_bytes=8)
    validate_tensor(inputs[1], index=1, name="attention_mask", data_type="int64",
                    dimensions=[1, -1], logical_bytes=8 * capacity,
                    bounds=[0, capacity])
    validate_tensor(inputs[2], index=2, name="position_ids", data_type="int64",
                    dimensions=[1, 1], logical_bytes=8)
    for layer in range(22):
        for kind_offset, kind in enumerate(("key", "value")):
            index = 3 + layer * 2 + kind_offset
            validate_tensor(
                inputs[index], index=index,
                name=f"past_key_values.{layer}.{kind}", data_type="float32",
                dimensions=[1, 4, -1, 64],
                logical_bytes=4 * 4 * (capacity - 1) * 64,
                bounds=[0, 0, capacity - 1, 0],
            )

    validate_tensor(outputs[0], index=0, name="logits", data_type="float32",
                    dimensions=[1, 1, 32000], logical_bytes=128000)
    for layer in range(22):
        for kind_offset, kind in enumerate(("key", "value")):
            index = 1 + layer * 2 + kind_offset
            validate_tensor(
                outputs[index], index=index, name=f"present.{layer}.{kind}",
                data_type="float32", dimensions=[1, 4, -1, 64],
                logical_bytes=4 * 4 * capacity * 64,
                bounds=[0, 0, capacity, 0],
            )

    # Every present is Concat(its past, the new row): the Host binds both to
    # one buffer and a run writes only its own row.
    require(binding.get("in_place_concats") == [
        {"input": 3 + offset, "output": 1 + offset, "axis": 2}
        for offset in range(44)
    ], "the presents are not in-place Concat pairs of their pasts")
    buffers = manifest.get("device", {}).get("buffers")
    require(isinstance(buffers, list), "artifact has no Device buffers")
    require(not any(item.get("public_access") for item in buffers),
            "conversation tensors must remain private")

    binding_source = artifact / program.get("binding_source", "binding.cs")
    require(binding_source.is_file(), "artifact has no generated U# binding")
    source = binding_source.read_text(encoding="utf-8")
    required_methods = (
        "StartInitialize(", "GetInitializeStatus(", "TryGetInitializeResult(",
        "StartReset(", "GetResetStatus(", "TryGetResetResult(",
        "TrySetPromptTokens(int[] token_ids, int token_count, out int result)",
        "StartGenerate(int max_new_tokens, UdonSharpBehaviour receiver,",
        "GetGenerateStatus(", "TryGetGenerateResult(",
        "TryGeneratedTokenCount(out int result)",
        "TryReadGeneratedTokens(int[] destination, out int result)",
        "TryShutdown(out int result)",
    )
    if profile == "chat":
        required_methods += (
            "TrySetChatPrompt(string system_text, string user_text, out int result)",
            "TryBeginChatPrompt(out int result)",
            "TryContinueChatPrompt(string system_text, string user_text, "
            "int max_work, out int result)",
            "TryChatPromptPending(out int result)",
            "TryBeginChatTurn(out int result)",
            "TryContinueChatTurn(string user_text, int max_work, out int result)",
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
    args = parser.parse_args()
    summary = validate_artifact(args.artifact.resolve(), args.profile, args.capacity)
    print(f"tinyllama_m1_{args.profile}_profile=passed capacity={args.capacity} "
          + " ".join(
        f"{name}={value}" for name, value in summary.items()
    ))


if __name__ == "__main__":
    main()
