"""Validate the pinned TinyLlama M=1 conversation-profile artifact contract."""

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


def validate_artifact(artifact: Path) -> dict[str, int]:
    manifest_path = artifact / "manifest.json"
    require(manifest_path.is_file(), f"artifact has no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    program = manifest.get("program", {})
    require(program.get("name") == "tinyllama_real_conversation",
            "artifact has an unexpected program identity")
    require(program.get("binding_class") == "UdonclTinyLlamaConversationBinding",
            "artifact has an unexpected generated U# binding class")

    binding = manifest.get("onnx_binding")
    require(isinstance(binding, dict), "artifact has no generated ONNX binding")
    inputs = binding.get("inputs")
    outputs = binding.get("outputs")
    require(isinstance(inputs, list) and len(inputs) == 47, "expected 47 inputs")
    require(isinstance(outputs, list) and len(outputs) == 45, "expected 45 outputs")

    validate_tensor(inputs[0], index=0, name="input_ids", data_type="int64",
                    dimensions=[1, 1], logical_bytes=8)
    validate_tensor(inputs[1], index=1, name="attention_mask", data_type="int64",
                    dimensions=[1, 64], logical_bytes=512)
    validate_tensor(inputs[2], index=2, name="position_ids", data_type="int64",
                    dimensions=[1, 1], logical_bytes=8)
    for layer in range(22):
        for kind_offset, kind in enumerate(("key", "value")):
            index = 3 + layer * 2 + kind_offset
            validate_tensor(
                inputs[index], index=index,
                name=f"past_key_values.{layer}.{kind}", data_type="float32",
                dimensions=[1, 4, 63, 64], logical_bytes=64512,
            )

    validate_tensor(outputs[0], index=0, name="logits", data_type="float32",
                    dimensions=[1, 1, 32000], logical_bytes=128000)
    for layer in range(22):
        for kind_offset, kind in enumerate(("key", "value")):
            index = 1 + layer * 2 + kind_offset
            validate_tensor(
                outputs[index], index=index, name=f"present.{layer}.{kind}",
                data_type="float32", dimensions=[1, 4, 64, 64],
                logical_bytes=65536,
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
    require(mutable_capacity == len(buffers) == 124,
            "conversation Device buffer pool does not match compiler metadata")
    require(not any(item.get("public_access") for item in buffers),
            "conversation tensors must remain private")

    binding_source = artifact / program.get("binding_source", "binding.cs")
    require(binding_source.is_file(), "artifact has no generated U# binding")
    source = binding_source.read_text(encoding="utf-8")
    for forbidden in ("TryWriteInput", "StartRead", "Tokenize", "udon_string_t"):
        require(forbidden not in source,
                f"conversation binding exposes internal model/tokenizer API: {forbidden}")

    return {"inputs": len(inputs), "outputs": len(outputs),
            "buffers": len(buffers)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    summary = validate_artifact(args.artifact.resolve())
    print("tinyllama_m1_conversation_profile=passed " + " ".join(
        f"{name}={value}" for name, value in summary.items()
    ))


if __name__ == "__main__":
    main()
