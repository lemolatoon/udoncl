"""Validate the artifact of the minimal TinyLlama chat example.

examples/tinyllama-chat builds program tinyllama_chat with the binding
UdonclTinyllamaChatBinding. Its public API is exactly the seven exports of
tinyllama_chat.c; only ReadStreamedText may run while Generate is pending.
The model runs one position per run on a past bounded by capacity - 1 rows,
every present is an in-place Concat of its past, every graph tensor stays a
private buffer, and the Host launches its own ArgMax kernel
(tinyllama_argmax, one work item over the 32000 logits).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_conversation_validator():
    path = Path(__file__).resolve().with_name("validate_tinyllama_conversation_artifact.py")
    spec = importlib.util.spec_from_file_location("validate_tinyllama_conversation_artifact", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_conversation = _load_conversation_validator()
require = _conversation.require
validate_tensor = _conversation.validate_tensor

PROGRAM = "tinyllama_chat"
BINDING_CLASS = "UdonclTinyllamaChatBinding"
# Public name -> (C symbol, parameter kinds, async).
EXPORTS = {
    "Initialize": ("initialize", [], True),
    "BeginMessage": ("begin_message", ["string"], False),
    "ContinueMessage": ("continue_message", [], False),
    "Generate": ("generate", [], True),
    "ReadStreamedText": ("read_streamed_text", ["i32_array", "i32"], False),
    "ReadGeneratedTokens": ("read_generated_tokens", ["i32_array"], False),
    "Dispose": ("dispose", [], False),
}
ALLOWED_WHILE_BUSY = ("ReadStreamedText",)
BINDING_METHODS = (
    "StartInitialize(UdonSharpBehaviour receiver, string completionEvent)",
    "GetInitializeStatus(",
    "TryGetInitializeResult(out int result)",
    "TryBeginMessage(string text, out int result)",
    "TryContinueMessage(out int result)",
    "StartGenerate(UdonSharpBehaviour receiver, string completionEvent)",
    "GetGenerateStatus(",
    "TryGetGenerateResult(out int result)",
    "TryReadStreamedText(int[] destination, int offset, out int result)",
    "TryReadGeneratedTokens(int[] destination, out int result)",
    "TryDispose(out int result)",
)
ARGMAX_KERNEL = "tinyllama_argmax"
VOCABULARY = 32000


def validate_exports(manifest: dict) -> None:
    exports = manifest.get("exports")
    require(isinstance(exports, list), "artifact has no public exports")
    by_name = {item.get("public_name"): item for item in exports}
    require(set(by_name) == set(EXPORTS),
            f"artifact has an unexpected public API: {sorted(by_name)}")
    for public_name, (symbol, parameter_kinds, is_async) in EXPORTS.items():
        export = by_name[public_name]
        require(export.get("c_symbol") == symbol, f"{public_name}: invalid C symbol")
        parameters = export.get("parameters")
        require(isinstance(parameters, list), f"{public_name}: missing parameter schema")
        require([item.get("kind") for item in parameters] == parameter_kinds,
                f"{public_name}: invalid parameter schema")
        require(bool(export.get("async")) is is_async,
                f"{public_name}: invalid async contract")
        require(export.get("result", {}).get("kind") == "i32",
                f"{public_name}: invalid result schema")
        require(bool(export.get("allowed_while_busy"))
                is (public_name in ALLOWED_WHILE_BUSY),
                f"{public_name}: invalid busy contract")


def validate_artifact(artifact: Path, capacity: int = 2048) -> dict[str, int]:
    require(capacity >= 2, "capacity must be at least 2")
    manifest_path = artifact / "manifest.json"
    require(manifest_path.is_file(), f"artifact has no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    program = manifest.get("program", {})
    require(program.get("name") == PROGRAM, "artifact has an unexpected program")
    require(program.get("binding_class") == BINDING_CLASS,
            "artifact has an unexpected generated U# binding class")
    validate_exports(manifest)

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
    validate_tensor(outputs[0], index=0, name="logits", data_type="float32",
                    dimensions=[1, 1, VOCABULARY], logical_bytes=4 * VOCABULARY)
    for layer in range(22):
        for kind_offset, kind in enumerate(("key", "value")):
            validate_tensor(
                inputs[3 + layer * 2 + kind_offset], index=3 + layer * 2 + kind_offset,
                name=f"past_key_values.{layer}.{kind}", data_type="float32",
                dimensions=[1, 4, -1, 64], logical_bytes=4 * 4 * (capacity - 1) * 64,
                bounds=[0, 0, capacity - 1, 0],
            )
            validate_tensor(
                outputs[1 + layer * 2 + kind_offset], index=1 + layer * 2 + kind_offset,
                name=f"present.{layer}.{kind}", data_type="float32",
                dimensions=[1, 4, -1, 64], logical_bytes=4 * 4 * capacity * 64,
                bounds=[0, 0, capacity, 0],
            )
    require(binding.get("in_place_concats") == [
        {"input": 3 + offset, "output": 1 + offset, "axis": 2} for offset in range(44)
    ], "the presents are not in-place Concat pairs of their pasts")

    device = manifest.get("device", {})
    buffers = device.get("buffers")
    require(isinstance(buffers, list), "artifact has no Device buffers")
    require(not any(item.get("public_access") for item in buffers),
            "the chat example has no public buffers")

    # The user ArgMax kernel: an id above the library kernels, one work
    # item, reading the logits and writing one word.
    kernels = {item.get("name"): item for item in device.get("kernels", [])}
    argmax = kernels.get(ARGMAX_KERNEL)
    require(argmax is not None, f"artifact does not list the {ARGMAX_KERNEL} kernel")
    require(isinstance(argmax.get("id"), int) and argmax["id"] >= 1024,
            f"{ARGMAX_KERNEL}: not a user kernel id")
    require(argmax.get("global_size") == [1, 1, 1], f"{ARGMAX_KERNEL}: invalid NDRange")
    require([item.get("required_bytes") for item in argmax.get("args", [])
             if item.get("kind") == "buffer"] == [4 * VOCABULARY, 4],
            f"{ARGMAX_KERNEL}: invalid buffer arguments")

    binding_source = artifact / program.get("binding_source", "binding.cs")
    require(binding_source.is_file(), "artifact has no generated U# binding")
    source = binding_source.read_text(encoding="utf-8")
    for method in BINDING_METHODS:
        require(method in source, f"chat binding is missing API: {method}")
    for forbidden in ("StartRun", "TryWrite", "StartRead", "udon_string_t"):
        require(forbidden not in source,
                f"chat binding exposes internal model API: {forbidden}")
    return {"inputs": len(inputs), "outputs": len(outputs), "buffers": len(buffers),
            "kernels": len(kernels), "argmax_id": argmax["id"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--capacity", type=int, default=2048)
    args = parser.parse_args()
    summary = validate_artifact(args.artifact.resolve(), args.capacity)
    print(f"tinyllama_chat_example=passed capacity={args.capacity} "
          + " ".join(f"{name}={value}" for name, value in summary.items()))


if __name__ == "__main__":
    main()
