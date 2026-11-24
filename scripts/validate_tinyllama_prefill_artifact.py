"""Validate the pinned TinyLlama fixed-prefill Unity artifact contract."""

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
    capacity_bytes: int | None = None,
) -> None:
    require(tensor.get("index") == index, f"{name}: invalid index")
    require(tensor.get("name") == name, f"input/output {index}: expected {name}")
    require(tensor.get("data_type") == data_type, f"{name}: invalid data type")
    require(tensor.get("dimensions") == dimensions, f"{name}: invalid dimensions")
    require(tensor.get("logical_bytes") == logical_bytes, f"{name}: invalid logical bytes")
    capacity = tensor.get("capacity_bytes")
    require(isinstance(capacity, int), f"{name}: missing capacity")
    require(capacity >= logical_bytes, f"{name}: capacity is smaller than its view")
    if capacity_bytes is not None:
        require(capacity == capacity_bytes, f"{name}: invalid capacity")


def validate_artifact(artifact: Path) -> dict[str, int]:
    manifest_path = artifact / "manifest.json"
    require(manifest_path.is_file(), f"artifact has no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    program = manifest.get("program", {})
    require(
        program.get("name") == "tinyllama_real_prefill",
        "artifact has an unexpected program identity",
    )
    require(
        program.get("binding_class") == "UdonclTinyLlamaPrefillBinding",
        "artifact has an unexpected generated U# binding class",
    )
    binding = manifest.get("onnx_binding")
    require(isinstance(binding, dict), "artifact has no generated ONNX binding")
    inputs = binding.get("inputs")
    outputs = binding.get("outputs")
    require(isinstance(inputs, list) and len(inputs) == 47, "expected 47 inputs")
    require(isinstance(outputs, list) and len(outputs) == 45, "expected 45 outputs")

    validate_tensor(inputs[0], index=0, name="input_ids", data_type="int64",
                    dimensions=[1, 4], logical_bytes=32, capacity_bytes=32)
    validate_tensor(inputs[1], index=1, name="attention_mask", data_type="int64",
                    dimensions=[1, 4], logical_bytes=32, capacity_bytes=32)
    validate_tensor(inputs[2], index=2, name="position_ids", data_type="int64",
                    dimensions=[1, 4], logical_bytes=32, capacity_bytes=32)
    for layer in range(22):
        for kind_offset, kind in enumerate(("key", "value")):
            index = 3 + layer * 2 + kind_offset
            validate_tensor(
                inputs[index], index=index,
                name=f"past_key_values.{layer}.{kind}", data_type="float32",
                dimensions=[1, 4, 0, 64], logical_bytes=0,
            )
            require(inputs[index]["capacity_bytes"] > 0,
                    f"{inputs[index]['name']}: empty view has no physical capacity")

    validate_tensor(outputs[0], index=0, name="logits", data_type="float32",
                    dimensions=[1, 4, 32000], logical_bytes=512000,
                    capacity_bytes=524288)
    for layer in range(22):
        for kind_offset, kind in enumerate(("key", "value")):
            index = 1 + layer * 2 + kind_offset
            validate_tensor(
                outputs[index], index=index, name=f"present.{layer}.{kind}",
                data_type="float32", dimensions=[1, 4, 4, 64],
                logical_bytes=4096, capacity_bytes=4096,
            )

    buffers = manifest.get("device", {}).get("buffers")
    capacity = manifest.get("device", {}).get("mutable_handle_capacity")
    require(isinstance(buffers, list), "artifact has no Device buffers")
    # The handle capacity also counts the graph tensors the ONNX plan
    # reserves; the Host binds its own buffers to them.
    require(isinstance(capacity, int) and capacity >= len(buffers),
            "mutable handle capacity is below the Device buffers")
    public = {item["name"]: item for item in buffers if item.get("public_access")}
    # The Host declares one UDONCL_PUBLIC buffer per model input and output
    # with an element, named after the tensor; the 44 empty past K/V inputs
    # share one private buffer.
    require(len(public) == 48,
            "expected one public buffer per non-empty model input and output")
    for tensor, access in ([(item, "input") for item in inputs[:3]] +
                           [(item, "output") for item in outputs]):
        name = tensor["name"].replace(".", "_")
        buffer = public.get(name)
        require(buffer is not None, f"{tensor['name']}: no public buffer {name}")
        require(buffer.get("public_access") == access,
                f"{name}: public access is not {access}")
        require(buffer.get("logical_bytes") == tensor["logical_bytes"],
                f"{name}: logical size differs from the tensor")
        require(buffer.get("source") == {"kind": "host_variable",
                                         "symbol": name,
                                         "onnx_tensor": tensor["name"]},
                f"{name}: buffer is not the Host variable bound to the tensor")
    require(not any(item["name"].startswith("past_key_values")
                    for item in public.values()),
            "an empty past K/V input has a public buffer")

    static_tensors = manifest.get("static_tensors")
    require(isinstance(static_tensors, list) and static_tensors,
            "model initializers were not externalized")
    static_bytes = sum(item["byte_length"] for item in static_tensors)
    require(static_bytes > 900_000_000, "static tensor payload is unexpectedly small")

    storage_path = artifact / program.get("storage_report", "storage-report.json")
    require(storage_path.is_file(), "artifact has no storage report")
    storage = json.loads(storage_path.read_text(encoding="utf-8"))
    require(storage.get("raw_heap_bytes", 1_000_001) <= 1_000_000,
            "model weights were duplicated into the Udon raw heap")
    raw_names = {item.get("name", "") for item in storage.get("raw_objects", [])}
    require(not any(name.startswith("__udon_onnx_static_") for name in raw_names),
            "static ONNX tensor is present in the Udon raw heap")

    binding_source = artifact / program.get("binding_source", "binding.cs")
    require(binding_source.is_file(), "artifact has no generated U# binding")
    source = binding_source.read_text(encoding="utf-8")
    required_methods = [
        "TryWriteInputIds(long[] source)",
        "TryWriteAttentionMask(long[] source)",
        "TryWritePositionIds(long[] source)",
        "StartReadLogits(float[] destination",
        "StartReadPresent0Key(float[] destination",
        "StartReadPresent21Value(float[] destination",
    ]
    for method in required_methods:
        require(method in source, f"generated U# binding is missing {method}")

    return {
        "inputs": len(inputs),
        "outputs": len(outputs),
        "buffers": len(buffers),
        "static_tensors": len(static_tensors),
        "static_bytes": static_bytes,
        "raw_heap_bytes": storage["raw_heap_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    summary = validate_artifact(args.artifact.resolve())
    print("tinyllama_prefill_artifact=passed " + " ".join(
        f"{name}={value}" for name, value in summary.items()
    ))


if __name__ == "__main__":
    main()
