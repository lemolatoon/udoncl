"""Dependency-free ONNX subset importer and Udon execution planner.

The wire reader accepts unknown protobuf fields and supports the tensor and
graph fields needed by inference models. The initial execution provider covers
the operators used by the checked-in MNIST network and records the exact
UdonDNN/UdonBLAS calls behind every scheduled Device pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def _varint(value: int) -> bytes:
    if value < 0:
        value &= (1 << 64) - 1
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def varint_field(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def bytes_field(number: int, value: bytes | str) -> bytes:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def packed_ints_field(number: int, values) -> bytes:
    payload = b"".join(_varint(value) for value in values)
    return bytes_field(number, payload)


def message_field(number: int, payload: bytes) -> bytes:
    return bytes_field(number, payload)


def _read_varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while position < len(data) and shift < 70:
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7
    raise ValueError("truncated or oversized protobuf varint")


def fields(data: bytes):
    position = 0
    while position < len(data):
        key, position = _read_varint(data, position)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise ValueError("protobuf field number zero is invalid")
        if wire == 0:
            value, position = _read_varint(data, position)
        elif wire == 1:
            if position + 8 > len(data):
                raise ValueError("truncated fixed64 protobuf field")
            value = data[position : position + 8]
            position += 8
        elif wire == 2:
            length, position = _read_varint(data, position)
            if position + length > len(data):
                raise ValueError("truncated length-delimited protobuf field")
            value = data[position : position + length]
            position += length
        elif wire == 5:
            if position + 4 > len(data):
                raise ValueError("truncated fixed32 protobuf field")
            value = data[position : position + 4]
            position += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def _string(value: bytes) -> str:
    return value.decode("utf-8")


def _packed_varints(value: bytes) -> list[int]:
    values = []
    position = 0
    while position < len(value):
        item, position = _read_varint(value, position)
        values.append(item)
    return values


DATA_TYPES = {
    1: ("float32", 4),
    2: ("uint8", 1),
    6: ("int32", 4),
    7: ("int64", 8),
    10: ("float16", 2),
    16: ("bfloat16", 2),
}


def tensor_proto(name: str, shape, values) -> bytes:
    raw = struct.pack(f"<{math.prod(shape)}f", *values)
    return (
        packed_ints_field(1, shape)
        + varint_field(2, 1)
        + bytes_field(8, name)
        + bytes_field(9, raw)
    )


def _string_pair(key: str, value: str) -> bytes:
    return bytes_field(1, key) + bytes_field(2, value)


def external_tensor_proto(
    name: str,
    shape,
    data_type: int,
    location: str,
    offset: int,
    length: int,
) -> bytes:
    if data_type not in DATA_TYPES:
        raise ValueError(f"unsupported external tensor data type {data_type}")
    if Path(location).is_absolute() or ".." in Path(location).parts:
        raise ValueError("external tensor location must be a safe relative path")
    payload = packed_ints_field(1, shape) + varint_field(2, data_type)
    payload += bytes_field(8, name)
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        payload += message_field(13, _string_pair(key, value))
    return payload + varint_field(14, 1)


def attribute_ints(name: str, values) -> bytes:
    return bytes_field(1, name) + packed_ints_field(8, values) + varint_field(20, 7)


def attribute_int(name: str, value: int) -> bytes:
    return bytes_field(1, name) + varint_field(3, value) + varint_field(20, 2)


def attribute_string(name: str, value: str) -> bytes:
    return bytes_field(1, name) + bytes_field(4, value) + varint_field(20, 3)


def node_proto(
    op_type: str, inputs, outputs, name: str, attributes=(), domain: str = ""
) -> bytes:
    payload = b"".join(bytes_field(1, value) for value in inputs)
    payload += b"".join(bytes_field(2, value) for value in outputs)
    payload += bytes_field(3, name) + bytes_field(4, op_type)
    payload += b"".join(message_field(5, attribute) for attribute in attributes)
    if domain:
        payload += bytes_field(7, domain)
    return payload


def value_info_proto(name: str, shape, data_type: int = 1) -> bytes:
    dimensions = b"".join(
        message_field(1, varint_field(1, dimension)) for dimension in shape
    )
    tensor_type = varint_field(1, data_type) + message_field(2, dimensions)
    return bytes_field(1, name) + message_field(2, message_field(1, tensor_type))


def model_proto(graph: bytes, opset: int = 13, custom_opsets=None) -> bytes:
    opset_imports = [bytes_field(1, "") + varint_field(2, opset)]
    for domain, version in (custom_opsets or {}).items():
        opset_imports.append(bytes_field(1, domain) + varint_field(2, version))
    return (
        varint_field(1, 8)
        + bytes_field(2, "udoncl")
        + message_field(7, graph)
        + b"".join(message_field(8, item) for item in opset_imports)
    )


def graph_proto(name: str, nodes, initializers, inputs, outputs) -> bytes:
    payload = b"".join(message_field(1, node) for node in nodes)
    payload += bytes_field(2, name)
    payload += b"".join(message_field(5, tensor) for tensor in initializers)
    payload += b"".join(message_field(11, value) for value in inputs)
    payload += b"".join(message_field(12, value) for value in outputs)
    return payload


def _parse_attribute(data: bytes) -> dict:
    result = {"name": "", "ints": []}
    for number, wire, value in fields(data):
        if number == 1 and wire == 2:
            result["name"] = _string(value)
        elif number == 3 and wire == 0:
            result["i"] = value
        elif number == 4 and wire == 2:
            result["s"] = _string(value)
        elif number == 8:
            result["ints"].extend(_packed_varints(value) if wire == 2 else [value])
    return result


def _parse_node(data: bytes) -> dict:
    result = {
        "inputs": [],
        "outputs": [],
        "name": "",
        "op_type": "",
        "domain": "",
        "attributes": {},
    }
    for number, wire, value in fields(data):
        if wire != 2:
            continue
        if number == 1:
            result["inputs"].append(_string(value))
        elif number == 2:
            result["outputs"].append(_string(value))
        elif number == 3:
            result["name"] = _string(value)
        elif number == 4:
            result["op_type"] = _string(value)
        elif number == 5:
            attribute = _parse_attribute(value)
            result["attributes"][attribute["name"]] = attribute
        elif number == 7:
            result["domain"] = _string(value)
    return result


def _parse_string_pair(data: bytes) -> tuple[str, str]:
    key, item = "", ""
    for number, wire, value in fields(data):
        if number == 1 and wire == 2:
            key = _string(value)
        elif number == 2 and wire == 2:
            item = _string(value)
    return key, item


def _parse_tensor(data: bytes) -> dict:
    result = {
        "name": "",
        "shape": [],
        "data_type": 0,
        "raw": b"",
        "external": {},
        "data_location": 0,
    }
    float_data = []
    for number, wire, value in fields(data):
        if number == 1:
            result["shape"].extend(_packed_varints(value) if wire == 2 else [value])
        elif number == 2 and wire == 0:
            result["data_type"] = value
        elif number == 4:
            if wire == 5:
                float_data.append(struct.unpack("<f", value)[0])
            elif wire == 2:
                float_data.extend(
                    struct.unpack(f"<{len(value) // 4}f", value)
                )
        elif number == 8 and wire == 2:
            result["name"] = _string(value)
        elif number == 9 and wire == 2:
            result["raw"] = value
        elif number == 13 and wire == 2:
            key, item = _parse_string_pair(value)
            if key in result["external"]:
                raise ValueError(f"duplicate external_data key {key!r}")
            result["external"][key] = item
        elif number == 14 and wire == 0:
            result["data_location"] = value
    if not result["raw"] and float_data:
        result["raw"] = struct.pack(f"<{len(float_data)}f", *float_data)
    if result["data_type"] not in DATA_TYPES:
        raise ValueError(
            f"tensor {result['name']!r} has unsupported data type {result['data_type']}"
        )
    expected = math.prod(result["shape"]) * DATA_TYPES[result["data_type"]][1]
    if result["data_location"] == 1:
        allowed = {"location", "offset", "length", "checksum", "basepath"}
        unknown = set(result["external"]) - allowed
        if unknown:
            raise ValueError(f"unsupported external_data keys {sorted(unknown)}")
        if "location" not in result["external"]:
            raise ValueError("external tensor has no location")
        location = Path(result["external"]["location"])
        if location.is_absolute() or ".." in location.parts:
            raise ValueError("external tensor location escapes the model directory")
        try:
            offset = int(result["external"].get("offset", "0"))
            length = int(result["external"].get("length", str(expected)))
        except ValueError as error:
            raise ValueError("external tensor offset and length must be integers") from error
        if offset < 0 or length < 0:
            raise ValueError("external tensor offset and length must be non-negative")
        result["external"].update(offset=offset, length=length)
        if result["raw"]:
            raise ValueError("external tensor unexpectedly contains raw_data")
    elif len(result["raw"]) != expected:
        raise ValueError(
            f"tensor {result['name']!r} has {len(result['raw'])} bytes; expected {expected}"
        )
    result["data_type_name"] = DATA_TYPES[result["data_type"]][0]
    result["logical_bytes"] = expected
    return result


def _parse_value_info(data: bytes) -> tuple[str, dict]:
    name, shape, elem_type = "", [], None
    for number, wire, value in fields(data):
        if number == 1 and wire == 2:
            name = _string(value)
        elif number == 2 and wire == 2:
            for type_number, type_wire, type_value in fields(value):
                if type_number != 1 or type_wire != 2:
                    continue
                for tensor_number, tensor_wire, tensor_value in fields(type_value):
                    if tensor_number == 1 and tensor_wire == 0:
                        elem_type = tensor_value
                    elif tensor_number == 2 and tensor_wire == 2:
                        for shape_number, shape_wire, shape_value in fields(tensor_value):
                            if shape_number != 1 or shape_wire != 2:
                                continue
                            dimension = None
                            for dim_number, dim_wire, dim_value in fields(shape_value):
                                if dim_number == 1 and dim_wire == 0:
                                    dimension = dim_value
                            if dimension is None:
                                raise ValueError("symbolic ONNX dimensions are not yet supported")
                            shape.append(dimension)
                if elem_type not in DATA_TYPES:
                    raise ValueError(f"value {name!r} has unsupported element type")
    return name, {"shape": shape, "data_type": elem_type}


def parse_model(data: bytes) -> dict:
    model = {"ir_version": 0, "opsets": {}, "graph": None}
    for number, wire, value in fields(data):
        if number == 1 and wire == 0:
            model["ir_version"] = value
        elif number == 7 and wire == 2:
            model["graph"] = value
        elif number == 8 and wire == 2:
            domain, version = "", 0
            for item_number, item_wire, item_value in fields(value):
                if item_number == 1 and item_wire == 2:
                    domain = _string(item_value)
                elif item_number == 2 and item_wire == 0:
                    version = item_value
            model["opsets"][domain] = version
    if model["graph"] is None:
        raise ValueError("ONNX ModelProto has no graph")
    if "" not in model["opsets"]:
        raise ValueError("ONNX model does not import the standard operator set")
    graph = {"name": "", "nodes": [], "initializers": {}, "inputs": {}, "outputs": {}}
    for number, wire, value in fields(model["graph"]):
        if number == 1 and wire == 2:
            graph["nodes"].append(_parse_node(value))
        elif number == 2 and wire == 2:
            graph["name"] = _string(value)
        elif number == 5 and wire == 2:
            tensor = _parse_tensor(value)
            if tensor["name"] in graph["initializers"]:
                raise ValueError(f"duplicate initializer {tensor['name']!r}")
            graph["initializers"][tensor["name"]] = tensor
        elif number in (11, 12) and wire == 2:
            name, value_info = _parse_value_info(value)
            (graph["inputs"] if number == 11 else graph["outputs"])[name] = value_info
    model["graph"] = graph
    return model


def validate_external_data(model: dict, base_directory: Path) -> dict:
    """Validate every external tensor path and byte range without loading it."""
    base = base_directory.resolve()
    files = {}
    tensors = 0
    for tensor in model["graph"]["initializers"].values():
        if tensor["data_location"] != 1:
            continue
        external = tensor["external"]
        target = (base / external["location"]).resolve()
        try:
            target.relative_to(base)
        except ValueError as error:
            raise ValueError("external tensor resolves outside the model directory") from error
        if not target.is_file():
            raise ValueError(f"external tensor file does not exist: {target}")
        size = files.setdefault(str(target), target.stat().st_size)
        end = external["offset"] + external["length"]
        if end > size:
            raise ValueError(
                f"external tensor {tensor['name']!r} ends at {end}, beyond {size} bytes"
            )
        tensors += 1
    return {"files": len(files), "tensors": tensors, "bytes": sum(files.values())}


def _broadcast(left, right):
    rank = max(len(left), len(right))
    left = [1] * (rank - len(left)) + list(left)
    right = [1] * (rank - len(right)) + list(right)
    result = []
    for a, b in zip(left, right):
        if a != b and a != 1 and b != 1:
            raise ValueError(f"incompatible broadcast dimensions {a} and {b}")
        result.append(max(a, b))
    return result


PROVIDERS = {
    "Conv": "udondnnConvolutionForward",
    "Add": "udondnnOpTensor",
    "Relu": "udondnnActivationForward",
    "AveragePool": "udondnnPoolingForward",
    "Flatten": "view",
    "Gemm": "udonblasSgemm",
}


LLAMA_DOMAIN = "com.lemolatoon.udoncl"


def llama_weight_spec(config: dict, storage_format: str = "q4") -> list[dict]:
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    kv = hidden * config["num_key_value_heads"] // config["num_attention_heads"]
    spec = [
        {"name": "model.embed_tokens.weight", "shape": [vocab, hidden]},
    ]
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        spec.extend(
            [
                {"name": f"{prefix}.input_layernorm.weight", "shape": [hidden]},
                {"name": f"{prefix}.self_attn.q_proj.weight", "shape": [hidden, hidden]},
                {"name": f"{prefix}.self_attn.k_proj.weight", "shape": [kv, hidden]},
                {"name": f"{prefix}.self_attn.v_proj.weight", "shape": [kv, hidden]},
                {"name": f"{prefix}.self_attn.o_proj.weight", "shape": [hidden, hidden]},
                {
                    "name": f"{prefix}.post_attention_layernorm.weight",
                    "shape": [hidden],
                },
                {"name": f"{prefix}.mlp.gate_proj.weight", "shape": [intermediate, hidden]},
                {"name": f"{prefix}.mlp.up_proj.weight", "shape": [intermediate, hidden]},
                {"name": f"{prefix}.mlp.down_proj.weight", "shape": [hidden, intermediate]},
            ]
        )
    spec.extend(
        [
            {"name": "model.norm.weight", "shape": [hidden]},
            {"name": "lm_head.weight", "shape": [vocab, hidden]},
        ]
    )
    for item in spec:
        item["parameters"] = math.prod(item["shape"])
        if storage_format == "float32":
            item["format"] = "float32"
            item["storage_bytes"] = item["parameters"] * 4
        elif len(item["shape"]) == 1:
            item["format"] = "bf16"
            item["storage_bytes"] = item["parameters"] * 2
        else:
            item["format"] = "q4_block32"
            blocks = (item["parameters"] + 31) // 32
            item["storage_bytes"] = blocks * 18  # 16 packed bytes + fp16 scale.
    return spec


def _compile_llama(model: dict) -> dict:
    graph = model["graph"]
    if len(graph["nodes"]) != 1:
        raise ValueError("the Udon Llama profile requires one fused decoder node")
    node = graph["nodes"][0]
    if node["domain"] != LLAMA_DOMAIN or node["op_type"] != "LlamaDecoder":
        raise ValueError("unsupported custom ONNX execution-provider node")
    architecture = node["attributes"].get("architecture", {}).get("s")
    llama_8b = {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "context_length": 1,
        "rope_theta": 500000,
        "q4_group_size": 32,
    }
    tiny_llama = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 16,
        "max_position_embeddings": 4,
        "context_length": 4,
        "rope_theta": 10000,
        "q4_group_size": 0,
    }
    profiles = {
        "llama-3.1-8b": (llama_8b, "q4", "llama-3.1-8b-q4"),
        "udon-tiny-llama": (tiny_llama, "float32", "udon-tiny-llama-f32"),
    }
    if architecture not in profiles:
        raise ValueError("unsupported LlamaDecoder architecture")
    required, storage_format, profile = profiles[architecture]
    config = {}
    for name, expected in required.items():
        attribute = node["attributes"].get(name)
        if attribute is None or attribute.get("i") != expected:
            raise ValueError(f"LlamaDecoder attribute {name!r} must be {expected}")
        config[name] = expected
    token_info = graph["inputs"].get("token_ids")
    logits_info = graph["outputs"].get("logits")
    if token_info != {
        "shape": [1, config["context_length"]],
        "data_type": 6,
    }:
        raise ValueError("Llama token_ids shape or type does not match the profile")
    if logits_info != {"shape": [1, config["vocab_size"]], "data_type": 1}:
        raise ValueError("Llama logits shape or type does not match the profile")
    spec = llama_weight_spec(config, storage_format)
    if node["inputs"] != ["token_ids"] + [item["name"] for item in spec]:
        raise ValueError("LlamaDecoder inputs do not match the canonical weight order")
    if node["outputs"] != ["logits"]:
        raise ValueError("LlamaDecoder must produce logits")
    expected_names = {item["name"] for item in spec}
    if set(graph["initializers"]) != expected_names:
        raise ValueError("Llama external initializers do not match the architecture")
    for item in spec:
        tensor = graph["initializers"][item["name"]]
        expected_type = {"bf16": 16, "q4_block32": 2, "float32": 1}[item["format"]]
        expected_shape = (
            [item["storage_bytes"]]
            if item["format"] == "q4_block32"
            else item["shape"]
        )
        if tensor["data_type"] != expected_type or tensor["shape"] != expected_shape:
            raise ValueError(f"invalid storage tensor for {item['name']!r}")
        if storage_format == "q4":
            if tensor["data_location"] != 1:
                raise ValueError(f"Llama weight {item['name']!r} must use external_data")
            if tensor["external"]["length"] != item["storage_bytes"]:
                raise ValueError(f"external_data length mismatch for {item['name']!r}")
        elif tensor["data_location"] != 0:
            raise ValueError("Tiny Llama validation weights must be embedded")

    tiny = architecture == "udon-tiny-llama"
    kernel_prefix = "tiny_llama" if tiny else "llama"
    first_id = 30 if tiny else 1000
    weight_matmul_provider = "udonblasSgemm" if tiny else "udonblasSgemvQ4"
    activation_matmul_provider = "udonblasSgemm"

    def q4_matmul(weight: str, output_features: int, input_features: int) -> dict:
        return {
            "provider": "udonblasSgemvQ4",
            "weight": weight,
            "weight_format": "q4_block32",
            "output_features": output_features,
            "input_features": input_features,
            "m_tile": 64,
            "k_tile": 128,
            "m_tiles": (output_features + 63) // 64,
            "k_tiles": (input_features + 127) // 128,
            "accumulation": "beta=0 for first K tile; beta=1 thereafter",
        }

    passes = [
        {
            "id": first_id,
            "kernel": f"{kernel_prefix}_embed" if tiny else "llama_embedding_gather",
            "layer": None,
            "providers": ["udondnnGatherForward"],
            "fragments": config["hidden_size"],
        }
    ]
    operations = [
        ("qkv", ["udondnnRmsNormalizationForward", weight_matmul_provider]),
        ("attention", ["udondnnSoftmaxForward", activation_matmul_provider]),
        ("project", [weight_matmul_provider, "udondnnOpTensor"]),
        (
            "ffn_up",
            [
                "udondnnRmsNormalizationForward",
                weight_matmul_provider,
                "udondnnActivationForward",
                "udondnnOpTensor",
            ],
        ),
        ("ffn_down", [weight_matmul_provider, "udondnnOpTensor"]),
    ]
    for layer in range(config["num_hidden_layers"]):
        for operation, providers in operations:
            device_pass = {
                "id": first_id + len(passes),
                "kernel": f"{kernel_prefix}_{operation}",
                "layer": layer,
                "providers": providers,
                "fragments": (
                    config["intermediate_size"]
                    if operation == "ffn_up"
                    else config["hidden_size"]
                ),
            }
            if not tiny:
                prefix = f"model.layers.{layer}"
                if operation == "qkv":
                    kv = (
                        config["hidden_size"]
                        * config["num_key_value_heads"]
                        // config["num_attention_heads"]
                    )
                    device_pass["matmuls"] = [
                        q4_matmul(
                            f"{prefix}.self_attn.q_proj.weight",
                            config["hidden_size"],
                            config["hidden_size"],
                        ),
                        q4_matmul(
                            f"{prefix}.self_attn.k_proj.weight",
                            kv,
                            config["hidden_size"],
                        ),
                        q4_matmul(
                            f"{prefix}.self_attn.v_proj.weight",
                            kv,
                            config["hidden_size"],
                        ),
                    ]
                elif operation == "project":
                    device_pass["matmuls"] = [
                        q4_matmul(
                            f"{prefix}.self_attn.o_proj.weight",
                            config["hidden_size"],
                            config["hidden_size"],
                        )
                    ]
                elif operation == "ffn_up":
                    device_pass["matmuls"] = [
                        q4_matmul(
                            f"{prefix}.mlp.gate_proj.weight",
                            config["intermediate_size"],
                            config["hidden_size"],
                        ),
                        q4_matmul(
                            f"{prefix}.mlp.up_proj.weight",
                            config["intermediate_size"],
                            config["hidden_size"],
                        ),
                    ]
                elif operation == "ffn_down":
                    device_pass["matmuls"] = [
                        q4_matmul(
                            f"{prefix}.mlp.down_proj.weight",
                            config["hidden_size"],
                            config["intermediate_size"],
                        )
                    ]
            passes.append(device_pass)
    logits_pass = {
        "id": first_id + len(passes),
        "kernel": f"{kernel_prefix}_logits",
        "layer": None,
        "providers": ["udondnnRmsNormalizationForward", weight_matmul_provider],
        "fragments": config["vocab_size"],
    }
    if not tiny:
        logits_pass["matmuls"] = [
            q4_matmul(
                "lm_head.weight", config["vocab_size"], config["hidden_size"]
            )
        ]
    passes.append(logits_pass)
    parameter_count = sum(item["parameters"] for item in spec)
    weight_bytes = sum(item["storage_bytes"] for item in spec)
    return {
        "schema_version": 1,
        "runtime": "udon-onnx",
        "profile": profile,
        "graph": graph["name"],
        "ir_version": model["ir_version"],
        "opset": model["opsets"][""],
        "custom_opset": model["opsets"].get(LLAMA_DOMAIN),
        "config": config,
        "nodes": [
            {
                "name": node["name"],
                "domain": node["domain"],
                "op_type": node["op_type"],
                "provider": "UdonLlamaExecutionProvider",
                "output": "logits",
                "shape": [1, config["vocab_size"]],
            }
        ],
        "passes": passes,
        "weights": {
            "parameters": parameter_count,
            "storage_bytes": weight_bytes,
            "format": (
                "q4_block32+bfloat16_norm" if storage_format == "q4" else "float32"
            ),
            "tensors": len(spec),
            "page_bytes": 16 * 1024 * 1024,
            "pages": (weight_bytes + 16 * 1024 * 1024 - 1) // (16 * 1024 * 1024),
        },
        "blas": {
            "weight_provider": weight_matmul_provider,
            "activation_provider": activation_matmul_provider,
            **(
                {
                    "q4_tile": {
                        "m": 64,
                        "k": 128,
                        "block_values": 32,
                        "expanded_block_words": 17,
                        "accumulator": "float32",
                    }
                }
                if not tiny
                else {}
            ),
        },
        "initializers": {
            item["name"]: {
                "logical_shape": item["shape"],
                "format": item["format"],
                "bytes": item["storage_bytes"],
                "external": graph["initializers"][item["name"]]["external"],
            }
            for item in spec
        },
        "inputs": {name: info["shape"] for name, info in graph["inputs"].items()},
        "outputs": {name: info["shape"] for name, info in graph["outputs"].items()},
    }


def _compile_mnist(model: dict) -> dict:
    graph = model["graph"]
    shapes = {name: tensor["shape"] for name, tensor in graph["initializers"].items()}
    shapes.update({name: info["shape"] for name, info in graph["inputs"].items()})
    nodes = []
    produced = set(shapes)
    for node in graph["nodes"]:
        operation = node["op_type"]
        if operation not in PROVIDERS:
            raise ValueError(f"unsupported ONNX operator {operation!r}")
        if any(name not in produced for name in node["inputs"]):
            missing = [name for name in node["inputs"] if name not in produced]
            raise ValueError(f"node {node['name']!r} has unavailable inputs {missing}")
        inputs = [shapes[name] for name in node["inputs"]]
        attributes = node["attributes"]
        if operation == "Conv":
            image, weight = inputs[:2]
            pads = attributes.get("pads", {"ints": [0, 0, 0, 0]})["ints"]
            strides = attributes.get("strides", {"ints": [1, 1]})["ints"]
            dilations = attributes.get("dilations", {"ints": [1, 1]})["ints"]
            if len(image) != 4 or len(weight) != 4 or image[1] != weight[1]:
                raise ValueError("Conv requires compatible rank-4 NCHW tensors")
            output = [
                image[0],
                weight[0],
                1 + (image[2] + pads[0] + pads[2] - dilations[0] * (weight[2] - 1) - 1) // strides[0],
                1 + (image[3] + pads[1] + pads[3] - dilations[1] * (weight[3] - 1) - 1) // strides[1],
            ]
        elif operation in ("Add",):
            output = _broadcast(inputs[0], inputs[1])
        elif operation == "Relu":
            output = inputs[0]
        elif operation == "AveragePool":
            image = inputs[0]
            kernel = attributes["kernel_shape"]["ints"]
            strides = attributes.get("strides", {"ints": [1, 1]})["ints"]
            pads = attributes.get("pads", {"ints": [0, 0, 0, 0]})["ints"]
            output = [
                image[0],
                image[1],
                1 + (image[2] + pads[0] + pads[2] - kernel[0]) // strides[0],
                1 + (image[3] + pads[1] + pads[3] - kernel[1]) // strides[1],
            ]
        elif operation == "Flatten":
            axis = attributes.get("axis", {"i": 1})["i"]
            output = [math.prod(inputs[0][:axis]), math.prod(inputs[0][axis:])]
        elif operation == "Gemm":
            left, right = inputs[:2]
            trans_a = attributes.get("transA", {"i": 0})["i"]
            trans_b = attributes.get("transB", {"i": 0})["i"]
            m, k = (left[1], left[0]) if trans_a else (left[0], left[1])
            right_k, n = (right[1], right[0]) if trans_b else (right[0], right[1])
            if k != right_k:
                raise ValueError("Gemm inner dimensions do not match")
            output = [m, n]
        else:
            raise AssertionError(operation)
        if len(node["outputs"]) != 1:
            raise ValueError(f"operator {operation} must have exactly one output")
        output_name = node["outputs"][0]
        if output_name in produced:
            raise ValueError(f"duplicate graph value {output_name!r}")
        shapes[output_name] = output
        produced.add(output_name)
        nodes.append(
            {
                "name": node["name"],
                "op_type": operation,
                "provider": PROVIDERS[operation],
                "inputs": node["inputs"],
                "output": output_name,
                "shape": output,
            }
        )
    for name, value_info in graph["outputs"].items():
        expected = value_info["shape"]
        if name not in shapes or shapes[name] != expected:
            raise ValueError(
                f"graph output {name!r} inferred as {shapes.get(name)}; expected {expected}"
            )
    expected_ops = ["Conv", "Add", "Relu", "AveragePool", "Flatten", "Gemm", "Add"]
    if [node["op_type"] for node in nodes] != expected_ops:
        raise ValueError("initial Udon execution provider requires the MNIST operator sequence")
    passes = [
        {
            "id": 20,
            "kernel": "mnist_conv",
            "providers": [nodes[index]["provider"] for index in (0, 1, 2)],
            "fusion": "Conv+Add+Relu",
            "output": nodes[2]["output"],
            "shape": nodes[2]["shape"],
            "fragments": math.prod(nodes[2]["shape"]),
        },
        {
            "id": 21,
            "kernel": "mnist_pool",
            "providers": [nodes[3]["provider"]],
            "fusion": None,
            "output": nodes[3]["output"],
            "shape": nodes[3]["shape"],
            "fragments": math.prod(nodes[3]["shape"]),
        },
        {
            "id": 22,
            "kernel": "mnist_dense",
            "providers": [nodes[index]["provider"] for index in (5, 6)],
            "fusion": "Gemm+Add",
            "output": nodes[6]["output"],
            "shape": nodes[6]["shape"],
            "fragments": math.prod(nodes[6]["shape"]),
        },
    ]
    return {
        "schema_version": 1,
        "runtime": "udon-onnx",
        "graph": graph["name"],
        "ir_version": model["ir_version"],
        "opset": model["opsets"][""],
        "nodes": nodes,
        "passes": passes,
        "initializers": {
            name: {
                "shape": tensor["shape"],
                "bytes": len(tensor["raw"]),
                "sha256": hashlib.sha256(tensor["raw"]).hexdigest(),
            }
            for name, tensor in graph["initializers"].items()
        },
        "inputs": {name: info["shape"] for name, info in graph["inputs"].items()},
        "outputs": {name: info["shape"] for name, info in graph["outputs"].items()},
    }


class ExecutionProviderRegistry:
    """Ordered public registry for graph-level Udon execution providers."""

    def __init__(self):
        self._providers = []

    def register(self, name, supports, compile_graph):
        if not name or any(item[0] == name for item in self._providers):
            raise ValueError(f"duplicate or empty execution provider name {name!r}")
        self._providers.append((name, supports, compile_graph))

    def compile(self, model: dict) -> dict:
        matches = [item for item in self._providers if item[1](model)]
        if not matches:
            operators = [
                (node["domain"], node["op_type"]) for node in model["graph"]["nodes"]
            ]
            raise ValueError(f"no Udon execution provider accepts graph operators {operators}")
        if len(matches) != 1:
            raise ValueError(
                "multiple Udon execution providers accept this graph: "
                + ", ".join(item[0] for item in matches)
            )
        name, _, compiler = matches[0]
        plan = compiler(model)
        plan["execution_provider"] = name
        return plan


def _supports_mnist(model: dict) -> bool:
    return [
        (node["domain"], node["op_type"]) for node in model["graph"]["nodes"]
    ] == [
        ("", "Conv"),
        ("", "Add"),
        ("", "Relu"),
        ("", "AveragePool"),
        ("", "Flatten"),
        ("", "Gemm"),
        ("", "Add"),
    ]


def _supports_llama(model: dict) -> bool:
    nodes = model["graph"]["nodes"]
    return len(nodes) == 1 and nodes[0]["domain"] == LLAMA_DOMAIN and nodes[0][
        "op_type"
    ] == "LlamaDecoder"


def default_registry() -> ExecutionProviderRegistry:
    registry = ExecutionProviderRegistry()
    registry.register("UdonLlamaExecutionProvider", _supports_llama, _compile_llama)
    registry.register("UdonMnistExecutionProvider", _supports_mnist, _compile_mnist)
    return registry


def compile_model(model: dict, registry=None) -> dict:
    return (registry or default_registry()).compile(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--external-data-dir",
        type=Path,
        help="validate external tensor paths and ranges before writing the plan",
    )
    args = parser.parse_args()
    model = parse_model(args.model.read_bytes())
    external = (
        validate_external_data(model, args.external_data_dir)
        if args.external_data_dir
        else None
    )
    plan = compile_model(model)
    if external is not None:
        plan["external_data_validation"] = external
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        f"udon_onnx_nodes={len(plan['nodes'])} device_passes={len(plan['passes'])} "
        f"output_shape={plan['outputs']['logits']}"
    )


if __name__ == "__main__":
    main()
