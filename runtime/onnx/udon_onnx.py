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


def tensor_int64_proto(name: str, shape, values) -> bytes:
    raw = struct.pack(f"<{math.prod(shape)}q", *values)
    return (
        packed_ints_field(1, shape)
        + varint_field(2, 7)
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


def attribute_float(name: str, value: float) -> bytes:
    return (
        bytes_field(1, name)
        + _varint((2 << 3) | 5)
        + struct.pack("<f", value)
        + varint_field(20, 1)
    )


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
        elif number == 2 and wire == 5:
            result["f"] = struct.unpack("<f", value)[0]
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
    "Add": "udonatenAdd",
    "Mul": "udonatenMul",
    "Sub": "udonatenSub",
    "Relu": "udonatenRelu",
    "Sigmoid": "udonatenSigmoid",
    "Tanh": "udonatenTanh",
    "AveragePool": "udondnnPoolingForward",
    "Softmax": "udondnnSoftmaxForward",
    "RMSNormalization": "udonatenRmsNorm",
    "Gather": "udonatenEmbedding",
    "Flatten": "view",
    "Identity": "view",
    "Reshape": "view",
    "Squeeze": "view",
    "Unsqueeze": "view",
    "Tile": "udonatenTile",
    "Gemm": "udonblasSgemm",
    "MatMul": "udonblasSgemm",
}

POINTWISE_PROVIDERS = {
    "udonatenAdd",
    "udonatenMul",
    "udonatenSub",
    "udonatenRelu",
    "udonatenSigmoid",
    "udonatenTanh",
}
POINTWISE_TILE_WORDS = 512
ARENA_POINTWISE_WORDS = 16384
ARENA_POINTWISE_OPERATIONS = {
    "udonatenAdd": 0,
    "udonatenMul": 1,
    "udonatenSigmoid": 2,
}
RMS_NORM_TILE_WORDS = 512


LLAMA_DOMAIN = "com.lemolatoon.udoncl"

EXTENSION_PROVIDERS = {
    (LLAMA_DOMAIN, "WeightInt4packMM"): "udonatenWeightInt4packMm",
}


def _operator_provider(node: dict) -> str | None:
    if node["domain"] == "":
        return PROVIDERS.get(node["op_type"])
    return EXTENSION_PROVIDERS.get((node["domain"], node["op_type"]))


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
            if item["shape"][1] % 32:
                raise ValueError("ATen INT4 weights require K divisible by 32")
            item["format"] = "q4_int4pack"
            item["packed_storage_bytes"] = item["parameters"] // 2
            item["scales_and_zeros_name"] = item["name"] + ".scales_and_zeros"
            item["scales_and_zeros_shape"] = [
                item["shape"][1] // 32,
                item["shape"][0],
                2,
            ]
            item["scales_and_zeros_storage_bytes"] = (
                math.prod(item["scales_and_zeros_shape"]) * 2
            )
            item["storage_bytes"] = (
                item["packed_storage_bytes"]
                + item["scales_and_zeros_storage_bytes"]
            )
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
    expected_inputs = ["token_ids"]
    for item in spec:
        expected_inputs.append(item["name"])
        if item["format"] == "q4_int4pack":
            expected_inputs.append(item["scales_and_zeros_name"])
    if node["inputs"] != expected_inputs:
        raise ValueError("LlamaDecoder inputs do not match the canonical weight order")
    if node["outputs"] != ["logits"]:
        raise ValueError("LlamaDecoder must produce logits")
    expected_names = {item["name"] for item in spec}
    expected_names.update(
        item["scales_and_zeros_name"]
        for item in spec
        if item["format"] == "q4_int4pack"
    )
    if set(graph["initializers"]) != expected_names:
        raise ValueError("Llama external initializers do not match the architecture")
    for item in spec:
        tensor = graph["initializers"][item["name"]]
        expected_type = {"bf16": 16, "q4_int4pack": 2, "float32": 1}[
            item["format"]
        ]
        expected_shape = (
            [item["shape"][0], item["shape"][1] // 2]
            if item["format"] == "q4_int4pack"
            else item["shape"]
        )
        if tensor["data_type"] != expected_type or tensor["shape"] != expected_shape:
            raise ValueError(f"invalid storage tensor for {item['name']!r}")
        if storage_format == "q4":
            if tensor["data_location"] != 1:
                raise ValueError(f"Llama weight {item['name']!r} must use external_data")
            expected_length = item.get("packed_storage_bytes", item["storage_bytes"])
            if tensor["external"]["length"] != expected_length:
                raise ValueError(f"external_data length mismatch for {item['name']!r}")
            if item["format"] == "q4_int4pack":
                qparams = graph["initializers"][item["scales_and_zeros_name"]]
                if (
                    qparams["data_type"] != 16
                    or qparams["shape"] != item["scales_and_zeros_shape"]
                    or qparams["data_location"] != 1
                    or qparams["external"]["length"]
                    != item["scales_and_zeros_storage_bytes"]
                ):
                    raise ValueError(
                        f"invalid scales_and_zeros for {item['name']!r}"
                    )
        elif tensor["data_location"] != 0:
            raise ValueError("Tiny Llama validation weights must be embedded")

    tiny = architecture == "udon-tiny-llama"
    kernel_prefix = "tiny_llama" if tiny else "llama"
    first_id = 30 if tiny else 1000
    weight_matmul_provider = (
        "udonblasSgemm" if tiny else "udonatenWeightInt4packMm"
    )
    activation_matmul_provider = "udonblasSgemm"

    def q4_matmul(weight: str, output_features: int, input_features: int) -> dict:
        return {
            "provider": "udonatenWeightInt4packMm",
            "weight": weight,
            "scales_and_zeros": weight + ".scales_and_zeros",
            "group_size": 32,
            "weight_format": "q4_int4pack",
            "output_features": output_features,
            "input_features": input_features,
            "m_tile": 64,
            "k_tile": 128,
            "m_tiles": (output_features + 63) // 64,
            "k_tiles": (input_features + 127) // 128,
            "accumulation": "first K tile replaces; later K tiles add internally",
        }

    passes = [
        {
            "id": first_id,
            "kernel": f"{kernel_prefix}_embed" if tiny else "llama_embedding_gather",
            "layer": None,
            "providers": ["udonatenEmbedding"],
            "fragments": config["hidden_size"],
        }
    ]
    operations = [
        ("qkv", ["udonatenRmsNorm", weight_matmul_provider]),
        ("attention", ["udondnnSoftmaxForward", activation_matmul_provider]),
        ("project", [weight_matmul_provider, "udonatenAdd"]),
        (
            "ffn_up",
            [
                "udonatenRmsNorm",
                weight_matmul_provider,
                "udonatenSigmoid",
                "udonatenMul",
            ],
        ),
        ("ffn_down", [weight_matmul_provider, "udonatenAdd"]),
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
        "providers": ["udonatenRmsNorm", weight_matmul_provider],
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
    initializers = {}
    for item in spec:
        packed_bytes = item.get("packed_storage_bytes", item["storage_bytes"])
        initializers[item["name"]] = {
            "logical_shape": item["shape"],
            "format": item["format"],
            "bytes": packed_bytes,
            "external": graph["initializers"][item["name"]]["external"],
        }
        if item["format"] == "q4_int4pack":
            initializers[item["scales_and_zeros_name"]] = {
                "logical_shape": item["scales_and_zeros_shape"],
                "format": "bf16_scale_zero",
                "bytes": item["scales_and_zeros_storage_bytes"],
                "external": graph["initializers"][item["scales_and_zeros_name"]][
                    "external"
                ],
            }
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
                "aten_int4pack+bfloat16_scale_zero" if storage_format == "q4" else "float32"
            ),
            "tensors": len(initializers),
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
                        "qparams": "[K/group_size,N,2] bfloat16",
                        "accumulator": "float32",
                    }
                }
                if not tiny
                else {}
            ),
        },
        "initializers": initializers,
        "inputs": {name: info["shape"] for name, info in graph["inputs"].items()},
        "outputs": {name: info["shape"] for name, info in graph["outputs"].items()},
    }


def _gemm_schedule(node: dict, values: dict, tile_size: int = 16) -> list[dict]:
    """Return the physical SGEMM tiles used by every AOT backend."""
    inputs = node["inputs"]
    if len(inputs) not in (2, 3):
        raise ValueError("ONNX Gemm requires two or three inputs")
    alpha = node["attributes"].get("alpha", {"f": 1.0})["f"]
    beta = node["attributes"].get("beta", {"f": 1.0})["f"]
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("ONNX Gemm alpha and beta must be finite")
    left_shape, right_shape = values[inputs[0]], values[inputs[1]]
    if len(left_shape) != 2 or len(right_shape) != 2:
        raise ValueError("Udon ONNX Gemm requires rank-2 input tensors")
    trans_a = node["attributes"].get("transA", {"i": 0})["i"]
    trans_b = node["attributes"].get("transB", {"i": 0})["i"]
    m, k = (left_shape[1], left_shape[0]) if trans_a else left_shape
    right_k, n = (right_shape[1], right_shape[0]) if trans_b else right_shape
    if k != right_k:
        raise ValueError("Gemm AOT dimensions do not match")
    tiles = []
    for m0 in range(0, m, tile_size):
        for n0 in range(0, n, tile_size):
            for k0 in range(0, k, tile_size):
                tiles.append(
                    {
                        "m_offset": m0,
                        "n_offset": n0,
                        "k_offset": k0,
                        "m": min(tile_size, m - m0),
                        "n": min(tile_size, n - n0),
                        "k": min(tile_size, k - k0),
                        "alpha": alpha,
                        "beta": 0 if k0 == 0 else 1,
                    }
                )
    return tiles


def _softmax_mode(node: dict, shape: list[int]) -> str:
    axis = node["attributes"].get("axis", {"i": -1})["i"]
    axis = axis + len(shape) if axis < 0 else axis
    if len(shape) in (1, 2) and axis == len(shape) - 1:
        return "UDONDNN_SOFTMAX_MODE_INSTANCE"
    if len(shape) == 4 and axis == 1:
        return "UDONDNN_SOFTMAX_MODE_CHANNEL"
    raise ValueError(
        f"Softmax axis {axis} for rank-{len(shape)} cannot map to a cuDNN softmax mode"
    )


def _unravel(index: int, shape: list[int]) -> list[int]:
    coordinates = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        coordinates[axis] = index % shape[axis]
        index //= shape[axis]
    return coordinates


def _batch_offset(coordinates: list[int], shape: list[int]) -> int:
    padded = [1] * (len(coordinates) - len(shape)) + list(shape)
    offset = 0
    for coordinate, dimension in zip(coordinates, padded):
        offset = offset * dimension + (0 if dimension == 1 else coordinate)
    return offset


def _matmul_shape(left: list[int], right: list[int]) -> list[int]:
    if not 2 <= len(left) <= 4 or not 2 <= len(right) <= 4:
        raise ValueError("Udon ONNX MatMul currently requires rank-2 through rank-4 tensors")
    if left[-1] != right[-2]:
        raise ValueError("MatMul inner dimensions do not match")
    return _broadcast(left[:-2], right[:-2]) + [left[-2], right[-1]]


def _matmul_schedule(node: dict, values: dict, tile_size: int = 16) -> list[dict]:
    left_shape, right_shape = (values[name] for name in node["inputs"])
    output_shape = _matmul_shape(left_shape, right_shape)
    batch_shape = output_shape[:-2]
    m, k, n = left_shape[-2], left_shape[-1], right_shape[-1]
    tiles = []
    batch_count = math.prod(batch_shape) if batch_shape else 1
    for batch in range(batch_count):
        coordinates = _unravel(batch, batch_shape)
        left_base = _batch_offset(coordinates, left_shape[:-2]) * m * k
        right_base = _batch_offset(coordinates, right_shape[:-2]) * k * n
        output_base = batch * m * n
        for m0 in range(0, m, tile_size):
            for n0 in range(0, n, tile_size):
                for k0 in range(0, k, tile_size):
                    tiles.append(
                        {
                            "batch": batch,
                            "m_offset": m0,
                            "n_offset": n0,
                            "k_offset": k0,
                            "m": min(tile_size, m - m0),
                            "n": min(tile_size, n - n0),
                            "k": min(tile_size, k - k0),
                            "left_offset": left_base + m0 * k + k0,
                            "right_offset": right_base + k0 * n + n0,
                            "output_offset": output_base + m0 * n + n0,
                            "alpha": 1.0,
                            "beta": 0 if k0 == 0 else 1,
                        }
                    )
    return tiles


def _pointwise_tile_words(remaining: int) -> int:
    for candidate in range(min(POINTWISE_TILE_WORDS, remaining), 0, -1):
        for n in range(1, 16):
            for c in range(1, 16):
                for h in range(1, 16):
                    divisor = n * c * h
                    if candidate % divisor == 0 and candidate // divisor <= 15:
                        return candidate
    raise AssertionError("every positive word count has a packable unit tile")


def _pointwise_schedule(node: dict, values: dict) -> list[dict | None]:
    """Describe the private Device launches behind one public ATen call.

    Linear slicing is valid only when every tensor has the output shape. A
    broadcast operand needs coordinate-aware addressing and remains one
    logical pass until that backend is implemented.
    """
    words = math.prod(node["shape"])
    exact_shape = all(values[name] == node["shape"] for name in node["inputs"])
    if (
        node["provider"] in ARENA_POINTWISE_OPERATIONS
        and POINTWISE_TILE_WORDS < words <= ARENA_POINTWISE_WORDS
        and exact_shape
    ):
        return [
            {
                "arena_words": words,
                "storage_words": ARENA_POINTWISE_WORDS,
                "width": 8192,
                "height": 2,
                "operation": ARENA_POINTWISE_OPERATIONS[node["provider"]],
            }
        ]
    if (
        node["provider"] not in POINTWISE_PROVIDERS
        or words <= POINTWISE_TILE_WORDS
        or not exact_shape
    ):
        return [None]
    tiles = []
    offset = 0
    while offset < words:
        tile_words = _pointwise_tile_words(words - offset)
        tiles.append(
            {
                "offset": offset,
                "words": tile_words,
                "tile_words": POINTWISE_TILE_WORDS,
            }
        )
        offset += tile_words
    return tiles


def _initializer_ints(graph: dict, name: str) -> list[int]:
    tensor = graph["initializers"].get(name)
    if tensor is None or tensor["data_location"] == 1:
        raise ValueError(f"shape input {name!r} must be an embedded initializer")
    if tensor["data_type"] == 7:
        code, width = "q", 8
    elif tensor["data_type"] == 6:
        code, width = "i", 4
    else:
        raise ValueError(f"shape input {name!r} must contain int32 or int64 values")
    count = math.prod(tensor["shape"])
    if len(tensor["raw"]) != count * width:
        raise ValueError(f"shape input {name!r} has an invalid byte length")
    return list(struct.unpack(f"<{count}{code}", tensor["raw"]))


def _reshape_shape(source: list[int], requested: list[int], allow_zero: bool) -> list[int]:
    result = []
    inferred = None
    known = 1
    for axis, dimension in enumerate(requested):
        if dimension == 0 and not allow_zero:
            if axis >= len(source):
                raise ValueError("Reshape zero dimension has no corresponding input axis")
            dimension = source[axis]
        if dimension == -1:
            if inferred is not None:
                raise ValueError("Reshape may infer at most one dimension")
            inferred = axis
            result.append(1)
        elif dimension < 0:
            raise ValueError("Reshape dimensions must be non-negative or -1")
        else:
            result.append(dimension)
            known *= dimension
    words = math.prod(source)
    if inferred is not None:
        if known == 0 or words % known:
            raise ValueError("Reshape inferred dimension is not integral")
        result[inferred] = words // known
    elif math.prod(result) != words:
        raise ValueError("Reshape changes the tensor element count")
    return result


def _linear_texture_layout(words: int) -> dict:
    if not isinstance(words, int) or words <= 0:
        raise ValueError("linear texture word count must be positive")
    return {"width": min(8192, words), "height": math.ceil(words / 8192)}


def _plan_texture_arena(
    graph: dict,
    calls: list[dict],
    shapes: dict,
    value_types: dict,
    scratch_buffers: list[dict],
) -> dict:
    """Color activation lifetimes into reusable linear Device textures.

    Lifetimes are deliberately inclusive. An input whose last consumer is a
    call cannot share that call's output slot, because fragment kernels do not
    promise safe in-place reads and writes. Shape-only views alias their source
    and therefore consume no additional texture.
    """
    initializers = set(graph["initializers"])
    graph_inputs = set(graph["inputs"])
    graph_outputs = set(graph["outputs"])
    aliases = {
        call["output"]: call["inputs"][0]
        for call in calls
        if call.get("kind") == "view"
    }

    def root(name: str) -> str:
        visited = set()
        while name in aliases:
            if name in visited:
                raise ValueError("AOT view aliases contain a cycle")
            visited.add(name)
            name = aliases[name]
        return name

    lifetimes = {}

    def touch(name: str, call_index: int, *, produced: bool = False):
        physical = root(name)
        if physical in initializers:
            return
        if physical not in shapes:
            raise ValueError(f"AOT value {physical!r} has no inferred shape")
        data_type = value_types.get(physical)
        if data_type not in ("float32", "int32"):
            raise ValueError(
                f"AOT activation {physical!r} has unsupported Device type {data_type!r}"
            )
        start = call_index if produced else (-1 if physical in graph_inputs else call_index)
        lifetime = lifetimes.setdefault(
            physical,
            {
                "name": physical,
                "first_call": start,
                "last_call": call_index,
                "words": math.prod(shapes[physical]),
                "data_type": data_type,
                "aliases": [],
                "scratch": False,
            },
        )
        lifetime["first_call"] = min(lifetime["first_call"], start)
        lifetime["last_call"] = max(lifetime["last_call"], call_index)

    for call_index, call in enumerate(calls):
        if call.get("index") != call_index:
            raise ValueError("AOT call indexes must be contiguous and topological")
        for name in call.get("inputs", []):
            touch(name, call_index)
        if call.get("kind") != "view":
            touch(call["output"], call_index, produced=True)

    for name in graph_outputs:
        physical = root(name)
        if physical in initializers:
            continue
        if physical not in lifetimes:
            touch(name, len(calls))
        lifetimes[physical]["last_call"] = len(calls)

    for alias, source in aliases.items():
        physical = root(source)
        if physical in lifetimes:
            lifetimes[physical]["aliases"].append(alias)

    scratch_names = set()
    for scratch in scratch_buffers:
        name = scratch.get("name")
        words = scratch.get("words")
        if not isinstance(name, str) or not name or name in scratch_names:
            raise ValueError("scratch buffer names must be unique non-empty strings")
        if not isinstance(words, int) or words <= 0:
            raise ValueError("scratch buffer word counts must be positive integers")
        if scratch.get("data_type") != "float32":
            raise ValueError("Device scratch buffers must use float32 words")
        suffix = "__rms_partials"
        if not name.endswith(suffix):
            raise ValueError("scratch buffer is not associated with an RMSNorm output")
        output = name[: -len(suffix)]
        matches = [call["index"] for call in calls if call.get("output") == output]
        if len(matches) != 1:
            raise ValueError("scratch buffer must have exactly one producing AOT call")
        call_index = matches[0]
        scratch_name = "$scratch:" + name
        lifetimes[scratch_name] = {
            "name": scratch_name,
            "first_call": call_index,
            "last_call": call_index,
            "words": words,
            "data_type": scratch.get("data_type"),
            "aliases": [],
            "scratch": True,
            "public_name": name,
        }
        scratch_names.add(name)

    slots = []
    bindings = {}
    ordered = sorted(
        lifetimes.values(), key=lambda item: (item["first_call"], item["name"])
    )
    for lifetime in ordered:
        reusable = [
            (max(slot["capacity_words"], lifetime["words"]), slot["index"])
            for slot in slots
            if slot["last_call"] < lifetime["first_call"]
        ]
        if reusable:
            _, slot_index = min(reusable)
            slot = slots[slot_index]
            slot["capacity_words"] = max(
                slot["capacity_words"], lifetime["words"]
            )
            slot["last_call"] = lifetime["last_call"]
        else:
            slot_index = len(slots)
            slot = {
                "index": slot_index,
                "capacity_words": lifetime["words"],
                "last_call": lifetime["last_call"],
                "values": [],
            }
            slots.append(slot)
        public_name = lifetime.get("public_name", lifetime["name"])
        slot["values"].append(public_name)
        binding = {
            "slot": slot_index,
            "offset_words": 0,
            "words": lifetime["words"],
            "data_type": lifetime["data_type"],
            "first_call": lifetime["first_call"],
            "last_call": lifetime["last_call"],
        }
        bindings[public_name] = binding
        for alias in sorted(lifetime["aliases"]):
            bindings[alias] = {
                **binding,
                "words": math.prod(shapes[alias]),
                "alias_of": lifetime["name"],
            }

    for slot in slots:
        slot.update(_linear_texture_layout(slot["capacity_words"]))
        del slot["last_call"]
    total_words = sum(slot["capacity_words"] for slot in slots)
    return {
        "abi": "linear-rgba32-words",
        "max_texture_width": 8192,
        "slot_count": len(slots),
        "total_words": total_words,
        "total_bytes": total_words * 4,
        "slots": slots,
        "bindings": bindings,
    }


def _compile_standard_graph(model: dict, *, fuse_mnist: bool) -> dict:
    graph = model["graph"]
    shapes = {name: tensor["shape"] for name, tensor in graph["initializers"].items()}
    shapes.update({name: info["shape"] for name, info in graph["inputs"].items()})
    value_types = {
        name: tensor["data_type_name"] for name, tensor in graph["initializers"].items()
    }
    value_types.update(
        {name: DATA_TYPES[info["data_type"]][0] for name, info in graph["inputs"].items()}
    )
    nodes = []
    produced = set(shapes)
    for node in graph["nodes"]:
        operation = node["op_type"]
        provider = _operator_provider(node)
        if provider is None:
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
        elif operation in ("Add", "Mul", "Sub"):
            output = _broadcast(inputs[0], inputs[1])
        elif operation in ("Relu", "Sigmoid", "Tanh"):
            output = inputs[0]
        elif operation == "Softmax":
            _softmax_mode(node, inputs[0])
            output = inputs[0]
        elif operation == "RMSNormalization":
            if len(inputs) != 2:
                raise ValueError("ONNX RMSNormalization requires X and scale inputs")
            axis = attributes.get("axis", {"i": -1})["i"]
            axis = axis + len(inputs[0]) if axis < 0 else axis
            epsilon = attributes.get("epsilon", {"f": 1.0e-5})["f"]
            stash_type = attributes.get("stash_type", {"i": 1})["i"]
            if axis < 0 or axis >= len(inputs[0]):
                raise ValueError("RMSNormalization axis is outside the input rank")
            if stash_type != 1 or not math.isfinite(epsilon) or epsilon < 0.0:
                raise ValueError("Udon RMSNormalization requires finite epsilon and stash_type=1")
            if _broadcast(inputs[0], inputs[1]) != inputs[0]:
                raise ValueError("RMSNormalization scale cannot broadcast to X")
            if math.prod(inputs[0][axis:]) != math.prod(inputs[1]):
                raise ValueError(
                    "Udon RMSNormalization scale must cover the normalized dimensions"
                )
            output = inputs[0]
        elif operation == "Gather":
            if len(inputs) != 2:
                raise ValueError("ONNX Gather requires data and indices inputs")
            if value_types[node["inputs"][0]] != "float32":
                raise ValueError("Udon Gather currently requires float32 data")
            if value_types[node["inputs"][1]] != "int32":
                raise ValueError("Udon Gather currently requires int32 indices")
            axis = attributes.get("axis", {"i": 0})["i"]
            axis = axis + len(inputs[0]) if axis < 0 else axis
            if axis < 0 or axis >= len(inputs[0]):
                raise ValueError("Gather axis is outside the data rank")
            output = inputs[0][:axis] + inputs[1] + inputs[0][axis + 1 :]
            if len(output) > 4:
                raise ValueError("Udon Gather output rank must not exceed four")
            if axis != 0 or len(inputs[0]) != 2:
                raise ValueError(
                    "UdonATen currently lowers ONNX Gather only as a rank-2 embedding lookup"
                )
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
        elif operation == "Identity":
            if len(inputs) != 1:
                raise ValueError("ONNX Identity requires exactly one input")
            output = inputs[0]
        elif operation == "Reshape":
            if len(inputs) != 2:
                raise ValueError("ONNX Reshape requires data and shape inputs")
            requested = _initializer_ints(graph, node["inputs"][1])
            output = _reshape_shape(
                inputs[0], requested, bool(attributes.get("allowzero", {"i": 0})["i"])
            )
        elif operation in ("Squeeze", "Unsqueeze"):
            if len(inputs) == 2:
                axes = _initializer_ints(graph, node["inputs"][1])
            else:
                axes = attributes.get("axes", {"ints": []})["ints"]
            if operation == "Squeeze":
                if not axes:
                    axes = [axis for axis, dimension in enumerate(inputs[0]) if dimension == 1]
                normalized = sorted(
                    {axis + len(inputs[0]) if axis < 0 else axis for axis in axes}
                )
                if len(normalized) != len(axes) or any(
                    axis < 0 or axis >= len(inputs[0]) or inputs[0][axis] != 1
                    for axis in normalized
                ):
                    raise ValueError("Squeeze axes must be unique dimensions of size one")
                output = [
                    dimension
                    for axis, dimension in enumerate(inputs[0])
                    if axis not in normalized
                ]
            else:
                output_rank = len(inputs[0]) + len(axes)
                normalized = sorted(
                    {axis + output_rank if axis < 0 else axis for axis in axes}
                )
                if len(normalized) != len(axes) or any(
                    axis < 0 or axis >= output_rank for axis in normalized
                ):
                    raise ValueError("Unsqueeze axes must be unique output dimensions")
                output = list(inputs[0])
                for axis in normalized:
                    output.insert(axis, 1)
        elif operation == "Tile":
            if len(inputs) != 2:
                raise ValueError("ONNX Tile requires data and repeats inputs")
            repeats = _initializer_ints(graph, node["inputs"][1])
            if len(repeats) != len(inputs[0]) or any(repeat <= 0 for repeat in repeats):
                raise ValueError("Tile repeats must be positive and match the data rank")
            output = [dimension * repeat for dimension, repeat in zip(inputs[0], repeats)]
            if len(output) > 4:
                raise ValueError("Udon Tile output rank must not exceed four")
            attributes = dict(attributes)
            attributes["repeats"] = {"ints": repeats}
        elif operation == "Gemm":
            if len(node["inputs"]) not in (2, 3):
                raise ValueError("ONNX Gemm requires two or three inputs")
            alpha = attributes.get("alpha", {"f": 1.0})["f"]
            beta = attributes.get("beta", {"f": 1.0})["f"]
            if not math.isfinite(alpha) or not math.isfinite(beta):
                raise ValueError("ONNX Gemm alpha and beta must be finite")
            left, right = inputs[:2]
            trans_a = attributes.get("transA", {"i": 0})["i"]
            trans_b = attributes.get("transB", {"i": 0})["i"]
            m, k = (left[1], left[0]) if trans_a else (left[0], left[1])
            right_k, n = (right[1], right[0]) if trans_b else (right[0], right[1])
            if k != right_k:
                raise ValueError("Gemm inner dimensions do not match")
            output = [m, n]
            if len(inputs) == 3 and _broadcast(output, inputs[2]) != output:
                raise ValueError("Gemm C input cannot broadcast to the matrix output")
        elif operation == "MatMul":
            if len(inputs) != 2:
                raise ValueError("ONNX MatMul requires exactly two inputs")
            output = _matmul_shape(inputs[0], inputs[1])
        elif node["domain"] == LLAMA_DOMAIN and operation == "WeightInt4packMM":
            if len(inputs) != 3 or len(inputs[0]) != 2 or inputs[0][0] != 1:
                raise ValueError(
                    "Udon WeightInt4packMM requires input, packed weight, and scales_and_zeros"
                )
            output_features = attributes.get("output_features", {}).get("i")
            input_features = attributes.get("input_features", {}).get("i")
            group_size = attributes.get("group_size", {}).get("i")
            if (
                not isinstance(output_features, int)
                or not isinstance(input_features, int)
                or group_size != 32
                or output_features <= 0
                or input_features <= 0
                or input_features % group_size
                or inputs[0][1] != input_features
            ):
                raise ValueError(
                    "Udon WeightInt4packMM attributes do not match its activation"
                )
            weight = graph["initializers"].get(node["inputs"][1])
            qparams = graph["initializers"].get(node["inputs"][2])
            expected_bytes = output_features * input_features // 2
            if (
                weight is None
                or weight["data_type"] != 2
                or weight["shape"] != [output_features, input_features // 2]
                or weight["data_location"] != 1
                or weight["external"]["length"] != expected_bytes
            ):
                raise ValueError(
                    "Udon WeightInt4packMM weight does not match packed INT4 storage"
                )
            expected_qparams_shape = [
                input_features // group_size,
                output_features,
                2,
            ]
            if (
                qparams is None
                or qparams["data_type"] != 16
                or qparams["shape"] != expected_qparams_shape
                or qparams["data_location"] != 1
                or qparams["external"]["length"]
                != math.prod(expected_qparams_shape) * 2
            ):
                raise ValueError(
                    "Udon WeightInt4packMM scales_and_zeros do not match ATen storage"
                )
            output = [1, output_features]
        else:
            raise AssertionError(operation)
        if len(node["outputs"]) != 1:
            raise ValueError(f"operator {operation} must have exactly one output")
        output_name = node["outputs"][0]
        if output_name in produced:
            raise ValueError(f"duplicate graph value {output_name!r}")
        shapes[output_name] = output
        value_types[output_name] = value_types[node["inputs"][0]]
        produced.add(output_name)
        nodes.append(
            {
                "name": node["name"],
                "domain": node["domain"],
                "op_type": operation,
                "provider": provider,
                "inputs": node["inputs"],
                "output": output_name,
                "shape": output,
                "attributes": attributes,
            }
        )
    for name, value_info in graph["outputs"].items():
        expected = value_info["shape"]
        if name not in shapes or shapes[name] != expected:
            raise ValueError(
                f"graph output {name!r} inferred as {shapes.get(name)}; expected {expected}"
            )
    if fuse_mnist:
        expected_ops = [
            "Conv",
            "Add",
            "Relu",
            "AveragePool",
            "Flatten",
            "Gemm",
            "Add",
        ]
        if [node["op_type"] for node in nodes] != expected_ops:
            raise ValueError(
                "initial Udon execution provider requires the MNIST operator sequence"
            )
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
    else:
        passes = []
        for node_index, node in enumerate(nodes):
            if node["provider"] == "view":
                continue
            if (
                node["provider"] == "udonatenRmsNorm"
                and math.prod(node["shape"]) > RMS_NORM_TILE_WORDS
                and math.prod(node["shape"]) <= ARENA_POINTWISE_WORDS
                and math.prod(shapes[node["inputs"][1]]) <= 4096
                and math.prod(node["shape"])
                % math.prod(shapes[node["inputs"][1]])
                == 0
            ):
                words = math.prod(node["shape"])
                normalized_words = math.prod(shapes[node["inputs"][1]])
                epsilon = node["attributes"].get("epsilon", {"f": 1.0e-5})["f"]
                partials = f"{node['output']}__rms_partials"
                passes.append(
                    {
                        "id": 200 + len(passes),
                        "kernel": "udonaten_arena_rms_norm_reduce",
                        "shader": "kernel_217.shader",
                        "providers": [node["provider"]],
                        "fusion": None,
                        "call_index": node_index,
                        "phase": "reduce",
                        "inputs": [node["inputs"][0]],
                        "output": partials,
                        "shape": node["shape"],
                        "fragments": ARENA_POINTWISE_WORDS,
                        "state_words": ARENA_POINTWISE_WORDS,
                        "material_ints": {
                            "_VRC_Arg2": words,
                            "_VRC_Arg3": normalized_words,
                        },
                    }
                )
                passes.append(
                    {
                        "id": 200 + len(passes),
                        "kernel": "udonaten_arena_rms_norm_finalize",
                        "shader": "kernel_218.shader",
                        "providers": [node["provider"]],
                        "fusion": None,
                        "call_index": node_index,
                        "phase": "finalize",
                        "inputs": [
                            node["inputs"][0],
                            partials,
                            node["inputs"][1],
                        ],
                        "static_inputs": [
                            {
                                "property": "_VRC_Input2",
                                "initializer": node["inputs"][1],
                            }
                        ],
                        "output": node["output"],
                        "shape": node["shape"],
                        "fragments": ARENA_POINTWISE_WORDS,
                        "state_words": ARENA_POINTWISE_WORDS,
                        "material_ints": {
                            "_VRC_Arg4": words,
                            "_VRC_Arg5": normalized_words,
                            "_VRC_Arg7": 1,
                        },
                        "material_floats": {"_VRC_Arg6": epsilon},
                    }
                )
                continue
            if node["op_type"] == "Gemm":
                tiles = _gemm_schedule(node, shapes)
            elif node["op_type"] == "MatMul":
                tiles = _matmul_schedule(node, shapes)
            elif node["provider"] in POINTWISE_PROVIDERS:
                tiles = _pointwise_schedule(node, shapes)
            elif (
                node["provider"] == "udonatenTile"
                and math.prod(node["shape"]) <= ARENA_POINTWISE_WORDS
            ):
                input_shape = [1] * (4 - len(shapes[node["inputs"][0]])) + shapes[
                    node["inputs"][0]
                ]
                repeats = [1] * (4 - len(node["attributes"]["repeats"]["ints"])) + node[
                    "attributes"
                ]["repeats"]["ints"]
                tiles = [
                    {
                        "arena_tile": True,
                        "storage_words": ARENA_POINTWISE_WORDS,
                        "width": 8192,
                        "height": 2,
                        "input_shape": input_shape,
                        "repeats": repeats,
                        "output_words": math.prod(node["shape"]),
                    }
                ]
            else:
                tiles = [None]
            for tile_index, tile in enumerate(tiles):
                suffix = "" if tile is None else f"_tile_{tile_index}"
                device_pass = {
                    "id": 200 + len(passes),
                    "kernel": f"onnx_{node_index}_{node['op_type'].lower()}{suffix}",
                    "providers": [node["provider"]],
                    "fusion": None,
                    "call_index": node_index,
                    "output": node["output"],
                    "shape": node["shape"],
                    "fragments": (
                        math.prod(node["shape"])
                        if tile is None
                        else (
                            tile["words"]
                            if "words" in tile
                            else (
                                tile["storage_words"]
                                if "storage_words" in tile
                                else tile["m"] * tile["n"]
                            )
                        )
                    ),
                }
                if tile is not None:
                    device_pass["tile"] = tile
                if tile is not None and "arena_words" in tile:
                    device_pass.update(
                        {
                            "kernel": "udonaten_arena_pointwise",
                            "shader": "kernel_215.shader",
                            "inputs": (
                                node["inputs"]
                                if len(node["inputs"]) == 2
                                else [node["inputs"][0], node["inputs"][0]]
                            ),
                            "fragments": tile["storage_words"],
                            "material_ints": {
                                "_VRC_Arg3": tile["arena_words"],
                                "_VRC_Arg4": tile["operation"],
                            },
                        }
                    )
                if tile is not None and tile.get("arena_tile"):
                    device_pass.update(
                        {
                            "kernel": "udonaten_arena_tile",
                            "shader": "kernel_216.shader",
                            "inputs": [node["inputs"][0]],
                            "fragments": tile["storage_words"],
                            "material_ints": {
                                **{
                                    f"_VRC_Arg{index + 2}": value
                                    for index, value in enumerate(
                                        tile["input_shape"] + tile["repeats"]
                                    )
                                },
                                "_VRC_Arg10": tile["output_words"],
                            },
                        }
                    )
                if node["provider"] == "udonatenRmsNorm" and math.prod(
                    node["shape"]
                ) > RMS_NORM_TILE_WORDS:
                    device_pass["supported"] = False
                    device_pass["reason"] = (
                        "large multi-row RMSNorm is not implemented"
                    )
                if (
                    node["domain"] == LLAMA_DOMAIN
                    and node["op_type"] == "WeightInt4packMM"
                ):
                    output_features = node["attributes"]["output_features"]["i"]
                    input_features = node["attributes"]["input_features"]["i"]
                    device_pass["tile_grid"] = {
                        "m": 64,
                        "k": 128,
                        "m_tiles": (output_features + 63) // 64,
                        "k_tiles": (input_features + 127) // 128,
                        "accumulation": "first K tile replaces; later K tiles add internally",
                    }
                passes.append(device_pass)
            if node["op_type"] == "Gemm" and len(node["inputs"]) == 3:
                passes.append(
                    {
                        "id": 200 + len(passes),
                        "kernel": f"onnx_{node_index}_gemm_bias",
                        "providers": ["udonatenAdd"],
                        "fusion": None,
                        "call_index": node_index,
                        "phase": "bias",
                        "beta": node["attributes"].get("beta", {"f": 1.0})["f"],
                        "output": node["output"],
                        "shape": node["shape"],
                        "fragments": math.prod(node["shape"]),
                    }
                )
    calls = []
    for node_index, node in enumerate(nodes):
        if (
            node["domain"] == LLAMA_DOMAIN
            and node["op_type"] == "WeightInt4packMM"
        ):
            output_features = node["attributes"]["output_features"]["i"]
            input_features = node["attributes"]["input_features"]["i"]
            calls.append(
                {
                    "index": len(calls),
                    "node_index": node_index,
                    "kind": "tiled_library",
                    "function": "udonatenWeightInt4packMm",
                    "inputs": node["inputs"],
                    "output": node["output"],
                    "output_shape": node["shape"],
                    "attributes": node["attributes"],
                    "weight_format": "q4_int4pack",
                    "group_size": node["attributes"]["group_size"]["i"],
                    "tile_grid": {
                        "m": 64,
                        "k": 128,
                        "m_tiles": (output_features + 63) // 64,
                        "k_tiles": (input_features + 127) // 128,
                        "accumulation": "first K tile replaces; later K tiles add internally",
                    },
                    "paging": {
                        "state_words": 8192,
                        "input_offset_words": 6144,
                        "result_offset_words": 7168,
                        "row_stride_words": 72,
                        "accumulate_argument": 5,
                        "packed_weight": {
                            "tensor": node["inputs"][1],
                            "external": dict(
                                graph["initializers"][node["inputs"][1]]["external"]
                            ),
                            "source_row_stride_bytes": input_features // 2,
                            "tile_row_bytes": 64,
                        },
                        "scales_and_zeros": {
                            "tensor": node["inputs"][2],
                            "external": dict(
                                graph["initializers"][node["inputs"][2]]["external"]
                            ),
                            "source_group_stride_bytes": output_features * 4,
                            "source_row_stride_bytes": 4,
                            "tile_groups": 4,
                            "destination_group_stride_words": 18,
                            "scale_word": 16,
                            "zero_word": 17,
                        },
                    },
                }
            )
            continue
        if node["op_type"] in ("Gemm", "MatMul"):
            tiles = (
                _gemm_schedule(node, shapes)
                if node["op_type"] == "Gemm"
                else _matmul_schedule(node, shapes)
            )
            for tile in tiles:
                calls.append(
                    {
                        "index": len(calls),
                        "node_index": node_index,
                        "kind": "library",
                        "function": "udonblasSgemm",
                        "inputs": node["inputs"][:2],
                        "output": node["output"],
                        "output_shape": node["shape"],
                        "attributes": node["attributes"],
                        "tile": tile,
                    }
                )
            if node["op_type"] == "Gemm" and len(node["inputs"]) == 3:
                calls.append(
                    {
                        "index": len(calls),
                        "node_index": node_index,
                        "kind": "library",
                        "function": "udonatenAdd",
                        "phase": "bias",
                        "inputs": [node["output"], node["inputs"][2]],
                        "output": node["output"],
                        "output_shape": node["shape"],
                        "attributes": {"beta": node["attributes"].get("beta", {"f": 1.0})},
                    }
                )
            continue
        call = {
            "index": len(calls),
            "node_index": node_index,
            "kind": "view" if node["provider"] == "view" else "library",
            "function": node["provider"],
            "inputs": node["inputs"],
            "output": node["output"],
            "output_shape": node["shape"],
            "attributes": node["attributes"],
        }
        if node["provider"] in POINTWISE_PROVIDERS:
            tiles = _pointwise_schedule(node, shapes)
            if tiles != [None] and "arena_words" in tiles[0]:
                call["device_profile"] = {
                    "supported": True,
                    "kernel": "udonaten_arena_pointwise",
                    "shader": "kernel_215.shader",
                    "storage_words": ARENA_POINTWISE_WORDS,
                    "width": 8192,
                    "height": 2,
                    "operation": tiles[0]["operation"],
                }
            elif tiles != [None]:
                call["device_tiles"] = {
                    "max_words": POINTWISE_TILE_WORDS,
                    "count": len(tiles),
                }
            elif math.prod(node["shape"]) > POINTWISE_TILE_WORDS:
                call["device_tiles"] = {
                    "supported": False,
                    "reason": "broadcast addressing is not implemented for large tensors",
                }
        elif node["provider"] == "udonatenRmsNorm":
            if math.prod(node["shape"]) > RMS_NORM_TILE_WORDS:
                normalized_words = math.prod(shapes[node["inputs"][1]])
                supported = (
                    math.prod(node["shape"]) <= ARENA_POINTWISE_WORDS
                    and normalized_words <= 4096
                    and math.prod(node["shape"]) % normalized_words == 0
                )
                call["device_profile"] = (
                    {
                        "supported": True,
                        "state_words": ARENA_POINTWISE_WORDS,
                        "passes": 2,
                        "reduction_chunk_words": 64,
                        "kernels": [
                            "udonaten_arena_rms_norm_reduce",
                            "udonaten_arena_rms_norm_finalize",
                        ],
                        "shaders": ["kernel_217.shader", "kernel_218.shader"],
                        "weight_input": "_VRC_Input2",
                    }
                    if supported
                    else {
                        "supported": False,
                        "reason": "RMSNorm exceeds the arena profile",
                    }
                )
        elif node["provider"] == "udonatenTile" and math.prod(
            node["shape"]
        ) <= ARENA_POINTWISE_WORDS:
            call["device_profile"] = {
                "supported": True,
                "kernel": "udonaten_arena_tile",
                "shader": "kernel_216.shader",
                "storage_words": ARENA_POINTWISE_WORDS,
                "width": 8192,
                "height": 2,
            }
        calls.append(call)
    scratch_buffers = [
        {
            "name": device_pass["output"],
            "words": device_pass["state_words"],
            "data_type": "float32",
            "lifetime": "until matching RMSNorm finalize pass",
        }
        for device_pass in passes
        if device_pass.get("phase") == "reduce"
        and device_pass["providers"] == ["udonatenRmsNorm"]
    ]
    texture_arena = _plan_texture_arena(
        graph, calls, shapes, value_types, scratch_buffers
    )
    return {
        "schema_version": 1,
        "runtime": "udon-onnx",
        "graph": graph["name"],
        "ir_version": model["ir_version"],
        "opset": model["opsets"][""],
        "nodes": nodes,
        "passes": passes,
        "aot": {
            "abi_version": 1,
            "ordering": "topological",
            "calls": calls,
        },
        "values": shapes,
        "value_types": value_types,
        "scratch_buffers": scratch_buffers,
        "texture_arena": texture_arena,
        "initializers": {
            name: (
                {
                    "shape": tensor["shape"],
                    "data_type": tensor["data_type_name"],
                    "bytes": tensor["logical_bytes"],
                    "storage": "external",
                    "external": dict(tensor["external"]),
                }
                if tensor["data_location"] == 1
                else {
                    "shape": tensor["shape"],
                    "data_type": tensor["data_type_name"],
                    "bytes": len(tensor["raw"]),
                    "storage": "embedded",
                    "sha256": hashlib.sha256(tensor["raw"]).hexdigest(),
                }
            )
            for name, tensor in graph["initializers"].items()
        },
        "inputs": {name: info["shape"] for name, info in graph["inputs"].items()},
        "outputs": {name: info["shape"] for name, info in graph["outputs"].items()},
    }


def _compile_mnist(model: dict) -> dict:
    return _compile_standard_graph(model, fuse_mnist=True)


def _compile_operators(model: dict) -> dict:
    return _compile_standard_graph(model, fuse_mnist=False)


def compile_operators(model: dict) -> dict:
    """Compile a graph from standard operators and explicit Udon library extensions."""
    nodes = model["graph"]["nodes"]
    unsupported = [
        (node["domain"], node["op_type"])
        for node in nodes
        if _operator_provider(node) is None
    ]
    if not nodes or unsupported:
        raise ValueError(f"unsupported operators for UdonOperatorExecutionProvider: {unsupported}")
    plan = _compile_operators(model)
    plan["execution_provider"] = "UdonOperatorExecutionProvider"
    return plan


def _c_identifiers(names) -> dict[str, str]:
    result = {}
    for index, name in enumerate(sorted(names)):
        stem = "".join(character if character.isalnum() else "_" for character in name)
        if not stem or stem[0].isdigit():
            stem = "value_" + stem
        result[name] = f"v{index}_{stem}"
    return result


def _tensor_4d(shape) -> list[int]:
    if not 0 <= len(shape) <= 4 or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"UdonDNN requires a static rank-0 through rank-4 tensor: {shape}")
    if len(shape) == 0:
        return [1, 1, 1, 1]
    if len(shape) == 1:
        return [1, shape[0], 1, 1]
    if len(shape) == 2:
        return [shape[0], shape[1], 1, 1]
    if len(shape) == 3:
        return [shape[0], shape[1], shape[2], 1]
    return list(shape)


def _attribute(node, name: str, field: str, default):
    return node["attributes"].get(name, {field: default}).get(field, default)


def _c_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("generated C scalar must be finite")
    return repr(float(value)) + "f"


def emit_c_host(plan: dict) -> str:
    """Emit an AOT C host that invokes public UdonBLAS/UdonDNN/UdonATen APIs."""
    if plan.get("runtime") != "udon-onnx" or "aot" not in plan:
        raise ValueError("execution plan has no standard Udon ONNX AOT schedule")
    if any(call["kind"] == "tiled_library" for call in plan["aot"]["calls"]):
        raise ValueError(
            "Q4 tiled-library calls require the external-weight paging Host emitter"
        )
    values = plan.get("values") or {}
    value_types = plan.get("value_types") or {name: "float32" for name in values}
    names = _c_identifiers(values)
    lines = [
        "/* Generated by udon-onnx. Compile this Host through UdonCL. */",
        '#include "udonblas.h"',
        '#include "udondnn.h"',
        '#include "udonaten.h"',
        "#include <stdint.h>",
        "",
        "typedef struct udon_onnx_buffers_t {",
    ]
    c_types = {
        "float32": "float",
        "int32": "int32_t",
        "int64": "int64_t",
        "uint8": "uint8_t",
        "float16": "uint16_t",
        "bfloat16": "uint16_t",
    }
    lines.extend(
        f"  {c_types[value_types[name]]} *{names[name]};" for name in sorted(values)
    )
    lines.extend(
        [
            "} udon_onnx_buffers_t;",
            "",
            "int udon_onnx_run(udon_onnx_buffers_t *buffers) {",
            "  int status = 0;",
            "  const float zero = 0.0f;",
            "  const float one = 1.0f;",
            "  udonblasHandle_t blas = 0;",
            "  udondnnHandle_t dnn = 0;",
        ]
    )
    tensor_descriptors = {}
    for name in sorted(values):
        if value_types[name] != "float32":
            continue
        tensor_descriptors[name] = "tensor_" + names[name]
        lines.append(f"  udondnnTensorDescriptor_t {tensor_descriptors[name]} = 0;")
    operation_descriptors = []
    filter_descriptors = []
    for index, node in enumerate(plan["nodes"]):
        operation = node["op_type"]
        if operation == "Conv":
            filter_descriptors.append((index, f"filter_{index}"))
            operation_descriptors.append(
                ("udondnnDestroyConvolutionDescriptor", f"convolution_{index}")
            )
            lines.append(f"  udondnnFilterDescriptor_t filter_{index} = 0;")
            lines.append(f"  udondnnConvolutionDescriptor_t convolution_{index} = 0;")
        elif operation == "AveragePool":
            operation_descriptors.append(
                ("udondnnDestroyPoolingDescriptor", f"pooling_{index}")
            )
            lines.append(f"  udondnnPoolingDescriptor_t pooling_{index} = 0;")
    lines.extend(
        [
            "#define CHECK_BLAS(call) do { status = (int)(call); if (status != UDONBLAS_STATUS_SUCCESS) goto cleanup; } while (0)",
            "#define CHECK_DNN(call) do { status = (int)(call); if (status != UDONDNN_STATUS_SUCCESS) goto cleanup; } while (0)",
            "  CHECK_BLAS(udonblasCreate(&blas));",
            "  CHECK_DNN(udondnnCreate(&dnn));",
        ]
    )
    for name in sorted(tensor_descriptors):
        n, c, h, w = _tensor_4d(values[name])
        descriptor = tensor_descriptors[name]
        lines.append(f"  CHECK_DNN(udondnnCreateTensorDescriptor(&{descriptor}));")
        lines.append(
            "  CHECK_DNN(udondnnSetTensor4dDescriptor("
            f"{descriptor}, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT, {n}, {c}, {h}, {w}));"
        )
    for index, node in enumerate(plan["nodes"]):
        operation = node["op_type"]
        inputs = node["inputs"]
        output = node["output"]
        if node["provider"] == "view":
            lines.append(f"  buffers->{names[output]} = buffers->{names[inputs[0]]};")
        elif operation == "Add":
            lines.append(
                "  CHECK_DNN(udonatenAdd("
                f"dnn, {tensor_descriptors[inputs[0]]}, buffers->{names[inputs[0]]}, "
                f"{tensor_descriptors[inputs[1]]}, buffers->{names[inputs[1]]}, 1.0f, "
                f"{tensor_descriptors[output]}, buffers->{names[output]}));"
            )
        elif operation == "Mul":
            lines.append(
                "  CHECK_DNN(udonatenMul("
                f"dnn, {tensor_descriptors[inputs[0]]}, buffers->{names[inputs[0]]}, "
                f"{tensor_descriptors[inputs[1]]}, buffers->{names[inputs[1]]}, "
                f"{tensor_descriptors[output]}, buffers->{names[output]}));"
            )
        elif operation == "Sub":
            lines.append(
                "  CHECK_DNN(udonatenSub("
                f"dnn, {tensor_descriptors[inputs[0]]}, buffers->{names[inputs[0]]}, "
                f"{tensor_descriptors[inputs[1]]}, buffers->{names[inputs[1]]}, 1.0f, "
                f"{tensor_descriptors[output]}, buffers->{names[output]}));"
            )
        elif operation == "Sigmoid":
            lines.append(
                "  CHECK_DNN(udonatenSigmoid("
                f"dnn, {tensor_descriptors[inputs[0]]}, buffers->{names[inputs[0]]}, "
                f"{tensor_descriptors[output]}, buffers->{names[output]}));"
            )
        elif operation in ("Relu", "Tanh"):
            function = "udonatenRelu" if operation == "Relu" else "udonatenTanh"
            lines.append(
                f"  CHECK_DNN({function}("
                f"dnn, {tensor_descriptors[inputs[0]]}, buffers->{names[inputs[0]]}, "
                f"{tensor_descriptors[output]}, buffers->{names[output]}));"
            )
        elif operation == "Softmax":
            mode = _softmax_mode(node, values[inputs[0]])
            lines.append(
                "  CHECK_DNN(udondnnSoftmaxForward("
                f"dnn, UDONDNN_SOFTMAX_ACCURATE, {mode}, &one, "
                f"{tensor_descriptors[inputs[0]]}, buffers->{names[inputs[0]]}, &zero, "
                f"{tensor_descriptors[output]}, buffers->{names[output]}));"
            )
        elif operation == "RMSNormalization":
            epsilon = _attribute(node, "epsilon", "f", 1.0e-5)
            normalized_shape = values[inputs[1]]
            normalized_values = ", ".join(str(value) for value in normalized_shape)
            lines.append(
                "  CHECK_DNN(udonatenRmsNorm("
                f"dnn, {tensor_descriptors[inputs[0]]}, buffers->{names[inputs[0]]}, "
                f"(const int64_t[]){{{normalized_values}}}, {len(normalized_shape)}, "
                f"{tensor_descriptors[inputs[1]]}, buffers->{names[inputs[1]]}, "
                f"(const float[]){{{_c_float(epsilon)}}}, {tensor_descriptors[output]}, "
                f"buffers->{names[output]}));"
            )
        elif operation == "Gather":
            indices_count = math.prod(values[inputs[1]])
            lines.append(
                "  CHECK_DNN(udonatenEmbedding("
                f"dnn, {tensor_descriptors[inputs[0]]}, buffers->{names[inputs[0]]}, "
                f"buffers->{names[inputs[1]]}, {indices_count}, -1, 0, 0, "
                f"{tensor_descriptors[output]}, buffers->{names[output]}));"
            )
        elif operation == "Tile":
            repeats = _attribute(node, "repeats", "ints", None)
            if repeats is None:
                raise ValueError("Tile AOT emission requires constant repeats")
            repeat_values = ", ".join(str(value) for value in repeats)
            lines.append(
                "  CHECK_DNN(udonatenTile("
                f"dnn, {tensor_descriptors[inputs[0]]}, buffers->{names[inputs[0]]}, "
                f"(const int64_t[]){{{repeat_values}}}, {len(repeats)}, "
                f"{tensor_descriptors[output]}, buffers->{names[output]}));"
            )
        elif operation == "AveragePool":
            kernel = _attribute(node, "kernel_shape", "ints", None)
            if not kernel or len(kernel) != 2:
                raise ValueError("AveragePool requires a two-dimensional kernel_shape")
            pads = _attribute(node, "pads", "ints", [0, 0, 0, 0])
            strides = _attribute(node, "strides", "ints", [1, 1])
            if pads[0] != pads[2] or pads[1] != pads[3]:
                raise ValueError("UdonDNN pooling requires symmetric ONNX padding")
            include = _attribute(node, "count_include_pad", "i", 0)
            mode = (
                "UDONDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING"
                if include
                else "UDONDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING"
            )
            lines.append(f"  CHECK_DNN(udondnnCreatePoolingDescriptor(&pooling_{index}));")
            lines.append(
                "  CHECK_DNN(udondnnSetPooling2dDescriptor("
                f"pooling_{index}, {mode}, UDONDNN_PROPAGATE_NAN, {kernel[0]}, {kernel[1]}, "
                f"{pads[0]}, {pads[1]}, {strides[0]}, {strides[1]}));"
            )
            lines.append(
                "  CHECK_DNN(udondnnPoolingForward("
                f"dnn, pooling_{index}, &one, {tensor_descriptors[inputs[0]]}, buffers->{names[inputs[0]]}, "
                f"&zero, {tensor_descriptors[output]}, buffers->{names[output]}));"
            )
        elif operation == "Conv":
            image_shape = values[inputs[0]]
            filter_shape = values[inputs[1]]
            if len(image_shape) != 4 or len(filter_shape) != 4:
                raise ValueError("Conv AOT emission requires rank-4 tensors")
            pads = _attribute(node, "pads", "ints", [0, 0, 0, 0])
            strides = _attribute(node, "strides", "ints", [1, 1])
            dilations = _attribute(node, "dilations", "ints", [1, 1])
            if pads[0] != pads[2] or pads[1] != pads[3]:
                raise ValueError("UdonDNN convolution requires symmetric ONNX padding")
            k, c, h, w = filter_shape
            lines.append(f"  CHECK_DNN(udondnnCreateFilterDescriptor(&filter_{index}));")
            lines.append(
                "  CHECK_DNN(udondnnSetFilter4dDescriptor("
                f"filter_{index}, UDONDNN_DATA_FLOAT, UDONDNN_TENSOR_NCHW, {k}, {c}, {h}, {w}));"
            )
            lines.append(f"  CHECK_DNN(udondnnCreateConvolutionDescriptor(&convolution_{index}));")
            lines.append(
                "  CHECK_DNN(udondnnSetConvolution2dDescriptor("
                f"convolution_{index}, {pads[0]}, {pads[1]}, {strides[0]}, {strides[1]}, "
                f"{dilations[0]}, {dilations[1]}, UDONDNN_CROSS_CORRELATION, UDONDNN_DATA_FLOAT));"
            )
            lines.append(
                "  CHECK_DNN(udondnnConvolutionForward("
                f"dnn, &one, {tensor_descriptors[inputs[0]]}, buffers->{names[inputs[0]]}, "
                f"filter_{index}, buffers->{names[inputs[1]]}, convolution_{index}, "
                "UDONDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 0, 0, &zero, "
                f"{tensor_descriptors[output]}, buffers->{names[output]}));"
            )
        elif operation == "Gemm":
            left_shape, right_shape = values[inputs[0]], values[inputs[1]]
            trans_a = _attribute(node, "transA", "i", 0)
            trans_b = _attribute(node, "transB", "i", 0)
            n = node["shape"][1]
            alpha = _attribute(node, "alpha", "f", 1.0)
            lines.append(f"  const float gemm_alpha_{index} = {_c_float(alpha)};")
            for tile in _gemm_schedule(node, values):
                m0, n0, k0 = (
                    tile["m_offset"],
                    tile["n_offset"],
                    tile["k_offset"],
                )
                left_offset = k0 * left_shape[1] + m0 if trans_a else m0 * left_shape[1] + k0
                right_offset = n0 * right_shape[1] + k0 if trans_b else k0 * right_shape[1] + n0
                beta = "zero" if tile["beta"] == 0 else "one"
                lines.append(
                    "  CHECK_BLAS(udonblasSgemm("
                    f"blas, {'UDONBLAS_OP_T' if trans_b else 'UDONBLAS_OP_N'}, "
                    f"{'UDONBLAS_OP_T' if trans_a else 'UDONBLAS_OP_N'}, {tile['n']}, {tile['m']}, {tile['k']}, &gemm_alpha_{index}, "
                    f"buffers->{names[inputs[1]]} + {right_offset}, {right_shape[1]}, "
                    f"buffers->{names[inputs[0]]} + {left_offset}, {left_shape[1]}, &{beta}, "
                    f"buffers->{names[output]} + {m0 * n + n0}, {n}));"
                )
            if len(inputs) == 3:
                bias = inputs[2]
                beta = _attribute(node, "beta", "f", 1.0)
                lines.append(f"  const float gemm_beta_{index} = {_c_float(beta)};")
                lines.append(
                    "  CHECK_DNN(udonatenAdd("
                    f"dnn, {tensor_descriptors[output]}, buffers->{names[output]}, "
                    f"{tensor_descriptors[bias]}, buffers->{names[bias]}, gemm_beta_{index}, "
                    f"{tensor_descriptors[output]}, buffers->{names[output]}));"
                )
        elif operation == "MatMul":
            left_shape, right_shape = values[inputs[0]], values[inputs[1]]
            k, n = left_shape[-1], right_shape[-1]
            for tile in _matmul_schedule(node, values):
                beta = "zero" if tile["beta"] == 0 else "one"
                lines.append(
                    "  CHECK_BLAS(udonblasSgemm("
                    f"blas, UDONBLAS_OP_N, UDONBLAS_OP_N, {tile['n']}, {tile['m']}, {tile['k']}, &one, "
                    f"buffers->{names[inputs[1]]} + {tile['right_offset']}, {n}, "
                    f"buffers->{names[inputs[0]]} + {tile['left_offset']}, {k}, &{beta}, "
                    f"buffers->{names[output]} + {tile['output_offset']}, {n}));"
                )
        else:
            raise ValueError(f"cannot emit AOT host call for {operation!r}")
    lines.append("cleanup:")
    for destroy, descriptor in reversed(operation_descriptors):
        lines.append(f"  if ({descriptor}) (void){destroy}({descriptor});")
    for _, descriptor in reversed(filter_descriptors):
        lines.append(f"  if ({descriptor}) (void)udondnnDestroyFilterDescriptor({descriptor});")
    for name in reversed(sorted(tensor_descriptors)):
        descriptor = tensor_descriptors[name]
        lines.append(f"  if ({descriptor}) (void)udondnnDestroyTensorDescriptor({descriptor});")
    lines.extend(
        [
            "  if (dnn) (void)udondnnDestroy(dnn);",
            "  if (blas) (void)udonblasDestroy(blas);",
            "  return status;",
            "#undef CHECK_DNN",
            "#undef CHECK_BLAS",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def emit_uasm_host(plan: dict) -> str:
    """Emit the Udon Host that dispatches an execution plan's Device passes."""
    passes = plan.get("passes") or []
    if plan.get("runtime") != "udon-onnx" or not passes:
        raise ValueError("execution plan has no Udon ONNX Device passes")
    material_names = [f"pass{item['id']}Material" for item in passes]
    lines = [
        ".data_start",
        "    .export initialState",
        "    .export stateA",
        "    .export stateB",
        "    .export resultState",
    ]
    lines.extend(f"    .export {name}" for name in material_names)
    lines.extend(
        [
            "",
            "    initialState: %UnityEngineTexture, null",
            "    stateA: %UnityEngineRenderTexture, null",
            "    stateB: %UnityEngineRenderTexture, null",
            "    resultState: %UnityEngineTexture, null",
        ]
    )
    lines.extend(f"    {name}: %UnityEngineMaterial, null" for name in material_names)
    lines.extend(
        [
            "    __source: %UnityEngineTexture, null",
            '    __inputName: %SystemString, "_VRC_Input"',
            ".data_end",
            "",
            ".code_start",
            "    .export _start",
            "    _start:",
            "        JUMP, 0xFFFFFFFC",
            "",
            "    .export _run",
            "    _run:",
            "        PUSH, initialState",
            "        PUSH, __source",
            "        COPY",
        ]
    )
    for index, material in enumerate(material_names):
        target = "stateA" if index % 2 == 0 else "stateB"
        lines.extend(
            [
                f"        PUSH, {material}",
                "        PUSH, __inputName",
                "        PUSH, __source",
                '        EXTERN, "UnityEngineMaterial.__SetTexture__SystemString_UnityEngineTexture__SystemVoid"',
                "        PUSH, __source",
                f"        PUSH, {target}",
                f"        PUSH, {material}",
                '        EXTERN, "VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid"',
                f"        PUSH, {target}",
                "        PUSH, __source",
                "        COPY",
            ]
        )
    lines.extend(
        [
            "        PUSH, __source",
            "        PUSH, resultState",
            "        COPY",
            "        JUMP, 0xFFFFFFFC",
            ".code_end",
            "",
        ]
    )
    return "\n".join(lines)


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


def _supports_operators(model: dict) -> bool:
    nodes = model["graph"]["nodes"]
    return bool(nodes) and not _supports_mnist(model) and all(
        _operator_provider(node) is not None for node in nodes
    )


def default_registry() -> ExecutionProviderRegistry:
    registry = ExecutionProviderRegistry()
    registry.register("UdonLlamaExecutionProvider", _supports_llama, _compile_llama)
    registry.register("UdonMnistExecutionProvider", _supports_mnist, _compile_mnist)
    registry.register(
        "UdonOperatorExecutionProvider", _supports_operators, _compile_operators
    )
    return registry


def compile_model(model: dict, registry=None) -> dict:
    return (registry or default_registry()).compile(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--emit-host-c",
        type=Path,
        help="write the AOT UdonBLAS/UdonDNN Host C call sequence",
    )
    parser.add_argument(
        "--emit-host-uasm",
        type=Path,
        help="write the executable Udon Host Device-pass dispatcher",
    )
    parser.add_argument(
        "--execution-provider",
        choices=("auto", "operators"),
        default="auto",
        help="select the generic standard-operator path instead of graph fusion",
    )
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
    plan = compile_operators(model) if args.execution_provider == "operators" else compile_model(model)
    if external is not None:
        plan["external_data_validation"] = external
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.emit_host_c:
        args.emit_host_c.parent.mkdir(parents=True, exist_ok=True)
        args.emit_host_c.write_text(emit_c_host(plan), encoding="utf-8")
    if args.emit_host_uasm:
        args.emit_host_uasm.parent.mkdir(parents=True, exist_ok=True)
        args.emit_host_uasm.write_text(emit_uasm_host(plan), encoding="utf-8")
    print(
        f"udon_onnx_nodes={len(plan['nodes'])} device_passes={len(plan['passes'])} "
        f"outputs={plan['outputs']}"
    )


if __name__ == "__main__":
    main()
