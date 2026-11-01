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


def tensor_proto(name: str, shape, values) -> bytes:
    raw = struct.pack(f"<{math.prod(shape)}f", *values)
    return (
        packed_ints_field(1, shape)
        + varint_field(2, 1)
        + bytes_field(8, name)
        + bytes_field(9, raw)
    )


def attribute_ints(name: str, values) -> bytes:
    return bytes_field(1, name) + packed_ints_field(8, values) + varint_field(20, 7)


def attribute_int(name: str, value: int) -> bytes:
    return bytes_field(1, name) + varint_field(3, value) + varint_field(20, 2)


def node_proto(op_type: str, inputs, outputs, name: str, attributes=()) -> bytes:
    payload = b"".join(bytes_field(1, value) for value in inputs)
    payload += b"".join(bytes_field(2, value) for value in outputs)
    payload += bytes_field(3, name) + bytes_field(4, op_type)
    payload += b"".join(message_field(5, attribute) for attribute in attributes)
    return payload


def value_info_proto(name: str, shape) -> bytes:
    dimensions = b"".join(
        message_field(1, varint_field(1, dimension)) for dimension in shape
    )
    tensor_type = varint_field(1, 1) + message_field(2, dimensions)
    return bytes_field(1, name) + message_field(2, message_field(1, tensor_type))


def model_proto(graph: bytes, opset: int = 13) -> bytes:
    opset_import = bytes_field(1, "") + varint_field(2, opset)
    return (
        varint_field(1, 8)
        + bytes_field(2, "udoncl")
        + message_field(7, graph)
        + message_field(8, opset_import)
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
        elif number == 8:
            result["ints"].extend(_packed_varints(value) if wire == 2 else [value])
    return result


def _parse_node(data: bytes) -> dict:
    result = {"inputs": [], "outputs": [], "name": "", "op_type": "", "attributes": {}}
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
        elif number == 7 and _string(value):
            raise ValueError("only the standard ONNX operator domain is supported")
    return result


def _parse_tensor(data: bytes) -> dict:
    result = {"name": "", "shape": [], "data_type": 0, "raw": b""}
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
    if not result["raw"] and float_data:
        result["raw"] = struct.pack(f"<{len(float_data)}f", *float_data)
    if result["data_type"] != 1:
        raise ValueError(f"tensor {result['name']!r} is not FLOAT")
    expected = math.prod(result["shape"]) * 4
    if len(result["raw"]) != expected:
        raise ValueError(
            f"tensor {result['name']!r} has {len(result['raw'])} bytes; expected {expected}"
        )
    return result


def _parse_value_info(data: bytes) -> tuple[str, list[int]]:
    name, shape = "", []
    for number, wire, value in fields(data):
        if number == 1 and wire == 2:
            name = _string(value)
        elif number == 2 and wire == 2:
            for type_number, type_wire, type_value in fields(value):
                if type_number != 1 or type_wire != 2:
                    continue
                elem_type = None
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
                if elem_type != 1:
                    raise ValueError(f"value {name!r} is not FLOAT")
    return name, shape


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
            name, shape = _parse_value_info(value)
            (graph["inputs"] if number == 11 else graph["outputs"])[name] = shape
    model["graph"] = graph
    return model


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


def compile_model(model: dict) -> dict:
    graph = model["graph"]
    shapes = {name: tensor["shape"] for name, tensor in graph["initializers"].items()}
    shapes.update(graph["inputs"])
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
    for name, expected in graph["outputs"].items():
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
        "inputs": graph["inputs"],
        "outputs": graph["outputs"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    plan = compile_model(parse_model(args.model.read_bytes()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        f"udon_onnx_nodes={len(plan['nodes'])} device_passes={len(plan['passes'])} "
        f"output_shape={plan['outputs']['logits']}"
    )


if __name__ == "__main__":
    main()
