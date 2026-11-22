"""Dependency-free ONNX protobuf builders for checked-in model generators.

The examples and test fixtures generate their ONNX models with these helpers
instead of depending on the Python onnx package. Each function returns the
serialized protobuf message; compose them into a ModelProto and write the
bytes of model_proto() to a .onnx file.
"""

from __future__ import annotations

import math
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


def tensor_uint8_proto(name: str, shape, values) -> bytes:
    raw = bytes(values)
    if len(raw) != math.prod(shape):
        raise ValueError(f"uint8 tensor {name!r} value count does not match shape")
    return (
        packed_ints_field(1, shape)
        + varint_field(2, 2)
        + bytes_field(8, name)
        + bytes_field(9, raw)
    )


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
        message_field(
            1,
            bytes_field(2, dimension)
            if isinstance(dimension, str)
            else varint_field(1, dimension),
        )
        for dimension in shape
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
