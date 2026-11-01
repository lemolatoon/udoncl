"""Validate generic safetensors to ONNX external-weight packing."""

import json
from pathlib import Path
import struct
import sys

import numpy as np


root, output = (Path(value).resolve() for value in sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
checkpoint = output / "checkpoint"
checkpoint.mkdir(exist_ok=True)
sys.path.insert(0, str(root / "runtime/onnx"))
from udon_weights import pack_checkpoint  # noqa: E402


def bfloat16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2")


matrix = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
norm = np.asarray([0.5, -0.25, 1.0, -1.5, 2.0, 0.0, 3.0, -4.0], dtype=np.float32)
copied = np.asarray([1.25, -2.5, 0.125, 8.0], dtype="<f2")
payloads = {
    "matrix": ("BF16", [2, 32], bfloat16(matrix).tobytes()),
    "norm": ("F32", [8], norm.astype("<f4").tobytes()),
    "copied": ("F16", [4], copied.tobytes()),
}
header = {}
data = bytearray()
for name, (dtype, shape, payload) in payloads.items():
    start = len(data)
    data.extend(payload)
    header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(data)]}
encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
(checkpoint / "model.safetensors").write_bytes(
    struct.pack("<Q", len(encoded)) + encoded + data
)

index = {
    "schema_version": 1,
    "storage_bytes_with_alignment": 144,
    "tensors": [
        {
            "name": "matrix",
            "logical_shape": [2, 32],
            "format": "q4_int4pack",
            "offset": 0,
            "length": 32,
        },
        {
            "name": "matrix.scales_and_zeros",
            "source_name": "matrix",
            "logical_shape": [2, 32],
            "format": "q4_scale_zero_bf16",
            "offset": 32,
            "length": 8,
        },
        {
            "name": "norm",
            "logical_shape": [8],
            "format": "bf16",
            "offset": 64,
            "length": 16,
        },
        {
            "name": "copied",
            "logical_shape": [4],
            "format": "float32",
            "offset": 128,
            "length": 16,
        },
    ],
}
index_path = output / "weights-index.json"
index_path.write_text(json.dumps(index), encoding="utf-8")
packed_path = output / "weights.q4"
manifest = pack_checkpoint(checkpoint, index_path, packed_path)
assert manifest["bytes"] == 144
assert len(manifest["tensors"]) == 4
packed = packed_path.read_bytes()
decoded = []
qparams_bits = np.frombuffer(packed[32:40], dtype="<u2").astype(np.uint32)
qparams = (qparams_bits << np.uint32(16)).view(np.float32).reshape(1, 2, 2)
for row in range(2):
    block_data = packed[row * 16 : row * 16 + 16]
    scale, zero = qparams[0, row]
    assert zero == 0.0
    values = []
    for byte in block_data:
        for nibble in (byte & 15, byte >> 4):
            values.append(nibble - 16 if nibble >= 8 else nibble)
    decoded.extend(value * scale + zero for value in values)
    source = matrix[row * 32 : row * 32 + 32]
    error = np.max(np.abs(source - np.asarray(decoded[-32:])))
    assert error <= scale * 0.51 + 1.0e-6
norm_bits = np.frombuffer(packed[64:80], dtype="<u2").astype(np.uint32) << np.uint32(16)
np.testing.assert_allclose(norm_bits.view(np.float32), norm, rtol=0.004, atol=0.004)
np.testing.assert_array_equal(
    np.frombuffer(packed[128:144], dtype="<f4"), copied.astype(np.float32)
)

bad_index = dict(index)
bad_index["tensors"] = [dict(item) for item in index["tensors"]]
bad_index["tensors"][0]["logical_shape"] = [64]
bad_path = output / "bad-index.json"
bad_path.write_text(json.dumps(bad_index), encoding="utf-8")
sentinel = output / "sentinel.q4"
sentinel.write_bytes(b"preserve")
try:
    pack_checkpoint(checkpoint, bad_path, sentinel)
except ValueError as error:
    assert "shape mismatch" in str(error)
else:
    raise AssertionError("shape-mismatched checkpoint was packed")
assert sentinel.read_bytes() == b"preserve"
print("udon_onnx_weight_packer=passed q4=64 bf16=8 float32=4 transactional=passed")
