"""Test the model-independent raw ONNX oracle sidecar contract."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np


root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
shutil.rmtree(output, ignore_errors=True)
output.mkdir(parents=True)
sys.path.insert(0, str(root / "scripts"))

from write_onnx_oracle import write_float32_oracle  # noqa: E402


values = [
    np.array([[1.25, -2.5]], dtype=np.float32),
    np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2),
]
write_float32_oracle(
    output,
    model_sha256="abc123",
    inputs={"input_ids": np.array([[1, 2]], dtype=np.int64)},
    outputs=[("logits", values[0]), ("present.0.key", values[1])],
)

manifest = json.loads((output / "oracle.json").read_text(encoding="utf-8"))
assert manifest["schema"] == "udoncl.onnx-oracle.v1"
assert manifest["model_sha256"] == "abc123"
assert manifest["inputs"] == [
    {"dtype": "int64", "name": "input_ids", "shape": [1, 2], "values": [1, 2]}
]
assert manifest["outputs"] == [
    {
        "dtype": "float32",
        "element_count": 2,
        "file": "output-000.f32.bytes",
        "index": 0,
        "name": "logits",
        "shape": [1, 2],
    },
    {
        "dtype": "float32",
        "element_count": 8,
        "file": "output-001.f32.bytes",
        "index": 1,
        "name": "present.0.key",
        "shape": [1, 2, 2, 2],
    },
]
for index, expected in enumerate(values):
    actual = np.fromfile(output / f"output-{index:03d}.f32.bytes", dtype="<f4")
    np.testing.assert_array_equal(actual, expected.reshape(-1))

print("onnx_oracle_sidecar=passed outputs=2 values=10")
