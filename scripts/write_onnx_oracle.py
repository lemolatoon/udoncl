"""Write ONNX Runtime results in a Unity-readable raw sidecar format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


_RESERVED_KEYS = {"schema", "model_sha256", "inputs", "outputs"}


def write_float32_oracle(
    directory: Path,
    *,
    model_sha256: str,
    inputs: Mapping[str, np.ndarray],
    outputs: Sequence[tuple[str, np.ndarray]],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write named FP32 outputs and a JSON manifest without requiring an NPZ reader."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("output-*.f32.bytes"):
        stale.unlink()

    input_entries = []
    for name, value in inputs.items():
        array = np.asarray(value)
        input_entries.append(
            {
                "dtype": array.dtype.name,
                "name": name,
                "shape": list(array.shape),
                "values": array.reshape(-1).tolist(),
            }
        )

    output_entries = []
    for index, (name, value) in enumerate(outputs):
        array = np.asarray(value)
        if array.dtype != np.float32:
            raise TypeError(f"ONNX oracle output {name!r} must be float32, got {array.dtype}")
        filename = f"output-{index:03d}.f32.bytes"
        contiguous = np.ascontiguousarray(array, dtype="<f4")
        (directory / filename).write_bytes(contiguous.tobytes(order="C"))
        output_entries.append(
            {
                "dtype": "float32",
                "element_count": int(array.size),
                "file": filename,
                "index": index,
                "name": name,
                "shape": list(array.shape),
            }
        )

    manifest: dict[str, object] = {
        "schema": "udoncl.onnx-oracle.v1",
        "model_sha256": model_sha256,
        "inputs": input_entries,
        "outputs": output_entries,
    }
    if metadata:
        overlap = _RESERVED_KEYS.intersection(metadata)
        if overlap:
            raise ValueError(f"ONNX oracle metadata uses reserved keys: {sorted(overlap)}")
        manifest.update(metadata)

    manifest_path = directory / "oracle.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path
