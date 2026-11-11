"""Run the pinned TinyLlama Q4 model with ONNX Runtime's CPU provider."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort


root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "scripts"))

from write_onnx_oracle import write_float32_oracle  # noqa: E402


spec = importlib.util.spec_from_file_location(
    "fetch_tinyllama_onnx", root / "scripts" / "fetch_tinyllama_onnx.py"
)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
model = fetch.ensure(fetch.default_cache())

options = ort.SessionOptions()
options.intra_op_num_threads = 4
session = ort.InferenceSession(
    str(model), sess_options=options, providers=["CPUExecutionProvider"]
)
tokens = np.array([[1, 450, 4996, 17354]], dtype=np.int64)
feed = {
    "input_ids": tokens,
    "attention_mask": np.ones((1, 4), dtype=np.int64),
    "position_ids": np.arange(4, dtype=np.int64).reshape(1, 4),
}
empty = np.zeros((1, 4, 0, 64), dtype=np.float32)
for layer in range(22):
    feed[f"past_key_values.{layer}.key"] = empty
    feed[f"past_key_values.{layer}.value"] = empty

started = time.perf_counter()
results = session.run(None, feed)
elapsed = time.perf_counter() - started
assert len(results) == 45
logits = results[0]
assert logits.shape == (1, 4, 32000)
assert np.isfinite(logits).all()
argmax = np.argmax(logits, axis=-1).tolist()
assert argmax == [[29958, 390, 17354, 1701]], argmax
for index, value in enumerate(results[1:]):
    assert value.shape == (1, 4, 4, 64), (index, value.shape)
    assert np.isfinite(value).all(), index

named_results = [
    (item.name, value) for item, value in zip(session.get_outputs(), results)
]
np.savez(output / "oracle.npz", **dict(named_results))
write_float32_oracle(
    output,
    model_sha256=fetch.MODEL_SHA256,
    inputs=feed,
    outputs=named_results,
    metadata={
        "logits": int(logits.size),
        "argmax": argmax,
        "kv_outputs": 44,
        "kv_values": sum(int(value.size) for value in results[1:]),
        "seconds": elapsed,
    },
)
manifest = json.loads((output / "oracle.json").read_text(encoding="utf-8"))
assert len(manifest["outputs"]) == 45
assert manifest["outputs"][0]["element_count"] == 128000
assert all(item["element_count"] == 1024 for item in manifest["outputs"][1:])
print(
    f"tinyllama_oracle_logits={logits.size} kv_outputs=44 "
    f"argmax={argmax[0]} seconds={elapsed:.6f}"
)
