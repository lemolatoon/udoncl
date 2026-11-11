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

# Prove the fixed-shape conversation profile independently of the compiler.
# A capacity-sized present tensor is rolled by one row into the next fixed
# past tensor. Unused left-padding rows are hidden by the ONNX attention mask.
capacity = 8
rolling_past = [
    np.zeros((1, 4, capacity - 1, 64), dtype=np.float32) for _ in range(44)
]
reference_past = [empty for _ in range(44)]
step_argmax: list[int] = []
step_logit_error: list[float] = []
step_kv_error: list[float] = []


def run_fixed_m1(token: int, position: int) -> list[np.ndarray]:
    valid = position + 1
    assert 0 < valid <= capacity
    mask = np.zeros((1, capacity), dtype=np.int64)
    mask[:, capacity - valid :] = 1
    step_feed = {
        "input_ids": np.array([[token]], dtype=np.int64),
        "attention_mask": mask,
        "position_ids": np.array([[position]], dtype=np.int64),
    }
    for item, value in zip(session.get_inputs()[3:], rolling_past):
        step_feed[item.name] = value
    return session.run(None, step_feed)


def run_variable_m1(token: int, position: int) -> list[np.ndarray]:
    step_feed = {
        "input_ids": np.array([[token]], dtype=np.int64),
        "attention_mask": np.ones((1, position + 1), dtype=np.int64),
        "position_ids": np.array([[position]], dtype=np.int64),
    }
    for item, value in zip(session.get_inputs()[3:], reference_past):
        step_feed[item.name] = value
    return session.run(None, step_feed)


for position, token in enumerate(tokens[0].tolist()):
    step_results = run_fixed_m1(token, position)
    reference_results = run_variable_m1(token, position)
    step_logits = step_results[0]
    assert step_logits.shape == (1, 1, 32000)
    assert np.isfinite(step_logits).all()
    step_argmax.append(int(np.argmax(step_logits[0, 0])))
    step_logit_error.append(
        float(np.max(np.abs(step_logits[0, 0] - reference_results[0][0, 0])))
    )
    kv_error = 0.0
    for reference_value, step_value in zip(
        reference_results[1:], step_results[1:]
    ):
        assert step_value.shape == (1, 4, capacity, 64)
        valid_step = step_value[:, :, capacity - position - 1 :, :]
        kv_error = max(
            kv_error, float(np.max(np.abs(valid_step - reference_value)))
        )
    step_kv_error.append(kv_error)
    rolling_past = [value[:, :, 1:, :].copy() for value in step_results[1:]]
    reference_past = reference_results[1:]

assert step_argmax == argmax[0], step_argmax
assert max(step_logit_error) < 2.0e-5, step_logit_error
assert max(step_kv_error) < 2.0e-5, step_kv_error

# Fix the first decode continuation as an oracle for the forthcoming C Host
# loop. Tokenization and detokenization deliberately remain outside this test.
generated_tokens = [step_argmax[-1]]
position = int(tokens.shape[1])
while len(generated_tokens) < 4 and generated_tokens[-1] != 2:
    step_results = run_fixed_m1(generated_tokens[-1], position)
    generated_tokens.append(int(np.argmax(step_results[0][0, 0])))
    rolling_past = [value[:, :, 1:, :].copy() for value in step_results[1:]]
    position += 1
assert generated_tokens == [1701, 29916, 432, 17204], generated_tokens

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
(output / "m1-oracle.json").write_text(
    json.dumps(
        {
            "capacity": capacity,
            "prompt_tokens": tokens[0].tolist(),
            "prompt_argmax": step_argmax,
            "generated_tokens": generated_tokens,
            "max_logit_error_vs_batch": max(step_logit_error),
            "max_kv_error_vs_batch": max(step_kv_error),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(
    f"tinyllama_oracle_logits={logits.size} kv_outputs=44 "
    f"argmax={argmax[0]} m1_generated={generated_tokens} "
    f"seconds={elapsed:.6f}"
)
