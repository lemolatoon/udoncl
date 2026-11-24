"""Run the pinned TinyLlama Q4 model with ONNX Runtime's CPU provider.

usage: tinyllama_real_onnx.py ROOT OUTPUT [--capacity N] [--skip-step-sidecars]

The fixed-shape M=1 profile and its oracles use the Context Capacity N of the
conversation artifacts they check (default 512). The step sidecars hold every
input as JSON, including 44 past tensors of capacity - 1 rows;
--skip-step-sidecars leaves them out for a capacity whose step artifact is not
built.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("root", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--capacity", type=int, default=512)
parser.add_argument("--skip-step-sidecars", action="store_true")
arguments = parser.parse_args()
root = arguments.root.resolve()
output = arguments.output.resolve()
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
# A capacity-sized present tensor is materialized by slicing away its first
# sequence row into the next fixed past tensor. Unused left-padding rows are
# hidden by the ONNX attention mask. The capacity is the Context Capacity of
# the graph of the conversation and M=1 step artifacts (512 by default in
# examples/onnx-transformer-tiny, 4096 for its -4k artifacts).
capacity = arguments.capacity
assert capacity >= 16, capacity
rolling_past = [
    np.zeros((1, 4, capacity - 1, 64), dtype=np.float32) for _ in range(44)
]
reference_past = [empty for _ in range(44)]
step_argmax: list[int] = []
step_logit_error: list[float] = []
step_kv_error: list[float] = []
# The M=1 logits of every prompt position, for the chunked profile below.
step_results_by_position: dict[int, np.ndarray] = {}


# Step-level M=1 sidecars: every input of the fixed-capacity graph,
# including the rolling past state, and every output. A Unity run of the
# public-buffer M=1 step artifact compares all logits and present K/V of these
# steps independently of the C Host loop.
step_sidecar_positions = () if arguments.skip_step_sidecars else (3, 4)
step_sidecars: dict[int, tuple[dict[str, np.ndarray], list[np.ndarray]]] = {}


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
    step_results = session.run(None, step_feed)
    if position in step_sidecar_positions:
        step_sidecars[position] = (dict(step_feed), step_results)
    return step_results


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
    step_results_by_position[position] = step_logits[0, 0]
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


# The profile of a Conversation Artifact of `rows` positions per run
# (UDONCL_TINYLLAMA_SEQUENCE_ROWS): the same past of capacity - 1 rows, a
# window of capacity + rows - 1, the prompt in chunks of `rows` and every
# decode step a run of one real row. A run of fewer real rows repeats its
# last token at its position in the rest, and only the real rows are carried
# (__udononnx_transitional_advance_sequence_state_rows). Greedy tokens must
# equal the M=1 ones.
def chunked_generation(rows: int) -> tuple[list[int], float]:
    window = capacity + rows - 1
    past = [np.zeros((1, 4, capacity - 1, 64), dtype=np.float32) for _ in range(44)]

    def run(chunk: list[int], position: int) -> np.ndarray:
        nonlocal past
        real = len(chunk)
        mask = np.zeros((1, window), dtype=np.int64)
        mask[:, window - position - rows :] = 1
        positions = [position + index for index in range(real)]
        chunk_feed = {
            "input_ids": np.array([chunk + [chunk[-1]] * (rows - real)], dtype=np.int64),
            "attention_mask": mask,
            "position_ids": np.array(
                [positions + [positions[-1]] * (rows - real)], dtype=np.int64
            ),
        }
        for item, value in zip(session.get_inputs()[3:], past):
            chunk_feed[item.name] = value
        chunk_results = session.run(None, chunk_feed)
        assert chunk_results[0].shape == (1, rows, 32000)
        past = [
            value[:, :, real : real + capacity - 1, :].copy()
            for value in chunk_results[1:]
        ]
        return chunk_results[0][0, real - 1]

    prompt = tokens[0].tolist()
    for start in range(0, len(prompt), rows):
        logits = run(prompt[start : start + rows], start)
    error = float(np.max(np.abs(logits - step_results_by_position[len(prompt) - 1])))
    chunked = [int(np.argmax(logits))]
    position = len(prompt)
    while len(chunked) < 4 and chunked[-1] != 2:
        logits = run([chunked[-1]], position)
        chunked.append(int(np.argmax(logits)))
        position += 1
    return chunked, error


chunked_errors = {}
for rows in (3, 4):
    chunked_tokens, chunked_errors[rows] = chunked_generation(rows)
    assert chunked_tokens == generated_tokens, (rows, chunked_tokens)
    assert chunked_errors[rows] < 1.0e-3, (rows, chunked_errors[rows])

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
assert sorted(step_sidecars) == list(step_sidecar_positions), sorted(step_sidecars)
for position, (step_feed, step_results) in step_sidecars.items():
    step_directory = output / "m1-steps" / f"step-{position:03d}"
    write_float32_oracle(
        step_directory,
        model_sha256=fetch.MODEL_SHA256,
        inputs=step_feed,
        outputs=[
            (item.name, value)
            for item, value in zip(session.get_outputs(), step_results)
        ],
        metadata={
            "capacity": capacity,
            "position": position,
            "argmax": int(np.argmax(step_results[0][0, 0])),
        },
    )
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
    f"chunked_logit_error={max(chunked_errors.values()):.2e} "
    f"seconds={elapsed:.6f}"
)
