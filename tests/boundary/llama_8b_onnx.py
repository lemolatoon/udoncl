"""Validate the Llama 3.1 8B ONNX external-weight execution plan."""

import json
from pathlib import Path
import subprocess
import sys


root, output = (Path(value).resolve() for value in sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
model = output / "llama-8b.onnx"
plan_path = output / "onnx-plan.json"
operator_model = output / "llama-8b-operators.onnx"
operator_plan_path = output / "operator-plan.json"
result = subprocess.run(
    [
        sys.executable,
        root / "examples/llama-8b-onnx/generate_onnx.py",
        "-o",
        model,
        "--plan",
        plan_path,
        "--operator-output",
        operator_model,
        "--operator-plan",
        operator_plan_path,
    ],
    text=True,
    capture_output=True,
    timeout=60,
)
assert result.returncode == 0, (result.stdout, result.stderr)
plan = json.loads(plan_path.read_text(encoding="utf-8"))
assert plan["runtime"] == "udon-onnx"
assert plan["profile"] == "llama-3.1-8b-q4"
assert plan["execution_provider"] == "UdonLlamaExecutionProvider"
assert plan["config"]["num_hidden_layers"] == 32
assert plan["config"]["hidden_size"] == 4096
assert plan["config"]["vocab_size"] == 128256
assert plan["weights"]["parameters"] == 8030261248
assert plan["weights"]["storage_bytes"] == 5019279360
assert plan["weights"]["tensors"] == 517
assert len(plan["passes"]) == 162
assert [item["id"] for item in plan["passes"]] == list(range(1000, 1162))
assert plan["passes"][0]["kernel"] == "llama_embedding_gather"
assert plan["passes"][-1]["kernel"] == "llama_logits"
assert {item["layer"] for item in plan["passes"][1:-1]} == set(range(32))
providers = {
    provider for device_pass in plan["passes"] for provider in device_pass["providers"]
}
assert "udonatenWeightInt4packMm" in providers
assert "udonblasSgemm" in providers
assert "udondnnRmsNormalizationForward" in providers
assert "udondnnSoftmaxForward" in providers
assert "udondnnOpTensor" in providers
assert plan["blas"] == {
    "weight_provider": "udonatenWeightInt4packMm",
    "activation_provider": "udonblasSgemm",
    "q4_tile": {
        "m": 64,
        "k": 128,
        "block_values": 32,
        "qparams": "[K/group_size,N,2] bfloat16",
        "accumulator": "float32",
    },
}
matmuls = [
    matmul
    for device_pass in plan["passes"]
    for matmul in device_pass.get("matmuls", [])
]
assert len(matmuls) == 225
assert all(matmul["provider"] == "udonatenWeightInt4packMm" for matmul in matmuls)
assert all(matmul["weight_format"] == "q4_int4pack" for matmul in matmuls)
assert all(matmul["group_size"] == 32 for matmul in matmuls)
assert all(matmul["scales_and_zeros"] in plan["initializers"] for matmul in matmuls)
assert {matmul["weight"] for matmul in matmuls} <= set(plan["initializers"])
q_projection = plan["passes"][1]["matmuls"][0]
assert q_projection["weight"] == "model.layers.0.self_attn.q_proj.weight"
assert (q_projection["m_tiles"], q_projection["k_tiles"]) == (64, 32)
logits = plan["passes"][-1]["matmuls"][0]
assert logits["weight"] == "lm_head.weight"
assert (logits["m_tiles"], logits["k_tiles"]) == (2004, 32)

operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
assert operator_plan["execution_provider"] == "UdonOperatorExecutionProvider"
assert operator_plan["graph"] == "llama_3_1_8b_operator_decoder_single_token"
assert operator_plan["inputs"] == {"hidden_state": [1, 4096]}
assert operator_plan["outputs"] == {"logits": [1, 128256]}
assert len(operator_plan["nodes"]) == 546
assert len(operator_plan["passes"]) == 482
assert all(node["op_type"] != "LlamaDecoder" for node in operator_plan["nodes"])
operator_q4_calls = [
    call
    for call in operator_plan["aot"]["calls"]
    if call["function"] == "udonatenWeightInt4packMm"
]
assert len(operator_q4_calls) == 225
assert all(call["kind"] == "tiled_library" for call in operator_q4_calls)
assert all(call["weight_format"] == "q4_int4pack" for call in operator_q4_calls)
assert all(call["group_size"] == 32 for call in operator_q4_calls)
assert operator_q4_calls[0]["tile_grid"] == {
    "m": 64,
    "k": 128,
    "m_tiles": 64,
    "k_tiles": 32,
    "accumulation": "first K tile replaces; later K tiles add internally",
}
assert operator_q4_calls[-1]["tile_grid"]["m_tiles"] == 2004
assert operator_q4_calls[-1]["tile_grid"]["k_tiles"] == 32
assert operator_q4_calls[0]["paging"]["state_words"] == 8192
assert operator_q4_calls[0]["paging"]["input_offset_words"] == 6144
assert operator_q4_calls[0]["paging"]["result_offset_words"] == 7168
assert operator_q4_calls[0]["paging"]["row_stride_words"] == 72
assert operator_q4_calls[0]["paging"]["accumulate_argument"] == 5
assert operator_q4_calls[0]["paging"]["packed_weight"]["source_row_stride_bytes"] == 2048
assert (
    operator_q4_calls[0]["paging"]["scales_and_zeros"][
        "source_group_stride_bytes"
    ]
    == 16384
)
assert (
    operator_q4_calls[-1]["paging"]["scales_and_zeros"][
        "source_group_stride_bytes"
    ]
    == 513024
)
assert sum(node["op_type"] == "Tile" for node in operator_plan["nodes"]) == 32

sys.path.insert(0, str(root / "runtime/onnx"))
from udon_onnx import (  # noqa: E402
    emit_c_host,
    external_tensor_proto,
    parse_model,
    validate_external_data,
)

try:
    emit_c_host(operator_plan)
except ValueError as error:
    assert "external-weight paging" in str(error)
else:
    raise AssertionError("8B Q4 operator plan was emitted without a paging Host")

try:
    external_tensor_proto("bad", [1], 2, "../escape.bin", 0, 1)
except ValueError:
    pass
else:
    raise AssertionError("unsafe ONNX external_data location was accepted")

parsed = parse_model(model.read_bytes())
locations = {
    tensor["external"]["location"]
    for tensor in parsed["graph"]["initializers"].values()
}
assert locations == {"weights.q4"}
try:
    validate_external_data(parsed, output)
except ValueError as error:
    assert "does not exist" in str(error)
else:
    raise AssertionError("missing 8B external weights were accepted as executable")
print(result.stdout, end="")
print("llama_8b_onnx_plan=passed providers=udondnn,udonaten-q4 external_data=safe")
