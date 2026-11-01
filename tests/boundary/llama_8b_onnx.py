"""Validate the Llama 3.1 8B ONNX external-weight execution plan."""

import json
from pathlib import Path
import subprocess
import sys


root, output = (Path(value).resolve() for value in sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
model = output / "llama-8b.onnx"
plan_path = output / "onnx-plan.json"
result = subprocess.run(
    [
        sys.executable,
        root / "examples/llama-8b-onnx/generate_onnx.py",
        "-o",
        model,
        "--plan",
        plan_path,
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
assert plan["weights"]["storage_bytes"] == 4517404672
assert plan["weights"]["tensors"] == 291
assert len(plan["passes"]) == 162
assert [item["id"] for item in plan["passes"]] == list(range(1000, 1162))
assert plan["passes"][0]["kernel"] == "llama_embedding_gather"
assert plan["passes"][-1]["kernel"] == "llama_logits"
assert {item["layer"] for item in plan["passes"][1:-1]} == set(range(32))
providers = {
    provider for device_pass in plan["passes"] for provider in device_pass["providers"]
}
assert "udonblasSgemvQ4" in providers
assert "udonblasSgemm" in providers
assert "udondnnRmsNormalizationForward" in providers
assert "udondnnSoftmaxForward" in providers
assert "udondnnOpTensor" in providers
assert plan["blas"] == {
    "weight_provider": "udonblasSgemvQ4",
    "activation_provider": "udonblasSgemm",
    "q4_tile": {
        "m": 64,
        "k": 128,
        "block_values": 32,
        "expanded_block_words": 17,
        "accumulator": "float32",
    },
}
matmuls = [
    matmul
    for device_pass in plan["passes"]
    for matmul in device_pass.get("matmuls", [])
]
assert len(matmuls) == 225
assert all(matmul["provider"] == "udonblasSgemvQ4" for matmul in matmuls)
assert all(matmul["weight_format"] == "q4_block32" for matmul in matmuls)
assert {matmul["weight"] for matmul in matmuls} <= set(plan["initializers"])
q_projection = plan["passes"][1]["matmuls"][0]
assert q_projection["weight"] == "model.layers.0.self_attn.q_proj.weight"
assert (q_projection["m_tiles"], q_projection["k_tiles"]) == (64, 32)
logits = plan["passes"][-1]["matmuls"][0]
assert logits["weight"] == "lm_head.weight"
assert (logits["m_tiles"], logits["k_tiles"]) == (2004, 32)

sys.path.insert(0, str(root / "runtime/onnx"))
from udon_onnx import external_tensor_proto, parse_model, validate_external_data  # noqa: E402

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
print("llama_8b_onnx_plan=passed providers=udondnn,udonblas-q4 external_data=safe")
