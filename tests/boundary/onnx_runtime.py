"""Validate the dependency-free ONNX importer and MNIST Udon provider plan."""

import json
from pathlib import Path
import subprocess
import sys


root, output = (Path(value).resolve() for value in sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "runtime/onnx"))

from udon_onnx import bytes_field, compile_model, parse_model


model_path = output / "mnist.onnx"
result = subprocess.run(
    [
        sys.executable,
        root / "examples/mnist-cnn/generate_onnx.py",
        "-o",
        model_path,
    ],
    text=True,
    capture_output=True,
    timeout=60,
)
assert result.returncode == 0, (result.stdout, result.stderr)
data = model_path.read_bytes()
model = parse_model(data + bytes_field(99, b"unknown-field-is-skipped"))
plan = compile_model(model)
assert plan["runtime"] == "udon-onnx"
assert plan["inputs"] == {"image": [1, 1, 28, 28]}
assert plan["outputs"] == {"logits": [1, 10]}
assert [node["op_type"] for node in plan["nodes"]] == [
    "Conv",
    "Add",
    "Relu",
    "AveragePool",
    "Flatten",
    "Gemm",
    "Add",
]
assert [node["provider"] for node in plan["nodes"]] == [
    "udondnnConvolutionForward",
    "udondnnOpTensor",
    "udondnnActivationForward",
    "udondnnPoolingForward",
    "view",
    "udonblasSgemm",
    "udondnnOpTensor",
]
assert [item["fragments"] for item in plan["passes"]] == [9216, 144, 10]
assert plan["nodes"][0]["shape"] == [1, 16, 24, 24]
assert plan["nodes"][3]["shape"] == [1, 16, 3, 3]
assert plan["nodes"][4]["shape"] == [1, 144]
assert set(plan["initializers"]) == {
    "conv.weight",
    "conv.bias",
    "dense.weight",
    "dense.bias",
}
try:
    parse_model(data[:-1])
except ValueError:
    pass
else:
    raise AssertionError("truncated ONNX protobuf was accepted")
(output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
print("udon_onnx_import=passed nodes=7 providers=udondnn,udonblas passes=3")
