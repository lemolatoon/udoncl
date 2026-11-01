"""Run the trained MNIST ONNX graph through the Udon Device provider."""

import json
from pathlib import Path
import subprocess
import sys


build, output = (Path(value).resolve() for value in sys.argv[1:])
root = Path(__file__).resolve().parents[2]
result = subprocess.run(
    [
        sys.executable,
        root / "examples/mnist-cnn/run_device.py",
        "--build-dir",
        build,
        "--output",
        output,
    ],
    text=True,
    capture_output=True,
    timeout=600,
)
assert result.returncode == 0, (result.stdout, result.stderr)
assert "mnist_device_predictions=10/10" in result.stdout
assert "max_cpu_gpu_error=" in result.stdout
plan = json.loads((output / "onnx-plan.json").read_text(encoding="utf-8"))
assert plan["runtime"] == "udon-onnx"
assert [node["provider"] for node in plan["nodes"]] == [
    "udondnnConvolutionForward",
    "udonatenAdd",
    "udondnnActivationForward",
    "udondnnPoolingForward",
    "view",
    "udonblasSgemm",
    "udonatenAdd",
]
assert [(item["id"], item["fragments"]) for item in plan["passes"]] == [
    (20, 9216),
    (21, 144),
    (22, 10),
]
print(result.stdout, end="")
