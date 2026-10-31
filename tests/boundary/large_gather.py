"""Prove that a non-power-of-two, CUDA-style large gather lowers safely."""

import json
from pathlib import Path
import shutil
import subprocess
import sys


compiler, clang, output = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
ir = output / "device.ll"


def run(command, error=None):
    result = subprocess.run(
        [str(value) for value in command], text=True, capture_output=True, timeout=120
    )
    if error is None:
        assert result.returncode == 0, (command, result.stdout, result.stderr)
    else:
        assert result.returncode != 0 and error in result.stderr, (
            command,
            result.stdout,
            result.stderr,
        )
    return result


run(
    [
        clang,
        "-target",
        "spir-unknown-unknown",
        "-x",
        "cl",
        "-cl-std=CL1.2",
        "-cl-kernel-arg-info",
        "-DMNIST_CONV=1",
        "-O0",
        "-Xclang",
        "-disable-O0-optnone",
        "-fomit-frame-pointer",
        "-ffp-contract=off",
        "-S",
        "-emit-llvm",
        root / "examples/mnist-cnn/device.cl",
        "-o",
        ir,
    ]
)
config = {
    "schema_version": 1,
    "profile": "gather-f32",
    "words": 10800,
    "kernel": "mnist_conv",
    "kernel_id": 20,
    "global_size": [10800, 1, 1],
    "local_size": [1, 1, 1],
    "global_offset": [0, 0, 0],
}
config_path = output / "large.json"
config_path.write_text(json.dumps(config), encoding="utf-8")
bundle = output / "bundle"
if bundle.exists():
    shutil.rmtree(bundle)
run([compiler, "--emit-native-device-bundle", config_path, "--device-llvm", ir, "-o", bundle])
manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
kernel = manifest["device"]["kernels"][0]
assert kernel["global_size"] == [10800, 1, 1]
assert [argument["required_bytes"] for argument in kernel["args"][:2]] == [
    43200,
    43200,
]
hlsl = (bundle / "kernels/kernel_20.hlsl").read_text()
assert "uint(position.x)" in hlsl and "%" in hlsl and "/" in hlsl

too_large = dict(config, words=16385, global_size=[16385, 1, 1])
config_path.write_text(json.dumps(too_large), encoding="utf-8")
run(
    [compiler, "--emit-native-device-bundle", config_path, "--device-llvm", ir, "-o", output / "too-large"],
    error="unsupported config version or launch profile",
)
print("large_gather_words=10800 upper_bound_rejection=passed")
