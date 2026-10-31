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

math_source = output / "math.cl"
math_source.write_text(
    "__kernel void math_probe(__global const float *input, __global float *output) {\n"
    "  uint gid = get_global_id(0);\n"
    "  float value = input[gid];\n"
    "  output[gid] = cos(value) + sin(value) + exp(value);\n"
    "}\n",
    encoding="utf-8",
)
math_ir = output / "math.ll"
run(
    [
        clang,
        "-target",
        "spir-unknown-unknown",
        "-x",
        "cl",
        "-cl-std=CL1.2",
        "-cl-kernel-arg-info",
        "-O0",
        "-Xclang",
        "-disable-O0-optnone",
        "-fomit-frame-pointer",
        "-ffp-contract=off",
        "-S",
        "-emit-llvm",
        math_source,
        "-o",
        math_ir,
    ]
)
math_config = dict(
    config,
    words=16,
    kernel="math_probe",
    kernel_id=23,
    global_size=[16, 1, 1],
)
config_path.write_text(json.dumps(math_config), encoding="utf-8")
math_bundle = output / "math-bundle"
if math_bundle.exists():
    shutil.rmtree(math_bundle)
run(
    [
        compiler,
        "--emit-native-device-bundle",
        config_path,
        "--device-llvm",
        math_ir,
        "-o",
        math_bundle,
    ]
)
math_hlsl = (math_bundle / "kernels/kernel_23.hlsl").read_text()
assert all(f"{name}(" in math_hlsl for name in ["cos", "sin", "exp"])

too_large = dict(config, words=16385, global_size=[16385, 1, 1])
config_path.write_text(json.dumps(too_large), encoding="utf-8")
run(
    [compiler, "--emit-native-device-bundle", config_path, "--device-llvm", ir, "-o", output / "too-large"],
    error="unsupported config version or launch profile",
)
print("large_gather_words=10800 upper_bound_rejection=passed")
