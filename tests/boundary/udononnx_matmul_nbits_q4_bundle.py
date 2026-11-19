"""Lower the one-dispatch Q4 MatMulNBits kernel to a Unity Device bundle.

The kernel accumulates a whole K-block segment per dispatch with constant,
guarded loops. The bundle must be accepted for the library capacities, emit
exactly the expected Loads (no loop unrolling, no extra reads), and be
rejected when a declared input capacity cannot hold the proven index range.
"""

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


compiler, clang, output = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
device_ir = output / "matmul-nbits-q4.ll"


def run(command, expect_success=True):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        timeout=300,
    )
    if expect_success:
        assert result.returncode == 0, (command, result.stdout, result.stderr)
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
        "-O0",
        "-Xclang",
        "-disable-O0-optnone",
        "-fomit-frame-pointer",
        "-ffp-contract=off",
        "-S",
        "-emit-llvm",
        root / "libraries/udononnxoperators/device/matmul_nbits.cl",
        "-o",
        device_ir,
    ]
)


def config(m, n, k, input_words):
    blocks = (k + 31) // 32
    return {
        "schema_version": 1,
        "profile": "gather-f32",
        "words": 131072,
        "kernel": "udononnx_matmul_nbits_q4",
        "kernel_id": 235,
        "input_words": input_words,
        "argument_bounds": [
            {"index": 5, "min": 1, "max": m},
            {"index": 6, "min": 1, "max": n},
            {"index": 7, "min": 1, "max": k},
            {"index": 8, "min": 0, "max": blocks - 1},
            {"index": 9, "min": 1, "max": blocks},
            {"index": 10, "min": 0, "max": 1},
        ],
        "argument_kinds": ["buffer"] * 5 + ["i32"] * 6,
        "global_size": [8192, 16, 1],
        "local_size": [1, 1, 1],
        "global_offset": [0, 0, 0],
    }


def lower(name, kernel_config, expect_success=True):
    config_path = output / f"{name}.json"
    config_path.write_text(json.dumps(kernel_config), encoding="utf-8")
    bundle = output / name
    if bundle.exists():
        shutil.rmtree(bundle)
    result = run(
        [
            compiler,
            "--emit-native-device-bundle",
            config_path,
            "--device-llvm",
            device_ir,
            "-o",
            bundle,
        ],
        expect_success,
    )
    return bundle, result


def capacities(m, n, k):
    blocks = (k + 31) // 32
    return [m * k, n * blocks * 4, n * blocks, 131072]


accepted = []
for name, (m, n, k) in {
    "k5632": (4, 32000, 5632),
    "k2048": (4, 33, 2048),
    "k145": (2, 37, 145),
}.items():
    bundle, _ = lower(name, config(m, n, k, capacities(m, n, k)))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    kernel = manifest["device"]["kernels"][0]
    assert kernel["name"] == "udononnx_matmul_nbits_q4"
    assert kernel["global_size"] == [8192, 16, 1]
    assert len(kernel["args"]) == 11
    assert [argument["access"] for argument in kernel["args"][:5]] == [
        "read",
        "read",
        "read",
        "read",
        "write",
    ]
    hlsl = (bundle / "kernels/kernel_235.hlsl").read_text(encoding="utf-8")
    # Chunk, block-in-chunk and K-tail loops stay rolled; the full-block body
    # is expanded by the source macros: 1 scale + 4 packed words + 32
    # activations, plus 1 scale + 1 packed word + 1 activation in the K tail
    # and the optional previous partial sum.
    loops = hlsl.count("[loop]")
    loads = len(re.findall(r"\.Load\(", hlsl))
    assert loops == 3, loops
    assert loads == 41, loads
    accepted.append(f"{name}:loops={loops},loads={loads}")

# The scale texture is one word short of n * ceil(k / 32): the index proof
# must reject the kernel instead of wrapping or clamping the access.
short = capacities(1, 64, 2048)
short[2] -= 1
_, rejected = lower("short-scales", config(1, 64, 2048, short), False)
assert rejected.returncode != 0, rejected.stdout
assert (
    "buffer index bounds are not proven within the declared extent for "
    "argument 2" in rejected.stderr
), rejected.stderr
print("matmul_nbits_q4_bundle=" + " ".join(accepted) + " short_scales=rejected")
