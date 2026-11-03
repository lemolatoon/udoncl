"""Lower the public UdonDNN large-convolution backend to a Device bundle."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

compiler, clang, output = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
device_ir = output / "large-convolution.ll"


def run(command):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, (command, result.stdout, result.stderr)


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
        root / "libraries/udondnn/device/large_convolution.cl",
        "-o",
        device_ir,
    ]
)
config = output / "large-convolution.json"
config.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "profile": "gather-f32",
            "words": 16384,
            "kernel": "udondnn_large_convolution_forward",
            "kernel_id": 232,
            "global_size": [8192, 2, 1],
            "local_size": [1, 1, 1],
            "global_offset": [0, 0, 0],
        }
    ),
    encoding="utf-8",
)
bundle = output / "bundle"
if bundle.exists():
    shutil.rmtree(bundle)
run(
    [
        compiler,
        "--emit-native-device-bundle",
        config,
        "--device-llvm",
        device_ir,
        "-o",
        bundle,
    ]
)
manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
kernel = manifest["device"]["kernels"][0]
assert kernel["name"] == "udondnn_large_convolution_forward"
assert kernel["global_size"] == [8192, 2, 1]
assert len(kernel["args"]) == 16
assert [argument["access"] for argument in kernel["args"][:4]] == [
    "read",
    "read",
    "read",
    "write",
]
assert kernel["descriptor"]["scalar_payload_bytes"] == 48
print("large_convolution_bundle=8192x2 arguments=16 scalar_bytes=48")
