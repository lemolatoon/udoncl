"""Lower both public UdonDNN large pointwise backends to Device bundles."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

compiler, clang, output = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
device_ir = output / "large-pointwise.ll"


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
        root / "libraries/udondnn/device/large_pointwise.cl",
        "-o",
        device_ir,
    ]
)
for kernel_id, name, argument_count, scalar_bytes in [
    (233, "udondnn_large_activation_forward", 9, 24),
    (234, "udondnn_large_op_tensor", 15, 44),
]:
    words = 131072 if kernel_id == 234 else 16384
    global_size = [8192, 16, 1] if kernel_id == 234 else [8192, 2, 1]
    config = output / f"{name}.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "gather-f32",
                "words": words,
                "kernel": name,
                "kernel_id": kernel_id,
                "global_size": global_size,
                "local_size": [1, 1, 1],
                "global_offset": [0, 0, 0],
            }
        ),
        encoding="utf-8",
    )
    bundle = output / f"{name}-bundle"
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
    assert kernel["name"] == name
    assert kernel["global_size"] == global_size
    assert len(kernel["args"]) == argument_count
    assert kernel["descriptor"]["scalar_payload_bytes"] == scalar_bytes
print("large_pointwise_bundles=2 max_dimensions=8192x16")
