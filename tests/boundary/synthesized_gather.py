"""Check a synthesized MLIR gather kernel against the library dispatches it fuses.

The UdonDNN large OpTensor kernel goes through the OpenCL C/LLVM path and the
hand-written synthesized kernel through --device-mlir. Both fragments are then
compiled and run by the native fragment renderer, which must accept the
generated HLSL and return bit-identical output words.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

compiler, clang, runner, output = sys.argv[1:]
output = Path(output).resolve()
output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
environment = dict(os.environ)
if sys.platform != "win32":
    icds = list(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json"))
    assert icds, "install mesa-vulkan-drivers and libvulkan-dev"
    environment["VK_DRIVER_FILES"] = environment["VK_ICD_FILENAMES"] = str(icds[0])


def run(command, expected=0):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        env=environment,
        timeout=300,
    )
    assert result.returncode == expected, (command, result.stdout, result.stderr)
    return result


GEOMETRY = {
    "schema_version": 1,
    "profile": "gather-f32",
    "words": 131072,
    "global_size": [8192, 16, 1],
    "local_size": [1, 1, 1],
    "global_offset": [0, 0, 0],
}


def emit(name, kernel_id, source_option, source, extra=None, expected=0):
    config = output / f"{name}.json"
    config.write_text(
        json.dumps(
            {**GEOMETRY, "kernel": name, "kernel_id": kernel_id, **(extra or {})}
        ),
        encoding="utf-8",
    )
    bundle = output / f"{name}-bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    result = run(
        [
            compiler,
            "--emit-native-device-bundle",
            config,
            source_option,
            source,
            "-o",
            bundle,
        ],
        expected,
    )
    return bundle, result


device_ir = output / "large-pointwise.ll"
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
library, _ = emit("udondnn_large_op_tensor", 234, "--device-llvm", device_ir)

synthesized_source = root / "tests/boundary/synthesized_rotary.mlir"
bounds = [{"index": 4, "min": 0, "max": 256}]
synthesized, _ = emit(
    "synthesized_rotary",
    300,
    "--device-mlir",
    synthesized_source,
    {"input_words": [256, 64, 64], "argument_bounds": bounds},
)
manifest = json.loads((synthesized / "manifest.json").read_text(encoding="utf-8"))
assert manifest["artifact_stage"] == "native-device-prototype"
kernel = manifest["device"]["kernels"][0]
assert kernel["name"] == "synthesized_rotary"
assert kernel["id"] == 300
assert kernel["global_size"] == [8192, 16, 1]
assert len(kernel["args"]) == 5
assert kernel["descriptor"]["scalar_payload_bytes"] == 4
recovered = (synthesized / "frontend.recovered.mlir").read_text(encoding="utf-8")
assert "vrc.gather_proven" in recovered and "vrc.synthesized_kernel" in recovered
library_hlsl = library / "kernels/kernel_234.hlsl"
synthesized_hlsl = synthesized / "kernels/kernel_300.hlsl"
hlsl = synthesized_hlsl.read_text(encoding="utf-8")
assert "precise float" in hlsl
assert not re.search(r"\b(mad|fma)\s*\(", hlsl)

# The config may not disagree with the contract the kernel declares, and an
# unproven kernel must not reach the lowering even if its text claims a proof.
for extra, message in [
    ({"words": 65536, "global_size": [8192, 8, 1]}, "global_size does not match"),
    ({"input_words": [256, 64, 32]}, "input_words does not match"),
    ({"argument_bounds": [{"index": 4, "min": 0, "max": 512}]}, "argument_bounds"),
    ({"kernel": "missing"}, "no gpu.func @missing"),
]:
    _, result = emit(
        "synthesized_rotary", 300, "--device-mlir", synthesized_source, extra, 1
    )
    assert message in result.stderr, (extra, result.stderr)
unproven = output / "unproven.mlir"
unproven.write_text(
    synthesized_source.read_text(encoding="utf-8")
    .replace("max = 256 : i32", "max = 257 : i32")
    .replace("vrc.profile", "vrc.gather_proven, vrc.profile"),
    encoding="utf-8",
)
assert "vrc.gather_proven" in unproven.read_text(encoding="utf-8")
_, result = emit(
    "synthesized_rotary",
    300,
    "--device-mlir",
    unproven,
    {"argument_bounds": [{"index": 4, "min": 0, "max": 257}]},
    1,
)
assert "inferred [0, 256] but capacity is 256" in result.stderr, result.stderr
synthesized, _ = emit("synthesized_rotary", 300, "--device-mlir", synthesized_source)

vertex = output / "fullscreen.hlsl"
vertex.write_text(
    "float4 vert(uint id : SV_VertexID) : SV_Position { return float4(id == 2 ? 3.0 : -1.0, id == 1 ? 3.0 : -1.0, 0, 1); }\n",
    encoding="utf-8",
)
fragments = [library_hlsl, synthesized_hlsl]
if sys.platform != "win32":
    glslang = shutil.which("glslangValidator")
    assert glslang, "install glslang-tools"
    compiled = []
    for index, (asset, stage, entry) in enumerate(
        [
            (fragments[0], "frag", "frag"),
            (fragments[1], "frag", "frag"),
            (vertex, "vert", "vert"),
        ]
    ):
        binary = output / f"shader_{index}.spv"
        run(
            [
                glslang,
                "-D",
                "-V",
                "-S",
                stage,
                "-e",
                entry,
                "--auto-map-locations",
                asset,
                "-o",
                binary,
            ]
        )
        compiled.append(binary)
    *fragments, vertex = compiled

result = run([runner, *fragments, vertex])
assert "synthesized_bitexact=256/256" in result.stdout, result.stdout
assert "tail_zero=1" in result.stdout, result.stdout
assert "draws=4" in result.stdout, result.stdout
host = re.search(r"host_bitexact=(\d+)/(\d+)", result.stdout)
assert host and host.group(1) == host.group(2) and int(host.group(2)) > 200
# The inputs must actually reach the signed-zero and overflow cases.
for counter in ["negative_zero", "nonfinite"]:
    assert int(re.search(rf"{counter}=(\d+)", result.stdout).group(1)) > 0
print(result.stdout, end="")
