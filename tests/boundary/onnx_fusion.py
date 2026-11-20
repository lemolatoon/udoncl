"""Check the fused kernels of ONNX Host finalization against the library.

Host finalization synthesizes one gather kernel for every fusion group of
onnx_fusion_graph.mlir. The carrier gpu.module is taken from its output as it
is after the passes that follow finalization, each kernel goes through
--device-mlir (which proves it), and the unfused library kernels go through
the OpenCL C/LLVM path. The native fragment renderer then runs both and must
return bit-identical output words.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

compiler, optimizer, clang, runner, output = sys.argv[1:]
output = Path(output).resolve()
output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
environment = dict(os.environ)
if sys.platform != "win32":
    icds = list(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json"))
    assert icds, "install mesa-vulkan-drivers and libvulkan-dev"
    environment["VK_DRIVER_FILES"] = environment["VK_ICD_FILENAMES"] = str(icds[0])


def run(command):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        env=environment,
        timeout=300,
    )
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    return result


SURFACE = {"words": 131072, "global_size": [8192, 16, 1]}
# udononnx_unary_f32 addresses its buffers with the first global id alone,
# and a fused kernel launches over the rows its result occupies: one here.
ROW = {"words": 8192, "global_size": [8192, 1, 1]}


def emit(name, kernel_id, geometry, source_option, source, extra=None):
    config = output / f"{name}.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "gather-f32",
                "local_size": [1, 1, 1],
                "global_offset": [0, 0, 0],
                "kernel": name,
                "kernel_id": kernel_id,
                **geometry,
                **(extra or {}),
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
            source_option,
            source,
            "-o",
            bundle,
        ]
    )
    return bundle / f"kernels/kernel_{kernel_id}.hlsl"


def compile_opencl(source):
    device_ir = output / (source.stem + ".ll")
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
            source,
            "-o",
            device_ir,
        ]
    )
    return device_ir


pointwise = compile_opencl(root / "libraries/udondnn/device/large_pointwise.cl")
unary = compile_opencl(root / "libraries/udononnxoperators/device/unary.cl")
fragments = [
    emit("udondnn_large_op_tensor", 234, SURFACE, "--device-llvm", pointwise),
    emit("udondnn_large_activation_forward", 233, SURFACE, "--device-llvm", pointwise),
    emit("udononnx_unary_f32", 222, ROW, "--device-llvm", unary),
]

# The pipeline of scripts/compile_onnx_mlir.py from Host planning on.
PIPELINE = [
    "--udoncl-validate-onnx-host-graph",
    "--udoncl-plan-onnx-host-graph",
    "--udoncl-materialize-onnx-host-globals",
    "--udoncl-materialize-onnx-host-model-registry",
    "--udoncl-attach-onnx-host-metadata",
    "--udoncl-finalize-onnx-host",
    "--udoncl-cleanup-onnx-host-source",
    "--canonicalize",
]
host = output / "host.mlir"
run([optimizer, *PIPELINE, root / "tests/boundary/onnx_fusion_graph.mlir", "-o", host])
text = host.read_text(encoding="utf-8")
# One dispatch per result and no library dispatch: every node is fused.
dispatches = re.findall(r"call @__vrc_opencl_dispatch_(\w+)\(", text)
assert dispatches == [f"udononnx_fused_{index}" for index in range(4)], dispatches

start = text.index("gpu.module @udononnx_fused_kernels {")
depth = 0
for end in range(text.index("{", start), len(text)):
    depth += {"{": 1, "}": -1}.get(text[end], 0)
    if depth == 0:
        break
carrier = output / "fused-kernels.mlir"
carrier.write_text(text[start : end + 1] + "\n", encoding="utf-8")

# Leaf buffers of the four kernels: x, cos, sin; x, y; k; x, y.
expected_inputs = [[256, 64, 64], [256, 256], [32], [256, 256]]
kernels = re.findall(r"gpu\.func @(\w+)\((.*?)\) kernel", carrier.read_text("utf-8"))
assert [name for name, _ in kernels] == [f"udononnx_fused_{i}" for i in range(4)]
for index, ((name, arguments), inputs) in enumerate(zip(kernels, expected_inputs)):
    sizes = [int(size) for size in re.findall(r"memref<(\d+)xf32>", arguments)]
    assert sizes == inputs + [8192], (name, sizes)
    fragment = emit(
        name, 300 + index, ROW, "--device-mlir", carrier, {"input_words": inputs}
    )
    hlsl = fragment.read_text(encoding="utf-8")
    assert "precise float" in hlsl
    assert not re.search(r"\b(mad|fma)\s*\(", hlsl)
    recovered = (fragment.parents[1] / "frontend.recovered.mlir").read_text("utf-8")
    assert "vrc.gather_proven" in recovered and "vrc.synthesized_kernel" in recovered
    fragments.append(fragment)

vertex = output / "fullscreen.hlsl"
vertex.write_text(
    "float4 vert(uint id : SV_VertexID) : SV_Position { return float4(id == 2 ? 3.0 : -1.0, id == 1 ? 3.0 : -1.0, 0, 1); }\n",
    encoding="utf-8",
)
if sys.platform != "win32":
    glslang = shutil.which("glslangValidator")
    assert glslang, "install glslang-tools"
    compiled = []
    for index, (asset, stage) in enumerate(
        [(fragment, "frag") for fragment in fragments] + [(vertex, "vert")]
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
                stage,
                "--auto-map-locations",
                asset,
                "-o",
                binary,
            ]
        )
        compiled.append(binary)
    *fragments, vertex = compiled

result = subprocess.run(
    [str(runner), *map(str, fragments), str(vertex)],
    text=True,
    capture_output=True,
    env=environment,
    timeout=300,
)
print(result.stdout, end="")
assert result.returncode == 0, (result.stdout, result.stderr)
for name, count in [("rotated", 256), ("gated", 256), ("keys", 96), ("mixed", 256)]:
    assert f"{name}_bitexact={count}/{count}" in result.stdout, result.stdout
    assert f"{name}_tail_zero=1" in result.stdout, result.stdout
    # The inputs must actually reach the non-finite cases.
    for counter in ["nan", "infinite"]:
        found = re.search(rf"{name}_{counter}=(\d+)", result.stdout)
        assert int(found.group(1)) > 0, (name, counter, result.stdout)
assert "draws=19" in result.stdout, result.stdout
