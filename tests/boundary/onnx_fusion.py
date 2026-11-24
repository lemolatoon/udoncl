"""Check the fused kernels of ONNX Host finalization against the library.

Host finalization synthesizes one gather kernel for every fusion group of
onnx_fusion_graph.mlir. The carrier gpu.module is taken from its output as it
is after the passes that follow finalization, each kernel goes through
--device-mlir (which proves it), and the unfused library kernels go through
the OpenCL C/LLVM path. The native fragment renderer then runs both and must
return bit-identical output words. The two-pass Softmax of
onnx_fusion_two_pass_graph.mlir regroups the sum of exponentials, so it must
instead stay within a few rounding errors of the library and be deterministic.
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


def emit(name, kernel_id, geometry, source_option, source, extra=None, label=None):
    label = label or name
    config = output / f"{label}.json"
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
    bundle = output / f"{label}-bundle"
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
softmax = compile_opencl(root / "libraries/udondnn/device/arena_softmax.cl")
# The arena Softmax as udonclc derives it from its library contract
# (tools/vrccl/UnityDeviceKernels.cpp).
SOFTMAX = {"words": 16384, "global_size": [8192, 2, 1]}
SOFTMAX_BOUNDS = {
    "argument_bounds": [
        {"index": 2, "min": 1, "max": 16384},
        {"index": 3, "min": 1, "max": 512},
        {"index": 4, "min": 1, "max": 16384},
        {"index": 5, "min": 1, "max": 16384},
        {"index": 6, "min": 0, "max": 2},
        {"index": 7, "min": 0, "max": 1},
    ]
}
fragments = [
    emit("udondnn_large_op_tensor", 234, SURFACE, "--device-llvm", pointwise),
    emit("udondnn_large_activation_forward", 233, SURFACE, "--device-llvm", pointwise),
    emit("udononnx_unary_f32", 222, ROW, "--device-llvm", unary),
    emit(
        "udondnn_arena_softmax_forward",
        236,
        SOFTMAX,
        "--device-llvm",
        softmax,
        SOFTMAX_BOUNDS,
    ),
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
def finalize(graph, name):
    """Returns the dispatched callees and the carrier of the finalized graph."""
    host = output / f"{name}.mlir"
    run([optimizer, *PIPELINE, root / "tests/boundary" / graph, "-o", host])
    text = host.read_text(encoding="utf-8")
    dispatches = re.findall(r"call @__vrc_opencl_dispatch_(\w+)\(", text)
    start = text.index("gpu.module @udononnx_fused_kernels {")
    depth = 0
    for end in range(text.index("{", start), len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            break
    carrier = output / f"{name}-kernels.mlir"
    carrier.write_text(text[start : end + 1] + "\n", encoding="utf-8")
    kernels = re.findall(
        r"gpu\.func @(\w+)\((.*?)\) kernel", carrier.read_text("utf-8")
    )
    return dispatches, carrier, kernels


dispatches, carrier, kernels = finalize("onnx_fusion_graph.mlir", "host")
# One dispatch per result and no library dispatch: every node is fused.
assert dispatches == [f"udononnx_fused_{index}" for index in range(5)], dispatches

# Leaf buffers of the five kernels: x, cos, sin; x, y; k; x, y; x, cos.
expected_inputs = [[256, 64, 64], [256, 256], [32], [256, 256], [256, 64]]
assert [name for name, _ in kernels] == [f"udononnx_fused_{i}" for i in range(5)]
for index, (source, (name, arguments), inputs) in enumerate(
    [(carrier, kernel, inputs) for kernel, inputs in zip(kernels, expected_inputs)]
):
    sizes = [int(size) for size in re.findall(r"memref<(\d+)xf32>", arguments)]
    assert sizes == inputs + [8192], (name, sizes)
    fragment = emit(
        name,
        300 + index,
        ROW,
        "--device-mlir",
        source,
        {"input_words": inputs},
        label=f"{source.stem}-{name}",
    )
    hlsl = fragment.read_text(encoding="utf-8")
    assert "precise float" in hlsl
    assert not re.search(r"\b(mad|fma)\s*\(", hlsl)
    recovered = (fragment.parents[1] / "frontend.recovered.mlir").read_text("utf-8")
    assert "vrc.gather_proven" in recovered and "vrc.synthesized_kernel" in recovered
    fragments.append(fragment)

# A Softmax over rows of 512 runs in two passes: the statistics kernel reads
# x and the mask and writes 2 x 32 x chunks words into one surface row, and
# the elementwise kernel also reads that row and writes the 32 x 512 result.
two_pass_dispatches, two_pass_carrier, two_pass_kernels = finalize(
    "onnx_fusion_two_pass_graph.mlir", "two-pass"
)
assert two_pass_dispatches == ["udononnx_fused_0", "udononnx_fused_1"], (
    two_pass_dispatches
)
assert [name for name, _ in two_pass_kernels] == [
    "udononnx_fused_0",
    "udononnx_fused_1",
]
statistics_sizes, elementwise_sizes = (
    [int(size) for size in re.findall(r"memref<(\d+)xf32>", arguments)]
    for _, arguments in two_pass_kernels
)
assert statistics_sizes[:2] == [16384, 512] and statistics_sizes[2] == 8192, (
    statistics_sizes
)
# The elementwise kernel reads the statistics as the whole surface row the
# statistics kernel writes, which is how the operator scratch texture is laid
# out.
assert elementwise_sizes == [16384, 512, 8192, 16384], elementwise_sizes
TWO_ROWS = {"words": 16384, "global_size": [8192, 2, 1]}
for index, ((name, _), inputs, geometry) in enumerate(
    zip(
        two_pass_kernels,
        [[16384, 512], [16384, 512, 8192]],
        [ROW, TWO_ROWS],
    )
):
    fragment = emit(
        name,
        310 + index,
        geometry,
        "--device-mlir",
        two_pass_carrier,
        {"input_words": inputs},
        label=f"{two_pass_carrier.stem}-{name}",
    )
    hlsl = fragment.read_text(encoding="utf-8")
    assert "precise float" in hlsl
    assert not re.search(r"\b(mad|fma)\s*\(", hlsl)
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
for name, count, counters in [
    ("rotated", 256, ["nan", "infinite"]),
    ("gated", 256, ["nan", "infinite"]),
    ("keys", 96, ["nan", "infinite"]),
    ("mixed", 256, ["nan", "infinite"]),
    # A probability is at most 1: the masked rows reach NaN and zeros.
    ("masked", 256, ["nan", "zero"]),
]:
    assert f"{name}_bitexact={count}/{count}" in result.stdout, result.stdout
    assert f"{name}_tail_zero=1" in result.stdout, result.stdout
    # The inputs must actually reach the non-finite cases.
    for counter in counters:
        found = re.search(rf"{name}_{counter}=(\d+)", result.stdout)
        assert int(found.group(1)) > 0, (name, counter, result.stdout)
# The two-pass Softmax is within a few rounding errors of the library kernel
# (the chunk sums are rescaled to the row maximum), NaN exactly where the
# library is, and deterministic: a second run is bit-identical.
assert "long_within_tolerance=16384/16384" in result.stdout, result.stdout
assert "long_nan_match=1" in result.stdout, result.stdout
assert "long_deterministic=1" in result.stdout, result.stdout
for counter in ["nan", "zero", "nonzero"]:
    found = re.search(rf"long_{counter}=(\d+)", result.stdout)
    assert int(found.group(1)) > 0, (counter, result.stdout)
assert "draws=28" in result.stdout, result.stdout
