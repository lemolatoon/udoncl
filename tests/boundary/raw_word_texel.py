"""Check that the RGBA8 word texel decode and encode are exact for every word.

On Windows the same copies also run with four words per RGBA32 float texel
(`texel_format: rgba32_float`), which must move every bit pattern unchanged
through a point Load and a render-target write.

Two OpenCL C copy kernels (uint32 and float32 words) go through the gather
lowering; the native fragment renderer runs them over a full surface whose
words cover every byte value in every position, NaN payloads, infinities,
signed zeros and denormals. Every output word must equal its input word.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

compiler, clang, runner, output = sys.argv[1:]
output = Path(output).resolve()
output.mkdir(parents=True, exist_ok=True)
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


def kernel(name, kernel_id, element, texel_format=None):
    source = output / f"{name}.cl"
    source.write_text(
        f"__kernel void {name}(__global const {element} *input, "
        f"__global {element} *output) {{\n"
        "  uint gid = (uint)get_global_id(1) * 8192u + (uint)get_global_id(0);\n"
        "  output[gid] = input[gid];\n"
        "}\n",
        encoding="utf-8",
    )
    ir = output / f"{name}.ll"
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
            ir,
        ]
    )
    config = output / f"{name}.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "gather-f32",
                "words": 131072,
                "global_size": [8192, 16, 1],
                "local_size": [1, 1, 1],
                "global_offset": [0, 0, 0],
                "kernel": name,
                "kernel_id": kernel_id,
                **({"texel_format": texel_format} if texel_format else {}),
            }
        ),
        encoding="utf-8",
    )
    bundle = output / f"{name}-{texel_format or 'rgba8'}-bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    run([compiler, "--emit-native-device-bundle", config, "--device-llvm", ir, "-o", bundle])
    hlsl = bundle / f"kernels/kernel_{kernel_id}.hlsl"
    text = hlsl.read_text(encoding="utf-8")
    if texel_format:
        # Words are bits: read with asuint, written with asfloat.
        assert "Texture2D<float4>" in text and "asfloat(_vrc_out)" in text, text
        return hlsl
    # Two exact 16-bit halves, not four rounded bytes.
    assert "round(" not in text and "uint2 bytes" in text, text
    return hlsl


fragments = [kernel("copy_words", 40, "uint"), kernel("copy_floats", 41, "float")]
vertex = output / "fullscreen.hlsl"
vertex.write_text(
    "float4 vert(uint id : SV_VertexID) : SV_Position { return float4(id == 2 ? 3.0 : -1.0, id == 1 ? 3.0 : -1.0, 0, 1); }\n",
    encoding="utf-8",
)
if sys.platform != "win32":
    glslang = shutil.which("glslangValidator")
    assert glslang, "install glslang-tools"
    compiled = []
    for index, (asset, stage, entry) in enumerate(
        [(fragments[0], "frag", "frag"), (fragments[1], "frag", "frag"), (vertex, "vert", "vert")]
    ):
        binary = output / f"shader_{index}.spv"
        run([glslang, "-D", "-V", "-S", stage, "-e", entry, "--auto-map-locations", asset, "-o", binary])
        compiled.append(binary)
    *fragments, vertex = compiled

if sys.platform == "win32":
    fragments += [
        kernel("copy_words", 42, "uint", "rgba32_float"),
        kernel("copy_floats", 43, "float", "rgba32_float"),
    ]
result = run([runner, *fragments, vertex])
if sys.platform == "win32":
    assert "word_uint_exact=131072/131072" in result.stdout, result.stdout
    assert "word_float_exact=131072/131072" in result.stdout, result.stdout
assert "uint_exact=131072/131072" in result.stdout, result.stdout
assert "float_exact=131072/131072" in result.stdout, result.stdout
assert f"draws={len(fragments)}" in result.stdout, result.stdout
print(result.stdout, end="")
