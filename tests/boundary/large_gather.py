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

tile_source = output / "tile-2d.cl"
tile_source.write_text(
    "__kernel void tile_2d(__global const float *input, __global float *output) {\n"
    "  uint x = get_global_id(0);\n"
    "  uint y = get_global_id(1);\n"
    "  uint gid = y * 8u + x;\n"
    "  output[gid] = input[(gid + 1u) % 32u] + (float)(x + y);\n"
    "}\n",
    encoding="utf-8",
)
tile_ir = output / "tile-2d.ll"
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
        tile_source,
        "-o",
        tile_ir,
    ]
)
tile_config = dict(
    config,
    words=32,
    kernel="tile_2d",
    kernel_id=24,
    global_size=[8, 4, 1],
)
config_path.write_text(json.dumps(tile_config), encoding="utf-8")
tile_bundle = output / "tile-2d-bundle"
if tile_bundle.exists():
    shutil.rmtree(tile_bundle)
run(
    [
        compiler,
        "--emit-native-device-bundle",
        config_path,
        "--device-llvm",
        tile_ir,
        "-o",
        tile_bundle,
    ]
)
tile_manifest = json.loads((tile_bundle / "manifest.json").read_text())
tile_kernel = tile_manifest["device"]["kernels"][0]
assert tile_kernel["global_size"] == [8, 4, 1]
assert tile_kernel["target_texture"]["width"] == 8
assert tile_kernel["target_texture"]["height"] == 4
tile_hlsl = (tile_bundle / "kernels/kernel_24.hlsl").read_text()
assert "uint(position.x)" in tile_hlsl
assert "uint(position.y)" in tile_hlsl
assert "_VRC_Input.Load(int3(" in tile_hlsl
assert "%" in tile_hlsl and "/" in tile_hlsl

multi_source = output / "multi-input.cl"
multi_source.write_text(
    "__kernel void add_scaled(__global const float *lhs, "
    "__global const float *rhs, __global float *result, float scale) {\n"
    "  uint gid = get_global_id(0);\n"
    "  result[gid] = (lhs[gid] + rhs[gid]) * scale;\n"
    "}\n",
    encoding="utf-8",
)
multi_ir = output / "multi-input.ll"
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
        multi_source,
        "-o",
        multi_ir,
    ]
)
multi_config = dict(
    config,
    words=16,
    input_words=[32, 64],
    kernel="add_scaled",
    kernel_id=25,
    global_size=[16, 1, 1],
)
config_path.write_text(json.dumps(multi_config), encoding="utf-8")
multi_bundle = output / "multi-input-bundle"
if multi_bundle.exists():
    shutil.rmtree(multi_bundle)
run(
    [
        compiler,
        "--emit-native-device-bundle",
        config_path,
        "--device-llvm",
        multi_ir,
        "-o",
        multi_bundle,
    ]
)
multi_manifest = json.loads((multi_bundle / "manifest.json").read_text())
multi_kernel = multi_manifest["device"]["kernels"][0]
assert [argument["required_bytes"] for argument in multi_kernel["args"][:3]] == [
    128,
    256,
    64,
]
assert [argument["access"] for argument in multi_kernel["args"][:3]] == [
    "read",
    "read",
    "write",
]
assert [argument["binding"] for argument in multi_kernel["args"]] == [
    "_VRC_Input",
    "_VRC_Input1",
    "SV_Target",
    "_VRC_Arg3",
]
assert multi_kernel["args"][3]["scalar_payload_offset"] == 0
assert multi_kernel["descriptor"]["resource_order"] == [0, 1, 2]
assert multi_kernel["descriptor"]["scalar_payload_bytes"] == 4
assert multi_kernel["passes"][0] == {
    "id": 0,
    "output_arg": 2,
    "reads": [0, 1],
    "writes": [2],
}
multi_hlsl = (multi_bundle / "kernels/kernel_25.hlsl").read_text()
assert "Texture2D<float4> _VRC_Input : register(t0);" in multi_hlsl
assert "Texture2D<float4> _VRC_Input1 : register(t1);" in multi_hlsl
assert "cbuffer Params : register(b13)" in multi_hlsl
assert "_VRC_Input.Load" in multi_hlsl and "_VRC_Input1.Load" in multi_hlsl
assert "float _VRC_Arg3;" in multi_hlsl
multi_shader = (multi_bundle / "kernels/kernel_25.shader").read_text()
assert '_VRC_Input1 ("Input 1", 2D)' in multi_shader

bad_input_words = dict(multi_config, input_words=[32])
config_path.write_text(json.dumps(bad_input_words), encoding="utf-8")
run(
    [
        compiler,
        "--emit-native-device-bundle",
        config_path,
        "--device-llvm",
        multi_ir,
        "-o",
        output / "bad-input-words",
    ],
    error="input_words must provide one positive capacity per input",
)

subview_source = output / "subview.cl"
subview_source.write_text(
    "__kernel void subview(__global const float *input, "
    "__global float *output, int base) {\n"
    "  uint gid = get_global_id(0);\n"
    "  output[gid] = input[base + gid];\n"
    "}\n",
    encoding="utf-8",
)
subview_ir = output / "subview.ll"
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
        subview_source,
        "-o",
        subview_ir,
    ]
)
subview_config = dict(
    config,
    words=16,
    input_words=[64],
    argument_bounds=[{"index": 2, "min": 8, "max": 16}],
    kernel="subview",
    kernel_id=26,
    global_size=[16, 1, 1],
)
config_path.write_text(json.dumps(subview_config), encoding="utf-8")
subview_bundle = output / "subview-bundle"
if subview_bundle.exists():
    shutil.rmtree(subview_bundle)
run(
    [
        compiler,
        "--emit-native-device-bundle",
        config_path,
        "--device-llvm",
        subview_ir,
        "-o",
        subview_bundle,
    ]
)
subview_kernel = json.loads(
    (subview_bundle / "manifest.json").read_text(encoding="utf-8")
)["device"]["kernels"][0]
assert [argument["required_bytes"] for argument in subview_kernel["args"][:2]] == [
    256,
    64,
]
assert subview_kernel["args"][2]["minimum"] == 8
assert subview_kernel["args"][2]["maximum"] == 16
subview_hlsl = (subview_bundle / "kernels/kernel_26.hlsl").read_text(
    encoding="utf-8"
)
assert "64u;" in subview_hlsl

bad_subview_config = dict(
    subview_config,
    argument_bounds=[{"index": 2, "min": 8, "max": 60}],
)
config_path.write_text(json.dumps(bad_subview_config), encoding="utf-8")
run(
    [
        compiler,
        "--emit-native-device-bundle",
        config_path,
        "--device-llvm",
        subview_ir,
        "-o",
        output / "bad-subview-bounds",
    ],
    error="buffer index bounds are not proven within the declared extent",
)

packed_source = output / "packed.cl"
packed_source.write_text(
    "__kernel void packed(__global const uint *input, "
    "__global float *output, int base) {\n"
    "  uint gid = get_global_id(0);\n"
    "  uint word = input[base + gid];\n"
    "  output[gid] = (float)(word & 15u);\n"
    "}\n",
    encoding="utf-8",
)
packed_ir = output / "packed.ll"
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
        packed_source,
        "-o",
        packed_ir,
    ]
)
packed_config = dict(
    subview_config,
    kernel="packed",
    kernel_id=27,
)
config_path.write_text(json.dumps(packed_config), encoding="utf-8")
packed_bundle = output / "packed-bundle"
if packed_bundle.exists():
    shutil.rmtree(packed_bundle)
run(
    [
        compiler,
        "--emit-native-device-bundle",
        config_path,
        "--device-llvm",
        packed_ir,
        "-o",
        packed_bundle,
    ]
)
packed_kernel = json.loads(
    (packed_bundle / "manifest.json").read_text(encoding="utf-8")
)["device"]["kernels"][0]
assert packed_kernel["args"][0]["element_type"] == "uint32"
assert packed_kernel["args"][1]["element_type"] == "float32"
packed_hlsl = (packed_bundle / "kernels/kernel_27.hlsl").read_text(
    encoding="utf-8"
)
assert "asfloat(bytes" not in packed_hlsl
assert " = bytes" in packed_hlsl and " = float(" in packed_hlsl

two_outputs_source = output / "two-outputs.cl"
two_outputs_source.write_text(
    "__kernel void two_outputs(__global const float *input, "
    "__global float *first, __global float *second) {\n"
    "  uint gid = get_global_id(0);\n"
    "  first[gid] = input[gid];\n"
    "  second[gid] = input[gid];\n"
    "}\n",
    encoding="utf-8",
)
two_outputs_ir = output / "two-outputs.ll"
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
        two_outputs_source,
        "-o",
        two_outputs_ir,
    ]
)
two_outputs_config = dict(
    config,
    words=16,
    kernel="two_outputs",
    kernel_id=26,
    global_size=[16, 1, 1],
)
config_path.write_text(json.dumps(two_outputs_config), encoding="utf-8")
run(
    [
        compiler,
        "--emit-native-device-bundle",
        config_path,
        "--device-llvm",
        two_outputs_ir,
        "-o",
        output / "two-outputs-bundle",
    ],
    error="exactly one trailing float output",
)

invalid_tile = tile_source.read_text().replace("output[gid]", "output[x]")
invalid_source = output / "invalid-tile-2d.cl"
invalid_source.write_text(invalid_tile, encoding="utf-8")
invalid_ir = output / "invalid-tile-2d.ll"
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
        invalid_source,
        "-o",
        invalid_ir,
    ]
)
config_path.write_text(json.dumps(tile_config), encoding="utf-8")
run(
    [
        compiler,
        "--emit-native-device-bundle",
        config_path,
        "--device-llvm",
        invalid_ir,
        "-o",
        output / "invalid-tile-2d",
    ],
    error="requires exactly one complete output[global_id] store",
)

large_2d_source = output / "large-2d.cl"
large_2d_source.write_text(
    "__kernel void large_2d(__global const float *input, __global float *output) {\n"
    "  uint x = get_global_id(0);\n"
    "  uint y = get_global_id(1);\n"
    "  uint gid = y * 8192u + x;\n"
    "  output[gid] = input[gid];\n"
    "}\n",
    encoding="utf-8",
)
large_2d_ir = output / "large-2d.ll"
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
        large_2d_source,
        "-o",
        large_2d_ir,
    ]
)
large_2d_config = dict(
    config,
    words=32768,
    kernel="large_2d",
    kernel_id=25,
    global_size=[8192, 4, 1],
)
config_path.write_text(json.dumps(large_2d_config), encoding="utf-8")
large_2d_bundle = output / "large-2d-bundle"
if large_2d_bundle.exists():
    shutil.rmtree(large_2d_bundle)
run(
    [
        compiler,
        "--emit-native-device-bundle",
        config_path,
        "--device-llvm",
        large_2d_ir,
        "-o",
        large_2d_bundle,
    ]
)
large_2d_kernel = json.loads(
    (large_2d_bundle / "manifest.json").read_text(encoding="utf-8")
)["device"]["kernels"][0]
assert large_2d_kernel["global_size"] == [8192, 4, 1]
assert large_2d_kernel["target_texture"]["height"] == 4

too_large = dict(config, words=32769, global_size=[8192, 4, 1])
config_path.write_text(json.dumps(too_large), encoding="utf-8")
run(
    [compiler, "--emit-native-device-bundle", config_path, "--device-llvm", ir, "-o", output / "too-large"],
    error="unsupported config version or launch profile",
)
print(
    "large_gather_words=10800 max_words=32768 tile_2d=8x4 multi_input=2 "
    "upper_bound_rejection=passed"
)
