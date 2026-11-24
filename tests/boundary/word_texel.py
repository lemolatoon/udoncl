"""Check that four-word texels change only where the words are stored.

Every library kernel the ONNX transformer artifacts dispatch is lowered twice
through the gather path: with one word per RGBA8 texel (the default) and with
four words per RGBA32 float texel (`texel_format: rgba32_float`), where
one fragment computes the four words of one texel. The native fragment
renderer runs both over the same 8192 x 16 word inputs with the scalar
arguments of real dispatches (and shapes that are not multiples of four, which
exercise the per-lane fallback), and every output word must be bit-identical.
The word-texel Q4 MatMulNBits kernel must also read each 32-element block with
the documented number of texel Loads.

A synthesized partial-output kernel (partial_output_kernel.mlir) draws onto a
target that already holds words: in both formats exactly the rows it keeps
change, and every other word stays as it was (HLSL `discard`).
"""

import json
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys

compiler, clang, runner, output = sys.argv[1:]
output = Path(output).resolve()
if output.exists():
    shutil.rmtree(output)
output.mkdir(parents=True)
root = Path(__file__).resolve().parents[2]
# The entries udonclc derives for these kernels when the TinyLlama artifacts
# dispatch them (tools/vrccl/UnityDeviceKernels.cpp).
library = json.loads(
    (root / "tests/boundary/word_texel_kernels.json").read_text(encoding="utf-8")
)
configs = {entry["kernel"]: entry for entry in library["device_kernels"]}
SURFACE = 131072


def run(command, timeout=600):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    return result


sources = {
    "udonaten_arena_rms_norm_reduce": "udonaten/device/arena_rms_norm.cl",
    "udonaten_arena_rms_norm_finalize": "udonaten/device/arena_rms_norm.cl",
    "udonaten_argmax_reduce": "udonaten/device/argmax.cl",
    "udonaten_argmax_finalize": "udonaten/device/argmax.cl",
    "udonblas_arena_sgemm": "udonblas/device/arena_sgemm.cl",
    "udonblas_arena_sgemm_strided_batched": "udonblas/device/arena_sgemm.cl",
    "udondnn_arena_softmax_forward": "udondnn/device/arena_softmax.cl",
    "udondnn_large_op_tensor": "udondnn/device/large_pointwise.cl",
    "udondnn_large_activation_forward": "udondnn/device/large_pointwise.cl",
    "udononnx_matmul_nbits_q4": "udononnxoperators/device/matmul_nbits.cl",
    "udononnx_concat2_f32": "udononnxoperators/device/reindex.cl",
    "udononnx_gather_axis0_f32": "udononnxoperators/device/reindex.cl",
    "udononnx_cast_i64_f32": "udononnxoperators/device/reindex.cl",
    "udononnx_equal_f32": "udononnxoperators/device/reindex.cl",
    "udononnx_where_f32": "udononnxoperators/device/reindex.cl",
    "udononnx_unary_f32": "udononnxoperators/device/unary.cl",
}
compiled = {}
for source in sorted(set(sources.values())):
    ir = output / (source.replace("/", "_") + ".ll")
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
            root / "libraries" / source,
            "-o",
            ir,
        ]
    )
    compiled[source] = ir


def surface(kernel):
    return 16384 if configs[kernel]["global_size"][1] == 1 else SURFACE


def lower(kernel, bounds, texel_format):
    config = dict(configs[kernel])
    buffers = config["argument_kinds"].count("buffer")
    # One-dimensional kernels keep a one-dimensional launch of two rows.
    words = surface(kernel)
    config.update(
        words=words,
        global_size=[words, 1, 1] if words < SURFACE else [8192, 16, 1],
        input_words=[words] * (buffers - 1),
    )
    if bounds is not None:
        config["argument_bounds"] = bounds
    name = f"{kernel}-{texel_format}"
    if texel_format != "rgba8_unorm":
        config["texel_format"] = texel_format
    path = output / f"{name}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    bundle = output / name
    run(
        [
            compiler,
            "--emit-native-device-bundle",
            path,
            "--device-llvm",
            compiled[sources[kernel]],
            "-o",
            bundle,
        ]
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["device"]["kernels"][0]
    expected_format = texel_format
    assert entry["target_texture"]["format"] == expected_format, entry
    assert entry["target_texture"]["width"] == (
        8192 if texel_format == "rgba8_unorm" else 2048
    ), entry
    for argument in entry["args"]:
        if argument["kind"] == "buffer" and texel_format != "rgba8_unorm":
            assert argument["format"] == texel_format, argument
    return bundle / f"kernels/kernel_{config['kernel_id']}.hlsl", config


rng = random.Random(0x7E7E1)
SPECIALS = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x00000001,
            0x807FFFFF, 0x3F800000]


def words(kind, count):
    """One input surface: floats, small integers, or int64 pairs."""
    if kind == "f":
        values = [struct.unpack("<I", struct.pack("<f", rng.uniform(-4, 4)))[0]
                  for _ in range(count)]
        for index in range(501, count, 997):
            values[index] = SPECIALS[index % len(SPECIALS)]
        return values
    if kind == "p":
        return [struct.unpack("<I", struct.pack("<f", rng.uniform(0.25, 4)))[0]
                for _ in range(count)]
    if kind == "u":
        return [rng.getrandbits(32) for _ in range(count)]
    if kind.startswith("i64:"):
        limit = int(kind[4:])
        values = []
        while len(values) < count:
            value = rng.randrange(-limit, limit)
            values += [value & 0xFFFFFFFF, 0 if value >= 0 else 0xFFFFFFFF]
        return values
    if kind.startswith("b"):
        return [struct.unpack("<I", struct.pack("<f", float(rng.random() < 0.5)))[0]
                for _ in range(count)]
    raise AssertionError(kind)


def payload(kernel, scalars):
    kinds = [kind for kind in configs[kernel]["argument_kinds"] if kind != "buffer"]
    assert len(kinds) == len(scalars), (kernel, scalars)
    data = b""
    for kind, value in zip(kinds, scalars):
        data += struct.pack("<f" if kind == "f32" else "<i", value)
    return data.hex() or "-"


q4_bounds = [
    {"index": 5, "min": 1, "max": 4},
    {"index": 6, "min": 1, "max": 512},
    {"index": 7, "min": 1, "max": 2048},
    {"index": 8, "min": 0, "max": 64},
    {"index": 9, "min": 1, "max": 64},
    {"index": 10, "min": 0, "max": 64},
]
# The table and the indices of a Gather fit one 16384-word surface.
gather_bounds = [
    {"index": 3, "min": 1, "max": 8192},
    {"index": 4, "min": 1, "max": 2048},
    {"index": 5, "min": 1, "max": 8},
    {"index": 6, "min": 1, "max": 2},
]
cast_bounds = [{"index": 2, "min": 1, "max": 8192}]
# (case name, kernel, scalars, input kinds, argument bounds or None)
cases = [
    ("rms_reduce", "udonaten_arena_rms_norm_reduce", [2048, 2048], ["f"], None),
    ("rms_reduce_odd", "udonaten_arena_rms_norm_reduce", [300, 75], ["f"], None),
    ("rms_finalize", "udonaten_arena_rms_norm_finalize",
     [2048, 2048, 1e-5, 1], ["f", "p", "f"], None),
    ("rms_finalize_odd", "udonaten_arena_rms_norm_finalize",
     [300, 75, 1e-5, 1], ["f", "p", "f"], None),
    ("sgemm_vector", "udonblas_arena_sgemm", [32, 1, 1, 0, 0, 1.0], ["f", "f"], None),
    ("sgemm", "udonblas_arena_sgemm", [8, 64, 48, 0, 0, 1.0], ["f", "f"], None),
    ("sgemm_odd", "udonblas_arena_sgemm", [7, 13, 9, 1, 1, 0.5], ["f", "f"], None),
    ("sgemm_qk", "udonblas_arena_sgemm_strided_batched",
     [8, 512, 64, 512, 32768, 4096, 4, 0, 0, 1.0], ["f", "f"], None),
    ("sgemm_pv", "udonblas_arena_sgemm_strided_batched",
     [8, 64, 512, 4096, 32768, 512, 4, 0, 0, 1.0], ["f", "f"], None),
    ("sgemm_batched_odd", "udonblas_arena_sgemm_strided_batched",
     [5, 7, 33, 165, 231, 35, 3, 0, 1, 1.0], ["f", "f"], None),
    ("softmax_rows", "udondnn_arena_softmax_forward", [32, 64, 1, 1, 0, 1], ["f"], None),
    ("softmax_odd", "udondnn_arena_softmax_forward", [4, 37, 1, 3, 1, 1], ["f"], None),
    ("softmax_instance", "udondnn_arena_softmax_forward", [6, 50, 1, 1, 2, 0], ["f"], None),
    ("op_tensor", "udondnn_large_op_tensor",
     [2048, 65537, 134217729, 65537, 134217729, 65537, 134217729, 256, 1.0, 1.0, 0.0],
     ["f", "f", "f"], None),
    ("op_tensor_mul", "udondnn_large_op_tensor",
     [512, 65537, 33554433, 65537, 33554433, 65537, 33554433, 257, 1.0, 1.0, 0.0],
     ["f", "f", "f"], None),
    ("q4_partial", "udononnx_matmul_nbits_q4", [1, 64, 2048, 16, 4, 0],
     ["f", "u", "f", "f"], q4_bounds),
    ("q4_sum", "udononnx_matmul_nbits_q4", [1, 64, 2048, 0, 1, 4],
     ["f", "u", "f", "f"], q4_bounds),
    ("q4_serial_odd", "udononnx_matmul_nbits_q4", [3, 33, 2048, 64, 1, 0],
     ["f", "u", "f", "f"], q4_bounds),
    ("q4_tail_odd", "udononnx_matmul_nbits_q4", [2, 37, 145, 3, 2, 0],
     ["f", "u", "f", "f"], q4_bounds),
    ("q4_tail_sum_odd", "udononnx_matmul_nbits_q4", [2, 37, 145, 0, 1, 2],
     ["f", "u", "f", "f"], q4_bounds),
    ("q4_wide", "udononnx_matmul_nbits_q4", [1, 512, 512, 4, 4, 0],
     ["f", "u", "f", "f"], q4_bounds),
    ("concat", "udononnx_concat2_f32", [64, 1, 32, 32], ["f", "f"], None),
    ("concat_odd", "udononnx_concat2_f32", [90, 3, 13, 17], ["f", "f"], None),
    ("gather", "udononnx_gather_axis0_f32", [2048, 512, 8, 2], ["f", "i64:8"],
     gather_bounds),
    ("gather_odd", "udononnx_gather_axis0_f32", [1501, 79, 7, 2], ["f", "i64:7"],
     gather_bounds),
    ("cast", "udononnx_cast_i64_f32", [512], ["u"], cast_bounds),
    ("cast_odd", "udononnx_cast_i64_f32", [37], ["i64:100000"], cast_bounds),
    ("equal", "udononnx_equal_f32",
     [512, 260, 65537, 33554433, 65537, 65537, 65537, 33554433], ["b", "b"], None),
    ("where", "udononnx_where_f32",
     [512, 2308, 65537, 33554433, 65537, 65537, 65537, 33554433, 65537, 33554433],
     ["b", "f", "f"], None),
    ("unary_cos", "udononnx_unary_f32", [1, 64], ["f"], None),
    ("unary_sin", "udononnx_unary_f32", [2, 6000], ["f"], None),
    ("unary_neg", "udononnx_unary_f32", [3, 16383], ["f"], None),
    ("argmax_reduce", "udonaten_argmax_reduce", [0, 16000], ["f"],
     [{"index": 2, "min": 0, "max": 0}, {"index": 3, "min": 1, "max": 16384}]),
    ("argmax_finalize", "udonaten_argmax_finalize", [0, 16000, 1], ["f", "f"],
     [{"index": 3, "min": 0, "max": 0}, {"index": 4, "min": 1, "max": 16384},
      {"index": 5, "min": 0, "max": 1}]),
]
lowered = {}
lines = []
for name, kernel, scalars, inputs, bounds in cases:
    key = (kernel, json.dumps(bounds))
    if key not in lowered:
        narrow, _ = lower(kernel, bounds, "rgba8_unorm")
        wide, _ = lower(kernel, bounds, "rgba32_float")
        lowered[key] = (narrow, wide)
    narrow, wide = lowered[key]
    assert len(inputs) == configs[kernel]["argument_kinds"].count("buffer") - 1
    count = surface(kernel)
    data = b"".join(struct.pack(f"<{count}I", *words(kind, count)[:count])
                    for kind in inputs)
    path = output / f"{name}.bin"
    path.write_bytes(data)
    lines.append(
        "\t".join([name, str(narrow), str(wide), str(len(inputs)),
                   payload(kernel, scalars), str(path), str(count)])
    )
# The partial-output kernel draws rows [first, first + count) of 64 words onto
# a target of random words; the second case crosses the surface row.
partial_config = {
    "schema_version": 1,
    "profile": "gather-f32",
    "local_size": [1, 1, 1],
    "global_offset": [0, 0, 0],
    "kernel": "partial_rows",
    "kernel_id": 9,
    "words": 16384,
    "global_size": [8192, 2, 1],
    "input_words": [16384],
}
partial = {}
for texel_format in ("rgba8_unorm", "rgba32_float"):
    config = dict(partial_config)
    if texel_format != "rgba8_unorm":
        config["texel_format"] = texel_format
    path = output / f"partial_rows-{texel_format}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    bundle = output / f"partial_rows-{texel_format}"
    run(
        [
            compiler,
            "--emit-native-device-bundle",
            path,
            "--device-mlir",
            root / "tests/boundary/partial_output_kernel.mlir",
            "-o",
            bundle,
        ]
    )
    partial[texel_format] = bundle / "kernels/kernel_9.hlsl"
    assert "discard;" in partial[texel_format].read_text(encoding="utf-8")
partial_cases = []
for name, first, count in (("partial_block", 37, 5), ("partial_straddle", 120, 20)):
    source = words("p", 16384)
    destination = words("u", 16384)
    path = output / f"{name}.bin"
    path.write_bytes(struct.pack("<16384I", *source))
    target = output / f"{name}.target"
    target.write_bytes(struct.pack("<16384I", *destination))
    lines.append(
        "	".join([name, str(partial["rgba8_unorm"]), str(partial["rgba32_float"]),
                   "1", struct.pack("<ii", first, count).hex(), str(path),
                   "16384", str(target)])
    )
    partial_cases.append((name, first, count, source, destination, target))
case_file = output / "cases.txt"
case_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
vertex = output / "fullscreen.hlsl"
vertex.write_text(
    "float4 vert(uint id : SV_VertexID) : SV_Position { return float4(id == 2 "
    "? 3.0 : -1.0, id == 1 ? 3.0 : -1.0, 0, 1); }\n",
    encoding="utf-8",
)
result = subprocess.run(
    [str(runner), str(vertex), str(case_file)],
    text=True,
    capture_output=True,
    timeout=3000,
)
print(result.stdout, end="")
assert result.returncode == 0, (result.stdout, result.stderr)
for name, *_ in cases:
    count = surface(dict((case[0], case[1]) for case in cases)[name])
    assert re.search(rf"^{name} exact={count}/{count} ", result.stdout, re.M), name
for name, first, count, source, destination, target in partial_cases:
    assert re.search(rf"^{name} exact=16384/16384 ", result.stdout, re.M), name
    drawn = struct.unpack("<16384I", (output / f"{name}.target.out").read_bytes())
    for index in range(16384):
        if first * 64 <= index < (first + count) * 64:
            value = struct.unpack("<f", struct.pack("<I", source[index]))[0]
            expected = struct.unpack("<I", struct.pack("<f", value + value))[0]
        else:
            expected = destination[index]
        assert drawn[index] == expected, (name, index, drawn[index], expected)

# A full Q4 block of the word-texel kernel: 1 scale and 1 packed-weight texel
# per lane, and the 32 shared activations from 9 texels of one funnelled run.
q4 = lowered[("udononnx_matmul_nbits_q4", json.dumps(q4_bounds))][1].read_text(
    encoding="utf-8"
)
assert "_VRC_Funnel(" in q4 and "float4 frag(" in q4 and "Texture2D<float4>" in q4
