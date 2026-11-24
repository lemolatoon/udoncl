"""Run the bounded decoder fixture natively against ONNX Runtime.

tests/boundary/onnx_bounded_kv_fixture.py writes a decoder attention layer
whose past is a bounded dynamic dimension, and ONNX Runtime's outputs at
several past lengths. This test compiles the fixture through the ONNX Host
pipeline once with the past bounded and once with it fixed at the bound,
interprets the generated Create, BindMemory and Run of each
(onnx_host_interpreter.py) to record the Device work of a run, and executes
that work with onnx-host-trace: synthesized kernels on the native fragment
renderer, library kernels as their CPU builds. It requires

- the bounded library at every past length of the oracle, with the past and
  present bound densely and as leading blocks of buffers of the bound, to
  match ONNX Runtime (present exactly, attention within float rounding),
  without reading rows outside the leading blocks (they hold NaN);
- the bounded library at the bound to equal the fixed library bit for bit;
- Run to reject a present whose extent disagrees with the past;
- in-place Concat: with each past and present bound to one buffer of the
  bound (the past its leading block), runs that grow the past, restart from
  a smaller one without clearing anything (a reset) and grow again (a
  second turn) to match ONNX Runtime on the rows the buffer holds, with the
  rows written only once each; one buffer bound with two pitches is
  UDONONNX_STATUS_INVALID_VALUE.
"""

import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

compiler, optimizer, frontend, runner, output = sys.argv[1:]
output = Path(output).resolve()
if output.exists():
    shutil.rmtree(output)
output.mkdir(parents=True)
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_host_interpreter import Host, NULL, Pointer  # noqa: E402

environment = dict(os.environ)
if sys.platform != "win32":
    icds = list(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json"))
    assert icds, "install mesa-vulkan-drivers and libvulkan-dev"
    environment["VK_DRIVER_FILES"] = environment["VK_ICD_FILENAMES"] = str(icds[0])

BOUND = 8
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
POISON = 0x7FC00001
INVALID_VALUE = 1


def run(command, **kwargs):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        env=environment,
        timeout=600,
        **kwargs,
    )
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    return result


model = output / "bounded_kv.onnx"
oracle_path = output / "oracle.json"
run(
    [
        sys.executable,
        root / "tests/boundary/onnx_bounded_kv_fixture.py",
        "-o",
        model,
        "--oracle",
        oracle_path,
        "--bound",
        BOUND,
    ]
)
oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
assert [case["past"] for case in oracle["cases"]] == [0, 1, 5, BOUND]

vertex = output / "fullscreen.hlsl"
vertex.write_text(
    "float4 vert(uint id : SV_VertexID) : SV_Position { return float4(id == 2 ? 3.0 : -1.0, id == 1 ? 3.0 : -1.0, 0, 1); }\n",
    encoding="utf-8",
)


def spirv(asset, stage):
    """The fragment or vertex shader in the form the native renderer takes."""
    if sys.platform == "win32":
        return asset
    glslang = shutil.which("glslangValidator")
    assert glslang, "install glslang-tools"
    binary = asset.with_suffix(".spv")
    run([glslang, "-D", "-V", "-S", stage, "-e", stage, "--auto-map-locations", asset, "-o", binary])
    return binary


vertex_shader = spirv(vertex, "vert")


partial_kernels = {}


def compile_library(name, dimension):
    """The finalized Host module of the fixture and its fused kernels."""
    host = output / f"{name}.mlir"
    run(
        [
            sys.executable,
            root / "scripts/compile_onnx_mlir.py",
            model,
            "-o",
            host,
            "--frontend",
            frontend,
            "--dimension",
            "batch=1",
            "--dimension",
            "sequence=1",
            *dimension,
            "--lower-to-host",
        ]
    )
    generic = output / f"{name}.generic.mlir"
    pretty = output / f"{name}.pretty.mlir"
    run([optimizer, host, "--mlir-print-op-generic", "-o", generic])
    run([optimizer, host, "-o", pretty])
    text = pretty.read_text(encoding="utf-8")
    start = text.index("gpu.module @udononnx_fused_kernels {")
    depth = 0
    for end in range(text.index("{", start), len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            break
    carrier = output / f"{name}-kernels.mlir"
    carrier.write_text(text[start : end + 1] + "\n", encoding="utf-8")
    # The partial-output kernels: an in-place Concat writes only new rows.
    partial_kernels[name] = re.findall(
        r"gpu\.func @(\w+)\(.*vrc\.output_granule", text[start : end + 1]
    )
    kernels = []
    for index, match in enumerate(
        re.finditer(
            r"gpu\.func @(\w+)\((.*?)\) kernel attributes \{(.*?)known_grid_size = array<i32: (\d+), (\d+), (\d+)>",
            carrier.read_text(encoding="utf-8"),
        )
    ):
        kernel, arguments, _, width, height, depth_ = match.groups()
        sizes = [int(size) for size in re.findall(r"memref<(\d+)xf32>", arguments)]
        scalars = len(re.findall(r":\s*i32", arguments))
        width, height = int(width), int(height)
        assert int(depth_) == 1 and sizes[-1] == width * height, (kernel, sizes)
        config = output / f"{name}-{kernel}.json"
        kernel_id = 4096 + index
        config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": "gather-f32",
                    "local_size": [1, 1, 1],
                    "global_offset": [0, 0, 0],
                    "kernel": kernel,
                    "kernel_id": kernel_id,
                    "words": width * height,
                    "global_size": [width, height, 1],
                    "input_words": sizes[:-1],
                }
            ),
            encoding="utf-8",
        )
        bundle = output / f"{name}-{kernel}-bundle"
        run(
            [
                compiler,
                "--emit-native-device-bundle",
                config,
                "--device-mlir",
                carrier,
                "-o",
                bundle,
            ]
        )
        fragment = spirv(bundle / f"kernels/kernel_{kernel_id}.hlsl", "frag")
        kernels.append(
            f"fused {kernel} {fragment} {width * height} {width} {height} {len(sizes) - 1} {scalars}"
        )
    return Host(generic.read_text(encoding="utf-8")), kernels


def flatten(values):
    if isinstance(values, list):
        for value in values:
            yield from flatten(value)
    else:
        yield values


def float_words(values):
    return list(struct.unpack(f"<{len(values)}I", struct.pack(f"<{len(values)}f", *values)))


class Session:
    """One interpreted Host and the trace of everything it ran."""

    def __init__(self, name, host, kernels):
        self.name = name
        self.host = host
        self.lines = [f"vertex {vertex_shader}", *kernels]
        self.saves = []
        self.queue = Pointer("opaque", 0x51)
        assert host.call("udononnxGeneratedCreate", 0, NULL, self.queue) == 0
        self.flush()

    def flush(self):
        for entry in self.host.trace:
            if entry[0] == "buffer":
                self.lines.append(f"buffer {entry[1]:x} {entry[2]}")
            elif entry[0] == "words":
                self.lines.append(
                    f"words {entry[1]:x} {len(entry[2])} " + " ".join(f"{word:x}" for word in entry[2])
                )
            elif entry[0] == "dispatch":
                self.lines.append(
                    f"dispatch {entry[1]} {len(entry[2])} " + " ".join(f"{word:x}" for word in entry[2])
                )
            elif entry[0] == "copy":
                self.lines.append("copy " + " ".join(f"{value:x}" if index < 2 else str(value) for index, value in enumerate(entry[1:])))
            else:
                raise AssertionError(entry)
        self.host.trace.clear()

    def buffer(self, words, contents=None, poison=True):
        pointer = self.host.create_buffer(words)
        self.flush()
        if poison:
            self.lines.append(f"fill {pointer.key:x} {POISON:x}")
        if contents is not None:
            self.lines.append(
                f"words {pointer.key:x} {len(contents)} " + " ".join(f"{word:x}" for word in contents)
            )
        return pointer

    def bind(self, output_, index, pointer, extent=0, pitch=0):
        return self.host.call("udononnxGeneratedBindMemory", 0, output_, index, pointer, extent, pitch)

    def run(self):
        status = self.host.call("udononnxGeneratedRun", 0, self.queue, Pointer("alloca", 0))
        self.flush()
        return status

    def save(self, pointer, words, label):
        path = output / f"{self.name}-{label}.bin"
        self.lines.append(f"save {pointer.key:x} {words} {path}")
        self.saves.append(path)
        return path

    def execute(self):
        trace = output / f"{self.name}.trace"
        trace.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        result = run([runner, trace])
        print(f"{self.name}: {result.stdout.strip()}")


def pitched(values, rows, pitch):
    """[1, KV_HEADS, rows, HEAD_DIM] words in a buffer of `pitch` rows, NaN elsewhere."""
    words = [POISON] * (KV_HEADS * max(pitch, 1) * HEAD_DIM)
    flat = float_words(list(flatten(values)))
    for head in range(KV_HEADS):
        for row in range(rows):
            for column in range(HEAD_DIM):
                words[(head * pitch + row) * HEAD_DIM + column] = flat[(head * rows + row) * HEAD_DIM + column]
    return words


def unpitched(words, rows, pitch):
    return [
        words[(head * pitch + row) * HEAD_DIM + column]
        for head in range(KV_HEADS)
        for row in range(rows)
        for column in range(HEAD_DIM)
    ]


def mask_words(length, pitch):
    words = [POISON] * (2 * max(pitch, 1))
    for column in range(length):
        words[2 * column : 2 * column + 2] = [1, 0]
    return words


def bind_case(session, case, past_pitch, present_pitch, mask_pitch, present_extent=None):
    """Binds the inputs and outputs of one oracle case; returns the outputs."""
    past = case["past"]
    inputs = case["inputs"]
    query = session.buffer(HEADS * HEAD_DIM, float_words(list(flatten(inputs["query"]))))
    key = session.buffer(KV_HEADS * HEAD_DIM, float_words(list(flatten(inputs["key"]))))
    value = session.buffer(KV_HEADS * HEAD_DIM, float_words(list(flatten(inputs["value"]))))
    mask = session.buffer(2 * max(mask_pitch, 1), mask_words(past + 1, mask_pitch))
    past_key = session.buffer(
        KV_HEADS * max(past_pitch, 1) * HEAD_DIM, pitched(inputs["past_key"], past, past_pitch)
    )
    past_value = session.buffer(
        KV_HEADS * max(past_pitch, 1) * HEAD_DIM, pitched(inputs["past_value"], past, past_pitch)
    )
    attention = session.buffer(HEADS * HEAD_DIM)
    present_key = session.buffer(KV_HEADS * present_pitch * HEAD_DIM)
    present_value = session.buffer(KV_HEADS * present_pitch * HEAD_DIM)
    present_extent = past + 1 if present_extent is None else present_extent
    dynamic = session.host.globals.get("__udon_onnx_bound_input_extent_3") is not None
    statuses = [
        session.bind(0, 0, query),
        session.bind(0, 1, key),
        session.bind(0, 2, value),
        session.bind(0, 3, mask, past + 1, mask_pitch) if dynamic else session.bind(0, 3, mask),
        session.bind(0, 4, past_key, past, past_pitch) if dynamic else session.bind(0, 4, past_key),
        session.bind(0, 5, past_value, past, past_pitch) if dynamic else session.bind(0, 5, past_value),
        session.bind(1, 0, attention),
        session.bind(1, 1, present_key, present_extent, present_pitch) if dynamic else session.bind(1, 1, present_key),
        session.bind(1, 2, present_value, present_extent, present_pitch) if dynamic else session.bind(1, 2, present_value),
    ]
    assert statuses == [0] * 9, statuses
    return attention, present_key, present_value


def read_words(path):
    data = path.read_bytes()
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def as_float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


bounded_host, bounded_kernels = compile_library("bounded", ["--dimension-bound", f"past={BOUND}"])
fixed_host, fixed_kernels = compile_library("fixed", ["--dimension", f"past={BOUND}"])

bounded = Session("bounded", bounded_host, bounded_kernels)
checks = []
for case in oracle["cases"]:
    past = case["past"]
    for layout, (past_pitch, present_pitch) in {
        "dense": (past, past + 1),
        "leading": (BOUND + 1, BOUND + 1),
    }.items():
        attention, present_key, present_value = bind_case(
            bounded, case, past_pitch, present_pitch, present_pitch
        )
        assert bounded.run() == 0
        label = f"t{past}-{layout}"
        checks.append(
            (
                case,
                present_pitch,
                bounded.save(attention, HEADS * HEAD_DIM, f"{label}-attention"),
                bounded.save(present_key, KV_HEADS * present_pitch * HEAD_DIM, f"{label}-present_key"),
                bounded.save(present_value, KV_HEADS * present_pitch * HEAD_DIM, f"{label}-present_value"),
            )
        )
# A present that is not one row longer than the past.
bind_case(bounded, oracle["cases"][2], 5, 7, 6, present_extent=7)
assert bounded.run() == INVALID_VALUE

# In-place Concat: one buffer of the bound holds each past and its present.
import numpy  # noqa: E402
import onnxruntime  # noqa: E402

reference = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
PITCH = BOUND + 1
cache_words = KV_HEADS * PITCH * HEAD_DIM
key_cache = bounded.buffer(cache_words)
value_cache = bounded.buffer(cache_words)
ones = bounded.buffer(2 * PITCH, mask_words(PITCH, PITCH))
# What the caches hold: NaN until a run writes a row.
held = {
    "key": numpy.full((1, KV_HEADS, PITCH, HEAD_DIM), numpy.nan, numpy.float32),
    "value": numpy.full((1, KV_HEADS, PITCH, HEAD_DIM), numpy.nan, numpy.float32),
}
generator = numpy.random.default_rng(0x1A7E)
in_place_checks = []
# Grow, reset to an empty past (nothing is cleared), grow again to the bound.
for step, past in enumerate([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8]):
    feeds = {
        name: generator.uniform(-1.0, 1.0, shape).astype(numpy.float32)
        for name, shape in (
            ("query", (1, HEADS, 1, HEAD_DIM)),
            ("key", (1, KV_HEADS, 1, HEAD_DIM)),
            ("value", (1, KV_HEADS, 1, HEAD_DIM)),
        )
    }
    query = bounded.buffer(HEADS * HEAD_DIM, float_words(feeds["query"].ravel().tolist()))
    key = bounded.buffer(KV_HEADS * HEAD_DIM, float_words(feeds["key"].ravel().tolist()))
    value = bounded.buffer(KV_HEADS * HEAD_DIM, float_words(feeds["value"].ravel().tolist()))
    attention = bounded.buffer(HEADS * HEAD_DIM)
    statuses = [
        bounded.bind(0, 0, query),
        bounded.bind(0, 1, key),
        bounded.bind(0, 2, value),
        bounded.bind(0, 3, ones, past + 1, PITCH),
        bounded.bind(0, 4, key_cache, past, PITCH),
        bounded.bind(0, 5, value_cache, past, PITCH),
        bounded.bind(1, 0, attention),
        bounded.bind(1, 1, key_cache, past + 1, PITCH),
        bounded.bind(1, 2, value_cache, past + 1, PITCH),
    ]
    assert statuses == [0] * 9, statuses
    assert bounded.run() == 0
    feeds["attention_mask"] = numpy.ones((1, past + 1), numpy.int64)
    feeds["past_key"] = held["key"][:, :, :past]
    feeds["past_value"] = held["value"][:, :, :past]
    expected = dict(
        zip(
            [output.name for output in reference.get_outputs()],
            reference.run(None, feeds),
        )
    )
    held["key"][:, :, past] = feeds["key"][:, :, 0]
    held["value"][:, :, past] = feeds["value"][:, :, 0]
    in_place_checks.append(
        (
            past,
            expected,
            bounded.save(attention, HEADS * HEAD_DIM, f"in-place{step}-attention"),
            bounded.save(key_cache, cache_words, f"in-place{step}-present_key"),
            bounded.save(value_cache, cache_words, f"in-place{step}-present_value"),
        )
    )
# One buffer bound with two pitches is not a leading block.
bounded.bind(0, 4, key_cache, 3, BOUND)
bounded.bind(1, 1, key_cache, 4, PITCH)
bounded.bind(0, 3, ones, 4, PITCH)
bounded.bind(0, 5, value_cache, 3, PITCH)
bounded.bind(1, 2, value_cache, 4, PITCH)
assert bounded.run() == INVALID_VALUE
# Every in-place run wrote the present key and value with the partial-output
# kernel.
assert len(partial_kernels["bounded"]) == 1, partial_kernels
assert partial_kernels["fixed"] == [], partial_kernels
in_place_dispatches = sum(
    line.startswith(f"dispatch {partial_kernels['bounded'][0]} ")
    for line in bounded.lines
)
assert in_place_dispatches == 2 * len(in_place_checks), in_place_dispatches
bounded.execute()

fixed = Session("fixed", fixed_host, fixed_kernels)
case = oracle["cases"][-1]
fixed_outputs = bind_case(fixed, case, BOUND, BOUND + 1, BOUND + 1)
assert fixed.run() == 0
fixed_paths = [
    fixed.save(fixed_outputs[0], HEADS * HEAD_DIM, "attention"),
    fixed.save(fixed_outputs[1], KV_HEADS * (BOUND + 1) * HEAD_DIM, "present_key"),
    fixed.save(fixed_outputs[2], KV_HEADS * (BOUND + 1) * HEAD_DIM, "present_value"),
]
fixed.execute()

worst = 0.0
for case, pitch, attention_path, key_path, value_path in checks:
    past = case["past"]
    expected = case["outputs"]
    for name, path in (("present_key", key_path), ("present_value", value_path)):
        words = unpitched(read_words(path), past + 1, pitch)
        assert words == float_words(list(flatten(expected[name]))), (past, pitch, name)
    attention = [as_float(word) for word in read_words(attention_path)]
    for got, want in zip(attention, flatten(expected["attention"])):
        assert math.isfinite(got), (past, pitch, attention)
        worst = max(worst, abs(got - want))
        assert abs(got - want) <= 1e-5 + 1e-5 * abs(want), (past, pitch, got, want)
    print(f"bounded past={past} pitch={pitch}: present exact, attention within {worst:.3g}")

for past, expected, attention_path, key_path, value_path in in_place_checks:
    for name, path in (("present_key", key_path), ("present_value", value_path)):
        words = unpitched(read_words(path), past + 1, PITCH)
        assert words == float_words(expected[name].ravel().tolist()), (past, name)
    attention = [as_float(word) for word in read_words(attention_path)]
    for got, want in zip(attention, expected["attention"].ravel().tolist()):
        worst = max(worst, abs(got - want))
        assert abs(got - want) <= 1e-5 + 1e-5 * abs(want), (past, got, want)
print(f"in-place runs={len(in_place_checks)}: present exact, attention within {worst:.3g}")

# At the bound the bounded library computes exactly what the fixed one does.
dense_bound = next(check for check in checks if check[0]["past"] == BOUND and check[1] == BOUND + 1)
for label, bounded_path, fixed_path in zip(("attention", "present_key", "present_value"), dense_bound[2:], fixed_paths):
    assert read_words(bounded_path) == read_words(fixed_path), label
print(f"bounded_kv=passed cases={len(checks)} bit_identical_at_bound=1 worst_attention_error={worst:.3g}")
