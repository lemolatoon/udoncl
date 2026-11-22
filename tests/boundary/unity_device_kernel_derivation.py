"""Derive the device_kernels of a Unity library from its Host dispatches.

A Unity Host that dispatches UdonCL Device library kernels through the OpenCL
AOT intrinsics needs no hand-written device_kernels entry: udonclc takes the
launch of each kernel from its dispatch sites and the access contract from
the library kernel contract. The derived bundle must equal the one built from
the equivalent hand-written entry, a hand-written entry must still win over
the derived one and keep a kernel the Host does not dispatch, and every case
the compiler cannot derive must say which entry to write by hand.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys

compiler, clang, output = sys.argv[1:]
output = Path(output).resolve()
if output.exists():
    shutil.rmtree(output)
output.mkdir(parents=True)
root = Path(__file__).resolve().parents[2]


def run(command, expected=0):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == expected, (command, result.stdout, result.stderr)
    return result


def device_ir(name, source):
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
            root / source,
            "-o",
            ir,
        ]
    )
    return ir


pointwise = device_ir("large-pointwise", "libraries/udondnn/device/large_pointwise.cl")
sgemm = device_ir("arena-sgemm", "libraries/udonblas/device/arena_sgemm.cl")

WORDS = 131072
GRID = [8192, 16, 1]
OP_TENSOR = {
    "schema_version": 1,
    "profile": "gather-f32",
    "words": WORDS,
    "kernel": "udondnn_large_op_tensor",
    "kernel_id": 234,
    "input_words": [WORDS, WORDS, WORDS],
    "argument_kinds": ["buffer"] * 4 + ["i32"] * 8 + ["f32"] * 3,
    "global_size": GRID,
    "local_size": [1, 1, 1],
    "global_offset": [0, 0, 0],
}
ACTIVATION = {
    "schema_version": 1,
    "profile": "gather-f32",
    "words": 32768,
    "kernel": "udondnn_large_activation_forward",
    "kernel_id": 233,
    "argument_kinds": ["buffer"] * 3 + ["i32"] * 3 + ["f32"] * 3,
    "global_size": [8192, 4, 1],
    "local_size": [1, 1, 1],
    "global_offset": [0, 0, 0],
}
config = {
    "schema_version": 1,
    "program": "derived_device_kernels",
    "exports": [{"symbol": "scale", "public_name": "Scale", "id": 1, "parameters": []}],
    "buffers": [
        {"id": identifier, "name": f"tensor{identifier}", "byte_size": WORDS * 4}
        for identifier in range(1, 5)
    ],
}


def host(dispatches, declarations=""):
    return f"""
func.func private @__vrc_opencl_buffer_alloc(i32, i32) -> i32
{declarations}
func.func @scale() {{
  %bytes = arith.constant {WORDS * 4} : i32
  %id1 = arith.constant 1 : i32
  %id2 = arith.constant 2 : i32
  %id3 = arith.constant 3 : i32
  %id4 = arith.constant 4 : i32
  %a = func.call @__vrc_opencl_buffer_alloc(%id1, %bytes) : (i32, i32) -> i32
  %b = func.call @__vrc_opencl_buffer_alloc(%id2, %bytes) : (i32, i32) -> i32
  %c = func.call @__vrc_opencl_buffer_alloc(%id3, %bytes) : (i32, i32) -> i32
  %x = func.call @__vrc_opencl_buffer_alloc(%id4, %bytes) : (i32, i32) -> i32
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %count = arith.constant 256 : i32
  %alpha = arith.constant 1.0 : f32
  %beta = arith.constant 0.0 : f32
{dispatches}
  return
}}
"""


OP_TENSOR_DECLARATION = (
    "func.func private @__vrc_opencl_dispatch_udondnn_large_op_tensor("
    + ", ".join(["i32"] * 12 + ["f32"] * 3)
    + ") -> i32"
)
OP_TENSOR_CALL = (
    "  %scaled = func.call @__vrc_opencl_dispatch_udondnn_large_op_tensor("
    "%a, %b, %c, %x, %zero, %count, %count, %count, %zero, %zero, %zero, "
    "%one, %alpha, %alpha, %beta) : ("
    + ", ".join(["i32"] * 12 + ["f32"] * 3)
    + ") -> i32"
)
op_tensor_host = host(OP_TENSOR_CALL, OP_TENSOR_DECLARATION)


def build(name, kernels=None, host_source=op_tensor_host, device=pointwise, expected=0):
    variant = dict(config)
    if kernels is not None:
        variant["device_kernels"] = kernels
    config_path = output / f"{name}.json"
    config_path.write_text(json.dumps(variant), encoding="utf-8")
    host_path = output / f"{name}.mlir"
    host_path.write_text(host_source, encoding="utf-8", newline="\n")
    bundle = output / name
    result = run(
        [
            compiler,
            "--emit-unity-library-bundle",
            config_path,
            "--host-mlir",
            host_path,
            *(["--device-llvm", device] if device else []),
            "-o",
            bundle,
        ],
        expected,
    )
    return bundle, result


def manifest(bundle):
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


# A hand-written entry and the derived one build the same bundle.
explicit, _ = build("explicit", [OP_TENSOR])
derived, _ = build("derived")
for path in sorted(explicit.rglob("*")):
    if path.is_file():
        relative = path.relative_to(explicit)
        assert (derived / relative).read_bytes() == path.read_bytes(), relative
assert sorted(p.relative_to(derived) for p in derived.rglob("*")) == sorted(
    p.relative_to(explicit) for p in explicit.rglob("*")
)
kernels = manifest(derived)["device"]["kernels"]
assert [(kernel["id"], kernel["name"]) for kernel in kernels] == [
    (234, "udondnn_large_op_tensor")
], kernels
assert kernels[0]["global_size"] == GRID, kernels[0]

# A hand-written entry is kept even when the Host does not dispatch it, and
# the kernels it does dispatch without an entry are still derived.
mixed, _ = build("mixed", [ACTIVATION])
assert sorted(kernel["id"] for kernel in manifest(mixed)["device"]["kernels"]) == [
    233,
    234,
]
assert (mixed / "kernels/kernel_234.shader").read_bytes() == (
    explicit / "kernels/kernel_234.shader"
).read_bytes()

# A kernel that is no UdonCL library kernel needs its entry.
custom = host(
    "  %custom = func.call @dispatch_custom(%a, %x, %count) : (i32, i32, i32) -> i32",
    "func.func private @dispatch_custom(i32, i32, i32) -> i32 attributes {"
    "vrc.device_dispatch = {kernel_id = 77 : i32, buffer_count = 2 : i32, "
    "global_size = array<i32: 256, 1, 1>}}",
)
_, unknown = build("unknown", host_source=custom, expected=1)
assert (
    "the Unity Host dispatches Device kernel id 77, which is not a UdonCL "
    "Device library kernel; describe it with a device_kernels entry"
) in unknown.stderr, unknown.stderr

# A site outside the library contract the kernel is compiled against cannot
# be derived: arena SGEMM walks at most 8192 K elements.
SGEMM_DECLARATION = (
    "func.func private @__vrc_opencl_dispatch_udonblas_arena_sgemm("
    "i32, i32, i32, i32, i32, i32, i32, i32, f32) -> i32"
)


def sgemm_call(k):
    return (
        f"  %k = arith.constant {k} : i32\n"
        "  %product = func.call @__vrc_opencl_dispatch_udonblas_arena_sgemm("
        "%a, %b, %x, %one, %one, %k, %zero, %zero, %alpha) : "
        "(i32, i32, i32, i32, i32, i32, i32, i32, f32) -> i32"
    )


sgemm_ok, _ = build(
    "sgemm", host_source=host(sgemm_call(8192), SGEMM_DECLARATION), device=sgemm
)
(sgemm_kernel,) = manifest(sgemm_ok)["device"]["kernels"]
assert sgemm_kernel["name"] == "udonblas_arena_sgemm", sgemm_kernel
assert sgemm_kernel["args"][5]["maximum"] == 8192, sgemm_kernel["args"][5]
_, outside = build(
    "sgemm-outside",
    host_source=host(sgemm_call(8193), SGEMM_DECLARATION),
    device=sgemm,
    expected=1,
)
assert (
    "a dispatch of Device kernel udonblas_arena_sgemm (id 221) passes 8193 as "
    "argument 5, outside its library contract [1, 8192]; declare its "
    "device_kernels entry by hand"
) in outside.stderr, outside.stderr
# The hand-written entry is the way out: it replaces the contract.
build(
    "sgemm-explicit",
    [
        {
            "schema_version": 1,
            "profile": "gather-f32",
            "words": 16384,
            "kernel": "udonblas_arena_sgemm",
            "kernel_id": 221,
            "argument_kinds": ["buffer"] * 3 + ["i32"] * 5 + ["f32"],
            "global_size": [8192, 2, 1],
            "local_size": [1, 1, 1],
            "global_offset": [0, 0, 0],
        }
    ],
    host_source=host(sgemm_call(8193), SGEMM_DECLARATION),
    device=sgemm,
)

# Derived library kernels need their Device library.
_, missing = build("missing-device", device=None, expected=1)
assert (
    "the Unity Host dispatches the Device library kernels "
    "udondnn_large_op_tensor but no Device library is linked"
) in missing.stderr, missing.stderr

print("unity_device_kernel_derivation=ok")
