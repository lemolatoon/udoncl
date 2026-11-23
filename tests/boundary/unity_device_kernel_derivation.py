"""Derive the device_kernels of a Unity library from its Host dispatches.

A Unity Host that dispatches UdonCL Device library kernels through the OpenCL
AOT intrinsics describes them by its dispatch sites alone: udonclc takes the
launch of each kernel from its dispatch sites and the access contract from
the library kernel contract. Every case the compiler cannot derive is an
error that says why.
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


def host(dispatches, declarations=""):
    # The Host names its program and export in the module and its four
    # Device buffers are compiler-assigned textures, whose OpenCL tokens the
    # dispatches take; no config lists them.
    buffers = "".join(
        f'  %{name}_buffer = "vrc.assigned_buffer"() <{{capacity_bytes = '
        f"{WORDS * 4} : i64, format = \"rgba8_unorm\", resource_id = {index} : "
        f"i32, texture_id = {index + 1} : i32}}> : () -> !vrc.buffer\n"
        f"  %{name} = builtin.unrealized_conversion_cast %{name}_buffer : "
        "!vrc.buffer to i32\n"
        for index, name in enumerate(["a", "b", "c", "x"])
    )
    return f"""
module attributes {{udoncl.program = "derived_device_kernels"}} {{
{declarations}
func.func @scale() attributes {{udoncl.export}} {{
{buffers}  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %count = arith.constant 256 : i32
  %alpha = arith.constant 1.0 : f32
  %beta = arith.constant 0.0 : f32
{dispatches}
  return
}}
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


def build(name, host_source=op_tensor_host, device=pointwise, expected=0):
    host_path = output / f"{name}.mlir"
    host_path.write_text(host_source, encoding="utf-8", newline="\n")
    bundle = output / name
    result = run(
        [
            compiler,
            "--emit-unity-library-bundle",
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


# The OpTensor kernel is derived from its dispatch site.
derived, _ = build("derived")
kernels = manifest(derived)["device"]["kernels"]
assert [(kernel["id"], kernel["name"]) for kernel in kernels] == [
    (234, "udondnn_large_op_tensor")
], kernels
assert kernels[0]["global_size"] == GRID, kernels[0]
assert (derived / "kernels/kernel_234.shader").is_file()

# A kernel that is no UdonCL library kernel cannot be derived.
custom = host(
    "  %custom = func.call @dispatch_custom(%a, %x, %count) : (i32, i32, i32) -> i32",
    "func.func private @dispatch_custom(i32, i32, i32) -> i32 attributes {"
    "vrc.device_dispatch = {kernel_id = 77 : i32, buffer_count = 2 : i32, "
    "global_size = array<i32: 256, 1, 1>}}",
)
_, unknown = build("unknown", host_source=custom, expected=1)
assert (
    "the Unity Host dispatches Device kernel id 77, which is neither a "
    "UdonCL Device library kernel nor an OpenCL C kernel of the Host's Device "
    "libraries"
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
    "argument 5, outside its library contract [1, 8192]"
) in outside.stderr, outside.stderr
# Derived library kernels need their Device library.
_, missing = build("missing-device", device=None, expected=1)
assert (
    "the Unity Host dispatches the Device library kernels "
    "udondnn_large_op_tensor but no Device library is linked"
) in missing.stderr, missing.stderr

print("unity_device_kernel_derivation=ok")
