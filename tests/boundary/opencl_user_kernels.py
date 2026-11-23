"""Declare the user OpenCL C kernels a Host C launches through the runtime.

`udonclc --emit-opencl-user-kernels host --device-llvm device -o dir` writes
the two headers the OpenCL AOT runtime is compiled with: every kernel the
Host creates by a constant name that the Device libraries define and no UdonCL
library provides gets an id from 1024 in name order and an enqueue entry
point. A Host that creates only library kernels gets an empty header, which
leaves the runtime unchanged, and a kernel whose arguments Udon cannot bind
fails with its name.
"""

from pathlib import Path
import shutil
import subprocess
import sys

compiler, clang, output = sys.argv[1:]
output = Path(output).resolve()
if output.exists():
    shutil.rmtree(output)
output.mkdir(parents=True)


def run(command, expected=0):
    result = subprocess.run(
        [str(value) for value in command], text=True, capture_output=True,
        timeout=300,
    )
    assert result.returncode == expected, (command, result.stdout, result.stderr)
    return result


def compile_source(name, language, text):
    source = output / f"{name}.{'c' if language == 'c' else 'cl'}"
    source.write_text(text, encoding="utf-8", newline="\n")
    module = output / f"{name}.ll"
    flags = (
        ["-x", "c", "-std=c11", "-O1", "-fno-inline",
         "-fno-discard-value-names"]
        if language == "c"
        else ["-x", "cl", "-cl-std=CL1.2", "-cl-kernel-arg-info", "-O0",
              "-Xclang", "-disable-O0-optnone"]
    )
    run([clang, "-target", "spir-unknown-unknown", *flags, "-S", "-emit-llvm",
         source, "-o", module])
    return module


def host(*names):
    calls = "\n".join(
        f'  kernels[{index}] = clCreateKernel(program, "{name}", 0);'
        for index, name in enumerate(names)
    )
    return compile_source(
        "host_" + "_".join(names),
        "c",
        "typedef struct _cl_program *cl_program;\n"
        "typedef struct _cl_kernel *cl_kernel;\n"
        "cl_kernel clCreateKernel(cl_program, const char *, int *);\n"
        f"cl_kernel kernels[{len(names)}];\n"
        "void create(cl_program program) {\n" + calls + "\n}\n",
    )


device = compile_source(
    "device",
    "cl",
    "__kernel void user_copy(__global const float *input,\n"
    "                        __global float *output, int n, float scale) {\n"
    "  uint gid = (uint)get_global_id(0);\n"
    "  output[gid] = gid < (uint)n ? scale * input[gid] : 0.0f;\n"
    "}\n"
    "__kernel void user_fill(__global float *output, float value) {\n"
    "  output[get_global_id(0)] = value;\n"
    "}\n"
    "__kernel void user_scalar_first(float value, __global float *output) {\n"
    "  output[get_global_id(0)] = value;\n"
    "}\n"
    "__kernel void udonblas_sgemm(__global const float *input,\n"
    "                             __global float *output) {\n"
    "  output[get_global_id(0)] = input[get_global_id(0)];\n"
    "}\n",
)


def declare(name, host_module, expected=0):
    directory = output / name
    result = run(
        [compiler, "--emit-opencl-user-kernels", host_module, "--device-llvm",
         device, "-o", directory],
        expected,
    )
    return directory, result


# User kernels, in name order from 1024; the library kernel and a name no
# Device library defines stay with the runtime.
directory, _ = declare(
    "users", host("user_fill", "udonblas_sgemm", "missing", "user_copy")
)
declarations = (directory / "udoncl_aot_user_kernels.h").read_text(encoding="utf-8")
dispatch = (directory / "udoncl_aot_user_dispatch.h").read_text(encoding="utf-8")
assert "#define UDONCL_AOT_USER_KERNEL_COUNT 2" in declarations, declarations
# The Device libraries define a UdonBLAS kernel, so the BLAS dispatch path
# stays; the other libraries' paths are left out.
assert "UDONCL_AOT_ENABLE_BLAS" not in declarations, declarations
for disabled in ["DNN", "ONNX"]:
    assert f"#define UDONCL_AOT_ENABLE_{disabled} 0" in declarations, declarations
for expected in [
    'if (udoncl_string_equal(name, "user_copy"))\n    return 1024;',
    'if (udoncl_string_equal(name, "user_fill"))\n    return 1025;',
    "extern cl_int __vrc_opencl_dispatch_user_1024_2_user_copy(cl_uint, "
    "cl_uint, cl_int, cl_float);",
    "extern cl_int __vrc_opencl_dispatch_user_1025_1_user_fill(cl_uint, "
    "cl_float);",
    "cl_int __vrc_enqueue_user_1024_2_user_copy(cl_command_queue queue, "
    "cl_mem buffer0, cl_mem buffer1, cl_int scalar2, cl_float scalar3, "
    "cl_uint wait_count, const cl_event *wait_list, cl_event *event) {",
    "UDONCL_RETURN_IF_READ_WRITE_ALIAS(buffer1, buffer0);",
    "if (udoncl_prepare_resources(buffer0, 0, 0) != CL_SUCCESS)",
]:
    assert expected in dispatch, (expected, dispatch)
assert "udonblas_sgemm" not in dispatch and "missing" not in dispatch, dispatch

# Only library kernels: an empty declaration header.
directory, _ = declare("library", host("udonblas_sgemm"))
declarations = (directory / "udoncl_aot_user_kernels.h").read_text(encoding="utf-8")
assert "#define" not in declarations, declarations

# A scalar before a buffer cannot be bound to a Udon dispatch.
_, failure = declare("scalar-first", host("user_scalar_first"), expected=1)
assert (
    "OpenCL kernel user_scalar_first argument 1 cannot be dispatched on Udon: "
    "a kernel takes its __global buffers (inputs, then the output) followed "
    "by int, uint or float scalars"
) in failure.stderr, failure.stderr

print("opencl_user_kernels=ok")
