"""Generate execution-plan-free UdonCL Host C directly from an ONNX model."""

import argparse
import math
from pathlib import Path
import struct

from udon_onnx import parse_model


def _c_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("ONNX initializer contains a non-finite float")
    return repr(float(value)) + "f"


def _float_initializer(tensor: dict) -> list[float]:
    if tensor["data_type_name"] != "float32" or tensor["data_location"] == 1:
        raise ValueError("MNIST Dense requires embedded float32 initializers")
    return list(struct.unpack(f"<{len(tensor['raw']) // 4}f", tensor["raw"]))


def _array(lines: list[str], declaration: str, values: list[float]) -> None:
    lines.append(declaration + " = {")
    for offset in range(0, len(values), 6):
        lines.append("    " + ", ".join(_c_float(value) for value in values[offset : offset + 6]) + ",")
    lines.append("};")


def emit_mnist_dense_host(model: dict) -> str:
    """Lower the trained MNIST Dense tail to the public UdonBLAS SGEMM ABI."""
    nodes = model["graph"]["nodes"]
    dense_index = next(
        (index for index, node in enumerate(nodes) if node["op_type"] == "Gemm"),
        None,
    )
    if dense_index is None or dense_index + 1 >= len(nodes):
        raise ValueError("MNIST Dense profile requires Gemm followed by Add")
    dense = nodes[dense_index]
    bias_add = nodes[dense_index + 1]
    if (
        bias_add["op_type"] != "Add"
        or dense["outputs"][0] not in bias_add["inputs"]
        or len(dense["inputs"]) != 2
        or len(bias_add["inputs"]) != 2
    ):
        raise ValueError("MNIST Dense profile requires Gemm output plus one bias")
    _, weight_name = dense["inputs"]
    bias_name = next(name for name in bias_add["inputs"] if name != dense["outputs"][0])
    initializers = model["graph"]["initializers"]
    if initializers[weight_name]["shape"] != [144, 10]:
        raise ValueError("MNIST Dense profile requires [1,144] x [144,10]")
    weight = _float_initializer(initializers[weight_name])
    bias = _float_initializer(initializers[bias_name])
    if len(weight) != 1440 or len(bias) != 10:
        raise ValueError("MNIST Dense initializer size mismatch")

    # ONNX stores [K,N] row-major. cuBLAS consumes the same logical matrix as
    # [K,N] column-major, so pack one contiguous K column per output feature.
    packed_weight = [weight[feature * 10 + column] for column in range(10) for feature in range(144)]
    lines = [
        "/* Generated from ONNX by udon-onnx-host; no execution plan is used. */",
        "#include <udonblas.h>",
        "",
        "static float mnist_dense_input[144];",
    ]
    _array(lines, "static float mnist_dense_weight[1440]", packed_weight)
    lines.extend(
        [
            "static float mnist_dense_output[10];",
            "",
            "int write_mnist_dense_input(int index, float value) {",
            "  mnist_dense_input[index] = value;",
            "  return UDONBLAS_STATUS_SUCCESS;",
            "}",
            "",
            "int run_mnist_dense(void) {",
            "  udonblasHandle_t handle = 0;",
            "  const float alpha = 1.0f;",
            "  const float beta = 1.0f;",
            "  udonblasStatus_t status;",
            *[
                f"  mnist_dense_output[{column}] = {_c_float(bias[column])};"
                for column in range(10)
            ],
            "  status = udonblasCreate(&handle);",
            "  if (status != UDONBLAS_STATUS_SUCCESS)",
            "    return status;",
            "  status = udonblasSgemm(handle, UDONBLAS_OP_N, UDONBLAS_OP_N, 1, 10, 144,",
            "                         &alpha, mnist_dense_input, 1, mnist_dense_weight, 144,",
            "                         &beta, mnist_dense_output, 1);",
            "  if (status != UDONBLAS_STATUS_SUCCESS)",
            "    return status;",
            "  return udonblasDestroy(handle);",
            "}",
            "",
            "float read_mnist_dense_output(int index) {",
            "  return mnist_dense_output[index];",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def emit_mnist_host(model: dict) -> str:
    """Lower the complete trained MNIST graph to three OpenCL AOT kernels."""
    nodes = model["graph"]["nodes"]
    if [node["op_type"] for node in nodes] != [
        "Conv",
        "Add",
        "Relu",
        "AveragePool",
        "Flatten",
        "Gemm",
        "Add",
    ]:
        raise ValueError("MNIST profile requires Conv/Add/Relu/Pool/Flatten/Gemm/Add")
    initializers = model["graph"]["initializers"]
    required = {
        "conv.weight": [16, 1, 5, 5],
        "conv.bias": [1, 16, 1, 1],
        "dense.weight": [144, 10],
        "dense.bias": [10],
    }
    for name, shape in required.items():
        if name not in initializers or initializers[name]["shape"] != shape:
            raise ValueError(f"MNIST initializer {name!r} must have shape {shape}")
    conv_weight = _float_initializer(initializers["conv.weight"])
    conv_bias = _float_initializer(initializers["conv.bias"])
    dense_weight = _float_initializer(initializers["dense.weight"])
    dense_bias = _float_initializer(initializers["dense.bias"])
    packed = [0.0] * 10800
    packed[784:1184] = conv_weight
    packed[1184:1200] = conv_bias
    packed[1200:2640] = [
        dense_weight[feature * 10 + digit]
        for digit in range(10)
        for feature in range(144)
    ]
    packed[2640:2650] = dense_bias

    lines = [
        "/* Generated from ONNX by udon-onnx-host; no execution plan is used. */",
        "#include <CL/cl.h>",
        "",
    ]
    _array(lines, "static float mnist_packed[10800]", packed)
    lines.extend(
        [
            "static float mnist_logits[10];",
            "",
            "int write_mnist_input(int index, float value) {",
            "  if (index < 0 || index >= 784)",
            "    return CL_INVALID_VALUE;",
            "  mnist_packed[index] = value;",
            "  return CL_SUCCESS;",
            "}",
            "",
            "int run_mnist(void) {",
            "  cl_platform_id platform = 0;",
            "  cl_device_id device = 0;",
            "  cl_context context = 0;",
            "  cl_command_queue queue = 0;",
            "  cl_program program = 0;",
            "  cl_kernel conv = 0, pool = 0, dense = 0;",
            "  cl_mem first = 0, second = 0;",
            "  const char *source = \"udoncl-aot\";",
            "  const size_t global = 10800, local = 1;",
            "  cl_int status = CL_SUCCESS;",
            "  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||",
            "      clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, 0) != CL_SUCCESS)",
            "    return CL_DEVICE_NOT_FOUND;",
            "  context = clCreateContext(0, 1, &device, 0, 0, &status);",
            "  if (!context || status != CL_SUCCESS)",
            "    return status;",
            "  queue = clCreateCommandQueue(context, device, 0, &status);",
            "  first = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(mnist_packed), 0, &status);",
            "  second = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(mnist_packed), 0, &status);",
            "  program = clCreateProgramWithSource(context, 1, &source, 0, &status);",
            "  if (!queue || !first || !second || !program || status != CL_SUCCESS ||",
            "      clBuildProgram(program, 0, 0, \"-cl-std=CL1.2\", 0, 0) != CL_SUCCESS)",
            "    return CL_OUT_OF_RESOURCES;",
            "  conv = clCreateKernel(program, \"mnist_conv\", &status);",
            "  pool = clCreateKernel(program, \"mnist_pool\", &status);",
            "  dense = clCreateKernel(program, \"mnist_dense\", &status);",
            "  if (!conv || !pool || !dense || status != CL_SUCCESS)",
            "    return status;",
            "#define SET_BUFFERS(kernel, input, output) \\",
            "  do { \\",
            "    if (clSetKernelArg(kernel, 0, sizeof(input), &(input)) != CL_SUCCESS || \\",
            "        clSetKernelArg(kernel, 1, sizeof(output), &(output)) != CL_SUCCESS) \\",
            "      return CL_INVALID_KERNEL_ARGS; \\",
            "  } while (0)",
            "  SET_BUFFERS(conv, first, second);",
            "  SET_BUFFERS(pool, second, first);",
            "  SET_BUFFERS(dense, first, second);",
            "#undef SET_BUFFERS",
            "  if (clEnqueueWriteBuffer(queue, first, CL_TRUE, 0, sizeof(mnist_packed),",
            "                           mnist_packed, 0, 0, 0) != CL_SUCCESS ||",
            "      clEnqueueNDRangeKernel(queue, conv, 1, 0, &global, &local, 0, 0, 0) != CL_SUCCESS ||",
            "      clEnqueueNDRangeKernel(queue, pool, 1, 0, &global, &local, 0, 0, 0) != CL_SUCCESS ||",
            "      clEnqueueNDRangeKernel(queue, dense, 1, 0, &global, &local, 0, 0, 0) != CL_SUCCESS ||",
            "      clEnqueueReadBuffer(queue, second, CL_TRUE, 0, sizeof(mnist_logits),",
            "                          mnist_logits, 0, 0, 0) != CL_SUCCESS ||",
            "      clFinish(queue) != CL_SUCCESS)",
            "    return CL_OUT_OF_RESOURCES;",
            "  clReleaseKernel(dense);",
            "  clReleaseKernel(pool);",
            "  clReleaseKernel(conv);",
            "  clReleaseProgram(program);",
            "  clReleaseMemObject(second);",
            "  clReleaseMemObject(first);",
            "  clReleaseCommandQueue(queue);",
            "  clReleaseContext(context);",
            "  return CL_SUCCESS;",
            "}",
            "",
            "float read_mnist_output(int index) {",
            "  return mnist_logits[index];",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("mnist", "mnist-dense"), required=True)
    args = parser.parse_args()
    model = parse_model(args.model.read_bytes())
    source = emit_mnist_host(model) if args.profile == "mnist" else emit_mnist_dense_host(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")
    print(f"udon_onnx_host_profile={args.profile} bytes={len(source.encode())}")


if __name__ == "__main__":
    main()
