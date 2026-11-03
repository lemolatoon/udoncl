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


def _mlir_i8_array(data: bytes) -> str:
    signed = (value if value < 128 else value - 256 for value in data)
    return "array<i8: " + ", ".join(str(value) for value in signed) + ">"


def _udonhl_global(name: str, storage_type: str, data: bytes) -> str:
    return (
        '  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, '
        f"initial_value = {_mlir_i8_array(data)}, linkage = \"internal\", "
        f"storage_type = {storage_type}, sym_name = \"{name}\"}}> : () -> ()"
    )


def emit_mnist_dense_mlir(model: dict) -> str:
    """Lower the MNIST Dense tail to UdonHL and the public UdonBLAS ABI."""
    nodes = model["graph"]["nodes"]
    dense_index = next(
        (index for index, node in enumerate(nodes) if node["op_type"] == "Gemm"),
        None,
    )
    if dense_index is None or dense_index + 1 >= len(nodes):
        raise ValueError("MNIST Dense MLIR requires Gemm followed by Add")
    dense = nodes[dense_index]
    bias_add = nodes[dense_index + 1]
    if (
        bias_add["op_type"] != "Add"
        or dense["outputs"][0] not in bias_add["inputs"]
        or len(dense["inputs"]) != 2
        or len(bias_add["inputs"]) != 2
    ):
        raise ValueError("MNIST Dense MLIR requires Gemm output plus one bias")
    _, weight_name = dense["inputs"]
    bias_name = next(name for name in bias_add["inputs"] if name != dense["outputs"][0])
    initializers = model["graph"]["initializers"]
    if initializers[weight_name]["shape"] != [144, 10]:
        raise ValueError("MNIST Dense MLIR requires [1,144] x [144,10]")
    weight = _float_initializer(initializers[weight_name])
    bias = _float_initializer(initializers[bias_name])
    if len(weight) != 1440 or len(bias) != 10:
        raise ValueError("MNIST Dense MLIR initializer size mismatch")
    packed_weight = [
        weight[feature * 10 + column]
        for column in range(10)
        for feature in range(144)
    ]

    weight_bytes = struct.pack(f"<{len(packed_weight)}f", *packed_weight)
    lines = [
        'module attributes {udoncl.onnx_lowering = "library-calls-v1", '
        'udoncl.onnx_profile = "mnist-dense"} {',
        _udonhl_global("mnist_dense_input", "!llvm.array<144 x f32>", bytes(144 * 4)),
        _udonhl_global("mnist_dense_weight", "!llvm.array<1440 x f32>", weight_bytes),
        _udonhl_global("mnist_dense_output", "!llvm.array<10 x f32>", bytes(10 * 4)),
        '',
        '  func.func private @udonblasCreate(!udonhl.ptr<0>) -> i32 attributes '
        '{udoncl.api = "udonblasCreate", udoncl.library = "UdonBLAS"}',
        '  func.func private @udonblasDestroy(i32) -> i32 attributes '
        '{udoncl.api = "udonblasDestroy", udoncl.library = "UdonBLAS"}',
        '  func.func private @udonblasSgemm(i32, i32, i32, i32, i32, i32, '
        '!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, '
        '!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> i32 attributes '
        '{udoncl.api = "udonblasSgemm", udoncl.library = "UdonBLAS"}',
        '',
        '  func.func @write_mnist_dense_input(%index: i32, %value: f32) -> i32 attributes {',
        '    vrc.export_id = 1 : i32,',
        '    vrc.parameter_names = ["index", "value"],',
        '    vrc.public_name = "WriteMnistDenseInput"',
        '  } {',
        '    %four = arith.constant 4 : i32',
        '    %input = udonhl.address_of @mnist_dense_input : <0>',
        '    %offset = arith.muli %index, %four : i32',
        '    %position = "udonhl.ptr_add"(%input, %offset) : '
        '(!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>',
        '    "udonhl.store"(%value, %position) <{alignment = 4 : i64}> : '
        '(f32, !udonhl.ptr<0>) -> ()',
        '    %success = arith.constant 0 : i32',
        '    return %success : i32',
        '  }',
        '',
        '  func.func @run_mnist_dense() -> i32 attributes {',
        '    vrc.async_effect = "may_suspend",',
        '    vrc.export_id = 2 : i32,',
        '    vrc.public_name = "RunMnistDense"',
        '  } {',
        '    %four = arith.constant 4 : i32',
        '    %input = udonhl.address_of @mnist_dense_input : <0>',
        '    %weight = udonhl.address_of @mnist_dense_weight : <0>',
        '    %output = udonhl.address_of @mnist_dense_output : <0>',
        '    %handle_slot = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : '
        '(i32) -> !udonhl.ptr<0>',
        '    %alpha = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : '
        '(i32) -> !udonhl.ptr<0>',
        '    %beta = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : '
        '(i32) -> !udonhl.ptr<0>',
        '    %one = arith.constant 1.0 : f32',
        '    "udonhl.store"(%one, %alpha) <{alignment = 4 : i64}> : '
        '(f32, !udonhl.ptr<0>) -> ()',
        '    "udonhl.store"(%one, %beta) <{alignment = 4 : i64}> : '
        '(f32, !udonhl.ptr<0>) -> ()',
    ]
    for index in range(10):
        lines.extend(
            [
                f'    %offset{index} = arith.constant {index * 4} : i32',
                f'    %output{index} = "udonhl.ptr_add"(%output, %offset{index}) : '
                '(!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>',
                f'    %bias{index} = arith.constant {_c_float(bias[index])[:-1]} : f32',
                f'    "udonhl.store"(%bias{index}, %output{index}) '
                '<{alignment = 4 : i64}> : (f32, !udonhl.ptr<0>) -> ()',
            ]
        )
    lines.extend(
        [
            '    %success = arith.constant 0 : i32',
            '    %op_n = arith.constant 0 : i32',
            '    %m = arith.constant 1 : i32',
            '    %n = arith.constant 10 : i32',
            '    %k = arith.constant 144 : i32',
            '    %lda = arith.constant 1 : i32',
            '    %ldb = arith.constant 144 : i32',
            '    %ldc = arith.constant 1 : i32',
            '    %create_status = func.call @udonblasCreate(%handle_slot) : '
            '(!udonhl.ptr<0>) -> i32',
            '    %created = arith.cmpi eq, %create_status, %success : i32',
            '    cf.cond_br %created, ^sgemm, ^return(%create_status : i32)',
            '  ^sgemm:',
            '    %handle = "udonhl.load"(%handle_slot) <{alignment = 4 : i64}> : '
            '(!udonhl.ptr<0>) -> i32',
            '    %sgemm_status = func.call @udonblasSgemm('
            '%handle, %op_n, %op_n, %m, %n, %k, %alpha, %input, %lda, '
            '%weight, %ldb, %beta, %output, %ldc) : '
            '(i32, i32, i32, i32, i32, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, '
            'i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> i32',
            '    %computed = arith.cmpi eq, %sgemm_status, %success : i32',
            '    cf.cond_br %computed, ^destroy, ^return(%sgemm_status : i32)',
            '  ^destroy:',
            '    %destroy_status = func.call @udonblasDestroy(%handle) : (i32) -> i32',
            '    cf.br ^return(%destroy_status : i32)',
            '  ^return(%status: i32):',
            '    return %status : i32',
            '  }',
            '',
            '  func.func @read_mnist_dense_output(%index: i32) -> f32 attributes {',
            '    vrc.export_id = 3 : i32,',
            '    vrc.parameter_names = ["index"],',
            '    vrc.public_name = "ReadMnistDenseOutput"',
            '  } {',
            '    %four = arith.constant 4 : i32',
            '    %output = udonhl.address_of @mnist_dense_output : <0>',
            '    %offset = arith.muli %index, %four : i32',
            '    %position = "udonhl.ptr_add"(%output, %offset) : '
            '(!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>',
            '    %value = "udonhl.load"(%position) <{alignment = 4 : i64}> : '
            '(!udonhl.ptr<0>) -> f32',
            '    return %value : f32',
            '  }',
            '}',
            '',
        ]
    )
    return "\n".join(lines)


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
    parser.add_argument("--mlir-output", type=Path)
    parser.add_argument("--profile", choices=("mnist", "mnist-dense"), required=True)
    args = parser.parse_args()
    model = parse_model(args.model.read_bytes())
    source = emit_mnist_host(model) if args.profile == "mnist" else emit_mnist_dense_host(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")
    if args.mlir_output is not None:
        if args.profile != "mnist-dense":
            raise ValueError("MLIR output currently supports the mnist-dense profile")
        mlir = emit_mnist_dense_mlir(model)
        args.mlir_output.parent.mkdir(parents=True, exist_ok=True)
        args.mlir_output.write_text(mlir, encoding="utf-8")
    print(f"udon_onnx_host_profile={args.profile} bytes={len(source.encode())}")


if __name__ == "__main__":
    main()
