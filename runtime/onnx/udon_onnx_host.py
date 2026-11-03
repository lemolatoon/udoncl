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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("mnist-dense",), required=True)
    args = parser.parse_args()
    model = parse_model(args.model.read_bytes())
    source = emit_mnist_dense_host(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")
    print(f"udon_onnx_host_profile={args.profile} bytes={len(source.encode())}")


if __name__ == "__main__":
    main()
