"""Numerically validate execution-plan-free ONNX Host C generation."""

import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys


root, output, compiler, opt = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    Path(sys.argv[4]),
)
if output.exists():
    shutil.rmtree(output)
output.mkdir(parents=True)
model_path = output / "mnist.onnx"
host_path = output / "host.c"
mlir_path = output / "host.mlir"
full_host_path = output / "full-host.c"
full_mlir_path = output / "full-host.mlir"
harness_path = output / "harness.c"
executable = output / ("onnx-host.exe" if sys.platform == "win32" else "onnx-host")


def run(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    return result.stdout


run([sys.executable, root / "examples/mnist-cnn/generate_onnx.py", "-o", model_path])
run(
    [
        sys.executable,
        root / "runtime/onnx/udon_onnx_host.py",
        model_path,
        "--profile",
        "mnist-dense",
        "-o",
        host_path,
        "--mlir-output",
        mlir_path,
    ]
)
run([opt, mlir_path, "--verify-each", "-o", output / "verified.mlir"])
mlir = mlir_path.read_text(encoding="utf-8")
assert 'udoncl.onnx_lowering = "library-calls-v1"' in mlir
assert 'udoncl.library = "UdonBLAS"' in mlir
assert "func.call @udonblasSgemm" in mlir
assert "!udonhl.ptr<0>" in mlir
assert "udonhl.address_of @mnist_dense_weight" in mlir
assert "memref.global" not in mlir
assert "vrc.public_name = \"RunMnistDense\"" in mlir
run(
    [
        sys.executable,
        root / "runtime/onnx/udon_onnx_host.py",
        model_path,
        "--profile",
        "mnist",
        "-o",
        full_host_path,
        "--mlir-output",
        full_mlir_path,
    ]
)
run([opt, full_mlir_path, "--verify-each", "-o", output / "full-verified.mlir"])
full_mlir = full_mlir_path.read_text(encoding="utf-8")
assert 'udoncl.onnx_profile = "mnist"' in full_mlir
for call in (
    "udondnnConvolutionForward",
    "udondnnOpTensor",
    "udondnnActivationForward",
    "udondnnPoolingForward",
    "udonblasSgemm",
):
    assert f"func.call @{call}" in full_mlir
assert "mnist_conv_output" in full_mlir
assert "!llvm.array<9216 x f32>" in full_mlir
assert "execution-plan" not in full_mlir
source = host_path.read_text(encoding="utf-8")
assert "UDONBLAS_OP_N, UDONBLAS_OP_N, 1, 10, 144" in source
assert "execution plan" in source
assert not list(output.glob("*.json"))

harness_path.write_text(
    r'''#include <udonblas.h>
#include <stdio.h>

int write_mnist_dense_input(int index, float value);
int run_mnist_dense(void);
float read_mnist_dense_output(int index);

udonblasStatus_t udonblasCreate(udonblasHandle_t *handle) {
  *handle = 1;
  return UDONBLAS_STATUS_SUCCESS;
}

udonblasStatus_t udonblasDestroy(udonblasHandle_t handle) {
  return handle == 1 ? UDONBLAS_STATUS_SUCCESS : UDONBLAS_STATUS_INVALID_VALUE;
}

udonblasStatus_t udonblasSgemm(
    udonblasHandle_t handle, udonblasOperation_t transa,
    udonblasOperation_t transb, int m, int n, int k, const float *alpha,
    const float *a, int lda, const float *b, int ldb, const float *beta,
    float *c, int ldc) {
  int row, column, inner;
  if (handle != 1 || transa != UDONBLAS_OP_N || transb != UDONBLAS_OP_N ||
      m != 1 || n != 10 || k != 144 || lda != 1 || ldb != 144 || ldc != 1)
    return UDONBLAS_STATUS_INVALID_VALUE;
  for (column = 0; column < n; ++column)
    for (row = 0; row < m; ++row) {
      float sum = 0.0f;
      for (inner = 0; inner < k; ++inner)
        sum += a[row + inner * lda] * b[inner + column * ldb];
      c[row + column * ldc] = *alpha * sum + *beta * c[row + column * ldc];
    }
  return UDONBLAS_STATUS_SUCCESS;
}

int main(void) {
  int index;
  for (index = 0; index < 144; ++index)
    if (write_mnist_dense_input(index, (float)(index % 16 + 1)) != 0)
      return 1;
  if (run_mnist_dense() != 0)
    return 2;
  for (index = 0; index < 10; ++index)
    printf("%.9g\n", read_mnist_dense_output(index));
  return 0;
}
''',
    encoding="utf-8",
)
run(
    [
        compiler,
        "-std=c11",
        "-I",
        root / "libraries/udonblas/include",
        host_path,
        harness_path,
        "-o",
        executable,
    ]
)
actual = [float(value) for value in run([executable]).splitlines()]

sys.path.insert(0, str(root / "runtime/onnx"))
from udon_onnx import parse_model  # noqa: E402

model = parse_model(model_path.read_bytes())
initializers = model["graph"]["initializers"]
weight = struct.unpack("<1440f", initializers["dense.weight"]["raw"])
bias = struct.unpack("<10f", initializers["dense.bias"]["raw"])
inputs = [float(index % 16 + 1) for index in range(144)]
expected = [
    bias[column]
    + sum(inputs[feature] * weight[feature * 10 + column] for feature in range(144))
    for column in range(10)
]
maximum_error = max(abs(left - right) for left, right in zip(actual, expected))
assert math.isclose(maximum_error, 0.0, abs_tol=2.0e-4), (actual, expected, maximum_error)
print(f"onnx_host=passed outputs=10 maximum_error={maximum_error:.9g}")
