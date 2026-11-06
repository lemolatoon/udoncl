#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys


opt = Path(sys.argv[1])
host_opt = Path(sys.argv[2])
source = Path(sys.argv[3])
result = subprocess.run(
    [
        opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-lower-onnx-to-host",
        "--one-shot-bufferize=bufferize-function-boundaries copy-before-write",
        source,
    ],
    text=True,
    capture_output=True,
)
assert result.returncode == 0, (result.stdout, result.stderr)
assert "onnx.Add" not in result.stdout
assert "onnx.Return" not in result.stdout
assert 'udon.onnx.provider = "udonatenAdd"' in result.stdout
assert "call @__udon_onnx_add_0" in result.stdout
assert "return %" in result.stdout
assert "tensor<" not in result.stdout
assert "memref<1x4xf32" in result.stdout
assert 'udon.onnx.provider = "udononnxMatMulNBits"' in result.stdout
assert "memref<1x4x16xf32" in result.stdout
assert 'udon.onnx.provider = "udonatenRmsNorm"' in result.stdout
assert "memref<1x4x8xf32" in result.stdout
portable = subprocess.run(
    [opt, "--udoncl-infer-onnx-shapes", "--udoncl-lower-onnx-to-host", source],
    text=True,
    capture_output=True,
    check=True,
).stdout
host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=portable,
    text=True,
    capture_output=True,
)
assert host.returncode == 0, (host.stdout, host.stderr)
assert "func.func @udon_onnx_session_write_input_float" not in host.stdout
assert "func.func @udon_onnx_session_run" not in host.stdout
assert "func.func @udon_onnx_session_read_output_float" not in host.stdout
assert "func.func @udononnxGeneratedEnqueueInitialize" in host.stdout
assert "func.func @udononnxGeneratedEnqueueRun" in host.stdout
assert "func.func @udononnxGeneratedDispose" in host.stdout
assert "func.func @udononnxGeneratedGetModelCount" in host.stdout
assert "func.func @udononnxGeneratedGetModel" in host.stdout
assert "call @udononnxBindingInitializeTensorMetadata" in host.stdout
assert "call @udononnxBindingInitializeModel" in host.stdout
assert "func.func @udon_onnx_binding_create" in host.stdout
assert "func.func @udon_onnx_binding_bind_tensor" in host.stdout
assert "func.func @udon_onnx_binding_enqueue_initialize" in host.stdout
assert "func.func @udon_onnx_binding_enqueue_run" in host.stdout
assert "func.func @udon_onnx_binding_wait_active_event" not in host.stdout
assert "func.func @udon_onnx_binding_dispose" in host.stdout
assert "call @udononnxSessionResolveInput" in host.stdout
assert "call @udononnxSessionResolveOutput" in host.stdout
assert "call @clEnqueueMarkerWithWaitList" in host.stdout
assert "call @__vrc_enqueue_udondnn_large_op_tensor_device_args" in host.stdout
assert "udoncl.onnx_lowering = \"library-calls-v2\"" in host.stdout
assert "udoncl.onnx_device_graph = true" in host.stdout
assert "udoncl.onnx_device_unsupported = []" in host.stdout
assert "udoncl.onnx_operator_counts" in host.stdout

static_source = source.parents[1] / "fixtures" / "onnx-static-initializer.mlir"
static_portable = subprocess.run(
    [
        opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-lower-onnx-to-host",
        static_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
assert "onnx.Constant" not in static_portable
assert "udoncl.static_tensors" in static_portable
assert 'name = "weight"' in static_portable
assert 'data_type = "float32"' in static_portable
assert "byte_length = 16 : i64" in static_portable
assert 'layout = "row_major_contiguous"' in static_portable
assert "alignment = 4 : i64" in static_portable
assert "byte_strides = array<i64: 8, 4>" in static_portable
assert "ad73b9acd6e4a74b2f5bb5386658ce3bb146cd040a1867646ab3b973fb6632b1" in static_portable
assert "call @__udoncl_static_tensor_0" in static_portable
static_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=static_portable,
    text=True,
    capture_output=True,
)
assert static_host.returncode == 0, (static_host.stdout, static_host.stderr)
assert "udoncl.static_tensors" in static_host.stdout
assert "__udon_onnx_static_1" not in static_host.stdout
assert "call @__vrc_opencl_static_tensor" in static_host.stdout
assert "call @udonclAdoptStaticBuffer" in static_host.stdout
assert "@__udon_onnx_device_buffer_1" in static_host.stdout
assert "func.func @udononnxGeneratedDispose" in static_host.stdout
assert "call @clReleaseMemObject" in static_host.stdout

expand_source = source.with_name("onnx_expand_shape.mlir")
expand_inferred = subprocess.run(
    [opt, "--udoncl-infer-onnx-shapes", expand_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
assert "-> tensor<1x32x1xf32>" in expand_inferred
assert "-> tensor<*xf32>" not in expand_inferred

alias_source = source.with_name("onnx_metadata_alias.mlir")
alias_portable = subprocess.run(
    [
        opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-materialize-onnx-constants",
        "--udoncl-lower-onnx-to-host",
        alias_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
assert "onnx.Range" not in alias_portable
assert "onnx.ConstantOfShape" not in alias_portable
assert "onnx.Greater" not in alias_portable
assert "onnx.Where" not in alias_portable
assert "onnx.Slice" not in alias_portable
assert "onnx.Trilu" not in alias_portable
alias_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=alias_portable,
    text=True,
    capture_output=True,
)
assert alias_host.returncode == 0, (alias_host.stdout, alias_host.stderr)
assert "func.func @udon_onnx_session_run" not in alias_host.stdout
assert "call @clEnqueueCopyBuffer" in alias_host.stdout
assert "call @__vrc_enqueue_udondnn" not in alias_host.stdout

cast_source = source.with_name("onnx_cast.mlir")
cast_portable = subprocess.run(
    [opt, "--udoncl-infer-onnx-shapes", "--udoncl-lower-onnx-to-host", cast_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
cast_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=cast_portable,
    text=True,
    capture_output=True,
)
assert cast_host.returncode == 0, (cast_host.stdout, cast_host.stderr)
assert "call @udononnxCast" in cast_host.stdout
assert "func.func private @udononnxCast" in cast_host.stdout
assert "call @udonatenMul" in cast_host.stdout
assert "call @udononnxExpand" in cast_host.stdout
assert "call @udononnxTranspose" in cast_host.stdout
assert "call @udononnxCos" in cast_host.stdout
assert "call @udononnxSin" in cast_host.stdout
assert "call @udononnxNeg" in cast_host.stdout
assert "call @udonatenSigmoid" in cast_host.stdout
assert "udoncl.onnx_device_graph = false" in cast_host.stdout
assert 'operation = "Cast"' in cast_host.stdout
assert 'provider = "udononnxCast"' in cast_host.stdout

unary_source = source.with_name("onnx_unary_device.mlir")
unary_portable = subprocess.run(
    [opt, "--udoncl-infer-onnx-shapes", "--udoncl-lower-onnx-to-host", unary_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
unary_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=unary_portable,
    text=True,
    capture_output=True,
)
assert unary_host.returncode == 0, (unary_host.stdout, unary_host.stderr)
assert "udoncl.onnx_device_graph = true" in unary_host.stdout
assert "udoncl.onnx_device_unsupported = []" in unary_host.stdout
assert unary_host.stdout.count("call @__vrc_enqueue_udondnn_large_op_tensor_device_args") == 1
assert unary_host.stdout.count("call @__vrc_enqueue_udondnn_large_activation_device_args") == 2
assert unary_host.stdout.count("@__udon_onnx_device_buffer_") == 6
assert unary_host.stdout.count("call @__vrc_enqueue_udononnx_unary_device_args") == 3
assert "call @udononnxCos" not in unary_host.stdout
assert "call @udononnxSin" not in unary_host.stdout
assert "call @udononnxNeg" not in unary_host.stdout

transpose_source = source.with_name("onnx_transpose_device.mlir")
transpose_portable = subprocess.run(
    [
        opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-lower-onnx-to-host",
        transpose_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
transpose_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=transpose_portable,
    text=True,
    capture_output=True,
)
assert transpose_host.returncode == 0, (
    transpose_host.stdout,
    transpose_host.stderr,
)
assert "udoncl.onnx_device_graph = true" in transpose_host.stdout
assert "udoncl.onnx_device_unsupported = []" in transpose_host.stdout
assert (
    transpose_host.stdout.count(
        "call @__vrc_enqueue_udononnx_transpose_device_args"
    )
    == 1
)
assert "call @udononnxTranspose" not in transpose_host.stdout

large_pointwise_source = source.with_name("onnx_large_pointwise.mlir")
large_pointwise_portable = subprocess.run(
    [
        opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-lower-onnx-to-host",
        large_pointwise_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
large_pointwise_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=large_pointwise_portable,
    text=True,
    capture_output=True,
)
assert large_pointwise_host.returncode == 0, (
    large_pointwise_host.stdout,
    large_pointwise_host.stderr,
)
assert "udoncl.onnx_device_graph = true" in large_pointwise_host.stdout
assert "udoncl.onnx_device_unsupported = []" in large_pointwise_host.stdout
assert (
    large_pointwise_host.stdout.count(
        "call @__vrc_enqueue_udondnn_large_op_tensor_device_args"
    )
    == 1
)
assert (
    large_pointwise_host.stdout.count(
        "call @__vrc_enqueue_udondnn_large_activation_device_args"
    )
    == 2
)

q4_source = source.with_name("onnx_matmul_nbits_device.mlir")
q4_portable = subprocess.run(
    [
        opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-lower-onnx-to-host",
        q4_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
assert "onnx.Constant" not in q4_portable
assert "udoncl.static_tensors" in q4_portable
q4_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=q4_portable,
    text=True,
    capture_output=True,
)
assert q4_host.returncode == 0, (q4_host.stdout, q4_host.stderr)
assert "udoncl.onnx_device_graph = true" in q4_host.stdout
assert "udoncl.onnx_device_unsupported = []" in q4_host.stdout
assert (
    q4_host.stdout.count(
        "call @__vrc_enqueue_udononnx_matmul_nbits_q4_device_args"
    )
    == 1
)
assert "call @udononnxMatMulNBits" not in q4_host.stdout
assert "@__udon_onnx_operator_scratch" in q4_host.stdout
assert q4_host.stdout.count("arith.constant 524288 : i32") >= 2
assert "call @clEnqueueCopyBuffer" in q4_host.stdout
assert "arith.constant 128 : i32" in q4_host.stdout

rmsnorm_source = source.with_name("onnx_rmsnorm_pattern.mlir")
rmsnorm_canonical = subprocess.run(
    [opt, "--udoncl-canonicalize-onnx-library-patterns", rmsnorm_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
assert 'function_name = "RmsNorm"' in rmsnorm_canonical
assert 'domain_name = "com.lemolatoon.udoncl"' in rmsnorm_canonical

rmsnorm_device_source = source.with_name("onnx_rmsnorm_device.mlir")
rmsnorm_device_portable = subprocess.run(
    [
        opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-lower-onnx-to-host",
        rmsnorm_device_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
rmsnorm_device_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=rmsnorm_device_portable,
    text=True,
    capture_output=True,
)
assert rmsnorm_device_host.returncode == 0, (
    rmsnorm_device_host.stdout,
    rmsnorm_device_host.stderr,
)
assert "udoncl.onnx_device_graph = true" in rmsnorm_device_host.stdout
assert "udoncl.onnx_device_unsupported = []" in rmsnorm_device_host.stdout
assert (
    rmsnorm_device_host.stdout.count(
        "call @__vrc_enqueue_udonaten_arena_rms_norm_reduce_device_args"
    )
    == 1
)
assert (
    rmsnorm_device_host.stdout.count(
        "call @__vrc_enqueue_udonaten_arena_rms_norm_finalize_device_args"
    )
    == 1
)
assert "call @udonatenRmsNorm" not in rmsnorm_device_host.stdout
assert "@__udon_onnx_operator_scratch" in rmsnorm_device_host.stdout
assert "arith.constant 65536 : i32" in rmsnorm_device_host.stdout

batched_matmul_source = source.with_name("onnx_batched_matmul.mlir")
batched_matmul_portable = subprocess.run(
    [
        opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-lower-onnx-to-host",
        batched_matmul_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
batched_matmul_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=batched_matmul_portable,
    text=True,
    capture_output=True,
)
assert batched_matmul_host.returncode == 0, (
    batched_matmul_host.stdout,
    batched_matmul_host.stderr,
)
assert "udoncl.onnx_device_graph = true" in batched_matmul_host.stdout
assert "udoncl.onnx_device_unsupported = []" in batched_matmul_host.stdout
assert (
    batched_matmul_host.stdout.count(
        "call @__vrc_enqueue_udonblas_arena_sgemm_strided_batched_device_args"
    )
    == 1
)
assert "call @udonblasSgemmStridedBatched" not in batched_matmul_host.stdout

concat_source = source.with_name("onnx_concat.mlir")
concat_portable = subprocess.run(
    [opt, "--udoncl-lower-onnx-to-host", concat_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
concat_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=concat_portable,
    text=True,
    capture_output=True,
)
assert concat_host.returncode == 0, (concat_host.stdout, concat_host.stderr)
assert concat_host.stdout.count("call @udononnxConcat") == 2

slice_source = source.with_name("onnx_slice.mlir")
slice_portable = subprocess.run(
    [opt, "--udoncl-materialize-onnx-constants", "--udoncl-lower-onnx-to-host", slice_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
slice_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=slice_portable,
    text=True,
    capture_output=True,
)
assert slice_host.returncode == 0, (slice_host.stdout, slice_host.stderr)
assert "call @udononnxSlice" in slice_host.stdout

where_source = source.with_name("onnx_equal_where.mlir")
where_portable = subprocess.run(
    [opt, "--udoncl-materialize-onnx-constants", "--udoncl-lower-onnx-to-host", where_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
where_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=where_portable,
    text=True,
    capture_output=True,
)
assert where_host.returncode == 0, (where_host.stdout, where_host.stderr)
assert "call @udononnxEqual" in where_host.stdout
assert "call @udononnxWhere" in where_host.stdout

integer_add_source = source.with_name("onnx_integer_add.mlir")
integer_add_portable = subprocess.run(
    [opt, "--udoncl-lower-onnx-to-host", integer_add_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
integer_add_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=integer_add_portable,
    text=True,
    capture_output=True,
)
assert integer_add_host.returncode == 0, (
    integer_add_host.stdout,
    integer_add_host.stderr,
)
assert "call @udononnxAdd" in integer_add_host.stdout

scatter_source = source.with_name("onnx_scatter_nd.mlir")
scatter_portable = subprocess.run(
    [opt, "--udoncl-materialize-onnx-constants", "--udoncl-lower-onnx-to-host", scatter_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
scatter_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=scatter_portable,
    text=True,
    capture_output=True,
)
assert scatter_host.returncode == 0, (scatter_host.stdout, scatter_host.stderr)
assert "call @udononnxScatterND" in scatter_host.stdout

softmax_source = source.with_name("onnx_softmax.mlir")
softmax_portable = subprocess.run(
    [opt, "--udoncl-lower-onnx-to-host", softmax_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
softmax_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=softmax_portable,
    text=True,
    capture_output=True,
)
assert softmax_host.returncode == 0, (softmax_host.stdout, softmax_host.stderr)
assert "udoncl.onnx_device_graph = true" in softmax_host.stdout
assert "udoncl.onnx_device_unsupported = []" in softmax_host.stdout
assert (
    softmax_host.stdout.count(
        "call @__vrc_enqueue_udondnn_arena_softmax_device_args"
    )
    == 1
)
assert "call @udondnnSoftmaxForward" not in softmax_host.stdout

unsupported = source.with_name("onnx_unsupported.mlir")
result = subprocess.run(
    [opt, "--verify-each=false", "--udoncl-lower-onnx-to-host", unsupported],
    text=True,
    capture_output=True,
)
assert result.returncode != 0, result.stdout
assert "unsupported ONNX operator Log" in result.stderr, result.stderr
print("onnx_cpp_lowering=passed")
