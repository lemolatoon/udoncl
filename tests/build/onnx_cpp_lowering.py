#!/usr/bin/env python3

from pathlib import Path
import re
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
assert "func.func @udon_onnx_binding_bind_all" in host.stdout
assert "func.func @udon_onnx_binding_enqueue_initialize" in host.stdout
assert "func.func @udon_onnx_binding_enqueue_run" in host.stdout
assert "func.func @udon_onnx_binding_wait_active_event" not in host.stdout
assert "func.func @udon_onnx_binding_dispose" in host.stdout
assert "call @udononnxSessionResolveInput" in host.stdout
assert "call @udononnxSessionResolveOutput" in host.stdout
assert "call @clEnqueueMarkerWithWaitList" in host.stdout
assert "call @__vrc_opencl_dispatch_udondnn_large_op_tensor" in host.stdout
assert "udoncl.onnx_lowering = \"library-calls-v2\"" in host.stdout
assert "udoncl.onnx_device_graph = true" in host.stdout
assert "udoncl.onnx_device_unsupported = []" in host.stdout
assert "udoncl.onnx_operator_counts" in host.stdout
assert "logical_bytes" in host.stdout
assert "capacity_bytes" in host.stdout
assert "udoncl.mutable_handle_capacity" in host.stdout

sequence_state_source = source.with_name("onnx_sequence_state.mlir")
sequence_state = subprocess.run(
    [opt, "--udoncl-identify-onnx-sequence-state", sequence_state_source],
    text=True,
    capture_output=True,
)
assert sequence_state.returncode == 0, (
    sequence_state.stdout,
    sequence_state.stderr,
)
assert "udoncl.onnx_sequence_states" in sequence_state.stdout
assert "input_index = 1 : i64" in sequence_state.stdout
assert "output_index = 0 : i64" in sequence_state.stdout
assert "input_index = 2 : i64" in sequence_state.stdout
assert "output_index = 2 : i64" in sequence_state.stdout
assert sequence_state.stdout.count("axis = 2 : i64") == 2
assert sequence_state.stdout.count("step = 1 : i64") == 2
assert sequence_state.stdout.count("capacity = 8 : i64") == 2
assert "input_index = 3 : i64" not in sequence_state.stdout

sequence_state_after_standard_return = subprocess.run(
    [
        opt,
        "--standard-func-return",
        "--udoncl-identify-onnx-sequence-state",
        sequence_state_source,
    ],
    text=True,
    capture_output=True,
)
assert sequence_state_after_standard_return.returncode == 0, (
    sequence_state_after_standard_return.stdout,
    sequence_state_after_standard_return.stderr,
)
assert "input_index = 1 : i64" in sequence_state_after_standard_return.stdout
assert "output_index = 0 : i64" in sequence_state_after_standard_return.stdout
assert "input_index = 2 : i64" in sequence_state_after_standard_return.stdout
assert "output_index = 2 : i64" in sequence_state_after_standard_return.stdout
assert sequence_state_after_standard_return.stdout.count("axis = 2 : i64") == 2
assert sequence_state_after_standard_return.stdout.count("step = 1 : i64") == 2
assert sequence_state_after_standard_return.stdout.count("capacity = 8 : i64") == 2

sequence_state_portable = subprocess.run(
    [
        opt,
        "--udoncl-identify-onnx-sequence-state",
        "--udoncl-lower-onnx-to-host",
        sequence_state_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
sequence_state_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=sequence_state_portable,
    text=True,
    capture_output=True,
)
assert sequence_state_host.returncode == 0, (
    sequence_state_host.stdout,
    sequence_state_host.stderr,
)
assert "func.func @udon_onnx_binding_enqueue_advance_sequence_state" in (
    sequence_state_host.stdout
)
assert "func.func @udon_onnx_binding_enqueue_reset_sequence_state" in (
    sequence_state_host.stdout
)
reset_body = sequence_state_host.stdout.split(
    "func.func @udon_onnx_binding_enqueue_reset_sequence_state", 1
)[1].split("func.func", 1)[0]
assert reset_body.count("call @__vrc_opencl_buffer_upload") == 2
assert reset_body.count("call @__vrc_opencl_await") == 2
assert "call @__vrc_opencl_buffer_copy" not in reset_body
assert "call @__vrc_opencl_buffer_readback" not in reset_body
assert "arith.constant 7168 : i32" in reset_body
assert "arith.constant 7 : i32" in reset_body
assert "arith.constant 12 : i32" in reset_body
advance_body = sequence_state_host.stdout.split(
    "func.func @udon_onnx_binding_enqueue_advance_sequence_state", 1
)[1].split("func.func", 1)[0]
assert advance_body.count("call @__vrc_opencl_dispatch_udononnx_slice_f32") == 2
assert advance_body.count("call @__vrc_opencl_await") == 2
assert "call @__vrc_opencl_buffer_copy" not in advance_body
assert "1792" in advance_body
assert "4194312" in advance_body
assert "4194311" in advance_body
assert "__udon_onnx_sequence_state_owned" in sequence_state_host.stdout
assert "__udon_onnx_sequence_state_pending" in sequence_state_host.stdout
assert "__udon_onnx_sequence_state_poisoned" in sequence_state_host.stdout
assert "arith.constant 7 : i32" in advance_body
assert "arith.constant 12 : i32" in advance_body

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
assert "call @__vrc_opencl_static_tensor" not in static_host.stdout
assert "call @udonclAdoptStaticBuffer" not in static_host.stdout
assert "@__udon_onnx_device_buffer_1" not in static_host.stdout
assert "func.func @udononnxGeneratedDispose" in static_host.stdout
assert "call @clReleaseMemObject" not in static_host.stdout

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
assert "call @__vrc_opencl_buffer_copy" in alias_host.stdout
assert "call @__vrc_opencl_await" in alias_host.stdout
assert "call @clEnqueueCopyBuffer" not in alias_host.stdout
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
assert "call @__vrc_opencl_dispatch_udononnx_cast_i64_f32" in cast_host.stdout
assert "call @udononnxCast" not in cast_host.stdout
assert "udoncl.onnx_device_graph = true" in cast_host.stdout
assert "udoncl.onnx_device_unsupported = []" in cast_host.stdout

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
assert unary_host.stdout.count("call @__vrc_opencl_dispatch_udondnn_large_op_tensor") == 1
assert unary_host.stdout.count("call @__vrc_opencl_dispatch_udondnn_large_activation") == 2
assert unary_host.stdout.count("@__udon_onnx_device_buffer_") == 6
assert unary_host.stdout.count("call @__vrc_opencl_dispatch_udononnx_unary_f32") == 3
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
        "call @__vrc_opencl_dispatch_udononnx_transpose_f32"
    )
    == 1
)
assert "call @udononnxTranspose" not in transpose_host.stdout

slice_source = source.with_name("onnx_slice_device.mlir")
slice_portable = subprocess.run(
    [
        opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-materialize-onnx-constants",
        "--udoncl-lower-onnx-to-host",
        slice_source,
    ],
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
assert "udoncl.onnx_device_graph = true" in slice_host.stdout
assert "udoncl.onnx_device_unsupported = []" in slice_host.stdout
assert slice_host.stdout.count("call @__vrc_opencl_dispatch_udononnx_slice_f32") == 1
assert "call @udononnxSlice" not in slice_host.stdout

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
        "call @__vrc_opencl_dispatch_udondnn_large_op_tensor"
    )
    == 1
)
assert (
    large_pointwise_host.stdout.count(
        "call @__vrc_opencl_dispatch_udondnn_large_activation"
    )
    == 2
)


def call_arguments(mlir, callee):
    return [
        [argument.strip() for argument in match.group(1).split(",")]
        for match in re.finditer(rf"call @{callee}\(([^)]*)\)", mlir)
    ]


pointwise_calls = call_arguments(
    large_pointwise_host.stdout,
    "__vrc_opencl_dispatch_udondnn_large_op_tensor",
)
activation_calls = call_arguments(
    large_pointwise_host.stdout,
    "__vrc_opencl_dispatch_udondnn_large_activation",
)
assert len(pointwise_calls) == 1
assert len(activation_calls) == 2
assert pointwise_calls[0][2] != pointwise_calls[0][3]
assert all(arguments[1] != arguments[2] for arguments in activation_calls)

gemm_bias_source = source.with_name("onnx_gemm_bias_device.mlir")
gemm_bias_portable = subprocess.run(
    [
        opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-lower-onnx-to-host",
        gemm_bias_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
gemm_bias_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=gemm_bias_portable,
    text=True,
    capture_output=True,
)
assert gemm_bias_host.returncode == 0, (
    gemm_bias_host.stdout,
    gemm_bias_host.stderr,
)
gemm_calls = call_arguments(
    gemm_bias_host.stdout, "__vrc_opencl_dispatch_udonblas_arena_sgemm"
)
gemm_bias_calls = call_arguments(
    gemm_bias_host.stdout,
    "__vrc_opencl_dispatch_udondnn_large_op_tensor",
)
assert len(gemm_calls) == 1
assert len(gemm_bias_calls) == 1
assert "@__udon_onnx_operator_scratch" in gemm_bias_host.stdout
assert gemm_calls[0][2] == gemm_bias_calls[0][0]
assert gemm_bias_calls[0][0] == gemm_bias_calls[0][2]
assert gemm_bias_calls[0][2] != gemm_bias_calls[0][3]

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
        "call @__vrc_opencl_dispatch_udononnx_matmul_nbits_q4"
    )
    == 2
)
assert "call @udononnxMatMulNBits" not in q4_host.stdout
assert "@__udon_onnx_operator_scratch" in q4_host.stdout
assert q4_host.stdout.count("arith.constant 524288 : i32") >= 2
assert "call @__vrc_opencl_buffer_copy" in q4_host.stdout
assert "call @__vrc_opencl_await" in q4_host.stdout
assert "call @clEnqueueCopyBuffer" not in q4_host.stdout
assert "arith.constant 128 : i32" in q4_host.stdout
assert "arith.constant 256 : i32" in q4_host.stdout
assert "cf.br ^" in q4_host.stdout
assert "cf.cond_br" in q4_host.stdout

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
        "call @__vrc_opencl_dispatch_udonaten_arena_rms_norm_reduce"
    )
    == 1
)
assert (
    rmsnorm_device_host.stdout.count(
        "call @__vrc_opencl_dispatch_udonaten_arena_rms_norm_finalize"
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
        "call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched"
    )
    == 1
)
assert "call @udonblasSgemmStridedBatched" not in batched_matmul_host.stdout

m1_attention_matmul_source = source.with_name("onnx_m1_attention_matmul.mlir")
m1_attention_matmul_portable = subprocess.run(
    [opt, "--udoncl-lower-onnx-to-host", m1_attention_matmul_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
m1_attention_matmul_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=m1_attention_matmul_portable,
    text=True,
    capture_output=True,
)
assert m1_attention_matmul_host.returncode == 0, (
    m1_attention_matmul_host.stdout,
    m1_attention_matmul_host.stderr,
)
assert "udoncl.onnx_device_graph = true" in m1_attention_matmul_host.stdout
assert (
    "call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched"
    in m1_attention_matmul_host.stdout
)

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
assert "udoncl.onnx_device_graph = true" in concat_host.stdout
assert concat_host.stdout.count("call @__vrc_opencl_dispatch_udononnx_concat2_f32") == 1
assert "call @udononnxConcat" not in concat_host.stdout

concat_device_source = source.with_name("onnx_concat_device.mlir")
concat_device_portable = subprocess.run(
    [opt, "--udoncl-lower-onnx-to-host", concat_device_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
concat_device_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=concat_device_portable,
    text=True,
    capture_output=True,
)
assert concat_device_host.returncode == 0, (
    concat_device_host.stdout,
    concat_device_host.stderr,
)
assert "udoncl.onnx_device_graph = true" in concat_device_host.stdout
assert "udoncl.onnx_device_unsupported = []" in concat_device_host.stdout
assert (
    concat_device_host.stdout.count(
        "call @__vrc_opencl_dispatch_udononnx_concat2_f32"
    )
    == 1
)
assert "call @udononnxConcat" not in concat_device_host.stdout

gather_device_source = source.with_name("onnx_gather_device.mlir")
gather_device_portable = subprocess.run(
    [
        opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-lower-onnx-to-host",
        gather_device_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
gather_device_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=gather_device_portable,
    text=True,
    capture_output=True,
)
assert gather_device_host.returncode == 0, (
    gather_device_host.stdout,
    gather_device_host.stderr,
)
assert "udoncl.onnx_device_graph = true" in gather_device_host.stdout
assert "udoncl.onnx_device_unsupported = []" in gather_device_host.stdout
assert (
    gather_device_host.stdout.count(
        "call @__vrc_opencl_dispatch_udononnx_gather_axis0_f32"
    )
    == 1
)
assert "call @udononnxGather" not in gather_device_host.stdout

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
assert "udoncl.onnx_device_graph = true" in slice_host.stdout
assert slice_host.stdout.count("call @__vrc_opencl_dispatch_udononnx_slice_f32") == 1
assert "call @udononnxSlice" not in slice_host.stdout

expand_device_source = source.with_name("onnx_expand_device.mlir")
expand_device_portable = subprocess.run(
    [
        opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-lower-onnx-to-host",
        expand_device_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
expand_device_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=expand_device_portable,
    text=True,
    capture_output=True,
)
assert expand_device_host.returncode == 0, (
    expand_device_host.stdout,
    expand_device_host.stderr,
)
assert "udoncl.onnx_device_graph = true" in expand_device_host.stdout
assert "udoncl.onnx_device_unsupported = []" in expand_device_host.stdout
assert (
    expand_device_host.stdout.count(
        "call @__vrc_opencl_dispatch_udononnx_expand_f32"
    )
    == 1
)
assert "call @udononnxExpand" not in expand_device_host.stdout

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
assert "udoncl.onnx_device_graph = true" in where_host.stdout
assert "udoncl.onnx_device_unsupported = []" in where_host.stdout
assert where_host.stdout.count("call @__vrc_opencl_dispatch_udononnx_equal_f32") == 1
assert where_host.stdout.count("call @__vrc_opencl_dispatch_udononnx_where_f32") == 1
assert "call @udononnxEqual" not in where_host.stdout
assert "call @udononnxWhere" not in where_host.stdout

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

device_pointer_fallback_source = source.with_name(
    "onnx_device_pointer_fallback.mlir"
)
device_pointer_fallback_portable = subprocess.run(
    [opt, "--udoncl-lower-onnx-to-host", device_pointer_fallback_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
device_pointer_fallback_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=device_pointer_fallback_portable,
    text=True,
    capture_output=True,
)
assert device_pointer_fallback_host.returncode != 0
assert (
    "ONNX Host fallback cannot call Device-pointer provider udonblasSgemm "
    "for MatMul"
    in device_pointer_fallback_host.stderr
), device_pointer_fallback_host.stderr

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
assert "udoncl.onnx_device_graph = true" in scatter_host.stdout
assert "udoncl.onnx_device_unsupported = []" in scatter_host.stdout
assert scatter_host.stdout.count("call @__vrc_opencl_dispatch_udononnx_scatter_nd_f32") == 1
assert "call @udononnxScatterND" not in scatter_host.stdout

gqa_source = source.with_name("onnx_m1_gqa_materialization.mlir")
gqa_portable = subprocess.run(
    [opt, "--udoncl-lower-onnx-to-host", gqa_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
gqa_host = subprocess.run(
    [host_opt, "--udoncl-finalize-onnx-host", "-"],
    input=gqa_portable,
    text=True,
    capture_output=True,
)
assert gqa_host.returncode == 0, (gqa_host.stdout, gqa_host.stderr)
assert "udoncl.onnx_device_graph = true" in gqa_host.stdout
assert "udoncl.onnx_device_unsupported = []" in gqa_host.stdout
assert gqa_host.stdout.count(
    "call @__vrc_opencl_dispatch_udononnx_expand_f32"
) == 1
assert gqa_host.stdout.count(
    "call @__vrc_opencl_dispatch_udononnx_transpose_f32"
) == 1
assert gqa_host.stdout.count(
    "call @__vrc_opencl_dispatch_udondnn_large_op_tensor"
) == 1

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
        "call @__vrc_opencl_dispatch_udondnn_arena_softmax"
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
