#!/usr/bin/env python3

from pathlib import Path
import re
import subprocess
import sys


host_opt = Path(sys.argv[1])
source = Path(sys.argv[2])
onnx_opt = [host_opt]

# A fixture whose nodes form a fusion group checks the library lowering of
# each operator with fusion off, and the synthesized kernel with it on.
UNFUSED_PLAN = "--udoncl-plan-onnx-host-graph=fuse-device-nodes=false"


def finalize_fused(portable):
    result = subprocess.run(
        [
            host_opt,
            "--udoncl-validate-onnx-host-graph",
            "--udoncl-plan-onnx-host-graph",
            "--udoncl-materialize-onnx-host-globals",
            "--udoncl-materialize-onnx-host-model-registry",
            "--udoncl-attach-onnx-host-metadata",
            "--udoncl-finalize-onnx-host",
            "--udoncl-cleanup-onnx-host-source",
            "-",
        ],
        input=portable,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "udoncl.onnx_device_graph = true" in result.stdout
    return result.stdout


def device_dispatches(mlir):
    return re.findall(r"call @__vrc_opencl_dispatch_(\w+)\(", mlir)


def global_definition(text, symbol):
    return next(
        line
        for line in text.splitlines()
        if '"udonhl.global"' in line and f'sym_name = "{symbol}"' in line
    )


def function_definition(text, symbol):
    match = re.search(
        rf"(?:^|\n)  func\.func @{re.escape(symbol)}\b.*?(?=\n  func\.func |\n}})",
        text,
        re.S,
    )
    assert match is not None, symbol
    return match.group(0)


def assert_pointer_global_accesses(text, symbol):
    loads = []
    stores = []
    address_count = 0
    for function in re.split(r"(?=  func\.func )", text):
        addresses = re.findall(
            rf"(%[A-Za-z0-9_.$]+) = udonhl\.address_of @{re.escape(symbol)} ",
            function,
        )
        address_count += len(addresses)
        for line in function.splitlines():
            if any(f'"udonhl.load"({address})' in line for address in addresses):
                loads.append(line)
            if any(
                '"udonhl.store"' in line and f", {address})" in line
                for address in addresses
            ):
                stores.append(line)
    assert address_count and loads and stores, (symbol, address_count)
    assert all("-> !udonhl.ptr<0>" in line for line in loads), loads
    assert all(
        "(!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()" in line for line in stores
    ), stores


def assert_owned_global_lifecycle(text, symbol):
    initialize = function_definition(text, "udononnxGeneratedEnqueueInitialize")
    dispose = function_definition(text, "udononnxGeneratedDispose")
    initialize_addresses = re.findall(
        rf"(%[A-Za-z0-9_.$]+) = udonhl\.address_of @{re.escape(symbol)} ",
        initialize,
    )
    dispose_addresses = re.findall(
        rf"(%[A-Za-z0-9_.$]+) = udonhl\.address_of @{re.escape(symbol)} ",
        dispose,
    )
    assert initialize_addresses and dispose_addresses, symbol

    initialize_stores = re.findall(
        r'"udonhl\.store"\((%[A-Za-z0-9_.$]+), (%[A-Za-z0-9_.$]+)\)',
        initialize,
    )
    initialize_values = [
        value for value, address in initialize_stores if address in initialize_addresses
    ]
    assert initialize_values, symbol
    assert any(
        re.search(
            rf"{re.escape(value)} = arith\.constant 1 : i32\b", initialize
        )
        for value in initialize_values
    ), symbol

    dispose_loads = []
    for address in dispose_addresses:
        dispose_loads.extend(
            re.findall(
                rf'(%[A-Za-z0-9_.$]+) = "udonhl\.load"\({re.escape(address)}\).*?-> i32',
                dispose,
            )
        )
    assert dispose_loads, symbol
    compare_results = []
    for loaded in dispose_loads:
        compare_results.extend(
            re.findall(
                rf"(%[A-Za-z0-9_.$]+) = arith\.cmpi ne, {re.escape(loaded)}, ",
                dispose,
            )
        )
    assert compare_results, symbol
    assert any(f"cf.cond_br {result}" in dispose for result in compare_results), symbol

    dispose_stores = re.findall(
        r'"udonhl\.store"\((%[A-Za-z0-9_.$]+), (%[A-Za-z0-9_.$]+)\)',
        dispose,
    )
    dispose_values = [
        value for value, address in dispose_stores if address in dispose_addresses
    ]
    assert dispose_values, symbol
    assert any(
        re.search(rf"{re.escape(value)} = arith\.constant 0 : i32\b", dispose)
        for value in dispose_values
    ), symbol


result = subprocess.run(
    [
        *onnx_opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
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
    [*onnx_opt, "--udoncl-infer-onnx-shapes", "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", source],
    text=True,
    capture_output=True,
    check=True,
).stdout
host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
assert "call @udononnxBindingInitializeTensorMetadata" not in host.stdout
assert "call @udononnxBindingInitializeModel" not in host.stdout
assert "func.func private @udononnxBindingInitializeTensorMetadata" not in host.stdout
assert "func.func private @udononnxBindingInitializeModel" not in host.stdout
assert "udonhl.store" in host.stdout
assert "arith.cmpi ne" in host.stdout
assert "func.func @udon_onnx_binding_create" in host.stdout
assert "func.func @udon_onnx_binding_bind_tensor" in host.stdout
assert "func.func @udon_onnx_binding_bind_all" in host.stdout
bind_all_start = host.stdout.index("func.func @udon_onnx_binding_bind_all")
bind_all_end = host.stdout.index("func.func", bind_all_start + 10)
bind_all = host.stdout[bind_all_start:bind_all_end]
assert "call @udononnxBindingStateBindTensor" not in bind_all
assert "call @udononnxBindingStateBeginStaticBind" in bind_all
assert "call @udononnxBindingStateCompleteStaticBind" in bind_all
assert bind_all.count("call @udonclDeviceBufferView") == 3
assert "udonhl.alloca" not in bind_all
assert host.stdout.count("vrc.static_public_resource_binding") == 3
assert host.stdout.count('public_access = "input"') >= 2
assert host.stdout.count('public_access = "output"') >= 1
assert host.stdout.count('resource_source = "call_allocation"') == 3
assert "func.func @udon_onnx_binding_enqueue_initialize" in host.stdout
assert "func.func @udon_onnx_binding_enqueue_run" in host.stdout
assert "func.func @udon_onnx_binding_wait_active_event" not in host.stdout
assert "func.func @udon_onnx_binding_dispose" in host.stdout
assert "call @udononnxSessionResolveInput" in host.stdout
assert "call @udononnxSessionResolveOutput" in host.stdout
assert "call @clEnqueueMarkerWithWaitList" in host.stdout
# Without a wait list the run only issues what the caller enqueued before
# it (clFlush); it waits for a marker only over a caller wait list.
assert host.stdout.count("call @clFlush") == 1
assert "call @__vrc_opencl_dispatch_udondnn_large_op_tensor" in host.stdout
assert "udoncl.onnx_lowering = \"library-calls-v2\"" in host.stdout
assert "udoncl.onnx_device_graph = true" in host.stdout
assert "udoncl.onnx_device_unsupported = []" in host.stdout
assert "udoncl.onnx_operator_counts" in host.stdout
assert "logical_bytes" in host.stdout
assert "capacity_bytes" in host.stdout
assert "udoncl.mutable_handle_capacity" in host.stdout
assert host.stdout.count("vrc.buffer_view_layout") == 6
metadata_global = next(
    line
    for line in host.stdout.splitlines()
    if 'sym_name = "__udon_onnx_input_metadata"' in line
)
assert "vrc.buffer_view_layout" in metadata_global
assert "count = 0 : i32" in metadata_global
binding_global = next(
    line
    for line in host.stdout.splitlines()
    if 'sym_name = "__udon_onnx_binding_inputs"' in line
)
assert "vrc.buffer_view_layout" in binding_global
assert "schema_version = 1 : i32" in binding_global
assert "count = 2 : i32" in binding_global
assert "stride_bytes = 88 : i32" in binding_global
assert "offsets = array<i32: 0>" in binding_global
output_binding_global = next(
    line
    for line in host.stdout.splitlines()
    if 'sym_name = "__udon_onnx_binding_outputs"' in line
)
assert "count = 1 : i32" in output_binding_global

sequence_state_source = source.with_name("onnx_sequence_state.mlir")
sequence_state = subprocess.run(
    [*onnx_opt, "--udoncl-identify-onnx-sequence-state", sequence_state_source],
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

sequence_state_standard_return = subprocess.run(
    [
        host_opt,
        "--standard-func-return",
        sequence_state_source,
    ],
    text=True,
    capture_output=True,
)
assert sequence_state_standard_return.returncode == 0, (
    sequence_state_standard_return.stdout,
    sequence_state_standard_return.stderr,
)
sequence_state_after_standard_return = subprocess.run(
    [*onnx_opt, "--udoncl-identify-onnx-sequence-state", "-"],
    input=sequence_state_standard_return.stdout,
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
        *onnx_opt,
        "--udoncl-identify-onnx-sequence-state",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
        "--udoncl-lower-onnx-to-host",
        sequence_state_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
sequence_state_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
# The queue is drained once, then each state is cleared by one word-fill
# command with the 4-byte zero pattern; nothing is uploaded.
assert reset_body.count("call @udononnxBindingStateGetCommandQueue") == 1
assert reset_body.count("call @clFinish") == 1
assert reset_body.count("call @__vrc_opencl_buffer_fill") == 2
assert "call @__vrc_opencl_buffer_upload" not in reset_body
assert reset_body.count("call @__vrc_opencl_await") == 2
assert "arith.constant 4 : i32" in reset_body
assert reset_body.index("call @clFinish") < reset_body.index(
    "call @__vrc_opencl_buffer_fill"
)
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
        *onnx_opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
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
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
assert "__udon_onnx_device_buffer_owned_" not in static_host.stdout
assert "__udon_onnx_operator_scratch_owned" not in static_host.stdout
assert "func.func @udononnxGeneratedDispose" in static_host.stdout
assert "call @clReleaseMemObject" not in static_host.stdout

expand_source = source.with_name("onnx_expand_shape.mlir")
expand_inferred = subprocess.run(
    [*onnx_opt, "--udoncl-infer-onnx-shapes", expand_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
assert "-> tensor<1x32x1xf32>" in expand_inferred
assert "-> tensor<*xf32>" not in expand_inferred

alias_source = source.with_name("onnx_metadata_alias.mlir")
alias_portable = subprocess.run(
    [
        *onnx_opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
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
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
    [*onnx_opt, "--udoncl-infer-onnx-shapes", "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", cast_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
cast_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
    [*onnx_opt, "--udoncl-infer-onnx-shapes", "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", unary_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
unary_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        UNFUSED_PLAN,
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
    input=unary_portable,
    text=True,
    capture_output=True,
)
assert unary_host.returncode == 0, (unary_host.stdout, unary_host.stderr)
assert "udoncl.onnx_device_graph = true" in unary_host.stdout
assert "udoncl.onnx_device_unsupported = []" in unary_host.stdout
assert unary_host.stdout.count("call @__vrc_opencl_dispatch_udondnn_large_op_tensor") == 1
assert unary_host.stdout.count("call @__vrc_opencl_dispatch_udondnn_large_activation") == 2
device_symbols = re.findall(
    r'sym_name = "(__udon_onnx_device_buffer_[0-9]+)"', unary_host.stdout
)
assert len(device_symbols) == 2
owned_symbols = re.findall(
    r'sym_name = "(__udon_onnx_device_buffer_owned_[0-9]+)"', unary_host.stdout
)
expected_owned_symbols = {
    symbol.replace(
        "__udon_onnx_device_buffer_", "__udon_onnx_device_buffer_owned_"
    )
    for symbol in device_symbols
}
assert set(owned_symbols) == expected_owned_symbols
initialize_body = function_definition(
    unary_host.stdout, "udononnxGeneratedEnqueueInitialize"
)
dispose_body = function_definition(unary_host.stdout, "udononnxGeneratedDispose")
assert "udonhl.ptrtoint" not in initialize_body
assert "udonhl.ptrtoint" not in dispose_body
for symbol in device_symbols:
    definition = global_definition(unary_host.stdout, symbol)
    assert "storage_type = !udonhl.ptr<0>" in definition
    assert "initial_value = 0 : i32" in definition
    assert_pointer_global_accesses(unary_host.stdout, symbol)
    owned_symbol = symbol.replace(
        "__udon_onnx_device_buffer_", "__udon_onnx_device_buffer_owned_"
    )
    owned_definition = global_definition(unary_host.stdout, owned_symbol)
    assert "storage_type = i32" in owned_definition
    assert "initial_value = 0 : i32" in owned_definition
    assert owned_symbol in initialize_body
    assert owned_symbol in dispose_body
    assert_owned_global_lifecycle(unary_host.stdout, owned_symbol)
assert "storage_type = i32" in global_definition(
    unary_host.stdout, "__udon_onnx_binding_event"
)
assert unary_host.stdout.count("call @__vrc_opencl_dispatch_udononnx_unary_f32") == 3
assert "call @udononnxCos" not in unary_host.stdout
assert "call @udononnxSin" not in unary_host.stdout
assert "call @udononnxNeg" not in unary_host.stdout
# Fused, the six nodes are one kernel and no intermediate owns a buffer.
unary_fused = finalize_fused(unary_portable)
assert device_dispatches(unary_fused) == ["udononnx_fused_0"]
assert "__udon_onnx_device_buffer_" not in unary_fused
for operation, count in [
    ("math.cos", 1),
    ("math.sin", 1),
    ("arith.negf", 2),
    ("math.exp", 2),
    ("memref.load", 1),
]:
    assert unary_fused.count(operation) == count, operation

transpose_source = source.with_name("onnx_transpose_device.mlir")
transpose_portable = subprocess.run(
    [
        *onnx_opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
        "--udoncl-lower-onnx-to-host",
        transpose_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
transpose_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
        *onnx_opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
        "--udoncl-lower-onnx-to-host",
        slice_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
slice_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
        *onnx_opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
        "--udoncl-lower-onnx-to-host",
        large_pointwise_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
large_pointwise_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        UNFUSED_PLAN,
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
large_pointwise_fused = finalize_fused(large_pointwise_portable)
assert device_dispatches(large_pointwise_fused) == ["udononnx_fused_0"]
assert (
    "gpu.func @udononnx_fused_0(%arg0: memref<20000xf32>, "
    "%arg1: memref<24576xf32>) kernel" in large_pointwise_fused
)
# 20000 elements take three rows of the 8192 wide surface.
assert "known_grid_size = array<i32: 8192, 3, 1>" in large_pointwise_fused

gemm_bias_source = source.with_name("onnx_gemm_bias_device.mlir")
gemm_bias_portable = subprocess.run(
    [
        *onnx_opt,
        "--udoncl-infer-onnx-shapes",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
        "--udoncl-lower-onnx-to-host",
        gemm_bias_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
gemm_bias_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
assert_pointer_global_accesses(
    gemm_bias_host.stdout, "__udon_onnx_operator_scratch"
)
assert "storage_type = i32" in global_definition(
    gemm_bias_host.stdout, "__udon_onnx_operator_scratch_owned"
)
assert_owned_global_lifecycle(
    gemm_bias_host.stdout, "__udon_onnx_operator_scratch_owned"
)
assert gemm_calls[0][2] == gemm_bias_calls[0][0]
assert gemm_bias_calls[0][0] == gemm_bias_calls[0][2]
assert gemm_bias_calls[0][2] != gemm_bias_calls[0][3]

q4_source = source.with_name("onnx_matmul_nbits_device.mlir")
q4_portable = subprocess.run(
    [
        *onnx_opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
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
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
    input=q4_portable,
    text=True,
    capture_output=True,
)
assert q4_host.returncode == 0, (q4_host.stdout, q4_host.stderr)
assert "udoncl.onnx_device_graph = true" in q4_host.stdout
assert "udoncl.onnx_device_unsupported = []" in q4_host.stdout
# K = 256 is 8 blocks of 32: the whole K is accumulated by one dispatch
# (block_base 0, block_count 8, accumulate 0) that writes the output directly;
# no Host loop over K tiles remains.
q4_calls = call_arguments(
    q4_host.stdout, "__vrc_opencl_dispatch_udononnx_matmul_nbits_q4"
)
assert len(q4_calls) == 1, q4_calls
q4_call_offset = q4_host.stdout.index(
    "call @__vrc_opencl_dispatch_udononnx_matmul_nbits_q4"
)
q4_function = q4_host.stdout[
    q4_host.stdout.rindex("func.func @", 0, q4_call_offset):q4_call_offset
]
q4_constants = dict(
    re.findall(r"(%[\w#]+) = arith\.constant (-?\d+) : i32", q4_function)
)
assert len(q4_calls[0]) == 11, q4_calls
assert [q4_constants.get(argument) for argument in q4_calls[0][5:]] == [
    "1", "256", "256", "0", "8", "0",
], q4_calls
assert q4_calls[0][3] != q4_calls[0][4]
assert "call @udononnxMatMulNBits" not in q4_host.stdout
assert "@__udon_onnx_operator_scratch" in q4_host.stdout
scratch_definition = global_definition(
    q4_host.stdout, "__udon_onnx_operator_scratch"
)
assert "storage_type = !udonhl.ptr<0>" in scratch_definition
assert "initial_value = 0 : i32" in scratch_definition
assert_pointer_global_accesses(q4_host.stdout, "__udon_onnx_operator_scratch")
scratch_owned_definition = global_definition(
    q4_host.stdout, "__udon_onnx_operator_scratch_owned"
)
assert "storage_type = i32" in scratch_owned_definition
assert "initial_value = 0 : i32" in scratch_owned_definition
q4_initialize_body = function_definition(
    q4_host.stdout, "udononnxGeneratedEnqueueInitialize"
)
q4_dispose_body = function_definition(q4_host.stdout, "udononnxGeneratedDispose")
assert "__udon_onnx_operator_scratch_owned" in q4_initialize_body
assert "__udon_onnx_operator_scratch_owned" in q4_dispose_body
assert "udonhl.ptrtoint" not in q4_initialize_body
assert "udonhl.ptrtoint" not in q4_dispose_body
assert_owned_global_lifecycle(q4_host.stdout, "__udon_onnx_operator_scratch_owned")
assert q4_host.stdout.count("arith.constant 524288 : i32") >= 2
assert "call @__vrc_opencl_buffer_copy" in q4_host.stdout
assert "call @__vrc_opencl_await" in q4_host.stdout
assert "call @clEnqueueCopyBuffer" not in q4_host.stdout

rmsnorm_source = source.with_name("onnx_rmsnorm_pattern.mlir")
rmsnorm_canonical = subprocess.run(
    [*onnx_opt, "--udoncl-canonicalize-onnx-library-patterns", rmsnorm_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
assert 'function_name = "RmsNorm"' in rmsnorm_canonical
assert 'domain_name = "com.lemolatoon.udoncl"' in rmsnorm_canonical

rmsnorm_device_source = source.with_name("onnx_rmsnorm_device.mlir")
rmsnorm_device_portable = subprocess.run(
    [
        *onnx_opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
        "--udoncl-lower-onnx-to-host",
        rmsnorm_device_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
rmsnorm_device_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
assert_pointer_global_accesses(
    rmsnorm_device_host.stdout, "__udon_onnx_operator_scratch"
)
assert "storage_type = i32" in global_definition(
    rmsnorm_device_host.stdout, "__udon_onnx_operator_scratch_owned"
)
assert_owned_global_lifecycle(
    rmsnorm_device_host.stdout, "__udon_onnx_operator_scratch_owned"
)
assert "arith.constant 65536 : i32" in rmsnorm_device_host.stdout

batched_matmul_source = source.with_name("onnx_batched_matmul.mlir")
batched_matmul_portable = subprocess.run(
    [
        *onnx_opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
        "--udoncl-lower-onnx-to-host",
        batched_matmul_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
batched_matmul_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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


def lower_device_graph(name):
    portable = subprocess.run(
        [
            *onnx_opt,
            "--udoncl-materialize-onnx-constants",
            "--udoncl-externalize-onnx-initializers",
            "--udoncl-externalize-onnx-graph-constants",
            "--udoncl-lower-onnx-to-host",
            source.with_name(name),
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    host = subprocess.run(
        [
            host_opt,
            "--udoncl-validate-onnx-host-graph",
            "--udoncl-plan-onnx-host-graph",
            "--udoncl-materialize-onnx-host-globals",
            "--udoncl-materialize-onnx-host-model-registry",
            "--udoncl-attach-onnx-host-metadata",
            "--udoncl-finalize-onnx-host",
            "--udoncl-cleanup-onnx-host-source",
            "-",
        ],
        input=portable,
        text=True,
        capture_output=True,
    )
    assert host.returncode == 0, (host.stdout, host.stderr)
    assert "udoncl.onnx_device_graph = true" in host.stdout, name
    assert "udoncl.onnx_device_unsupported = []" in host.stdout, name
    return host.stdout


# K beyond one 256-wide chunk stays on the non-batched Device SGEMM.
long_k_host = lower_device_graph("onnx_matmul_long_k_device.mlir")
assert long_k_host.count("call @__vrc_opencl_dispatch_udonblas_arena_sgemm(") == 1
assert "arith.constant 1024 : i32" in long_k_host
# A 256 x 128 weight (32768 words) exceeds the 16384 words the non-batched
# kernel addresses: it must use the strided-batched kernel with one batch
# instead of wrapping its indices.
wide_host = lower_device_graph("onnx_matmul_wide_device.mlir")
assert "call @__vrc_opencl_dispatch_udonblas_arena_sgemm(" not in wide_host
assert (
    wide_host.count(
        "call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched("
    )
    == 1
)

m1_attention_matmul_source = source.with_name("onnx_m1_attention_matmul.mlir")
m1_attention_matmul_portable = subprocess.run(
    [*onnx_opt, "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", m1_attention_matmul_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
m1_attention_matmul_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
    [*onnx_opt, "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", concat_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
concat_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
    [*onnx_opt, "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", concat_device_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
concat_device_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
        *onnx_opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
        "--udoncl-lower-onnx-to-host",
        gather_device_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
gather_device_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
    [*onnx_opt, "--udoncl-materialize-onnx-constants", "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", slice_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
slice_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
        *onnx_opt,
        "--udoncl-materialize-onnx-constants",
        "--udoncl-externalize-onnx-initializers",
        "--udoncl-externalize-onnx-graph-constants",
        "--udoncl-lower-onnx-to-host",
        expand_device_source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
expand_device_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
    [*onnx_opt, "--udoncl-materialize-onnx-constants", "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", where_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
where_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
    [*onnx_opt, "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", integer_add_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
integer_add_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
    [*onnx_opt, "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", device_pointer_fallback_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
device_pointer_fallback_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
    [*onnx_opt, "--udoncl-materialize-onnx-constants", "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", scatter_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
scatter_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
    [*onnx_opt, "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", gqa_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
gqa_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        UNFUSED_PLAN,
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
# Fused, the whole materialization is index arithmetic in front of the Loads
# of the input and of the scale, for every fragment of the full surface.
gqa_fused = finalize_fused(gqa_portable)
assert device_dispatches(gqa_fused) == ["udononnx_fused_0"]
assert (
    "gpu.func @udononnx_fused_0(%arg0: memref<16384xf32>, %arg1: memref<1xf32>, "
    "%arg2: memref<131072xf32>) kernel" in gqa_fused
)
assert "vrc.fragment_cost = {loads = 2 : i64}" in gqa_fused
assert gqa_fused.count("memref.load") == 2

softmax_source = source.with_name("onnx_softmax.mlir")
softmax_portable = subprocess.run(
    [*onnx_opt, "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", softmax_source],
    text=True,
    capture_output=True,
    check=True,
).stdout
softmax_host = subprocess.run(
    [
        host_opt,
        "--udoncl-validate-onnx-host-graph",
        "--udoncl-plan-onnx-host-graph",
        "--udoncl-materialize-onnx-host-globals",
        "--udoncl-materialize-onnx-host-model-registry",
        "--udoncl-attach-onnx-host-metadata",
        "--udoncl-finalize-onnx-host",
        "--udoncl-cleanup-onnx-host-source",
        "-",
    ],
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
    [*onnx_opt, "--verify-each=false", "--udoncl-externalize-onnx-initializers", "--udoncl-externalize-onnx-graph-constants", "--udoncl-lower-onnx-to-host", unsupported],
    text=True,
    capture_output=True,
)
assert result.returncode != 0, result.stdout
assert "unsupported ONNX operator Log" in result.stderr, result.stderr
print("onnx_cpp_lowering=passed")
