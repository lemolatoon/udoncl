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
assert "func.func @udon_onnx_session_write_input_float" in host.stdout
assert "func.func @udon_onnx_session_run" in host.stdout
assert "func.func @udon_onnx_session_read_output_float" in host.stdout
assert "func.func @udononnxGeneratedEnqueueInitialize" in host.stdout
assert "func.func @udononnxGeneratedEnqueueRun" in host.stdout
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
assert "call @udonatenAdd" in host.stdout
assert "udoncl.onnx_lowering = \"library-calls-v2\"" in host.stdout
unsupported = source.with_name("onnx_unsupported.mlir")
result = subprocess.run(
    [opt, "--verify-each=false", "--udoncl-lower-onnx-to-host", unsupported],
    text=True,
    capture_output=True,
)
assert result.returncode != 0, result.stdout
assert "unsupported ONNX operator Log" in result.stderr, result.stderr
print("onnx_cpp_lowering=passed")
