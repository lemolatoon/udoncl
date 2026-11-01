"""Validate the dependency-free ONNX importer and MNIST Udon provider plan."""

import json
from pathlib import Path
import subprocess
import sys


root, output = (Path(value).resolve() for value in sys.argv[1:])
c_compiler = root / ".deps/llvm23/bin" / (
    "clang.exe" if sys.platform == "win32" else "clang"
)
assert c_compiler.is_file(), f"pinned Clang is missing: {c_compiler}"
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "runtime/onnx"))

from udon_onnx import (
    ExecutionProviderRegistry,
    attribute_float,
    bytes_field,
    compile_model,
    compile_operators,
    emit_c_host,
    emit_uasm_host,
    external_tensor_proto,
    graph_proto,
    model_proto,
    node_proto,
    parse_model,
    tensor_proto,
    tensor_int64_proto,
    value_info_proto,
)


model_path = output / "mnist.onnx"
result = subprocess.run(
    [
        sys.executable,
        root / "examples/mnist-cnn/generate_onnx.py",
        "-o",
        model_path,
    ],
    text=True,
    capture_output=True,
    timeout=60,
)
assert result.returncode == 0, (result.stdout, result.stderr)
data = model_path.read_bytes()
model = parse_model(data + bytes_field(99, b"unknown-field-is-skipped"))
plan = compile_model(model)
assert plan["runtime"] == "udon-onnx"
assert plan["execution_provider"] == "UdonMnistExecutionProvider"
assert plan["inputs"] == {"image": [1, 1, 28, 28]}
assert plan["outputs"] == {"logits": [1, 10]}
assert [node["op_type"] for node in plan["nodes"]] == [
    "Conv",
    "Add",
    "Relu",
    "AveragePool",
    "Flatten",
    "Gemm",
    "Add",
]
assert [node["provider"] for node in plan["nodes"]] == [
    "udondnnConvolutionForward",
    "udondnnOpTensor",
    "udondnnActivationForward",
    "udondnnPoolingForward",
    "view",
    "udonblasSgemm",
    "udondnnOpTensor",
]
assert [item["fragments"] for item in plan["passes"]] == [9216, 144, 10]
mnist_calls = [call["function"] for call in plan["aot"]["calls"]]
assert mnist_calls[:5] == [
    "udondnnConvolutionForward",
    "udondnnOpTensor",
    "udondnnActivationForward",
    "udondnnPoolingForward",
    "view",
]
assert mnist_calls[5:-1] == ["udonblasSgemm"] * 9
assert mnist_calls[-1] == "udondnnOpTensor"
assert plan["nodes"][0]["shape"] == [1, 16, 24, 24]
assert plan["nodes"][3]["shape"] == [1, 16, 3, 3]
assert plan["nodes"][4]["shape"] == [1, 144]
assert set(plan["initializers"]) == {
    "conv.weight",
    "conv.bias",
    "dense.weight",
    "dense.bias",
}

generic_mnist_plan = compile_operators(model)
assert generic_mnist_plan["execution_provider"] == "UdonOperatorExecutionProvider"
assert [item["providers"][0] for item in generic_mnist_plan["passes"][:4]] == [
    "udondnnConvolutionForward",
    "udondnnOpTensor",
    "udondnnActivationForward",
    "udondnnPoolingForward",
]
assert len(generic_mnist_plan["passes"]) == 14
assert sum(
    item["providers"] == ["udonblasSgemm"] for item in generic_mnist_plan["passes"]
) == 9
generic_mnist_uasm = emit_uasm_host(generic_mnist_plan)
assert generic_mnist_uasm.count("VRCSDKBaseVRCGraphics.__Blit__") == 14
(output / "mnist-operators-host.uasm").write_text(generic_mnist_uasm, encoding="utf-8")

operator_model = parse_model(
    model_proto(
        graph_proto(
            "generic_operator_graph",
            [
                node_proto("Gemm", ["x", "weight"], ["matrix"], "gemm"),
                node_proto("Add", ["matrix", "bias"], ["biased"], "bias"),
                node_proto("Relu", ["biased"], ["y"], "relu"),
            ],
            [
                tensor_proto("weight", [3, 4], [0.25] * 12),
                tensor_proto("bias", [4], [0.5] * 4),
            ],
            [value_info_proto("x", [2, 3])],
            [value_info_proto("y", [2, 4])],
        )
    )
)
operator_plan = compile_model(operator_model)
assert operator_plan["execution_provider"] == "UdonOperatorExecutionProvider"
assert [item["function"] for item in operator_plan["aot"]["calls"]] == [
    "udonblasSgemm",
    "udondnnOpTensor",
    "udondnnActivationForward",
]
assert [item["fragments"] for item in operator_plan["passes"]] == [8, 8, 8]
host_c = emit_c_host(operator_plan)
assert "udonblasSgemm(" in host_c
assert "udondnnOpTensor(" in host_c
assert "udondnnActivationForward(" in host_c
assert host_c.count("udonblasSgemm(") == 1
(output / "generic-host.c").write_text(host_c, encoding="utf-8")
host_uasm = emit_uasm_host(operator_plan)
assert host_uasm.count("VRCSDKBaseVRCGraphics.__Blit__") == 3
assert ".export pass200Material" in host_uasm
assert ".export pass201Material" in host_uasm
assert ".export pass202Material" in host_uasm
assert ".export resultState" in host_uasm
(output / "generic-host.uasm").write_text(host_uasm, encoding="utf-8")

tiled_model = parse_model(
    model_proto(
        graph_proto(
            "tiled_gemm_graph",
            [node_proto("Gemm", ["left", "right"], ["result"], "tiled_gemm")],
            [tensor_proto("right", [33, 18], [0.125] * (33 * 18))],
            [value_info_proto("left", [17, 33])],
            [value_info_proto("result", [17, 18])],
        )
    )
)
tiled_plan = compile_model(tiled_model)
tiled_host_c = emit_c_host(tiled_plan)
tiled_host_uasm = emit_uasm_host(tiled_plan)
assert tiled_plan["execution_provider"] == "UdonOperatorExecutionProvider"
assert len(tiled_plan["passes"]) == 12
assert [item["id"] for item in tiled_plan["passes"]] == list(range(200, 212))
assert tiled_plan["passes"][0]["tile"] == {
    "m_offset": 0,
    "n_offset": 0,
    "k_offset": 0,
    "m": 16,
    "n": 16,
    "k": 16,
    "alpha": 1.0,
    "beta": 0,
}
assert tiled_plan["passes"][1]["tile"]["beta"] == 1
assert tiled_plan["passes"][-1]["tile"] == {
    "m_offset": 16,
    "n_offset": 16,
    "k_offset": 32,
    "m": 1,
    "n": 2,
    "k": 1,
    "alpha": 1.0,
    "beta": 1,
}
assert tiled_host_c.count("udonblasSgemm(") == 12
assert "buffers->v0_left + 528" in tiled_host_c
assert tiled_host_uasm.count("VRCSDKBaseVRCGraphics.__Blit__") == 12
assert ".export pass211Material" in tiled_host_uasm
(output / "tiled-gemm-host.c").write_text(tiled_host_c, encoding="utf-8")
(output / "tiled-gemm-host.uasm").write_text(tiled_host_uasm, encoding="utf-8")

biased_gemm_model = parse_model(
    model_proto(
        graph_proto(
            "biased_scaled_gemm",
            [
                node_proto(
                    "Gemm",
                    ["x", "weight", "bias"],
                    ["y"],
                    "gemm",
                    [attribute_float("alpha", 0.5), attribute_float("beta", 2.0)],
                )
            ],
            [
                tensor_proto("weight", [3, 4], [0.25] * 12),
                tensor_proto("bias", [4], [0.5] * 4),
            ],
            [value_info_proto("x", [2, 3])],
            [value_info_proto("y", [2, 4])],
        )
    )
)
biased_gemm_plan = compile_model(biased_gemm_model)
assert [item["function"] for item in biased_gemm_plan["aot"]["calls"]] == [
    "udonblasSgemm",
    "udondnnOpTensor",
]
assert biased_gemm_plan["aot"]["calls"][0]["tile"]["alpha"] == 0.5
assert biased_gemm_plan["aot"]["calls"][1]["phase"] == "bias"
assert biased_gemm_plan["passes"][1]["phase"] == "bias"
assert biased_gemm_plan["passes"][1]["beta"] == 2.0
biased_gemm_host_c = emit_c_host(biased_gemm_plan)
assert "const float gemm_alpha_0 = 0.5f;" in biased_gemm_host_c
assert "const float gemm_beta_0 = 2.0f;" in biased_gemm_host_c
assert biased_gemm_host_c.count("udonblasSgemm(") == 1
assert biased_gemm_host_c.count("udondnnOpTensor(") == 1
(output / "biased-gemm-host.c").write_text(biased_gemm_host_c, encoding="utf-8")

external_gemm_model = parse_model(
    model_proto(
        graph_proto(
            "external_weight_gemm",
            [node_proto("Gemm", ["x", "weight"], ["y"], "gemm")],
            [external_tensor_proto("weight", [3, 4], 1, "weights.bin", 16, 48)],
            [value_info_proto("x", [2, 3])],
            [value_info_proto("y", [2, 4])],
        )
    )
)
external_gemm_plan = compile_model(external_gemm_model)
assert external_gemm_plan["initializers"]["weight"] == {
    "shape": [3, 4],
    "data_type": "float32",
    "bytes": 48,
    "storage": "external",
    "external": {"location": "weights.bin", "offset": 16, "length": 48},
}

rl_model_path = output / "rl-policy.onnx"
rl_generate = subprocess.run(
    [
        sys.executable,
        root / "examples/reinforcement-learning/generate_onnx.py",
        "-o",
        rl_model_path,
    ],
    text=True,
    capture_output=True,
    timeout=60,
)
assert rl_generate.returncode == 0, (rl_generate.stdout, rl_generate.stderr)
rl_plan = compile_operators(parse_model(rl_model_path.read_bytes()))
assert rl_plan["execution_provider"] == "UdonOperatorExecutionProvider"
assert [item["op_type"] for item in rl_plan["nodes"]] == [
    "Gemm",
    "Add",
    "Tanh",
    "Sigmoid",
    "Mul",
    "Gemm",
    "Add",
    "Sub",
    "Mul",
    "Softmax",
]
assert [item["function"] for item in rl_plan["aot"]["calls"]] == [
    "udonblasSgemm",
    "udondnnOpTensor",
    "udondnnActivationForward",
    "udondnnActivationForward",
    "udondnnOpTensor",
    "udonblasSgemm",
    "udondnnOpTensor",
    "udondnnOpTensor",
    "udondnnOpTensor",
    "udondnnSoftmaxForward",
]
assert len(rl_plan["passes"]) == 10
rl_host_c = emit_c_host(rl_plan)
assert "UDONDNN_ACTIVATION_TANH" in rl_host_c
assert "UDONDNN_ACTIVATION_SIGMOID" in rl_host_c
assert "UDONDNN_OP_TENSOR_MUL" in rl_host_c
assert "&minus_one" in rl_host_c
assert "UDONDNN_SOFTMAX_ACCURATE, UDONDNN_SOFTMAX_MODE_INSTANCE" in rl_host_c
(output / "rl-policy-host.c").write_text(rl_host_c, encoding="utf-8")
rl_host_uasm = emit_uasm_host(rl_plan)
assert rl_host_uasm.count("VRCSDKBaseVRCGraphics.__Blit__") == 10
(output / "rl-policy-host.uasm").write_text(rl_host_uasm, encoding="utf-8")

attention_matmul_model = parse_model(
    model_proto(
        graph_proto(
            "broadcast_batched_attention_matmul",
            [node_proto("MatMul", ["query", "key"], ["scores"], "attention")],
            [],
            [
                value_info_proto("query", [2, 1, 3, 4]),
                value_info_proto("key", [1, 5, 4, 6]),
            ],
            [value_info_proto("scores", [2, 5, 3, 6])],
        )
    )
)
attention_matmul_plan = compile_operators(attention_matmul_model)
assert attention_matmul_plan["nodes"][0]["shape"] == [2, 5, 3, 6]
assert len(attention_matmul_plan["passes"]) == 10
assert len(attention_matmul_plan["aot"]["calls"]) == 10
assert attention_matmul_plan["passes"][0]["tile"]["left_offset"] == 0
assert attention_matmul_plan["passes"][1]["tile"]["left_offset"] == 0
assert attention_matmul_plan["passes"][1]["tile"]["right_offset"] == 24
assert attention_matmul_plan["passes"][5]["tile"]["left_offset"] == 12
assert attention_matmul_plan["passes"][-1]["tile"]["right_offset"] == 96
assert attention_matmul_plan["passes"][-1]["tile"]["output_offset"] == 162
attention_matmul_host_c = emit_c_host(attention_matmul_plan)
assert attention_matmul_host_c.count("udonblasSgemm(") == 10
(output / "attention-matmul-host.c").write_text(
    attention_matmul_host_c, encoding="utf-8"
)

llama_view_model = parse_model(
    model_proto(
        graph_proto(
            "llama_shape_views",
            [
                node_proto("Reshape", ["x", "reshape.shape"], ["flat"], "reshape"),
                node_proto("Unsqueeze", ["flat", "unsqueeze.axes"], ["expanded"], "unsqueeze"),
                node_proto("Squeeze", ["expanded", "squeeze.axes"], ["squeezed"], "squeeze"),
                node_proto("Identity", ["squeezed"], ["y"], "identity"),
            ],
            [
                tensor_int64_proto("reshape.shape", [2], [4, -1]),
                tensor_int64_proto("unsqueeze.axes", [1], [0]),
                tensor_int64_proto("squeeze.axes", [1], [0]),
            ],
            [value_info_proto("x", [1, 4, 8])],
            [value_info_proto("y", [4, 8])],
        )
    )
)
llama_view_plan = compile_operators(llama_view_model)
assert [node["shape"] for node in llama_view_plan["nodes"]] == [
    [4, 8],
    [1, 4, 8],
    [4, 8],
    [4, 8],
]
assert not llama_view_plan["passes"]
assert {item["kind"] for item in llama_view_plan["aot"]["calls"]} == {"view"}
assert llama_view_plan["value_types"]["reshape.shape"] == "int64"
llama_view_host_c = emit_c_host(llama_view_plan)
assert "int64_t *" in llama_view_host_c
assert llama_view_host_c.count(" = buffers->") == 4
(output / "llama-view-host.c").write_text(llama_view_host_c, encoding="utf-8")

rms_model = parse_model(
    model_proto(
        graph_proto(
            "standard_rms_normalization",
            [node_proto("RMSNormalization", ["x", "scale"], ["y"], "rms")],
            [tensor_proto("scale", [8], [0.5 + index * 0.125 for index in range(8)])],
            [value_info_proto("x", [2, 8])],
            [value_info_proto("y", [2, 8])],
        ),
        opset=23,
    )
)
rms_plan = compile_operators(rms_model)
assert rms_plan["aot"]["calls"][0]["function"] == "udondnnExtRmsNormalizationForward"
assert rms_plan["passes"][0]["providers"] == ["udondnnExtRmsNormalizationForward"]
rms_host_c = emit_c_host(rms_plan)
assert "udondnnExtRmsNormalizationForward(" in rms_host_c
(output / "rms-normalization-host.c").write_text(rms_host_c, encoding="utf-8")

gather_model = parse_model(
    model_proto(
        graph_proto(
            "standard_embedding_gather",
            [node_proto("Gather", ["embedding", "token_ids"], ["hidden"], "embed")],
            [tensor_proto("embedding", [16, 8], [index * 0.03125 for index in range(128)])],
            [value_info_proto("token_ids", [1, 4], 6)],
            [value_info_proto("hidden", [1, 4, 8])],
        ),
        opset=23,
    )
)
gather_plan = compile_operators(gather_model)
assert gather_plan["execution_provider"] == "UdonOperatorExecutionProvider"
assert gather_plan["nodes"][0]["shape"] == [1, 4, 8]
assert gather_plan["nodes"][0]["provider"] == "udondnnExtGatherForward"
assert gather_plan["aot"]["calls"][0]["function"] == "udondnnExtGatherForward"
gather_host_c = emit_c_host(gather_plan)
assert "udondnnExtGatherForward(" in gather_host_c
assert "int32_t *" in gather_host_c
(output / "gather-host.c").write_text(gather_host_c, encoding="utf-8")

tiny_custom_model_path = output / "tiny-llama-custom.onnx"
tiny_custom_plan_path = output / "tiny-llama-custom-plan.json"
tiny_standard_model_path = output / "tiny-llama-standard.onnx"
tiny_standard_plan_path = output / "tiny-llama-standard-plan.json"
tiny_result = subprocess.run(
    [
        sys.executable,
        root / "examples/tiny-llama/generate_onnx.py",
        "-o",
        tiny_custom_model_path,
        "--plan",
        tiny_custom_plan_path,
        "--standard-output",
        tiny_standard_model_path,
        "--standard-plan",
        tiny_standard_plan_path,
    ],
    text=True,
    capture_output=True,
    timeout=60,
)
assert tiny_result.returncode == 0, (tiny_result.stdout, tiny_result.stderr)
tiny_standard_plan = compile_operators(parse_model(tiny_standard_model_path.read_bytes()))
assert tiny_standard_plan["execution_provider"] == "UdonOperatorExecutionProvider"
assert len(tiny_standard_plan["nodes"]) == 16
assert tiny_standard_plan["nodes"][0]["op_type"] == "Gather"
assert tiny_standard_plan["nodes"][-1]["op_type"] == "Gemm"
assert tiny_standard_plan["outputs"] == {"logits": [1, 16]}
assert {node["op_type"] for node in tiny_standard_plan["nodes"]} == {
    "Gather",
    "Reshape",
    "RMSNormalization",
    "Gemm",
    "Add",
    "Sigmoid",
    "Mul",
}
tiny_standard_host_c = emit_c_host(tiny_standard_plan)
tiny_standard_host_uasm = emit_uasm_host(tiny_standard_plan)
(output / "tiny-llama-standard-host.c").write_text(
    tiny_standard_host_c, encoding="utf-8"
)
(output / "tiny-llama-standard-host.uasm").write_text(
    tiny_standard_host_uasm, encoding="utf-8"
)
generated_sources = [
    output / "generic-host.c",
    output / "tiled-gemm-host.c",
    output / "biased-gemm-host.c",
    output / "rl-policy-host.c",
    output / "attention-matmul-host.c",
    output / "llama-view-host.c",
    output / "rms-normalization-host.c",
    output / "gather-host.c",
    output / "tiny-llama-standard-host.c",
]
syntax_command = [
    c_compiler,
    "-std=c11",
    "-fsyntax-only",
    "-I",
    root / "libraries/udonblas/include",
    "-I",
    root / "libraries/udondnn/include",
    *generated_sources,
]
syntax = subprocess.run(syntax_command, text=True, capture_output=True, timeout=60)
assert syntax.returncode == 0, (syntax.stdout, syntax.stderr)
try:
    parse_model(data[:-1])
except ValueError:
    pass
else:
    raise AssertionError("truncated ONNX protobuf was accepted")

registry = ExecutionProviderRegistry()
registry.register(
    "FixtureProvider",
    lambda candidate: candidate["graph"]["name"] == "mnist_udondnn_udonblas",
    lambda candidate: {"schema_version": 1, "runtime": "fixture"},
)
custom = compile_model(model, registry)
assert custom["execution_provider"] == "FixtureProvider"
assert custom["runtime"] == "fixture"
try:
    registry.register("FixtureProvider", lambda candidate: True, lambda candidate: {})
except ValueError:
    pass
else:
    raise AssertionError("duplicate execution provider name was accepted")
(output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
print("udon_onnx_import=passed nodes=7 providers=udondnn,udonblas passes=3")
