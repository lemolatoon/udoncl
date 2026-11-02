"""Validate the dependency-free ONNX importer and MNIST Udon provider plan."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys


root, output = (Path(value).resolve() for value in sys.argv[1:])
unity_template_value = os.environ.get("UDONCL_ONNX_UNITY_TEMPLATE")
unity_template_directory = (
    Path(unity_template_value).resolve() if unity_template_value else None
)
c_compiler = root / ".deps/llvm23/bin" / (
    "clang.exe" if sys.platform == "win32" else "clang"
)
assert c_compiler.is_file(), f"pinned Clang is missing: {c_compiler}"
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "runtime/onnx"))

from udon_onnx import (
    ExecutionProviderRegistry,
    attribute_float,
    attribute_int,
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
from udon_pages import (
    decode_int4_compact_page,
    emit_int4_call_uasm,
    emit_int4_model_uasm,
    emit_model_uasm,
    encode_int4_page,
    int4_atlas_location,
    linear_texture_layout,
    materialize_int4_call,
    materialize_int4_compact_page,
    materialize_int4_model,
    materialize_model,
    materialize_int4_page,
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
    "udonatenAdd",
    "udonatenRelu",
    "udondnnPoolingForward",
    "view",
    "udonblasSgemm",
    "udonatenAdd",
]
assert [item["fragments"] for item in plan["passes"]] == [9216, 144, 10]
mnist_calls = [call["function"] for call in plan["aot"]["calls"]]
assert mnist_calls[:5] == [
    "udondnnConvolutionForward",
    "udonatenAdd",
    "udonatenRelu",
    "udondnnPoolingForward",
    "view",
]
assert mnist_calls[5:-1] == ["udonblasSgemm"]
assert mnist_calls[-1] == "udonatenAdd"
assert plan["nodes"][0]["shape"] == [1, 16, 24, 24]
assert plan["nodes"][3]["shape"] == [1, 16, 3, 3]
assert plan["nodes"][4]["shape"] == [1, 144]
assert set(plan["initializers"]) == {
    "conv.weight",
    "conv.bias",
    "dense.weight",
    "dense.bias",
}
pooling_call = next(
    call for call in plan["aot"]["calls"] if call["function"] == "udondnnPoolingForward"
)
assert pooling_call["device_profile"]["supported"] is False
assert "udondnn_arena_pooling_forward" in pooling_call["device_profile"]["reason"]

try:
    compile_operators(model)
except NotImplementedError as error:
    assert "udondnn_arena_pooling_forward" in str(error)
    assert "2026-11-02-unity-mnist-averagepool-zero.md" in str(error)
else:
    raise AssertionError("ONNX arena AveragePool must fail explicitly while unresolved")

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
    "udonatenAdd",
    "udonatenRelu",
]
assert [item["fragments"] for item in operator_plan["passes"]] == [16384, 16384, 16384]
host_c = emit_c_host(operator_plan)
assert "udonblasSgemm(" in host_c
assert "udonatenAdd(" in host_c
assert "udonatenRelu(" in host_c
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
assert len(tiled_plan["passes"]) == 1
assert [item["id"] for item in tiled_plan["passes"]] == [200]
assert tiled_plan["passes"][0]["tile"] == {
    "arena_sgemm": True,
    "storage_words": 16384,
    "width": 8192,
    "height": 2,
    "m": 17,
    "n": 18,
    "k": 33,
    "transa": 0,
    "transb": 0,
    "alpha": 1.0,
}
assert tiled_host_c.count("udonblasSgemm(") == 12
assert "buffers->v0_left + 528" in tiled_host_c
assert tiled_host_uasm.count("VRCSDKBaseVRCGraphics.__Blit__") == 1
assert ".export pass200Material" in tiled_host_uasm
(output / "tiled-gemm-host.c").write_text(tiled_host_c, encoding="utf-8")
(output / "tiled-gemm-host.uasm").write_text(tiled_host_uasm, encoding="utf-8")

matrix_vector_model = parse_model(
    model_proto(
        graph_proto(
            "matrix_vector_graph",
            [node_proto("MatMul", ["matrix", "vector"], ["result"], "matvec")],
            [],
            [value_info_proto("matrix", [18, 4]), value_info_proto("vector", [4])],
            [value_info_proto("result", [18])],
        )
    )
)
matrix_vector_plan = compile_model(matrix_vector_model)
assert matrix_vector_plan["nodes"][0]["provider"] == "udonblasSgemv"
assert [call["function"] for call in matrix_vector_plan["aot"]["calls"]] == [
    "udonblasSgemv",
    "udonblasSgemv",
]
assert [item["fragments"] for item in matrix_vector_plan["passes"]] == [16, 2]
assert matrix_vector_plan["aot"]["calls"][1]["tile"] == {
    "batch": 0,
    "trans": "UDONBLAS_OP_T",
    "m": 4,
    "n": 2,
    "lda": 4,
    "matrix_input": 0,
    "matrix_offset": 64,
    "vector_input": 1,
    "vector_offset": 0,
    "output_offset": 16,
    "output_length": 2,
    "alpha": 1.0,
    "beta": 0,
}
matrix_vector_host_c = emit_c_host(matrix_vector_plan)
assert matrix_vector_host_c.count("udonblasSgemv(") == 2
assert "UDONBLAS_OP_T, 4, 16" in matrix_vector_host_c
assert "buffers->v0_matrix + 64, 4" in matrix_vector_host_c

vector_batched_matrix_model = parse_model(
    model_proto(
        graph_proto(
            "vector_batched_matrix_graph",
            [node_proto("MatMul", ["vector", "matrix"], ["result"], "vecmat")],
            [],
            [value_info_proto("vector", [4]), value_info_proto("matrix", [2, 4, 5])],
            [value_info_proto("result", [2, 5])],
        )
    )
)
vector_batched_matrix_plan = compile_model(vector_batched_matrix_model)
assert vector_batched_matrix_plan["nodes"][0]["provider"] == "udonblasSgemv"
assert [call["function"] for call in vector_batched_matrix_plan["aot"]["calls"]] == [
    "udonblasSgemv",
    "udonblasSgemv",
]
assert [item["fragments"] for item in vector_batched_matrix_plan["passes"]] == [5, 5]
assert vector_batched_matrix_plan["aot"]["calls"][1]["tile"]["matrix_offset"] == 20
assert vector_batched_matrix_plan["aot"]["calls"][1]["tile"]["output_offset"] == 5
vector_batched_matrix_host_c = emit_c_host(vector_batched_matrix_plan)
assert vector_batched_matrix_host_c.count("udonblasSgemv(") == 2
assert "UDONBLAS_OP_N, 5, 4" in vector_batched_matrix_host_c
assert "buffers->v0_matrix + 20, 5" in vector_batched_matrix_host_c

vector_dot_model = parse_model(
    model_proto(
        graph_proto(
            "vector_dot_graph",
            [node_proto("MatMul", ["left", "right"], ["result"], "dot")],
            [],
            [value_info_proto("left", [4]), value_info_proto("right", [4])],
            [value_info_proto("result", [])],
        )
    )
)
vector_dot_plan = compile_model(vector_dot_model)
assert vector_dot_plan["nodes"][0]["provider"] == "udonblasSdot"
assert vector_dot_plan["nodes"][0]["shape"] == []
assert [call["function"] for call in vector_dot_plan["aot"]["calls"]] == [
    "udonblasSdot"
]
assert len(vector_dot_plan["passes"]) == 9
assert [item["tile"]["phase"] for item in vector_dot_plan["passes"]] == [
    "product",
    "reduce",
    "reduce",
    "reduce",
    "reduce",
    "reduce",
    "reduce",
    "reduce",
    "reduce",
]
assert [item["fragments"] for item in vector_dot_plan["passes"]] == [
    4,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]
vector_dot_host_c = emit_c_host(vector_dot_plan)
assert vector_dot_host_c.count("udonblasSdot(") == 1
assert "blas, 4, buffers->v0_left + 0, 1" in vector_dot_host_c

oversized_dot_model = parse_model(
    model_proto(
        graph_proto(
            "oversized_vector_dot_graph",
            [node_proto("MatMul", ["left", "right"], ["result"], "dot")],
            [],
            [value_info_proto("left", [257]), value_info_proto("right", [257])],
            [value_info_proto("result", [])],
        )
    )
)
try:
    compile_model(oversized_dot_model)
except ValueError as error:
    assert "SDOT Device profile" in str(error)
else:
    raise AssertionError("oversized vector-vector MatMul must reject the SDOT profile")

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
    "udonatenAdd",
]
assert biased_gemm_plan["aot"]["calls"][0]["tile"]["alpha"] == 0.5
assert biased_gemm_plan["aot"]["calls"][1]["phase"] == "bias"
assert biased_gemm_plan["passes"][1]["phase"] == "bias"
assert biased_gemm_plan["passes"][1]["beta"] == 2.0
biased_gemm_host_c = emit_c_host(biased_gemm_plan)
assert "const float gemm_alpha_0 = 0.5f;" in biased_gemm_host_c
assert "const float gemm_beta_0 = 2.0f;" in biased_gemm_host_c
assert biased_gemm_host_c.count("udonblasSgemm(") == 1
assert biased_gemm_host_c.count("udonatenAdd(") == 1
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

int4_n, int4_k = 70, 160
int4_weight_bytes = int4_n * int4_k // 2
int4_qparam_bytes = (int4_k // 32) * int4_n * 4
int4_external = bytearray(6000 + int4_qparam_bytes)
for index in range(int4_weight_bytes):
    int4_external[32 + index] = (index * 17 + 9) & 255


def bfloat16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<H", bits >> 16)


for group in range(int4_k // 32):
    for row in range(int4_n):
        offset = 6000 + (group * int4_n + row) * 4
        int4_external[offset : offset + 2] = bfloat16(0.5 + group * 0.125)
        int4_external[offset + 2 : offset + 4] = bfloat16((row % 7 - 3) * 0.25)
(output / "int4-weights.bin").write_bytes(int4_external)
int4_model = parse_model(
    model_proto(
        graph_proto(
            "external_weight_int4_mm",
            [
                node_proto(
                    "WeightInt4packMM",
                    ["x", "weight", "weight.scales_and_zeros"],
                    ["y"],
                    "int4_mm",
                    [
                        attribute_int("output_features", int4_n),
                        attribute_int("input_features", int4_k),
                        attribute_int("group_size", 32),
                    ],
                    "com.lemolatoon.udoncl",
                )
            ],
            [
                external_tensor_proto(
                    "weight",
                    [int4_n, int4_k // 2],
                    2,
                    "int4-weights.bin",
                    32,
                    int4_weight_bytes,
                ),
                external_tensor_proto(
                    "weight.scales_and_zeros",
                    [int4_k // 32, int4_n, 2],
                    16,
                    "int4-weights.bin",
                    6000,
                    int4_qparam_bytes,
                ),
            ],
            [value_info_proto("x", [1, int4_k])],
            [value_info_proto("y", [1, int4_n])],
        )
    )
)
int4_plan = compile_operators(int4_model)
int4_call = int4_plan["aot"]["calls"][0]
assert int4_call["function"] == "udonatenWeightInt4packMm"
assert int4_call["tile_grid"]["m_tiles"] == 2
assert int4_call["tile_grid"]["k_tiles"] == 2
assert int4_call["paging"]["packed_weight"] == {
    "tensor": "weight",
    "external": {"location": "int4-weights.bin", "offset": 32, "length": 5600},
    "source_row_stride_bytes": 80,
    "tile_row_bytes": 64,
}
assert int4_call["paging"]["scales_and_zeros"]["source_group_stride_bytes"] == 280
activation = [index * 0.125 for index in range(int4_k)]
accumulator = [index * -0.25 for index in range(int4_n)]
int4_page = materialize_int4_page(
    int4_plan, 0, 1, 1, output, activation, accumulator
)
assert int4_page["n"] == 6
assert int4_page["k"] == 32
assert int4_page["m_offset"] == 64
assert int4_page["k_offset"] == 128
assert int4_page["row_stride"] == 72
assert len(encode_int4_page(int4_page)) == 8192 * 4
first_source = 64 * (int4_k // 2) + 128 // 2
assert int4_page["state"][0] == float((first_source * 17 + 9) & 255)
assert int4_page["state"][16] == 1.0
assert int4_page["state"][17] == -0.5
assert int4_page["state"][6144] == activation[128]
assert int4_page["state"][6144 + 31] == activation[159]
assert int4_page["state"][7168] == accumulator[64]
assert int4_page["state"][7168 + 5] == accumulator[69]
try:
    materialize_int4_page(int4_plan, 0, 2, 0, output)
except ValueError as error:
    assert "outside" in str(error)
else:
    raise AssertionError("out-of-grid INT4 page coordinates were accepted")
int4_compact_page = materialize_int4_compact_page(int4_plan, 0, 1, 1, output)
assert len(int4_compact_page["data"]) == 5120
assert decode_int4_compact_page(int4_compact_page)[:6144] == int4_page["state"][:6144]

int4_call_bundle = output / "int4-call-bundle"
int4_call_manifest = materialize_int4_call(
    int4_plan, 0, output, int4_call_bundle
)
assert int4_call_manifest["artifact_stage"] == "udon-onnx-int4-call"
assert int4_call_manifest["input_features"] == 160
assert int4_call_manifest["output_features"] == 70
assert int4_call_manifest["activation_layout"] == {"width": 160, "height": 1}
assert int4_call_manifest["result_layout"] == {"width": 70, "height": 1}
assert int4_call_manifest["page_encoding"] == {
    "name": "q4-bf16-row80-rgba32",
    "row_bytes": 80,
    "page_bytes": 5120,
    "atlas_width": 1280,
    "max_pages_per_atlas": 8192,
}
assert len(int4_call_manifest["resources"]) == 1
assert {
    (tile["m_tile"], tile["k_tile"], tile["atlas"], tile["row"], tile["n"], tile["k"])
    for tile in int4_call_manifest["tiles"]
} == {
    (0, 0, 0, 0, 64, 128),
    (0, 1, 0, 1, 64, 32),
    (1, 0, 0, 2, 6, 128),
    (1, 1, 0, 3, 6, 32),
}
int4_atlas = int4_call_manifest["resources"][0]
assert int4_atlas["width"] == 1280
assert int4_atlas["height"] == 4
assert int4_atlas["bytes"] == 4 * 5120
assert len(int4_atlas["sha256"]) == 64
assert (int4_call_bundle / int4_atlas["path"]).stat().st_size == 4 * 5120
int4_call_uasm = emit_int4_call_uasm(int4_plan, 0)
assert int4_call_uasm == (int4_call_bundle / "host.uasm").read_text(encoding="utf-8")
assert int4_call_uasm.count("VRCSDKBaseVRCGraphics.__Blit__") == 4
assert int4_call_uasm.count(".export pageAtlas") == 1
assert "__compactPageName: %SystemString, \"_CompactPage\"" in int4_call_uasm
assert "__mLoop:" in int4_call_uasm
assert "__kLoop:" in int4_call_uasm
assert "__atlasMaxPages: %SystemInt32, 8192" in int4_call_uasm
assert "SystemInt32.__op_Addition" in int4_call_uasm
assert "SystemInt32.__op_LessThan" in int4_call_uasm
assert (
    "PUSH, __mTile\n        PUSH, __lastMTile\n        PUSH, __condition\n"
    in int4_call_uasm
)
assert (
    "PUSH, __pageRow\n        PUSH, __one\n        PUSH, __pageRow\n"
    in int4_call_uasm
)
assert (
    "PUSH, accumulatorState\n        PUSH, __previousResult\n        COPY\n"
    in int4_call_uasm
)
assert "__pageRow3" not in int4_call_uasm
assert "PUSH, resultState" in int4_call_uasm
assert len(materialize_int4_call(int4_plan, 0, output, int4_call_bundle)["tiles"]) == 4

assert linear_texture_layout(20000) == {"width": 8192, "height": 3}
assert int4_atlas_location(8191) == {"atlas": 0, "row": 8191}
assert int4_atlas_location(8192) == {"atlas": 1, "row": 0}
try:
    materialize_int4_call(int4_plan, 0, output, output)
except ValueError as error:
    assert "external data directory" in str(error)
else:
    raise AssertionError("INT4 call materializer accepted its external data root as output")
int4_plan_path = output / "int4-plan.json"
int4_plan_path.write_text(json.dumps(int4_plan), encoding="utf-8")
int4_cli_bundle = output / "int4-cli-call-bundle"
int4_cli = subprocess.run(
    [
        sys.executable,
        root / "runtime/onnx/udon_pages.py",
        int4_plan_path,
        output,
        "--call",
        "0",
        "--all",
        "-o",
        int4_cli_bundle,
    ],
    text=True,
    capture_output=True,
    timeout=60,
)
assert int4_cli.returncode == 0, (int4_cli.stdout, int4_cli.stderr)
assert "pages=4" in int4_cli.stdout
assert "atlases=1" in int4_cli.stdout
assert (int4_cli_bundle / "host.uasm").is_file()

int4_model_plan = json.loads(json.dumps(int4_plan))
int4_model_call0 = int4_model_plan["aot"]["calls"][0]
int4_model_call0["output"] = "projection_a"
int4_model_call1 = json.loads(json.dumps(int4_model_call0))
int4_model_call1["index"] = 1
int4_model_call1["output"] = "projection_b"
int4_model_merge = {
    "index": 2,
    "node_index": 2,
    "kind": "library",
    "function": "udonatenAdd",
    "inputs": ["projection_a", "projection_b"],
    "output": "merged",
    "output_shape": [1, int4_n],
    "attributes": {},
}
int4_model_plan["aot"]["calls"] = [
    int4_model_call0,
    int4_model_call1,
    int4_model_merge,
]
int4_model_plan["passes"] = [
    {
        **int4_model_plan["passes"][0],
        "call_index": 0,
        "output": "projection_a",
    },
    {
        **int4_model_plan["passes"][0],
        "id": 201,
        "call_index": 1,
        "output": "projection_b",
    },
    {
        "id": 202,
        "kernel": "onnx_2_add",
        "providers": ["udonatenAdd"],
        "fusion": None,
        "call_index": 2,
        "inputs": ["projection_a", "projection_b"],
        "output": "merged",
        "shape": [1, int4_n],
        "fragments": int4_n,
    },
]
int4_model_plan["outputs"] = {"merged": [1, int4_n]}
int4_model_plan["texture_arena"] = {
    "abi": "linear-rgba32-words",
    "max_texture_width": 8192,
    "slot_count": 3,
    "total_words": int4_k + int4_n * 2,
    "total_bytes": (int4_k + int4_n * 2) * 4,
    "slots": [
        {
            "index": 0,
            "capacity_words": int4_k,
            "values": ["x", "merged"],
            "width": int4_k,
            "height": 1,
        },
        {
            "index": 1,
            "capacity_words": int4_n,
            "values": ["projection_a"],
            "width": int4_n,
            "height": 1,
        },
        {
            "index": 2,
            "capacity_words": int4_n,
            "values": ["projection_b"],
            "width": int4_n,
            "height": 1,
        },
    ],
    "bindings": {
        "x": {
            "slot": 0,
            "offset_words": 0,
            "words": int4_k,
            "data_type": "float32",
            "first_call": -1,
            "last_call": 1,
        },
        "projection_a": {
            "slot": 1,
            "offset_words": 0,
            "words": int4_n,
            "data_type": "float32",
            "first_call": 0,
            "last_call": 2,
        },
        "projection_b": {
            "slot": 2,
            "offset_words": 0,
            "words": int4_n,
            "data_type": "float32",
            "first_call": 1,
            "last_call": 2,
        },
        "merged": {
            "slot": 0,
            "offset_words": 0,
            "words": int4_n,
            "data_type": "float32",
            "first_call": 2,
            "last_call": 3,
        },
    },
}
int4_model_bundle = output / "int4-model-bundle"
int4_model_manifest = materialize_int4_model(
    int4_model_plan, output, int4_model_bundle
)
assert int4_model_manifest["artifact_stage"] == "udon-onnx-int4-model"
assert int4_model_manifest["aot_call_count"] == 3
assert int4_model_manifest["int4_call_count"] == 2
assert int4_model_manifest["int4_atlas_count"] == 2
assert int4_model_manifest["int4_page_count"] == 8
assert int4_model_manifest["int4_atlas_bytes"] == 8 * 5120
assert int4_model_manifest["complete_graph"] is False
assert int4_model_manifest["assembly"] == "host.uasm"
assert int4_model_manifest["host_schedule"] == {
    "assembly": "host.uasm",
    "arena_slots": 3,
    "int4_runners": 2,
    "direct_device_passes": 1,
    "static_initializer_textures": 0,
    "zero_copy_views": 0,
}
assert int4_model_manifest["static_inputs"] == []
assert [item["call_index"] for item in int4_model_manifest["call_artifacts"]] == [0, 1]
assert all(item["atlases"] == 1 for item in int4_model_manifest["call_artifacts"])
assert int4_model_manifest["unmaterialized_calls"] == [
    {
        "index": 2,
        "kind": "library",
        "function": "udonatenAdd",
        "output": "merged",
    }
]
assert (int4_model_bundle / "calls/call-0000/host.uasm").is_file()
assert (int4_model_bundle / "calls/call-0001/host.uasm").is_file()
assert (int4_model_bundle / "host.uasm").is_file()
assert all(
    item["arena_binding"]["activation_slot"] == 0
    for item in int4_model_manifest["call_artifacts"]
)
assert len(int4_model_manifest["execution_plan_sha256"]) == 64
unity_int4_plan = json.loads(json.dumps(int4_model_plan))
unity_int4_plan["aot"]["calls"] = [unity_int4_plan["aot"]["calls"][0]]
unity_int4_plan["passes"] = [unity_int4_plan["passes"][0]]
unity_int4_plan["outputs"] = {"projection_a": [1, int4_n]}
unity_int4_bundle = output / "unity-int4-model-bundle"
unity_int4_manifest = None
if unity_template_directory is not None:
    unity_int4_manifest = materialize_int4_model(
        unity_int4_plan,
        output,
        unity_int4_bundle,
        unity_template_directory=unity_template_directory,
        model_path=model_path,
        program_name="onnx_int4_fixture",
    )
unity_int4_activation = [index * 0.125 for index in range(int4_k)]
unity_int4_reference = []
for row in range(int4_n):
    total = 0.0
    for inner in range(int4_k):
        packed = int4_external[32 + row * (int4_k // 2) + inner // 2]
        nibble = packed >> 4 if inner % 2 else packed & 15
        quantized = nibble - 16 if nibble >= 8 else nibble
        group = inner // 32
        qparam_offset = 6000 + (group * int4_n + row) * 4
        scale_bits, zero_bits = struct.unpack_from("<2H", int4_external, qparam_offset)
        scale = struct.unpack("<f", struct.pack("<I", scale_bits << 16))[0]
        zero = struct.unpack("<f", struct.pack("<I", zero_bits << 16))[0]
        total += (quantized * scale + zero) * unity_int4_activation[inner]
    unity_int4_reference.append(total)
(output / "unity-int4-activation.bytes").write_bytes(
    struct.pack(f"<{int4_k}f", *unity_int4_activation)
)
(output / "unity-int4-reference.bytes").write_bytes(
    struct.pack(f"<{int4_n}f", *unity_int4_reference)
)
if unity_int4_manifest is not None:
    assert unity_int4_manifest["artifact_stage"] == "unity-udon-onnx-model"
    assert unity_int4_manifest["ready_for_unity"] is True
    assert unity_int4_manifest["complete_graph"] is True
    assert len(unity_int4_manifest["call_artifacts"]) == 1
    assert unity_int4_manifest["model_inputs"] == [
        {
            "index": 0,
            "name": "x",
            "data_type": "float32",
            "words": int4_k,
            "slot": 0,
        }
    ]
    assert unity_int4_manifest["model_outputs"] == [
        {
            "index": 0,
            "name": "projection_a",
            "data_type": "float32",
            "words": int4_n,
            "slot": 1,
        }
    ]
    assert unity_int4_manifest["program"]["buffer_copy_shader"] == "buffer-copy.shader"
    assert (unity_int4_bundle / "buffer-copy.shader").is_file()
    assert (unity_int4_bundle / "model.onnx.bytes").is_file()
    assert (unity_int4_bundle / "execution-plan.json.bytes").is_file()
    assert (unity_int4_bundle / "kernels/kernel_208.shader").is_file()

arena_add_bytes = model_proto(
    graph_proto(
        "arena_add",
        [node_proto("Add", ["left", "right"], ["sum"], "add")],
        [],
        [
            value_info_proto("left", [1, 16384]),
            value_info_proto("right", [1, 16384]),
        ],
        [value_info_proto("sum", [1, 16384])],
    )
)
arena_add_path = output / "arena-add.onnx"
arena_add_path.write_bytes(arena_add_bytes)
arena_add_plan = compile_operators(parse_model(arena_add_bytes))
assert arena_add_plan["execution_provider"] == "UdonOperatorExecutionProvider"
assert arena_add_plan["passes"][0]["shader"] == "kernel_215.shader"
assert emit_model_uasm(arena_add_plan) == emit_int4_model_uasm(arena_add_plan)
arena_add_uasm = emit_model_uasm(arena_add_plan)
assert "PUSH, __zero\n        PUSH, __passIndex\n        COPY" in arena_add_uasm
assert "PUSH, arenaSlot2\n        PUSH, resultState\n        COPY" in arena_add_uasm
if unity_template_directory is not None:
    arena_add_bundle = output / "unity-arena-add-model-bundle"
    arena_add_manifest = materialize_model(
        arena_add_plan,
        output,
        arena_add_bundle,
        unity_template_directory=unity_template_directory,
        model_path=arena_add_path,
        program_name="onnx_arena_add_fixture",
    )
    assert arena_add_manifest["artifact_stage"] == "unity-udon-onnx-model"
    assert arena_add_manifest["complete_graph"] is True
    assert arena_add_manifest["int4_call_count"] == 0
    assert arena_add_manifest["call_artifacts"] == []
    assert arena_add_manifest["unmaterialized_calls"] == []
    assert arena_add_manifest["model_inputs"] == [
        {
            "index": 0,
            "name": "left",
            "data_type": "float32",
            "words": 16384,
            "slot": 0,
        },
        {
            "index": 1,
            "name": "right",
            "data_type": "float32",
            "words": 16384,
            "slot": 1,
        },
    ]
    assert arena_add_manifest["model_outputs"] == [
        {
            "index": 0,
            "name": "sum",
            "data_type": "float32",
            "words": 16384,
            "slot": 2,
        }
    ]
    assert arena_add_manifest["device"]["kernels"] == [
        {
            "id": 215,
            "name": "udonaten_arena_pointwise",
            "shader": "kernels/kernel_215.shader",
        }
    ]
    assert (arena_add_bundle / "host.uasm").is_file()
    assert (arena_add_bundle / "model.onnx.bytes").read_bytes() == arena_add_bytes
    assert (arena_add_bundle / "execution-plan.json.bytes").is_file()
    assert (arena_add_bundle / "kernels/kernel_215.shader").is_file()
    assert (arena_add_bundle / "buffer-copy.shader").is_file()
    arena_add_plan_path = output / "arena-add-plan.json"
    arena_add_plan_path.write_text(json.dumps(arena_add_plan), encoding="utf-8")
    arena_add_cli_bundle = output / "arena-add-cli-model-bundle"
    arena_add_cli = subprocess.run(
        [
            sys.executable,
            root / "runtime/onnx/udon_pages.py",
            arena_add_plan_path,
            output,
            "--model",
            "--unity-template-directory",
            unity_template_directory,
            "--model-path",
            arena_add_path,
            "--program-name",
            "onnx_arena_add_cli_fixture",
            "-o",
            arena_add_cli_bundle,
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert arena_add_cli.returncode == 0, (arena_add_cli.stdout, arena_add_cli.stderr)
    assert "calls=1 complete=True" in arena_add_cli.stdout
    assert (arena_add_cli_bundle / "kernels/kernel_215.shader").is_file()
int4_model_plan_path = output / "int4-model-plan.json"
int4_model_plan_path.write_text(json.dumps(int4_model_plan), encoding="utf-8")
int4_model_cli_bundle = output / "int4-model-cli-bundle"
int4_model_cli = subprocess.run(
    [
        sys.executable,
        root / "runtime/onnx/udon_pages.py",
        int4_model_plan_path,
        output,
        "--all-calls",
        "-o",
        int4_model_cli_bundle,
    ],
    text=True,
    capture_output=True,
    timeout=60,
)
assert int4_model_cli.returncode == 0, (
    int4_model_cli.stdout,
    int4_model_cli.stderr,
)
assert "calls=2 complete=False" in int4_model_cli.stdout
assert (int4_model_cli_bundle / "execution-plan.json").is_file()

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
assert rl_plan["texture_arena"]["slot_count"] == 3
assert rl_plan["texture_arena"]["total_words"] == 24
assert [
    slot["capacity_words"] for slot in rl_plan["texture_arena"]["slots"]
] == [8, 8, 8]
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
    "udonatenAdd",
    "udonatenTanh",
    "udonatenSigmoid",
    "udonatenMul",
    "udonblasSgemm",
    "udonatenAdd",
    "udonatenSub",
    "udonatenMul",
    "udondnnSoftmaxForward",
]
assert len(rl_plan["passes"]) == 10
assert {item["shader"] for item in rl_plan["passes"]} == {
    "kernel_215.shader",
    "kernel_221.shader",
    "kernel_222.shader",
}
rl_unity_bundle = output / "unity-rl-policy-model-bundle"
if unity_template_directory is not None:
    rl_unity_manifest = materialize_model(
        rl_plan,
        output,
        rl_unity_bundle,
        unity_template_directory=unity_template_directory,
        model_path=rl_model_path,
        program_name="onnx_rl_policy_fixture",
    )
    assert rl_unity_manifest["artifact_stage"] == "unity-udon-onnx-model"
    assert rl_unity_manifest["complete_graph"] is True
    assert rl_unity_manifest["unmaterialized_calls"] == []
    assert len(rl_unity_manifest["static_inputs"]) == 6
    assert {item["id"] for item in rl_unity_manifest["device"]["kernels"]} == {
        215,
        221,
        222,
    }
    assert (rl_unity_bundle / "kernels/kernel_221.shader").is_file()
    assert (rl_unity_bundle / "kernels/kernel_222.shader").is_file()
rl_host_c = emit_c_host(rl_plan)
assert "udonatenTanh(" in rl_host_c
assert "udonatenSigmoid(" in rl_host_c
assert rl_host_c.count("udonatenMul(") == 2
assert "udonatenSub(" in rl_host_c
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

tile_model = parse_model(
    model_proto(
        graph_proto(
            "standard_tile",
            [node_proto("Tile", ["x", "repeats"], ["y"], "tile")],
            [tensor_int64_proto("repeats", [2], [2, 3])],
            [value_info_proto("x", [2, 3])],
            [value_info_proto("y", [4, 9])],
        )
    )
)
tile_plan = compile_operators(tile_model)
assert tile_plan["nodes"][0]["provider"] == "udonatenTile"
assert tile_plan["nodes"][0]["shape"] == [4, 9]
assert tile_plan["nodes"][0]["attributes"]["repeats"]["ints"] == [2, 3]
assert tile_plan["aot"]["calls"][0]["function"] == "udonatenTile"
tile_host_c = emit_c_host(tile_plan)
assert "udonatenTile(" in tile_host_c
assert "(const int64_t[]){2, 3}, 2" in tile_host_c
(output / "tile-host.c").write_text(tile_host_c, encoding="utf-8")

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
assert rms_plan["aot"]["calls"][0]["function"] == "udonatenRmsNorm"
assert rms_plan["passes"][0]["providers"] == ["udonatenRmsNorm"]
rms_host_c = emit_c_host(rms_plan)
assert "udonatenRmsNorm(" in rms_host_c
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
assert gather_plan["nodes"][0]["provider"] == "udonatenEmbedding"
assert gather_plan["aot"]["calls"][0]["function"] == "udonatenEmbedding"
gather_host_c = emit_c_host(gather_plan)
assert "udonatenEmbedding(" in gather_host_c
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
    output / "tile-host.c",
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
    "-I",
    root / "libraries/udonaten/include",
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
print("udon_onnx_import=passed nodes=7 providers=udondnn,udonblas,udonaten passes=3")
