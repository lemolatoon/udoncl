"""Validate the Llama 3.1 8B ONNX external-weight execution plan."""

import json
import math
from pathlib import Path
import struct
import subprocess
import sys


root, output = (Path(value).resolve() for value in sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
model = output / "llama-8b.onnx"
plan_path = output / "onnx-plan.json"
operator_model = output / "llama-8b-operators.onnx"
operator_plan_path = output / "operator-plan.json"
result = subprocess.run(
    [
        sys.executable,
        root / "examples/llama-8b-onnx/generate_onnx.py",
        "-o",
        model,
        "--plan",
        plan_path,
        "--operator-output",
        operator_model,
        "--operator-plan",
        operator_plan_path,
    ],
    text=True,
    capture_output=True,
    timeout=60,
)
assert result.returncode == 0, (result.stdout, result.stderr)
plan = json.loads(plan_path.read_text(encoding="utf-8"))
assert plan["runtime"] == "udon-onnx"
assert plan["profile"] == "llama-3.1-8b-q4"
assert plan["execution_provider"] == "UdonLlamaExecutionProvider"
assert plan["config"]["num_hidden_layers"] == 32
assert plan["config"]["hidden_size"] == 4096
assert plan["config"]["vocab_size"] == 128256
assert plan["weights"]["parameters"] == 8030261248
assert plan["weights"]["storage_bytes"] == 5019279360
assert plan["weights"]["tensors"] == 517
assert len(plan["passes"]) == 162
assert [item["id"] for item in plan["passes"]] == list(range(1000, 1162))
assert plan["passes"][0]["kernel"] == "llama_embedding_gather"
assert plan["passes"][-1]["kernel"] == "llama_logits"
assert {item["layer"] for item in plan["passes"][1:-1]} == set(range(32))
providers = {
    provider for device_pass in plan["passes"] for provider in device_pass["providers"]
}
assert "udonatenWeightInt4packMm" in providers
assert "udonblasSgemm" in providers
assert "udonatenRmsNorm" in providers
assert "udondnnSoftmaxForward" in providers
assert "udonatenAdd" in providers
assert "udonatenMul" in providers
assert "udonatenSigmoid" in providers
assert plan["blas"] == {
    "weight_provider": "udonatenWeightInt4packMm",
    "activation_provider": "udonblasSgemm",
    "q4_tile": {
        "m": 64,
        "k": 128,
        "block_values": 32,
        "qparams": "[K/group_size,N,2] bfloat16",
        "accumulator": "float32",
    },
}
matmuls = [
    matmul
    for device_pass in plan["passes"]
    for matmul in device_pass.get("matmuls", [])
]
assert len(matmuls) == 225
assert all(matmul["provider"] == "udonatenWeightInt4packMm" for matmul in matmuls)
assert all(matmul["weight_format"] == "q4_int4pack" for matmul in matmuls)
assert all(matmul["group_size"] == 32 for matmul in matmuls)
assert all(matmul["scales_and_zeros"] in plan["initializers"] for matmul in matmuls)
assert {matmul["weight"] for matmul in matmuls} <= set(plan["initializers"])
q_projection = plan["passes"][1]["matmuls"][0]
assert q_projection["weight"] == "model.layers.0.self_attn.q_proj.weight"
assert (q_projection["m_tiles"], q_projection["k_tiles"]) == (64, 32)
logits = plan["passes"][-1]["matmuls"][0]
assert logits["weight"] == "lm_head.weight"
assert (logits["m_tiles"], logits["k_tiles"]) == (2004, 32)

operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
assert operator_plan["execution_provider"] == "UdonOperatorExecutionProvider"
assert operator_plan["graph"] == "llama_3_1_8b_operator_decoder_single_token"
assert operator_plan["inputs"] == {"hidden_state": [1, 4096]}
assert operator_plan["outputs"] == {"logits": [1, 128256]}
assert len(operator_plan["nodes"]) == 546
assert all(node["op_type"] != "LlamaDecoder" for node in operator_plan["nodes"])
pointwise_calls = [
    call
    for call in operator_plan["aot"]["calls"]
    if call["function"] in {"udonatenAdd", "udonatenMul", "udonatenSigmoid"}
]
assert len(pointwise_calls) == 160
assert all(call["device_profile"]["supported"] is True for call in pointwise_calls)
assert all(
    call["device_profile"]["kernel"] == "udonaten_arena_pointwise"
    and call["device_profile"]["shader"] == "kernel_215.shader"
    and call["device_profile"]["storage_words"] == 16384
    and call["device_profile"]["width"] == 8192
    and call["device_profile"]["height"] == 2
    for call in pointwise_calls
)
assert {call["device_profile"]["operation"] for call in pointwise_calls} == {
    0,
    1,
    2,
}
pointwise_passes = [
    device_pass
    for device_pass in operator_plan["passes"]
    if device_pass["providers"][0]
    in {"udonatenAdd", "udonatenMul", "udonatenSigmoid"}
]
assert len(pointwise_passes) == 160
assert all(
    device_pass["kernel"] == "udonaten_arena_pointwise"
    and device_pass["shader"] == "kernel_215.shader"
    and device_pass["fragments"] == 16384
    and len(device_pass["inputs"]) == 2
    and device_pass["material_ints"]["_VRC_Arg3"]
    == device_pass["tile"]["arena_words"]
    and device_pass["material_ints"]["_VRC_Arg4"]
    == device_pass["tile"]["input_words"][0]
    and device_pass["material_ints"]["_VRC_Arg5"]
    == device_pass["tile"]["input_words"][-1]
    and device_pass["material_ints"]["_VRC_Arg6"]
    == device_pass["tile"]["operation"]
    for device_pass in pointwise_passes
)
assert {device_pass["tile"]["arena_words"] for device_pass in pointwise_passes} == {
    4096,
    14336,
}
rms_calls = [
    call
    for call in operator_plan["aot"]["calls"]
    if call["function"] == "udonatenRmsNorm"
]
assert len(rms_calls) == 65
assert all(
    call["device_profile"]
    == {
        "supported": True,
        "state_words": 16384,
        "passes": 2,
        "reduction_chunk_words": 64,
        "kernels": [
            "udonaten_arena_rms_norm_reduce",
            "udonaten_arena_rms_norm_finalize",
        ],
        "shaders": ["kernel_217.shader", "kernel_218.shader"],
        "weight_input": "_VRC_Input2",
    }
    for call in rms_calls
)
rms_passes = [
    device_pass
    for device_pass in operator_plan["passes"]
    if device_pass["providers"] == ["udonatenRmsNorm"]
]
assert len(rms_passes) == 130
assert {device_pass["phase"] for device_pass in rms_passes} == {
    "reduce",
    "finalize",
}
assert all(device_pass["fragments"] == 16384 for device_pass in rms_passes)
assert all(device_pass["state_words"] == 16384 for device_pass in rms_passes)
rms_finalize_passes = [
    device_pass for device_pass in rms_passes if device_pass["phase"] == "finalize"
]
assert all(
    device_pass["kernel"] == "udonaten_arena_rms_norm_finalize"
    and device_pass["shader"] == "kernel_218.shader"
    and device_pass["static_inputs"]
    == [
        {
            "property": "_VRC_Input2",
            "initializer": device_pass["inputs"][2],
        }
    ]
    and device_pass["material_floats"] == {"_VRC_Arg6": 1.0e-5}
    for device_pass in rms_finalize_passes
)
assert len(operator_plan["scratch_buffers"]) == 65
assert all(
    scratch["words"] == 16384 and scratch["data_type"] == "float32"
    for scratch in operator_plan["scratch_buffers"]
)
assert len(operator_plan["passes"]) == 547
host_material_passes = [
    device_pass
    for device_pass in operator_plan["passes"]
    if device_pass["host_material_index"] >= 0
]
assert len(host_material_passes) == 322
assert [item["host_material_index"] for item in host_material_passes] == list(
    range(322)
)
assert sum(item["host_material_index"] < 0 for item in operator_plan["passes"]) == 225
texture_arena = operator_plan["texture_arena"]
assert texture_arena["abi"] == "linear-rgba32-words"
assert texture_arena["max_texture_width"] == 8192
assert texture_arena["slot_count"] == 5
assert texture_arena["total_words"] == 177408
assert texture_arena["total_bytes"] == 709632
assert operator_plan["passes"][0]["material_int_bindings"] == [
    {"property": "_VRC_Arg2", "value": 4096},
    {"property": "_VRC_Arg3", "value": 4096},
]
assert operator_plan["passes"][1]["material_float_bindings"] == [
    {"property": "_VRC_Arg6", "value": 1.0e-5}
]
assert [slot["capacity_words"] for slot in texture_arena["slots"]] == [
    128256,
    16384,
    4096,
    14336,
    14336,
]
assert [
    (slot["width"], slot["height"]) for slot in texture_arena["slots"]
] == [(8192, 16), (8192, 2), (4096, 1), (8192, 2), (8192, 2)]
assert texture_arena["bindings"]["hidden_state"] == {
    "slot": 0,
    "offset_words": 0,
    "words": 4096,
    "data_type": "float32",
    "first_call": -1,
    "last_call": 8,
}
assert texture_arena["bindings"]["layer.0.value_heads"]["alias_of"] == (
    "layer.0.value"
)
assert texture_arena["bindings"]["layer.0.value_heads"]["slot"] == (
    texture_arena["bindings"]["layer.0.value"]["slot"]
)
assert texture_arena["bindings"]["logits"]["slot"] == 0
assert texture_arena["bindings"]["logits"]["last_call"] == 546
for device_pass in operator_plan["passes"]:
    assert device_pass["output"] in texture_arena["bindings"]
    assert all(
        name in texture_arena["bindings"] or name in operator_plan["initializers"]
        for name in device_pass.get("inputs", [])
    )
for slot in texture_arena["slots"]:
    intervals = sorted(
        (
            texture_arena["bindings"][name]["first_call"],
            texture_arena["bindings"][name]["last_call"],
            name,
        )
        for name in slot["values"]
    )
    assert all(left[1] < right[0] for left, right in zip(intervals, intervals[1:]))
operator_q4_calls = [
    call
    for call in operator_plan["aot"]["calls"]
    if call["function"] == "udonatenWeightInt4packMm"
]
assert len(operator_q4_calls) == 225
assert all(call["kind"] == "tiled_library" for call in operator_q4_calls)
assert all(call["weight_format"] == "q4_int4pack" for call in operator_q4_calls)
assert all(call["group_size"] == 32 for call in operator_q4_calls)
assert operator_q4_calls[0]["tile_grid"] == {
    "m": 64,
    "k": 128,
    "m_tiles": 64,
    "k_tiles": 32,
    "accumulation": "first K tile replaces; later K tiles add internally",
}
assert operator_q4_calls[-1]["tile_grid"]["m_tiles"] == 2004
assert operator_q4_calls[-1]["tile_grid"]["k_tiles"] == 32
assert operator_q4_calls[0]["paging"]["state_words"] == 8192
assert operator_q4_calls[0]["paging"]["input_offset_words"] == 6144
assert operator_q4_calls[0]["paging"]["result_offset_words"] == 7168
assert operator_q4_calls[0]["paging"]["row_stride_words"] == 72
assert operator_q4_calls[0]["paging"]["accumulate_argument"] == 5
assert operator_q4_calls[0]["paging"]["packed_weight"]["source_row_stride_bytes"] == 2048
assert (
    operator_q4_calls[0]["paging"]["scales_and_zeros"][
        "source_group_stride_bytes"
    ]
    == 16384
)
assert (
    operator_q4_calls[-1]["paging"]["scales_and_zeros"][
        "source_group_stride_bytes"
    ]
    == 513024
)
assert sum(node["op_type"] == "Tile" for node in operator_plan["nodes"]) == 32
tile_calls = [
    call for call in operator_plan["aot"]["calls"] if call["function"] == "udonatenTile"
]
assert len(tile_calls) == 32
assert all(math.prod(call["output_shape"]) <= 4096 for call in tile_calls)
assert all(call["attributes"]["repeats"]["ints"] == [1, 1, 4, 1] for call in tile_calls)
assert all(
    call["device_profile"]
    == {
        "supported": True,
        "kernel": "udonaten_arena_tile",
        "shader": "kernel_216.shader",
        "storage_words": 16384,
        "width": 8192,
        "height": 2,
    }
    for call in tile_calls
)
tile_passes = [
    device_pass
    for device_pass in operator_plan["passes"]
    if device_pass["providers"] == ["udonatenTile"]
]
assert len(tile_passes) == 32
assert all(
    device_pass["kernel"] == "udonaten_arena_tile"
    and device_pass["shader"] == "kernel_216.shader"
    and device_pass["inputs"] == [
        operator_plan["aot"]["calls"][device_pass["call_index"]]["inputs"][0]
    ]
    and device_pass["fragments"] == 16384
    and device_pass["tile"]["input_shape"] == [1, 8, 1, 128]
    and device_pass["tile"]["repeats"] == [1, 1, 4, 1]
    and device_pass["material_ints"]["_VRC_Arg10"] == 4096
    for device_pass in tile_passes
)

sys.path.insert(0, str(root / "runtime/onnx"))
from udon_onnx import (  # noqa: E402
    emit_c_host,
    external_tensor_proto,
    parse_model,
    validate_external_data,
)
from udon_pages import (  # noqa: E402
    emit_int4_call_uasm,
    emit_int4_model_uasm,
    materialize_static_inputs,
)
from udon_pages import plan_int4_model_bundle  # noqa: E402

static_source = output / "static-input-source"
static_source.mkdir(exist_ok=True)
(static_source / "weights.bin").write_bytes(struct.pack("<2H", 0x3F80, 0xC000))
static_output = output / "static-input-artifact"
static_artifacts = materialize_static_inputs(
    {
        "passes": [
            {
                "static_inputs": [
                    {"property": "_VRC_Input2", "initializer": "scale"}
                ]
            }
        ],
        "initializers": {
            "scale": {
                "shape": [2],
                "data_type": "bfloat16",
                "bytes": 4,
                "storage": "external",
                "external": {
                    "location": "weights.bin",
                    "offset": 0,
                    "length": 4,
                },
            }
        },
    },
    static_source,
    static_output,
)
assert len(static_artifacts) == 1
assert static_artifacts[0]["bytes"] == 8
assert (static_output / static_artifacts[0]["resource"]).read_bytes() == struct.pack(
    "<2f", 1.0, -2.0
)

int4_bundle_plan = plan_int4_model_bundle(operator_plan)
assert int4_bundle_plan["aot_call_count"] == len(operator_plan["aot"]["calls"])
assert int4_bundle_plan["int4_call_count"] == 225
assert int4_bundle_plan["complete_graph"] is False
assert int4_bundle_plan["int4_page_count"] == sum(
    call["tile_grid"]["m_tiles"] * call["tile_grid"]["k_tiles"]
    for call in operator_q4_calls
)
assert int4_bundle_plan["int4_atlas_bytes"] == (
    int4_bundle_plan["int4_page_count"] * 5120
)
assert int4_bundle_plan["int4_page_count"] == 916096
assert int4_bundle_plan["int4_atlas_count"] == 232
assert int4_bundle_plan["int4_atlas_bytes"] == 4690411520
assert int4_bundle_plan["texture_arena"] == {
    "abi": "linear-rgba32-words",
    "slot_count": 5,
    "total_words": 177408,
    "total_bytes": 709632,
    "slots": [
        {"index": 0, "capacity_words": 128256, "width": 8192, "height": 16},
        {"index": 1, "capacity_words": 16384, "width": 8192, "height": 2},
        {"index": 2, "capacity_words": 4096, "width": 4096, "height": 1},
        {"index": 3, "capacity_words": 14336, "width": 8192, "height": 2},
        {"index": 4, "capacity_words": 14336, "width": 8192, "height": 2},
    ],
}
assert int4_bundle_plan["host_schedule"] == {
    "assembly": "host.uasm",
    "arena_slots": 5,
    "int4_runners": 225,
    "direct_device_passes": 322,
    "static_initializer_textures": 65,
    "zero_copy_views": 64,
}
assert len(int4_bundle_plan["static_inputs"]) == 65
assert all(
    item["properties"] == ["_VRC_Input2"]
    and item["shape"] == [4096]
    and item["data_type"] == "bfloat16"
    and item["source_bytes"] == 8192
    and item["storage"] == "external"
    and item["texture"]
    == {
        "abi": "linear-rgba32-words",
        "words": 4096,
        "width": 4096,
        "height": 1,
    }
    for item in int4_bundle_plan["static_inputs"]
)
assert int4_bundle_plan["call_artifacts"][-1]["pages"] == 2004 * 32
assert int4_bundle_plan["call_artifacts"][-1]["atlases"] == 8
logits_uasm = emit_int4_call_uasm(operator_plan, operator_q4_calls[-1]["index"])
assert len(logits_uasm) < 50000
assert logits_uasm.count(".export pageAtlas") == 8
assert logits_uasm.count("VRCSDKBaseVRCGraphics.__Blit__") == 4
assert "__mTiles: %SystemInt32, 2004" in logits_uasm
assert "__kTiles: %SystemInt32, 32" in logits_uasm
key_uasm = emit_int4_call_uasm(
    operator_plan,
    operator_q4_calls[1]["index"],
    activation_storage_width=8192,
    result_storage_width=8192,
)
assert "__activationWidth: %SystemInt32, 8192" in key_uasm
assert "__outputWidth: %SystemInt32, 8192" in key_uasm
model_uasm = emit_int4_model_uasm(operator_plan)
assert len(model_uasm) < 700000
assert model_uasm.count(".export arenaSlot") == 5
assert ".export int4Runners" in model_uasm
assert ".export passMaterials" in model_uasm
assert model_uasm.count("IUdonEventReceiverArray.__Get") == 225
assert model_uasm.count("MaterialArray.__Get") == 322
assert model_uasm.count("__SetProgramVariable__") == 450
assert model_uasm.count("__SendCustomEvent__") == 225
assert model_uasm.count("VRCSDKBaseVRCGraphics.__Blit__") == 322
assert model_uasm.count("UnityEngineMaterial.__SetInt__") == 1253
assert model_uasm.count("UnityEngineMaterial.__SetFloat__") == 65
assert "__float0: %SystemSingle, 0.00001" in model_uasm
assert "PUSH, __zero\n        PUSH, __runnerIndex\n        COPY" in model_uasm
assert "PUSH, __zero\n        PUSH, __passIndex\n        COPY" in model_uasm
assert "PUSH, arenaSlot0\n        PUSH, resultState\n        COPY" in model_uasm
(output / "model-host.uasm").write_text(model_uasm, encoding="utf-8")

try:
    emit_c_host(operator_plan)
except ValueError as error:
    assert "external-weight paging" in str(error)
else:
    raise AssertionError("8B Q4 operator plan was emitted without a paging Host")

try:
    external_tensor_proto("bad", [1], 2, "../escape.bin", 0, 1)
except ValueError:
    pass
else:
    raise AssertionError("unsafe ONNX external_data location was accepted")

parsed = parse_model(model.read_bytes())
locations = {
    tensor["external"]["location"]
    for tensor in parsed["graph"]["initializers"].values()
}
assert locations == {"weights.q4"}
try:
    validate_external_data(parsed, output)
except ValueError as error:
    assert "does not exist" in str(error)
else:
    raise AssertionError("missing 8B external weights were accepted as executable")
print(result.stdout, end="")
print("llama_8b_onnx_plan=passed providers=udondnn,udonblas,udonaten external_data=safe")
