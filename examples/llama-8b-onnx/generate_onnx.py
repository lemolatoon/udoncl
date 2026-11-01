"""Generate the canonical Llama 3.1 8B Udon ONNX descriptor.

The protobuf is small because trained weights remain ONNX external data.  The
separate JSON index is suitable for a licensed checkpoint converter to fill.
"""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime/onnx"))

from udon_onnx import (  # noqa: E402
    LLAMA_DOMAIN,
    attribute_int,
    attribute_string,
    compile_model,
    compile_operators,
    external_tensor_proto,
    graph_proto,
    llama_weight_spec,
    model_proto,
    node_proto,
    parse_model,
    tensor_int64_proto,
    value_info_proto,
)


CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "context_length": 1,
    "rope_theta": 500000,
    "q4_group_size": 32,
}


def align(value: int, alignment: int = 65536) -> int:
    return (value + alignment - 1) // alignment * alignment


def build_weight_tensors():
    tensors = []
    index = []
    offset = 0
    spec = llama_weight_spec(CONFIG)
    for item in spec:
        offset = align(offset)
        data_type = 16 if item["format"] == "bf16" else 2
        storage_shape = (
            item["shape"]
            if data_type == 16
            else [item["shape"][0], item["shape"][1] // 2]
        )
        storage_bytes = item.get("packed_storage_bytes", item["storage_bytes"])
        tensors.append(
            external_tensor_proto(
                item["name"],
                storage_shape,
                data_type,
                "weights.q4",
                offset,
                storage_bytes,
            )
        )
        index.append(
            {
                "name": item["name"],
                "logical_shape": item["shape"],
                "parameters": item["parameters"],
                "format": item["format"],
                "location": "weights.q4",
                "offset": offset,
                "length": storage_bytes,
            }
        )
        offset += storage_bytes
        if item["format"] == "q4_int4pack":
            offset = align(offset)
            tensors.append(
                external_tensor_proto(
                    item["scales_and_zeros_name"],
                    item["scales_and_zeros_shape"],
                    16,
                    "weights.q4",
                    offset,
                    item["scales_and_zeros_storage_bytes"],
                )
            )
            index.append(
                {
                    "name": item["scales_and_zeros_name"],
                    "source_name": item["name"],
                    "logical_shape": item["shape"],
                    "format": "q4_scale_zero_bf16",
                    "storage_shape": item["scales_and_zeros_shape"],
                    "location": "weights.q4",
                    "offset": offset,
                    "length": item["scales_and_zeros_storage_bytes"],
                }
            )
            offset += item["scales_and_zeros_storage_bytes"]
    return tensors, index, offset, spec


def build_model():
    tensors, index, offset, spec = build_weight_tensors()
    attributes = [attribute_string("architecture", "llama-3.1-8b")]
    attributes.extend(attribute_int(name, value) for name, value in CONFIG.items())
    inputs = ["token_ids"]
    for item in spec:
        inputs.append(item["name"])
        if item["format"] == "q4_int4pack":
            inputs.append(item["scales_and_zeros_name"])
    node = node_proto(
        "LlamaDecoder",
        inputs,
        ["logits"],
        "decoder",
        attributes,
        LLAMA_DOMAIN,
    )
    graph = graph_proto(
        "llama_3_1_8b",
        [node],
        tensors,
        [value_info_proto("token_ids", [1, 1], 6)],
        [value_info_proto("logits", [1, CONFIG["vocab_size"]])],
    )
    return model_proto(graph, opset=23, custom_opsets={LLAMA_DOMAIN: 1}), index, offset


def build_operator_model():
    """Build the embedding-independent one-token decoder as explicit operators."""
    tensors, index, offset, spec = build_weight_tensors()
    tensors.extend(
        [
            tensor_int64_proto("hidden.shape", [2], [1, CONFIG["hidden_size"]]),
            tensor_int64_proto(
                "value.heads.shape",
                [4],
                [
                    1,
                    CONFIG["num_key_value_heads"],
                    1,
                    CONFIG["hidden_size"] // CONFIG["num_attention_heads"],
                ],
            ),
            tensor_int64_proto(
                "value.heads.repeats",
                [4],
                [
                    1,
                    1,
                    CONFIG["num_attention_heads"] // CONFIG["num_key_value_heads"],
                    1,
                ],
            ),
        ]
    )

    def q4_gemm(name, activation, weight, output, output_features, input_features):
        return node_proto(
            "WeightInt4packMM",
            [activation, weight, weight + ".scales_and_zeros"],
            [output],
            name,
            [
                attribute_int("output_features", output_features),
                attribute_int("input_features", input_features),
                attribute_int("group_size", CONFIG["q4_group_size"]),
            ],
            LLAMA_DOMAIN,
        )

    nodes = []
    hidden = "hidden_state"
    kv = (
        CONFIG["hidden_size"]
        * CONFIG["num_key_value_heads"]
        // CONFIG["num_attention_heads"]
    )
    for layer in range(CONFIG["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        stem = f"layer.{layer}"
        attention_norm = f"{stem}.attention_norm"
        query = f"{stem}.query"
        key = f"{stem}.key"
        value = f"{stem}.value"
        value_heads = f"{stem}.value_heads"
        repeated_heads = f"{stem}.repeated_value_heads"
        attention_value = f"{stem}.attention_value"
        attention = f"{stem}.attention"
        residual = f"{stem}.attention_residual"
        feed_forward_norm = f"{stem}.feed_forward_norm"
        gate = f"{stem}.gate"
        gate_sigmoid = f"{stem}.gate_sigmoid"
        silu = f"{stem}.silu"
        up = f"{stem}.up"
        activated = f"{stem}.activated"
        down = f"{stem}.down"
        output = f"{stem}.output"
        nodes.extend(
            [
                node_proto(
                    "RMSNormalization",
                    [hidden, f"{prefix}.input_layernorm.weight"],
                    [attention_norm],
                    f"{prefix}.input_layernorm",
                ),
                q4_gemm(
                    f"{prefix}.self_attn.q_proj",
                    attention_norm,
                    f"{prefix}.self_attn.q_proj.weight",
                    query,
                    CONFIG["hidden_size"],
                    CONFIG["hidden_size"],
                ),
                q4_gemm(
                    f"{prefix}.self_attn.k_proj",
                    attention_norm,
                    f"{prefix}.self_attn.k_proj.weight",
                    key,
                    kv,
                    CONFIG["hidden_size"],
                ),
                q4_gemm(
                    f"{prefix}.self_attn.v_proj",
                    attention_norm,
                    f"{prefix}.self_attn.v_proj.weight",
                    value,
                    kv,
                    CONFIG["hidden_size"],
                ),
                node_proto(
                    "Reshape",
                    [value, "value.heads.shape"],
                    [value_heads],
                    f"{prefix}.self_attn.value_heads",
                ),
                node_proto(
                    "Tile",
                    [value_heads, "value.heads.repeats"],
                    [repeated_heads],
                    f"{prefix}.self_attn.repeat_value_heads",
                ),
                node_proto(
                    "Reshape",
                    [repeated_heads, "hidden.shape"],
                    [attention_value],
                    f"{prefix}.self_attn.flatten_value_heads",
                ),
                q4_gemm(
                    f"{prefix}.self_attn.o_proj",
                    attention_value,
                    f"{prefix}.self_attn.o_proj.weight",
                    attention,
                    CONFIG["hidden_size"],
                    CONFIG["hidden_size"],
                ),
                node_proto("Add", [hidden, attention], [residual], f"{prefix}.attention_add"),
                node_proto(
                    "RMSNormalization",
                    [residual, f"{prefix}.post_attention_layernorm.weight"],
                    [feed_forward_norm],
                    f"{prefix}.post_attention_layernorm",
                ),
                q4_gemm(
                    f"{prefix}.mlp.gate_proj",
                    feed_forward_norm,
                    f"{prefix}.mlp.gate_proj.weight",
                    gate,
                    CONFIG["intermediate_size"],
                    CONFIG["hidden_size"],
                ),
                node_proto("Sigmoid", [gate], [gate_sigmoid], f"{prefix}.mlp.sigmoid"),
                node_proto("Mul", [gate, gate_sigmoid], [silu], f"{prefix}.mlp.silu"),
                q4_gemm(
                    f"{prefix}.mlp.up_proj",
                    feed_forward_norm,
                    f"{prefix}.mlp.up_proj.weight",
                    up,
                    CONFIG["intermediate_size"],
                    CONFIG["hidden_size"],
                ),
                node_proto("Mul", [silu, up], [activated], f"{prefix}.mlp.activate"),
                q4_gemm(
                    f"{prefix}.mlp.down_proj",
                    activated,
                    f"{prefix}.mlp.down_proj.weight",
                    down,
                    CONFIG["hidden_size"],
                    CONFIG["intermediate_size"],
                ),
                node_proto("Add", [residual, down], [output], f"{prefix}.mlp_add"),
            ]
        )
        hidden = output
    nodes.extend(
        [
            node_proto(
                "RMSNormalization",
                [hidden, "model.norm.weight"],
                ["final_norm"],
                "model.norm",
            ),
            q4_gemm(
                "lm_head",
                "final_norm",
                "lm_head.weight",
                "logits",
                CONFIG["vocab_size"],
                CONFIG["hidden_size"],
            ),
        ]
    )
    graph = graph_proto(
        "llama_3_1_8b_operator_decoder_single_token",
        nodes,
        tensors,
        [value_info_proto("hidden_state", [1, CONFIG["hidden_size"]])],
        [value_info_proto("logits", [1, CONFIG["vocab_size"]])],
    )
    model = model_proto(graph, opset=23, custom_opsets={LLAMA_DOMAIN: 1})
    return model, index, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--operator-output", type=Path)
    parser.add_argument("--operator-plan", type=Path)
    args = parser.parse_args()
    model, index, storage_bytes = build_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model)
    index_path = args.index or args.output.with_name("weights-index.json")
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture": "llama-3.1-8b",
                "config": CONFIG,
                "storage_bytes_with_alignment": storage_bytes,
                "tensors": index,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plan = compile_model(parse_model(model))
    if args.plan:
        args.plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if bool(args.operator_output) != bool(args.operator_plan):
        parser.error("--operator-output and --operator-plan must be provided together")
    if args.operator_output:
        operator_model, _, _ = build_operator_model()
        operator_plan = compile_operators(parse_model(operator_model))
        args.operator_output.parent.mkdir(parents=True, exist_ok=True)
        args.operator_output.write_bytes(operator_model)
        args.operator_plan.write_text(
            json.dumps(operator_plan, indent=2) + "\n", encoding="utf-8"
        )
        q4_calls = sum(
            call["function"] == "udonatenWeightInt4packMm"
            for call in operator_plan["aot"]["calls"]
        )
        print(
            f"llama_8b_operator_nodes={len(operator_plan['nodes'])} "
            f"q4_calls={q4_calls} passes={len(operator_plan['passes'])}"
        )
    print(
        f"llama_8b_onnx_bytes={len(model)} parameters={plan['weights']['parameters']} "
        f"external_bytes={plan['weights']['storage_bytes']} passes={len(plan['passes'])}"
    )


if __name__ == "__main__":
    main()
