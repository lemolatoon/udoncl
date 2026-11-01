"""Generate the numerical Tiny Llama fixture as a Udon ONNX model."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime/onnx"))

from udon_onnx import (  # noqa: E402
    LLAMA_DOMAIN,
    attribute_int,
    attribute_string,
    compile_model,
    compile_operators,
    graph_proto,
    llama_weight_spec,
    model_proto,
    node_proto,
    parse_model,
    tensor_proto,
    tensor_int64_proto,
    value_info_proto,
)


CONFIG = {
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "vocab_size": 16,
    "max_position_embeddings": 4,
    "context_length": 4,
    "rope_theta": 10000,
    "q4_group_size": 0,
}


def build_standard_single_token_model(values, spec):
    """Build the sequence-length-one decoder from standard ONNX operators."""
    tensors = [tensor_proto(item["name"], item["shape"], values[item["name"]]) for item in spec]
    tensors.append(tensor_int64_proto("hidden.shape", [2], [1, CONFIG["hidden_size"]]))
    nodes = [
        node_proto(
            "Gather",
            ["model.embed_tokens.weight", "token_ids"],
            ["embedded.3d"],
            "embedding",
        ),
        node_proto(
            "Reshape",
            ["embedded.3d", "hidden.shape"],
            ["layer.0.input"],
            "embedding.reshape",
        ),
    ]
    hidden = "layer.0.input"
    for layer in range(CONFIG["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        attention_norm = f"layer.{layer}.attention_norm"
        value = f"layer.{layer}.value"
        attention = f"layer.{layer}.attention"
        residual = f"layer.{layer}.attention_residual"
        feed_forward_norm = f"layer.{layer}.feed_forward_norm"
        gate = f"layer.{layer}.gate"
        gate_sigmoid = f"layer.{layer}.gate_sigmoid"
        silu = f"layer.{layer}.silu"
        up = f"layer.{layer}.up"
        activated = f"layer.{layer}.activated"
        down = f"layer.{layer}.down"
        output = f"layer.{layer}.output"
        nodes.extend(
            [
                node_proto(
                    "RMSNormalization",
                    [hidden, f"{prefix}.input_layernorm.weight"],
                    [attention_norm],
                    f"{prefix}.input_layernorm",
                ),
                node_proto(
                    "Gemm",
                    [attention_norm, f"{prefix}.self_attn.v_proj.weight"],
                    [value],
                    f"{prefix}.self_attn.v_proj",
                    [attribute_int("transB", 1)],
                ),
                node_proto(
                    "Gemm",
                    [value, f"{prefix}.self_attn.o_proj.weight"],
                    [attention],
                    f"{prefix}.self_attn.o_proj",
                    [attribute_int("transB", 1)],
                ),
                node_proto("Add", [hidden, attention], [residual], f"{prefix}.attention_add"),
                node_proto(
                    "RMSNormalization",
                    [residual, f"{prefix}.post_attention_layernorm.weight"],
                    [feed_forward_norm],
                    f"{prefix}.post_attention_layernorm",
                ),
                node_proto(
                    "Gemm",
                    [feed_forward_norm, f"{prefix}.mlp.gate_proj.weight"],
                    [gate],
                    f"{prefix}.mlp.gate_proj",
                    [attribute_int("transB", 1)],
                ),
                node_proto("Sigmoid", [gate], [gate_sigmoid], f"{prefix}.mlp.sigmoid"),
                node_proto("Mul", [gate, gate_sigmoid], [silu], f"{prefix}.mlp.silu"),
                node_proto(
                    "Gemm",
                    [feed_forward_norm, f"{prefix}.mlp.up_proj.weight"],
                    [up],
                    f"{prefix}.mlp.up_proj",
                    [attribute_int("transB", 1)],
                ),
                node_proto("Mul", [silu, up], [activated], f"{prefix}.mlp.activate"),
                node_proto(
                    "Gemm",
                    [activated, f"{prefix}.mlp.down_proj.weight"],
                    [down],
                    f"{prefix}.mlp.down_proj",
                    [attribute_int("transB", 1)],
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
            node_proto(
                "Gemm",
                ["final_norm", "lm_head.weight"],
                ["logits"],
                "lm_head",
                [attribute_int("transB", 1)],
            ),
        ]
    )
    graph = graph_proto(
        "udon_tiny_llama_standard_single_token",
        nodes,
        tensors,
        [value_info_proto("token_ids", [1, 1], 6)],
        [value_info_proto("logits", [1, CONFIG["vocab_size"]])],
    )
    return model_proto(graph, opset=23)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--standard-output", type=Path)
    parser.add_argument("--standard-plan", type=Path)
    args = parser.parse_args()
    reference_spec = importlib.util.spec_from_file_location("tiny_llama_reference", SOURCE / "run.py")
    reference = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference)
    source_weights = reference.make_weights(0x4C4C414D)
    values = {
        "model.embed_tokens.weight": source_weights["embedding"],
        "model.layers.0.input_layernorm.weight": source_weights["attention_norm"],
        "model.layers.0.self_attn.q_proj.weight": source_weights["wq"],
        "model.layers.0.self_attn.k_proj.weight": source_weights["wk"],
        "model.layers.0.self_attn.v_proj.weight": source_weights["wv"],
        "model.layers.0.self_attn.o_proj.weight": source_weights["wo"],
        "model.layers.0.post_attention_layernorm.weight": source_weights["ffn_norm"],
        "model.layers.0.mlp.gate_proj.weight": source_weights["w1"],
        "model.layers.0.mlp.up_proj.weight": source_weights["w3"],
        "model.layers.0.mlp.down_proj.weight": source_weights["w2"],
        "model.norm.weight": source_weights["final_norm"],
        "lm_head.weight": source_weights["output"],
    }
    spec = llama_weight_spec(CONFIG, "float32")
    tensors = [tensor_proto(item["name"], item["shape"], values[item["name"]]) for item in spec]
    attributes = [attribute_string("architecture", "udon-tiny-llama")]
    attributes.extend(attribute_int(name, value) for name, value in CONFIG.items())
    node = node_proto(
        "LlamaDecoder",
        ["token_ids"] + [item["name"] for item in spec],
        ["logits"],
        "decoder",
        attributes,
        LLAMA_DOMAIN,
    )
    graph = graph_proto(
        "udon_tiny_llama",
        [node],
        tensors,
        [value_info_proto("token_ids", [1, 4], 6)],
        [value_info_proto("logits", [1, 16])],
    )
    model = model_proto(graph, opset=23, custom_opsets={LLAMA_DOMAIN: 1})
    plan = compile_model(parse_model(model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model)
    args.plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if bool(args.standard_output) != bool(args.standard_plan):
        parser.error("--standard-output and --standard-plan must be provided together")
    if args.standard_output:
        standard_model = build_standard_single_token_model(values, spec)
        standard_plan = compile_operators(parse_model(standard_model))
        args.standard_output.parent.mkdir(parents=True, exist_ok=True)
        args.standard_output.write_bytes(standard_model)
        args.standard_plan.write_text(
            json.dumps(standard_plan, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"tiny_llama_standard_onnx_bytes={len(standard_model)} "
            f"nodes={len(standard_plan['nodes'])} passes={len(standard_plan['passes'])}"
        )
    print(f"tiny_llama_onnx_bytes={len(model)} passes={len(plan['passes'])}")


if __name__ == "__main__":
    main()
