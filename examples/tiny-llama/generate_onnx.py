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
    graph_proto,
    llama_weight_spec,
    model_proto,
    node_proto,
    parse_model,
    tensor_proto,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
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
    print(f"tiny_llama_onnx_bytes={len(model)} passes={len(plan['passes'])}")


if __name__ == "__main__":
    main()
