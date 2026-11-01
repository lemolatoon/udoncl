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
    external_tensor_proto,
    graph_proto,
    llama_weight_spec,
    model_proto,
    node_proto,
    parse_model,
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


def build_model():
    tensors = []
    index = []
    offset = 0
    spec = llama_weight_spec(CONFIG)
    for item in spec:
        offset = align(offset)
        data_type = 16 if item["format"] == "bf16" else 2
        storage_shape = item["shape"] if data_type == 16 else [item["storage_bytes"]]
        tensors.append(
            external_tensor_proto(
                item["name"],
                storage_shape,
                data_type,
                "weights.q4",
                offset,
                item["storage_bytes"],
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
                "length": item["storage_bytes"],
            }
        )
        offset += item["storage_bytes"]
    attributes = [attribute_string("architecture", "llama-3.1-8b")]
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
        "llama_3_1_8b",
        [node],
        tensors,
        [value_info_proto("token_ids", [1, 1], 6)],
        [value_info_proto("logits", [1, CONFIG["vocab_size"]])],
    )
    return model_proto(graph, opset=23, custom_opsets={LLAMA_DOMAIN: 1}), index, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--plan", type=Path)
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
    print(
        f"llama_8b_onnx_bytes={len(model)} parameters={plan['weights']['parameters']} "
        f"external_bytes={plan['weights']['storage_bytes']} passes={len(plan['passes'])}"
    )


if __name__ == "__main__":
    main()
