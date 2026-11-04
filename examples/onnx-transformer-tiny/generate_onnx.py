"""Generate a bounded dynamic hidden-size-8 Transformer-style ONNX fixture."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime" / "onnx"))

from udon_onnx import (  # noqa: E402
    graph_proto,
    model_proto,
    node_proto,
    tensor_proto,
    value_info_proto,
)


HIDDEN = 8
VOCAB = 16


def matrix(seed):
    return [
        ((row * 5 + column * 3 + seed) % 17 - 8) / 32.0
        for row in range(HIDDEN)
        for column in range(HIDDEN)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    initializers = [
        tensor_proto(
            "embedding.weight",
            [VOCAB, HIDDEN],
            [((index * 7) % 23 - 11) / 32.0 for index in range(VOCAB * HIDDEN)],
        ),
        tensor_proto("query.weight", [HIDDEN, HIDDEN], matrix(1)),
        tensor_proto("value.weight", [HIDDEN, HIDDEN], matrix(2)),
        tensor_proto("output.weight", [HIDDEN, HIDDEN], matrix(3)),
    ]
    nodes = [
        node_proto(
            "Gather", ["embedding.weight", "input_ids"], ["hidden"], "embedding"
        ),
        node_proto("MatMul", ["hidden", "query.weight"], ["query"], "query"),
        node_proto("Add", ["query", "past_key"], ["present_key"], "key.cache"),
        node_proto("MatMul", ["hidden", "value.weight"], ["value"], "value"),
        node_proto(
            "Add", ["value", "past_value"], ["present_value"], "value.cache"
        ),
        node_proto(
            "Add", ["present_key", "present_value"], ["attention"], "attention"
        ),
        node_proto("Tanh", ["attention"], ["activated"], "activation"),
        node_proto(
            "MatMul", ["activated", "output.weight"], ["logits"], "projection"
        ),
    ]
    dynamic_hidden = [1, "sequence", HIDDEN]
    graph = graph_proto(
        "bounded_dynamic_transformer",
        nodes,
        initializers,
        [
            value_info_proto("input_ids", [1, "sequence"], 6),
            value_info_proto("past_key", dynamic_hidden),
            value_info_proto("past_value", dynamic_hidden),
        ],
        [
            value_info_proto("logits", dynamic_hidden),
            value_info_proto("present_key", dynamic_hidden),
            value_info_proto("present_value", dynamic_hidden),
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model_proto(graph, opset=13))
    print(f"transformer_onnx_bytes={args.output.stat().st_size} nodes={len(nodes)}")


if __name__ == "__main__":
    main()
