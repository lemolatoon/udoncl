"""Generate a bounded dynamic hidden-size-8 Transformer-style ONNX fixture."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime" / "onnx"))

from udon_onnx import (  # noqa: E402
    attribute_int,
    graph_proto,
    model_proto,
    node_proto,
    tensor_proto,
    tensor_uint8_proto,
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


def quantized_projection():
    packed = []
    for column in range(HIDDEN):
        values = []
        for inner in range(32):
            signed = (
                max(-8, min(7, (inner * 5 + column * 3 + 3) % 17 - 8))
                if inner < HIDDEN
                else 0
            )
            values.append(signed + 8)
        packed.extend(
            values[index] | (values[index + 1] << 4)
            for index in range(0, 32, 2)
        )
    return packed


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
        tensor_uint8_proto(
            "output.weight.q4", [HIDDEN, 1, 16], quantized_projection()
        ),
        tensor_proto("output.weight.scale", [HIDDEN, 1], [1.0 / 32.0] * HIDDEN),
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
            "MatMulNBits",
            [
                "activated",
                "output.weight.q4",
                "output.weight.scale",
                "",
                "",
                "",
            ],
            ["logits"],
            "projection",
            attributes=(
                attribute_int("K", HIDDEN),
                attribute_int("N", HIDDEN),
                attribute_int("block_size", 32),
                attribute_int("bits", 4),
            ),
            domain="com.microsoft",
        ),
    ]
    dynamic_hidden = [1, "sequence", HIDDEN]
    graph = graph_proto(
        "bounded_dynamic_transformer",
        nodes,
        initializers,
        [
            value_info_proto("input_ids", [1, "sequence"], 7),
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
    args.output.write_bytes(
        model_proto(graph, opset=13, custom_opsets={"com.microsoft": 1})
    )
    print(f"transformer_onnx_bytes={args.output.stat().st_size} nodes={len(nodes)}")


if __name__ == "__main__":
    main()
