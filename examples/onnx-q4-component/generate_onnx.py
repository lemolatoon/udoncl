"""Generate a focused MatMulNBits model with TinyLlama-depth accumulation."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime" / "onnx"))

from onnx_builder import (  # noqa: E402
    attribute_int,
    graph_proto,
    model_proto,
    node_proto,
    tensor_proto,
    tensor_uint8_proto,
    value_info_proto,
)


M = 4
N = 33
K = 2048
BLOCK = 32
BLOCKS = K // BLOCK


def packed_weights():
    packed = []
    for column in range(N):
        for block in range(BLOCKS):
            values = [
                (column * 3 + (block * BLOCK + inner) * 7) & 15
                for inner in range(BLOCK)
            ]
            packed.extend(
                values[index] | (values[index + 1] << 4)
                for index in range(0, BLOCK, 2)
            )
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    scales = [
        ((column + block) % 7 + 1) / 23.0
        for column in range(N)
        for block in range(BLOCKS)
    ]
    graph = graph_proto(
        "assigned_q4_component",
        [
            node_proto(
                "MatMulNBits",
                ["input", "weight.q4", "weight.scale", "", "", ""],
                ["output"],
                "projection",
                attributes=(
                    attribute_int("K", K),
                    attribute_int("N", N),
                    attribute_int("block_size", BLOCK),
                    attribute_int("bits", 4),
                ),
                domain="com.microsoft",
            )
        ],
        [
            tensor_uint8_proto("weight.q4", [N, BLOCKS, 16], packed_weights()),
            tensor_proto("weight.scale", [N, BLOCKS], scales),
        ],
        [value_info_proto("input", [M, K])],
        [value_info_proto("output", [M, N])],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        model_proto(graph, opset=13, custom_opsets={"com.microsoft": 1})
    )
    print(f"q4_component_onnx_bytes={args.output.stat().st_size} m={M} n={N} k={K}")


if __name__ == "__main__":
    main()
