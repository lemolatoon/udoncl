"""Generate a focused Transpose model at a TinyLlama attention shape."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime" / "onnx"))

from udon_onnx import (  # noqa: E402
    attribute_ints,
    graph_proto,
    model_proto,
    node_proto,
    value_info_proto,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    graph = graph_proto(
        "assigned_transpose_component",
        [
            node_proto(
                "Transpose",
                ["input"],
                ["output"],
                "attention_transpose",
                attributes=(attribute_ints("perm", [0, 2, 1, 3]),),
            )
        ],
        [],
        [value_info_proto("input", [1, 4, 32, 64])],
        [value_info_proto("output", [1, 32, 4, 64])],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model_proto(graph, opset=23))
    print(f"transpose_component_onnx_bytes={args.output.stat().st_size} words=8192")


if __name__ == "__main__":
    main()
