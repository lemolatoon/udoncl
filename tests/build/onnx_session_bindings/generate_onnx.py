"""Generate a model with three inputs and one output: sum = ((a + b) + c)^T."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "runtime" / "onnx"))

from onnx_builder import (  # noqa: E402
    attribute_ints,
    graph_proto,
    model_proto,
    node_proto,
    value_info_proto,
)

SHAPE = [32, 256]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    graph = graph_proto(
        "onnx_session_bindings",
        [
            node_proto("Add", ["a", "b"], ["partial"], "first"),
            node_proto("Add", ["partial", "c"], ["total"], "second"),
            node_proto(
                "Transpose",
                ["total"],
                ["sum"],
                "transpose",
                attributes=(attribute_ints("perm", [1, 0]),),
            ),
        ],
        [],
        [value_info_proto(name, SHAPE) for name in ("a", "b", "c")],
        [value_info_proto("sum", list(reversed(SHAPE)))],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model_proto(graph, opset=23))


if __name__ == "__main__":
    main()
