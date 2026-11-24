"""Generate a model with three inputs and one output: sum = ((a + b) + c)^T.

With --bounded, generate instead present = Concat(past, step) of a past of
`past` rows of 8 floats, a dimension the build bounds."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "runtime" / "onnx"))

from onnx_builder import (  # noqa: E402
    attribute_int,
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
    parser.add_argument("--bounded", action="store_true")
    args = parser.parse_args()
    if args.bounded:
        graph = graph_proto(
            "onnx_session_bindings_bounded",
            [
                node_proto(
                    "Concat",
                    ["past", "step"],
                    ["present"],
                    "present",
                    attributes=(attribute_int("axis", 0),),
                ),
            ],
            [],
            [
                value_info_proto("past", ["past", 8]),
                value_info_proto("step", [1, 8]),
            ],
            [value_info_proto("present", ["past + 1", 8])],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(model_proto(graph, opset=23))
        return
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
