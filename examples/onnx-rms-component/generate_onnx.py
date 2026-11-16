"""Generate a focused RMSNormalization model at TinyLlama hidden width."""

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


ROWS = 4
WIDTH = 2048


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    scale = [0.75 + (index % 17) / 32.0 for index in range(WIDTH)]
    graph = graph_proto(
        "assigned_rms_component",
        [
            node_proto(
                "RmsNorm",
                ["input", "scale", "epsilon"],
                ["output"],
                "rms",
                domain="com.lemolatoon.udoncl",
            )
        ],
        [
            tensor_proto("scale", [WIDTH], scale),
            tensor_proto("epsilon", [], [1.0e-5]),
        ],
        [value_info_proto("input", [ROWS, WIDTH])],
        [value_info_proto("output", [ROWS, WIDTH])],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        model_proto(
            graph,
            opset=23,
            custom_opsets={"com.lemolatoon.udoncl": 1},
        )
    )
    print(f"rms_component_onnx_bytes={args.output.stat().st_size} rows={ROWS} width={WIDTH}")


if __name__ == "__main__":
    main()
