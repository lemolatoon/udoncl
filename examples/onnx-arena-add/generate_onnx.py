"""Generate a small standard-operator ONNX model for the Udon runtime."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime/onnx"))

from udon_onnx import (  # noqa: E402
    graph_proto,
    model_proto,
    node_proto,
    value_info_proto,
)


WORDS = 512


def build_model() -> bytes:
    return model_proto(
        graph_proto(
            "udoncl_arena_add",
            [node_proto("Add", ["left", "right"], ["sum"], "add")],
            [],
            [
                value_info_proto("left", [1, WORDS]),
                value_info_proto("right", [1, WORDS]),
            ],
            [value_info_proto("sum", [1, WORDS])],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    args.output.write_bytes(model)
    print(f"onnx_arena_add_bytes={len(model)} words={WORDS}")


if __name__ == "__main__":
    main()
