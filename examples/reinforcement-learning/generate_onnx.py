"""Generate a dependency-free continuous-control policy ONNX fixture."""

import argparse
from pathlib import Path
import sys


root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "runtime/onnx"))

from udon_onnx import (  # noqa: E402
    attribute_int,
    graph_proto,
    model_proto,
    node_proto,
    tensor_proto,
    value_info_proto,
)


def build_model() -> bytes:
    nodes = [
        node_proto("Gemm", ["observation", "hidden.weight"], ["hidden.linear"], "hidden_gemm"),
        node_proto("Add", ["hidden.linear", "hidden.bias"], ["hidden.preact"], "hidden_bias"),
        node_proto("Tanh", ["hidden.preact"], ["hidden.tanh"], "hidden_tanh"),
        node_proto("Sigmoid", ["hidden.preact"], ["hidden.gate"], "hidden_gate"),
        node_proto("Mul", ["hidden.tanh", "hidden.gate"], ["hidden"], "hidden_mul"),
        node_proto("Gemm", ["hidden", "policy.weight"], ["policy.linear"], "policy_gemm"),
        node_proto("Add", ["policy.linear", "policy.bias"], ["policy.biased"], "policy_bias"),
        node_proto("Sub", ["policy.biased", "policy.baseline"], ["policy.centered"], "policy_center"),
        node_proto("Mul", ["policy.centered", "policy.temperature"], ["policy.scaled"], "policy_scale"),
        node_proto(
            "Softmax",
            ["policy.scaled"],
            ["action_probabilities"],
            "policy_softmax",
            [attribute_int("axis", 1)],
        ),
    ]
    initializers = [
        tensor_proto("hidden.weight", [4, 8], [((index % 9) - 4) / 16 for index in range(32)]),
        tensor_proto("hidden.bias", [8], [0.05 * (index - 3) for index in range(8)]),
        tensor_proto("policy.weight", [8, 3], [((index % 7) - 3) / 12 for index in range(24)]),
        tensor_proto("policy.bias", [3], [0.1, -0.05, 0.025]),
        tensor_proto("policy.baseline", [3], [0.02, 0.02, 0.02]),
        tensor_proto("policy.temperature", [], [0.75]),
    ]
    return model_proto(
        graph_proto(
            "udon_rl_policy",
            nodes,
            initializers,
            [value_info_proto("observation", [1, 4])],
            [value_info_proto("action_probabilities", [1, 3])],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = build_model()
    args.output.write_bytes(data)
    print(f"rl_policy_onnx_bytes={len(data)} nodes=10")


if __name__ == "__main__":
    main()
