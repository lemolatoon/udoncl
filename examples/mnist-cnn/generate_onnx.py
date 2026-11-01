"""Generate the trained MNIST graph as a real dependency-free ONNX model."""

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime/onnx"))

from udon_onnx import (  # noqa: E402
    attribute_int,
    attribute_ints,
    graph_proto,
    model_proto,
    node_proto,
    tensor_proto,
    value_info_proto,
)


def read_weight_array(header, name, expected):
    match = re.search(
        rf"static const float {name}\[{expected}\] = \{{(.*?)\}};",
        header,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"could not find {name}[{expected}] in weights.h")
    values = [
        float(token.rstrip("f"))
        for token in match.group(1).split(",")
        if token.strip()
    ]
    if len(values) != expected:
        raise RuntimeError(f"{name} contains {len(values)} values; expected {expected}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    header = (source / "weights.h").read_text(encoding="utf-8")
    conv_weight = read_weight_array(header, "kMnistConvWeight", 400)
    conv_bias = read_weight_array(header, "kMnistConvBias", 16)
    original_dense = read_weight_array(header, "kMnistDenseWeight", 1440)
    dense_bias = read_weight_array(header, "kMnistDenseBias", 10)
    dense_weight = []
    for channel in range(16):
        for pool_pixel in range(9):
            original_feature = pool_pixel * 16 + channel
            for digit in range(10):
                dense_weight.append(original_dense[digit * 144 + original_feature])

    nodes = [
        node_proto(
            "Conv",
            ["image", "conv.weight"],
            ["conv.raw"],
            "conv",
            [
                attribute_ints("kernel_shape", [5, 5]),
                attribute_ints("pads", [0, 0, 0, 0]),
                attribute_ints("strides", [1, 1]),
            ],
        ),
        node_proto(
            "Add", ["conv.raw", "conv.bias"], ["conv.biased"], "conv.bias.add"
        ),
        node_proto("Relu", ["conv.biased"], ["relu"], "relu"),
        node_proto(
            "AveragePool",
            ["relu"],
            ["pooled"],
            "pool",
            [
                attribute_ints("kernel_shape", [8, 8]),
                attribute_ints("strides", [8, 8]),
            ],
        ),
        node_proto(
            "Flatten", ["pooled"], ["features"], "flatten", [attribute_int("axis", 1)]
        ),
        node_proto(
            "Gemm",
            ["features", "dense.weight"],
            ["logits.raw"],
            "dense",
        ),
        node_proto(
            "Add", ["logits.raw", "dense.bias"], ["logits"], "dense.bias.add"
        ),
    ]
    initializers = [
        tensor_proto("conv.weight", [16, 1, 5, 5], conv_weight),
        tensor_proto("conv.bias", [1, 16, 1, 1], conv_bias),
        tensor_proto("dense.weight", [144, 10], dense_weight),
        tensor_proto("dense.bias", [10], dense_bias),
    ]
    graph = graph_proto(
        "mnist_udondnn_udonblas",
        nodes,
        initializers,
        [value_info_proto("image", [1, 1, 28, 28])],
        [value_info_proto("logits", [1, 10])],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model_proto(graph, opset=13))
    print(f"mnist_onnx_bytes={args.output.stat().st_size} nodes={len(nodes)}")


if __name__ == "__main__":
    main()
