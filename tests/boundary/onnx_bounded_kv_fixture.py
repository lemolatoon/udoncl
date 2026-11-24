"""Generate a tiny decoder attention layer with a bounded dynamic past.

The graph follows the Hugging Face Llama export that stage 3 compiles with a
bounded `past` dimension, at a size that runs anywhere:

- present = Concat(past, new) along the sequence axis, as a graph output;
- the causal mask is built from the shapes of `attention_mask` and the past
  (Shape, Gather, Range, ConstantOfShape, Trilu, Greater, Unsqueeze, Expand,
  Slice), padded with Cast(attention_mask) (Add, Equal, Where) and written
  with a ScatterND over the whole mask;
- repeat_kv is Unsqueeze, Expand and Reshape of the present key and value;
- the scores are MatMul(query, Transpose(key) * scale) plus the mask, then
  Softmax, and the result is MatMul(probabilities, value).

Every tensor that depends on the past has one dynamic axis whose extent is
`past` or `past + 1`. With --oracle, ONNX Runtime evaluates the model at
several past lengths and writes the inputs and outputs as JSON; that is the
oracle of the native execution tests. The model itself is written with the
dependency-free builders of runtime/onnx.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime" / "onnx"))

from onnx_builder import (  # noqa: E402
    attribute_float,
    attribute_int,
    graph_proto,
    message_field,
    model_proto,
    node_proto,
    tensor_int64_proto,
    tensor_proto,
    value_info_proto,
)

HEADS = 4
KV_HEADS = 2
GROUPS = HEADS // KV_HEADS
HEAD_DIM = 8
MINIMUM = -3.4028234663852886e38
FLOAT, INT64 = 1, 7


def attribute_tensor(name: str, tensor: bytes) -> bytes:
    from onnx_builder import bytes_field, varint_field

    return bytes_field(1, name) + message_field(5, tensor) + varint_field(20, 4)


def build_model() -> bytes:
    nodes = []
    initializers = []

    def constant(name, values, shape=None, integer=True):
        shape = [len(values)] if shape is None else shape
        proto = tensor_int64_proto if integer else tensor_proto
        initializers.append(proto(name, shape, values))
        return name

    def node(op, inputs, outputs, attributes=()):
        nodes.append(node_proto(op, inputs, outputs, f"/fixture/{outputs[0]}", attributes))
        return outputs[0]

    zero = constant("zero", [0], [])
    one = constant("one", [1], [])
    two = constant("two", [2], [])
    axis0 = constant("axis0", [0])
    axes01 = constant("axes01", [0, 1])
    axes12 = constant("axes12", [1, 2])
    axis2 = constant("axis2", [2])
    axis3 = constant("axis3", [3])
    start0 = constant("start0", [0])
    step1 = constant("step1", [1])
    column = constant("column", [-1, 1])
    expand_ones = constant("expand_ones", [1, 1, 1, 1])
    leading = constant("leading", [1, KV_HEADS, GROUPS])
    head_dim = constant("head_dim", [HEAD_DIM])
    query_heads = constant("query_heads", [1, HEADS])
    zero_f = constant("zero_f", [0.0], [], integer=False)
    minimum = constant("minimum", [MINIMUM], [], integer=False)
    scale = constant("scale", [1.0 / math.sqrt(HEAD_DIM)], [1], integer=False)
    zero_index = constant("zero_index", [0], [1, 1, 1, 1, 1])

    # present = Concat(past, new) along the sequence axis.
    node("Concat", ["past_key", "key"], ["present_key"], [attribute_int("axis", -2)])
    node("Concat", ["past_value", "value"], ["present_value"], [attribute_int("axis", -2)])

    # The causal mask from the shapes of the past and the attention mask.
    node("Shape", ["past_key"], ["past_shape"])
    node("Gather", ["past_shape", two], ["past_length"], [attribute_int("axis", 0)])
    node("Shape", ["query"], ["query_shape"])
    node("Gather", ["query_shape", two], ["sequence_length"], [attribute_int("axis", 0)])
    node("Shape", ["attention_mask"], ["mask_shape"])
    node("Gather", ["mask_shape", one], ["mask_length"], [attribute_int("axis", 0)])
    node("Range", [zero, "mask_length", one], ["columns"])
    node("Add", ["past_length", "sequence_length"], ["row_end"])
    node("Range", ["past_length", "row_end", one], ["rows"])
    node("Unsqueeze", ["sequence_length", axis0], ["sequence_vector"])
    node("Unsqueeze", ["mask_length", axis0], ["mask_vector"])
    node("Concat", ["sequence_vector", "mask_vector"], ["mask_extent"], [attribute_int("axis", 0)])
    nodes.append(
        node_proto(
            "ConstantOfShape",
            ["mask_extent"],
            ["full"],
            "/fixture/full",
            [attribute_tensor("value", tensor_proto("minimum_value", [1], [MINIMUM]))],
        )
    )
    node("Trilu", ["full", one], ["upper"], [attribute_int("upper", 1)])
    node("Reshape", ["rows", column], ["row_column"])
    node("Greater", ["columns", "row_column"], ["future"])
    node("Cast", ["future"], ["future_f"], [attribute_int("to", FLOAT)])
    node("Mul", ["upper", "future_f"], ["causal"])
    node("Unsqueeze", ["causal", axes01], ["causal4"])
    node("Expand", ["causal4", expand_ones], ["causal_expanded"])
    node("Slice", ["causal_expanded", start0, "mask_vector", axis3, step1], ["causal_slice"])
    node("Unsqueeze", ["attention_mask", axes12], ["mask4"])
    node("Cast", ["mask4"], ["mask4_f"], [attribute_int("to", FLOAT)])
    node("Add", ["causal_slice", "mask4_f"], ["padded"])
    node("Equal", ["padded", zero_f], ["padding"])
    node("Where", ["padding", minimum, "causal_slice"], ["masked"])
    # mask[..., :mask_length] = masked, as one ScatterND over the whole mask.
    node("Range", [zero, "mask_length", one], ["mask_columns"])
    node("Reshape", ["mask_columns", constant("index_shape", [1, 1, 1, -1, 1])], ["column_index"])
    node("Shape", ["column_index"], ["index_extent"])
    node("Expand", [zero_index, "index_extent"], ["zero_coordinate"])
    node(
        "Concat",
        ["zero_coordinate", "zero_coordinate", "zero_coordinate", "column_index"],
        ["indices"],
        [attribute_int("axis", 4)],
    )
    node("ScatterND", ["causal_expanded", "indices", "masked"], ["mask"])

    # repeat_kv: Unsqueeze, Expand and Reshape of the present key and value.
    node("Shape", ["present_key"], ["present_shape"])
    node("Gather", ["present_shape", two], ["key_length"], [attribute_int("axis", 0)])
    node("Unsqueeze", ["key_length", axis0], ["key_vector"])
    node("Concat", [leading, "key_vector", head_dim], ["repeat_extent"], [attribute_int("axis", 0)])
    node("Concat", [query_heads, "key_vector", head_dim], ["heads_extent"], [attribute_int("axis", 0)])
    for name in ("key", "value"):
        node("Unsqueeze", [f"present_{name}", axis2], [f"{name}_unsqueezed"])
        node("Expand", [f"{name}_unsqueezed", "repeat_extent"], [f"{name}_repeated"])
        node("Reshape", [f"{name}_repeated", "heads_extent"], [f"{name}_heads"])

    node("Transpose", ["key_heads"], ["key_transposed"], [_perm([0, 1, 3, 2])])
    node("Mul", ["key_transposed", scale], ["key_scaled"])
    node("MatMul", ["query", "key_scaled"], ["scores"])
    node("Slice", ["mask", start0, "key_vector", axis3, step1], ["score_mask"])
    node("Add", ["scores", "score_mask"], ["masked_scores"])
    node("Softmax", ["masked_scores"], ["probabilities"], [attribute_int("axis", -1)])
    node("MatMul", ["probabilities", "value_heads"], ["attention"])

    inputs = [
        value_info_proto("query", ["batch", HEADS, "sequence", HEAD_DIM]),
        value_info_proto("key", ["batch", KV_HEADS, "sequence", HEAD_DIM]),
        value_info_proto("value", ["batch", KV_HEADS, "sequence", HEAD_DIM]),
        value_info_proto("attention_mask", ["batch", "past + 1"], INT64),
        value_info_proto("past_key", ["batch", KV_HEADS, "past", HEAD_DIM]),
        value_info_proto("past_value", ["batch", KV_HEADS, "past", HEAD_DIM]),
    ]
    outputs = [
        value_info_proto("attention", ["batch", HEADS, "sequence", HEAD_DIM]),
        value_info_proto("present_key", ["batch", KV_HEADS, "past + 1", HEAD_DIM]),
        value_info_proto("present_value", ["batch", KV_HEADS, "past + 1", HEAD_DIM]),
    ]
    graph = graph_proto("bounded_kv", nodes, initializers, inputs, outputs)
    return model_proto(graph, opset=14)


def _perm(values):
    from onnx_builder import attribute_ints

    return attribute_ints("perm", values)


def oracle(model: Path, pasts) -> dict:
    import numpy
    import onnxruntime

    session = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    cases = []
    for past in pasts:
        generator = numpy.random.default_rng(past + 1)

        def sample(*shape):
            return generator.uniform(-1.0, 1.0, shape).astype(numpy.float32)

        feeds = {
            "query": sample(1, HEADS, 1, HEAD_DIM),
            "key": sample(1, KV_HEADS, 1, HEAD_DIM),
            "value": sample(1, KV_HEADS, 1, HEAD_DIM),
            "attention_mask": numpy.ones((1, past + 1), dtype=numpy.int64),
            "past_key": sample(1, KV_HEADS, past, HEAD_DIM),
            "past_value": sample(1, KV_HEADS, past, HEAD_DIM),
        }
        names = [output.name for output in session.get_outputs()]
        results = session.run(names, feeds)
        cases.append(
            {
                "past": past,
                "inputs": {name: value.tolist() for name, value in feeds.items()},
                "outputs": {name: value.tolist() for name, value in zip(names, results)},
            }
        )
    return {"cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, help="write ONNX Runtime outputs as JSON")
    parser.add_argument("--bound", type=int, default=8, help="largest past of the oracle")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(build_model())
    if args.oracle:
        pasts = sorted({0, 1, min(5, args.bound), args.bound})
        args.oracle.write_text(json.dumps(oracle(args.output, pasts)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
