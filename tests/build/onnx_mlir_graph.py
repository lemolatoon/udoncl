#!/usr/bin/env python3

import importlib.util
import math
from pathlib import Path
import struct
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime" / "onnx"))
MODULE_PATH = ROOT / "runtime" / "onnx" / "onnx_mlir_graph.py"
SPEC = importlib.util.spec_from_file_location("onnx_mlir_graph", MODULE_PATH)
graph_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = graph_module
SPEC.loader.exec_module(graph_module)
from onnx_mlir_lower import host_graph
from udon_onnx import emit_c_host


def initializer(shape):
    count = math.prod(shape)
    return {
        "shape": list(shape),
        "data_type_name": "float32",
        "data_location": 0,
        "raw": struct.pack(f"<{count}f", *([0.0] * count)),
    }


CANONICAL = r'''
module {
  func.func @main_graph(%arg0: tensor<?x4xf32> {onnx.name = "input"} loc(unknown)) -> (tensor<?x2xf32> {onnx.name = "output"}) {
    %0 = onnx.Constant dense<0.0> : tensor<4x2xf32> loc(#loc1)
    %1 = onnx.Constant dense<0.0> : tensor<2xf32> loc(#loc2)
    %2 = "onnx.Gemm"(%arg0, %0, %1) <{alpha = 1.000000e+00 : f32, transB = 0 : si64}> {onnx_node_name = "dense"} : (tensor<?x4xf32>, tensor<4x2xf32>, tensor<2xf32>) -> tensor<?x2xf32> loc(#loc3)
    %3 = "onnx.Relu"(%2) {onnx_node_name = "activation"} : (tensor<?x2xf32>) -> tensor<?x2xf32> loc(#loc4)
    onnx.Return %3 : tensor<?x2xf32> loc(unknown)
  }
}
#loc1 = loc("Initializer_weight")
#loc2 = loc("Initializer_bias")
#loc3 = loc("dense")
#loc4 = loc("activation")
'''


class ONNXMLIRGraphTests(unittest.TestCase):
    def test_parses_graph_without_serialized_execution_plan(self):
        graph = graph_module.parse(CANONICAL)
        self.assertEqual(graph.name, "main_graph")
        self.assertEqual(graph.inputs[0].name, "input")
        self.assertEqual(graph.inputs[0].type.shape, (None, 4))
        self.assertEqual(graph.outputs[0].name, "output")
        self.assertEqual(
            [(value.name, value.type.shape) for value in graph.initializers],
            [("weight", (4, 2)), ("bias", (2,))],
        )
        self.assertEqual([operator.op_type for operator in graph.operators], ["Gemm", "Relu"])
        self.assertEqual(graph.operators[0].inputs, ("%arg0", "%0", "%1"))
        self.assertEqual(graph.operators[0].attributes["alpha"], 1.0)
        self.assertEqual(graph.operators[0].attributes["transB"], 0)
        self.assertEqual(graph.operators[0].node_name, "dense")
        self.assertEqual(graph.operators[0].location, "dense")

    def test_rejects_undefined_ssa_input(self):
        broken = CANONICAL.replace('"onnx.Relu"(%2)', '"onnx.Relu"(%missing)')
        with self.assertRaisesRegex(ValueError, "undefined SSA value %missing"):
            graph_module.parse(broken)

    def test_rejects_unnamed_initializer(self):
        broken = CANONICAL.replace(" loc(#loc1)", " loc(unknown)", 1)
        with self.assertRaisesRegex(ValueError, "lacks initializer name"):
            graph_module.parse(broken)

    def test_structural_no_value_is_a_known_operand(self):
        with_no_value = CANONICAL.replace(
            '    %2 = "onnx.Gemm"(%arg0, %0, %1)',
            '    %none = "onnx.NoValue"() : () -> none loc(unknown)\n'
            '    %2 = "onnx.Gemm"(%arg0, %0, %1, %none)',
        ).replace(
            '(tensor<?x4xf32>, tensor<4x2xf32>, tensor<2xf32>)',
            '(tensor<?x4xf32>, tensor<4x2xf32>, tensor<2xf32>, none)',
        )
        graph = graph_module.parse(with_no_value)
        self.assertEqual(graph.operators[0].inputs[-1], "%none")

    def test_lowers_canonical_graph_directly_to_public_library_calls(self):
        static = CANONICAL.replace("?x", "1x")
        graph = graph_module.parse(static)
        model = {
            "graph": {
                "initializers": {
                    "weight": initializer((4, 2)),
                    "bias": initializer((2,)),
                }
            }
        }
        lowering = host_graph(graph, model)
        self.assertNotIn("aot", lowering)
        self.assertNotIn("passes", lowering)
        self.assertEqual(
            [node["provider"] for node in lowering["nodes"]],
            ["udonblasSgemm", "udonatenRelu"],
        )
        source = emit_c_host(lowering)
        self.assertIn("udonblasSgemm(", source)
        self.assertIn("udonatenRelu(", source)
        self.assertIn("int udon_onnx_session_run(void)", source)
        self.assertIn("udon_onnx_session_write_input_float", source)

    def test_unsupported_operator_reports_source_location(self):
        unsupported = CANONICAL.replace("?x", "1x").replace(
            "onnx.Relu", "onnx.Custom"
        )
        graph = graph_module.parse(unsupported)
        with self.assertRaisesRegex(
            ValueError, "unsupported canonical ONNX operator 'Custom' at activation"
        ):
            host_graph(
                graph,
                {
                    "graph": {
                        "initializers": {
                            "weight": initializer((4, 2)),
                            "bias": initializer((2,)),
                        }
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
