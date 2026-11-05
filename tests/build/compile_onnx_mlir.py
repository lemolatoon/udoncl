"""Check the generic ONNX MLIR pipeline orchestration and diagnostics."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'onnx_mlir_compiler', ROOT / 'scripts/compile_onnx_mlir.py')
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


class CompileONNXMLIRTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix='udoncl ONNX compile test ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / 'model.onnx'
        self.model.write_bytes(b'onnx')
        self.importer = self.root / 'udoncl-onnx'
        self.optimizer = self.root / 'onnx-mlir-opt'
        self.importer.write_text('tool', encoding='utf-8')
        self.optimizer.write_text('tool', encoding='utf-8')
        self.output = self.root / 'out/model.mlir'

    def run_pipeline(self, mlir, supported=(), dimensions=(), input_shapes=(),
                     lower_to_host=False):
        def run(command, check):
            self.assertTrue(check)
            output = Path(command[command.index('-o') + 1])
            if isinstance(mlir, bytes):
                output.write_bytes(mlir)
            else:
                output.write_text(mlir, encoding='utf-8')
            return mock.Mock(returncode=0)

        with mock.patch.object(compiler.subprocess, 'run', side_effect=run) as call:
            compiler.canonicalize(
                self.model, self.output, self.importer, self.optimizer,
                supported, dimensions, input_shapes, lower_to_host)
        return call.call_args_list

    def test_pipeline_imports_then_canonicalizes(self):
        calls = self.run_pipeline(
            '"onnx.Add"() : () -> ()\nonnx.Return\n', ['Add'])
        self.assertEqual(len(calls), 2)
        optimizer = calls[1].args[0]
        self.assertIn('--shape-inference', optimizer)
        self.assertIn('--simplify-shape-related-ops-onnx', optimizer)
        self.assertIn('--constprop-onnx', optimizer)
        self.assertIn('--standard-func-return', optimizer)
        self.assertIn('--mlir-elide-elementsattrs-if-larger=64', optimizer)
        self.assertIn('--canonicalize', optimizer)
        self.assertIn('--mlir-print-debuginfo', optimizer)
        self.assertTrue(self.output.is_file())

    def test_dimension_values_are_forwarded_to_the_importer(self):
        calls = self.run_pipeline(
            'onnx.Return\n', dimensions=['batch_size=1', 'past_sequence_length=0']
        )
        importer = calls[0].args[0]
        self.assertEqual(importer.count('--dim'), 2)
        self.assertIn('batch_size=1', importer)
        self.assertIn('past_sequence_length=0', importer)

    def test_host_lowering_is_entirely_in_the_cpp_pass_pipeline(self):
        calls = self.run_pipeline(b'host bytecode', lower_to_host=True)
        optimizer = calls[1].args[0]
        self.assertIn('--udoncl-infer-onnx-shapes', optimizer)
        self.assertIn('--udoncl-materialize-onnx-constants', optimizer)
        self.assertIn('--udoncl-lower-onnx-to-host', optimizer)
        self.assertIn(
            '--one-shot-bufferize=bufferize-function-boundaries copy-before-write',
            optimizer)
        self.assertIn('--udoncl-emit-bytecode', optimizer)
        self.assertNotIn('--mlir-elide-elementsattrs-if-larger=64', optimizer)
        self.assertEqual(self.output.read_bytes(), b'host bytecode')

    def test_input_shapes_are_forwarded_to_the_importer(self):
        calls = self.run_pipeline(
            'onnx.Return\n', input_shapes=['attention_mask=1,4']
        )
        importer = calls[0].args[0]
        self.assertIn('--input-shape', importer)
        self.assertIn('attention_mask=1,4', importer)

    def test_unsupported_operator_is_explicit_and_output_is_atomic(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_text('old output', encoding='utf-8')
        with self.assertRaisesRegex(
                RuntimeError, 'Unsupported ONNX operators.*Mystery'):
            self.run_pipeline(
                '"onnx.Add"() : () -> ()\n"onnx.Mystery"() : () -> ()\n',
                ['Add'])
        self.assertEqual(self.output.read_text(encoding='utf-8'), 'old output')

    def test_operator_scan_ignores_structural_onnx_ops(self):
        found = compiler.operators(
            '"onnx.EntryPoint"()\nonnx.Return\nonnx.Constant\nonnx.NoValue\n'
            '"onnx.MatMulNBits"() {onnx.name = "input"}\n')
        self.assertEqual(found, {'MatMulNBits'})


if __name__ == '__main__':
    unittest.main()
