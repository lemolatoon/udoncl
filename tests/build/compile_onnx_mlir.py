"""Check the generic repo-owned ONNX frontend orchestration."""
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
        self.frontend = self.root / 'udoncl-onnx-frontend'
        self.frontend.write_text('tool', encoding='utf-8')
        self.output = self.root / 'out/model.mlir'

    def run_pipeline(self, content, dimensions=(), input_shapes=(),
                     lower_to_host=False, static_data_directory=None):
        def run(command, check):
            self.assertTrue(check)
            prefix = ('--udoncl-externalize-onnx-initializers='
                      'static-data-directory=')
            for argument in command:
                if str(argument).startswith(prefix):
                    sidecar = Path(str(argument)[len(prefix):])
                    sidecar.mkdir(parents=True)
                    (sidecar / 'tensor_0.bytes').write_bytes(b'weights')
            output = Path(command[command.index('-o') + 1])
            if isinstance(content, bytes):
                output.write_bytes(content)
            else:
                output.write_text(content, encoding='utf-8')
            return mock.Mock(returncode=0)

        with mock.patch.object(compiler.subprocess, 'run', side_effect=run) as call:
            compiler.canonicalize(
                self.model, self.output, self.frontend, dimensions,
                input_shapes, lower_to_host, static_data_directory)
        return call.call_args.args[0]

    def test_frontend_imports_and_runs_upstream_passes_in_one_process(self):
        command = self.run_pipeline('onnx.Return\n')
        self.assertEqual(command[0], self.frontend)
        self.assertEqual(command[1], self.model.resolve())
        for option in ('--shape-inference',
                       '--simplify-shape-related-ops-onnx',
                       '--constprop-onnx', '--standard-func-return',
                       '--canonicalize'):
            self.assertIn(option, command)
        self.assertIn('--mlir-elide-elementsattrs-if-larger=64', command)
        self.assertTrue(self.output.is_file())

    def test_shape_specialization_is_repo_owned(self):
        command = self.run_pipeline(
            'onnx.Return\n', dimensions=['batch_size=1'],
            input_shapes=['attention_mask=1,4'])
        option = next(value for value in command if str(value).startswith(
            '--udoncl-specialize-onnx-inputs='))
        self.assertIn('dimensions=batch_size=1', option)
        self.assertIn('input-shapes=attention_mask=1,4', option)
        self.assertIn('--udoncl-infer-onnx-shapes', command)

    def test_host_lowering_is_one_repo_owned_pipeline(self):
        command = self.run_pipeline(b'host bytecode', lower_to_host=True)
        for option in ('--udoncl-materialize-onnx-constants',
                       '--udoncl-identify-onnx-sequence-state',
                       '--udoncl-lower-onnx-to-host',
                       '--udoncl-validate-onnx-host-graph',
                       '--udoncl-plan-onnx-host-graph',
                       '--udoncl-materialize-onnx-host-globals',
                       '--udoncl-materialize-onnx-host-model-registry',
                       '--udoncl-finalize-onnx-host',
                       '--udoncl-cleanup-onnx-host-source',
                       '--emit-bytecode'):
            self.assertIn(option, command)
        self.assertLess(command.index('--udoncl-validate-onnx-host-graph'),
                        command.index('--udoncl-plan-onnx-host-graph'))
        self.assertLess(command.index('--udoncl-plan-onnx-host-graph'),
                        command.index(
                            '--udoncl-materialize-onnx-host-globals'))
        self.assertLess(
            command.index('--udoncl-materialize-onnx-host-globals'),
            command.index('--udoncl-materialize-onnx-host-model-registry'))
        self.assertLess(
            command.index('--udoncl-materialize-onnx-host-model-registry'),
                        command.index('--udoncl-finalize-onnx-host'))
        self.assertLess(command.index('--udoncl-finalize-onnx-host'),
                        command.index('--udoncl-cleanup-onnx-host-source'))
        self.assertEqual(self.output.read_bytes(), b'host bytecode')

    def test_host_lowering_moves_static_payloads_transactionally(self):
        sidecar = self.root / 'out/static-tensors'
        self.run_pipeline(b'host bytecode', lower_to_host=True,
                          static_data_directory=sidecar)
        self.assertEqual((sidecar / 'tensor_0.bytes').read_bytes(), b'weights')


if __name__ == '__main__':
    unittest.main()
