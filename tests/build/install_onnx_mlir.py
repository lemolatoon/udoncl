"""Check the locked ONNX-MLIR frontend installer without building it."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'onnx_mlir_installer', ROOT / 'scripts/install_onnx_mlir.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallONNXMLIRTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix='udoncl ONNX MLIR test ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lock = json.loads(installer.LOCK_PATH.read_text(encoding='utf-8'))

    def publish(self, host='linux'):
        prefix = self.root / 'frontend'
        (prefix / 'bin').mkdir(parents=True)
        for name in self.lock['targets']:
            installer.executable(prefix, host, name).write_text(
                'frontend', encoding='utf-8')
        (prefix / installer.RECEIPT).write_text(json.dumps(
            installer.expected_receipt(self.lock, host)), encoding='utf-8')
        return prefix

    def test_locked_frontend_receipt_validates(self):
        prefix = self.publish()
        installer.validate(prefix, self.lock, 'linux')
        self.assertEqual(self.lock['targets'],
                         ['onnx-mlir-opt', 'udoncl-onnx'])

    def test_changed_revision_rejects_cached_frontend(self):
        prefix = self.publish()
        changed = dict(self.lock)
        changed['revision'] = '0' * 40
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            installer.validate(prefix, changed, 'linux')

    def test_missing_tool_rejects_cached_frontend(self):
        prefix = self.publish('windows')
        installer.executable(prefix, 'windows', 'udoncl-onnx').unlink()
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            installer.validate(prefix, self.lock, 'windows')

    def test_windows_configuration_has_required_compatibility_flag(self):
        command = installer.configure_command(
            self.lock, Path('source'), Path('build'), Path('llvm'),
            'windows', 'cmake')
        self.assertIn('-DMLIR_DIR=llvm\\lib\\cmake\\mlir', command)
        self.assertIn('-DCMAKE_CXX_FLAGS=/D_USE_MATH_DEFINES', command)
        self.assertIn('-DONNX_MLIR_ENABLE_STABLEHLO=OFF', command)
        self.assertIn('-DONNX_MLIR_BUILD_TESTS=OFF', command)

    def test_linux_configuration_does_not_use_msvc_flag(self):
        command = installer.configure_command(
            self.lock, Path('source'), Path('build'), Path('llvm'),
            'linux', 'cmake')
        self.assertFalse(any('_USE_MATH_DEFINES' in str(part)
                             for part in command))


if __name__ == '__main__':
    unittest.main()
