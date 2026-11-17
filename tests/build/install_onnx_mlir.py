"""Check the locked ONNX-MLIR frontend installer without building it."""
import importlib.util
import json
from pathlib import Path
import subprocess
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
        for relative in (
            'include/onnx-mlir/src/Builder/FrontendDialectTransformer.hpp',
            'include/onnx-mlir/src/Dialect/ONNX/ONNXOps.hpp',
            'include/onnx-mlir/src/Dialect/ONNX/ONNXOps.hpp.inc',
            'include/onnx-mlir/src/Pass/Passes.hpp',
            'include/onnx-mlir/third_party/onnx/onnx/onnx_pb.h',
            'include/onnx-mlir/build/third_party/onnx/onnx/onnx-ml.pb.h',
            'include/onnx-mlir/build/_deps/protobuf-src/src/google/protobuf/message.h',
            'lib/cmake/ONNXMLIR/ONNXMLIRConfig.cmake',
            'lib/OMBuilder.lib' if host == 'windows' else 'lib/libOMBuilder.a',
            'lib/OMCompilerOptions.lib' if host == 'windows' else 'lib/libOMCompilerOptions.a',
            'lib/OMCompilerDialects.lib' if host == 'windows' else 'lib/libOMCompilerDialects.a',
            'lib/OMONNXOps.lib' if host == 'windows' else 'lib/libOMONNXOps.a',
            'lib/OMShapeInferencePass.lib' if host == 'windows' else 'lib/libOMShapeInferencePass.a',
            'lib/OMONNXRewrite.lib' if host == 'windows' else 'lib/libOMONNXRewrite.a',
            'lib/OMONNXSimplifyShapeRelatedOps.lib' if host == 'windows' else 'lib/libOMONNXSimplifyShapeRelatedOps.a',
            'lib/OMONNXStandardFuncReturnPass.lib' if host == 'windows' else 'lib/libOMONNXStandardFuncReturnPass.a',
            'lib/onnx.lib' if host == 'windows' else 'lib/libonnx.a',
            'lib/libprotobuf.lib' if host == 'windows' else 'lib/libprotobuf.a',
        ):
            path = prefix / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('sdk', encoding='utf-8')
        (prefix / installer.RECEIPT).write_text(json.dumps(
            installer.expected_receipt(self.lock, host)), encoding='utf-8')
        return prefix

    def test_locked_frontend_receipt_validates(self):
        prefix = self.publish()
        installer.validate(prefix, self.lock, 'linux')
        self.assertEqual(self.lock['build_targets'], [
            'OMBuilder', 'OMCompilerOptions', 'OMCompilerDialects',
            'OMShapeInferencePass', 'OMONNXRewrite',
            'OMONNXSimplifyShapeRelatedOps', 'OMONNXStandardFuncReturnPass'])
        self.assertNotIn('extension_sources', self.lock)
        self.assertEqual(self.lock['sdk_version'], 3)
        self.assertNotIn(
            'cmake/patches/onnx-mlir-0.5.1.2-udoncl-plugin.patch',
            self.lock['patches'])
        self.assertFalse(
            (ROOT / 'cmake/patches/onnx-mlir-0.5.1.2-udoncl-plugin.patch').exists())

    def test_external_patches_do_not_own_udoncl_shape_semantics(self):
        for relative in self.lock['patches']:
            content = (ROOT / relative).read_text(encoding='utf-8')
            self.assertNotIn('specializeInputs', content)
            self.assertNotIn('MatMulNBits', content)

    def test_local_sources_are_not_part_of_frontend_receipt(self):
        before = installer.expected_receipt(self.lock, 'linux')
        for path in (
            ROOT / 'lib/Conversion/ONNXToUdonHost/FinalizeONNXHost.cpp',
            ROOT / 'lib/Conversion/ONNXToUdonHost/PlanONNXHostGraph.cpp',
            ROOT / 'lib/Conversion/ONNXToUdonHost/UdonCLLocalSources.cmake',
        ):
            self.assertNotIn(path.as_posix(), json.dumps(before))
        self.assertEqual(before, installer.expected_receipt(self.lock, 'linux'))

    def test_external_patch_change_changes_frontend_receipt(self):
        source = self.root / 'source'
        for relative in self.lock['patches']:
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
        config = source / 'cmake/ONNXMLIRConfig.cmake'
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_bytes((ROOT / 'cmake/ONNXMLIRConfig.cmake').read_bytes())
        before = installer.expected_receipt(self.lock, 'linux', source)
        patch = source / self.lock['patches'][0]
        patch.write_bytes(patch.read_bytes() + b'\n# changed\n')
        self.assertNotEqual(
            before, installer.expected_receipt(self.lock, 'linux', source))

    def test_external_target_list_change_changes_frontend_receipt(self):
        before = installer.expected_receipt(self.lock, 'linux')
        changed = dict(self.lock)
        changed['build_targets'] = [*self.lock['build_targets'], 'other-library']
        self.assertNotEqual(before, installer.expected_receipt(changed, 'linux'))

    def test_sdk_config_change_changes_frontend_receipt(self):
        source = self.root / 'source'
        for relative in self.lock['patches']:
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
        config = source / 'cmake/ONNXMLIRConfig.cmake'
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_bytes((ROOT / 'cmake/ONNXMLIRConfig.cmake').read_bytes())
        before = installer.expected_receipt(self.lock, 'linux', source)
        config.write_text('changed', encoding='utf-8')
        self.assertNotEqual(
            before, installer.expected_receipt(self.lock, 'linux', source))

    def test_sdk_header_publication_excludes_implementation_sources(self):
        source = self.root / 'headers'
        destination = self.root / 'published'
        (source / 'nested').mkdir(parents=True)
        (source / 'nested/api.hpp').write_text('header', encoding='utf-8')
        (source / 'nested/generated.inc').write_text('generated', encoding='utf-8')
        (source / 'nested/implementation.cpp').write_text('source', encoding='utf-8')
        installer.copy_sdk_headers(source, destination)
        self.assertTrue((destination / 'nested/api.hpp').is_file())
        self.assertTrue((destination / 'nested/generated.inc').is_file())
        self.assertFalse((destination / 'nested/implementation.cpp').exists())

    def test_sdk_header_layout_preserves_virtual_source_root(self):
        source = self.root / 'source/include'
        destination = self.root / 'sdk/include/onnx-mlir/include'
        header = source / 'onnx-mlir/Compiler/OMCompilerTypes.h'
        header.parent.mkdir(parents=True)
        header.write_text('header', encoding='utf-8')
        installer.copy_sdk_headers(source, destination)
        self.assertTrue((
            self.root /
            'sdk/include/onnx-mlir/include/onnx-mlir/Compiler/OMCompilerTypes.h'
        ).is_file())

    def test_sdk_publication_rejects_obsolete_tool_import_libraries(self):
        self.assertFalse(installer.publishable_sdk_library(
            Path('udoncl-onnx.lib')))
        self.assertFalse(installer.publishable_sdk_library(
            Path('onnx-mlir-opt.lib')))
        self.assertTrue(installer.publishable_sdk_library(
            Path('OMBuilder.lib')))

    def test_changed_revision_rejects_cached_frontend(self):
        prefix = self.publish()
        changed = dict(self.lock)
        changed['revision'] = '0' * 40
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            installer.validate(prefix, changed, 'linux')

    def test_missing_importer_library_rejects_cached_sdk(self):
        prefix = self.publish('windows')
        (prefix / 'lib/OMBuilder.lib').unlink()
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            installer.validate(prefix, self.lock, 'windows')

    def test_tool_only_prefix_is_not_an_sdk(self):
        prefix = self.publish()
        (prefix / 'lib/libOMONNXOps.a').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete ONNX-MLIR SDK'):
            installer.validate(prefix, self.lock, 'linux')

    def test_locked_patches_are_well_formed(self):
        for relative in self.lock['patches']:
            result = subprocess.run(
                ['git', 'apply', '--numstat', ROOT / relative],
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, (relative, result.stderr))

    def test_windows_configuration_has_required_compatibility_flag(self):
        command = installer.configure_command(
            self.lock, Path('source'), Path('build'), Path('llvm'),
            'windows', 'cmake')
        self.assertIn('-DMLIR_DIR=llvm\\lib\\cmake\\mlir', command)
        self.assertIn('-DCMAKE_CXX_FLAGS=/D_USE_MATH_DEFINES', command)
        self.assertIn('-DONNX_MLIR_ENABLE_STABLEHLO=OFF', command)
        self.assertIn('-DONNX_MLIR_BUILD_TESTS=OFF', command)
        self.assertFalse(any(
            str(part).startswith('-DUDONCL_ONNX_EXTENSION_SOURCE_DIR=')
            for part in command))

    def test_linux_configuration_does_not_use_msvc_flag(self):
        command = installer.configure_command(
            self.lock, Path('source'), Path('build'), Path('llvm'),
            'linux', 'cmake')
        self.assertFalse(any('_USE_MATH_DEFINES' in str(part)
                             for part in command))


if __name__ == '__main__':
    unittest.main()
