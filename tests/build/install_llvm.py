"""Check verified archive installation, SDK isolation, and failure atomicity."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('installer', ROOT / 'scripts/install_llvm.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='udoncl SDK test ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = self.root / 'installed SDK'
        self.downloads = self.root / 'downloads'

    def archive(self, name, files):
        path = self.root / name
        with tarfile.open(path, 'w:gz') as archive:
            for name, content in files.items():
                entry = tarfile.TarInfo(name)
                data = content.encode()
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
        return {'url': path.as_uri(), 'sha256': installer.digest(path)}

    def sdk(self, host, clang=True):
        suffix = '.exe' if host == 'windows' else ''
        libraries = ('LLVMCore.lib', 'MLIRIR.lib') if host == 'windows' else ('libLLVMCore.a', 'libMLIRIR.a')
        files = {name: 'SDK' for name in ['lib/cmake/llvm/LLVMConfig.cmake', 'lib/cmake/mlir/MLIRConfig.cmake',
                                        'include/llvm/IR/Module.h', 'include/mlir/IR/MLIRContext.h']}
        files['include/llvm/Config/llvm-config.h'] = '#define LLVM_VERSION_STRING "23.1.4"\n'
        files.update({f'lib/{name}': 'portable libraries' for name in libraries})
        files.update({f'bin/{name}{suffix}': 'SDK tool' for name in installer.TOOLS if clang or name != 'clang'})
        if clang: files['lib/clang/23/include/opencl-c.h'] = 'OpenCL resource header'
        return files

    def test_linux_install_and_reuse(self):
        files = {'LLVM-release/' + name: value for name, value in self.sdk('linux').items()}
        lock = {'version': '23.1.4', 'linux_x64_archive': self.archive('linux.tar.gz', files)}
        installer.install(lock, 'linux', self.prefix, self.downloads)
        installer.validate(self.prefix, '23.1.4', 'linux')
        installer.install(lock, 'linux', self.prefix, self.downloads)
        receipt = json.loads((self.prefix / installer.RECEIPT).read_text())
        self.assertEqual(receipt['archives']['linux_x64_archive']['sha256'], lock['linux_x64_archive']['sha256'])
        self.assertFalse(list(self.root.glob('installed SDK.unpack-*')))

    def test_windows_clang_overlay_keeps_sdk_libraries(self):
        core = self.archive('windows.tar.gz', {'./' + name: value for name, value in self.sdk('windows', clang=False).items()})
        clang = self.archive('clang.tar.gz', {'official/bin/clang.exe': 'official clang',
            'official/lib/clang/23/include/opencl-c.h': 'official resource header',
            'official/lib/LLVMCore.lib': 'MUST NOT OVERWRITE',
            'official/lib/cmake/llvm/LLVMConfig.cmake': 'MUST NOT OVERWRITE',
            'official/bin/opt.exe': 'MUST NOT OVERWRITE'})
        lock = {'version': '23.1.4', 'windows_x64_archive': core, 'windows_x64_clang_archive': clang}
        installer.install(lock, 'windows', self.prefix, self.downloads)
        self.assertEqual((self.prefix / 'bin/clang.exe').read_text(), 'official clang')
        self.assertEqual((self.prefix / 'bin/opt.exe').read_text(), 'SDK tool')
        self.assertEqual((self.prefix / 'lib/LLVMCore.lib').read_text(), 'portable libraries')
        self.assertEqual((self.prefix / 'lib/cmake/llvm/LLVMConfig.cmake').read_text(), 'SDK')
        self.assertEqual((self.prefix / 'lib/clang/23/include/opencl-c.h').read_text(), 'official resource header')

    def test_sha256_mismatch_never_extracts(self):
        archive = self.archive('bad-hash.tar.gz', self.sdk('linux'))
        archive['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
            installer.install({'version': '23.1.4', 'linux_x64_archive': archive}, 'linux', self.prefix, self.downloads)
        self.assertFalse(self.prefix.exists())
        self.assertFalse(list(self.root.glob('installed SDK.unpack-*')))

    def test_incomplete_archive_never_published(self):
        archive = self.archive('incomplete.tar.gz', {'lib/cmake/mlir/MLIRConfig.cmake': 'MLIR'})
        with self.assertRaisesRegex(RuntimeError, 'Incomplete LLVM/MLIR SDK'):
            installer.install({'version': '23.1.4', 'linux_x64_archive': archive}, 'linux', self.prefix, self.downloads)
        self.assertFalse(self.prefix.exists())
        self.assertFalse(list(self.root.glob('installed SDK.unpack-*')))

    def test_archive_traversal_rejected(self):
        archive = self.archive('traversal.tar.gz', {'../../escaped': 'MUST NOT WRITE'})
        with self.assertRaises(tarfile.FilterError):
            installer.install({'version': '23.1.4', 'linux_x64_archive': archive}, 'linux', self.prefix, self.downloads)
        self.assertFalse((self.root / 'escaped').exists())
        self.assertFalse(self.prefix.exists())

    def test_existing_unmanaged_sdk_left_unchanged(self):
        for name, value in self.sdk('linux').items():
            path = self.prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        archive = self.archive('sdk.tar.gz', self.sdk('linux'))
        with self.assertRaisesRegex(RuntimeError, 'matching install receipt'):
            installer.install({'version': '23.1.4', 'linux_x64_archive': archive}, 'linux', self.prefix, self.downloads)
        self.assertEqual((self.prefix / 'bin/clang').read_text(), 'SDK tool')
        self.assertFalse(self.downloads.exists())

    def test_ci_discard_removes_download_only_after_success(self):
        archive = self.archive('discard.tar.gz', self.sdk('linux'))
        installer.install({'version': '23.1.4', 'linux_x64_archive': archive},
                          'linux', self.prefix, self.downloads, discard_downloads=True)
        self.assertTrue(self.prefix.is_dir())
        self.assertFalse((self.downloads / 'discard.tar.gz').exists())
        self.assertTrue((self.root / 'discard.tar.gz').is_file())

    def test_changed_receipt_prevents_cache_reuse(self):
        archive = self.archive('receipt.tar.gz', self.sdk('linux'))
        lock = {'version': '23.1.4', 'linux_x64_archive': archive}
        installer.install(lock, 'linux', self.prefix, self.downloads)
        marker = self.prefix / installer.RECEIPT
        receipt = json.loads(marker.read_text())
        receipt['archives']['linux_x64_archive']['sha256'] = '0' * 64
        marker.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(RuntimeError, 'matching install receipt'):
            installer.install(lock, 'linux', self.prefix, self.downloads)
        self.assertEqual((self.prefix / 'bin/clang').read_text(), 'SDK tool')

    def test_header_version_mismatch(self):
        files = self.sdk('linux')
        files['include/llvm/Config/llvm-config.h'] = '#define LLVM_VERSION_STRING "22.1.0"\n'
        archive = self.archive('old-version.tar.gz', files)
        with self.assertRaisesRegex(RuntimeError, 'headers do not match'):
            installer.install({'version': '23.1.4', 'linux_x64_archive': archive}, 'linux', self.prefix, self.downloads)
        self.assertFalse(self.prefix.exists())


if __name__ == '__main__':
    unittest.main()
