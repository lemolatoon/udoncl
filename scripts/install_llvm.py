"""Install the locked prebuilt LLVM/MLIR SDK; never builds LLVM from source.

Windows uses Portable MLIR Toolchain for LLVM/MLIR and only the clang executable
and resource directory from the official LLVM archive. Their CMake packages and
link libraries are never mixed. Existing installations are never overwritten.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / 'cmake/llvm-lock.json'
RECEIPT = '.udoncl-sdk.json'
TOOLS = ('clang', 'opt', 'llvm-tblgen', 'mlir-tblgen', 'mlir-opt', 'mlir-translate')


def components(lock, host):
    keys = [f'{host}_x64_archive']
    if host == 'windows':
        keys.append('windows_x64_clang_archive')
    try:
        return {key: lock[key] for key in keys}
    except KeyError as error:
        raise RuntimeError(f'No prebuilt SDK is locked for {host} x64') from error


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(spec, directory):
    name = spec['url'].rsplit('/', 1)[-1]
    path = directory / name
    directory.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + '.part')
        try:
            print(f'Downloading {spec["url"]}', flush=True)
            with urllib.request.urlopen(spec['url'], timeout=120) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    actual = digest(path)
    if actual != spec['sha256']:
        raise RuntimeError(f'SHA256 mismatch for {path}: expected {spec["sha256"]}, got {actual}; file left unchanged')
    return path


@contextlib.contextmanager
def open_archive(path):
    if path.name.endswith('.zst'):
        try:
            import zstandard
        except ImportError as error:
            raise RuntimeError('Install extraction dependency: python -m pip install -r scripts/requirements-llvm.txt') from error
        # Official archives use long-distance matching with a 1 GiB window.
        # Stream directly to tarfile instead of materializing an uncompressed tar.
        with path.open('rb') as compressed:
            with zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(compressed) as stream:
                with tarfile.open(fileobj=stream, mode='r|') as archive:
                    yield archive
    else:
        with tarfile.open(path, mode='r|*') as archive:
            yield archive


def extract(path, directory, clang_only=False, major='23'):
    with open_archive(path) as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            # Official archives have one top-level directory; Portable MLIR's
            # archive is rooted at ./bin, ./include, ./lib.
            relative = parts if parts and parts[0] in ('bin', 'include', 'lib') else parts[1:]
            if clang_only and not (relative == ('bin', 'clang.exe') or
                                   relative[:3] == ('lib', 'clang', major)):
                continue
            archive.extract(member, directory, filter='data')


def find_prefix(directory, required):
    if (directory / required).is_file():
        return directory
    children = [child for child in directory.iterdir()
                if child.is_dir() and (child / required).is_file()]
    if len(children) != 1:
        raise RuntimeError(f'Archive does not contain exactly one SDK prefix with {required}')
    return children[0]


def validate(prefix, version, host):
    suffix = '.exe' if host == 'windows' else ''
    libraries = ['LLVMCore.lib', 'MLIRIR.lib'] if host == 'windows' else ['libLLVMCore.a', 'libMLIRIR.a']
    required = ['lib/cmake/llvm/LLVMConfig.cmake', 'lib/cmake/mlir/MLIRConfig.cmake',
                'include/llvm/Config/llvm-config.h', 'include/llvm/IR/Module.h',
                'include/mlir/IR/MLIRContext.h']
    required += [f'bin/{tool}{suffix}' for tool in TOOLS]
    required += [f'lib/{name}' for name in libraries]
    required += [f'lib/clang/{version.split(".")[0]}/include/opencl-c.h']
    missing = [name for name in required if not (prefix / name).is_file()]
    if missing:
        raise RuntimeError('Incomplete LLVM/MLIR SDK: ' + ', '.join(missing))
    header = (prefix / 'include/llvm/Config/llvm-config.h').read_text(encoding='utf-8')
    if not re.search(r'^#define LLVM_VERSION_STRING "' + re.escape(version) + r'"', header, re.MULTILINE):
        raise RuntimeError(f'SDK headers do not match LLVM {version}')


def install(lock, host, prefix, downloads, discard_downloads=False):
    selected = components(lock, host)
    receipt = {'version': lock['version'], 'platform': host,
               'archives': {key: {'url': spec['url'], 'sha256': spec['sha256']}
                            for key, spec in selected.items()}}
    if prefix.exists():
        validate(prefix, lock['version'], host)
        marker = prefix / RECEIPT
        if not marker.is_file() or json.loads(marker.read_text()) != receipt:
            raise RuntimeError(f'{prefix} exists without a matching install receipt; use --check to validate it or choose a new --prefix')
        print(f'Using previously verified LLVM/MLIR SDK: {prefix}')
        return
    # All archives must pass SHA256 verification before any extraction begins.
    archives = {key: download(spec, downloads) for key, spec in selected.items()}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=prefix.name + '.unpack-', dir=prefix.parent) as temporary:
        temporary = Path(temporary)
        sdk_directory = temporary / 'sdk'
        sdk_directory.mkdir()
        print('Extracting LLVM/MLIR SDK', flush=True)
        extract(archives[f'{host}_x64_archive'], sdk_directory)
        sdk = find_prefix(sdk_directory, 'lib/cmake/mlir/MLIRConfig.cmake')
        if host == 'windows':
            clang_directory = temporary / 'clang'
            clang_directory.mkdir()
            print('Extracting official Clang executable and resource headers', flush=True)
            extract(archives['windows_x64_clang_archive'], clang_directory,
                    clang_only=True, major=lock['version'].split('.')[0])
            clang = find_prefix(clang_directory, 'bin/clang.exe')
            shutil.copy2(clang / 'bin/clang.exe', sdk / 'bin/clang.exe')
            shutil.copytree(clang / 'lib/clang', sdk / 'lib/clang', dirs_exist_ok=True)
        validate(sdk, lock['version'], host)
        (sdk / RECEIPT).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        sdk.rename(prefix)
    if discard_downloads:
        for archive in archives.values():
            archive.unlink()
    print(f'Installed LLVM/MLIR {lock["version"]}: {prefix}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=ROOT / '.deps/llvm23')
    parser.add_argument('--download-dir', type=Path, default=ROOT / '.deps')
    parser.add_argument('--platform', choices=['linux', 'windows'],
                        default='windows' if platform.system() == 'Windows' else 'linux',
                        help='Archive platform; overriding permits content validation without executing its tools')
    parser.add_argument('--discard-downloads', action='store_true', help='Remove verified archives after a successful fresh install (CI disk space)')
    parser.add_argument('--check', action='store_true', help='Validate an existing SDK without downloading or changing it')
    args = parser.parse_args()
    lock = json.loads(LOCK_PATH.read_text())
    try:
        if platform.machine().lower() not in ('x86_64', 'amd64') or platform.system() not in ('Linux', 'Windows'):
            raise RuntimeError('Prebuilt installer supports Linux/Windows x64; use scripts/build_llvm.py on other hosts')
        if args.check:
            validate(args.prefix, lock['version'], args.platform)
            marker = args.prefix / RECEIPT
            if marker.is_file():
                receipt = json.loads(marker.read_text())
                selected = components(lock, args.platform)
                expected = {'version': lock['version'], 'platform': args.platform,
                            'archives': {key: {'url': spec['url'], 'sha256': spec['sha256']}
                                         for key, spec in selected.items()}}
                if receipt != expected:
                    raise RuntimeError('SDK install receipt does not match the locked archives')
            print(f'Validated LLVM/MLIR {lock["version"]}: {args.prefix}')
        else:
            install(lock, args.platform, args.prefix.resolve(), args.download_dir.resolve(), args.discard_downloads)
    except (OSError, RuntimeError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
