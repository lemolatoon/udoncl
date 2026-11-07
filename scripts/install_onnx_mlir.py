"""Build the locked ONNX-MLIR frontend against UdonCL's LLVM/MLIR SDK.

The published prefix is immutable and identified by a receipt. Source and build
directories are retained outside the prefix so interrupted builds can resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / 'cmake/onnx-mlir-lock.json'
RECEIPT = '.udoncl-onnx-mlir.json'


def run(command, cwd=None):
    print(' '.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def ensure_windows_toolchain():
    if sys.platform != 'win32':
        return
    program_files = Path(os.environ.get('ProgramFiles', r'C:\Program Files'))
    if not shutil.which('cmake') or not shutil.which('ninja'):
        for cmake_dir in program_files.glob(
                'Microsoft Visual Studio/*/*/Common7/IDE/CommonExtensions/'
                'Microsoft/CMake/CMake/bin'):
            ninja_dir = cmake_dir.parent.parent / 'Ninja'
            if ((cmake_dir / 'cmake.exe').is_file() and
                    (ninja_dir / 'ninja.exe').is_file()):
                os.environ['PATH'] = os.pathsep.join(
                    [str(cmake_dir), str(ninja_dir),
                     os.environ.get('PATH', '')])
                break
    if os.environ.get('VCToolsInstallDir'):
        return
    program_files_x86 = Path(os.environ.get(
        'ProgramFiles(x86)', r'C:\Program Files (x86)'))
    vswhere = (program_files_x86 / 'Microsoft Visual Studio/Installer/'
               'vswhere.exe')
    if not vswhere.is_file():
        raise RuntimeError(f'Visual Studio locator is missing: {vswhere}')
    visual_studio = subprocess.check_output([
        vswhere, '-latest', '-prerelease', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
        '-property', 'installationPath'], text=True).strip()
    if not visual_studio:
        raise RuntimeError('Visual Studio C++ tools are required on Windows')
    vsdevcmd = Path(visual_studio) / 'Common7/Tools/VsDevCmd.bat'
    with tempfile.TemporaryDirectory(prefix='udoncl-onnx-vsdevcmd-') as temporary:
        batch = Path(temporary) / 'environment.cmd'
        batch.write_text(
            f'@call "{vsdevcmd}" -no_logo -arch=x64 -host_arch=x64 >nul\n'
            '@set\n', encoding='utf-8')
        result = subprocess.run(
            ['cmd.exe', '/d', '/c', batch], check=True, text=True,
            capture_output=True)
    for line in result.stdout.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value


def patch_digest(lock):
    digest = hashlib.sha256()
    for relative in lock['patches']:
        digest.update(relative.encode())
        digest.update(b'\0')
        digest.update((ROOT / relative).read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def extension_digest(lock):
    digest = hashlib.sha256()
    for relative in lock.get('extension_sources', []):
        digest.update(relative.encode())
        digest.update(b'\0')
        digest.update((ROOT / relative).read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def lock_digest(lock):
    encoded = json.dumps(lock, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def expected_receipt(lock, host):
    return {
        'repository': lock['repository'],
        'revision': lock['revision'],
        'llvm_version': lock['llvm_version'],
        'platform': host,
        'lock_sha256': lock_digest(lock),
        'patch_sha256': patch_digest(lock),
        'extension_sha256': extension_digest(lock),
        'targets': lock['targets'],
    }


def executable(prefix, host, name='onnx-mlir-opt'):
    suffix = '.exe' if host == 'windows' else ''
    return prefix / 'bin' / f'{name}{suffix}'


def validate(prefix, lock, host):
    marker = prefix / RECEIPT
    missing = [executable(prefix, host, name) for name in lock['targets']
               if not executable(prefix, host, name).is_file()]
    if missing:
        raise RuntimeError(
            'Incomplete ONNX-MLIR frontend: missing ' +
            ', '.join(str(path) for path in missing))
    if not marker.is_file():
        raise RuntimeError(f'ONNX-MLIR frontend has no install receipt: {marker}')
    if json.loads(marker.read_text(encoding='utf-8')) != expected_receipt(lock, host):
        raise RuntimeError('ONNX-MLIR install receipt does not match the lock file')


def ensure_source(lock, source):
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', '--filter=blob:none', '--no-checkout',
             lock['repository'], source])
        run(['git', 'checkout', '--detach', lock['revision']], cwd=source)
    else:
        try:
            revision = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
            remote = subprocess.check_output(
                ['git', 'remote', 'get-url', 'origin'], cwd=source,
                text=True).strip()
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'{source} is not an ONNX-MLIR checkout') from error
        if revision != lock['revision'] or remote != lock['repository']:
            raise RuntimeError(
                f'{source} does not match locked ONNX-MLIR {lock["revision"]}')

    # The source checkout is a disposable cache. Recreate the complete locked
    # patch stack from its pinned revision instead of trying to recognize each
    # patch independently in a tree modified by later patches.
    run(['git', 'config', 'core.autocrlf', 'false'], cwd=source)
    run(['git', 'reset', '--hard', lock['revision']], cwd=source)
    run(['git', 'clean', '-ffd'], cwd=source)

    paths = list(lock['submodules'])
    run(['git', 'submodule', 'sync', '--', *paths], cwd=source)
    run(['git', 'submodule', 'update', '--init', '--depth', '1', '--', *paths],
        cwd=source)
    for relative, expected in lock['submodules'].items():
        actual = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=source / relative,
            text=True).strip()
        if actual != expected:
            raise RuntimeError(
                f'ONNX-MLIR submodule {relative} is {actual}, expected {expected}')

    for relative in lock['patches']:
        patch = ROOT / relative
        check = subprocess.run(
            ['git', 'apply', '-C0', '--check', patch], cwd=source,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode != 0:
            raise RuntimeError(f'Cannot apply locked ONNX-MLIR patch {patch}')
        run(['git', 'apply', '-C0', patch], cwd=source)


def configure_command(lock, source, build, llvm_prefix, host, cmake):
    command = [cmake, '-S', source, '-B', build, '-G', 'Ninja',
               f'-DMLIR_DIR={llvm_prefix / "lib/cmake/mlir"}',
               f'-DUDONCL_ONNX_EXTENSION_SOURCE_DIR={ROOT / "lib/Conversion/ONNXToUdonHost"}',
               '-DCMAKE_BUILD_TYPE=Release']
    command += [f'-D{name}={value}'
                for name, value in lock['cmake_options'].items()]
    if host == 'windows':
        command.append('-DCMAKE_CXX_FLAGS=/D_USE_MATH_DEFINES')
    return command


def install(lock, host, prefix, source, build, llvm_prefix, jobs):
    if prefix.exists():
        validate(prefix, lock, host)
        print(f'Using previously verified ONNX-MLIR frontend: {prefix}')
        return
    llvm_receipt = llvm_prefix / '.udoncl-sdk.json'
    if not llvm_receipt.is_file():
        raise RuntimeError(
            f'Install the locked LLVM/MLIR SDK first: {llvm_receipt} is missing')
    llvm = json.loads(llvm_receipt.read_text(encoding='utf-8'))
    if llvm.get('version') != lock['llvm_version']:
        raise RuntimeError(
            f'ONNX-MLIR requires LLVM {lock["llvm_version"]}, got '
            f'{llvm.get("version", "unknown")}')
    ensure_windows_toolchain()
    cmake = shutil.which('cmake')
    ninja = shutil.which('ninja')
    if not cmake or not ninja:
        raise RuntimeError('cmake and ninja must be available on PATH')
    ensure_source(lock, source)
    run(configure_command(lock, source, build, llvm_prefix, host, cmake))
    run([cmake, '--build', build, '--config', 'Release', '--parallel',
         str(jobs), '--target', *lock['targets']])

    built_tools = []
    for name in lock['targets']:
        built = build / 'Release/bin' / executable(Path(), host, name).name
        if not built.is_file():
            built = build / 'bin' / executable(Path(), host, name).name
        if not built.is_file():
            raise RuntimeError(f'ONNX-MLIR build did not produce {built.name}')
        built_tools.append(built)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=prefix.name + '.publish-', dir=prefix.parent) as temporary:
        publication = Path(temporary)
        (publication / 'bin').mkdir()
        for built in built_tools:
            shutil.copy2(built, publication / 'bin' / built.name)
        (publication / RECEIPT).write_text(
            json.dumps(expected_receipt(lock, host), indent=2) + '\n',
            encoding='utf-8')
        validate(publication, lock, host)
        publication.rename(prefix)
    print(f'Installed ONNX-MLIR frontend: {prefix}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path,
                        default=ROOT / '.deps/onnx-mlir')
    parser.add_argument('--source-dir', type=Path,
                        default=ROOT / '.deps/onnx-mlir-src')
    parser.add_argument('--build-dir', type=Path,
                        default=ROOT / 'build/onnx-mlir')
    parser.add_argument('--llvm-prefix', type=Path,
                        default=ROOT / '.deps/llvm23')
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--platform', choices=['linux', 'windows'],
                        default='windows' if platform.system() == 'Windows'
                        else 'linux')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    lock = json.loads(LOCK_PATH.read_text(encoding='utf-8'))
    try:
        if platform.machine().lower() not in ('x86_64', 'amd64'):
            raise RuntimeError('ONNX-MLIR frontend supports x64 hosts')
        if args.check:
            validate(args.prefix.resolve(), lock, args.platform)
            print(f'Validated ONNX-MLIR frontend: {args.prefix.resolve()}')
        else:
            install(lock, args.platform, args.prefix.resolve(),
                    args.source_dir.resolve(), args.build_dir.resolve(),
                    args.llvm_prefix.resolve(), args.jobs)
    except (OSError, RuntimeError, ValueError,
            subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
