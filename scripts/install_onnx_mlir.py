"""Build and publish the locked ONNX-MLIR SDK used by repo-owned tools.

The published prefix is immutable and identified by a receipt. Source and build
directories are retained outside the prefix so interrupted builds can resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / 'cmake/onnx-mlir-lock.json'
RECEIPT = '.udoncl-onnx-mlir.json'
SDK_CONFIG = ROOT / 'cmake/ONNXMLIRConfig.cmake'
PROJECT_INCLUDE = ROOT / 'cmake/onnx-mlir-project-include.cmake'
# LLVM tools that ONNX-MLIR's lit suites name as target dependencies. The
# suites are declared even with ONNX_MLIR_BUILD_TESTS=OFF, but the official
# Linux LLVM release archive does not export every tool.
LIT_TOOL_TARGETS = ('FileCheck', 'count', 'not', 'split-file', 'llvm-dis')
# Umbrella over ONNX-MLIR's public TableGen targets, from PROJECT_INCLUDE.
TABLEGEN_TARGET = 'udoncl-onnx-mlir-tablegen'


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


def patch_digest(lock, root=ROOT):
    digest = hashlib.sha256()
    for relative in lock['patches']:
        digest.update(relative.encode())
        digest.update(b'\0')
        digest.update((root / relative).read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def lock_digest(lock):
    build_lock = {key: value for key, value in lock.items()
                  if key != 'compatible_receipts'}
    encoded = json.dumps(
        build_lock, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def sdk_config_digest(root=ROOT):
    return hashlib.sha256((root / 'cmake/ONNXMLIRConfig.cmake').read_bytes()).hexdigest()


def expected_receipt(lock, host, root=ROOT):
    return {
        'repository': lock['repository'],
        'revision': lock['revision'],
        'llvm_version': lock['llvm_version'],
        'platform': host,
        'lock_sha256': lock_digest(lock),
        'patch_sha256': patch_digest(lock, root),
        'build_targets': lock['build_targets'],
        'sdk_version': lock['sdk_version'],
        'sdk_config_sha256': sdk_config_digest(root),
    }


def sdk_components(root=ROOT):
    """Return the static libraries that ONNXMLIRConfig.cmake links."""
    config = root / 'cmake/ONNXMLIRConfig.cmake'
    match = re.search(r'set\(_onnxmlir_component_names\s+([^)]*)\)',
                      config.read_text(encoding='utf-8'))
    if not match:
        raise RuntimeError(f'{config} does not list its SDK components')
    return match.group(1).split()


def sdk_library_name(component, host):
    if component == 'libprotobuf':
        return 'libprotobuf.lib' if host == 'windows' else 'libprotobuf.a'
    return f'{component}.lib' if host == 'windows' else f'lib{component}.a'


def copy_sdk_headers(source, destination):
    extensions = {'.def', '.h', '.hpp', '.inc', '.td'}
    for path in source.rglob('*'):
        if not path.is_file() or path.suffix not in extensions:
            continue
        output = destination / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)


def publishable_sdk_library(path):
    """Reject executable import libraries left by an older shared build tree."""
    return path.stem not in {'udoncl-onnx', 'onnx-mlir-opt'}


def validate(prefix, lock, host):
    marker = prefix / RECEIPT
    sdk_required = [
        prefix / 'include/onnx-mlir/src/Builder/FrontendDialectTransformer.hpp',
        prefix / 'include/onnx-mlir/src/Dialect/ONNX/ONNXOps.hpp',
        prefix / 'include/onnx-mlir/src/Dialect/ONNX/ONNXOps.hpp.inc',
        prefix / 'include/onnx-mlir/src/Pass/Passes.hpp',
        prefix / 'include/onnx-mlir/third_party/onnx/onnx/onnx_pb.h',
        prefix / 'include/onnx-mlir/build/third_party/onnx/onnx/onnx-ml.pb.h',
        prefix / 'include/onnx-mlir/build/_deps/protobuf-src/src/google/protobuf/message.h',
        prefix / 'lib/cmake/ONNXMLIR/ONNXMLIRConfig.cmake',
    ]
    for component in sdk_components():
        sdk_required.append(prefix / 'lib' / sdk_library_name(component, host))
    missing_sdk = [path for path in sdk_required if not path.is_file()]
    if missing_sdk:
        raise RuntimeError(
            'Incomplete ONNX-MLIR SDK: missing ' +
            ', '.join(str(path) for path in missing_sdk))
    if not marker.is_file():
        raise RuntimeError(f'ONNX-MLIR frontend has no install receipt: {marker}')
    receipt = json.loads(marker.read_text(encoding='utf-8'))
    if (receipt != expected_receipt(lock, host)
            and receipt not in lock.get('compatible_receipts', [])):
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


def exported_llvm_executables(llvm_prefix):
    exports = llvm_prefix / 'lib/cmake/llvm/LLVMExports.cmake'
    if not exports.is_file():
        raise RuntimeError(f'LLVM/MLIR SDK has no CMake exports: {exports}')
    return set(re.findall(r'add_executable\(\s*([^\s)]+)\s+IMPORTED',
                          exports.read_text(encoding='utf-8')))


def missing_lit_tools(llvm_prefix):
    exported = exported_llvm_executables(llvm_prefix)
    return [tool for tool in LIT_TOOL_TARGETS if tool not in exported]


def llvm_enables_rtti(llvm_prefix):
    config = llvm_prefix / 'lib/cmake/llvm/LLVMConfig.cmake'
    if not config.is_file():
        raise RuntimeError(f'LLVM/MLIR SDK has no CMake config: {config}')
    match = re.search(r'^set\(LLVM_ENABLE_RTTI\s+(\S+)\)',
                      config.read_text(encoding='utf-8'), re.MULTILINE)
    if not match:
        raise RuntimeError(f'{config} does not record LLVM_ENABLE_RTTI')
    return match.group(1).upper() in {'1', 'ON', 'YES', 'TRUE', 'Y'}


def configure_command(lock, source, build, llvm_prefix, host, cmake,
                      lit_tool_stubs=(), llvm_rtti=True):
    command = [cmake, '-S', source, '-B', build, '-G', 'Ninja',
               f'-DMLIR_DIR={llvm_prefix / "lib/cmake/mlir"}',
               '-DCMAKE_BUILD_TYPE=Release']
    command += [f'-D{name}={value}'
                for name, value in lock['cmake_options'].items()]
    if host == 'windows':
        command.append('-DCMAKE_CXX_FLAGS=/D_USE_MATH_DEFINES')
    elif not llvm_rtti:
        # ONNX-MLIR sets LLVM_REQUIRES_EH, which makes LLVM's build flags keep
        # the compiler's default RTTI. Its passes and dialects derive from
        # MLIR classes, so with RTTI they reference typeinfo symbols that an
        # SDK built without RTTI (the official Linux release archive) does
        # not define, and every consumer fails to link. Nothing built here
        # uses dynamic_cast or typeid, and protobuf and Abseil detect -fno-rtti.
        command.append('-DCMAKE_CXX_FLAGS=-fno-rtti')
    # The project include stubs lit tools the SDK lacks, so the unused
    # check-* suites stay generatable.
    command += [
        f'-DCMAKE_PROJECT_onnx-mlir_INCLUDE={PROJECT_INCLUDE.as_posix()}',
        '-DUDONCL_ONNX_MLIR_LIT_TOOL_STUBS=' + ';'.join(lit_tool_stubs)]
    return command


def build_commands(lock, build, cmake, jobs):
    # Building a static library does not build the static libraries it
    # links, so also name every library that ONNXMLIRConfig.cmake links.
    libraries = [*lock['build_targets'],
                 *(component for component in sdk_components()
                   if component not in lock['build_targets'])]
    # Generate every TableGen header first: ONNX-MLIR libraries include
    # generated headers whose generator they do not depend on.
    return [[cmake, '--build', build, '--config', 'Release', '--parallel',
             str(jobs), '--target', *targets]
            for targets in ([TABLEGEN_TARGET], libraries)]


def install(lock, host, prefix, source, build, llvm_prefix, jobs):
    if prefix.exists():
        validate(prefix, lock, host)
        print(f'Using previously verified ONNX-MLIR SDK: {prefix}')
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
    run(configure_command(lock, source, build, llvm_prefix, host, cmake,
                          missing_lit_tools(llvm_prefix),
                          llvm_enables_rtti(llvm_prefix)))
    for command in build_commands(lock, build, cmake, jobs):
        run(command)
    library_root = build / 'Release/lib'
    if not library_root.is_dir():
        library_root = build / 'lib'
    library_suffix = '.lib' if host == 'windows' else '.a'
    built_libraries = sorted(library_root.glob(f'*{library_suffix}'))
    dialect_library = ('OMONNXOps.lib' if host == 'windows'
                       else 'libOMONNXOps.a')
    if not any(path.name == dialect_library
               for path in built_libraries):
        raise RuntimeError('ONNX-MLIR build did not produce the dialect SDK')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=prefix.name + '.publish-', dir=prefix.parent) as temporary:
        publication = Path(temporary)
        include_root = publication / 'include/onnx-mlir'
        copy_sdk_headers(source / 'src', include_root / 'src')
        # Preserve ONNX-MLIR's virtual source root. Its internal SDK headers
        # include both `src/...` and `include/onnx-mlir/...` paths.
        copy_sdk_headers(source / 'include', include_root / 'include')
        copy_sdk_headers(build / 'src', include_root / 'src')
        copy_sdk_headers(source / 'third_party/onnx',
                         include_root / 'third_party/onnx')
        copy_sdk_headers(build / 'third_party/onnx',
                         include_root / 'build/third_party/onnx')
        copy_sdk_headers(build / '_deps/protobuf-src/src',
                         include_root / 'build/_deps/protobuf-src/src')
        copy_sdk_headers(build / '_deps/absl-src',
                         include_root / 'build/_deps/absl-src')
        library_destination = publication / 'lib'
        library_destination.mkdir()
        for built in built_libraries:
            if not publishable_sdk_library(built):
                continue
            shutil.copy2(built, library_destination / built.name)
        config_destination = library_destination / 'cmake/ONNXMLIR'
        config_destination.mkdir(parents=True)
        shutil.copy2(SDK_CONFIG,
                     config_destination / 'ONNXMLIRConfig.cmake')
        (publication / RECEIPT).write_text(
            json.dumps(expected_receipt(lock, host), indent=2) + '\n',
            encoding='utf-8')
        validate(publication, lock, host)
        publication.rename(prefix)
    print(f'Installed ONNX-MLIR SDK: {prefix}')


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
            print(f'Validated ONNX-MLIR SDK: {args.prefix.resolve()}')
        else:
            install(lock, args.platform, args.prefix.resolve(),
                    args.source_dir.resolve(), args.build_dir.resolve(),
                    args.llvm_prefix.resolve(), args.jobs)
    except (OSError, RuntimeError, ValueError,
            subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
