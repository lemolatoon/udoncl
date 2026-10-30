"""Build the pinned Clang/LLVM/MLIR on Linux, macOS or Windows.

Windows: run from an x64 Visual Studio Developer PowerShell with Ninja on PATH.
All commands use argument lists so paths containing spaces are supported.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'cmake/llvm-lock.json').read_text())

def run(*args):
    subprocess.run([str(a) for a in args], check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--configure-only', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    source = ROOT / '.deps/llvm-project'
    build = ROOT / '.deps/llvm-build'
    prefix = ROOT / '.deps/llvm23'
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        run('git', 'clone', '--depth', '1', '--branch', LOCK['tag'],
            LOCK['repository'], source)
    revision = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != LOCK['commit']:
        raise SystemExit(f'Expected LLVM {LOCK["commit"]}, found {revision}; checkout was left unchanged')
    run('cmake', '-S', source / 'llvm', '-B', build, '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_INSTALL_PREFIX={prefix}',
        '-DLLVM_ENABLE_PROJECTS=clang;mlir', '-DLLVM_TARGETS_TO_BUILD=Native',
        '-DLLVM_ENABLE_ASSERTIONS=ON', '-DLLVM_INCLUDE_TESTS=OFF',
        '-DMLIR_INCLUDE_TESTS=OFF', '-DCLANG_INCLUDE_TESTS=OFF',
        '-DLLVM_ENABLE_ZLIB=OFF', '-DLLVM_ENABLE_ZSTD=OFF',
        '-DLLVM_ENABLE_LIBXML2=OFF', '-DLLVM_ENABLE_LIBEDIT=OFF',
        '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL')
    if not args.configure_only:
        run('cmake', '--build', build, '--target', 'install', '--parallel', args.jobs)

if __name__ == '__main__':
    main()
