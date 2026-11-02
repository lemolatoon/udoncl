"""Compile real C/OpenCL C through LLVM/MLIR to a boundary shader bundle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=root / 'build/udoncl')
    parser.add_argument('--output', type=Path, default=root / 'build/c-boundary')
    parser.add_argument('--clang', type=Path, help='Override the configured SDK clang')
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    suffix = '.exe' if sys.platform == 'win32' else ''
    clang = args.clang
    if clang is None:
        cache = args.build_dir / 'CMakeCache.txt'
        if not cache.is_file():
            parser.error('configure the project first with cmake --preset udoncl')
        entries = dict(line.split('=', 1) for line in cache.read_text().splitlines()
                       if '=' in line and not line.startswith(('#', '//')))
        llvm_dir = next((v for k, v in entries.items() if k.startswith('LLVM_DIR:')), None)
        if llvm_dir is None:
            parser.error('LLVM_DIR is missing from the selected build directory')
        clang = Path(llvm_dir).parents[2] / 'bin' / ('clang' + suffix)
    compiler = args.build_dir / 'tools/vrccl' / ('udonclc' + suffix)
    if not clang.is_file() or not compiler.is_file():
        parser.error('build the project first, or specify --build-dir / --clang')
    args.output.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(command):
        result = subprocess.run([str(x) for x in command], text=True, capture_output=True)
        commands.append({'argv': [str(x) for x in command], 'returncode': result.returncode,
                         'stdout': result.stdout, 'stderr': result.stderr})
        (args.output / 'commands.json').write_text(json.dumps(commands, indent=2), encoding='utf-8')
        if result.returncode:
            sys.stderr.write(result.stdout + result.stderr)
            raise SystemExit(result.returncode)
        return result.stdout

    version = json.loads((root / 'cmake/llvm-lock.json').read_text())['version']
    if 'clang version ' + version not in run([clang, '--version']):
        parser.error('clang must match the pinned LLVM version ' + version)
    common = ['-target', 'spir-unknown-unknown', '-O0', '-Xclang', '-disable-O0-optnone',
              '-fomit-frame-pointer', '-ffp-contract=off', '-S', '-emit-llvm']
    host = args.output / 'host.ll'
    device = args.output / 'device.ll'
    print('Compiling host.c and device.cl with pinned Clang', flush=True)
    # SPIR supplies a portable 32-bit test layout. Explicit C calling convention
    # keeps scalar Host calls separate from SPIR device calls; final TargetInfo is pending.
    run([clang, *common, '-x', 'c', '-std=c11', '-Xclang', '-fdefault-calling-conv=cdecl',
         args.source_dir / 'host.c', '-o', host])
    run([clang, *common, '-x', 'cl', '-cl-std=CL1.2', '-cl-kernel-arg-info',
         args.source_dir / 'device.cl', '-o', device])
    bundle = args.output / 'bundle'
    print(run([compiler, '--emit-llvm-boundary-bundle', args.source_dir / 'boundary.json',
               '--host-llvm', host, '--device-llvm', device, '-o', bundle]), end='')
    print(run([compiler, '--check-boundary-request', bundle / 'manifest.json',
               '--request', args.source_dir / 'request.json']), end='')
    print('Inspect:', bundle / 'host-device.mlir', 'and', bundle / 'kernels/kernel_7.shader')


if __name__ == '__main__':
    main()
