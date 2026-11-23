"""Build a Unity library whose public Device buffers are UDONCL_PUBLIC Host
variables, and compile variants of its Host that must be rejected."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

# The Host flags of udoncl_add_host_library (UdonCLConfig.cmake).
HOST_FLAGS = [
    '-target', 'spir-unknown-unknown', '-x', 'c', '-std=c11',
    '-Xclang', '-fdefault-calling-conv=cdecl', '-O1', '-fno-inline',
    '-fno-discard-value-names', '-fomit-frame-pointer', '-ffp-contract=off',
    '-emit-llvm', '-c',
]


def run(command):
    subprocess.run([str(item) for item in command], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--project-build', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cmake', required=True)
    parser.add_argument('--generator', required=True)
    parser.add_argument('--make-program', required=True)
    parser.add_argument('--clang', required=True)
    parser.add_argument('--llvm-link', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    install = output / 'install'
    consumer = output / 'consumer'
    run([args.cmake, '--install', args.project_build.resolve(), '--prefix',
         install, '--component', 'UdonCLCompiler'])
    source = root / 'tests/build/public_buffers'
    run([
        args.cmake, '-S', source, '-B', consumer, '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={install / "lib/cmake/UdonCL"}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
    ])
    run([args.cmake, '--build', consumer, '--target', 'public_buffers'])

    suffix = '.exe' if sys.platform == 'win32' else ''
    compiler = install / 'bin' / f'udonclc{suffix}'
    target = consumer / 'udoncl/public_buffers'
    artifact = target / 'artifact'
    manifest = json.loads(
        (artifact / 'manifest.json').read_text(encoding='utf-8'))

    # The configuration lists no buffer: the three variables are the public
    # buffers, named after them, with their annotated element type and the
    # direction of their clCreateBuffer flags. Every Device buffer resolves at
    # compile time to its own Unity Texture.
    buffers = manifest['device']['buffers']
    public = {
        buffer['name']: buffer for buffer in buffers
        if 'public_access' in buffer
    }
    assert set(public) == {'x', 'y', 'result'}, buffers
    assert [public[name]['public_access'] for name in ['x', 'y', 'result']] \
        == ['input', 'input', 'output'], public
    for name, buffer in public.items():
        assert buffer['element_type'] == 'float32', buffer
        assert buffer['capacity_bytes'] == 128, buffer
        assert buffer['logical_bytes'] == 128, buffer
    assert len(buffers) == 3, buffers
    recovered = (artifact / 'frontend.recovered.mlir').read_text(
        encoding='utf-8')
    readiness = next(
        line for line in recovered.splitlines()
        if 'vrc.static_unity_resource_lowering_readiness' in line)
    assert 'applied = true' in readiness, readiness
    for symbol in ['x', 'y', 'result']:
        line = next(line for line in recovered.splitlines()
                    if f'sym_name = "{symbol}"' in line
                    and 'udonhl.global' in line)
        assert 'udoncl.public = "float32"' in line, line
    binding = (artifact / 'binding.cs').read_text(encoding='utf-8') \
        if (artifact / 'binding.cs').is_file() else ''

    def compile_variant(name, host_source, *extra):
        """Compiles a variant of host.c against the built runtime."""
        host = output / f'{name}.c'
        host.write_text(host_source, encoding='utf-8')
        module = output / f'{name}.bc'
        run([args.clang, *HOST_FLAGS, host, f'-I{install / "include"}',
             '-DCL_TARGET_OPENCL_VERSION=120', '-o', module])
        return subprocess.run([
            str(compiler), '--emit-unity-library-bundle',
            str(source / 'unity-library.json'),
            '--host-llvm', str(module),
            '--host-library-llvm',
            str(target / 'modules/host/capacity_runtime.bc'),
            '--device-llvm', str(target / 'device.bc'),
            '--no-runtime-validation', *extra,
            '-o', str(output / f'{name}-artifact'),
        ], text=True, capture_output=True)

    host = (source / 'host.c').read_text(encoding='utf-8')
    report = compile_variant('report', host, '--report-static-resources')
    assert report.returncode == 0, report.stderr
    assert ('udonclc-static-resources: static Unity resource lowering '
            'applied') in report.stderr, report.stderr
    assert 'blocker' not in report.stderr, report.stderr

    def rejected(name, host_source, diagnostic):
        result = compile_variant(name, host_source)
        assert result.returncode != 0, (name, result.stderr)
        assert diagnostic in result.stderr, (name, diagnostic, result.stderr)

    # A public buffer U# both writes and reads is not supported yet.
    rejected(
        'read-write',
        host.replace('result = create(CL_MEM_WRITE_ONLY',
                     'result = create(CL_MEM_READ_WRITE'),
        'UDONCL_PUBLIC variable result is a CL_MEM_READ_WRITE buffer')
    # A variable must hold the buffer of exactly one clCreateBuffer.
    rejected(
        'two-sites',
        host.replace('UDONCL_EXPORT int shutdown(void) {', '''\
UDONCL_EXPORT int recreate(void) {
  cl_int status;
  x = create(CL_MEM_READ_ONLY, &status);
  return status;
}

UDONCL_EXPORT int shutdown(void) {'''),
        'UDONCL_PUBLIC variable x holds the Device buffers of 2 allocation '
        'contexts')
    rejected(
        'no-buffer',
        host.replace('UDONCL_PUBLIC(float) cl_mem result;',
                     'UDONCL_PUBLIC(float) cl_mem result;\n'
                     'UDONCL_PUBLIC(int) cl_mem unused;'),
        'UDONCL_PUBLIC variable unused holds no Device buffer')
    print(f'public_buffers=passed buffers={len(buffers)}')


if __name__ == '__main__':
    main()
