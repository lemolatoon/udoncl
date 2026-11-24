"""Compile one small ONNX component with and without static initialization."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


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
    parser.add_argument('--onnx-frontend', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    project_build = args.project_build.resolve()
    output = args.output.resolve()
    run([sys.executable, root / 'scripts/build.py', '--build-dir', project_build])
    if output.exists():
        shutil.rmtree(output)
    install = output / 'install'
    consumer = output / 'consumer'
    run([args.cmake, '--install', project_build, '--prefix', install,
         '--component', 'UdonCLCompiler'])
    package_root = install / 'lib/cmake'
    # tests/build/static_initialization builds the transpose component three
    # ways from one Host C source (see its host.c).
    source = root / 'tests/build/static_initialization'
    run([
        args.cmake, '-S', source, '-B', consumer, '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package_root / "UdonCL"}',
        f'-DUdonONNXOperators_DIR={package_root / "UdonONNXOperators"}',
        f'-DUdonONNXRuntime_DIR={package_root / "UdonONNXRuntime"}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
        '-DUDONCL_TEST_EXAMPLE_DIRECTORY='
        f'{root / "examples/onnx-transpose-component"}',
    ])
    run([args.cmake, '--build', consumer, '--target', 'dynamic', 'static',
         'hidden'])

    def manifest_of(artifact):
        return json.loads(
            (artifact / 'manifest.json').read_text(encoding='utf-8'))

    def exports_of(manifest):
        return {item['public_name']: item for item in manifest['exports']}

    def defines(module, symbol):
        return re.search(
            rf'func\.func (private )?@{re.escape(symbol)}\(', module)

    def instructions(text):
        code = text[text.index('.code_start'):]
        return sum(
            line.strip().split(',')[0] in (
                'PUSH', 'POP', 'COPY', 'EXTERN', 'JUMP', 'JUMP_IF_FALSE',
                'JUMP_INDIRECT', 'NOP', 'ANNOTATION')
            for line in code.splitlines())

    # The dynamic artifact is the default: `initialize` is a plain export.
    dynamic = consumer / 'udoncl/dynamic/artifact'
    dynamic_manifest = manifest_of(dynamic)
    assert 'static_initialization' not in dynamic_manifest
    dynamic_exports = exports_of(dynamic_manifest)
    assert set(dynamic_exports) == {'Initialize', 'Run', 'Shutdown'}
    dynamic_assembly = (dynamic / 'host.uasm').read_text(encoding='utf-8')
    dynamic_heap = (dynamic / 'raw-heap.bytes').read_bytes()
    recovered = (dynamic / 'frontend.recovered.mlir').read_text(
        encoding='utf-8')
    assert not (dynamic / 'host.static-initialized.mlir').exists()

    def check_static(static, expected_exports):
        manifest = manifest_of(static)
        initialization = manifest['static_initialization']
        assert initialization['constructor'] == 'initialize'
        # The constructor ran in the compiler. This component primes no
        # Device buffer, so nothing of it is left but its status.
        assert initialization['evaluated_operations'] > 100
        assert initialization['residual_commands'] == 0
        exports = exports_of(manifest)
        assert set(exports) == expected_exports, exports
        # Every remaining export keeps its dynamic public ABI.
        for name, entry in exports.items():
            assert entry == dynamic_exports[name], name
        # The session, its binding table and every Device buffer handle are
        # initializer data: no code that creates them is left, and the
        # initial heap image holds what initialize used to store.
        evaluated = (static / 'host.static-initialized.mlir').read_text(
            encoding='utf-8')
        assert 'relocations = [' in evaluated
        for symbol in (
            'udononnxCreateSession', 'udononnxBindInputAt',
            'udononnxBindOutputAt', 'clCreateContext', 'clCreateCommandQueue',
        ):
            assert not defines(evaluated, symbol), symbol
            assert defines(recovered, symbol), symbol
        assembly = (static / 'host.uasm').read_text(encoding='utf-8')
        assert instructions(assembly) < instructions(dynamic_assembly)
        heap = (static / 'raw-heap.bytes').read_bytes()
        assert sum(map(bool, heap)) > sum(map(bool, dynamic_heap))
        return manifest, evaluated, assembly

    # An __attribute__((constructor)) export runs as static initialization
    # and stays exported, returning the status it recorded.
    static = consumer / 'udoncl/static/artifact'
    # The __attribute__((destructor)) export is omitted: the artifact is
    # never unloaded.
    manifest, evaluated, assembly = check_static(static, {'Initialize', 'Run'})
    assert 'vrc.evaluated_static_constructor' in evaluated
    assert manifest['static_initialization']['omitted_exports'] == [
        'shutdown']
    assert '.export Shutdown' not in assembly
    assert '.export Shutdown' in dynamic_assembly
    assert instructions(assembly) * 2 < instructions(dynamic_assembly)
    assert exports_of(manifest)['Initialize']['c_symbol'] == 'initialize'

    # A constructor that is not an export runs the same way and leaves the
    # U# API once it has run.
    hidden = consumer / 'udoncl/hidden/artifact'
    _, evaluated, assembly = check_static(hidden, {'Run'})
    assert not re.search(r'func\.func @initialize\(', evaluated)
    remaining = re.search(r'func\.func private @initialize\(.*', evaluated)
    assert not remaining or 'vrc.export_id' not in remaining.group(0)
    # initialize enqueues no Device work, so the export is synchronous.
    assert 'public int TryInitialize(' in (static / 'binding.cs').read_text(
        encoding='utf-8')
    assert 'Initialize(' not in (hidden / 'binding.cs').read_text(
        encoding='utf-8')
    assert '.export Initialize' not in assembly

    # The static_initialization config key that named a constructor and
    # omitted exports is removed with the config file; a constructor that
    # fails at build time is rejected by vrc-evaluate-host-constructors
    # (tests/lit/VRCToUdonHL/evaluate-host-constructors-invalid.mlir).
    print('static_initialization=passed variants=3')


if __name__ == '__main__':
    main()
