"""Compile one small ONNX component with and without static initialization."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

CONSTRUCTOR = 'transpose_component_initialize'
DESTRUCTOR = 'transpose_component_dispose'


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
    source = root / 'examples/onnx-transpose-component'
    run([
        args.cmake, '-S', source, '-B', consumer, '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package_root / "UdonCL"}',
        f'-DUdonONNXOperators_DIR={package_root / "UdonONNXOperators"}',
        f'-DUdonONNXRuntime_DIR={package_root / "UdonONNXRuntime"}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
    ])
    run([args.cmake, '--build', consumer, '--target', 'onnx-transpose-component'])

    # The dynamic artifact is the default: the same inputs compile again with
    # only the Unity library configuration changed.
    suffix = '.exe' if sys.platform == 'win32' else ''
    compiler = install / 'bin' / f'udonclc{suffix}'
    target = consumer / 'udoncl/onnx-transpose-component'
    model = consumer / 'udoncl-onnx/transpose_component_model_host'
    dynamic_config = json.loads(
        (source / 'unity-library.json').read_text(encoding='utf-8'))

    def compile_artifact(name, initialization):
        config = dict(dynamic_config)
        config['static_initialization'] = initialization
        config_path = output / f'{name}.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n',
                               encoding='utf-8')
        artifact = output / name
        result = subprocess.run([
            str(compiler), '--emit-unity-library-bundle', str(config_path),
            '--host-mlir', str(model / 'host.mlir'),
            '--host-llvm', str(target / 'host.bc'),
            '--host-library-llvm',
            str(target / 'modules/host/capacity_runtime.bc'),
            '--device-llvm', str(target / 'device.bc'),
            '--static-tensor-directory', str(model / 'static-tensors'),
            '--host-buffer-plan', str(target / 'host-buffer-plan.json'),
            '--no-runtime-validation', '-o', str(artifact),
        ], text=True, capture_output=True)
        return artifact, result

    def rejected(name, initialization, diagnostic):
        _, result = compile_artifact(name, initialization)
        assert result.returncode != 0, (name, result.stdout, result.stderr)
        assert diagnostic in result.stderr, (name, diagnostic, result.stderr)

    dynamic = target / 'artifact'
    dynamic_manifest = json.loads(
        (dynamic / 'manifest.json').read_text(encoding='utf-8'))
    assert 'static_initialization' not in dynamic_manifest
    dynamic_exports = {
        item['public_name']: item for item in dynamic_manifest['exports']
    }
    assert {'Initialize', 'Run', 'Shutdown'} <= set(dynamic_exports)

    static, result = compile_artifact('static', {
        'constructor': CONSTRUCTOR,
        'omit_exports': [DESTRUCTOR],
    })
    assert result.returncode == 0, (result.stdout, result.stderr)
    manifest = json.loads(
        (static / 'manifest.json').read_text(encoding='utf-8'))
    initialization = manifest['static_initialization']
    assert initialization['constructor'] == CONSTRUCTOR
    assert initialization['omitted_exports'] == [DESTRUCTOR]
    # The constructor ran in the compiler. This component primes no Device
    # buffer, so nothing of it is left but its status.
    assert initialization['evaluated_operations'] > 100
    assert initialization['residual_commands'] == 0
    exports = {item['public_name']: item for item in manifest['exports']}
    assert set(exports) == set(dynamic_exports) - {'Shutdown'}
    assert exports['Initialize']['c_symbol'] == CONSTRUCTOR
    # An omitted export has no Udon event and no generated wrapper; every
    # remaining export keeps its dynamic public ABI.
    for name, entry in exports.items():
        assert entry == dynamic_exports[name], name
    binding = (static / 'binding.cs').read_text(encoding='utf-8')
    assert 'Shutdown' not in binding
    assert 'Shutdown' in (dynamic / 'binding.cs').read_text(encoding='utf-8')
    assembly = (static / 'host.uasm').read_text(encoding='utf-8')
    assert '.export Shutdown' not in assembly
    dynamic_assembly = (dynamic / 'host.uasm').read_text(encoding='utf-8')
    assert '.export Shutdown' in dynamic_assembly

    # The session, its binding table and every Device buffer handle are
    # initializer data: no code that creates or releases them is left, and
    # the initial heap image holds what Initialize used to store.
    evaluated = (static / 'host.static-initialized.mlir').read_text(
        encoding='utf-8')
    assert 'vrc.evaluated_static_constructor' in evaluated
    assert 'relocations = [' in evaluated
    recovered = (dynamic / 'frontend.recovered.mlir').read_text(
        encoding='utf-8')

    def defines(module, symbol):
        return re.search(
            rf'func\.func (private )?@{re.escape(symbol)}\(', module)

    for symbol in (
        'udon_onnx_binding_create', 'udon_onnx_binding_bind_all',
        'udon_onnx_binding_dispose', 'udononnxCreateSession',
        'udononnxSessionDispose', 'clCreateContext', 'clCreateCommandQueue',
        'clReleaseMemObject', 'clReleaseContext', DESTRUCTOR,
    ):
        assert not defines(evaluated, symbol), symbol
        assert defines(recovered, symbol), symbol
    assert not (dynamic / 'host.static-initialized.mlir').exists()

    def instructions(text):
        code = text[text.index('.code_start'):]
        return sum(
            line.strip().split(',')[0] in (
                'PUSH', 'POP', 'COPY', 'EXTERN', 'JUMP', 'JUMP_IF_FALSE',
                'JUMP_INDIRECT', 'NOP', 'ANNOTATION')
            for line in code.splitlines())

    assert instructions(assembly) * 2 < instructions(dynamic_assembly)
    static_heap = (static / 'raw-heap.bytes').read_bytes()
    dynamic_heap = (dynamic / 'raw-heap.bytes').read_bytes()
    assert sum(map(bool, static_heap)) > sum(map(bool, dynamic_heap))

    # A constructor that fails is a build error: Run before Initialize
    # reports NOT_INITIALIZED.
    rejected('failing-constructor',
             {'constructor': 'transpose_component_run'},
             'static constructor fails at build time with status')

    rejected('missing-constructor', {'omit_exports': [DESTRUCTOR]},
             'static_initialization requires a constructor')
    rejected('unknown-constructor', {'constructor': 'not_an_export'},
             'static_initialization requires a constructor')
    rejected('internal-constructor',
             {'constructor': 'udon_onnx_binding_create'},
             'static_initialization requires a constructor')
    rejected('omitted-constructor',
             {'constructor': CONSTRUCTOR, 'omit_exports': [CONSTRUCTOR]},
             'static_initialization omit_exports must name distinct')
    rejected('duplicate-omission',
             {'constructor': CONSTRUCTOR,
              'omit_exports': [DESTRUCTOR, DESTRUCTOR]},
             'static_initialization omit_exports must name distinct')
    print(f'static_initialization=passed exports={len(exports)} omitted=1')


if __name__ == '__main__':
    main()
