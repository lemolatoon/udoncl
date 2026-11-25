"""Build and validate the assigned-buffer Transpose component in isolation."""

import argparse
import json
from pathlib import Path
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
    run([
        args.cmake, '-S', root / 'examples/onnx-transpose-component',
        '-B', consumer, '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package_root / "UdonCL"}',
        f'-DUdonONNXOperators_DIR={package_root / "UdonONNXOperators"}',
        f'-DUdonONNXRuntime_DIR={package_root / "UdonONNXRuntime"}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
    ])
    run([args.cmake, '--build', consumer, '--target', 'onnx-transpose-component'])
    artifact = consumer / 'udoncl/onnx-transpose-component/artifact'
    recovered = (artifact / 'frontend.recovered.mlir').read_text(encoding='utf-8')
    manifest = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
    # The Host C names the program (UDONCL_PROGRAM) and binds its
    # UDONCL_PUBLIC input and output to the graph tensors of the same names;
    # the binding class is the default one.
    assert manifest['program']['name'] == 'onnx_transpose_component'
    assert manifest['program']['binding_class'] == 'UdonclOnnxTransposeComponentBinding'
    assert 'vrc.assigned_buffer' in recovered
    assert 'vrc.buffer_alloc' not in recovered
    assert 'vrc.buffer_release' not in recovered
    buffers = manifest['device']['buffers']
    assert len(buffers) == 2
    assert all(item.get('heap_slot') for item in buffers)
    public = {item.get('public_access'): item for item in buffers}
    assert public['input']['logical_bytes'] == 32768
    assert public['output']['logical_bytes'] == 32768
    assert public['input']['capacity_bytes'] == 32768
    assert public['output']['capacity_bytes'] == 32768
    assert [item['kernel_id'] for item in manifest['dispatch_sites']] == [238]
    assert manifest['static_tensors'] == []
    kernel = next(item for item in manifest['device']['kernels'] if item['id'] == 238)
    assert [item['access'] for item in kernel['args'][:2]] == ['read', 'write']
    assert [item['required_bytes'] for item in kernel['args'][:2]] == [524288, 524288]
    assert kernel['global_size'] == [8192, 16, 1]
    # The functions of the generated ONNX binding are not U# exports; only
    # the component's own C exports are.
    assert sorted(item['public_name'] for item in manifest['exports']) == [
        'Dispose', 'Initialize', 'Run']
    print('onnx_transpose_component=passed assigned_buffers=2 words=8192')


if __name__ == '__main__':
    main()
