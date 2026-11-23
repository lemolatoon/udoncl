"""Build and validate the assigned-buffer RMS component in isolation."""

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
    run([
        args.cmake, '--install', project_build, '--prefix', install,
        '--component', 'UdonCLCompiler',
    ])
    package_root = install / 'lib/cmake'
    run([
        args.cmake,
        '-S', root / 'examples/onnx-rms-component',
        '-B', consumer,
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package_root / "UdonCL"}',
        f'-DUdonBLAS_DIR={package_root / "UdonBLAS"}',
        f'-DUdonDNN_DIR={package_root / "UdonDNN"}',
        f'-DUdonATen_DIR={package_root / "UdonATen"}',
        f'-DUdonONNXRuntime_DIR={package_root / "UdonONNXRuntime"}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
    ])
    run([args.cmake, '--build', consumer, '--target', 'onnx-rms-component'])

    artifact = consumer / 'udoncl/onnx-rms-component/artifact'
    recovered = (artifact / 'frontend.recovered.mlir').read_text(encoding='utf-8')
    manifest = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
    # The Host C names the program (UDONCL_PROGRAM) and CMake passes
    # ONNX_PUBLIC_TENSOR_BUFFERS; the binding class is the default one.
    assert manifest['program']['name'] == 'onnx_rms_component'
    assert manifest['program']['binding_class'] == 'UdonclOnnxRmsComponentBinding'
    assert 'vrc.assigned_buffer' in recovered
    assert 'vrc.buffer_alloc' not in recovered
    assert 'vrc.buffer_release' not in recovered
    buffers = manifest['device']['buffers']
    assert sum(item.get('public_access') == 'input' for item in buffers) == 1
    assert sum(item.get('public_access') == 'output' for item in buffers) == 1
    assert all(item.get('heap_slot') for item in buffers)
    assert [item['kernel_id'] for item in manifest['dispatch_sites']] == [217, 218]
    kernels = {item['id']: item for item in manifest['device']['kernels']}
    assert [item['required_bytes'] for item in kernels[217]['args'][:2]] == [
        65536, 65536,
    ]
    assert [item['required_bytes'] for item in kernels[218]['args'][:4]] == [
        65536, 65536, 16384, 65536,
    ]
    public = {item.get('public_access'): item for item in buffers if item.get('public_access')}
    assert public['input']['logical_bytes'] == 32768
    assert public['output']['logical_bytes'] == 32768
    # A public tensor buffer states the ONNX tensor it holds.
    for item in public.values():
        assert item['source'] == {
            'kind': 'onnx_tensor', 'tensor': item['name']}, item
    static_tensors = manifest['static_tensors']
    assert [item['name'] for item in static_tensors] == ['scale', 'epsilon']
    assert static_tensors[0]['shape'] == [2048]
    assert static_tensors[0]['byte_length'] == 8192
    assert static_tensors[1]['shape'] == []
    assert static_tensors[1]['byte_length'] == 4
    print(
        'onnx_rms_component=passed '
        f'assigned_buffers={len(buffers)} dispatch_sites=2 width=2048'
    )


if __name__ == '__main__':
    main()
