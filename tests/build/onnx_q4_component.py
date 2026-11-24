"""Build and validate the assigned-buffer Q4 component in isolation."""

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
    run([
        sys.executable,
        root / 'scripts/build.py',
        '--build-dir',
        project_build,
    ])
    if output.exists():
        shutil.rmtree(output)
    install = output / 'install'
    consumer = output / 'consumer'
    run([
        args.cmake,
        '--install',
        project_build,
        '--prefix',
        install,
        '--component',
        'UdonCLCompiler',
    ])
    package_root = install / 'lib/cmake'
    run([
        args.cmake,
        '-S', root / 'examples/onnx-q4-component',
        '-B', consumer,
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package_root / "UdonCL"}',
        f'-DUdonBLAS_DIR={package_root / "UdonBLAS"}',
        f'-DUdonDNN_DIR={package_root / "UdonDNN"}',
        f'-DUdonATen_DIR={package_root / "UdonATen"}',
        f'-DUdonONNXOperators_DIR={package_root / "UdonONNXOperators"}',
        f'-DUdonONNXRuntime_DIR={package_root / "UdonONNXRuntime"}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
    ])
    run([
        args.cmake,
        '--build',
        consumer,
        '--target',
        'onnx-q4-component',
    ])

    artifact = consumer / 'udoncl/onnx-q4-component/artifact'
    recovered = (artifact / 'frontend.recovered.mlir').read_text(encoding='utf-8')
    udon_async = (artifact / 'host.udon-async.mlir').read_text(encoding='utf-8')
    manifest = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
    # The Host C names the program (UDONCL_PROGRAM) and binds its
    # UDONCL_PUBLIC input and output to the graph tensors of the same names;
    # the binding class is the default one.
    assert manifest['program']['name'] == 'onnx_q4_component'
    assert manifest['program']['binding_class'] == 'UdonclOnnxQ4ComponentBinding'
    assert 'vrc.assigned_buffer' in recovered
    assert 'vrc.buffer_alloc' not in recovered
    assert 'vrc.buffer_release' not in recovered
    assert 'udonhl.checked_assigned_buffer_resolve' not in recovered
    # Split-K: the partial pass writes the operator scratch (buffer 4) and
    # the ordered sum pass reads it and writes the output (buffer 2).
    assert udon_async.count('vrc.unity_resource_heap_slots = ') == 2
    assert udon_async.count(
        'vrc.unity_resource_heap_slots = ["__udoncl_buffer_1", "", "", '
        '"__udoncl_buffer_2", "__udoncl_buffer_4"]'
    ) == 1
    assert udon_async.count(
        'vrc.unity_resource_heap_slots = ["__udoncl_buffer_1", "", "", '
        '"__udoncl_buffer_4", "__udoncl_buffer_2"]'
    ) == 1
    assert udon_async.count(
        'vrc.unity_static_tensor_ids = [-1 : i32, 0 : i32, 1 : i32, '
        '-1 : i32, -1 : i32]'
    ) == 2
    assert [
        (item['name'], item['logical_bytes'], item['capacity_bytes'],
         item.get('element_type'), item.get('public_access'),
         item.get('heap_slot'))
        for item in manifest['device']['buffers']
    ] == [
        ('input', 32768, 32768, 'float32', 'input', '__udoncl_buffer_1'),
        ('output', 528, 528, 'float32', 'output', '__udoncl_buffer_2'),
        ('private_3', 524288, 524288, None, None, '__udoncl_buffer_3'),
        ('private_4', 524288, 524288, None, None, '__udoncl_buffer_4'),
    ]
    # K = 2048 (64 blocks) over 4 x 33 outputs: the split-K cost model
    # chooses a partial pass and an ordered sum pass of the same kernel.
    assert [item['kernel_id'] for item in manifest['dispatch_sites']] == [
        235, 235,
    ]
    # The kernel entry is derived from the dispatches: its inputs hold the
    # library contract of the kernel (M <= 4, N <= 32000, K <= 5632), not
    # just this component's 33 x 2048 weight.
    kernel = next(item for item in manifest['device']['kernels'] if item['id'] == 235)
    assert [item['required_bytes'] for item in kernel['args'][:5]] == [
        4 * 5632 * 4, 32000 * 176 * 4 * 4, 32000 * 176 * 4, 524288, 524288,
    ]
    assert [item.get('maximum') for item in kernel['args'][5:]] == [
        4, 32000, 5632, 176, 64, 64,
    ]
    assert kernel['global_size'] == [8192, 16, 1]
    print('onnx_q4_component=passed assigned_buffers=4 dispatch_sites=2 k=2048')


if __name__ == '__main__':
    main()
