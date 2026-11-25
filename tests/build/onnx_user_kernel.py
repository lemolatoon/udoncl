"""Build a Unity library that links an ONNX Host and an OpenCL C kernel of
its own Host C: the MNIST model run through the UdonONNX session API, its
tensors in a cl_mem array created, bound and released in loops, and a user
ArgMax kernel created in initialize and launched by every run. This is the
model-free shape of the TinyLlama chat example."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


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
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    install = output / 'install'
    consumer = output / 'consumer'
    run([args.cmake, '--install', args.project_build.resolve(), '--prefix',
         install, '--component', 'UdonCLCompiler'])
    run([
        args.cmake, '-S', root / 'tests/build/onnx_user_kernel', '-B',
        consumer, '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DCMAKE_PREFIX_PATH={install}',
        f'-DUdonCL_DIR={install / "lib/cmake/UdonCL"}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
        f'-DUDONCL_MNIST_ONNX_GENERATOR={root / "examples/mnist-cnn/generate_onnx.py"}',
    ])
    run([args.cmake, '--build', consumer, '--target', 'onnx-user-kernel'])

    target = consumer / 'udoncl/onnx-user-kernel'
    manifest = json.loads(
        (target / 'artifact/manifest.json').read_text(encoding='utf-8'))
    assert manifest['ready_for_unity'] is True

    # The user kernel joins the library kernels, with an id above theirs;
    # it reads every word of the logits buffer its launch passes.
    kernels = {kernel['name']: kernel
               for kernel in manifest['device']['kernels']}
    argmax = kernels.get('mnist_argmax')
    assert argmax is not None, sorted(kernels)
    assert argmax['id'] >= 1024, argmax
    assert argmax['global_size'] == [1, 1, 1], argmax
    assert [arg['required_bytes'] for arg in argmax['args']
            if arg['kind'] == 'buffer'] == [40, 4], argmax['args']
    assert len(kernels) > 1, sorted(kernels)

    # Every Device buffer is private: the tensors of the array, the digit
    # and the buffers of the session.
    buffers = manifest['device']['buffers']
    assert not [buffer for buffer in buffers if 'public_access' in buffer], \
        buffers

    # The handle plan: the session's buffers, then one per allocation of
    # the C Host, the two array elements created in a loop and the digit.
    plan = json.loads(
        (target / 'host-buffer-plan.json').read_text(encoding='utf-8'))
    assert sorted(plan['extra_buffer_bytes']) == [4, 40, 3136], plan
    assert plan['total_capacity'] == plan['base_capacity'] + 3, plan
    assert manifest['device']['mutable_handle_capacity'] == \
        plan['total_capacity'], manifest['device']
    assert len(buffers) == plan['total_capacity'], buffers

    # Initialize and Dispose enqueue no Device work; Run does.
    exports = {export['public_name']: export for export in manifest['exports']}
    assert set(exports) == {'Initialize', 'Run', 'Dispose'}, sorted(exports)
    assert exports['Run']['async'] is True, exports['Run']
    assert not exports['Initialize'].get('async'), exports['Initialize']
    assert not exports['Dispose'].get('async'), exports['Dispose']
    print(f'onnx_user_kernel=valid kernel_id={argmax["id"]} '
          f'handles={plan["total_capacity"]}')


if __name__ == '__main__':
    main()
