"""Build a Unity library whose C Host allocates its Device buffers in
constant-trip loops and check that the Host buffer plan gives its OpenCL
runtime one buffer handle per allocation."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

LAYERS = 22
SCRATCH_BUFFERS = 4


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
    run([
        args.cmake, '-S', root / 'tests/build/host_buffer_plan', '-B',
        consumer, '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={install / "lib/cmake/UdonCL"}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
    ])
    run([args.cmake, '--build', consumer, '--target', 'host_buffer_plan'])

    suffix = '.exe' if sys.platform == 'win32' else ''
    compiler = install / 'bin' / f'udonclc{suffix}'
    target = consumer / 'udoncl/host_buffer_plan'
    manifest = json.loads(
        (target / 'artifact/manifest.json').read_text(encoding='utf-8'))
    buffers = manifest['device']['buffers']
    # result, two buffers per layer and the scratch buffers: every
    # allocation context is one Unity buffer.
    allocations = 1 + 2 * LAYERS + SCRATCH_BUFFERS
    assert len(buffers) == allocations, buffers
    # The OpenCL runtime of the library has one buffer handle per
    # allocation: the Host buffer plan CMake writes sizes its pools.
    assert manifest['device']['mutable_handle_capacity'] == allocations
    built_plan = json.loads(
        (target / 'host-buffer-plan.json').read_text(encoding='utf-8'))
    assert built_plan['total_capacity'] == allocations, built_plan
    assert (target / 'host-buffer-capacity.h').read_text(
        encoding='utf-8') == (
            f'#define UDONCL_AOT_BUFFER_CAPACITY {allocations}\n'
            f'#define UDONCL_DEVICE_BUFFER_CAPACITY {allocations}\n')

    # The plan of the runtime-free C Host, planned again: one handle per
    # allocation context, including the 2 * LAYERS buffers a helper called
    # in a loop allocates and the buffers of a clCreateBuffer directly in a
    # loop.
    plan_path = output / 'host-buffer-plan.json'
    header = output / 'host-buffer-capacity.h'
    run([compiler, '--plan-host-buffer-capacity', target / 'host.bc',
         '--capacity-header', header, '-o', plan_path])
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert plan['schema_version'] == 1, plan
    assert plan['base_capacity'] == 0, plan
    assert plan['total_capacity'] == allocations, plan
    assert sorted(plan['extra_buffer_bytes']) == sorted(
        [16] * (1 + SCRATCH_BUFFERS) + [256, 512] * LAYERS), plan
    assert header.read_text(encoding='utf-8') == (
        f'#define UDONCL_AOT_BUFFER_CAPACITY {allocations}\n'
        f'#define UDONCL_DEVICE_BUFFER_CAPACITY {allocations}\n')

    # The plan adds to the ONNX Host's handles.
    based = output / 'based-plan.json'
    run([compiler, '--plan-host-buffer-capacity', target / 'host.bc',
         '--base-mutable-handle-capacity', 3,
         '--capacity-header', output / 'based.h', '-o', based])
    based_plan = json.loads(based.read_text(encoding='utf-8'))
    assert based_plan['base_capacity'] == 3, based_plan
    assert based_plan['total_capacity'] == allocations + 3, based_plan
    assert based_plan['extra_buffer_bytes'] == plan['extra_buffer_bytes']
    # A plan that undercounts the allocation contexts, as the LLVM IR scan
    # of call sites did for the helper in the loop, is a compile error.
    undercount = output / 'undercount-plan.json'
    undercount.write_text(json.dumps({
        'schema_version': 1, 'base_capacity': 0,
        'extra_buffer_bytes': [16, 256, 512, 16],
        'total_capacity': 4,
    }), encoding='utf-8')
    rejected = subprocess.run([
        str(compiler), '--emit-unity-library-bundle',
        '--host-llvm', str(target / 'host.bc'),
        '--host-library-llvm',
        str(target / 'modules/host/capacity_runtime.bc'),
        '--host-buffer-plan', str(undercount), '--no-runtime-validation',
        '-o', str(output / 'undercount-artifact'),
    ], text=True, capture_output=True)
    assert rejected.returncode != 0, rejected.stderr
    assert (f'the OpenCL runtime has 4 buffer handles but the Host has '
            f'{allocations} allocation contexts, each of which needs one '
            f'(Host buffer plan {undercount})') in rejected.stderr, \
        rejected.stderr
    print(f'host_buffer_plan=valid handles={allocations}')


if __name__ == '__main__':
    main()
