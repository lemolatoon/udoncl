"""Build the public UdonCL CMake package as an external multi-file project."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run(command):
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise AssertionError((command, result.stdout, result.stderr))
    return result.stdout


def run_failure(command, diagnostic):
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0, (command, result.stdout, result.stderr)
    assert diagnostic in result.stderr, (command, diagnostic, result.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--project-build', required=True, type=Path)
    parser.add_argument('--cmake', required=True)
    parser.add_argument('--generator', required=True)
    parser.add_argument('--make-program', required=True)
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--clang', required=True)
    parser.add_argument('--llvm-link', required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    install = output / 'install'
    consumer = output / 'consumer'
    run([
        args.cmake,
        '--install', str(args.project_build.resolve()),
        '--prefix', str(install),
        '--component', 'UdonCLCompiler',
    ])
    suffix = '.exe' if Path(args.compiler).suffix == '.exe' else ''
    installed_compiler = install / 'bin' / f'udonclc{suffix}'
    assert installed_compiler.is_file(), installed_compiler
    package = install / 'lib/cmake/UdonCL'
    run([
        args.cmake,
        '-S', str(args.source.resolve()),
        '-B', str(consumer),
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
    ])
    run([args.cmake, '--build', str(consumer), '--target', 'hello-artifact'])
    run([args.cmake, '--build', str(consumer), '--target', 'hello-unity-library'])
    run([
        args.cmake, '--build', str(consumer),
        '--target', 'hello-buffer-copy-unity-library',
    ])

    build_root = consumer / 'udoncl/hello-artifact'
    artifact = build_root / 'artifact'
    manifest = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['artifact_stage'] == 'boundary-prototype', manifest
    assert manifest['exports'][0]['c_symbol'] == 'host_square_plus_one', manifest
    assert len(list((build_root / 'objects/host').rglob('*.bc'))) == 3
    assert len(list((build_root / 'objects/device').rglob('*.bc'))) == 3
    assert (build_root / 'host.bc').is_file()
    assert (build_root / 'device.bc').is_file()
    assert (artifact / 'host-device.mlir').is_file()
    assert (artifact / 'kernels/kernel_1.shader').is_file()
    unity_artifact = consumer / 'udoncl/hello-unity-library/artifact'
    unity_manifest = json.loads(
        (unity_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert unity_manifest['artifact_stage'] == 'unity-udoncl-library'
    assert unity_manifest['ready_for_unity'] is True
    assert unity_manifest['program']['assembly'] == 'host.uasm'
    assert unity_manifest['program']['display_shader'] == 'display.shader'
    assert unity_manifest['exports'][0]['public_name'] == 'SquarePlusOne'
    assert unity_manifest['device']['buffers'] == [{
        'id': 1,
        'name': 'scratch',
        'heap_slot': '__udoncl_buffer_1',
        'logical_bytes': 4,
        'width': 1,
        'height': 1,
        'format': 'rgba8_unorm',
        'srgb': False,
        'mipmaps': False,
        'filter': 'point',
    }]
    assembly = (unity_artifact / 'host.uasm').read_text(encoding='utf-8')
    assert '.export __0_SquarePlusOne' in assembly
    assert '.export __udoncl_buffer_1' in assembly
    assert '__udoncl_buffer_1: %UnityEngineRenderTexture, null' in assembly
    assert (unity_artifact / 'display.shader').is_file()
    copy_artifact = consumer / 'udoncl/hello-buffer-copy-unity-library/artifact'
    copy_manifest = json.loads(
        (copy_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert copy_manifest['program']['buffer_copy_shader'] == 'buffer-copy.shader'
    assert [buffer['name'] for buffer in copy_manifest['device']['buffers']] == [
        'source', 'destination',
    ]
    assert [
        (buffer['logical_bytes'], buffer['width'], buffer['height'])
        for buffer in copy_manifest['device']['buffers']
    ] == [(32772, 8192, 2), (32772, 8192, 2)]
    copy_assembly = (copy_artifact / 'host.uasm').read_text(encoding='utf-8')
    assert '.export CopyFull' in copy_assembly
    assert copy_assembly.count('VRCSDKBaseVRCGraphics.__Blit__') == 2
    assert (copy_artifact / 'buffer-copy.shader').is_file()
    assert (install / 'include/udoncl/unity_buffer.h').is_file()

    invalid_source = output / 'invalid-buffer-handle.c'
    invalid_module = output / 'invalid-buffer-handle.bc'
    invalid_source.write_text(
        '#include <udoncl/unity_buffer.h>\n'
        'udoncl_buffer copy_full(void) { return udonclBufferAlloc(4); }\n',
        encoding='utf-8',
    )
    run([
        args.clang, '-target', 'spir-unknown-unknown', '-x', 'c', '-std=c11',
        '-Xclang', '-fdefault-calling-conv=cdecl', '-O0',
        '-Xclang', '-disable-O0-optnone', '-fomit-frame-pointer',
        '-ffp-contract=off', '-emit-llvm', '-c', str(invalid_source),
        '-I' + str(install / 'include'), '-o', str(invalid_module),
    ])
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        args.source.resolve() / 'buffer-copy-library.json',
        '--host-llvm', invalid_module, '-o', output / 'invalid-buffer-artifact',
    ], 'must flow directly to one or more CopyBuffer calls')
    print(
        'cmake_package=passed host_sources=3 device_sources=3 '
        'artifacts=hello-artifact,hello-unity-library,hello-buffer-copy-unity-library'
    )


if __name__ == '__main__':
    main()
