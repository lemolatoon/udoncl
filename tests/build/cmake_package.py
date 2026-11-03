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
    saxpy_consumer = output / 'saxpy-consumer'
    pooling_consumer = output / 'average-pooling-consumer'
    run([
        args.cmake,
        '--install', str(args.project_build.resolve()),
        '--prefix', str(install),
        '--component', 'UdonCLCompiler',
    ])
    suffix = '.exe' if Path(args.compiler).suffix == '.exe' else ''
    installed_compiler = install / 'bin' / f'udonclc{suffix}'
    assert installed_compiler.is_file(), installed_compiler
    assert (install / 'include/CL/cl.h').is_file()
    assert (install / 'include/CL/cl_platform.h').is_file()
    assert (install / 'include/CL/cl_version.h').is_file()
    assert (install / 'share/licenses/UdonCL/OpenCL-Headers.LICENSE').is_file()
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
    run([
        args.cmake, '--build', str(consumer),
        '--target', 'hello-dispatch-unity-library',
    ])
    run([
        args.cmake, '--build', str(consumer),
        '--target', 'hello-int-dispatch-unity-library',
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
    assert [kernel['name'] for kernel in unity_manifest['device']['kernels']] == [
        'square_plus_one'
    ]
    assert unity_manifest['device']['kernels'][0]['shader'] == (
        'kernels/kernel_1.shader'
    )
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
    assert 'JUMP_INDIRECT' in assembly
    assert '__vrc_internal_1_return: %SystemUInt32, 0' in assembly
    assert (unity_artifact / 'display.shader').is_file()
    assert (unity_artifact / 'kernels/kernel_1.shader').is_file()
    assert (unity_artifact / 'kernels/kernel_1.hlsl').is_file()
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        args.source.resolve() / 'unity-library.json',
        '--host-llvm', consumer / 'udoncl/hello-unity-library/host.bc',
        '-o', output / 'missing-device-artifact',
    ], 'device_kernel and --device-llvm must be provided together')
    dispatch_artifact = output / 'typed-dispatch-artifact'
    run([
        installed_compiler, '--emit-unity-library-bundle',
        args.source.resolve() / 'dispatch-library.json',
        '--host-mlir', args.source.resolve().parents[1]
        / 'tests/boundary/unity_dispatch.mlir',
        '--device-llvm', consumer / 'udoncl/hello-unity-library/device.bc',
        '-o', dispatch_artifact,
    ])
    dispatch_manifest = json.loads(
        (dispatch_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert dispatch_manifest['dispatch_sites'] == [{
        'function': 'dispatch_square',
        'kernel_id': 1,
        'descriptor_abi': 1,
        'resources': 1,
        'completion': 'pending_event',
    }]
    dispatch_assembly = (dispatch_artifact / 'host.uasm').read_text(
        encoding='utf-8'
    )
    assert '.export __udoncl_kernel_material_1' in dispatch_assembly
    assert 'UnityEngineMaterial.__SetFloat__' in dispatch_assembly
    assert dispatch_assembly.count('VRCSDKBaseVRCGraphics.__Blit__') == 1
    opencl_dispatch_artifact = (
        consumer / 'udoncl/hello-dispatch-unity-library/artifact'
    )
    opencl_dispatch_manifest = json.loads(
        (opencl_dispatch_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert opencl_dispatch_manifest['dispatch_sites'] == [{
        'function': 'dispatch_square',
        'kernel_id': 1,
        'descriptor_abi': 1,
        'resources': 1,
        'completion': 'pending_event',
    }]
    assert opencl_dispatch_manifest['device']['buffers'][0][
        'element_type'
    ] == 'float32'
    assert opencl_dispatch_manifest['device']['buffers'][0][
        'public_access'
    ] == 'output'
    opencl_dispatch_assembly = (
        opencl_dispatch_artifact / 'host.uasm'
    ).read_text(encoding='utf-8')
    assert '.export __0_DispatchSquare' in opencl_dispatch_assembly
    assert '.export __udoncl_kernel_material_1' in opencl_dispatch_assembly
    assert opencl_dispatch_assembly.count('UnityEngineMaterial.__SetFloat__') == 1
    assert opencl_dispatch_assembly.count('VRCSDKBaseVRCGraphics.__Blit__') == 1
    assert (opencl_dispatch_artifact / 'kernels/kernel_1.shader').is_file()
    assert (opencl_dispatch_artifact / 'kernels/kernel_1.hlsl').is_file()
    int_dispatch_artifact = (
        consumer / 'udoncl/hello-int-dispatch-unity-library/artifact'
    )
    int_dispatch_manifest = json.loads(
        (int_dispatch_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert int_dispatch_manifest['device']['kernels'][0]['name'] == (
        'square_plus_one_i32'
    )
    assert int_dispatch_manifest['dispatch_sites'][0]['kernel_id'] == 2
    int_dispatch_assembly = (int_dispatch_artifact / 'host.uasm').read_text(
        encoding='utf-8'
    )
    assert '.export __0_DispatchSquareInt' in int_dispatch_assembly
    assert int_dispatch_assembly.count('UnityEngineMaterial.__SetInteger__') == 1
    assert int_dispatch_assembly.count('VRCSDKBaseVRCGraphics.__Blit__') == 1
    int_hlsl = (int_dispatch_artifact / 'kernels/kernel_2.hlsl').read_text(
        encoding='utf-8'
    )
    int_shader = (int_dispatch_artifact / 'kernels/kernel_2.shader').read_text(
        encoding='utf-8'
    )
    assert 'int _VRC_Arg1;' in int_hlsl
    assert 'int v0 = 1;' in int_hlsl
    assert '_VRC_Arg1 ("Arg1", Integer)' in int_shader
    copy_artifact = consumer / 'udoncl/hello-buffer-copy-unity-library/artifact'
    copy_manifest = json.loads(
        (copy_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert copy_manifest['program']['buffer_copy_shader'] == 'buffer-copy.shader'
    assert [buffer['name'] for buffer in copy_manifest['device']['buffers']] == [
        'source', 'intermediate', 'destination',
    ]
    assert [
        (buffer['logical_bytes'], buffer['width'], buffer['height'])
        for buffer in copy_manifest['device']['buffers']
    ] == [(32772, 8192, 2), (32772, 8192, 2), (32772, 8192, 2)]
    copy_assembly = (copy_artifact / 'host.uasm').read_text(encoding='utf-8')
    assert '.export CopyFull' in copy_assembly
    assert copy_assembly.count('VRCSDKBaseVRCGraphics.__Blit__') == 2
    first_copy = copy_assembly.index('PUSH, __udoncl_buffer_1')
    intermediate_write = copy_assembly.index('PUSH, __udoncl_buffer_2', first_copy)
    intermediate_read = copy_assembly.index(
        'PUSH, __udoncl_buffer_2', intermediate_write + 1
    )
    final_write = copy_assembly.index('PUSH, __udoncl_buffer_3', intermediate_read)
    assert first_copy < intermediate_write < intermediate_read < final_write
    assert (copy_artifact / 'buffer-copy.shader').is_file()
    assert (install / 'include/udoncl/unity_buffer.h').is_file()
    copy_recovered = (copy_artifact / 'frontend.recovered.mlir').read_text(
        encoding='utf-8'
    )
    assert copy_recovered.count('vrc.buffer_alloc ') == 3
    assert copy_recovered.count('vrc.buffer_copy ') == 2

    saxpy_source = args.source.resolve().parent / 'saxpy'
    run([
        args.cmake,
        '-S', str(saxpy_source),
        '-B', str(saxpy_consumer),
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
    ])
    run([
        args.cmake, '--build', str(saxpy_consumer),
        '--target', 'saxpy-unity-library',
    ])
    saxpy_artifact = (
        saxpy_consumer / 'udoncl/saxpy-unity-library/artifact'
    )
    saxpy_manifest = json.loads(
        (saxpy_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert saxpy_manifest['ready_for_unity'] is True
    assert saxpy_manifest['dispatch_sites'] == [{
        'function': 'run_saxpy',
        'kernel_id': 10,
        'descriptor_abi': 1,
        'resources': 2,
        'completion': 'pending_event',
    }]
    assert [
        (
            buffer['name'], buffer['logical_bytes'], buffer['element_type'],
            buffer['public_access'],
        )
        for buffer in saxpy_manifest['device']['buffers']
    ] == [
        ('input', 128, 'float32', 'input'),
        ('output', 128, 'float32', 'output'),
    ]
    assert saxpy_manifest['program']['buffer_copy_shader'] == 'buffer-copy.shader'
    assert (saxpy_artifact / 'buffer-copy.shader').is_file()
    saxpy_kernel = saxpy_manifest['device']['kernels'][0]
    assert saxpy_kernel['name'] == 'saxpy_demo'
    assert saxpy_kernel['profile'] == 'gather-f32'
    assert [arg['kind'] for arg in saxpy_kernel['args']] == [
        'buffer', 'buffer', 'i32', 'f32',
    ]
    assert [arg.get('access') for arg in saxpy_kernel['args'][:2]] == [
        'read', 'write',
    ]
    saxpy_assembly = (saxpy_artifact / 'host.uasm').read_text(
        encoding='utf-8'
    )
    assert '.export __0_RunSaxpy' in saxpy_assembly
    assert saxpy_assembly.count('UnityEngineMaterial.__SetTexture__') == 1
    assert saxpy_assembly.count('UnityEngineMaterial.__SetInteger__') == 1
    assert saxpy_assembly.count('UnityEngineMaterial.__SetFloat__') == 1
    assert saxpy_assembly.count('VRCSDKBaseVRCGraphics.__Blit__') == 1
    saxpy_hlsl = (saxpy_artifact / 'kernels/kernel_10.hlsl').read_text(
        encoding='utf-8'
    )
    assert 'Texture2D<float4> _VRC_Input' in saxpy_hlsl
    assert 'int _VRC_Arg2;' in saxpy_hlsl
    assert 'float _VRC_Arg3;' in saxpy_hlsl
    assert (saxpy_artifact / 'kernels/kernel_10.shader').is_file()

    pooling_source = args.source.resolve().parent / 'average-pooling'
    run([
        args.cmake,
        '-S', str(pooling_source),
        '-B', str(pooling_consumer),
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
    ])
    run([
        args.cmake, '--build', str(pooling_consumer),
        '--target', 'average-pooling-unity-library',
    ])
    pooling_artifact = (
        pooling_consumer / 'udoncl/average-pooling-unity-library/artifact'
    )
    pooling_manifest = json.loads(
        (pooling_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert pooling_manifest['ready_for_unity'] is True
    assert pooling_manifest['dispatch_sites'] == [{
        'function': 'run_average_pool',
        'kernel_id': 11,
        'descriptor_abi': 1,
        'resources': 2,
        'completion': 'pending_event',
    }]
    assert [
        (
            buffer['name'], buffer['logical_bytes'], buffer['element_type'],
            buffer['public_access'],
        )
        for buffer in pooling_manifest['device']['buffers']
    ] == [
        ('input', 32768, 'float32', 'input'),
        ('output', 32768, 'float32', 'output'),
    ]
    pooling_kernel = pooling_manifest['device']['kernels'][0]
    assert pooling_kernel['name'] == 'average_pool_8x8'
    assert pooling_kernel['profile'] == 'gather-f32'
    assert pooling_kernel['global_size'] == [8192, 1, 1]
    assert [arg['kind'] for arg in pooling_kernel['args']] == [
        'buffer', 'buffer', 'i32',
    ]
    pooling_assembly = (pooling_artifact / 'host.uasm').read_text(
        encoding='utf-8'
    )
    assert '.export __0_RunAveragePool' in pooling_assembly
    assert pooling_assembly.count('VRCSDKBaseVRCGraphics.__Blit__') == 1
    assert (pooling_artifact / 'kernels/kernel_11.shader').is_file()
    assert (pooling_artifact / 'buffer-copy.shader').is_file()

    invalid_saxpy_source = output / 'invalid-saxpy-access.c'
    invalid_saxpy_module = output / 'invalid-saxpy-access.bc'
    invalid_saxpy_source.write_text(
        (saxpy_source / 'host.c').read_text(encoding='utf-8').replace(
            'input = clCreateBuffer(context, CL_MEM_READ_ONLY,',
            'input = clCreateBuffer(context, CL_MEM_WRITE_ONLY,',
        ),
        encoding='utf-8',
    )
    run([
        args.clang, '-target', 'spir-unknown-unknown', '-x', 'c', '-std=c11',
        '-Xclang', '-fdefault-calling-conv=cdecl', '-O0',
        '-Xclang', '-disable-O0-optnone', '-fomit-frame-pointer',
        '-ffp-contract=off', '-emit-llvm', '-c', str(invalid_saxpy_source),
        '-I' + str(install / 'include'), '-o', str(invalid_saxpy_module),
    ])
    legacy_saxpy_config = output / 'legacy-saxpy-buffer-config.json'
    legacy_saxpy_document = json.loads(
        (saxpy_source / 'unity-library.json').read_text(encoding='utf-8')
    )
    for buffer in legacy_saxpy_document['buffers']:
        buffer.pop('element_type')
        buffer.pop('public_access')
    legacy_saxpy_config.write_text(
        json.dumps(legacy_saxpy_document), encoding='utf-8'
    )
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        legacy_saxpy_config,
        '--host-llvm', invalid_saxpy_module,
        '--device-llvm',
        saxpy_consumer / 'udoncl/saxpy-unity-library/device.bc',
        '-o', output / 'invalid-saxpy-access-artifact',
    ], 'requires read-only inputs and one write-only output')

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

    opencl_source = output / 'unsupported-opencl-read.c'
    opencl_module = output / 'unsupported-opencl-read.bc'
    opencl_source.write_text(
        '#define CL_TARGET_OPENCL_VERSION 120\n'
        '#include <CL/cl.h>\n'
        'void copy_full(void) {\n'
        '  (void)clEnqueueReadBuffer((cl_command_queue)0, (cl_mem)0, CL_TRUE, '
        '0, 4, 0, 0, 0, 0);\n'
        '}\n',
        encoding='utf-8',
    )
    run([
        args.clang, '-target', 'spir-unknown-unknown', '-x', 'c', '-std=c11',
        '-Xclang', '-fdefault-calling-conv=cdecl', '-O0',
        '-Xclang', '-disable-O0-optnone', '-fomit-frame-pointer',
        '-ffp-contract=off', '-emit-llvm', '-c', str(opencl_source),
        '-I' + str(install / 'include'), '-o', str(opencl_module),
    ])
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        args.source.resolve() / 'buffer-copy-library.json',
        '--host-llvm', opencl_module, '-o', output / 'unsupported-opencl-artifact',
    ], 'does not support Host API call clEnqueueReadBuffer')

    context_source = output / 'invalid-opencl-context.c'
    context_module = output / 'invalid-opencl-context.bc'
    context_source.write_text(
        '#define CL_TARGET_OPENCL_VERSION 120\n'
        '#include <CL/cl.h>\n'
        'void copy_full(void) {\n'
        '  (void)clCreateBuffer((cl_context)0, CL_MEM_READ_WRITE, 32772, 0, 0);\n'
        '}\n',
        encoding='utf-8',
    )
    run([
        args.clang, '-target', 'spir-unknown-unknown', '-x', 'c', '-std=c11',
        '-Xclang', '-fdefault-calling-conv=cdecl', '-O0',
        '-Xclang', '-disable-O0-optnone', '-fomit-frame-pointer',
        '-ffp-contract=off', '-emit-llvm', '-c', str(context_source),
        '-I' + str(install / 'include'), '-o', str(context_module),
    ])
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        args.source.resolve() / 'buffer-copy-library.json',
        '--host-llvm', context_module, '-o', output / 'invalid-context-artifact',
    ], 'clCreateBuffer must use the context returned by clCreateContext')

    status_source = output / 'unsupported-opencl-status-branch.c'
    status_module = output / 'unsupported-opencl-status-branch.bc'
    status_source.write_text(
        '#define CL_TARGET_OPENCL_VERSION 120\n'
        '#include <CL/cl.h>\n'
        'void copy_full(void) {\n'
        '  cl_platform_id platform;\n'
        '  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS) return;\n'
        '}\n',
        encoding='utf-8',
    )
    run([
        args.clang, '-target', 'spir-unknown-unknown', '-x', 'c', '-std=c11',
        '-Xclang', '-fdefault-calling-conv=cdecl', '-O0',
        '-Xclang', '-disable-O0-optnone', '-fomit-frame-pointer',
        '-ffp-contract=off', '-emit-llvm', '-c', str(status_source),
        '-I' + str(install / 'include'), '-o', str(status_module),
    ])
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        args.source.resolve() / 'buffer-copy-library.json',
        '--host-llvm', status_module, '-o', output / 'status-branch-artifact',
    ], 'does not support control flow around setup calls')

    wrong_kernel_source = output / 'wrong-dispatch-kernel.c'
    wrong_kernel_module = output / 'wrong-dispatch-kernel.bc'
    dispatch_source = (args.source.resolve() / 'dispatch.c').read_text(
        encoding='utf-8'
    )
    wrong_kernel_source.write_text(
        dispatch_source.replace(
            'clCreateKernel(program, "square_plus_one"',
            'clCreateKernel(program, "wrong_kernel"',
        ),
        encoding='utf-8',
    )
    run([
        args.clang, '-target', 'spir-unknown-unknown', '-x', 'c', '-std=c11',
        '-Xclang', '-fdefault-calling-conv=cdecl', '-O0',
        '-Xclang', '-disable-O0-optnone', '-fomit-frame-pointer',
        '-ffp-contract=off', '-emit-llvm', '-c', str(wrong_kernel_source),
        '-I' + str(install / 'include'), '-o', str(wrong_kernel_module),
    ])
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        args.source.resolve() / 'dispatch-library.json',
        '--host-llvm', wrong_kernel_module,
        '--device-llvm', consumer / 'udoncl/hello-unity-library/device.bc',
        '-o', output / 'wrong-dispatch-kernel-artifact',
    ], 'kernel name does not match device_kernel')

    wrong_queue_source = output / 'wrong-dispatch-queue.c'
    wrong_queue_module = output / 'wrong-dispatch-queue.bc'
    wrong_queue_source.write_text(
        dispatch_source.replace(
            'clEnqueueNDRangeKernel(queue, kernel',
            'clEnqueueNDRangeKernel((cl_command_queue)context, kernel',
        ),
        encoding='utf-8',
    )
    run([
        args.clang, '-target', 'spir-unknown-unknown', '-x', 'c', '-std=c11',
        '-Xclang', '-fdefault-calling-conv=cdecl', '-O0',
        '-Xclang', '-disable-O0-optnone', '-fomit-frame-pointer',
        '-ffp-contract=off', '-emit-llvm', '-c', str(wrong_queue_source),
        '-I' + str(install / 'include'), '-o', str(wrong_queue_module),
    ])
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        args.source.resolve() / 'dispatch-library.json',
        '--host-llvm', wrong_queue_module,
        '--device-llvm', consumer / 'udoncl/hello-unity-library/device.bc',
        '-o', output / 'wrong-dispatch-queue-artifact',
    ], 'enqueue and release handles must match their created objects')

    mismatched_launch_config = output / 'mismatched-dispatch-buffer.json'
    dispatch_config = json.loads(
        (args.source.resolve() / 'dispatch-library.json').read_text(
            encoding='utf-8'
        )
    )
    dispatch_config['buffers'][0]['byte_size'] = 8
    mismatched_launch_config.write_text(
        json.dumps(dispatch_config), encoding='utf-8'
    )
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        mismatched_launch_config,
        '--host-llvm', consumer / 'udoncl/hello-dispatch-unity-library/host.bc',
        '--device-llvm', consumer / 'udoncl/hello-unity-library/device.bc',
        '-o', output / 'mismatched-dispatch-buffer-artifact',
    ], 'requires one configured four-byte buffer')
    mismatched_semantics_config = output / 'mismatched-dispatch-semantics.json'
    dispatch_config = json.loads(
        (args.source.resolve() / 'dispatch-library.json').read_text(
            encoding='utf-8'
        )
    )
    dispatch_config['buffers'][0]['public_access'] = 'input'
    mismatched_semantics_config.write_text(
        json.dumps(dispatch_config), encoding='utf-8'
    )
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        mismatched_semantics_config,
        '--host-llvm', consumer / 'udoncl/hello-dispatch-unity-library/host.bc',
        '--device-llvm', consumer / 'udoncl/hello-unity-library/device.bc',
        '-o', output / 'mismatched-dispatch-semantics-artifact',
    ], 'public buffer semantics do not match the write-only kernel output')
    print(
        'cmake_package=passed host_sources=3 device_sources=3 '
        'artifacts=hello-artifact,hello-unity-library,'
        'hello-buffer-copy-unity-library,hello-dispatch-unity-library,'
        'hello-int-dispatch-unity-library,saxpy-unity-library'
    )


if __name__ == '__main__':
    main()
