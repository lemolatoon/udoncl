"""Build and run the MD OpenCL example through udoncl and native fragments."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=root / 'build/udoncl')
    parser.add_argument('--output', type=Path, default=root / 'build/md-crystal')
    parser.add_argument('--serve', action='store_true', help='Open an interactive local browser view')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    build, output = args.build_dir.resolve(), args.output.resolve()
    source = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if sys.platform != 'win32':
        icds = list(Path('/usr/share/vulkan/icd.d').glob('lvp_icd*.json'))
        if not icds:
            parser.error('install mesa-vulkan-drivers and libvulkan-dev')
        env['VK_DRIVER_FILES'] = env['VK_ICD_FILENAMES'] = str(icds[0])
    commands = []

    def run(command):
        command = [str(x) for x in command]
        result = subprocess.run(command, text=True, capture_output=True, env=env, timeout=600)
        commands.append(dict(argv=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        (output/'commands.json').write_text(json.dumps(commands, indent=2), encoding='utf-8')
        if result.returncode:
            sys.stderr.write(result.stdout + result.stderr)
            raise SystemExit(result.returncode)
        return result.stdout

    suffix = '.exe' if sys.platform == 'win32' else ''
    cache = (build/'CMakeCache.txt').read_text()
    sdk = Path(next(line.split('=', 1)[1] for line in cache.splitlines() if line.startswith('LLVM_DIR:'))).parents[2]
    clang = sdk/'bin'/('clang'+suffix)
    compiler = build/'tools/vrccl'/('udonclc'+suffix)
    opt = build/'tools/vrc-opt'/('udoncl-opt'+suffix)
    translate = build/'tools/vrc-translate'/('udoncl-translate'+suffix)
    llvm_translate = sdk/'bin'/('mlir-translate'+suffix)
    flags = ['-fms-runtime-lib=dll'] if sys.platform == 'win32' else []
    ext = '.obj' if sys.platform == 'win32' else '.o'
    text = (source/'device.cl').read_text(encoding='utf-8')
    (output/'device_source.h').write_text('#define UDONCL_DEVICE_SOURCE '+json.dumps(text)+'\n', encoding='utf-8')
    (output/'Bindings.h').write_text('inline const char *deviceSource() { return '+json.dumps(text)+'; }\n', encoding='utf-8')
    # Verify the unselected source submitted by Host is a valid program with
    # both kernels; the AOT selections below only specialize each entry point.
    program_ir = output/'program.ll'
    run([clang, '-target', 'spir-unknown-unknown', '-x', 'cl', '-cl-std=CL1.2',
         '-S', '-emit-llvm', source/'device.cl', '-o', program_ir])
    program_text = program_ir.read_text()
    if any('spir_kernel void @'+name+'(' not in program_text for name in ['md_drift', 'md_kick']):
        raise RuntimeError('the Host program source must contain both OpenCL entry points')
    fragments, objects, bindings = [], [], []
    for name, identifier, defines in [('md_drift', 10, ['-DMD_DRIFT=1']), ('md_kick', 11, ['-DMD_KICK=1'])]:
        ir = output/(name+'.ll')
        run([clang, '-target', 'spir-unknown-unknown', '-x', 'cl', '-cl-std=CL1.2', '-cl-kernel-arg-info', *defines,
             '-O0', '-Xclang', '-disable-O0-optnone', '-fomit-frame-pointer', '-ffp-contract=off', '-S', '-emit-llvm', source/'device.cl', '-o', ir])
        config = output/(name+'.json')
        config.write_text(json.dumps(dict(schema_version=1, profile='gather-f32', words=256, kernel=name, kernel_id=identifier,
                                         global_size=[256, 1, 1], local_size=[1, 1, 1], global_offset=[0, 0, 0])), encoding='utf-8')
        bundle = output/(name+'-bundle')
        # Publish only a complete regenerated bundle. This example owns its
        # named build artifacts; the compiler itself keeps immutable bundles.
        with tempfile.TemporaryDirectory(prefix=name+'-', dir=output) as staging:
            staged = Path(staging)/'bundle'
            run([compiler, '--emit-native-device-bundle', config, '--device-llvm', ir, '-o', staged])
            if bundle.exists():
                previous = json.loads((bundle/'manifest.json').read_text())
                if previous.get('artifact_stage') != 'native-device-prototype':
                    raise RuntimeError('refusing to replace a non-example bundle')
                shutil.rmtree(bundle)
            staged.rename(bundle)
        manifest = json.loads((bundle/'manifest.json').read_text())
        kernel = manifest['device']['kernels'][0]
        if kernel['args'][0]['access'] != 'read' or kernel['args'][1]['access'] != 'write' or kernel['target_texture']['width'] != 256:
            raise RuntimeError('gather manifest does not match native bindings')
        bindings.append([kernel['id'], kernel['target_texture']['width'],
                         kernel['descriptor']['scalar_payload_bytes']//4, kernel['name']])
        fragments.append(bundle/f'kernels/kernel_{identifier}.hlsl')
        obj = output/(name+ext)
        run([clang, *flags, '-x', 'c', '-std=c11', '-ffp-contract=off', *defines, '-c', source/'device.cl', '-o', obj])
        objects.append(obj)
    binding_code = []
    for record, path_name in zip(bindings, ['drift', 'kick']):
        binding_code.append('{'+', '.join(json.dumps(value) for value in record)+', '+path_name+'}')
    (output/'Bindings.h').write_text(
        'inline const char *deviceSource() { return '+json.dumps(text)+'; }\n'+
        'inline std::vector<native_opencl::GatherBinding> manifestBindings(const std::string &drift, const std::string &kick) { return {'+
        ', '.join(binding_code)+'}; }\n', encoding='utf-8')

    # Publish the Unity-facing host + two Device kernels as one validated
    # artifact. The native per-kernel manifests remain evidence for the
    # compiler boundary; this combined manifest is the Editor import contract.
    unity_bundle = output/'unity-bundle'
    with tempfile.TemporaryDirectory(prefix='unity-', dir=output) as staging:
        staged = Path(staging)/'bundle'
        (staged/'kernels').mkdir(parents=True)
        shutil.copy2(source/'host.uasm', staged/'host.uasm')
        shutil.copy2(source/'display.shader', staged/'display.shader')
        for identifier, bundle in [(10, output/'md_drift-bundle'),
                                   (11, output/'md_kick-bundle')]:
            shutil.copy2(bundle/f'kernels/kernel_{identifier}.shader',
                         staged/f'kernels/kernel_{identifier}.shader')
        unity_manifest = {
            'schema_version': 1,
            'compiler_version': '0.1.0-dev',
            'runtime_abi_version': 1,
            'artifact_stage': 'unity-lj-prototype',
            'ready_for_unity': True,
            'sdk_compatibility': {'sdk_version': '3.10.5', 'extern_db_hash': None},
            'target': {'host_pointer_bits': 32, 'device_pointer_bits': 32,
                       'endianness': 'little', 'platform': 'VRChat-PC'},
            'program': {'name': 'md_crystal', 'assembly': 'host.uasm',
                        'display_shader': 'display.shader'},
            'exports': [
                {'id': 1, 'c_symbol': 'md_init', 'udon_event': '_start',
                 'status_slot': '', 'async': False},
                {'id': 2, 'c_symbol': 'md_step', 'udon_event': '_update',
                 'status_slot': '', 'async': False},
                {'id': 3, 'c_symbol': 'md_heat', 'udon_event': '_interact',
                 'status_slot': '', 'async': False},
            ],
            'device': {'kernels': [
                {'id': 10, 'name': 'md_drift', 'shader': 'kernels/kernel_10.shader'},
                {'id': 11, 'name': 'md_kick', 'shader': 'kernels/kernel_11.shader'},
            ]},
        }
        (staged/'manifest.json').write_text(json.dumps(unity_manifest, indent=2),
                                           encoding='utf-8')
        if unity_bundle.exists():
            previous = json.loads((unity_bundle/'manifest.json').read_text())
            if previous.get('artifact_stage') != 'unity-lj-prototype':
                raise RuntimeError('refusing to replace a non-example Unity bundle')
            shutil.rmtree(unity_bundle)
        # Moving a populated directory out of TemporaryDirectory intermittently
        # fails with WinError 5 on Windows even after the prior bundle is gone.
        # Copy into the already validated, example-owned destination instead;
        # the temporary tree is removed when the context exits.
        shutil.copytree(staged, unity_bundle)
    host_ir = output/'host.ll'
    run([clang, *flags, '-x', 'c', '-std=c11', '-O0', '-Xclang', '-disable-O0-optnone', '-fomit-frame-pointer',
         '-I', root/'third_party/OpenCL-Headers', '-I', output, '-S', '-emit-llvm', source/'host.c', '-o', host_ir])
    imported, udonhl, lowered, native = (output/name for name in ['host.imported.mlir', 'host.udonhl.mlir', 'host.native.llvm.mlir', 'host.native.ll'])
    run([translate, '--import-llvm', host_ir, '-o', imported])
    run([opt, '--vrc-raise-native-host-calls', '--verify-each', imported, '-o', udonhl])
    run([opt, '--vrc-lower-native-host-calls', '--verify-each', udonhl, '-o', lowered])
    native.write_text(run([llvm_translate, '--mlir-to-llvmir', lowered]), encoding='utf-8')
    host_obj = output/('host'+ext)
    run([clang, *flags, '-c', native, '-o', host_obj])
    vertex = output/'fullscreen.hlsl'
    vertex.write_text('float4 vert(uint id : SV_VertexID) : SV_Position { return float4(id == 2 ? 3.0 : -1.0, id == 1 ? 3.0 : -1.0, 0, 1); }\n', encoding='utf-8')
    if sys.platform != 'win32':
        glslang = shutil.which('glslangValidator')
        if not glslang:
            parser.error('install glslang-tools')
        binaries = []
        for i, path in enumerate([*fragments, vertex]):
            binary = output/f'shader_{i}.spv'
            stage, entry = ('vert', 'vert') if i == 2 else ('frag', 'frag')
            run([glslang, '-D', '-V', '-S', stage, '-e', entry, '--auto-map-locations', path, '-o', binary])
            if shutil.which('spirv-val'):
                run([shutil.which('spirv-val'), binary])
            binaries.append(binary)
        fragments, vertex = binaries[:2], binaries[2]
    runtime_build = output/'runtime-build'
    run(['cmake', '-S', source, '-B', runtime_build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
         '-DUDONCL_HOST_OBJECT='+host_obj.as_posix(), '-DUDONCL_DRIFT_OBJECT='+objects[0].as_posix(),
         '-DUDONCL_KICK_OBJECT='+objects[1].as_posix(), '-DUDONCL_BINDINGS_DIR='+output.as_posix()])
    run([sys.executable, root/'scripts/build.py', '--build-dir', runtime_build])
    executable = [str(x) for x in [runtime_build/('md-crystal'+suffix), *fragments, vertex]]
    if args.serve:
        from viewer import serve
        serve(executable, env, args.port)
        return
    stdout = run(executable)
    print(stdout, end='')


if __name__ == '__main__':
    main()
