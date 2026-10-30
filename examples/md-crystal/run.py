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
    compiler = build/'tools/vrccl'/('vrccl'+suffix)
    opt = build/'tools/vrc-opt'/('vrc-opt'+suffix)
    translate = build/'tools/vrc-translate'/('vrc-translate'+suffix)
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
         '-DUDONCL_HOST_OBJECT='+str(host_obj), '-DUDONCL_DRIFT_OBJECT='+str(objects[0]),
         '-DUDONCL_KICK_OBJECT='+str(objects[1]), '-DUDONCL_BINDINGS_DIR='+str(output)])
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
