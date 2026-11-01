"""C -> UdonHL -> native Host binary -> real fragment rendering and readback."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=root / 'build/udoncl')
    parser.add_argument('--output', type=Path, default=root / 'build/native-fragment')
    parser.add_argument('--source-dir', type=Path, default=root / 'examples/c-boundary')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.build_dir = args.build_dir.resolve()
    args.source_dir = args.source_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    commands = []
    environment = dict(os.environ)
    if sys.platform != 'win32':
        icds = list(Path('/usr/share/vulkan/icd.d').glob('lvp_icd*.json'))
        if not icds:
            parser.error('install Mesa Lavapipe (mesa-vulkan-drivers) for the native fragment test')
        environment['VK_DRIVER_FILES'] = environment['VK_ICD_FILENAMES'] = str(icds[0])

    def run(command, error=None, text=None):
        result = subprocess.run([str(x) for x in command], input=text, text=True,
                                capture_output=True, env=environment, timeout=600)
        commands.append({'argv': [str(x) for x in command], 'returncode': result.returncode,
                         'stdout': result.stdout, 'stderr': result.stderr})
        (args.output / 'commands.json').write_text(json.dumps(commands, indent=2), encoding='utf-8')
        if error is None:
            if result.returncode:
                sys.stderr.write(result.stdout + result.stderr)
                raise SystemExit(result.returncode)
        elif result.returncode == 0 or error not in result.stderr:
            raise RuntimeError(f'expected rejection {error}: {result.stdout} {result.stderr}')
        return result.stdout

    suffix = '.exe' if sys.platform == 'win32' else ''
    cache = (args.build_dir / 'CMakeCache.txt').read_text()
    llvm_dir = next(line.split('=',1)[1] for line in cache.splitlines() if line.startswith('LLVM_DIR:'))
    sdk = Path(llvm_dir).parents[2]
    clang = sdk / 'bin' / ('clang' + suffix)
    upstream_opt = sdk / 'bin' / ('mlir-opt' + suffix)
    upstream_translate = sdk / 'bin' / ('mlir-translate' + suffix)
    opt = args.build_dir / 'tools/vrc-opt' / ('vrc-opt' + suffix)
    run([sys.executable, root / 'examples/c-boundary/run.py', '--build-dir', args.build_dir,
         '--source-dir', args.source_dir, '--output', args.output / 'compile'])
    bundle = args.output / 'compile/bundle'
    native_mlir = args.output / 'host.native.mlir'
    run([opt, '--vrc-lower-host-to-native-test', '--verify-each', bundle / 'host-device.mlir', '-o', native_mlir])
    lowered = run([upstream_opt, '--convert-arith-to-llvm', '--convert-func-to-llvm',
                   '--reconcile-unrealized-casts', '--verify-each', native_mlir])
    (args.output / 'host.native.llvm.mlir').write_text(lowered, encoding='utf-8')
    llvm_ir = args.output / 'host.native.ll'
    llvm_ir.write_text(run([upstream_translate, '--mlir-to-llvmir'], text=lowered), encoding='utf-8')
    native_ir = native_mlir.read_text()
    if 'udonhl.call ' in native_ir or 'vrc.dispatch ' in native_ir or '__vrc_native_test_dispatch' not in native_ir:
        raise RuntimeError('Host native lowering did not remove the custom call/dispatch operations')
    extension = '.obj' if sys.platform == 'win32' else '.o'
    host_object = args.output / ('host' + extension)
    oracle_object = args.output / ('oracle' + extension)
    runtime_flags = ['-fms-runtime-lib=dll'] if sys.platform == 'win32' else []
    run([clang, *runtime_flags, '-c', '-O0', '-Wno-override-module', llvm_ir, '-o', host_object])
    # This independent CPU oracle compiles the same restricted Device source as
    # C, erasing only the OpenCL kernel/address-space keywords. It never replaces
    # the Device shader or supplies a result to the generated Host dispatch.
    run([clang, *runtime_flags, '-x', 'c', '-std=c11', '-ffp-contract=off', '-D__kernel=', '-D__global=',
         '-c', args.source_dir / 'device.cl', '-o', oracle_object])
    config = json.loads((args.source_dir / 'boundary.json').read_text())
    for key in ['export', 'request', 'kernel']:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', config[key]):
            raise RuntimeError('native example requires C identifier symbol names')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    export = next(item for item in manifest['exports'] if item['c_symbol'] == config['export'])
    if [item['kind'] for item in export['parameters']] != ['i32','i32'] or export['result']['kind'] != 'i32':
        raise RuntimeError('native example scalar probe requires (i32,i32)->i32')
    kernel = next(item for item in manifest['device']['kernels'] if item['id'] == config['kernel_id'])
    texture = kernel['target_texture']
    binding = [kernel['id'], kernel['args'][1]['binding'], kernel['args'][0]['required_bytes'],
               texture['width'], texture['height'], texture['format'], texture['srgb'], texture['mipmaps']]
    literal = ', '.join(json.dumps(value) for value in binding)
    (args.output / 'Bindings.h').write_text(
        '#define NATIVE_ADD ' + config['export'] + '\n#define NATIVE_REQUEST ' + config['request'] +
        '\n#define NATIVE_ORACLE ' + config['kernel'] + '\n' +
        'inline native_test::KernelBinding manifestBinding() { return {' + literal + '}; }\n', encoding='utf-8')
    fragment = bundle / ('kernels/kernel_' + str(config['kernel_id']) + '.hlsl')
    vertex = args.output / 'fullscreen.hlsl'
    vertex.write_text('float4 vert(uint id : SV_VertexID) : SV_Position { return float4(id == 2 ? 3.0 : -1.0, id == 1 ? 3.0 : -1.0, 0.0, 1.0); }\n', encoding='utf-8')
    if sys.platform != 'win32':
        glslang = shutil.which('glslangValidator')
        if not glslang:
            parser.error('install glslang-tools to compile generated HLSL for the Vulkan test')
        for name, stage, entry, source in [('fragment','frag','frag',fragment), ('vertex','vert','vert',vertex)]:
            binary = args.output / (name + '.spv')
            run([glslang, '-D', '-V', '-S', stage, '-e', entry, '--auto-map-bindings', '--auto-map-locations', source, '-o', binary])
            validator = shutil.which('spirv-val')
            if validator: run([validator, binary])
        fragment, vertex = args.output / 'fragment.spv', args.output / 'vertex.spv'
    runtime_build = args.output / 'runtime-build'
    run(['cmake', '-S', root / 'examples/native-fragment', '-B', runtime_build, '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Release', '-DUDONCL_HOST_OBJECT=' + host_object.as_posix(),
         '-DUDONCL_ORACLE_OBJECT=' + oracle_object.as_posix(), '-DUDONCL_BINDINGS_DIR=' + args.output.as_posix()])
    run([sys.executable, root / 'scripts/build.py', '--build-dir', runtime_build])
    executable = runtime_build / ('native-fragment' + suffix)
    command = [executable, fragment, vertex, config['kernel_id']]
    stdout = run(command)
    print(stdout, end='')
    if 'passed=8 ' not in stdout:
        raise RuntimeError('missing native/fragment execution evidence')
    for mode, diagnostic in [('kernel','kernel id mismatch'), ('descriptor','invalid descriptor handle'),
                             ('buffer','invalid buffer handle'), ('generation','stale buffer generation')]:
        run([*command, mode], error=diagnostic)
    # Lowering must use the generated Host's kernel identity; the runtime does
    # not infer or replace it from the expected result or the Device source.
    report = {'host_backend':'UdonHL -> func/arith -> LLVM -> native object',
              'renderer': stdout.splitlines()[0], 'fragment_draw_readbacks':8,
              'bitwise_oracle':'same Device source compiled independently as C',
              'runtime_rejections':4, 'stdout':stdout,
              'unity_vrchat_execution':'not executed; this is a native test adapter'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
