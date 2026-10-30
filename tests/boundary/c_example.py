"""Exercise source compilation, real recovery, and paired lowering together."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

compiler, translate, opt, upstream_opt, upstream_translate, clang, build_dir, output = sys.argv[1:]
root = Path(__file__).resolve().parents[2]
output = Path(output).resolve()
output.mkdir(parents=True, exist_ok=True)
source = root / 'examples/c-boundary'


def run(args, text=None, error=None):
    result = subprocess.run([str(a) for a in args], input=text, capture_output=True, text=True, timeout=60)
    if error is None:
        assert result.returncode == 0, (args, result.stdout, result.stderr)
    else:
        assert result.returncode != 0 and error in result.stderr, (args, result.stdout, result.stderr)
        assert 'PLEASE submit a bug report' not in result.stderr
    return result.stdout


baseline = output / 'baseline'
if (baseline / 'bundle').exists():
    shutil.rmtree(baseline / 'bundle')
run([sys.executable, source / 'run.py', '--build-dir', build_dir, '--clang', clang, '--output', baseline])
bundle = baseline / 'bundle'
recovered = (bundle / 'frontend.recovered.mlir').read_text()
assert 'call @add_impl' in recovered and 'call @__vrc_dispatch' in recovered
assert '!vrc.buffer' in recovered and 'gpu.func @square' in recovered
assert 'memref.store' in recovered and 'arith.mulf' in recovered
assert 'llvm.func ' not in recovered and 'llvm.store ' not in recovered
assert 'vrc.host_frontend' in recovered and 'vrc.device_frontend' in recovered
manifest = json.loads((bundle / 'manifest.json').read_text())
assert manifest['ready_for_unity'] is False
shader = (bundle / 'kernels/kernel_7.hlsl').read_text()
assert ' * ' in shader and 'asuint' in shader
# Replaying the imported artifact through registered PassManager passes must
# reproduce recovery. The example never substitutes a hand-written GPU body.
pipeline = 'builtin.module(vrc-recover-c-boundary)'
replayed = run([opt, '--pass-pipeline=' + pipeline, bundle / 'frontend.imported.mlir'])
# LLVM's parser materializes the optional default unnamed_addr property;
# an in-memory import may omit it from the retained provenance dictionary.
normalize = lambda text: text.replace('unnamed_addr = 0 : i64, ', '')
assert normalize(run([opt], replayed)) == normalize(run([opt], recovered))

# Differential Host oracle includes the newly recovered direct call, normal C
# integer behavior and descriptor/buffer ordering through a native stub.
harness = output / 'harness.c'
harness.write_text('''#include <assert.h>
int add(int, int);
unsigned request_square(unsigned, unsigned);
unsigned __vrc_dispatch(unsigned desc, unsigned buffer) {
  assert(desc == 11 && buffer == 9); return 77;
}
int main(void) {
  assert(add(20,22)==42); assert(add(-10,3)==-7);
  assert(request_square(9,11)==77); return 0;
}
''')
imported = run([translate, '--import-llvm', baseline / 'host.ll'])
raised = run([opt, '--pass-pipeline=builtin.module(vrc-recover-high-level)'], imported)
back = run([upstream_opt, '--convert-arith-to-llvm', '--convert-func-to-llvm', '--reconcile-unrealized-casts'], raised)
llvm_back = output / 'host-roundtrip.ll'
llvm_back.write_text(run([upstream_translate, '--mlir-to-llvmir'], back))
suffix = '.exe' if sys.platform == 'win32' else ''
for name, implementation in [('original', source / 'host.c'), ('raised', llvm_back)]:
    executable = output / (name + suffix)
    run([clang, harness, implementation, '-Wno-override-module', '-o', executable])
    run([executable])

# Mutating C must change generated code, while exported Host ABI stays stable.
variant = output / 'variant-source'
variant.mkdir(exist_ok=True)
for path in source.glob('*'):
    if path.is_file(): shutil.copyfile(path, variant / path.name)
(variant / 'device.cl').write_text((source / 'device.cl').read_text().replace('x * x', 'x + x'))
variant_out = output / 'variant'
if (variant_out / 'bundle').exists(): shutil.rmtree(variant_out / 'bundle')
run([sys.executable, source / 'run.py', '--build-dir', build_dir, '--clang', clang,
     '--source-dir', variant, '--output', variant_out])
variant_shader = (variant_out / 'bundle/kernels/kernel_7.hlsl').read_text()
assert ' + ' in variant_shader and variant_shader != shader
assert json.loads((variant_out / 'bundle/manifest.json').read_text())['exports'] == manifest['exports']

shader_validation = 'unavailable'
glslang = shutil.which('glslangValidator')
if glslang:
    for name, asset in [('square', bundle), ('twice', variant_out / 'bundle')]:
        binary = output / (name + '.test.spv')
        run([glslang, '-D', '-V', '-S', 'frag', '-e', 'frag', '--auto-map-bindings',
             asset / 'kernels/kernel_7.hlsl', '-o', binary])
        validator = shutil.which('spirv-val')
        if validator: run([validator, binary])
    shader_validation = '2 generated HLSL fragments compiled; SPIR-V is only a test artifact'

# Recompile unsupported source patterns and require proof failures, not crashes.
rejected = 0
for file, old, new, diagnostic in [
    ('host.c', '__vrc_dispatch(desc, buffer)', '__vrc_dispatch(buffer, desc)', 'forwarding order'),
    ('host.c', '__vrc_dispatch(desc, buffer)', '__vrc_dispatch(desc + 1, buffer)', 'without arithmetic'),
    ('device.cl', 'output[0]', 'output[1]', 'unsupported kernel operation'),
    ('device.cl', 'x * x', 'output[0] + x', 'unsupported kernel operation'),
    ('device.cl', '__global float', '__local float', 'signature must be'),
    ('device.cl', '__global float *output', '__global int *output', 'pointee/launch metadata'),
    ('device.cl', '__kernel void', '__attribute__((reqd_work_group_size(2,1,1))) __kernel void', 'pointee/launch metadata'),
    ('device.cl', 'output[0] =', 'barrier(CLK_GLOBAL_MEM_FENCE); output[0] =', 'unsupported kernel operation'),
]:
    for path in ['host.c', 'device.cl']:
        (variant / path).write_text((source / path).read_text())
    (variant / file).write_text((variant / file).read_text().replace(old, new))
    run([sys.executable, source / 'run.py', '--build-dir', build_dir, '--clang', clang,
         '--source-dir', variant, '--output', output / ('reject-' + str(rejected))], error=diagnostic)
    rejected += 1
config = json.loads((source / 'boundary.json').read_text())
for key, value, diagnostic in [('schema_version', 2, 'config version'), ('global_size', [2,1,1], 'launch profile'),
                               ('kernel', 'missing', 'defined two-argument'), ('export', 'missing', 'configured C symbols')]:
    changed = dict(config)
    changed[key] = value
    path = output / ('bad-config-' + str(rejected) + '.json')
    path.write_text(json.dumps(changed))
    run([compiler, '--emit-llvm-boundary-bundle', path, '--host-llvm', baseline / 'host.ll',
         '--device-llvm', baseline / 'device.ll', '-o', output / 'must-not-exist'], error=diagnostic)
    rejected += 1
assert not (output / 'must-not-exist').exists()
(output / 'report.json').write_text(json.dumps({'c_to_bundle': 'passed', 'native_host_oracle': 'original and raised passed',
    'source_mutation_changes_shader': True, 'replayed_mlir_pipeline': 'passed', 'rejections': rejected,
    'shader_validation': shader_validation, 'gpu_execution': 'not executed'}, indent=2))
print('C/OpenCL C → LLVM → MLIR → paired boundary bundle, native Host oracle and 12 rejection cases passed')
