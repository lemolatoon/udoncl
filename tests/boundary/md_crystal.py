"""Real MD Host/Fragment execution, gather rejection, and interactive protocol."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

build, output = (Path(x).resolve() for x in sys.argv[1:])
root = Path(__file__).resolve().parents[2]
output.mkdir(parents=True, exist_ok=True)
suffix = '.exe' if sys.platform == 'win32' else ''
for name in ['md_drift-bundle', 'md_kick-bundle']:
    if (output/name).exists():
        shutil.rmtree(output/name)


def run(command, error=None):
    p = subprocess.run([str(x) for x in command], text=True, capture_output=True, timeout=600)
    if error is None:
        assert p.returncode == 0, (command, p.stdout, p.stderr)
    else:
        assert p.returncode != 0 and error in p.stderr, (command, p.stdout, p.stderr)
    return p.stdout


stdout = run([sys.executable, root/'examples/md-crystal/run.py', '--build-dir', build, '--output', output])
assert all(marker in stdout for marker in [
    'NVE_steps=1000', 'phase_cycle=ice-liquid-ice', 'oracle_steps=100',
    'draws=23200', 'max_cpu_gpu_error=0'])
udonhl = (output/'host.udonhl.mlir').read_text()
for api in ['clCreateBuffer', 'clSetKernelArg', 'clEnqueueNDRangeKernel', 'clEnqueueReadBuffer']:
    assert 'udonhl.call @'+api in udonhl
for name in ['md_drift', 'md_kick']:
    bundle = output/(name+'-bundle')
    recovered = (bundle/'frontend.recovered.mlir').read_text()
    assert 'scf.while' in recovered and 'memref.load' in recovered and 'vrc.gather_proven' in recovered
    manifest = json.loads((bundle/'manifest.json').read_text())
    assert manifest['artifact_stage'] == 'native-device-prototype' and manifest['ready_for_unity'] is False
    kernel = manifest['device']['kernels'][0]
    assert kernel['descriptor']['resource_order'] == [0, 1] and kernel['descriptor']['distinct_resources'] is True
    assert kernel['descriptor']['scalar_payload_bytes'] == 44
    assert kernel['args'][0]['required_bytes'] == kernel['args'][1]['required_bytes'] == 1024

commands = json.loads((output/'commands.json').read_text())
compile_device = next(item['argv'] for item in commands if '-DMD_DRIFT=1' in item['argv'] and '-emit-llvm' in item['argv'])
compiler = build/'tools/vrccl'/('vrccl'+suffix)
source = (root/'examples/md-crystal/device.cl').read_text()
variants = [
    ('output-oob', 'out[word] = value;', 'out[word + 1u] = value;', 'buffer index bounds'),
    ('input-oob', 'state[base]', 'state[base + 256u]', 'buffer index bounds'),
    ('scatter', 'out[word] = value;', 'out[word & 3u] = value;', 'output[global_id]'),
    ('write-input', 'out[word] = value;', '((__global float *)state)[word] = value;', 'output[global_id]'),
    ('barrier', 'out[word] = value;', 'barrier(CLK_GLOBAL_MEM_FENCE); out[word] = value;', 'unsupported builtin'),
    ('wrong-dimension', 'get_global_id(0)', 'get_global_id(1)', 'dimension-zero'),
    ('unbounded-loop', '++j)', 'j += 0u)', 'buffer index bounds'),
]
for name, old, new, error in variants:
    assert old in source
    path = output/(name+'.cl')
    path.write_text(source.replace(old, new), encoding='utf-8')
    ir = output/(name+'.ll')
    command = [str(x) for x in compile_device]
    command[command.index(str(root/'examples/md-crystal/device.cl'))] = str(path)
    command[-1] = str(ir)
    run(command)
    destination = output/(name+'-rejected')
    run([compiler, '--emit-native-device-bundle', output/'md_drift.json', '--device-llvm', ir, '-o', destination], error=error)
    assert not destination.exists(), 'failed proof must not publish a bundle'

# A different allocation stride or global-pointer index width must not be
# silently interpreted as the native profile's packed f32 array.
compiler_ir = (output/'md_drift.ll').read_text()
import re
for name, layout_suffix in [('padded-f32', '-f32:64'), ('global-pointer64', '-p1:64:64')]:
    ir = output/(name+'.ll')
    ir.write_text(re.sub(r'(target datalayout = "[^"]*)"',
                         lambda match: match.group(1)+layout_suffix+'"', compiler_ir), encoding='utf-8')
    destination = output/(name+'-rejected')
    run([compiler, '--emit-native-device-bundle', output/'md_drift.json', '--device-llvm', ir, '-o', destination],
        error='four-byte f32 and 32-bit global pointer/index layout')
    assert not destination.exists()

native = [str(output/'runtime-build'/('md-crystal'+suffix))]
if sys.platform == 'win32':
    native += [str(output/'md_drift-bundle/kernels/kernel_10.hlsl'), str(output/'md_kick-bundle/kernels/kernel_11.hlsl'), str(output/'fullscreen.hlsl')]
else:
    native += [str(output/f'shader_{i}.spv') for i in range(3)]
env = dict(os.environ)
if sys.platform != 'win32':
    env['VK_DRIVER_FILES'] = env['VK_ICD_FILENAMES'] = str(next(Path('/usr/share/vulkan/icd.d').glob('lvp_icd*.json')))
p = subprocess.run([*native, '--interactive'], input='step 8 4.5 3.8 -100 -100 0.04\nreset\nquit\n', text=True, capture_output=True, env=env, timeout=60)
assert p.returncode == 0, p.stderr
states = [json.loads(line) for line in p.stdout.splitlines()]
assert [s['frame'] for s in states] == [0, 8, 0]
assert states[0]['state'] != states[1]['state'] and states[0]['state'] == states[2]['state']
assert [s['phase'] for s in states] == ['ice', 'ice', 'ice']
assert all('temperature' in s and 'bond_order_6' in s for s in states)
report = dict(stdout=stdout, runtime_rejection_cases=6, rejected_gather_cases=[v[0] for v in variants]+['padded-f32', 'global-pointer64'], interactive_frames=[s['frame'] for s in states])
(output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(stdout, end='')
print('gather_rejections=9 interactive_step_reset=passed')
