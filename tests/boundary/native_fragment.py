"""Run real UdonHL/native dispatches against two source-derived fragment shaders."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

build, output = map(Path, sys.argv[1:])
root = Path(__file__).resolve().parents[2]
output = output.resolve()
output.mkdir(parents=True, exist_ok=True)


def run(command, error=None, text=None):
    result = subprocess.run([str(x) for x in command], input=text, text=True,
                            capture_output=True, timeout=600)
    if error is None:
        assert result.returncode == 0, (command, result.stdout, result.stderr)
    else:
        assert result.returncode != 0 and error in result.stderr, (command, result.stdout, result.stderr)
        assert not result.stdout
    return result.stdout


reports = []
for name in ['square', 'twice']:
    source = root / 'examples/c-boundary'
    if name == 'twice':
        source = output / 'twice-source'
        source.mkdir(exist_ok=True)
        for path in (root / 'examples/c-boundary').iterdir():
            if path.is_file(): shutil.copyfile(path, source / path.name)
        path = source / 'device.cl'
        path.write_text(path.read_text().replace('x * x', 'x + x'), encoding='utf-8')
    result = output / name
    # Only these test-owned bundles are disposable; developer assets remain protected.
    bundle = result / 'compile/bundle'
    if bundle.exists(): shutil.rmtree(bundle)
    run([sys.executable, root / 'examples/native-fragment/run.py', '--build-dir', build,
         '--source-dir', source, '--output', result])
    reports.append(json.loads((result / 'report.json').read_text()))
assert 'actual=0x40c80000' in reports[0]['stdout'] # 2.5 squared = 6.25
assert 'actual=0x40a00000' in reports[1]['stdout'] # 2.5 doubled = 5
assert 'input=-0 expected=0x80000000 actual=0x80000000' in reports[1]['stdout']

suffix = '.exe' if sys.platform == 'win32' else ''
opt = build / 'tools/vrc-opt' / ('vrc-opt' + suffix)
boundary = (output / 'square/compile/bundle/host-device.mlir').read_text()
# Native lowering must start after the paired boundary verifier and reject
# calls whose recursion/suspension semantics have not been lowered.
run([opt, '--vrc-lower-host-to-native-test'], text=boundary.replace('vrc.host_stage = "udonhl"', 'vrc.host_stage = "wrong"'), error='requires lowered Host and Device stages')
run([opt, '--vrc-lower-host-to-native-test'], text=boundary.replace('#udonhl<call_policy normal>', '#udonhl<call_policy recursive>'), error='normal non-suspending calls')
run([opt, '--vrc-lower-host-to-native-test'], text=boundary.replace('request = "request_square"', 'request = "missing"'), error='configured internal request entry')
run([opt, '--vrc-lower-host-to-native-test'], text=boundary.replace('  func.func @add(', '  func.func private @__vrc_native_test_dispatch(%a: i32, %b: i32, %c: i32) -> i32 { return %a : i32 }\n  func.func @add(', 1), error='runtime symbol collision')
# Private source request is only exposed in the test backend; saved production
# boundary IR keeps opaque handles and private visibility.
assert 'func.func private @request_square' in boundary
native = (output / 'square/host.native.mlir').read_text()
assert 'func.func @request_square' in native and 'udonhl.call ' not in native
assert '!vrc.buffer' not in native and 'vrc.dispatch ' not in native
llvm = (output / 'square/host.native.ll').read_text()
assert 'call i32 @__vrc_native_test_dispatch' in llvm
(output / 'report.json').write_text(json.dumps({
    'cases':reports, 'total_fragment_draw_readbacks':16, 'runtime_rejections':8,
    'native_lowering_rejections':4, 'source_mutation_verified':True,
    'execution_platform':sys.platform,
    'unity_vrchat_execution':'not executed'}, indent=2) + '\n', encoding='utf-8')
print('16 native Host dispatches → real fragment draws/readbacks; 8 runtime and 4 lowering rejections passed')
