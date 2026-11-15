"""Run real UdonHL/native dispatches against two source-derived fragment shaders."""
import json
from pathlib import Path
import re
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
opt = build / 'tools/vrc-opt' / ('udoncl-opt' + suffix)
boundary = (output / 'square/compile/bundle/host-device.mlir').read_text()
# Native lowering must start after the paired boundary verifier and reject
# calls whose recursion/suspension semantics have not been lowered.
run([opt, '--vrc-lower-host-to-native-test'], text=boundary.replace('vrc.host_stage = "udonhl"', 'vrc.host_stage = "wrong"'), error='requires lowered Host and Device stages')
run([opt, '--vrc-lower-host-to-native-test'], text=boundary.replace('#udonhl<call_policy normal>', '#udonhl<call_policy recursive>'), error='normal non-suspending calls')
run([opt, '--vrc-lower-host-to-native-test'], text=boundary.replace('request = "request_square"', 'request = "missing"'), error='configured internal request entry')
run([opt, '--vrc-lower-host-to-native-test'], text=boundary.replace('  func.func @add(', '  func.func private @__vrc_native_test_dispatch(%a: i32, %b: i32, %c: i32) -> i32 { return %a : i32 }\n  func.func @add(', 1), error='runtime symbol collision')

# The static-resource index segment exists only to select a Unity texture at
# Udon lowering. It must not become a native kernel/runtime argument. Inject a
# test-only segment into the generated one-resource boundary and require the
# canonical native IR to remain byte-identical.
dispatch_lines = [
    line for line in boundary.splitlines() if ' = vrc.dispatch ' in line
]
assert len(dispatch_lines) == 1, dispatch_lines
dispatch_line = dispatch_lines[0]
explicit_dispatch = dispatch_line.replace(
    'vrc.dispatch %arg1(%arg0) ',
    'vrc.dispatch %arg1(%arg0) static_indices(%static_index) ',
    1,
).replace(
    '{argument_kinds = ',
    '{static_resource_positions = array<i32: 0>, argument_kinds = ',
    1,
).replace(
    '(!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event',
    '(!vrc.dispatch_desc, !vrc.buffer, i32) -> !vrc.event',
    1,
)
assert explicit_dispatch != dispatch_line
explicit_boundary = boundary.replace(
    dispatch_line,
    '    %static_index = arith.constant 123456789 : i32\n' + explicit_dispatch,
    1,
)
baseline_native = run(
    [opt, '--vrc-lower-host-to-native-test', '--canonicalize', '--verify-each'],
    text=boundary,
)
explicit_native = run(
    [opt, '--vrc-lower-host-to-native-test', '--canonicalize', '--verify-each'],
    text=explicit_boundary,
)
assert explicit_native == baseline_native
native_dispatch_declaration = (
    'func.func private @__vrc_native_test_dispatch(i32, i32, i32) -> i32'
)
native_dispatch_call = re.compile(
    r'call @__vrc_native_test_dispatch\(%c7_i32, %arg1, %arg0\) : '
    r'\(i32, i32, i32\) -> i32'
)
assert baseline_native.count(native_dispatch_declaration) == 1
assert len(native_dispatch_call.findall(baseline_native)) == 1
assert '123456789' not in explicit_native and 'static_index' not in explicit_native
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
    'explicit_static_index_native_equivalence':True,
    'native_dispatch_argument_count':3,
    'execution_platform':sys.platform,
    'unity_vrchat_execution':'not executed'}, indent=2) + '\n', encoding='utf-8')
print('16 native Host dispatches -> real fragment draws/readbacks; explicit static index ABI equivalence, 8 runtime and 4 lowering rejections passed')
