"""Compile both sides of the boundary and reject incompatible contracts.

Shader validation is optional on developer machines, and mandatory in Linux CI
where glslang-tools is installed. SPIR-V here is only a test artifact.
"""
from pathlib import Path
import copy
import json
import re
import shutil
import subprocess
import sys

opt, compiler, output = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
# These are disposable test-owned bundle directories, not user assets.
for directory in ['bundle', 'variant-bundle']:
    if (output / directory).exists():
        shutil.rmtree(output / directory)
root = Path(__file__).resolve().parents[2]
source_path = root / 'examples/host-device-boundary/probe.mlir'
source = source_path.read_text()
request = json.loads((source_path.parent / 'request.json').read_text())
commands = []
rejections = []

def run(args, text=None, error=None):
    result = subprocess.run([str(a) for a in args], input=text, text=True,
                            capture_output=True, timeout=60)
    commands.append({'argv': [str(a) for a in args], 'returncode': result.returncode,
                     'stderr': result.stderr})
    (output / 'commands.json').write_text(json.dumps(commands, indent=2))
    if error is None:
        assert result.returncode == 0, (args, result.stdout, result.stderr)
    else:
        assert result.returncode != 0 and error in result.stderr, (args, error, result.stdout, result.stderr)
        rejections.append(error)
    return result.stdout

host_pass = 'vrc-lower-host-to-udonhl'
intrinsic_pass = 'vrc-substitute-runtime-intrinsics'
device_pass = 'vrc-lower-device-single-invocation'
verify_pass = 'vrc-verify-host-device'
pipeline = f'builtin.module({host_pass},{intrinsic_pass},{device_pass},{verify_pass})'
host = run([opt, '--pass-pipeline=builtin.module(' + host_pass + ')', '--verify-each'], source)
assert 'udonhl.call @add_impl' in host
assert 'udonhl.call @__vrc_dispatch' in host
assert 'gpu.func @square' in host and 'vrc.dispatch ' not in host
(output / 'host.mlir').write_text(host)
intrinsics = run([opt, '--pass-pipeline=builtin.module(' + intrinsic_pass + ')', '--verify-each'], host)
assert 'vrc.dispatch ' in intrinsics and '@__vrc_dispatch' not in intrinsics
assert 'gpu.func @square' in intrinsics
(output / 'host-intrinsics.mlir').write_text(intrinsics)
lowered = run([opt, '--pass-pipeline=' + pipeline, '--verify-each'], source)
assert '"vrcgpu.kernel"' in lowered and '"vrcgpu.output"' in lowered
assert '!vrcgpu.version<0>' in lowered and 'gpu.func' not in lowered
assert 'udonhl.call' in lowered and 'func.call ' not in lowered
roundtrip = run([opt, '--verify-each'], lowered)
assert run([opt, '--verify-each'], roundtrip) == roundtrip
(output / 'host-device.mlir').write_text(lowered)

# Runtime intrinsics must be substituted in Host order, and GPU functions must
# never be rewritten as Host calls just because both use standard func ops.
run([opt, '--pass-pipeline=builtin.module(' + intrinsic_pass + ')'], source, error='must follow UdonHL')
buffer_runtime = r'''
module attributes {vrc.host_stage = "udonhl"} {
  func.func private @allocate(i32) -> (!vrc.buffer, i32) attributes {vrc.intrinsic = "buffer_alloc", vrc.format = "rgba8_unorm", vrc.usage = "storage"}
  func.func private @copy(!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event attributes {vrc.intrinsic = "buffer_copy"}
  func.func @runtime(%bytes: i32, %source: !vrc.buffer, %zero: i32) -> !vrc.event {
    %destination, %status = udonhl.call @allocate(%bytes) {call_policy = #udonhl<call_policy normal>} : (i32) -> (!vrc.buffer, i32)
    %event = udonhl.call @copy(%source, %destination, %zero, %zero, %bytes) {call_policy = #udonhl<call_policy normal>} : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    return %event : !vrc.event
  }
}
'''
buffer_ops = run([opt, '--pass-pipeline=builtin.module(' + intrinsic_pass + ')',
                  '--verify-each'], buffer_runtime)
assert 'vrc.buffer_alloc ' in buffer_ops and 'vrc.buffer_copy ' in buffer_ops
assert '@allocate' not in buffer_ops and '@copy' not in buffer_ops
nested = source.replace('  gpu.module @device {', '''  gpu.module @device {
    func.func private @device_helper(%x: f32) -> f32 { return %x : f32 }
    func.func private @device_caller(%x: f32) -> f32 {
      %r = func.call @device_helper(%x) : (f32) -> f32
      return %r : f32
    }
''')
host_nested = run([opt, '--pass-pipeline=builtin.module(' + host_pass + ')'], nested)
assert 'call @device_helper' in host_nested and 'udonhl.call @device_helper' not in host_nested
run([opt, '--pass-pipeline=' + pipeline], nested, error='GPU module may contain only')

bundle = output / 'bundle'
run([compiler, '--emit-boundary-bundle', source_path, '-o', bundle])
manifest = json.loads((bundle / 'manifest.json').read_text())
assert manifest['schema_version'] == manifest['runtime_abi_version'] == 1
assert manifest['artifact_stage'] == 'boundary-prototype' and not manifest['ready_for_unity']
assert manifest['program']['assembly'] is None
assert manifest['sdk_compatibility']['sdk_version'] == '3.10.5'
assert manifest['sdk_compatibility']['unity_version'] == '2022.3.22f1'
assert manifest['physical_export_abi'] == 'udonsharp-worlds-sdk-3.10.5'
assert re.fullmatch(r'[0-9a-f]{64}', manifest['public_abi_fingerprint'])
assert manifest['target']['host_pointer_bits'] == manifest['target']['device_pointer_bits'] == 32
export = manifest['exports'][0]
assert export['c_symbol'] == 'add' and export['public_name'] == 'Add' and not export['async']
assert export['udon_event'] == '__0_Add'
assert [p['name'] for p in export['parameters']] == ['left', 'right']
assert [p['heap_slot'] for p in export['parameters']] == [
    '__0_left__param', '__0_right__param']
assert export['result']['heap_slot'] == '__0___0_Add__ret'
assert export['status_slot'] == '__udoncl_status_1'
assert export['parameters'][0]['type_schema'] == {
    'category': 'signed-integer', 'bits': 32}
kernel = manifest['device']['kernels'][0]
assert kernel['id'] == manifest['dispatch_sites'][0]['kernel_id'] == 7
assert kernel['name'] == 'square' and kernel['source_module'] == 'device'
assert kernel['symbol'] == 'device__square'
assert [arg['kind'] for arg in kernel['args']] == ['buffer', 'f32']
assert kernel['descriptor']['scalar_payload_bytes'] == 4
assert kernel['descriptor']['resource_order'] == [0]
assert kernel['target_texture']['format'] == 'rgba8_unorm' and not kernel['target_texture']['srgb']
assert kernel['cost'] == {'kind': 'static', 'pass_count': 1, 'texture_read_bytes': 0,
                          'texture_write_bytes': 4, 'scratch_bytes': 0}
hlsl = (bundle / 'kernels/kernel_7.hlsl').read_text()
shader = (bundle / 'kernels/kernel_7.shader').read_text()
assert hlsl in shader and 'Blend Off' in shader and '#pragma target 5.0' in shader
assert 'precise float v0 = _VRC_Arg1 * _VRC_Arg1;' in hlsl
assert 'asuint(v0)' in hlsl and all(shift in hlsl for shift in ['>> 8u', '>> 16u', '>> 24u'])
assert 'SV_Target' in hlsl

before = {str(p.relative_to(bundle)): p.read_bytes() for p in bundle.rglob('*') if p.is_file()}
run([compiler, '--emit-boundary-bundle', source_path, '-o', bundle])
assert before == {str(p.relative_to(bundle)): p.read_bytes() for p in bundle.rglob('*') if p.is_file()}

def check_request(doc, req, expected=None):
    path, request_path = output / 'check-manifest.json', output / 'check-request.json'
    path.write_text(json.dumps(doc))
    request_path.write_text(json.dumps(req))
    return run([compiler, '--check-boundary-request', path, '--request', request_path], error=expected)

assert 'no GPU dispatch executed' in check_request(manifest, request)
# ABI/schema migration and descriptor mismatches must fail explicitly.
for key in ['schema_version', 'runtime_abi_version']:
    bad = copy.deepcopy(manifest); bad[key] = 2
    check_request(bad, request, 'unsupported manifest/runtime ABI version')
bad = copy.deepcopy(manifest); bad['ready_for_unity'] = True
check_request(bad, request, 'readiness claim')
bad = copy.deepcopy(manifest); bad['target']['device_pointer_bits'] = 64
check_request(bad, request, 'address or byte-order ABI')
bad = copy.deepcopy(manifest); bad['device']['kernels'].append(copy.deepcopy(kernel))
check_request(bad, request, 'duplicate kernel id')
for path in ['../escape.shader', '/tmp/escape.shader', 'C:\\escape.shader']:
    bad = copy.deepcopy(manifest); bad['device']['kernels'][0]['shader'] = path
    check_request(bad, request, 'safe relative asset path')
for field, value in [('kind', 'i32'), ('scalar_payload_offset', 4)]:
    bad = copy.deepcopy(manifest); bad['device']['kernels'][0]['args'][1][field] = value
    check_request(bad, request, 'argument kinds/layout')
for field, value in [('scalar_payload_bytes', 8), ('resource_order', [1]),
                     ('argument_copy_policy', 'zero_copy'), ('byte_order', 'big')]:
    bad = copy.deepcopy(manifest); bad['device']['kernels'][0]['descriptor'][field] = value
    check_request(bad, request, 'descriptor schema')
bad = copy.deepcopy(manifest); bad['device']['kernels'][0]['args'][1]['binding'] = '_WrongBinding'
check_request(bad, request, 'argument kinds/layout')
for field, value, expected in [
    ('kernel_id', 99, 'missing kernel id'), ('generation', 2, 'stale generation'),
    ('scalar_payload', [0,0,32], 'payload length mismatch'),
    ('scalar_payload', [0,0,32,256], 'four copied bytes'),
    ('global_size', [2,1,1], 'NDRange'), ('local_size', [2,1,1], 'NDRange'),
    ('global_offset', [1,0,0], 'NDRange')]:
    bad = copy.deepcopy(request); bad[field] = value
    check_request(manifest, bad, expected)
for field, value in [('kind', 'host_ptr'), ('arg_index', 1), ('handle', 0),
                     ('handle', 4294967296), ('logical_bytes', 8)]:
    bad = copy.deepcopy(request); bad['resources'][0][field] = value
    check_request(manifest, bad, 'resource handle/kind/order/size')
for field, value in [('width', 2), ('format', 'rfloat'), ('srgb', True), ('mipmaps', True)]:
    bad = copy.deepcopy(request); bad['resources'][0]['texture'][field] = value
    check_request(manifest, bad, 'texture layout/format/color-space')

# Cross-component failures are checked before any asset publication.
negative_sources = [
    (source.replace('module {', 'module attributes {vrc.abi_version = 2 : i32} {', 1), 'unsupported Host/Device ABI version'),
    (source.replace('vrc.public_name = "Add"', 'vrc.public_name = "1Bad"', 1), 'invalid UdonSharp public export name'),
    (source.replace('vrc.parameter_names = ["left", "right"]', 'vrc.parameter_names = ["left"]', 1), 'invalid UdonSharp parameter names'),
    (source.replace('module {', 'module attributes {vrc.target = "quest"} {', 1), 'initial boundary target'),
    (source.replace('vrc.kernel_id = 7', 'vrc.kernel_id = 8', 1), 'missing Device kernel id'),
    (source.replace('vrc.argument_kinds = ["buffer", "f32"]', 'vrc.argument_kinds = ["buffer", "i32"]'), 'schema does not match'),
    (source.replace('vrc.argument_kinds = ["buffer", "f32"]', 'vrc.argument_kinds = ["buffer", "buffer", "f32"]'), 'resource count'),
    (source.replace('vrc.global_size = array<i32: 1, 1, 1>', 'vrc.global_size = array<i32: 2, 1, 1>'), 'schema does not match'),
    (source.replace('vrc.global_offset = array<i32: 0, 0, 0>', 'vrc.global_offset = array<i32: 1, 0, 0>'), 'schema does not match'),
    (source.replace('known_grid_size = array<i32: 1, 1, 1>', 'known_grid_size = array<i32: 2, 1, 1>'), 'one-invocation proof'),
    (source.replace('memref<1xf32>', 'memref<2xf32>'), 'initial Device ABI'),
    (source.replace('memref<1xf32>', 'memref<1xf32, \"unsupported\">'), 'initial Device ABI'),
    (source.replace('      memref.store', '      gpu.barrier\n      memref.store'), 'GPU legality/barrier'),
    (source.replace('      memref.store', '      %read = memref.load %output[%index] : memref<1xf32>\n      memref.store'), 'GPU legality/barrier'),
    (source.replace('      memref.store %value, %output[%index] : memref<1xf32>', '      memref.store %value, %output[%index] : memref<1xf32>\n      memref.store %value, %output[%index] : memref<1xf32>'), 'one complete f32 store'),
    (source.replace('arith.constant 0 : index', 'arith.constant 1 : index'), 'one complete f32 store'),
    (source.replace('arith.mulf %x, %x : f32', 'arith.mulf %x, %x fastmath<fast> : f32'), 'without fast-math'),
    (source.replace('vrc.intrinsic = "dispatch"', 'vrc.intrinsic = "clEnqueueNDRangeKernel"'), 'recognized internal runtime intrinsic'),
    (source.replace('%result = arith.addi %a, %b : i32', '%result = func.call @add_impl(%a, %b) : (i32, i32) -> i32'), 'recursive call graph'),
    (source.replace('vrc.export_id = 1 : i32', 'vrc.export_id = "wrong"'), 'must be an integer'),
    (source.replace('vrc.export_id = 1 : i32', 'vrc.export_id = 1 : i32, vrc.async_effect = "may_suspend"'), 'continuation lowering'),
]
for i, (bad_source, expected) in enumerate(negative_sources):
    path = output / f'invalid-{i}.mlir'; path.write_text(bad_source)
    run([compiler, '--emit-boundary-bundle', path, '-o', bundle], error=expected)
    assert before == {str(p.relative_to(bundle)): p.read_bytes() for p in bundle.rglob('*') if p.is_file()}

# Verify final IR itself rejects version, binding and forbidden GPU semantics.
for bad, expected in [
    (lowered.replace('!vrcgpu.version<0>', '!vrcgpu.version<1>'), 'output/version'),
    (lowered.replace('binding = 1 : i32', 'binding = 0 : i32'), 'binding collision'),
    (lowered.replace('call_policy normal', 'call_policy may_suspend', 1), 'propagate to the caller'),
]:
    if expected: run([opt, '--verify-each'], bad, error=expected)

# A second arithmetic pattern exercises exact f32 constants, SSA mapping and
# no-contraction variables, while the Host export ABI remains stable.
variant = source.replace('%value = arith.mulf %x, %x : f32', '''%half = arith.constant 0.5 : f32
      %square = arith.mulf %x, %x : f32
      %value = arith.addf %square, %half : f32''')
variant_path = output / 'variant.mlir'; variant_path.write_text(variant)
run([compiler, '--emit-boundary-bundle', variant_path, '-o', bundle], error='different content')
variant_bundle = output / 'variant-bundle'
run([compiler, '--emit-boundary-bundle', variant_path, '-o', variant_bundle])
variant_manifest = json.loads((variant_bundle / 'manifest.json').read_text())
assert variant_manifest['exports'] == manifest['exports']
variant_hlsl = (variant_bundle / 'kernels/kernel_7.hlsl').read_text()
assert 'asfloat(0x3F000000u)' in variant_hlsl and ' + ' in variant_hlsl

validation = {'glslang': 'unavailable; ShaderLab/SDK/client validation still pending'}
glslang = shutil.which('glslangValidator')
if glslang:
    validation['glslang'] = run([glslang, '--version']).strip()
    for name, asset in [('square', bundle), ('square-plus-half', variant_bundle)]:
        binary = output / (name + '.test.spv')
        run([glslang, '-D', '-V', '-S', 'frag', '-e', 'frag', '--auto-map-bindings',
             asset / 'kernels/kernel_7.hlsl', '-o', binary])
        spirv_val = shutil.which('spirv-val')
        if spirv_val: run([spirv_val, binary])
    validation['standalone_hlsl'] = '2 fragment shaders compiled; SPIR-V is a test artifact only'
(output / 'report.json').write_text(json.dumps({
    'llvm_version': '23.1.4', 'host_device_contract': 'passed',
    'pattern': 'Host scalar call and typed dispatch + 1-invocation Device f32 output',
    'rejected_cases': len(rejections), 'shader_validation': validation,
    'udon_assembly_sdk_client_gpu_execution': 'not implemented or not executed',
}, indent=2) + '\n')
print(f'Host/Device stages, schema/payload checks and {len(rejections)} rejection cases passed')
