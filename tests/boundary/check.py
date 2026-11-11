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
for directory in [
        'bundle', 'variant-bundle', 'unity-buffer-copy',
        'unity-static-tensor']:
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
opencl_aot_pass = 'vrc-lower-opencl-aot-intrinsics'
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
  func.func private @allocate(i32, i32) -> (!vrc.buffer, i32) attributes {vrc.intrinsic = "buffer_alloc", vrc.format = "rgba8_unorm", vrc.usage = "storage"}
  func.func private @release(!vrc.buffer) -> i32 attributes {vrc.intrinsic = "buffer_release"}
  func.func private @copy(!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event attributes {vrc.intrinsic = "buffer_copy"}
  func.func private @upload(!vrc.buffer, i32, i32, i32) -> !vrc.event attributes {vrc.intrinsic = "buffer_upload"}
  func.func private @readback(!vrc.buffer, i32, i32, i32) -> !vrc.event attributes {vrc.intrinsic = "buffer_readback"}
  func.func @runtime(%bytes: i32, %source: !vrc.buffer, %zero: i32, %pointer: i32) -> !vrc.event {
    %destination, %status = udonhl.call @allocate(%zero, %bytes) {call_policy = #udonhl<call_policy normal>} : (i32, i32) -> (!vrc.buffer, i32)
    %copied = udonhl.call @copy(%source, %destination, %zero, %zero, %bytes) {call_policy = #udonhl<call_policy normal>} : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    %uploaded = udonhl.call @upload(%destination, %zero, %bytes, %pointer) {call_policy = #udonhl<call_policy normal>} : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %read = udonhl.call @readback(%destination, %zero, %bytes, %pointer) {call_policy = #udonhl<call_policy normal>} : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %released = udonhl.call @release(%destination) {call_policy = #udonhl<call_policy normal>} : (!vrc.buffer) -> i32
    return %read : !vrc.event
  }
}
'''
buffer_ops = run([opt, '--pass-pipeline=builtin.module(' + intrinsic_pass + ')',
                  '--verify-each'], buffer_runtime)
for operation in ['vrc.buffer_alloc ', 'vrc.buffer_release ', 'vrc.buffer_copy ',
                  'vrc.buffer_upload ', 'vrc.buffer_readback ']:
    assert operation in buffer_ops
for prototype in ['@allocate', '@release', '@copy', '@upload', '@readback']:
    assert prototype not in buffer_ops
typed_buffer_source = buffer_runtime.replace(
    'module attributes {vrc.host_stage = "udonhl"}', 'module').replace(
    'udonhl.call', 'func.call').replace(
    ' {call_policy = #udonhl<call_policy normal>}', '')
typed_buffer_ops = run(
    [opt, '--pass-pipeline=builtin.module(' + host_pass + ',' + intrinsic_pass + ')',
     '--verify-each'], typed_buffer_source)
assert 'vrc.buffer_alloc ' in typed_buffer_ops
assert 'vrc.buffer_release ' in typed_buffer_ops
assert 'vrc.buffer_copy ' in typed_buffer_ops
assert 'vrc.buffer_upload ' in typed_buffer_ops
assert 'vrc.buffer_readback ' in typed_buffer_ops
assert 'func.call ' not in typed_buffer_ops and 'udonhl.call ' not in typed_buffer_ops

opencl_aot_runtime = r'''
module {
  func.func private @__vrc_opencl_static_tensor(i32, i32) -> i32
  func.func private @__vrc_opencl_buffer_alloc(i32, i32) -> i32
  func.func private @__vrc_opencl_buffer_release(i32) -> i32
  func.func private @__vrc_opencl_buffer_upload(i32, i32, i32, i32) -> i32
  func.func private @__vrc_opencl_buffer_readback(i32, i32, i32, i32) -> i32
  func.func private @__vrc_opencl_await(i32) -> i32
  func.func private @__vrc_opencl_dispatch_udonblas_sgemm(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32) -> i32
  func.func private @__vrc_opencl_dispatch_udonblas_arena_sgemm(i32, i32, i32, i32, i32, i32, i32, i32, f32) -> i32
  func.func private @__vrc_opencl_dispatch_udondnn_large_activation(i32, i32, i32, i32, i32, i32, f32, f32, f32) -> i32
  func.func private @__vrc_opencl_dispatch_udondnn_large_op_tensor(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32) -> i32
  func.func private @blocking_transfer(%token: i32, %offset: i32, %bytes: i32, %pointer: i32) -> i32 {
    %event = func.call @__vrc_opencl_buffer_readback(%token, %offset, %bytes, %pointer) : (i32, i32, i32, i32) -> i32
    %status = func.call @__vrc_opencl_await(%event) : (i32) -> i32
    return %status : i32
  }
  func.func @run(%bytes: i32, %zero: i32, %pointer: i32, %one: i32, %scalar: f32) -> i32 {
    %static_id = arith.constant 0 : i32
    %static_bytes = arith.constant 4 : i32
    %static = func.call @__vrc_opencl_static_tensor(%static_id, %static_bytes) : (i32, i32) -> i32
    %input = func.call @__vrc_opencl_buffer_alloc(%one, %bytes) : (i32, i32) -> i32
    %output = func.call @__vrc_opencl_buffer_alloc(%one, %bytes) : (i32, i32) -> i32
    %upload = func.call @__vrc_opencl_buffer_upload(%input, %zero, %bytes, %pointer) : (i32, i32, i32, i32) -> i32
    %dispatch = func.call @__vrc_opencl_dispatch_udonblas_sgemm(%input, %output, %one, %one, %one, %zero, %zero, %one, %one, %one, %scalar, %scalar) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32) -> i32
    %arena = func.call @__vrc_opencl_dispatch_udonblas_arena_sgemm(%static, %input, %output, %one, %one, %one, %zero, %zero, %scalar) : (i32, i32, i32, i32, i32, i32, i32, i32, f32) -> i32
    %activation = func.call @__vrc_opencl_dispatch_udondnn_large_activation(%input, %input, %output, %bytes, %zero, %one, %scalar, %scalar, %scalar) : (i32, i32, i32, i32, i32, i32, f32, f32, f32) -> i32
    %pointwise = func.call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%input, %static, %input, %output, %bytes, %one, %one, %one, %one, %one, %one, %zero, %scalar, %scalar, %scalar) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32) -> i32
    %status = func.call @blocking_transfer(%output, %zero, %bytes, %pointer) : (i32, i32, i32, i32) -> i32
    %released = func.call @__vrc_opencl_buffer_release(%output) : (i32) -> i32
    return %status : i32
  }
}
'''
opencl_aot_ops = run(
    [opt, '--pass-pipeline=builtin.module(' + opencl_aot_pass + ')',
     '--verify-each'], opencl_aot_runtime)
for operation in ['vrc.static_buffer', 'vrc.buffer_alloc ', 'vrc.buffer_release ',
                  'vrc.buffer_upload ', 'vrc.buffer_readback ',
                  'vrc.dispatch_desc_create ', 'vrc.dispatch ', 'vrc.await ']:
    assert operation in opencl_aot_ops
assert 'vrc.async_effect = "may_suspend"' in opencl_aot_ops
assert '@__vrc_opencl_' not in opencl_aot_ops
assert 'scalar_kinds = ["i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32"]' in opencl_aot_ops
assert 'kernel_id = 221 : i32' in opencl_aot_ops
assert opencl_aot_ops.count('global_size = array<i32: 8192, 4, 1>') == 1
assert opencl_aot_ops.count('global_size = array<i32: 8192, 16, 1>') == 1
assert 'kernel_id = 233 : i32' in opencl_aot_ops
assert 'kernel_id = 234 : i32' in opencl_aot_ops
assert 'resource_accesses = ["read", "read", "write"]' in opencl_aot_ops
assert 'resource_accesses = ["read", "read", "read", "write"]' in opencl_aot_ops

# Recovered C control flow remains in cf form on the Host path. Keep the public
# ABI narrow while allowing internal booleans, index values, block arguments,
# loops and branches to reach the existing Udon jump lowering.
control_flow_source = r'''
module {
  func.func @sum_to(%n: i32) -> i32 attributes {vrc.export_id = 3 : i32} {
    %zero = arith.constant 0 : i32
    cf.br ^loop(%zero, %zero : i32, i32)
  ^loop(%index: i32, %sum: i32):
    %condition = arith.cmpi slt, %index, %n : i32
    cf.cond_br %condition, ^body, ^exit(%sum : i32)
  ^body:
    %next_sum = arith.addi %sum, %index : i32
    %one = arith.constant 1 : i32
    %next_index = arith.addi %index, %one : i32
    cf.br ^loop(%next_index, %next_sum : i32, i32)
  ^exit(%result: i32):
    return %result : i32
  }
}
'''
control_flow_host = run([
    opt,
    '--pass-pipeline=builtin.module(' + host_pass + ')',
    '--verify-each',
], control_flow_source)
assert 'cf.cond_br ' in control_flow_host
control_flow_udon = run([
    opt,
    '--pass-pipeline=builtin.module(' + host_pass +
    ',vrc-lower-udonhl-to-udonasm)',
    '--verify-each',
], control_flow_source)
assert 'udonasm.jump_if_false' in control_flow_udon
assert 'SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean' in control_flow_udon

structured_control_flow_source = r'''
module {
  func.func @choose(%condition_value: i32, %value: i32) -> i32 attributes {vrc.export_id = 5 : i32} {
    %zero = arith.constant 0 : i32
    %condition = arith.cmpi ne, %condition_value, %zero : i32
    %poison = ub.poison : i32
    %result = scf.if %condition -> (i32) {
      scf.yield %value : i32
    } else {
      scf.yield %poison : i32
    }
    return %result : i32
  }
}
'''
structured_host = run([
    opt,
    '--pass-pipeline=builtin.module(' + host_pass + ')',
    '--verify-each',
], structured_control_flow_source)
assert 'scf.' not in structured_host and 'ub.poison' not in structured_host
assert 'cf.cond_br ' in structured_host
structured_udon = run([
    opt,
    '--pass-pipeline=builtin.module(' + host_pass +
    ',vrc-lower-udonhl-to-udonasm)',
    '--verify-each',
], structured_control_flow_source)
assert 'udonasm.jump_if_false' in structured_udon

internal_i64_source = r'''
module {
  func.func private @accept_flags(%flags: i64) -> i32 {
    %one = arith.constant 1 : i32
    return %one : i32
  }
  func.func @use_flags() -> i32 attributes {vrc.export_id = 6 : i32} {
    %flags = arith.constant 5 : i64
    %result = func.call @accept_flags(%flags) : (i64) -> i32
    return %result : i32
  }
}
'''
internal_i64_udon = run([
    opt,
    '--pass-pipeline=builtin.module(' + host_pass +
    ',vrc-lower-udonhl-to-udonasm)',
    '--verify-each',
], internal_i64_source)
assert 'type = "SystemInt64"' in internal_i64_udon

public_boolean = r'''
module {
  func.func @identity(%value: i1) -> i32 attributes {vrc.export_id = 4 : i32} {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
}
'''
run([
    opt,
    '--pass-pipeline=builtin.module(' + host_pass + ')',
], public_boolean, error='export id/signature')

continuation_source = r'''
module {
  func.func @resume(%execution: i32, %resume_state: i32, %event: !vrc.event, %value: i32) -> i32 attributes {vrc.async_effect = "may_suspend", vrc.execution_id_arg = 0 : i32, vrc.resume_state_arg = 1 : i32} {
    %status = vrc.await %event : (!vrc.event) -> i32
    %sum = arith.addi %status, %value : i32
    return %sum : i32
  }
}
'''
continuation_pass = 'vrc-lower-await-to-continuations'
continuation = run(
    [opt, '--pass-pipeline=builtin.module(' + continuation_pass + ')',
     '--verify-each'], continuation_source)
assert 'vrc.await ' not in continuation
for operation in ['udonhl.frame_create ', 'udonhl.frame_store ',
                  'udonhl.frame_load ', '"udonhl.resume_dispatch"',
                  '"udonhl.suspend"']:
    assert operation in continuation
assert 'vrc.continuation_stage = "udonhl-frame"' in continuation
assert run([opt, '--verify-each'], continuation) == continuation

nonpersistable_continuation = r'''
module {
  func.func @resume(%execution: i32, %resume_state: i32, %event: !vrc.event, %values: memref<?xi32>) -> i32 attributes {vrc.async_effect = "may_suspend", vrc.execution_id_arg = 0 : i32, vrc.resume_state_arg = 1 : i32} {
    %status = vrc.await %event : (!vrc.event) -> i32
    %zero = arith.constant 0 : index
    %value = memref.load %values[%zero] : memref<?xi32>
    %sum = arith.addi %status, %value : i32
    return %sum : i32
  }
}
'''
run([opt, '--pass-pipeline=builtin.module(' + continuation_pass + ')'],
    nonpersistable_continuation, error='not yet persistable')

event_runtime = r'''
module attributes {vrc.host_stage = "udonhl"} {
  func.func private @event_status(!vrc.event) -> i32 attributes {vrc.intrinsic = "event_status"}
  func.func private @await(!vrc.event) -> i32 attributes {vrc.intrinsic = "await", vrc.async_effect = "may_suspend"}
  func.func @runtime(%event: !vrc.event) -> i32 attributes {vrc.async_effect = "may_suspend"} {
    %current = udonhl.call @event_status(%event) {call_policy = #udonhl<call_policy normal>} : (!vrc.event) -> i32
    %completed = udonhl.call @await(%event) {call_policy = #udonhl<call_policy may_suspend>} : (!vrc.event) -> i32
    return %completed : i32
  }
}
'''
event_ops = run([
    opt, '--pass-pipeline=builtin.module(' + intrinsic_pass + ')',
    '--verify-each'
], event_runtime)
assert 'vrc.event_status ' in event_ops and 'vrc.await ' in event_ops
assert '@event_status' not in event_ops and '@await' not in event_ops
typed_event_source = event_runtime.replace(
    'module attributes {vrc.host_stage = "udonhl"}', 'module'
).replace(
    'udonhl.call', 'func.call'
).replace(
    ' {call_policy = #udonhl<call_policy normal>}', ''
).replace(
    ' {call_policy = #udonhl<call_policy may_suspend>}', ''
)
typed_event_ops = run([
    opt, '--pass-pipeline=builtin.module(' + host_pass + ',' + intrinsic_pass + ')',
    '--verify-each'
], typed_event_source)
assert 'vrc.event_status ' in typed_event_ops and 'vrc.await ' in typed_event_ops
assert 'func.call ' not in typed_event_ops and 'udonhl.call ' not in typed_event_ops
run([
    opt, '--pass-pipeline=builtin.module(' + intrinsic_pass + ')'
], event_runtime.replace(
    '#udonhl<call_policy may_suspend>', '#udonhl<call_policy normal>'
), error='may_suspend callee requires may_suspend call policy')

copy_artifact = output / 'unity-buffer-copy'
copy_config = root / 'tests/boundary/unity_buffer_copy.json'
copy_host = root / 'tests/boundary/unity_buffer_copy.mlir'
run([
    compiler, '--emit-unity-library-bundle', copy_config,
    '--host-mlir', copy_host, '-o', copy_artifact,
])
copy_manifest = json.loads((copy_artifact / 'manifest.json').read_text())
assert copy_manifest['artifact_stage'] == 'unity-udoncl-library'
assert copy_manifest['program']['buffer_copy_shader'] == 'buffer-copy.shader'
assert copy_manifest['program']['buffer_copy_program_slot'] is True
assert [buffer['logical_bytes'] for buffer in copy_manifest['device']['buffers']] == [4, 4]
copy_assembly = (copy_artifact / 'host.uasm').read_text()
for expected in [
    '.export CopyFull',
    '.export __udoncl_buffer_1',
    '.export __udoncl_buffer_2',
    '.export __udoncl_buffer_copy_material',
    'VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_'
    'UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid',
]:
    assert expected in copy_assembly, (expected, copy_assembly)
assert (copy_artifact / 'buffer-copy.shader').is_file()

static_artifact = output / 'unity-static-tensor'
run([
    compiler, '--emit-unity-library-bundle',
    root / 'tests/boundary/static_tensor_table.json',
    '--host-mlir', root / 'tests/boundary/static_tensor_table.mlir',
    '-o', static_artifact,
])
static_manifest = json.loads((static_artifact / 'manifest.json').read_text())
assert static_manifest['program']['static_tensor_table_slot'] == (
    '__udoncl_static_tensors')
assert static_manifest['static_tensors'][0]['name'] == 'weight'
assert (static_artifact / 'static-tensors/tensor_0.bytes').read_bytes() == (
    b'\x00\x00\x80\x3f')
static_assembly = (static_artifact / 'host.uasm').read_text()
assert '.export __udoncl_static_tensors' in static_assembly
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
array_export = manifest['exports'][1]
assert array_export['public_name'] == 'TouchArrays'
assert [parameter['kind'] for parameter in array_export['parameters']] == [
    'f32_array', 'i32_array'
]
assert array_export['parameters'][0]['type_schema'] == {
    'category': 'array', 'element_category': 'float', 'element_bits': 32,
    'rank': 1, 'elements': 4,
}
assert array_export['parameters'][1]['type_schema'] == {
    'category': 'array', 'element_category': 'signed-integer',
    'element_bits': 32, 'rank': 1, 'elements': 2,
}
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
