"""Lower Host runtime intrinsics, Host control flow and Unity buffer bundles.

Each case is a self-contained Host module: the runtime and OpenCL AOT
intrinsics a recovered Host calls, the cf control flow, internal values and
continuations recovered C keeps on the Host path, the event intrinsics, and
Unity library bundles of typed Host MLIR that copy buffers or read a static
tensor table.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys

opt, compiler, output = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
# These are disposable test-owned bundle directories, not user assets.
for directory in [
        'unity-buffer-copy', 'unity-buffer-range-copy', 'unity-static-tensor']:
    if (output / directory).exists():
        shutil.rmtree(output / directory)
root = Path(__file__).resolve().parents[2]
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
  func.func private @__vrc_opencl_dispatch_udonaten_argmax_reduce(i32, i32, i32, i32) -> i32
  func.func private @__vrc_opencl_dispatch_udonaten_argmax_finalize(i32, i32, i32, i32, i32, i32) -> i32
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
    %argmax_reduce = func.call @__vrc_opencl_dispatch_udonaten_argmax_reduce(%input, %output, %zero, %bytes) : (i32, i32, i32, i32) -> i32
    %argmax_finalize = func.call @__vrc_opencl_dispatch_udonaten_argmax_finalize(%input, %output, %static, %zero, %bytes, %one) : (i32, i32, i32, i32, i32, i32) -> i32
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
assert 'kernel_id = 247 : i32' in opencl_aot_ops
assert 'kernel_id = 248 : i32' in opencl_aot_ops
assert opencl_aot_ops.count('global_size = array<i32: 512, 1, 1>') == 1
assert opencl_aot_ops.count('global_size = array<i32: 1, 1, 1>') >= 1
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
copy_host = root / 'tests/boundary/unity_buffer_copy.mlir'
# The Host MLIR names its program, export and assigned buffers; no config.
run([
    compiler, '--emit-unity-library-bundle',
    '--host-mlir', copy_host, '-o', copy_artifact,
])
copy_manifest = json.loads((copy_artifact / 'manifest.json').read_text())
assert copy_manifest['artifact_stage'] == 'unity-udoncl-library'
assert copy_manifest['program']['buffer_copy_shader'] == 'buffer-copy.shader'
assert copy_manifest['program']['buffer_copy_program_slot'] is True
assert copy_manifest['program']['buffer_pool_slot'] == '__udoncl_buffer_pool'
assert [buffer['logical_bytes'] for buffer in copy_manifest['device']['buffers']] == [4, 4]
copy_assembly = (copy_artifact / 'host.uasm').read_text()
for expected in [
    '.export CopyFull',
    '.export __udoncl_buffer_pool',
    '__udoncl_buffer_pool: %UnityEngineRenderTextureArray, null',
    'SystemObjectArray.__Get__SystemInt32__SystemObject',
    '.export __udoncl_buffer_copy_material',
    'VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_'
    'UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid',
]:
    assert expected in copy_assembly, (expected, copy_assembly)
assert (copy_artifact / 'buffer-copy.shader').is_file()
assert '_VRC_TransferDelta' not in copy_assembly

# A copy of part of a buffer, or between differently shaped buffers, is a
# word-range Blit through the same transfer material.
range_copy_artifact = output / 'unity-buffer-range-copy'
run([
    compiler, '--emit-unity-library-bundle',
    '--host-mlir', root / 'tests/boundary/unity_buffer_range_copy.mlir',
    '-o', range_copy_artifact,
])
range_copy_manifest = json.loads(
    (range_copy_artifact / 'manifest.json').read_text())
assert range_copy_manifest['program']['buffer_copy_program_slot'] is True
range_copy_assembly = (range_copy_artifact / 'host.uasm').read_text()
for expected in [
    '.export CopyWord',
    '"_VRC_TransferMode"',
    '"_VRC_TransferFirst"',
    '"_VRC_TransferCount"',
    '"_VRC_TransferDelta"',
    'UnityEngineRenderTexture.__get_width__SystemInt32',
    'VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_'
    'UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid',
]:
    assert expected in range_copy_assembly, (expected, range_copy_assembly)

static_artifact = output / 'unity-static-tensor'
run([
    compiler, '--emit-unity-library-bundle',
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
print(f'Host runtime lowering and Unity buffer bundles passed with '
      f'{len(rejections)} rejection cases')
