"""Exercise scalar SSA to real Udon opcodes and textual assembly emission."""
import pathlib
import subprocess
import sys

opt, translate, output = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
output.mkdir(parents=True, exist_ok=True)

source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @add(%a: i32, %b: i32) -> i32 attributes {
    vrc.export_id = 1 : i32
  } {
    %sum = arith.addi %a, %b : i32
    return %sum : i32
  }
  func.func @scale(%x: f32, %factor: f32) -> f32 attributes {
    vrc.export_id = 2 : i32
  } {
    %result = arith.mulf %x, %factor : f32
    return %result : f32
  }
  func.func @sum_to(%n: i32) -> i32 attributes {
    vrc.export_id = 3 : i32
  } {
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
  func.func @phase(%temperature: f32) -> i32 attributes {
    vrc.export_id = 4 : i32
  } {
    %ice_threshold = arith.constant 8.000000e-02 : f32
    %is_ice = arith.cmpf olt, %temperature, %ice_threshold : f32
    cf.cond_br %is_ice, ^ice, ^liquid
  ^ice:
    %ice = arith.constant 0 : i32
    return %ice : i32
  ^liquid:
    %liquid = arith.constant 2 : i32
    return %liquid : i32
  }
  func.func @array_sum(%input: memref<4xf32>, %output: memref<1xf32>) -> f32 attributes {
    vrc.export_id = 5 : i32
  } {
    %zero_index = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    cf.br ^loop(%zero_index, %zero : index, f32)
  ^loop(%index: index, %sum: f32):
    %limit = arith.constant 4 : index
    %condition = arith.cmpi slt, %index, %limit : index
    cf.cond_br %condition, ^body, ^exit(%sum : f32)
  ^body:
    %value = memref.load %input[%index] : memref<4xf32>
    %next_sum = arith.addf %sum, %value : f32
    %one = arith.constant 1 : index
    %next_index = arith.addi %index, %one : index
    cf.br ^loop(%next_index, %next_sum : index, f32)
  ^exit(%result: f32):
    memref.store %result, %output[%zero_index] : memref<1xf32>
    return %result : f32
  }
}
'''

lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'], input=source,
    text=True, capture_output=True)
assert lowered.returncode == 0, lowered.stderr
text = lowered.stdout
for expected in [
    'udonasm.stage = "scalar-v1"',
    'udonasm.push "__0_arg0__param"',
    'udonasm.push "__0_arg1__param"',
    'SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32',
    'SystemSingle.__op_Multiplication__SystemSingle_SystemSingle__SystemSingle',
    'SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean',
    'SystemSingle.__op_LessThan__SystemSingle_SystemSingle__SystemBoolean',
    'SystemSingleArray.__Get__SystemInt32__SystemSingle',
    'SystemSingleArray.__Set__SystemInt32_SystemSingle__SystemVoid',
    'udonasm.jump_if_false',
    'udonasm.jump 4294967292',
]:
    assert expected in text, (expected, text)
assert 'func.func' not in text and 'arith.' not in text

mlir = output / 'scalar.udonasm.mlir'
uasm = output / 'scalar.uasm'
mlir.write_text(text, encoding='utf-8')
emitted = subprocess.run(
    [translate, '--emit-udon-assembly', mlir, '-o', uasm], text=True,
    capture_output=True)
assert emitted.returncode == 0, emitted.stderr
assembly = uasm.read_text(encoding='utf-8')
for expected in [
    '.data_start',
    '__0_arg0__param: %SystemInt32, null',
    '.export __0_add',
    '__0_add:',
    'EXTERN, "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"',
    '.export __0_scale',
    '.export __0_sum_to',
    '.export __0_phase',
    '.export __0_array_sum',
    '__4_arg0__param: %SystemSingleArray, null',
    '__2_arg1__param: %SystemSingleArray, null',
    'JUMP, 0xFFFFFFFC',
    '.code_end',
]:
    assert expected in assembly, (expected, assembly)

# Unity's MCP-driven integration test executes this checked-in lowering output
# on the SDK IUdonVM. Keep it byte-for-token aligned with the compiler while
# allowing presentation-only blank-line differences.
fixture = pathlib.Path(__file__).parents[2] / 'unity/Packages/com.lemolatoon.udoncl/Tests/Editor/Fixtures/scalar.uasm'
normalize = lambda value: '\n'.join(line.rstrip() for line in value.splitlines() if line.strip())
assert fixture.exists(), fixture
assert normalize(fixture.read_text(encoding='utf-8')) == normalize(assembly), \
    'Unity IUdonVM fixture is stale; regenerate it from scalar.uasm'

call_source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @call_double(%value: i32) -> i32 attributes {
    vrc.export_id = 1 : i32
  } {
    %result = udonhl.call @double(%value) {call_policy = #udonhl<call_policy normal>} : (i32) -> i32
    return %result : i32
  }
  func.func private @double(%value: i32) -> i32 {
    %values = memref.alloca() : memref<2xi32>
    %zero = arith.constant 0 : index
    memref.store %value, %values[%zero] : memref<2xi32>
    %stored = memref.load %values[%zero] : memref<2xi32>
    %result = arith.addi %stored, %stored : i32
    return %result : i32
  }
}'''
call_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=call_source, text=True, capture_output=True)
assert call_lowered.returncode == 0, call_lowered.stderr
for expected in [
    'name = "__vrc_internal_1_return"',
    'type = "SystemUInt32"',
    'udonasm.jump_indirect "__vrc_internal_1_return"',
    'SystemInt32Array.__ctor__SystemInt32__SystemInt32Array',
    'SystemInt32Array.__Get__SystemInt32__SystemInt32',
    'SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid',
    'SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32',
]:
    assert expected in call_lowered.stdout, (expected, call_lowered.stdout)

global_source = r'''module {
  memref.global "private" @weights : memref<2xf32> = dense<[1.0, 2.0]>
  func.func @read_weight() -> f32 attributes {vrc.export_id = 1 : i32} {
    %weights = memref.get_global @weights : memref<2xf32>
    %index = arith.constant 1 : index
    %value = memref.load %weights[%index] : memref<2xf32>
    return %value : f32
  }
}'''
global_lowered = subprocess.run(
    [
        opt,
        '--pass-pipeline=builtin.module(vrc-lower-host-to-udonhl,'
        'vrc-lower-udonhl-to-udonasm)',
        '--verify-each',
    ],
    input=global_source,
    text=True,
    capture_output=True,
)
assert global_lowered.returncode == 0, global_lowered.stderr
for expected in [
    'name = "__vrc_global_0"',
    'name = "__vrc_globals_initialized"',
    'SystemSingleArray.__ctor__SystemInt32__SystemSingleArray',
    'SystemSingleArray.__Set__SystemInt32_SystemSingle__SystemVoid',
    'SystemSingleArray.__Get__SystemInt32__SystemSingle',
    'udonasm.jump_if_false',
]:
    assert expected in global_lowered.stdout, (expected, global_lowered.stdout)
assert 'memref.global' not in global_lowered.stdout
assert 'memref.get_global' not in global_lowered.stdout

raw_memory_source = r'''module {
  "udonhl.global"() <{sym_name = "handle", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
  func.func @raw_roundtrip() -> i32 attributes {vrc.export_id = 1 : i32} {
    %address = udonhl.address_of @handle : !udonhl.ptr<0>
    %value = arith.constant 305419896 : i32
    "udonhl.store"(%value, %address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %loaded = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    return %loaded : i32
  }
  func.func @raw_float() -> f32 attributes {vrc.export_id = 2 : i32} {
    %bytes = arith.constant 4 : i32
    %pointer = "udonhl.alloca"(%bytes) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %value = arith.constant 1.25 : f32
    "udonhl.store"(%value, %pointer) <{alignment = 4 : i64}> : (f32, !udonhl.ptr<0>) -> ()
    %loaded = "udonhl.load"(%pointer) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> f32
    return %loaded : f32
  }
  func.func @raw_memcpy() -> i32 attributes {vrc.export_id = 3 : i32} {
    %source_bytes = arith.constant 4 : i32
    %source = "udonhl.alloca"(%source_bytes) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %destination_bytes = arith.constant 4 : i32
    %destination = "udonhl.alloca"(%destination_bytes) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %value = arith.constant 270544960 : i32
    "udonhl.store"(%value, %source) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    "udonhl.memcpy"(%destination, %source, %source_bytes) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> ()
    %copied = "udonhl.load"(%destination) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    return %copied : i32
  }
}'''
raw_memory_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=raw_memory_source, text=True, capture_output=True)
assert raw_memory_lowered.returncode == 0, raw_memory_lowered.stderr
for expected in [
    'name = "__vrc_raw_heap"',
    'initial = "20", name = "__vrc_raw_heap_length"',
    'SystemInt32Array.__ctor__SystemInt32__SystemInt32Array',
    'SystemInt32Array.__Get__SystemInt32__SystemInt32',
    'SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid',
    'SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_BitwiseAnd__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_BitwiseOr__SystemInt32_SystemInt32__SystemInt32',
    'SystemBitConverter.__SingleToInt32Bits__SystemSingle__SystemInt32',
    'SystemBitConverter.__Int32BitsToSingle__SystemInt32__SystemSingle',
    'name = "__vrc_raw_memcpy_',
]:
    assert expected in raw_memory_lowered.stdout, (
        expected, raw_memory_lowered.stdout)
for forbidden in ['udonhl.global', 'udonhl.address_of', 'udonhl.alloca',
                  'udonhl.load', 'udonhl.store', 'udonhl.memcpy']:
    assert forbidden not in raw_memory_lowered.stdout, (
        forbidden, raw_memory_lowered.stdout)

dynamic_memcpy = raw_memory_source.replace(
    'func.func @raw_memcpy() -> i32',
    'func.func @raw_memcpy(%dynamic_size: i32) -> i32').replace(
        '"udonhl.memcpy"(%destination, %source, %source_bytes)',
        '"udonhl.memcpy"(%destination, %source, %dynamic_size)')
rejected = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm'], input=dynamic_memcpy,
    text=True, capture_output=True)
assert rejected.returncode == 0, rejected.stderr
assert 'udonhl.memcpy' not in rejected.stdout
assert 'SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean' \
    in rejected.stdout

raw_llvm_source = r'''module {
  llvm.mlir.global internal @counter(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64} : i32
  llvm.func @raw_export() -> i32 attributes {vrc.export_id = 1 : i32} {
    %address = llvm.mlir.addressof @counter : !llvm.ptr
    %value = llvm.mlir.constant(42 : i32) : i32
    llvm.store %value, %address {alignment = 4 : i64} : i32, !llvm.ptr
    %result = llvm.load %address {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %result : i32
  }
}'''
raw_llvm_lowered = subprocess.run(
    [
        opt,
        '--pass-pipeline=builtin.module(vrc-recover-high-level,'
        'vrc-lower-host-to-udonhl,vrc-lower-udonhl-to-udonasm)',
        '--verify-each',
    ],
    input=raw_llvm_source,
    text=True,
    capture_output=True,
)
assert raw_llvm_lowered.returncode == 0, raw_llvm_lowered.stderr
assert 'initial = "42"' in raw_llvm_lowered.stdout
assert 'SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid' in raw_llvm_lowered.stdout
for forbidden in ['llvm.mlir.global', 'llvm.mlir.addressof', 'llvm.load',
                  'llvm.store', 'udonhl.global', 'udonhl.load',
                  'udonhl.store']:
    assert forbidden not in raw_llvm_lowered.stdout, (
        forbidden, raw_llvm_lowered.stdout)

# Pin the exact cross-UdonBehaviour naming observed from UdonSharp in Worlds
# SDK 3.10.5. Logical export ids remain separate from this physical ABI.
abi_source = r'''module {
  func.func @c_add(%left: i32, %right: i32) -> i32 attributes {
    vrc.export_id = 1 : i32,
    vrc.public_name = "Add",
    vrc.parameter_names = ["left", "right"]
  } {
    %sum = arith.addi %left, %right : i32
    return %sum : i32
  }
  func.func @c_ref_out(%value: i32, %copy: i32) -> i32 attributes {
    vrc.export_id = 2 : i32,
    vrc.public_name = "RefOutCall",
    vrc.parameter_names = ["value", "copy"]
  } {
    return %value : i32
  }
}'''
abi_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm'], input=abi_source, text=True,
    capture_output=True)
assert abi_lowered.returncode == 0, abi_lowered.stderr
for expected in [
    'name = "__0_Add"',
    'name = "__0_RefOutCall"',
    'name = "__0_left__param"',
    'name = "__0_right__param"',
    'name = "__0___0_Add__ret"',
    'name = "__0_value__param"',
    'name = "__0_copy__param"',
    'name = "__0___0_RefOutCall__ret"',
]:
    assert expected in abi_lowered.stdout, (expected, abi_lowered.stdout)

duplicate_name = abi_source.replace('vrc.public_name = "RefOutCall"',
                                    'vrc.public_name = "Add"')
rejected = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm'], input=duplicate_name, text=True,
    capture_output=True)
assert rejected.returncode != 0
assert 'duplicate UdonSharp public export name' in rejected.stderr

mismatched_event = abi_source.replace(
    'vrc.parameter_names = ["left", "right"]',
    'vrc.parameter_names = ["left", "right"], vrc.udon_event = "Wrong"',
    1)
rejected = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm'], input=mismatched_event, text=True,
    capture_output=True)
assert rejected.returncode != 0
assert 'vrc.udon_event does not match the pinned UdonSharp ABI' in rejected.stderr

invalid = source.replace('%sum = arith.addi %a, %b : i32', '%sum = arith.andi %a, %b : i32')
rejected = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm'], input=invalid, text=True,
    capture_output=True)
assert rejected.returncode != 0
assert 'extern legalization set' in rejected.stderr

buffer_source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @copy_full() attributes {vrc.export_id = 6 : i32} {
    %bytes = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %source, %source_status = vrc.buffer_alloc %bytes {
      format = "rgba8_unorm", usage = "storage",
      vrc.unity_buffer_id = 1 : i32,
      vrc.unity_heap_slot = "__udoncl_buffer_1",
      vrc.unity_logical_bytes = 4 : i32
    } : (i32) -> (!vrc.buffer, i32)
    %destination, %destination_status = vrc.buffer_alloc %bytes {
      format = "rgba8_unorm", usage = "storage",
      vrc.unity_buffer_id = 2 : i32,
      vrc.unity_heap_slot = "__udoncl_buffer_2",
      vrc.unity_logical_bytes = 4 : i32
    } : (i32) -> (!vrc.buffer, i32)
    %event = vrc.buffer_copy %source, %destination, %zero, %zero, %bytes {
      vrc.unity_copy_material_slot = "__udoncl_buffer_copy_material"
    } : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    return
  }
}'''
buffer_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=buffer_source, text=True, capture_output=True)
assert buffer_lowered.returncode == 0, buffer_lowered.stderr
for expected in [
    'name = "__udoncl_buffer_1"',
    'name = "__udoncl_buffer_2"',
    'name = "__udoncl_buffer_copy_material"',
    'udonasm.push "__udoncl_buffer_1"',
    'udonasm.push "__udoncl_buffer_2"',
    'udonasm.push "__udoncl_buffer_copy_material"',
    'VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_'
    'UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid',
]:
    assert expected in buffer_lowered.stdout, (expected, buffer_lowered.stdout)

continuation_source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @resume(%execution: i32, %resume_state: i32, %event: !vrc.event, %value: i32) -> i32 attributes {
    vrc.async_effect = "may_suspend",
    vrc.execution_id_arg = 0 : i32,
    vrc.export_id = 6 : i32,
    vrc.resume_state_arg = 1 : i32
  } {
    %status = vrc.await %event : (!vrc.event) -> i32
    %sum = arith.addi %status, %value : i32
    return %sum : i32
  }
}'''
continuation_lowered = subprocess.run(
    [
        opt,
        '--pass-pipeline=builtin.module(vrc-lower-await-to-continuations,vrc-lower-udonhl-to-udonasm)',
        '--verify-each',
    ],
    input=continuation_source,
    text=True,
    capture_output=True,
)
assert continuation_lowered.returncode == 0, continuation_lowered.stderr
for expected in [
    'name = "__vrc_f6_frame_0"',
    'name = "__vrc_f6_frame_0_slot_0"',
    'name = "__vrc_f6_frame_0_slot_1"',
    'name = "__vrc_f6_resume_0"',
    'name = "__vrc_f6_resume_1"',
    'SystemInt32.__op_Equality__SystemInt32_SystemInt32__SystemBoolean',
    'udonasm.jump 4294967292',
]:
    assert expected in continuation_lowered.stdout, (
        expected,
        continuation_lowered.stdout,
    )
continuation_mlir = output / 'continuation.udonasm.mlir'
continuation_uasm = output / 'continuation.uasm'
continuation_mlir.write_text(continuation_lowered.stdout, encoding='utf-8')
emitted = subprocess.run(
    [translate, '--emit-udon-assembly', continuation_mlir, '-o', continuation_uasm],
    text=True,
    capture_output=True,
)
assert emitted.returncode == 0, emitted.stderr
continuation_assembly = continuation_uasm.read_text(encoding='utf-8')
assert '.export __0_resume' in continuation_assembly
assert '__vrc_f6_frame_0_slot_0: %SystemInt32, 0' in continuation_assembly
assert '__vrc_f6_frame_0_slot_1: %SystemInt32, 0' in continuation_assembly

unbound_buffer = buffer_source.replace(
    'vrc.unity_heap_slot = "__udoncl_buffer_1",', '', 1)
rejected = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm'], input=unbound_buffer,
    text=True, capture_output=True)
assert rejected.returncode != 0
assert 'requires a bound heap slot' in rejected.stderr

for argument_type, diagnostic in [
    ('memref<2x2xf32>', 'unsupported scalar Udon argument type'),
    ('memref<2x2xi32>', 'unsupported scalar Udon argument type'),
]:
    unsupported = f'''module {{
      func.func @bad(%array: {argument_type}) attributes {{vrc.export_id = 99 : i32}} {{
        return
      }}
    }}'''
    rejected = subprocess.run(
        [opt, '--vrc-lower-udonhl-to-udonasm'], input=unsupported,
        text=True, capture_output=True)
    assert rejected.returncode != 0
    assert diagnostic in rejected.stderr, rejected.stderr
print('scalar Udon heap allocation, opcode lowering and .uasm emission passed')
