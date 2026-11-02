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
