"""Exercise scalar SSA to real Udon opcodes and textual assembly emission."""
import pathlib
import subprocess
import sys

opt, translate, output = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
output.mkdir(parents=True, exist_ok=True)

source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @add(%a: i32, %b: i32) -> i32 attributes {
    vrc.export_id = 1 : i32,
    vrc.udon_event = "VRC_Call_add_1"
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
}
'''

lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'], input=source,
    text=True, capture_output=True)
assert lowered.returncode == 0, lowered.stderr
text = lowered.stdout
for expected in [
    'udonasm.stage = "scalar-v1"',
    'udonasm.push "VRC_ARG_1_0"',
    'udonasm.push "VRC_ARG_1_1"',
    'SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32',
    'SystemSingle.__op_Multiplication__SystemSingle_SystemSingle__SystemSingle',
    'SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean',
    'SystemSingle.__op_LessThan__SystemSingle_SystemSingle__SystemBoolean',
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
    '.export VRC_ARG_1_0',
    'VRC_ARG_1_0: %SystemInt32, null',
    '.export VRC_Call_add_1',
    'VRC_Call_add_1:',
    'EXTERN, "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"',
    '.export VRC_Call_scale_2',
    '.export VRC_Call_sum_to_3',
    '.export VRC_Call_phase_4',
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

invalid = source.replace('%sum = arith.addi %a, %b : i32', '%sum = arith.andi %a, %b : i32')
rejected = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm'], input=invalid, text=True,
    capture_output=True)
assert rejected.returncode != 0
assert 'extern legalization set' in rejected.stderr
print('scalar Udon heap allocation, opcode lowering and .uasm emission passed')
