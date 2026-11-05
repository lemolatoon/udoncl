"""Exercise scalar SSA to real Udon opcodes and textual assembly emission."""
import pathlib
import re
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

remainder_source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @remainder(%x: i32, %divisor: i32) -> i32 attributes {
    vrc.export_id = 1 : i32
  } {
    %result = arith.remsi %x, %divisor : i32
    return %result : i32
  }
}'''
remainder = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=remainder_source, text=True, capture_output=True)
assert remainder.returncode == 0, remainder.stderr
for expected in [
    'SystemInt32.__op_Division__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_Multiplication__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_Subtraction__SystemInt32_SystemInt32__SystemInt32',
]:
    assert expected in remainder.stdout, (expected, remainder.stdout)
assert '__op_Modulus__' not in remainder.stdout

boolean_and_source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @both(%lhs: i1, %rhs: i1) -> i1 attributes {
    vrc.export_id = 1 : i32
  } {
    %result = arith.andi %lhs, %rhs : i1
    return %result : i1
  }
}'''
boolean_and = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=boolean_and_source, text=True, capture_output=True)
assert boolean_and.returncode == 0, boolean_and.stderr
assert 'udonasm.jump_if_false' in boolean_and.stdout
assert (
    'SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__SystemInt32'
    not in boolean_and.stdout
)

conversion_source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @to_float(%value: i32) -> f32 attributes {
    vrc.export_id = 1 : i32
  } {
    %result = arith.sitofp %value : i32 to f32
    return %result : f32
  }
}'''
conversion = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=conversion_source, text=True, capture_output=True)
assert conversion.returncode == 0, conversion.stderr
assert (
    'SystemConvert.__ToSingle__SystemInt32__SystemSingle'
    in conversion.stdout
)

unsigned_conversion_source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @to_float(%value: i32) -> f32 attributes {
    vrc.export_id = 1 : i32
  } {
    %result = arith.uitofp %value : i32 to f32
    return %result : f32
  }
}'''
unsigned_conversion = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=unsigned_conversion_source, text=True, capture_output=True)
assert unsigned_conversion.returncode == 0, unsigned_conversion.stderr
assert (
    'SystemConvert.__ToUInt32__SystemInt64__SystemUInt32'
    in unsigned_conversion.stdout
)
assert (
    'SystemConvert.__ToSingle__SystemUInt32__SystemSingle'
    in unsigned_conversion.stdout
)

large_float_source = r'''module attributes {
  vrc.async_stage = "udon-static-v1",
  vrc.host_stage = "udonhl"
} {
  func.func @large_float() -> f32 attributes {
    vrc.export_id = 1 : i32
  } {
    %value = arith.constant 3.40282347E+38 : f32
    return %value : f32
  }
}'''
large_float = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=large_float_source, text=True, capture_output=True)
assert large_float.returncode == 0, large_float.stderr
assert '340282346638528859811704183484516925440.0' not in large_float.stdout
assert (
    'SystemBitConverter.__Int32BitsToSingle__SystemInt32__SystemSingle'
    in large_float.stdout
)

int_min_source = r'''module attributes {
  vrc.async_stage = "udon-static-v1",
  vrc.host_stage = "udonhl"
} {
  func.func @int_min() -> i32 attributes {
    vrc.export_id = 1 : i32
  } {
    %value = arith.constant -2147483648 : i32
    return %value : i32
  }
}'''
int_min = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=int_min_source, text=True, capture_output=True)
assert int_min.returncode == 0, int_min.stderr
assert ': %SystemInt32, -2147483648' not in int_min.stdout

zero_extension_source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @zero_extend(%byte: i8, %word: i16) -> i32 attributes {
    vrc.export_id = 1 : i32
  } {
    %byte32 = arith.extui %byte : i8 to i32
    %word32 = arith.extui %word : i16 to i32
    %result = arith.addi %byte32, %word32 : i32
    return %result : i32
  }
}'''
zero_extension = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=zero_extension_source, text=True, capture_output=True)
assert zero_extension.returncode == 0, zero_extension.stderr
assert 'arith.extui' not in zero_extension.stdout

wide_zero_extension_source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @wide_zero_extend(%value: i32) -> i64 attributes {
    vrc.export_id = 1 : i32
  } {
    %result = arith.extui %value : i32 to i64
    return %result : i64
  }
}'''
wide_zero_extension = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=wide_zero_extension_source, text=True, capture_output=True)
assert wide_zero_extension.returncode == 0, wide_zero_extension.stderr
for expected in [
    'SystemConvert.__ToInt64__SystemInt32__SystemInt64',
    'SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean',
    'SystemInt64.__op_Addition__SystemInt64_SystemInt64__SystemInt64',
    'SystemInt64.__op_LeftShift__SystemInt64_SystemInt32__SystemInt64',
    'udonasm.jump_if_false',
]:
    assert expected in wide_zero_extension.stdout, (
        expected, wide_zero_extension.stdout)
assert 'initial = "4294967296"' not in wide_zero_extension.stdout

integer_bit_source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @integer_bits(%value: i32, %shift: i32, %mask: i32) -> i32 attributes {
    vrc.export_id = 1 : i32
  } {
    %left = arith.shli %value, %shift : i32
    %right = arith.shrsi %value, %shift : i32
    %logical = arith.shrui %value, %shift : i32
    %combined = arith.ori %left, %right : i32
    %either = arith.ori %combined, %logical : i32
    %result = arith.xori %either, %mask : i32
    return %result : i32
  }
}'''
integer_bits = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=integer_bit_source, text=True, capture_output=True)
assert integer_bits.returncode == 0, integer_bits.stderr
for expected in [
    'SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_LogicalXor__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__SystemInt32',
]:
    assert expected in integer_bits.stdout, (expected, integer_bits.stdout)

unordered_not_equal_source = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @is_nan_or_different(%lhs: f32, %rhs: f32) -> i1 attributes {
    vrc.export_id = 1 : i32
  } {
    %result = arith.cmpf une, %lhs, %rhs : f32
    return %result : i1
  }
}'''
unordered_not_equal = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=unordered_not_equal_source, text=True, capture_output=True)
assert unordered_not_equal.returncode == 0, unordered_not_equal.stderr
assert (
    'SystemSingle.__op_Inequality__SystemSingle_SystemSingle__SystemBoolean'
    in unordered_not_equal.stdout
)

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
    'name = "__vrc_internal_bank_0_return"',
    'type = "SystemUInt32"',
    'udonasm.jump_indirect "__vrc_internal_bank_0_return"',
    'SystemInt32Array.__ctor__SystemInt32__SystemInt32Array',
    'SystemInt32Array.__Get__SystemInt32__SystemInt32',
    'SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid',
    'SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32',
]:
    assert expected in call_lowered.stdout, (expected, call_lowered.stdout)

slot_bank_source = r'''module {
  func.func @run() -> i32 attributes {
    vrc.export_id = 1 : i32,
    vrc.parameter_names = [],
    vrc.public_name = "Run"
  } {
    %left = func.call @left() : () -> i32
    %right = func.call @right() : () -> i32
    %sum = arith.addi %left, %right : i32
    return %sum : i32
  }
  func.func private @left() -> i32 {
    %values = memref.alloca() : memref<2xi32>
    %zero = arith.constant 0 : index
    %value = arith.constant 2 : i32
    memref.store %value, %values[%zero] : memref<2xi32>
    %result = memref.load %values[%zero] : memref<2xi32>
    return %result : i32
  }
  func.func private @right() -> i32 {
    %values = memref.alloca() : memref<3xi32>
    %zero = arith.constant 0 : index
    %value = arith.constant 3 : i32
    memref.store %value, %values[%zero] : memref<3xi32>
    %result = memref.load %values[%zero] : memref<3xi32>
    return %result : i32
  }
}'''
slot_bank_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=slot_bank_source,
    text=True,
    capture_output=True,
)
assert slot_bank_lowered.returncode == 0, slot_bank_lowered.stderr
for expected in [
    'name = "__vrc_shared_r_SystemInt32Array_n2_0"',
    'name = "__vrc_shared_r_SystemInt32Array_n3_0"',
]:
    assert expected in slot_bank_lowered.stdout, (
        expected,
        slot_bank_lowered.stdout,
    )
assert '__vrc_shared_r_SystemInt32Array_n2_1' not in slot_bank_lowered.stdout

boolean_comparison_source = r'''module {
  func.func @compare(%left: i1, %right: i1) -> i1 attributes {
    vrc.export_id = 7 : i32,
    vrc.parameter_names = ["left", "right"],
    vrc.public_name = "Compare"
  } {
    %different = arith.cmpi ne, %left, %right : i1
    %same = arith.cmpi eq, %left, %right : i1
    %result = arith.andi %different, %same : i1
    return %result : i1
  }
}'''
boolean_comparison_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=boolean_comparison_source,
    text=True,
    capture_output=True,
)
assert boolean_comparison_lowered.returncode == 0, (
    boolean_comparison_lowered.stderr
)
assert 'SystemInt32.__op_Inequality__' not in (
    boolean_comparison_lowered.stdout
)
assert boolean_comparison_lowered.stdout.count('udonasm.jump_if_false') >= 5

parallel_call_source = r'''module attributes {vrc.async_stage = "udon-static-v1"} {
  func.func @run(%left: i32, %right: i32) -> i32 attributes {
    vrc.export_id = 2 : i32,
    vrc.parameter_names = ["left", "right"],
    vrc.public_name = "RunParallel"
  } {
    %zero = arith.constant 0 : i32
    %left_copy = arith.addi %left, %zero : i32
    %right_copy = arith.addi %right, %zero : i32
    %result = func.call @select_right(%left_copy, %right_copy) : (i32, i32) -> i32
    return %result : i32
  }
  func.func private @select_right(%left: i32, %right: i32) -> i32 {
    return %right : i32
  }
}'''
parallel_call_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=parallel_call_source,
    text=True,
    capture_output=True,
)
assert parallel_call_lowered.returncode == 0, parallel_call_lowered.stderr
for expected in [
    'name = "__vrc_shared_r_SystemInt32_0"',
    'name = "__vrc_shared_r_SystemInt32_1"',
    'name = "__vrc_shared_r_SystemInt32_2"',
    'name = "__vrc_shared_r_SystemInt32_3"',
]:
    assert expected in parallel_call_lowered.stdout, (
        expected,
        parallel_call_lowered.stdout,
    )
for fixed in ['__0_left__param', '__0_right__param']:
    assert re.search(
        r'fixed_named_slots = \[[^]]*name = "' + fixed,
        parallel_call_lowered.stdout,
    ), parallel_call_lowered.stdout
for scalar in [
    '__vrc_shared_r_SystemInt32_0',
    '__vrc_shared_r_SystemInt32_1',
]:
    assert re.search(
        r'scalar_registers = \[[^]]*name = "' + scalar,
        parallel_call_lowered.stdout,
    ), parallel_call_lowered.stdout

parallel_branch_source = r'''module attributes {
  vrc.async_stage = "udon-static-v1",
  vrc.host_stage = "udonhl",
  vrc.scalar_register_budget = 1 : i32
} {
  func.func @swap(%left: i32, %right: i32) -> i32 attributes {
    vrc.export_id = 9 : i32
  } {
    %zero = arith.constant 0 : i32
    %left_copy = arith.addi %left, %zero : i32
    %right_copy = arith.addi %right, %zero : i32
    cf.br ^bb1(%right_copy, %left_copy : i32, i32)
  ^bb1(%first: i32, %second: i32):
    %result = arith.subi %first, %second : i32
    return %result : i32
  }
}'''
parallel_branch_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=parallel_branch_source,
    text=True,
    capture_output=True,
)
assert parallel_branch_lowered.returncode == 0, parallel_branch_lowered.stderr
# Udon emits successor-argument assignments as sequential COPYs. The two
# branch sources must therefore interfere with both successor destinations;
# otherwise a destination COPY can overwrite the next source and violate SSA
# parallel-copy semantics.
for expected in [
    'minimum_simultaneous_registers = 4 : i64',
    'typed_spill_arrays = [{array_type = "SystemInt32Array", length = 3 : i64',
]:
    assert expected in parallel_branch_lowered.stdout, (
        expected,
        parallel_branch_lowered.stdout,
    )

typed_spill_source = r'''module attributes {
  vrc.async_stage = "udon-static-v1",
  vrc.host_stage = "udonhl",
  vrc.scalar_register_budget = 1 : i32
} {
  func.func @spill(%left: i32, %right: i32) -> i32 attributes {
    vrc.export_id = 1 : i32
  } {
    %zero = arith.constant 0 : i32
    %left_copy = arith.addi %left, %zero : i32
    %right_copy = arith.addi %right, %zero : i32
    %result = arith.addi %left_copy, %right_copy : i32
    return %result : i32
  }
}'''
typed_spill_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=typed_spill_source,
    text=True,
    capture_output=True,
)
assert typed_spill_lowered.returncode == 0, typed_spill_lowered.stderr
for expected in [
    'typed_spill_arrays = [{array_type = "SystemInt32Array"',
    'SystemInt32Array.__ctor__SystemInt32__SystemInt32Array',
    'SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid',
    'SystemInt32Array.__Get__SystemInt32__SystemInt32',
]:
    assert expected in typed_spill_lowered.stdout, (
        expected,
        typed_spill_lowered.stdout,
    )

capacity_source = typed_spill_source.replace(
    'vrc.scalar_register_budget = 1 : i32',
    'vrc.scalar_register_budget = 1 : i32, '
    'vrc.data_slot_capacity = 4 : i32',
)
capacity_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=capacity_source,
    text=True,
    capture_output=True,
)
assert capacity_lowered.returncode != 0
for expected in [
    'final Udon data table requires',
    'exceeding capacity 4',
    'fixed=',
    'scalar_registers=',
    'typed_arrays=',
    'constants=',
]:
    assert expected in capacity_lowered.stderr, (
        expected,
        capacity_lowered.stderr,
    )

raw_address_after_call_source = r'''module attributes {vrc.async_stage = "udon-static-v1"} {
  "udonhl.global"() <{sym_name = "value", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
  func.func @run() -> i32 attributes {
    vrc.export_id = 3 : i32,
    vrc.parameter_names = [],
    vrc.public_name = "RunRawAddressAfterCall"
  } {
    %address = udonhl.address_of @value : !udonhl.ptr<0>
    %ignored = func.call @helper() : () -> i32
    %result = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    return %result : i32
  }
  func.func private @helper() -> i32 {
    %value = arith.constant 7 : i32
    return %value : i32
  }
}'''
raw_address_after_call_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=raw_address_after_call_source,
    text=True,
    capture_output=True,
)
assert raw_address_after_call_lowered.returncode == 0, (
    raw_address_after_call_lowered.stderr
)
call_return = re.search(
    r'udonasm\.jump \d+\s+'
    r'udonasm\.push "__vrc_internal_bank_\d+_result_SystemInt32"\s+'
    r'udonasm\.push "__vrc_shared_r_SystemInt32_\d+"\s+'
    r'udonasm\.copy',
    raw_address_after_call_lowered.stdout,
)
assert call_return, raw_address_after_call_lowered.stdout
raw_load = raw_address_after_call_lowered.stdout.find(
    'SystemInt32Array.__Get__SystemInt32__SystemInt32',
    call_return.end(),
)
address_materialization = raw_address_after_call_lowered.stdout.find(
    'udonasm.push "__vrc_constant_0"',
    call_return.end(),
)
assert call_return.end() < address_materialization < raw_load, (
    raw_address_after_call_lowered.stdout
)

recursive_call_source = r'''module {
  func.func @run() attributes {
    vrc.export_id = 1 : i32,
    vrc.parameter_names = [],
    vrc.public_name = "Run"
  } {
    func.call @recurse() : () -> ()
    return
  }
  func.func private @recurse() {
    func.call @recurse() : () -> ()
    return
  }
}'''
recursive_call_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=recursive_call_source,
    text=True,
    capture_output=True,
)
assert recursive_call_lowered.returncode != 0
assert 'recursive internal Udon call graph has no static slot plan' in (
    recursive_call_lowered.stderr
)

global_source = r'''module {
  memref.global "private" @weights : memref<2xf32> = dense<[1.0, 2.0]>
  memref.global "private" @bias : memref<2xf32> = dense<[3.0, 4.0]>
  func.func @read_weight() -> f32 attributes {vrc.export_id = 1 : i32} {
    %weights = memref.get_global @weights : memref<2xf32>
    %bias = memref.get_global @bias : memref<2xf32>
    %index = arith.constant 1 : index
    %value = memref.load %weights[%index] : memref<2xf32>
    %bias_value = memref.load %bias[%index] : memref<2xf32>
    %result = arith.addf %value, %bias_value : f32
    return %result : f32
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
    'name = "__vrc_global_bank_f32"',
    'typed_global_arrays = [{element_type = "SystemSingle", length = 4 : i64',
    'name = "__vrc_typed_bank_base"',
    'name = "__vrc_globals_initialized"',
    'SystemSingleArray.__ctor__SystemInt32__SystemSingleArray',
    'SystemSingleArray.__Set__SystemInt32_SystemSingle__SystemVoid',
    'SystemSingleArray.__Get__SystemInt32__SystemSingle',
    'udonasm.jump_if_false',
]:
    assert expected in global_lowered.stdout, (expected, global_lowered.stdout)
assert 'memref.global' not in global_lowered.stdout
assert 'memref.get_global' not in global_lowered.stdout

integer_ops_source = r'''module {
  func.func @integer_ops(%value: i32, %other: i32) -> i32 attributes {vrc.export_id = 8 : i32} {
    %wide = arith.extsi %value : i32 to i64
    %mask = arith.constant 5 : i64
    %masked = arith.andi %wide, %mask : i64
    %narrow = arith.trunci %masked : i64 to i32
    %maximum = arith.maxsi %narrow, %other : i32
    %unsigned_maximum = arith.maxui %maximum, %value : i32
    %positive = arith.cmpi ugt, %unsigned_maximum, %value : i32
    %wide_positive = arith.cmpi ugt, %wide, %mask : i64
    %selected = arith.select %positive, %unsigned_maximum, %value : i32
    %product = arith.muli %selected, %other : i32
    %quotient = arith.divui %product, %other : i32
    %remainder = arith.remui %product, %other : i32
    %as_int = arith.extui %positive : i1 to i32
    %wide_as_int = arith.extui %wide_positive : i1 to i32
    %byte = arith.constant -1 : i8
    %signed_byte = arith.extsi %byte : i8 to i32
    %division_sum = arith.addi %quotient, %remainder : i32
    %sum = arith.addi %division_sum, %as_int : i32
    %sum2 = arith.addi %sum, %wide_as_int : i32
    %result = arith.addi %sum2, %signed_byte : i32
    return %result : i32
  }
}'''
integer_ops_lowered = subprocess.run(
    [opt, '--pass-pipeline=builtin.module(vrc-lower-host-to-udonhl,'
          'vrc-lower-udonhl-to-udonasm)', '--verify-each'],
    input=integer_ops_source, text=True, capture_output=True)
assert integer_ops_lowered.returncode == 0, integer_ops_lowered.stderr
for expected in [
    'SystemConvert.__ToInt64__SystemInt32__SystemInt64',
    'SystemInt64.__op_LogicalAnd__SystemInt64_SystemInt64__SystemInt64',
    'SystemInt64.__op_LogicalXor__SystemInt64_SystemInt64__SystemInt64',
    'SystemConvert.__ToInt32__SystemInt64__SystemInt32',
    'SystemInt32.__op_GreaterThanOrEqual__SystemInt32_SystemInt32__SystemBoolean',
    'SystemInt32.__op_LogicalXor__SystemInt32_SystemInt32__SystemInt32',
    'SystemConvert.__ToUInt32__SystemInt64__SystemUInt32',
    'SystemUInt32.__op_Division__SystemUInt32_SystemUInt32__SystemUInt32',
    'SystemConvert.__ToInt64__SystemUInt32__SystemInt64',
    'SystemInt32.__op_Multiplication__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_Subtraction__SystemInt32_SystemInt32__SystemInt32',
    'udonasm.jump_if_false',
]:
    assert expected in integer_ops_lowered.stdout, (
        expected, integer_ops_lowered.stdout)

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
    "udonhl.memcpy"(%source, %destination, %source_bytes) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> ()
    %copied = "udonhl.load"(%destination) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    return %copied : i32
  }
  func.func @raw_i64() -> i64 attributes {vrc.export_id = 4 : i32} {
    %bytes = arith.constant 8 : i32
    %pointer = "udonhl.alloca"(%bytes) <{alignment = 8 : i64}> : (i32) -> !udonhl.ptr<0>
    %value = arith.constant 72623859790382856 : i64
    "udonhl.store"(%value, %pointer) <{alignment = 8 : i64}> : (i64, !udonhl.ptr<0>) -> ()
    %loaded = "udonhl.load"(%pointer) <{alignment = 8 : i64}> : (!udonhl.ptr<0>) -> i64
    return %loaded : i64
  }
  func.func @raw_memset_zero() -> i32 attributes {vrc.export_id = 5 : i32} {
    %bytes = arith.constant 4 : i32
    %pointer = "udonhl.alloca"(%bytes) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %zero = arith.constant 0 : i8
    "udonhl.memset"(%pointer, %zero, %bytes) : (!udonhl.ptr<0>, i8, i32) -> ()
    %loaded = "udonhl.load"(%pointer) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    return %loaded : i32
  }
  func.func @raw_memset_nonzero() -> i32 attributes {vrc.export_id = 6 : i32} {
    %bytes = arith.constant 4 : i32
    %pointer = "udonhl.alloca"(%bytes) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %byte = arith.constant 127 : i8
    "udonhl.memset"(%pointer, %byte, %bytes) : (!udonhl.ptr<0>, i8, i32) -> ()
    %loaded = "udonhl.load"(%pointer) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    return %loaded : i32
  }
  func.func @raw_bool() -> i1 attributes {vrc.export_id = 7 : i32} {
    %bytes = arith.constant 1 : i32
    %pointer = "udonhl.alloca"(%bytes) <{alignment = 1 : i64}> : (i32) -> !udonhl.ptr<0>
    %value = arith.constant true
    "udonhl.store"(%value, %pointer) <{alignment = 1 : i64}> : (i1, !udonhl.ptr<0>) -> ()
    %loaded = "udonhl.load"(%pointer) <{alignment = 1 : i64}> : (!udonhl.ptr<0>) -> i1
    return %loaded : i1
  }
}'''
raw_memory_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=raw_memory_source, text=True, capture_output=True)
assert raw_memory_lowered.returncode == 0, raw_memory_lowered.stderr
assert 'udonasm.storage_report' in raw_memory_lowered.stdout
assert 'raw_heap_bytes = 41 : i64' in raw_memory_lowered.stdout
assert 'raw_heap_words = 11 : i64' in raw_memory_lowered.stdout
assert re.search(
    r'\{alignment = 4 : i64, base = 4 : i64, bytes = 4 : i64, '
    r'kind = "global", name = "handle", '
    r'reason = "address_observable_after_promotion"\}',
    raw_memory_lowered.stdout,
), raw_memory_lowered.stdout
assert re.search(
    r'kind = "alloca", name = "raw_memcpy:alloca\d+", '
    r'reason = "byte_or_bulk_memory"',
    raw_memory_lowered.stdout,
), raw_memory_lowered.stdout
for expected in [
    'name = "__vrc_raw_heap"',
    'initial = "11", name = "__vrc_raw_heap_length"',
    'SystemInt32Array.__ctor__SystemInt32__SystemInt32Array',
    'SystemInt32Array.__Get__SystemInt32__SystemInt32',
    'SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid',
    'SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt64.__op_LeftShift__SystemInt64_SystemInt32__SystemInt64',
    'SystemInt64.__op_RightShift__SystemInt64_SystemInt32__SystemInt64',
    'SystemInt64.__op_LogicalAnd__SystemInt64_SystemInt64__SystemInt64',
    'SystemInt64.__op_LogicalOr__SystemInt64_SystemInt64__SystemInt64',
    'SystemBitConverter.__SingleToInt32Bits__SystemSingle__SystemInt32',
    'SystemBitConverter.__Int32BitsToSingle__SystemInt32__SystemSingle',
    'SystemInt32.__op_Inequality__SystemInt32_SystemInt32__SystemBoolean',
    'name = "__vrc_raw_memcpy_counter"',
    'name = "__vrc_raw_memset_scratch_counter"',
]:
    assert expected in raw_memory_lowered.stdout, (
        expected, raw_memory_lowered.stdout)
for forbidden in ['__op_BitwiseAnd__', '__op_BitwiseOr__', '__op_Multiply__']:
    assert forbidden not in raw_memory_lowered.stdout, (
        forbidden, raw_memory_lowered.stdout)
for typed_scratch, udon_type in [
    ('__vrc_raw_load_i32_scratch_bits', 'SystemInt32'),
    ('__vrc_raw_store_i32_scratch_bits', 'SystemInt32'),
    ('__vrc_i64_scratch_0', 'SystemInt64'),
    ('__vrc_i64_scratch_1', 'SystemInt64'),
    ('__vrc_i64_scratch_2', 'SystemInt64'),
    ('__vrc_i64_scratch_3', 'SystemInt64'),
]:
    declaration = f'name = "{typed_scratch}", type = "{udon_type}"'
    assert declaration in raw_memory_lowered.stdout, (
        declaration, raw_memory_lowered.stdout)
for obsolete_scratch in [
    '__vrc_i64_constant_',
    'name = "__vrc_raw_load_i64_scratch_bits"',
    'name = "__vrc_raw_load_i64_scratch_wide"',
    'name = "__vrc_raw_load_i64_scratch_component"',
    'name = "__vrc_raw_load_i64_scratch_combined"',
    'name = "__vrc_raw_store_i64_scratch_bits"',
    'name = "__vrc_raw_store_i64_scratch_mask"',
    'name = "__vrc_raw_store_i64_scratch_component"',
    'name = "__vrc_raw_store_i64_scratch_byte_wide"',
]:
    assert obsolete_scratch not in raw_memory_lowered.stdout, (
        obsolete_scratch, raw_memory_lowered.stdout)
int64_initializers = re.findall(
    r'\{initial = "([^"]+)", name = "[^"]+", type = "SystemInt64"\}',
    raw_memory_lowered.stdout,
)
assert int64_initializers and set(int64_initializers) == {'null'}, \
    int64_initializers
assert 'SystemArray.__Copy__SystemArray_SystemInt32_SystemArray_' not in (
    raw_memory_lowered.stdout
)
for forbidden in ['udonhl.global', 'udonhl.address_of', 'udonhl.alloca',
                  'udonhl.load', 'udonhl.store', 'udonhl.memcpy']:
    assert forbidden not in raw_memory_lowered.stdout, (
        forbidden, raw_memory_lowered.stdout)

fast_direct_global_source = r'''module {
  "udonhl.global"() <{sym_name = "ordinary_internal_state", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
  func.func @fast_direct_global() -> i32 attributes {vrc.export_id = 1 : i32} {
    %address = udonhl.address_of @ordinary_internal_state : !udonhl.ptr<0>
    %value = arith.constant 42 : i32
    "udonhl.store"(%value, %address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %result = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    return %result : i32
  }
}'''
fast_direct_global = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=fast_direct_global_source, text=True, capture_output=True)
assert fast_direct_global.returncode == 0, fast_direct_global.stderr
assert fast_direct_global.stdout.count(
    'SystemInt32Array.__Get__SystemInt32__SystemInt32') == 1
assert fast_direct_global.stdout.count(
    'SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid') == 1
for forbidden in [
    '__vrc_raw_load_i32_scratch_byte',
    '__vrc_raw_store_i32_scratch_byte',
]:
    assert forbidden not in fast_direct_global.stdout, (
        forbidden, fast_direct_global.stdout)

zero_i32_store_source = r'''module {
  func.func @zero_i32_store() attributes {vrc.export_id = 1 : i32} {
    %size = arith.constant 4 : i32
    %address = "udonhl.alloca"(%size) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %zero = arith.constant 0 : i32
    "udonhl.store"(%zero, %address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    return
  }
}'''
zero_i32_store = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=zero_i32_store_source, text=True, capture_output=True)
assert zero_i32_store.returncode == 0, zero_i32_store.stderr
assert zero_i32_store.stdout.count(
    'SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid') == 1
assert '__vrc_raw_byte_store_word' not in zero_i32_store.stdout

boolean_bitwise_source = r'''module {
  func.func @boolean_bitwise(%lhs: i1, %rhs: i1) -> i1 attributes {vrc.export_id = 1 : i32} {
    %or = arith.ori %lhs, %rhs : i1
    %xor = arith.xori %lhs, %rhs : i1
    %result = arith.andi %or, %xor : i1
    return %result : i1
  }
}'''
boolean_bitwise = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=boolean_bitwise_source, text=True, capture_output=True)
assert boolean_bitwise.returncode == 0, boolean_bitwise.stderr
for forbidden in [
    'SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32',
    'SystemInt32.__op_LogicalXor__SystemInt32_SystemInt32__SystemInt32',
]:
    assert forbidden not in boolean_bitwise.stdout, (
        forbidden, boolean_bitwise.stdout)

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
assert 'name = "__vrc_raw_memcpy_counter"' in rejected.stdout

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

invalid = source.replace('%sum = arith.addi %a, %b : i32', '%sum = arith.maxui %a, %b : i32')
rejected = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm'], input=invalid, text=True,
    capture_output=True)
assert rejected.returncode != 0
assert 'extern legalization set' in rejected.stderr

buffer_source = r'''module attributes {
  vrc.host_stage = "udonhl",
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_buffer_1"},
    {id = 2 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_buffer_2"}
  ]
} {
  func.func @copy_full() attributes {vrc.export_id = 6 : i32} {
    %bytes = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %source_id = arith.constant 1 : i32
    %destination_id = arith.constant 2 : i32
    %source, %source_status = vrc.buffer_alloc %source_id, %bytes {
      format = "rgba8_unorm", usage = "storage",
      vrc.unity_buffer_id = 1 : i32,
      vrc.unity_heap_slot = "__udoncl_buffer_1",
      vrc.unity_logical_bytes = 4 : i32
    } : (i32, i32) -> (!vrc.buffer, i32)
    %destination, %destination_status = vrc.buffer_alloc %destination_id, %bytes {
      format = "rgba8_unorm", usage = "storage",
      vrc.unity_buffer_id = 2 : i32,
      vrc.unity_heap_slot = "__udoncl_buffer_2",
      vrc.unity_logical_bytes = 4 : i32
    } : (i32, i32) -> (!vrc.buffer, i32)
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
    'name = "__udoncl_completed_event"',
    'VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_'
    'UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid',
]:
    assert expected in buffer_lowered.stdout, (expected, buffer_lowered.stdout)
assert re.search(
    r'udonasm\.push "__udoncl_completed_event"\s+'
    r'udonasm\.push "[^"]+"\s+udonasm\.copy',
    buffer_lowered.stdout), buffer_lowered.stdout

async_buffer_source = r'''module attributes {
  vrc.host_stage = "udonhl",
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_buffer_1"}
  ]
} {
  func.func @transfer() -> i32 attributes {
    vrc.async_effect = "may_suspend",
    vrc.export_id = 7 : i32
  } {
    %id = arith.constant 1 : i32
    %bytes = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %pointer = arith.constant 16 : i32
    %buffer, %status = vrc.buffer_alloc %id, %bytes {
      format = "rgba8_unorm", usage = "storage"
    } : (i32, i32) -> (!vrc.buffer, i32)
    %loaded_id = arith.addi %id, %zero : i32
    %host_pointer = arith.addi %pointer, %zero : i32
    %recovered = builtin.unrealized_conversion_cast %loaded_id : i32 to !vrc.buffer
    %uploaded = vrc.buffer_upload %recovered, %zero, %bytes, %host_pointer : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %upload_status = vrc.await %uploaded : (!vrc.event) -> i32
    %read = vrc.buffer_readback %buffer, %zero, %bytes, %pointer : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %read_status = vrc.await %read : (!vrc.event) -> i32
    %result = arith.addi %upload_status, %read_status : i32
    return %result : i32
  }
}'''
async_buffer_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=async_buffer_source, text=True, capture_output=True)
assert async_buffer_lowered.returncode == 0, async_buffer_lowered.stderr
for expected in [
    'name = "__udoncl_async_resume_state"',
    'name = "__udoncl_async_completion_status"',
    'name = "__udoncl_async_command_kind"',
    'name = "__udoncl_async_command_buffer"',
    'name = "__udoncl_async_command_offset"',
    'name = "__udoncl_async_command_size"',
    'name = "__udoncl_async_command_pointer"',
    'name = "__vrc_f7_buffer_',
    'initial = "1", name = "__vrc_f7_upload_command"',
    'initial = "2", name = "__vrc_f7_readback_command"',
    'udonasm.jump 4294967292',
]:
    assert expected in async_buffer_lowered.stdout, (
        expected, async_buffer_lowered.stdout)
assert async_buffer_lowered.stdout.count('udonasm.jump 4294967292') == 3
command_buffer_copy = re.search(
    r'udonasm\.push "([^"]+)"\s+'
    r'udonasm\.push "__udoncl_async_command_buffer"\s+udonasm\.copy',
    async_buffer_lowered.stdout)
command_pointer_copy = re.search(
    r'udonasm\.push "([^"]+)"\s+'
    r'udonasm\.push "__udoncl_async_command_pointer"\s+udonasm\.copy',
    async_buffer_lowered.stdout)
assert command_buffer_copy and command_pointer_copy, async_buffer_lowered.stdout
assert command_buffer_copy.group(1) != command_pointer_copy.group(1), (
    command_buffer_copy.group(1), command_pointer_copy.group(1),
    async_buffer_lowered.stdout)

async_buffer_mlir = output / 'async-buffer.udonasm.mlir'
async_buffer_uasm = output / 'async-buffer.uasm'
async_buffer_mlir.write_text(async_buffer_lowered.stdout, encoding='utf-8')
emitted = subprocess.run(
    [translate, '--emit-udon-assembly', async_buffer_mlir, '-o', async_buffer_uasm],
    text=True,
    capture_output=True,
)
assert emitted.returncode == 0, emitted.stderr
async_buffer_assembly = async_buffer_uasm.read_text(encoding='utf-8')
for expected in [
    '.export transfer',
    '__vrc_f7_await_state_value: %SystemInt32, 0',
    'SystemInt32.__op_Equality__SystemInt32_SystemInt32__SystemBoolean',
]:
    assert expected in async_buffer_assembly, (expected, async_buffer_assembly)

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
for persistent in [
    '__vrc_f6_frame_0_slot_0',
    '__vrc_f6_frame_0_slot_1',
]:
    assert re.search(
        r'async_persistent_values = \[[^]]*name = "' + persistent,
        continuation_lowered.stdout,
    ), continuation_lowered.stdout
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

static_async_source = r'''module attributes {vrc.host_stage = "udon-static-v1"} {
  func.func @start(%frame: i32, %pc: i32) attributes {
    vrc.export_id = 8 : i32,
    vrc.parameter_names = ["frame", "pc"],
    vrc.public_name = "Start"
  } {
    "udonhl.async_resume"(%frame, %pc) : (i32, i32) -> ()
    return
  }
  func.func private @continuation() attributes {
    vrc.async_frame_id = 1 : i32
  } {
    cf.br ^resume
  ^resume:
    "udonhl.async_resume_point"() <{
      frame_id = 1 : i32,
      resume_pc = 2 : i32
    }> : () -> ()
    return
  }
  func.func @__udoncl_async_complete(%object: i32, %status: i32) attributes {
    vrc.compiler_async_completion,
    vrc.udon_event = "__udoncl_async_complete"
  } {
    "udonhl.async_resume"(%object, %status) : (i32, i32) -> ()
    return
  }
}'''
static_async_lowered = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm', '--verify-each'],
    input=static_async_source,
    text=True,
    capture_output=True,
)
assert static_async_lowered.returncode == 0, static_async_lowered.stderr
for expected in [
    'name = "__vrc_f8_async_route_0_0_frame"',
    'name = "__vrc_f8_async_route_0_0_pc"',
    'name = "__vrc_return_address_i32"',
    'name = "__udoncl_async_completion_object"',
    'name = "__udoncl_async_completion_status"',
    'name = "__udoncl_async_complete"',
    'SystemInt32.__op_Equality__SystemInt32_SystemInt32__SystemBoolean',
    'SystemConvert.__ToUInt32__SystemInt32__SystemUInt32',
    'udonasm.jump_indirect "__vrc_internal_bank_0_return"',
]:
    assert expected in static_async_lowered.stdout, (
        expected,
        static_async_lowered.stdout,
    )
assert 'udonhl.async_' not in static_async_lowered.stdout

bound_event = static_async_source.replace(
    '"udonhl.async_resume"(%frame, %pc) : (i32, i32) -> ()',
    '%event = builtin.unrealized_conversion_cast %frame : i32 to !vrc.event\n'
    '    "udonhl.async_bind_event"(%event) <{object_id = 1 : i32}> : '
    '(!vrc.event) -> ()',
)
rejected = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm'],
    input=bound_event,
    text=True,
    capture_output=True,
)
assert rejected.returncode == 0, rejected.stderr
assert 'name = "__udoncl_async_pending_object"' in rejected.stdout
assert 'udonhl.async_bind_event' not in rejected.stdout

unbound_buffer = buffer_source.replace(
    'vrc.unity_buffer_pool', 'vrc.unity_buffer_pool_missing', 1)
rejected = subprocess.run(
    [opt, '--vrc-lower-udonhl-to-udonasm'], input=unbound_buffer,
    text=True, capture_output=True)
assert rejected.returncode != 0
assert 'Unity copy has no physical buffer pool' in rejected.stderr

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
