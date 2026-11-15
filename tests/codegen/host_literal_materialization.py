import re
import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
SHIFT = "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__SystemInt32"
LOGICAL_OR = (
    "SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32"
)
ADD = "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"


def signed32(value):
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def recipe(value):
    bits = value & 0xFFFFFFFF
    highest = max(
        (nibble for nibble in range(8) if (bits >> (4 * nibble)) & 15),
        default=0,
    )
    seeds = {(bits >> (4 * highest)) & 15}
    externs = []
    if highest:
        seeds.add(4)
        externs.append(SHIFT)
    if any((bits >> (4 * nibble)) & 15 for nibble in range(highest)):
        externs.append(LOGICAL_OR)
        seeds.update(
            (bits >> (4 * nibble)) & 15
            for nibble in range(highest)
            if (bits >> (4 * nibble)) & 15
        )
    return bits, highest, sorted(seeds), externs


def source(value, logical_type="i32", omit_seed=None, with_consumer=False):
    bits, _, seeds, externs = recipe(value)
    constants = "\n".join(
        f"    %seed{seed} = arith.constant {seed} : i32"
        for seed in seeds
        if seed != omit_seed
    )
    bindings = ", ".join(
        "{color = %d : i32, literal_key = \"SystemInt32:0x%X\", "
        "physical_type = \"SystemInt32\", storage_domain = \"resident\"}"
        % (index, seed)
        for index, seed in enumerate(seeds)
    )
    required = ", ".join(f'"{name}"' for name in externs)
    resident_allocations = ", ".join(
        "{color = %d : i32, key_id = \"SystemInt32:0x%X\", "
        "physical_type = \"SystemInt32\", placement = \"resident_shared\"}"
        % (index, seed)
        for index, seed in enumerate(seeds)
    )
    direct_allocations = ""
    consumer = ""
    returned = "%value"
    if with_consumer:
        result_id = (
            "function:materialize:region:0:block:0:operation:"
            f"{len(seeds) + 1}:result:0"
        )
        direct_allocations = (
            "{allocation = \"direct_color_candidate\", color = 8 : i32, "
            'physical_type = "SystemInt32", storage_domain = "writable", '
            f'value_id = "{result_id}"}}'
        )
        consumer = "    %result = arith.addi %value, %value : i32\n"
        returned = "%result"
    literal = signed32(bits)
    return f'''module attributes {{
  vrc.async_stage = "udon-static-v1",
  vrc.host_direct_color_allocations = [{direct_allocations}],
  vrc.host_literal_allocation_version = 1 : i32,
  vrc.host_literal_allocations = [{resident_allocations}],
  vrc.host_stage = "udonhl",
  vrc.host_writable_allocation_version = 2 : i32
}} {{
  func.func @materialize() -> {logical_type} attributes {{vrc.export_id = 91 : i32}} {{
{constants}
    %value = "udonhl.literal_materialize"() <{{
      color = 7 : i32,
      generated_version = 1 : i32,
      helper_scratch = 0 : i32,
      instance_id = "test:{logical_type}:{bits:X}",
      literal = {literal} : {logical_type},
      literal_key = "SystemInt32:0x{bits:X}",
      physical_type = "SystemInt32",
      recipe = "rematerialize_i32",
      required_externs = [{required}],
      seed_bindings = [{bindings}],
      storage_domain = "writable"
    }}> : () -> {logical_type}
{consumer}    return {returned} : {logical_type}
  }}
}}
'''


def lower(text):
    return subprocess.run(
        [optimizer, "--vrc-lower-udonhl-to-udonasm", "--verify-each"],
        input=text,
        text=True,
        capture_output=True,
    )


def interpret(text):
    values = {}
    for initial, name in re.findall(
        r'\{initial = "([^"]+)", name = "([^"]+)", type = "SystemInt32"\}',
        text,
    ):
        values[name] = 0 if initial == "null" else int(initial)
    stack = []
    for line in text.splitlines():
        if match := re.fullmatch(r'\s*udonasm\.push "([^"]+)"', line):
            stack.append(match.group(1))
        elif re.fullmatch(r"\s*udonasm\.copy", line):
            destination = stack.pop()
            source_slot = stack.pop()
            values[destination] = values[source_slot]
        elif match := re.fullmatch(r'\s*udonasm\.extern "([^"]+)"', line):
            result = stack.pop()
            rhs = values[stack.pop()]
            lhs = values[stack.pop()]
            if match.group(1) == SHIFT:
                values[result] = signed32(lhs << rhs)
            elif match.group(1) == LOGICAL_OR:
                values[result] = signed32(lhs | rhs)
            elif match.group(1) == ADD:
                values[result] = signed32(lhs + rhs)
            else:
                raise AssertionError(match.group(1))
    return values["__0_materialize__ret"]


cases = [
    (0, "i32"),
    (15, "i32"),
    (16, "i32"),
    (22, "i32"),
    (33, "index"),
    (-1, "i8"),
    (-2, "i16"),
    (-2147483648, "i32"),
    (2147483647, "i32"),
]
for value, logical_type in cases:
    result = lower(source(value, logical_type))
    assert result.returncode == 0, (value, logical_type, result.stderr)
    assert 'name = "__vrc_shared_r_SystemInt32_7"' in result.stdout
    assert "udonhl.literal_materialize" not in result.stdout
    assert interpret(result.stdout) == signed32(value), (value, result.stdout)
    _, highest, seeds, _ = recipe(value)
    assert result.stdout.count(f'udonasm.extern "{SHIFT}"') == highest
    expected_or = sum(
        1
        for nibble in range(highest)
        if ((value & 0xFFFFFFFF) >> (4 * nibble)) & 15
    )
    assert result.stdout.count(f'udonasm.extern "{LOGICAL_OR}"') == expected_or
    assert len(re.findall(r'kind = "arith_constant"', result.stdout)) == len(
        seeds
    )
    for color, seed in enumerate(seeds):
        assert f'name = "__vrc_resident_SystemInt32_{color}"' in result.stdout

twenty_two = lower(source(22))
assert 'initial = "22"' not in twenty_two.stdout

missing = lower(source(22, omit_seed=6))
assert missing.returncode != 0

conflict = lower(source(22, with_consumer=True))
assert conflict.returncode == 0, conflict.stderr
assert 'name = "__vrc_shared_r_SystemInt32_7"' in conflict.stdout
assert 'name = "__vrc_shared_r_SystemInt32_8"' in conflict.stdout
assert interpret(conflict.stdout) == 44

repeat = lower(source(22))
assert repeat.returncode == 0
assert repeat.stdout == twenty_two.stdout

candidate_only = lower(
    '''module attributes {
  vrc.host_direct_color_allocations = [{color = 99 : i32}],
  vrc.host_writable_allocation_version = 2 : i32,
  vrc.host_stage = "udonhl"
} {
  func.func @candidate_only() -> i32 attributes {vrc.export_id = 92 : i32} {
    %value = arith.constant 22 : i32
    return %value : i32
  }
}
'''
)
assert candidate_only.returncode == 0, candidate_only.stderr
assert "__vrc_resident_" not in candidate_only.stdout
assert 'initial = "22"' in candidate_only.stdout

print(f"host_literal_materialization_codegen=passed cases={len(cases)}")
