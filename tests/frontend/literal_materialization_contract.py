import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2]).read_text()


def run(text):
    return subprocess.run(
        [optimizer, "-"], input=text, text=True, capture_output=True
    )


valid = run(source)
assert valid.returncode == 0, (valid.stdout, valid.stderr)
assert '"udonhl.literal_materialize"' in valid.stdout
assert 'recipe = "rematerialize_i32"' in valid.stdout
assert 'storage_domain = "writable"' in valid.stdout

SHIFT = "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__SystemInt32"
LOGICAL_OR = "SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32"


def contract(literal, logical_type, key, seeds, externs):
    seed_bindings = ",\n        ".join(
        "{color = %d : i32, literal_key = \"SystemInt32:0x%X\", "
        "physical_type = \"SystemInt32\", storage_domain = \"resident\"}"
        % (color, seed)
        for color, seed in enumerate(seeds)
    )
    required_externs = ", ".join(f'"{name}"' for name in externs)
    return f'''module {{
  func.func @case() -> {logical_type} {{
    %value = "udonhl.literal_materialize"() <{{
      color = 0 : i32,
      generated_version = 1 : i32,
      helper_scratch = 0 : i32,
      instance_id = "case:{logical_type}:{key}",
      literal = {literal} : {logical_type},
      literal_key = "SystemInt32:{key}",
      physical_type = "SystemInt32",
      recipe = "rematerialize_i32",
      required_externs = [{required_externs}],
      seed_bindings = [{seed_bindings}],
      storage_domain = "writable"
    }}> : () -> {logical_type}
    return %value : {logical_type}
  }}
}}
'''


for case in [
    ("0", "i32", "0x0", [0], []),
    ("1", "i32", "0x1", [1], []),
    ("15", "i32", "0xF", [15], []),
    ("16", "i32", "0x10", [1, 4], [SHIFT]),
    ("22", "i32", "0x16", [1, 4, 6], [SHIFT, LOGICAL_OR]),
    ("-1", "i32", "0xFFFFFFFF", [4, 15], [SHIFT, LOGICAL_OR]),
    ("-1", "i8", "0xFFFFFFFF", [4, 15], [SHIFT, LOGICAL_OR]),
    ("-1", "i16", "0xFFFFFFFF", [4, 15], [SHIFT, LOGICAL_OR]),
    ("22", "index", "0x16", [1, 4, 6], [SHIFT, LOGICAL_OR]),
]:
    result = run(contract(*case))
    assert result.returncode == 0, (case, result.stdout, result.stderr)


def reject(old, new, count=1):
    changed = source.replace(old, new, count)
    assert changed != source
    result = run(changed)
    assert result.returncode != 0, (old, new, result.stdout, result.stderr)


reject("literal = 22 : i32", "literal = 23 : i32")
reject('literal_key = "SystemInt32:0x16"', 'literal_key = "SystemInt32:0x17"')
reject('storage_domain = "writable"', 'storage_domain = "resident"', 1)
reject("color = 0 : i32", "color = -1 : i32", 1)
reject("generated_version = 1 : i32", "generated_version = 2 : i32")
reject('recipe = "rematerialize_i32"', 'recipe = "unknown"')
reject('instance_id = "function:selected_literal:consumer:0:literal:SystemInt32:0x16"', 'instance_id = ""')
reject("helper_scratch = 0 : i32", "helper_scratch = 1 : i32")
reject('literal_key = "SystemInt32:0x4"', 'literal_key = "SystemInt32:0x5"')
reject("color = 2 : i32", "color = 1 : i32", 1)
reject('storage_domain = "resident"', 'storage_domain = "writable"', 1)
reject('physical_type = "SystemInt32", storage_domain = "resident"', 'physical_type = "SystemSingle", storage_domain = "resident"', 1)
reject("color = 1 : i32, literal_key = \"SystemInt32:0x4\"", "color = -1 : i32, literal_key = \"SystemInt32:0x4\"")
reject('literal_key = "SystemInt32:0x1"', 'literal_key = "SystemInt32:0x6"', 1)
reject(
    '{color = 2 : i32, literal_key = "SystemInt32:0x6", physical_type = "SystemInt32", storage_domain = "resident"}',
    '',
)
reject(
    '"SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32"',
    '"SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"',
)
reject(
    ',\n        "SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32"',
    '',
)
reject(
    '"SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32"',
    '"SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32", '
    '"SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"',
)
reject(": () -> i32", ": () -> i64", 1)

print("literal_materialization_contract=passed")
