import re
import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
pipeline = [
    optimizer,
    "--vrc-discover-host-rematerialization",
    "--vrc-allocate-host-storage",
    source,
]
allocated = subprocess.run(
    pipeline, text=True, capture_output=True, check=True
).stdout


def entries(text, name, next_name):
    start = text.index(f"{name} = [")
    end = text.index(next_name, start)
    return re.findall(r"\{([^{}]+)\}", text[start:end])


scratch_entries = entries(
    allocated,
    "vrc.host_literal_scratch_allocations",
    "vrc.host_literal_scratch_instances",
)
scratch = {
    re.search(r'instance_id = "([^"]+)"', entry).group(1): entry
    for entry in scratch_entries
}
resident_entries = entries(
    allocated,
    "vrc.host_literal_allocations",
    "vrc.host_literal_consumer_groups",
)
resident_colors = {
    re.search(r'key_id = "([^"]+)"', entry).group(1): int(
        re.search(r"color = (\d+) : i32", entry).group(1)
    )
    for entry in resident_entries
}

SHIFT = "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__SystemInt32"
LOGICAL_OR = (
    "SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32"
)
recipes = {
    "SystemInt32:0x16": ([1, 4, 6], [SHIFT, LOGICAL_OR]),
    "SystemInt32:0x21": ([1, 2, 4], [SHIFT, LOGICAL_OR]),
    "SystemInt32:0xFFFFFFFF": ([4, 15], [SHIFT, LOGICAL_OR]),
}


def field(entry, name):
    match = re.search(rf'{name} = "([^"]+)"', entry)
    assert match, (name, entry)
    return match.group(1)


def selection(instance_id):
    entry = scratch[instance_id]
    literal_key = field(entry, "literal_key")
    logical_type = field(entry, "logical_type")
    physical_type = field(entry, "physical_type")
    color = int(re.search(r"color = (\d+) : i32", entry).group(1))
    site_text = re.search(r"site_ids = \[([^]]+)\]", entry).group(1)
    site_ids = re.findall(r'"([^"]+)"', site_text)
    seeds, externs = recipes[literal_key]
    seed_bindings = ", ".join(
        "{color = %d : i32, literal_key = \"SystemInt32:0x%X\", "
        "physical_type = \"SystemInt32\", storage_domain = \"resident\"}"
        % (resident_colors[f"SystemInt32:0x{seed:X}"], seed)
        for seed in seeds
    )
    required_externs = ", ".join(f'"{extern}"' for extern in externs)
    sites = ", ".join(f'"{site}"' for site in site_ids)
    return (
        "{color = %d : i32, generated_version = 1 : i32, "
        "helper_scratch = 0 : i32, instance_id = \"%s\", "
        "literal_key = \"%s\", logical_type = \"%s\", "
        "physical_type = \"%s\", placement = \"rematerialized\", "
        "recipe = \"rematerialize_i32\", required_externs = [%s], "
        "seed_bindings = [%s], site_ids = [%s], "
        "storage_domain = \"writable\"}"
        % (
            color,
            instance_id,
            literal_key,
            logical_type,
            physical_type,
            required_externs,
            seed_bindings,
            sites,
        )
    )


def selected(instance_ids):
    materializations = ", ".join(selection(item) for item in sorted(instance_ids))
    return allocated.replace(
        "module attributes {",
        "module attributes {vrc.host_literal_selected_placement_version = "
        "1 : i32, vrc.host_literal_selected_placements = ["
        f"{materializations}], ",
        1,
    )


def run(text):
    return subprocess.run(
        [optimizer, "--vrc-materialize-host-storage", "-"],
        input=text,
        text=True,
        capture_output=True,
    )


candidate_only = run(allocated)
assert candidate_only.returncode == 0
assert '"udonhl.literal_materialize"' not in candidate_only.stdout
assert "vrc.host_literal_materialization_version" not in candidate_only.stdout

prefix = "function:materialization_cases:region:0:block:0:operation:"
duplicate = f"{prefix}10:literal:SystemInt32:0x16:logical:i32"
distinct22 = f"{prefix}11:literal:SystemInt32:0x16:logical:i32"
distinct33 = f"{prefix}11:literal:SystemInt32:0x21:logical:i32"
call22 = f"{prefix}12:literal:SystemInt32:0x16:logical:i32"
call33 = f"{prefix}12:literal:SystemInt32:0x21:logical:i32"
mixed8 = f"{prefix}13:literal:SystemInt32:0xFFFFFFFF:logical:i8"
mixed32 = f"{prefix}13:literal:SystemInt32:0xFFFFFFFF:logical:i32"

one = run(selected([duplicate]))
assert one.returncode == 0, (one.stdout, one.stderr)
assert one.stdout.count('"udonhl.literal_materialize"') == 1
same_operand = re.search(r"arith\.addi (%\w+), (%\w+) : i32", one.stdout)
assert same_operand and same_operand.group(1) == same_operand.group(2)
assert "vrc.host_literal_materialization_version = 1 : i32" in one.stdout

all_instances = [
    duplicate,
    distinct22,
    distinct33,
    call22,
    call33,
    mixed8,
    mixed32,
]
all_selected = run(selected(all_instances))
assert all_selected.returncode == 0, (all_selected.stdout, all_selected.stderr)
assert all_selected.stdout.count('"udonhl.literal_materialize"') == 7
assert re.search(r'"udonhl\.literal_materialize".*: \(\) -> i8', all_selected.stdout)
assert re.search(r'"udonhl\.literal_materialize".*: \(\) -> i32', all_selected.stdout)
assert "arith.constant 22 : i32" not in all_selected.stdout
assert "arith.constant 33 : i32" not in all_selected.stdout
assert re.search(r"call @sink_i32\(%\w+, %\w+\)", all_selected.stdout)
assert not re.search(r"call @sink_i32\(%c", all_selected.stdout)
assert re.search(
    rf'color = 2 : i32.*instance_id = "{re.escape(distinct22)}"',
    all_selected.stdout,
)
assert re.search(
    rf'color = 3 : i32.*instance_id = "{re.escape(distinct33)}"',
    all_selected.stdout,
)
call22_result = re.search(
    rf'(%\w+) = "udonhl\.literal_materialize".*instance_id = "{re.escape(call22)}"',
    all_selected.stdout,
).group(1)
call33_result = re.search(
    rf'(%\w+) = "udonhl\.literal_materialize".*instance_id = "{re.escape(call33)}"',
    all_selected.stdout,
).group(1)
assert f"call @sink_i32({call22_result}, {call33_result})" in all_selected.stdout


def reject(text):
    result = run(text)
    assert result.returncode != 0, (result.stdout, result.stderr)


reject(allocated.replace(
    "module attributes {",
    "module attributes {vrc.host_literal_selected_placement_version = 1 : i32, ",
    1,
))
reject(
    selected([duplicate]).replace(
        "vrc.host_literal_selected_placement_version = 1 : i32, ", "", 1
    )
)
wrong_color = selected([duplicate]).replace("color = 0 : i32", "color = 99 : i32", 1)
reject(wrong_color)
reject(
    selected([duplicate]).replace(
        "color = 1 : i32, generated_version", "color = 99 : i32, generated_version", 1
    )
)
partial = selected([duplicate]).replace(
    f'"{prefix}10:operand:1"', "", 1
)
reject(partial)
branch_instance = (
    "function:branch_literal:region:0:block:0:operation:1:"
    "literal:SystemInt32:0x7:logical:i32"
)
branch = selected([duplicate]).replace(duplicate, branch_instance, 1)
reject(branch)
reject(
    selected([duplicate]).replace(
        "vrc.host_literal_inventory_version = 1 : i32",
        "vrc.host_literal_inventory_version = 2 : i32",
        1,
    )
)
reject(
    selected([duplicate]).replace(
        "vrc.host_literal_allocation_version = 1 : i32",
        "vrc.host_literal_allocation_version = 2 : i32",
        1,
    )
)
reject(selected([duplicate]).replace("helper_scratch = 0", "helper_scratch = 1", 1))
reject(selected([duplicate]).replace("generated_version = 1", "generated_version = 2", 1))
reject(selected([duplicate]).replace(SHIFT, LOGICAL_OR, 1))
seed1 = (
    '{color = 0 : i32, literal_key = "SystemInt32:0x1", '
    'physical_type = "SystemInt32", storage_domain = "resident"}'
)
seed4 = (
    '{color = 4 : i32, literal_key = "SystemInt32:0x4", '
    'physical_type = "SystemInt32", storage_domain = "resident"}'
)
reject(selected([duplicate]).replace(seed1 + ", ", "", 1))
reject(selected([duplicate]).replace(seed4, seed1, 1))
reject(selected([duplicate]).replace("SystemInt32:0x1", "SystemInt32:0x2", 1))
reject(
    selected([duplicate]).replace(
        "%c22_i32 = arith.constant 22 : i32",
        "%c22_i32 = arith.constant 23 : i32",
        1,
    )
)

repeat = run(selected(all_instances))
assert repeat.returncode == 0
assert repeat.stdout == all_selected.stdout
print("host_storage_materialization=passed selected=7")
