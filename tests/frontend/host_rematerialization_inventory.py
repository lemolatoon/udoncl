import re
import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [optimizer, "--vrc-discover-host-rematerialization", source]
double_command = [
    optimizer,
    "--vrc-discover-host-rematerialization",
    "--vrc-discover-host-rematerialization",
    source,
]
baseline_command = [optimizer, source]
allocation_command = [
    optimizer,
    "--vrc-discover-host-rematerialization",
    "--vrc-allocate-host-storage",
    source,
]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
allocated = subprocess.run(
    allocation_command, text=True, capture_output=True, check=True
).stdout
allocated_twice = subprocess.run(
    [
        optimizer,
        "--vrc-discover-host-rematerialization",
        "--vrc-allocate-host-storage",
        "--vrc-allocate-host-storage",
        source,
    ],
    text=True,
    capture_output=True,
    check=True,
).stdout
assert first == second
double = subprocess.run(double_command, text=True, capture_output=True, check=True).stdout
baseline = subprocess.run(
    baseline_command, text=True, capture_output=True, check=True
).stdout
assert first == double
assert allocated == allocated_twice

stable_ids = re.findall(r'use_id = "([^"]+)"', first)
assert stable_ids == sorted(stable_ids), stable_ids
assert stable_ids == [
    "backend:unity_buffer_id:3",
    "backend:unity_buffer_id:7",
    "backend:unity_buffer_pool_index:3",
    "backend:unity_buffer_pool_index:7",
    "backend:unity_buffer_token:3",
    "backend:unity_buffer_token:7",
    "function:alpha:region:0:block:0:operation:0:result:0",
    "function:alpha:region:0:block:0:operation:1:result:0",
    "function:alpha:region:0:block:0:operation:2:result:0",
    "function:alpha:region:0:block:0:operation:3:result:0",
    "function:alpha:region:0:block:0:operation:4:result:0",
    "function:alpha:region:0:block:0:operation:5:result:0",
    "function:alpha:region:0:block:0:operation:6:region:0:block:0:operation:0:result:0",
    "function:alpha:region:0:block:0:operation:6:region:1:block:0:operation:0:result:0",
    "function:zeta:region:0:block:0:operation:0:result:0",
    "function:zeta:region:0:block:0:operation:1:result:0",
]
assert "vrc.host_literal_inventory_version = 1 : i32" in first
assert "vrc.host_literal_keys" in first
assert "vrc.host_literal_uses" in first
assert 'origin = "backend"' in first
assert 'origin = "source"' in first
assert 'kind = "unity_buffer_id"' in first
assert 'kind = "unity_buffer_pool_index"' in first
assert 'kind = "unity_buffer_token"' in first
assert 'kind = "arith_constant"' in first
assert first.count('value = "0xFFFFFFFF"') == 1
assert 'value = "0x80000000"' in first
assert 'value = "0x1"' in first
assert 'value = "0x100000001"' in first
assert first.count('value = "0x2A"') == 1
assert 'literal_key = "SystemInt32:0x2A", logical_type = "i32"' in first
assert 'literal_key = "SystemInt32:0x2A", logical_type = "index"' in first
assert 'literal_key = "SystemInt32:0xFFFFFFFF", logical_type = "i8"' in first
assert 'literal_key = "SystemInt32:0xFFFFFFFF", logical_type = "i16"' in first
assert 'literal_key = "unsupported:i2:0x1", logical_type = "i2"' in first
assert 'context_free_recipes = [], key_id = "unsupported:i2:0x1"' in first
assert 'physical_type = "unsupported", supported = false' in first
assert '"rematerialize_i32"' in first
assert '"rematerialize_f32_bits"' in first
assert '"rematerialize_i1"' in first
assert '"rematerialize_i64"' in first

allocation_start = allocated.index("vrc.host_literal_allocations = [")
allocation_end = allocated.index("vrc.host_literal_inventory_version", allocation_start)
allocation_text = allocated[allocation_start:allocation_end]
expected_supported = first.count("supported = true")
assert allocation_text.count(
    'reason = "compatibility_resident_baseline"'
) == expected_supported
assert "unsupported:i2:0x1" not in allocation_text
entries = re.findall(r"\{([^{}]+)\}", allocation_text)
colors_by_type = {}
for entry in entries:
    physical_type = re.search(r'physical_type = "([^"]+)"', entry)
    color = re.search(r"color = (\d+) : i32", entry)
    if physical_type and color:
        colors_by_type.setdefault(physical_type.group(1), []).append(
            int(color.group(1))
        )
for colors in colors_by_type.values():
    assert sorted(colors) == list(range(len(colors))), colors
shared_minus_one = next(
    entry for entry in entries if 'key_id = "SystemInt32:0xFFFFFFFF"' in entry
)
assert "inventory_use_count = 3 : i64" in shared_minus_one

# Allocation annotates only module metadata and preserves executable IR.
assert (
    allocated[allocated.index("  func.func") :]
    == baseline[baseline.index("  func.func") :]
)

# Discovery annotates the module but does not rewrite executable operations.
assert first[first.index("  func.func") :] == baseline[baseline.index("  func.func") :]

duplicate_pool_id = source.read_text().replace("{id = 3 : i32}", "{id = 7 : i32}")
invalid = subprocess.run(
    [optimizer, "--vrc-discover-host-rematerialization", "-"],
    input=duplicate_pool_id,
    text=True,
    capture_output=True,
)
assert invalid.returncode != 0


def reject_allocation(text):
    rejected = subprocess.run(
        [optimizer, "--vrc-allocate-host-storage", "-"],
        input=text,
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0


reject_allocation(first.replace(
    "vrc.host_literal_inventory_version = 1 : i32",
    "vrc.host_literal_inventory_version = 2 : i32",
))
reject_allocation(first.replace("supported = true", "supported = false", 1))
reject_allocation(first.replace('physical_type = "SystemBoolean", ', "", 1))
reject_allocation(first.replace('value = "0x1"', 'value = "0x2"', 1))
reject_allocation(first.replace(
    'context_free_recipes = [], key_id = "unsupported:i2:0x1"',
    'context_free_recipes = ["arbitrary"], '
    'key_id = "unsupported:i2:0x1"',
))
reject_allocation(first.replace('origin = "backend", ', "", 1))
reject_allocation(
    'module attributes {vrc.host_literal_allocation_version = 1 : i32, '
    'vrc.host_literal_allocations = []} {}'
)
print("host_rematerialization_inventory=passed requests=16")
