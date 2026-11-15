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


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
double = subprocess.run(double_command, text=True, capture_output=True, check=True).stdout
baseline = subprocess.run(
    baseline_command, text=True, capture_output=True, check=True
).stdout
assert first == double

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
print("host_rematerialization_inventory=passed requests=16")
