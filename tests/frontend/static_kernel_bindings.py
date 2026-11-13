import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-discover-static-device-resources",
    "--vrc-propagate-device-resource-provenance",
    "--vrc-resolve-static-kernel-bindings",
    source,
]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert "vrc.static_kernel_bindings" in first
assert "snapshot_count = 2 : i32" in first
calls = [
    line
    for line in first.splitlines()
    if "call @enqueue" in line and "vrc.static_kernel_binding_snapshot" in line
]
assert len(calls) == 2, first
assert "kernel_id = 7 : i32" in calls[0]
assert 'profile = "fixture"' in calls[0]
assert "operand_index = 0 : i32, resource_id = 0 : i32" in calls[0]
assert "operand_index = 1 : i32, resource_id = 1 : i32" in calls[0]
assert "operand_index = 1 : i32, resource_id = 2 : i32" in calls[1]
assert "operand_index = 1 : i32, resource_id = 2 : i32" not in calls[0]
assert "operand_index = 2 : i32, type = i32" in calls[0]
assert "%c3_i32" in calls[0]
assert "%c5_i32" in calls[1]
print("static_kernel_bindings=passed snapshots=2")
