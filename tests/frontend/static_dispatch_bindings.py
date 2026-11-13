import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [optimizer, "--vrc-resolve-static-kernel-bindings", source]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert "snapshot_count = 3 : i32" in first
assert "candidate_count = 5 : i32" in first
assert "unresolved_count = 2 : i32" in first
assert first.count("vrc.static_kernel_binding_snapshot") == 3
dispatch = next(
    line
    for line in first.splitlines()
    if "vrc.dispatch " in line
    and "kernel_id = 7 : i32" in line
    and "vrc.static_kernel_binding_snapshot" in line
)
assert "kernel_id = 7 : i32" in dispatch
assert 'kind = "static_tensor"' in dispatch
assert "operand_index = 0 : i32, resource_id = 9 : i32" in dispatch
assert 'kind = "allocation"' in dispatch
assert "operand_index = 1 : i32, resource_id = 2 : i32" in dispatch
assert "descriptor_operand_index = 0 : i32, type = i32" in dispatch
physical_dispatch = next(
    line
    for line in first.splitlines()
    if "kernel_id = 9 : i32" in line
    and "vrc.static_kernel_binding_snapshot" in line
)
assert "operand_index = 0 : i32, resource_id = 1 : i32" in physical_dispatch
assert "operand_index = 1 : i32, resource_id = 0 : i32" in physical_dispatch
assert "operand_index = 2 : i32, resource_id = 2 : i32" in physical_dispatch
logical_dispatch = next(
    line
    for line in first.splitlines()
    if "kernel_id = 10 : i32" in line
    and "vrc.static_kernel_binding_snapshot" in line
)
assert 'kind = "logical_buffer"' in logical_dispatch
assert "operand_index = 0 : i32, resource_id = 5 : i32" in logical_dispatch
assert not any(
    "kernel_id = 11 : i32" in line
    and "vrc.static_kernel_binding_snapshot" in line
    for line in first.splitlines()
)
print("static_dispatch_bindings=passed snapshots=3 unresolved=2")
