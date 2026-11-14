import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [optimizer, "--vrc-materialize-static-dispatch-resources", source]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert "candidate_dispatch_count = 3 : i32" in first
assert "snapshotted_dispatch_count = 2 : i32" in first
assert "unresolved_dispatch_count = 1 : i32" in first
assert "materialized_operand_count = 4 : i32" in first
assert "unresolved_operand_count = 0 : i32" in first
assert first.count('"vrc.assigned_buffer"') == 4
assert first.count("resource_id = 9 : i32, texture_id = 1 : i32") == 2
assert first.count("resource_id = 2 : i32, texture_id = 2 : i32") == 2
assert first.count('"vrc.static_buffer"') == 1
assert "kernel_id = 10 : i32" in first
print("static_dispatch_resource_materialization=passed operands=4 unresolved=1")
