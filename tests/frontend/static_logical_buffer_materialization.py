import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [optimizer, "--vrc-materialize-static-logical-buffers", source]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert "candidate_count = 3 : i32" in first
assert "materialized_count = 2 : i32" in first
assert "unresolved_count = 1 : i32" in first
assert first.count('"vrc.assigned_buffer"') == 2
assert "resource_id = 7 : i32, texture_id = 3 : i32" in first
assert "resource_id = 9 : i32, texture_id = 4 : i32" in first
assert first.count("vrc.buffer_alloc") == 1
assert "vrc.buffer_copy" in first
assert "arith.constant 0 : i32" in first
print("static_logical_buffer_materialization=passed materialized=2 unresolved=1")
