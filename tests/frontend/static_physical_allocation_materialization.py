import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
pipeline = (
    "builtin.module("
    "vrc-plan-static-physical-allocation-specialization,"
    "vrc-specialize-static-physical-allocation-paths,"
    "vrc-materialize-static-physical-allocations,"
    "vrc-materialize-static-physical-allocations)"
)
command = [optimizer, f"--pass-pipeline={pipeline}", "--split-input-file", source]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert first.count("vrc.static_physical_allocation_materialization") == 3
assert first.count("materialized_count = 2 : i32") == 1
assert first.count("materialized_count = 0 : i32") == 2
assert first.count('"vrc.assigned_buffer"') == 2
assert first.count("vrc.static_physical_allocation_assignment") == 0
assert "call @create(" in first
assert first.count("vrc.buffer_alloc") == 3
assert "arith.constant 0 : i32" in first
print("static_physical_allocation_materialization=passed assigned=2 legacy=3")
