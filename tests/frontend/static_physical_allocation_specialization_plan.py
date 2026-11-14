import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-plan-static-physical-allocation-specialization",
    "--split-input-file",
    source,
]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert first.count("vrc.static_physical_allocation_specialization_plan") == 3
assert "blocker_count = 0 : i32" in first
assert "candidate_count = 2 : i32" in first
assert "planned_count = 2 : i32" in first
assert "resource_id = 0 : i32" in first
assert "resource_id = 1 : i32" in first
assert "texture_id = 1 : i32" in first
assert "texture_id = 2 : i32" in first
assert "callee = @create, caller = @root, occurrence = 0 : i32" in first
assert "callee = @physical, caller = @create, occurrence = 0 : i32" in first
assert first.count("blocker_count = 1 : i32") == 2
assert first.count("candidate_count = 1 : i32") == 2
assert first.count("planned_count = 0 : i32") == 2
print("static_physical_allocation_specialization_plan=passed planned=2 blocked=2")
