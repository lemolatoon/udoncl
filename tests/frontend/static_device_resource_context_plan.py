import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-plan-static-device-resource-context-specialization",
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
assert first.count("vrc.static_device_resource_context_plan") == 2
assert "context_count = 3 : i32" in first
assert "allocation_site_count = 2 : i32" in first
assert "singleton_site_count = 1 : i32" in first
assert "shared_site_count = 1 : i32" in first
assert "required_clone_count = 1 : i32" in first
assert 'disposition = "clone_per_context"' in first
assert 'resource_ids = array<i32: 0, 1>' in first
assert 'disposition = "singleton"' in first
assert 'resource_ids = array<i32: 2>' in first
assert "context_count = 0 : i32" in first
assert "allocation_site_count = 0 : i32" in first
print("static_device_resource_context_plan=passed shared=1 clone=1 empty=passed")
