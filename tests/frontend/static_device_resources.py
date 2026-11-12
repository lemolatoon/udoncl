import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-discover-static-device-resources",
    source,
]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert first.count("context_id =") == 8, first
assert first.count("root = @RunBranch") == 2, first
assert first.count("root = @RunDirect") == 2, first
assert first.count("root = @RunNested") == 1, first
assert first.count("root = @RunOther") == 1, first
assert first.count("root = @RunRepeat") == 2, first
assert "callee = @allocate, caller = @RunRepeat, occurrence = 0 : i32" in first
assert "callee = @allocate, caller = @RunRepeat, occurrence = 1 : i32" in first
assert "function = @RunDirect, occurrence = 0 : i32, primitive = \"clCreateBuffer\"" in first
assert (
    "function = @RunDirect, occurrence = 0 : i32, "
    'primitive = "udonclCreateBufferForQueue"'
) in first
assert "callee = @nested_allocate, caller = @RunNested, occurrence = 0 : i32" in first
assert "callee = @allocate, caller = @nested_allocate, occurrence = 0 : i32" in first
assert first.count("vrc.static_device_allocation_site") == 3, first
assert first.count("vrc.static_device_call_site") == 8, first
assert "schema_version = 1 : i32" in first
print("static_device_resource_discovery=passed contexts=8 deterministic=true")
