import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-discover-static-device-resources",
    "--vrc-propagate-device-resource-provenance",
    source,
]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
assert first == run()
rerun = subprocess.run(
    command[:-1] + ["--vrc-propagate-device-resource-provenance", source],
    text=True,
    capture_output=True,
)
assert rerun.returncode == 0, (rerun.stdout, rerun.stderr)
assert rerun.stdout == first
release_lines = [line for line in first.splitlines() if "call @clReleaseMemObject" in line]
assert "resource_ids = [0 : i32]" in release_lines[0] and "resource_ids = [1 : i32]" not in release_lines[0]
assert "resource_ids = [1 : i32]" in release_lines[1] and "resource_ids = [0 : i32]" not in release_lines[1]
assert "resource_ids = [1 : i32]" in release_lines[2] and "resource_ids = [0 : i32]" not in release_lines[2]
assert "resource_ids = [0 : i32, 1 : i32]" in release_lines[3]
assert "resource_ids = [0 : i32]" in release_lines[4]
assert "may_be_null = true" in release_lines[4]
assert "resource_ids = [2 : i32, 3 : i32]" in release_lines[5]
passthrough = [line for line in first.splitlines() if "call @passthrough" in line]
assert "resource_ids = [0 : i32, 1 : i32]" in passthrough[0]
assert "unknown = true" in passthrough[0]
assert "resource_ids = [4 : i32]" in passthrough[1]
assert "unknown = true" in passthrough[1]
helper = next(line for line in first.splitlines() if "func.func @through_cell" in line)
assert "resource_ids = [0 : i32]" in helper and "resource_ids = [1 : i32]" in helper
print("static_device_resource_local_cells=passed contexts=2 strong_update=true")
