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
release = next(line for line in first.splitlines() if "call @clReleaseMemObject" in line)
assert "resource_ids = [0 : i32, 1 : i32]" in release
assert "may_be_null = true" in release
ordinary = next(line for line in first.splitlines() if "call @consumeOrdinaryObject" in line)
assert "vrc.static_device_operand_provenance" not in ordinary
print("static_device_resource_global_cells=passed contexts=2 fixed_point=true")
