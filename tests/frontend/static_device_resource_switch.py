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
switch_start = first.index("cf.switch")
switch = first[switch_start : first.index("\n  ^bb1", switch_start)]
assert "index = 1 : i32" in switch and "resource_ids = [0 : i32]" in switch
assert "index = 2 : i32" in switch and "resource_ids = [1 : i32]" in switch
assert "index = 3 : i32" in switch and switch.count(
    "resource_ids = [0 : i32]"
) == 2
release = next(
    line for line in first.splitlines() if "call @clReleaseMemObject" in line
)
assert "resource_ids = [0 : i32, 1 : i32]" in release, release
print("static_device_resource_switch=passed contexts=2 join=finite")
