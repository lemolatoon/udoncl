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
for callee in (
    "udononnxCopyBufferView",
    "udonclBufferViewResolve",
    "clReleaseMemObject",
):
    line = next(line for line in first.splitlines() if f"call @{callee}" in line)
    assert "resource_ids = [0 : i32]" in line, line

print("static_device_resource_aggregate_pointer=passed index=1 stride=88")
