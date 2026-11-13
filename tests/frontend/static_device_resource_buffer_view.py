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
invalid = source.with_name("static_device_resource_buffer_view_invalid.mlir")
rejected = subprocess.run(
    command[:-1] + [invalid], text=True, capture_output=True
)
assert rejected.returncode != 0, rejected.stdout
for callee in (
    "udonclDeviceBufferView",
    "udononnxCopyBufferView",
    "udonclBufferViewResolve",
    "clReleaseMemObject",
):
    line = next(line for line in first.splitlines() if f"call @{callee}" in line)
    assert "resource_ids = [0 : i32]" in line, line

copy_lines = [
    line for line in first.splitlines()
    if "call @udononnxCopyBufferView" in line
]
resolve_lines = [
    line for line in first.splitlines()
    if "call @udonclBufferViewResolve" in line
]
assert len(copy_lines) == 2 and all(
    "resource_ids = [0 : i32]" in line for line in copy_lines), copy_lines
assert len(resolve_lines) == 2 and all(
    "resource_ids = [0 : i32]" in line for line in resolve_lines), resolve_lines

print(
    "static_device_resource_buffer_view=passed contexts=1 "
    "copy=strong direct_fields=checked"
)
