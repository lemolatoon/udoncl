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
copy_lines = [
    line for line in first.splitlines()
    if "call @udononnxCopyBufferView" in line
    and "vrc.static_device_operand_provenance" in line
]
resolve_lines = [
    line for line in first.splitlines()
    if "call @udonclBufferViewResolve" in line
]
assert len(copy_lines) == 2, copy_lines
assert len(resolve_lines) == 2, resolve_lines
for lines in (copy_lines, resolve_lines):
    assert any("resource_ids = [0 : i32]" in line for line in lines), lines
    assert any("resource_ids = [1 : i32]" in line for line in lines), lines
release = next(
    line for line in first.splitlines()
    if "call @clReleaseMemObject" in line
)
assert "resource_ids = [1 : i32]" in release, release

invalid = source.with_name("static_device_resource_aggregate_pointer_invalid.mlir")
rejected = subprocess.run(command[:-1] + [invalid], text=True, capture_output=True)
assert rejected.returncode != 0

print(
    "static_device_resource_aggregate_pointer=passed "
    "index=1 stride=88 cfg_join=checked status_refinement=checked "
    "cross_export=checked"
)
