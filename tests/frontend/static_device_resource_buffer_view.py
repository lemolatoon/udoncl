import re
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


def run_source(path):
    result = subprocess.run(command[:-1] + [path], text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


def run():
    return run_source(source)


first = run()
assert first == run()
loop = run_source(source.with_name("static_device_resource_buffer_view_loop.mlir"))
loop_resolve = next(
    line for line in loop.splitlines()
    if "call @udonclBufferViewResolve" in line
)
assert re.search(r"resource_ids = \[\d+ : i32\]", loop_resolve), loop_resolve
assert "unknown = false" in loop_resolve, loop_resolve
partial_loop = source.with_name(
    "static_device_resource_buffer_view_loop_invalid.mlir"
)
rejected = subprocess.run(
    command[:-1] + [partial_loop], text=True, capture_output=True
)
assert rejected.returncode != 0, rejected.stdout
invalid = source.with_name("static_device_resource_buffer_view_invalid.mlir")
rejected = subprocess.run(
    command[:-1] + [invalid], text=True, capture_output=True
)
assert rejected.returncode != 0, rejected.stdout
partial_copy = source.with_name(
    "static_device_resource_buffer_view_memcpy_invalid.mlir"
)
rejected = subprocess.run(
    command[:-1] + [partial_copy], text=True, capture_output=True
)
assert rejected.returncode != 0, rejected.stdout
for callee in (
    "udonclDeviceBufferView",
    "udononnxCopyBufferView",
    "udonclBufferViewResolve",
    "clReleaseMemObject",
):
    line = next(line for line in first.splitlines() if f"call @{callee}" in line)
    assert re.search(r"resource_ids = \[\d+ : i32\]", line), line
    assert "unknown = false" in line, line

copy_lines = [
    line for line in first.splitlines()
    if "call @udononnxCopyBufferView" in line
    and "vrc.static_device_operand_provenance" in line
]
resolve_lines = [
    line for line in first.splitlines()
    if "call @udonclBufferViewResolve" in line
]
assert len(copy_lines) == 2 and all(
    re.search(r"resource_ids = \[\d+ : i32\]", line) for line in copy_lines
), copy_lines
assert len(resolve_lines) == 5 and all(
    re.search(r"resource_ids = \[\d+ : i32\]", line) for line in resolve_lines
), resolve_lines
hidden = first.split("func.func @HiddenDynamicBinding", 1)[1]
assert "vrc.static_device_operand_provenance" not in hidden, hidden

print(
    "static_device_resource_buffer_view=passed contexts=1 "
    "copy=strong direct_fields=checked"
)
