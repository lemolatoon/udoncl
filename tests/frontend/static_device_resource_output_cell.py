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
result = subprocess.run(command, text=True, capture_output=True)
assert result.returncode == 0, (result.stdout, result.stderr)

resolve_view = next(
    line
    for line in result.stdout.splitlines()
    if "call @udonclBufferViewResolve" in line
)
resolve_physical = next(
    line
    for line in result.stdout.splitlines()
    if "call @udonclResolvePhysicalBuffer" in line
)
view_ids = re.search(r"resource_ids = \[([^]]+)\]", resolve_view)
physical_ids = re.search(r"resource_ids = \[([^]]+)\]", resolve_physical)
assert view_ids, resolve_view
assert physical_ids, resolve_physical
assert view_ids.group(1) == physical_ids.group(1), (
    resolve_view,
    resolve_physical,
)
print("static_device_resource_output_cell=passed wrappers=2 status=refined")
