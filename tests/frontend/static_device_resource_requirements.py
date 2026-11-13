import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-discover-static-device-resources",
    "--vrc-infer-static-device-resource-requirements",
    source,
]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert "vrc.static_device_resource_requirements" in first
for resource_id, capacity, flags in [(0, 32, 4), (1, 12, 2), (2, 64, 1)]:
    marker = f"capacity_bytes = {capacity} : i64"
    position = first.index(marker)
    entry = first[position : position + 240]
    assert f"memory_flags = {flags} : i64" in entry
    assert f"resource_id = {resource_id} : i32" in entry
    assert 'resource_kind = "allocation"' in entry
    assert 'format = "rgba8_unorm"' in entry
    assert 'usage = "storage"' in entry
print("static_device_resource_requirements=passed resources=3")
