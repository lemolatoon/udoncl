import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-discover-static-device-resources",
    "--vrc-infer-static-device-resource-requirements",
    "--vrc-propagate-device-resource-provenance",
    "--vrc-classify-static-public-device-resources",
    source,
]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
assert first == run()
assert 'primitive = "udonclDeviceBufferCreate"' in first
assert "capacity_bytes = 96 : i64" in first
create = next(
    line for line in first.splitlines() if "call @udonclDeviceBufferCreate" in line
)
resolve = next(
    line for line in first.splitlines() if "call @udonclDeviceBufferGetRaw" in line
)
assert "vrc.static_device_operand_provenance" in create
assert "resource_ids = [0 : i32]" in create
assert "vrc.static_public_resource_binding" in create
assert "vrc.static_device_operand_provenance" in resolve
assert "resource_ids = [0 : i32]" in resolve
public = first.split("vrc.static_device_resource_public_bindings", 1)[1].split(
    "schema_version = 1 : i32", 1
)[0]
assert 'name = "values"' in public
assert "logical_bytes = 0 : i64" in public
assert 'public_access = "input"' in public
assert "public_index = 0 : i32" in public
assert "resource_id = 0 : i32" in public
print("static_device_resource_device_create=passed resource=0 bytes=96")
