import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-discover-static-device-resources",
    "--vrc-plan-static-device-resource-context-specialization",
    "--vrc-specialize-static-allocation-contexts",
    "--vrc-discover-static-device-resources",
    "--vrc-infer-static-device-resource-requirements",
    "--vrc-propagate-device-resource-provenance",
    "--vrc-classify-annotated-public-device-resources",
    "--vrc-classify-static-public-device-resources",
    "--vrc-analyze-static-device-resource-persistence",
    "--vrc-build-static-device-resource-inventory",
    source,
]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
assert first == run()
header = first.splitlines()[0]
bindings = header.split("vrc.static_device_resource_public_bindings = ")[1]
bindings = bindings.split("schema_version")[0]
# Each UDONCL_PUBLIC variable is the one allocation context it holds, with
# the element type of its annotation; CL_MEM_READ_ONLY is an input U# writes,
# CL_MEM_WRITE_ONLY an output U# reads, indexed per direction in declaration
# order.
expected = [
    '{capacity_bytes = 128 : i64, data_type = "float32", logical_bytes = 128 : i64, name = "x", public_access = "input", public_index = 0 : i32, resource_id = 0 : i32, resource_kind = "allocation"}',
    '{capacity_bytes = 128 : i64, data_type = "float32", logical_bytes = 128 : i64, name = "y", public_access = "input", public_index = 1 : i32, resource_id = 1 : i32, resource_kind = "allocation"}',
    '{capacity_bytes = 128 : i64, data_type = "int32", logical_bytes = 128 : i64, name = "result", public_access = "output", public_index = 0 : i32, resource_id = 2 : i32, resource_kind = "allocation"}',
]
for entry in expected:
    assert entry in bindings, bindings
assert bindings.count("resource_kind") == 3, bindings
assert (
    "vrc.static_declared_public_resources = "
    "{resource_ids = array<i32: 0, 1, 2>"
) in header
# The inventory marks them public and the private scratch buffer not.
inventory = header.split("vrc.static_device_resource_inventory = ")[1]
assert inventory.count("public = true") == 3, inventory
assert "public = false, resource_id = 3 : i32" in inventory, inventory
print("static_annotated_public_resources=passed public=3")
