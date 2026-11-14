import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-build-static-device-resource-inventory",
    "--vrc-allocate-unity-textures",
    "--split-input-file",
    source,
]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
assert first == run()
onnx, udoncl = first.split("// -----")
assert onnx.index("resource_id = 2 : i32") < onnx.index("resource_id = 5 : i32")
assert onnx.index("resource_id = 5 : i32") < onnx.index("resource_id = 9 : i32")
assert 'binding_capacity_bytes = 32 : i64' in onnx
assert 'logical_bytes = 24 : i64' in onnx
assert 'public_access = "input"' in onnx
public_entry = onnx[onnx.index('binding_capacity_bytes') : onnx.index('resource_id = 5')]
assert "persistent = true" in public_entry
for resource_id, texture_id in [(2, 1), (5, 2), (9, 3)]:
    marker = f"resource_id = {resource_id} : i32"
    position = onnx.rindex(marker)
    assignment = onnx[max(0, position - 180) : position + 180]
    assert f"texture_id = {texture_id} : i32" in assignment
assert 'public = false' in udoncl
assert 'resource_id = 4 : i32' in udoncl
assert 'texture_id = 1 : i32' in udoncl
print("static_device_resource_inventory=passed onnx=3 udoncl=1")
