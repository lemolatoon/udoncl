import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [optimizer, "--vrc-classify-onnx-public-device-resources", source]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert "vrc.static_device_resource_public_bindings" in first
public = first.split("vrc.static_device_resource_public_bindings", 1)[1].split(
    "schema_version = 1 : i32", 1
)[0]
assert 'name = "tokens"' in public
assert 'onnx_index = 1 : i32' in public
assert 'public_access = "input"' in public
assert 'resource_id = 7 : i32' in public
assert 'name = "scores"' in public
assert 'public_access = "output"' in public
assert 'resource_id = 9 : i32' in public
assert 'resource_id = 11 : i32' not in public
assert 'name = "unused_input"' not in public
print("onnx_public_device_resources=passed public=2 private=1")
