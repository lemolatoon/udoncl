import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [optimizer, "--vrc-analyze-static-device-resource-persistence", source]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
assert first == run()
attribute = first.split("vrc.static_device_resource_persistence", 1)[1]
attribute = attribute.split("resources = [", 1)[1].split(
    "], schema_version = 1 : i32", 1
)[0]
for resource_id, persistent, escaped in [(0, False, False), (1, True, False), (2, True, True)]:
    marker = f"resource_id = {resource_id} : i32"
    end = attribute.index(marker)
    entry = attribute[max(0, end - 300) : end + len(marker)]
    assert f"persistent = {'true' if persistent else 'false'}" in entry
    assert f"global_escape = {'true' if escaped else 'false'}" in entry
assert "observed_roots = [@Initialize, @Run]" in attribute
assert "observed_roots = [@Repeat]" in attribute
print("static_device_resource_persistence=passed transient=1 persistent=2")
