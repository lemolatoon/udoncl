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
adopt = next(line for line in first.splitlines() if "call @udonclDeviceBufferAdopt" in line)
assert "resource_ids = [0 : i32]" in adopt
assert "resource_ids = [1 : i32]" in adopt
assert "resource_ids = [2 : i32]" in adopt

get_raw = [line for line in first.splitlines() if "call @udonclDeviceBufferGetRaw" in line]
destroy = [line for line in first.splitlines() if "call @udonclDeviceBufferDestroy" in line]
release = [line for line in first.splitlines() if "call @clReleaseMemObject" in line]
assert len(get_raw) == len(destroy) == len(release) == 3
for context_id, lines in enumerate(zip(get_raw, destroy, release)):
    for line in lines:
        assert f"resource_ids = [{context_id} : i32]" in line
        assert "may_be_null = true" in line

print("static_device_resource_device_memory=passed contexts=3 handles=local,array,struct")
