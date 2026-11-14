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
rerun = subprocess.run(
    command[:-1] + ["--vrc-propagate-device-resource-provenance", source],
    text=True,
    capture_output=True,
)
assert rerun.returncode == 0, (rerun.stdout, rerun.stderr)
assert rerun.stdout == first

releases = [
    line for line in first.splitlines() if "call @clReleaseMemObject" in line
]
assert len(releases) == 5, releases
for release in releases:
    marker = "resource_ids = ["
    resource_ids = release[release.index(marker) + len(marker) : release.index("]", release.index(marker))]
    assert resource_ids.count(": i32") == 1, release

assert len({release.split("resource_ids = [", 1)[1].split("]", 1)[0] for release in releases}) == 5, releases
print("static_device_resource_known_control_flow=passed cond=true switch=true loop=true")
