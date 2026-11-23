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
release = next(line for line in first.splitlines() if "call @clReleaseMemObject" in line)
assert "resource_ids = [0 : i32, 1 : i32]" in release
assert "may_be_null = true" in release
ordinary = next(line for line in first.splitlines() if "call @consumeOrdinaryObject" in line)
assert "vrc.static_device_operand_provenance" not in ordinary
# The module records what each global cell may hold across every context.
summary = next(line for line in first.splitlines()
               if "vrc.static_device_global_cell_provenance" in line)
for cell in [
    "{global = @forwarded, may_be_null = true, resource_ids = [0 : i32, 1 : i32], unknown = false}",
    "{global = @primary, may_be_null = true, resource_ids = [0 : i32, 1 : i32], unknown = false}",
    "{global = @single, may_be_null = true, resource_ids = [2 : i32], unknown = false}",
]:
    assert cell in summary, summary
assert "@ordinary_pointer" not in summary.split("vrc.static_device_global_cell_provenance")[1].split("schema_version")[0]
print("static_device_resource_global_cells=passed contexts=2 fixed_point=true")
