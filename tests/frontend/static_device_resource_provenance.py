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
second = run()
assert first == second
rerun = subprocess.run(
    command[:-1] + ["--vrc-propagate-device-resource-provenance", source],
    text=True,
    capture_output=True,
)
assert rerun.returncode == 0, (rerun.stdout, rerun.stderr)
assert rerun.stdout == first
assert "vrc.static_device_resource_provenance" in first
assert "discovery_schema_version = 1 : i32" in first
assert first.count("vrc.static_device_argument_provenance") == 1, first
assert first.count("vrc.static_device_function_result_provenance") == 2, first
assert first.count("vrc.static_device_result_provenance") >= 5, first
assert first.count("vrc.static_device_operand_provenance") >= 8, first
identity_calls = [line for line in first.splitlines() if "call @identity" in line]
assert "occurrence = 0" in identity_calls[0] and "resource_ids = [0 : i32]" in identity_calls[0]
assert "resource_ids = [1 : i32]" not in identity_calls[0]
assert "occurrence = 1" in identity_calls[1] and "resource_ids = [1 : i32]" in identity_calls[1]
assert "resource_ids = [0 : i32]" not in identity_calls[1]
selects = [line for line in first.splitlines() if "arith.select" in line]
assert "resource_ids = [0 : i32, 1 : i32]" in selects[0]
assert "may_be_null = true" in selects[1] and "resource_ids = [0 : i32]" in selects[1]
allocation = next(line for line in first.splitlines() if "call @clCreateBuffer" in line)
assert "resource_ids = [0 : i32]" in allocation and "resource_ids = [1 : i32]" in allocation
assert "resource_ids = [0 : i32, 1 : i32]" not in allocation
unknown_identity = identity_calls[-1]
assert "unknown = true" in unknown_identity
assert first.count("callee = @make, caller = @Run") >= 4, first

loop_source = source.with_name("static_device_resource_provenance_loop.mlir")
loop = subprocess.run(command[:-1] + [loop_source], text=True, capture_output=True)
assert loop.returncode == 0, (loop.stdout, loop.stderr)
release = next(line for line in loop.stdout.splitlines() if "call @clReleaseMemObject" in line)
assert "resource_ids = [0 : i32, 1 : i32]" in release, release
print("static_device_resource_provenance=passed contexts=2 join=finite")
