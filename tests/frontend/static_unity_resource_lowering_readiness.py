import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-assess-static-unity-resource-lowering",
    "--split-input-file",
    source,
]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert first.count("vrc.static_unity_resource_lowering_readiness") == 3
assert first.count("resource_operands_ready = true") == 2
assert first.count("pool_replacement_ready = true") == 1
assert first.count("pool_replacement_ready = false") == 2
assert "dynamic_allocation_count = 1 : i32" in first
assert "dynamic_allocation_count = 2 : i32" in first
assert "dynamic_release_count = 1 : i32" in first
assert "typed_copy_count = 1 : i32" in first
assert "unresolved_typed_transfer_operand_count = 2 : i32" in first
print("static_unity_resource_lowering_readiness=passed resource=2 pool=1 typed=1")
