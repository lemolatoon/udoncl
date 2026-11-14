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
    "--vrc-plan-static-device-resource-context-specialization",
    source,
]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert "func.func private @allocate.__udoncl_ctx_1" in first
assert first.count("call @allocate()") == 1
assert first.count("call @allocate.__udoncl_ctx_1()") == 1
assert "cloned_function_count = 1 : i32" in first
assert "rewritten_call_count = 1 : i32" in first
assert "specialized_context_count = 1 : i32" in first
assert "context_count = 2 : i32" in first
assert "allocation_site_count = 2 : i32" in first
assert "singleton_site_count = 2 : i32" in first
assert "shared_site_count = 0 : i32" in first
assert "required_clone_count = 0 : i32" in first
print("static_allocation_context_specialization=passed clones=1 singleton=2")
