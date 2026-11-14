import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
pipeline = (
    "builtin.module("
    "vrc-plan-static-physical-allocation-specialization,"
    "vrc-specialize-static-physical-allocation-paths,"
    "vrc-specialize-static-physical-allocation-paths)"
)
command = [optimizer, f"--pass-pipeline={pipeline}", "--split-input-file", source]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
for name in [
    "create.__udoncl_phys_r0_f0",
    "physical.__udoncl_phys_r0_f1",
    "create.__udoncl_phys_r1_f0",
    "physical.__udoncl_phys_r1_f1",
]:
    assert name in first
assert first.count("vrc.static_physical_allocation_assignment") == 2
assert first.count("specialized_context_count = 2 : i32") == 1
assert "cloned_function_count = 4 : i32" in first
assert "rewritten_call_count = 4 : i32" in first
assert "annotated_sink_count = 2 : i32" in first
assert "call @create(" in first
assert first.count('vrc.async_effect = "may_suspend"') == 3
assert first.count("vrc.llvm_function") == 1
assert first.count("specialized_context_count = 0 : i32") == 2
assert first.count("blocker_count = 1 : i32") >= 2
print("static_physical_allocation_path_specialization=passed contexts=2 clones=4")
