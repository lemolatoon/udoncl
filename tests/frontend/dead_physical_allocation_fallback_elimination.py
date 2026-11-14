import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
pipeline = "builtin.module(vrc-eliminate-dead-physical-allocation-fallbacks,vrc-eliminate-dead-physical-allocation-fallbacks)"


def run():
    result = subprocess.run(
        [optimizer, f"--pass-pipeline={pipeline}", source],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
assert first == run()
assert "@allocator_sink" not in first
assert "@allocator_wrapper" not in first
assert "vrc.buffer_alloc" not in first
assert "@dead_dispatch" in first
assert first.count("vrc.dispatch") >= 1
assert "vrc.static_kernel_binding_snapshot" in first
assert "vrc.static_kernel_bindings" in first
print("dead_physical_allocation_fallback_elimination=passed")
