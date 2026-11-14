import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-plan-static-unity-buffer-layout",
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
assert first.count("vrc.static_unity_buffer_layout_plan") == 2
assert "planned_count = 2 : i32" in first
assert "legacy_configured_count = 3 : i32" in first
assert "removable_legacy_count = 1 : i32" in first
assert "ready = true" in first
assert 'heap_slot = "__udoncl_buffer_1"' in first
assert 'public_access = "input", public_index = 0 : i32' in first
assert "planned_count = 1 : i32" in first
assert "unresolved_public_count = 1 : i32" in first
assert "ready = false" in first
print("static_unity_buffer_layout_plan=passed ready=1 blocked=1")
