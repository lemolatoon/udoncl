import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-reconcile-static-unity-buffers",
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
assert first.count("vrc.static_unity_buffer_reconciliation") == 2
assert "configured_count = 3 : i32" in first
assert "resolved_alias_count = 1 : i32" in first
assert "unresolved_public_count = 1 : i32" in first
assert "unresolved_private_count = 1 : i32" in first
assert "unused_alias_count = 1 : i32" in first
assert (
    'buffer_id = 1 : i32, heap_slot = "resolved_input", public_access = "input", '
    'public_index = 7 : i32, resolution = "static_alias", resource_id = 9 : i32, '
    'resource_kind = "allocation", texture_id = 1 : i32'
) in first
assert (
    'buffer_id = 3 : i32, heap_slot = "legacy_same_capacity", public_access = '
    '"input", public_index = 8 : i32, resolution = "legacy_public"'
) in first
assert first.index("buffer_id = 1 : i32") < first.index("buffer_id = 3 : i32")
assert "configured_count = 1 : i32" in first
assert 'resolution = "legacy_public"' in first
print("static_unity_buffer_reconciliation=passed resolved=1 legacy=3")
