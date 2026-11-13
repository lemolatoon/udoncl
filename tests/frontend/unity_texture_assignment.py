import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [optimizer, "--vrc-allocate-unity-textures", source]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert "vrc.unity_texture_assignments" in first
assert first.count('resource_kind = "allocation"') == 6
assignments = first.split("vrc.unity_texture_assignments = ", 1)[1]
expected = [
    (0, 1, 16, 'public_access = "input"'),
    (1, 2, 32, 'public_access = "output"'),
    (2, 3, 64, None),
]
positions = []
for resource_id, texture_id, capacity, public_access in expected:
    marker = (
        f"capacity_bytes = {capacity} : i64, format = \"rgba8_unorm\""
    )
    position = assignments.index(marker)
    positions.append(position)
    entry = assignments[position : position + 260]
    assert f"resource_id = {resource_id} : i32" in entry
    assert f"texture_id = {texture_id} : i32" in entry
    assert 'resource_kind = "allocation"' in entry
    if public_access:
        assert public_access in entry
assert positions == sorted(positions)
assert 'resource_kind = "static_tensor"' not in assignments.split("}", 1)[0]
print("unity_texture_assignment=passed assignments=3")
