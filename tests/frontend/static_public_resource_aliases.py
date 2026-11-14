import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-resolve-static-public-resource-aliases",
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
assert first.count("vrc.static_public_resource_aliases") == 2
assert "resolved_count = 2 : i32" in first
assert "resolved_count = 0 : i32" in first
assert 'public_access = "input", public_index = 7 : i32' in first
assert 'resource_id = 9 : i32, resource_kind = "allocation", texture_id = 1 : i32' in first
assert 'public_access = "output", public_index = 0 : i32' in first
assert 'resource_id = 2 : i32, resource_kind = "allocation", texture_id = 4 : i32' in first
alias_sections = [
    section.split("resolved_count", 1)[0]
    for section in first.split("vrc.static_public_resource_aliases")[1:]
]
assert all("resource_id = 5 : i32" not in section for section in alias_sections)
print("static_public_resource_aliases=passed resolved=2 private=1 empty=passed")
