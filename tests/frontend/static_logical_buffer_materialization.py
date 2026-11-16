import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [optimizer, "--vrc-materialize-static-logical-buffers"]


def run(source_text=None, *, success=True, diagnostic=None):
    arguments = command if source_text is not None else [*command, source]
    result = subprocess.run(
        arguments, input=source_text, text=True, capture_output=True
    )
    assert (result.returncode == 0) == success, (result.stdout, result.stderr)
    if diagnostic is not None:
        assert diagnostic in result.stderr, result.stderr
    return result.stdout


first = run()
second = run()
assert first == second
assert "candidate_count = 3 : i32" in first
assert "materialized_count = 2 : i32" in first
assert "unresolved_count = 1 : i32" in first
assert first.count('"vrc.assigned_buffer"') == 2
assert "resource_id = 7 : i32, texture_id = 3 : i32" in first
assert "resource_id = 9 : i32, texture_id = 4 : i32" in first
assert first.count("vrc.buffer_alloc") == 1
assert "vrc.buffer_copy" in first
assert "arith.constant 0 : i32" in first

source_text = source.read_text(encoding="utf-8")
run(
    source_text.replace(
        "resource_id = 7 : i32, resource_kind = \"allocation\", "
        "texture_id = 3 : i32}",
        "resource_id = 7 : i32, resource_kind = \"allocation\", "
        "texture_id = 6 : i32}",
        1,
    ),
    success=False,
    diagnostic="reconciled logical buffer has no canonical layout",
)
print("static_logical_buffer_materialization=passed materialized=2 unresolved=1")
