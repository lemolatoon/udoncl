import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
pipeline = "builtin.module(vrc-materialize-static-physical-resolves,vrc-materialize-static-physical-resolves)"
command = [optimizer, f"--pass-pipeline={pipeline}", source]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
assert first == run()
assert "vrc.static_physical_resolve_materialization" in first
assert "materialized_count = 1 : i32" in first
assert "unresolved_count = 1 : i32" in first
assert first.count("call @udonclResolvePhysicalBuffer") == 1
assert first.count('"vrc.assigned_buffer"') == 1
assert "resource_id = 7 : i32" in first
assert "texture_id = 2 : i32" in first
assert "builtin.unrealized_conversion_cast" in first
assert '"udonhl.store"' in first
assert "arith.constant 0 : i32" in first
print("static_physical_resolve_materialization=passed materialized=1 unresolved=1")
