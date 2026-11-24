import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [optimizer, "--vrc-resolve-static-transfer-bindings", source]


def run():
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


first = run()
second = run()
assert first == second
assert "candidate_count = 6 : i32" in first
# A handle that may be null resolves like any other: the transfer checks its
# handles before it moves data. Only the non-allocation stays unresolved.
assert "resolved_count = 5 : i32" in first
assert "unresolved_count = 1 : i32" in first
assert "requires_context_specialization_count = 1 : i32" in first
assert first.count("vrc.static_transfer_binding_snapshot") == 5
assert 'transfer_kind = "write"' in first
assert 'transfer_kind = "read"' in first
assert 'transfer_kind = "copy"' in first
assert 'access = "read", operand_index = 1 : i32' in first
assert 'access = "write", operand_index = 2 : i32' in first
assert "resource_id = 9 : i32" in first
assert "resource_id = 2 : i32" in first
assert "requires_context_specialization = true" in first
print("static_transfer_bindings=passed resolved=5 unresolved=1 specialization=1")
