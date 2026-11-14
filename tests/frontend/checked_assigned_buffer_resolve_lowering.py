import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
pipeline = "builtin.module(vrc-lower-checked-assigned-buffer-resolves,vrc-lower-checked-assigned-buffer-resolves)"


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
assert "checked_assigned_buffer_resolve" not in first
assert '"udonhl.ptrtoint"' in first
assert '"udonhl.load"' in first
assert '"udonhl.store"' in first
assert first.count("cf.cond_br") == 2
assert "arith.constant -38 : i32" in first
assert "builtin.unrealized_conversion_cast" in first
assert "cf.br" in first
print("checked_assigned_buffer_resolve_lowering=passed")
