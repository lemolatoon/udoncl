import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
fixed_source = Path(sys.argv[2])
generic_source = Path(sys.argv[3])
pipeline = "builtin.module(vrc-materialize-fixed-physical-releases,vrc-materialize-fixed-physical-releases)"


def run(source):
    result = subprocess.run(
        [optimizer, f"--pass-pipeline={pipeline}", source],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


fixed = run(fixed_source)
assert fixed == run(fixed_source)
assert "vrc.buffer_release" not in fixed
assert "arith.constant 0 : i32" in fixed
assert "materialized_count = 1 : i32" in fixed
assert "unresolved_count = 0 : i32" in fixed

generic = run(generic_source)
assert generic == run(generic_source)
assert "vrc.buffer_release" in generic
assert "materialized_count = 0 : i32" in generic
assert "unresolved_count = 1 : i32" in generic
print("fixed_physical_release_materialization=passed fixed=1 generic_unresolved=1")
