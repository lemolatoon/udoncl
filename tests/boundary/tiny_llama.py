"""Run the complete tiny LLaMA C graph through the native adapter."""

from pathlib import Path
import subprocess
import sys


build, output = (Path(value).resolve() for value in sys.argv[1:])
root = Path(__file__).resolve().parents[2]
result = subprocess.run(
    [
        sys.executable,
        root / "examples/tiny-llama/run.py",
        "--build-dir",
        build,
        "--output",
        output,
    ],
    text=True,
    capture_output=True,
    timeout=300,
)
assert result.returncode == 0, (result.stdout, result.stderr)
for marker in [
    "architecture=llama-rmsnorm-rope-mha-swiglu",
    "generated=2,2,2,2,2,2,2,2",
    "python_oracle_max_error=",
    "native_udonhl_roundtrip=passed",
]:
    assert marker in result.stdout, (marker, result.stdout)
print(result.stdout, end="")
