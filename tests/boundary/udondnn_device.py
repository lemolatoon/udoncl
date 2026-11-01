"""Run UdonDNN convolution through the UdonCL Device fragment backend."""

from pathlib import Path
import re
import subprocess
import sys


build, output = (Path(value).resolve() for value in sys.argv[1:])
root = Path(__file__).resolve().parents[2]
result = subprocess.run(
    [
        sys.executable,
        root / "libraries/udondnn/run_device.py",
        "--build-dir",
        build,
        "--output",
        output,
    ],
    text=True,
    capture_output=True,
    timeout=600,
)
assert result.returncode == 0, (result.stdout, result.stderr)
assert "udondnn_convolution_outputs=171" in result.stdout
assert "convolution_cases=5" in result.stdout
assert "convolution_max_error=0" in result.stdout
assert "activation_outputs=512" in result.stdout
assert "activation_cases=8" in result.stdout
activation_error = re.search(r"activation_max_error=([0-9.eE+-]+)", result.stdout)
assert activation_error and float(activation_error.group(1)) < 0.0005
assert "draws=13" in result.stdout
print(result.stdout, end="")
