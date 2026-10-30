"""Run the trained MNIST CNN through the Device fragment backend."""

from pathlib import Path
import subprocess
import sys


build, output = (Path(value).resolve() for value in sys.argv[1:])
root = Path(__file__).resolve().parents[2]
result = subprocess.run(
    [
        sys.executable,
        root / "examples/mnist-cnn/run_device.py",
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
assert "mnist_device_predictions=10/10" in result.stdout
assert "max_cpu_gpu_error=" in result.stdout
print(result.stdout, end="")
