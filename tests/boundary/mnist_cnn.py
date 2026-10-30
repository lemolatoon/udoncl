"""Exercise the trained MNIST CNN through the native UdonHL adapter."""

from pathlib import Path
import subprocess
import sys


build, output = (Path(value).resolve() for value in sys.argv[1:])
root = Path(__file__).resolve().parents[2]
result = subprocess.run(
    [
        sys.executable,
        root / "examples/mnist-cnn/run.py",
        "--build-dir",
        build,
        "--output",
        output,
    ],
    text=True,
    capture_output=True,
    timeout=180,
)
assert result.returncode == 0, (result.stdout, result.stderr)
assert "mnist_fixture_accuracy=10/10" in result.stdout
assert "native_udonhl_roundtrip=passed" in result.stdout
print(result.stdout, end="")
