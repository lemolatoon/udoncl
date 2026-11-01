"""Run UdonBLAS SGEMM and quantized GEMV through the Device backend."""

from pathlib import Path
import subprocess
import sys


build, output = (Path(value).resolve() for value in sys.argv[1:])
root = Path(__file__).resolve().parents[2]
result = subprocess.run(
    [
        sys.executable,
        root / "libraries/udonblas/run_device.py",
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
assert "udonblas_sgemm_outputs=323" in result.stdout
assert "cases=5" in result.stdout
assert "max_cpu_gpu_error=0" in result.stdout
assert "q4_outputs=5" in result.stdout
assert "q4_max_cpu_gpu_error=0" in result.stdout
assert "draws=6" in result.stdout
print(result.stdout, end="")
