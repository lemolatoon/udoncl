"""Run UdonBLAS Level-1 and Level-3 kernels on Device."""

from pathlib import Path
import subprocess
import sys


build, output = (Path(value).resolve() for value in sys.argv[1:])
root = Path(__file__).resolve().parents[2]
udonblas_header = (root / "libraries/udonblas/include/udonblas.h").read_text(
    encoding="utf-8"
)
assert "udonblasSetStream(udonblasHandle_t handle," in udonblas_header
assert "udonblasGetStream(udonblasHandle_t handle," in udonblas_header
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
assert "sgemm_m1n10k144_outputs=10" in result.stdout
assert "sgemm_m1n10k144_tiles=9" in result.stdout
assert "sgemm_m1n10k144_max_cpu_gpu_error=0" in result.stdout
assert "sgemv_outputs=28" in result.stdout
assert "sgemv_cases=4" in result.stdout
assert "sgemv_max_cpu_gpu_error=0" in result.stdout
assert "saxpy_outputs=280" in result.stdout
assert "saxpy_cases=4" in result.stdout
assert "saxpy_max_cpu_gpu_error=0" in result.stdout
assert "draws=58" in result.stdout
print(result.stdout, end="")
