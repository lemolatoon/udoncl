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
assert "pooling_outputs=41" in result.stdout
assert "pooling_cases=6" in result.stdout
pooling_error = re.search(r"pooling_max_error=([0-9.eE+-]+)", result.stdout)
assert pooling_error and float(pooling_error.group(1)) < 0.0005
assert "softmax_outputs=80" in result.stdout
assert "softmax_cases=6" in result.stdout
softmax_error = re.search(r"softmax_max_error=([0-9.eE+-]+)", result.stdout)
assert softmax_error and float(softmax_error.group(1)) < 0.0005
assert "batchnorm_outputs=68" in result.stdout
assert "batchnorm_cases=4" in result.stdout
batchnorm_error = re.search(r"batchnorm_max_error=([0-9.eE+-]+)", result.stdout)
assert batchnorm_error and float(batchnorm_error.group(1)) < 0.0005
assert "optensor_outputs=60" in result.stdout
assert "optensor_cases=4" in result.stdout
optensor_error = re.search(r"optensor_max_error=([0-9.eE+-]+)", result.stdout)
assert optensor_error and float(optensor_error.group(1)) < 0.0005
assert "int4_outputs=64" in result.stdout
int4_error = re.search(r"int4_max_error=([0-9.eE+-]+)", result.stdout)
assert int4_error and float(int4_error.group(1)) < 0.0005
assert "draws=36" in result.stdout
print(result.stdout, end="")
