"""Generate the trained MNIST ONNX fixture used by this CMake example."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    generator = Path(__file__).resolve().parents[1] / "mnist-cnn" / "generate_onnx.py"
    runpy.run_path(str(generator), run_name="__main__")
