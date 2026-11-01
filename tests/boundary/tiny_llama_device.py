"""Run Tiny LLaMA through the Device fragment backend."""

from pathlib import Path
import json
import subprocess
import sys


build, output = (Path(value).resolve() for value in sys.argv[1:])
root = Path(__file__).resolve().parents[2]
result = subprocess.run(
    [
        sys.executable,
        root / "examples/tiny-llama/run_device.py",
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
assert "tiny_llama_device_logits=64/64" in result.stdout
assert "prompts=4" in result.stdout
assert "generated=2,2,2,2,2,2,2,2" in result.stdout
assert "draws=84" in result.stdout
plan = json.loads((output / "onnx-plan.json").read_text(encoding="utf-8"))
assert plan["runtime"] == "udon-onnx"
assert plan["profile"] == "udon-tiny-llama-f32"
assert len(plan["passes"]) == 7
assert plan["nodes"][0]["provider"] == "UdonLlamaExecutionProvider"
print(result.stdout, end="")
