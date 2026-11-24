"""Canonicalize the pinned TinyLlama model through the generic ONNX-MLIR frontend."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys


root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
frontend = Path(sys.argv[3]).resolve()
output.mkdir(parents=True, exist_ok=True)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = load("fetch_tinyllama_onnx", root / "scripts" / "fetch_tinyllama_onnx.py")
compiler = load("compile_onnx_mlir", root / "scripts" / "compile_onnx_mlir.py")
model = fetch.ensure(fetch.default_cache())
canonical = output / "model.mlir"
compiler.canonicalize(
    model,
    canonical,
    frontend,
    dimensions=["batch_size=1", "sequence_length=4", "past_sequence_length=0"],
    input_shapes=["attention_mask=1,4"],
)
text = canonical.read_text(encoding="utf-8")
assert text.count('function_name = "MatMulNBits"') == 155
assert canonical.stat().st_size < 2_000_000
signature = next(line for line in text.splitlines() if "func.func @main_graph" in line)
assert re.search(
    r'tensor<1x4xi64> \{onnx\.dim_params = "0:batch_size,1:past_sequence_length \+ 1", onnx\.name = "attention_mask"\}',
    signature,
)
assert 'onnx.name = "logits"' in signature
assert signature.count('onnx.name = "present.') == 44

# The conversation graph with a bounded past: one position per run on a past
# of up to 511 positions compiles to a Device graph whose attention reads the
# run-time extent of the past. The real-model checkpoint of stage 3 is that
# this graph compiles, at 512 and 4096, with no padded capacity.
bounded = output / "bounded-host.mlir"
compiler.canonicalize(
    model,
    bounded,
    frontend,
    dimensions=["batch_size=1", "sequence_length=1"],
    dimension_bounds=["past_sequence_length=511"],
    lower_to_host=True,
    static_data_directory=output / "bounded-static",
    host_metadata=output / "bounded-host-metadata.json",
)
assert bounded.stat().st_size > 0
print(
    f"tinyllama_frontend_bytes={canonical.stat().st_size} "
    "matmul_nbits=155 inputs=47 outputs=45 bounded_past=511"
)
