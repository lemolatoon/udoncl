import importlib.util
import json
from pathlib import Path
import sys


root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = load(
    "fetch_tinyllama_tokenizer", root / "scripts/fetch_tinyllama_tokenizer.py"
)
compiler = load(
    "compile_tokenizer_tables", root / "scripts/compile_tokenizer_tables.py"
)

bundle = fetch.ensure(fetch.default_cache())
first = output / "first"
second = output / "second"
compiler.compile_bundle(bundle, first, symbol="tinyllama_tokenizer")
compiler.compile_bundle(bundle, second, symbol="tinyllama_tokenizer")

for name in ("tinyllama_tokenizer.h", "tinyllama_tokenizer.c", "manifest.json"):
    assert (first / name).read_bytes() == (second / name).read_bytes(), name

manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
assert manifest["vocab_count"] == 32_000
assert manifest["merge_count"] == 61_249
assert manifest["byte_token_count"] == 256
assert manifest["normalizer"] == {
    "prepend": "▁",
    "replace_from": " ",
    "replace_to": "▁",
}
assert manifest["special_token_ids"] == {"bos": 1, "eos": 2, "unk": 0}
print(
    "tinyllama_tokenizer_tables=passed "
    f"vocab={manifest['vocab_count']} merges={manifest['merge_count']}"
)
