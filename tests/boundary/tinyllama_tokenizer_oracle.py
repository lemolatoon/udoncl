import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from tokenizers import Tokenizer


root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
compiler_path = Path(sys.argv[3]).resolve()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utf16_units(text: str) -> list[int]:
    encoded = text.encode("utf-16-le")
    return [
        int.from_bytes(encoded[index : index + 2], "little")
        for index in range(0, len(encoded), 2)
    ]


fetch = load(
    "fetch_tinyllama_tokenizer", root / "scripts/fetch_tinyllama_tokenizer.py"
)
table_compiler = load(
    "compile_tokenizer_tables", root / "scripts/compile_tokenizer_tables.py"
)
bundle = fetch.ensure(fetch.default_cache())
generated = output / "generated"
table_compiler.compile_bundle(bundle, generated, symbol="tinyllama_tokenizer")

tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
cases = ["", "Hello", "日本語 🍜", " café\nline", "e\u0301", "  "]
oracle = [(utf16_units(text), tokenizer.encode(text).ids) for text in cases]

lines = [
    "#include <stdint.h>",
    "#include <stdio.h>",
    "#include <udoncl/array.h>",
    "#include <udoncl/string.h>",
    '#include "tinyllama_tokenizer.h"',
    "",
    "int main(void) {",
    "  udontokenizers_model_t model;",
    "  tinyllama_tokenizerInitializeModel(&model);",
]
for case_index, (units, expected) in enumerate(oracle):
    unit_values = ", ".join(f"0x{value:04x}" for value in units) or "0"
    expected_values = ", ".join(str(value) for value in expected)
    lines.extend(
        [
            f"  static const uint16_t input_{case_index}[] = {{{unit_values}}};",
            f"  static const int32_t expected_{case_index}[] = {{{expected_values}}};",
            f"  udon_string_native_t string_{case_index} = {{input_{case_index}, {len(units)}}};",
            f"  int32_t values_{case_index}[128] = {{0}};",
            f"  udon_int32_array_native_t output_{case_index} = {{values_{case_index}, 128}};",
            f"  int32_t count_{case_index} = -1;",
            f"  if (udontokenizersEncode(&model, &string_{case_index}, &output_{case_index}, 128, &count_{case_index}) != UDONTOKENIZERS_STATUS_SUCCESS)",
            f"    return {case_index * 3 + 1};",
            f"  if (count_{case_index} != {len(expected)}) return {case_index * 3 + 2};",
            f"  for (int32_t index = 0; index < count_{case_index}; ++index)",
            f"    if (values_{case_index}[index] != expected_{case_index}[index]) return {case_index * 3 + 3};",
        ]
    )
lines.extend(["  return 0;", "}", ""])
output.mkdir(parents=True, exist_ok=True)
harness = output / "oracle.c"
harness.write_text("\n".join(lines), encoding="utf-8", newline="\n")
executable = output / ("oracle.exe" if os.name == "nt" else "oracle")

if compiler_path.name.lower() in ("cl", "cl.exe"):
    command = [
        compiler_path,
        "/nologo",
        "/std:c11",
        f"/I{root / 'include'}",
        f"/I{root / 'libraries/udontokenizers/include'}",
        f"/I{generated}",
        harness,
        generated / "tinyllama_tokenizer.c",
        root / "libraries/udontokenizers/src/udontokenizers.c",
        f"/Fe:{executable}",
    ]
else:
    command = [
        compiler_path,
        "-std=c11",
        "-I",
        root / "include",
        "-I",
        root / "libraries/udontokenizers/include",
        "-I",
        generated,
        harness,
        generated / "tinyllama_tokenizer.c",
        root / "libraries/udontokenizers/src/udontokenizers.c",
        "-o",
        executable,
    ]
subprocess.run([str(item) for item in command], check=True)
result = subprocess.run([str(executable)])
assert result.returncode == 0, f"TinyLlama tokenizer oracle case failed: {result.returncode}"
print(f"tinyllama_tokenizer_oracle=passed cases={len(cases)}")
