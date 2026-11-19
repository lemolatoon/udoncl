"""Native test of the TinyLlama conversation text layer against Hugging Face.

The token-array conversation Host is replaced by a small fake of its internal
C interface, so this test isolates conversation_text.c: chat-template prompt
construction into the bounded prompt workspace, capacity failure, prompt
readback, and decoding of a completed generation to UTF-16.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from jinja2.sandbox import ImmutableSandboxedEnvironment
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


def c_values(values) -> str:
    return ", ".join(str(value) for value in values) or "0"


fetch = load("fetch_tinyllama_tokenizer", root / "scripts/fetch_tinyllama_tokenizer.py")
table_compiler = load("compile_tokenizer_tables", root / "scripts/compile_tokenizer_tables.py")
bundle = fetch.ensure(fetch.default_cache())
generated = output / "generated"
table_compiler.compile_bundle(bundle, generated, symbol="tinyllama_tokenizer")
tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
config = json.loads((bundle / "tokenizer_config.json").read_text(encoding="utf-8"))
template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
    config["chat_template"]
)


def chat_tokens(system: str, user: str) -> list[int]:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    rendered = template.render(
        messages=messages, add_generation_prompt=True, eos_token="</s>", bos_token="<s>"
    )
    return tokenizer.encode(rendered).ids


cases = [
    ("", "What is your name?"),
    ("You are a friendly chatbot.", "こんにちは \U0001f35c"),
]
long_user = "word " * 80
assert len(chat_tokens("", long_user)) > 64
reply_ids = tokenizer.encode("My name is Sarah. 日本 \U0001f35c", add_special_tokens=False).ids + [2]
reply_units = utf16_units(tokenizer.decode(reply_ids, skip_special_tokens=True))

lines = [
    "#include <stdint.h>",
    "#include <stdio.h>",
    "#include <udoncl/array.h>",
    "#include <udoncl/onnx_types.h>",
    "#include <udoncl/string.h>",
    '#include "conversation_host.h"',
    "",
    "#define EXPECT(x) do { if (!(x)) { fprintf(stderr, \"line %d: %s\\n\", __LINE__, #x); return 1; } } while (0)",
    "",
    "udononnxStatus_t tinyllama_set_chat_prompt(udon_string_t, udon_string_t);",
    "uint32_t tinyllama_prompt_token_count(void);",
    "udononnxStatus_t tinyllama_read_prompt_tokens(udon_int32_array_t);",
    "uint32_t tinyllama_generated_text_length(void);",
    "udononnxStatus_t tinyllama_read_generated_text(udon_int32_array_t);",
    "int32_t tinyllama_last_tokenizer_status(void);",
    "",
    "static int32_t prompt[TINYLLAMA_CONTEXT_CAPACITY];",
    "static uint32_t prompt_count, committed;",
    "static int32_t generated[TINYLLAMA_GENERATED_CAPACITY];",
    "static uint32_t generated_count;",
    "static int completed;",
    "",
    "udononnxStatus_t tinyllamaConversationPreparePrompt(int32_t **workspace) {",
    "  prompt_count = 0;",
    "  *workspace = prompt;",
    "  return UDONONNX_STATUS_SUCCESS;",
    "}",
    "udononnxStatus_t tinyllamaConversationCommitPrompt(uint32_t count) {",
    "  if (count > TINYLLAMA_CONTEXT_CAPACITY) return UDONONNX_STATUS_CAPACITY_EXCEEDED;",
    "  prompt_count = count;",
    "  ++committed;",
    "  return UDONONNX_STATUS_SUCCESS;",
    "}",
    "udononnxStatus_t tinyllamaConversationPrompt(const int32_t **tokens, uint32_t *count) {",
    "  *tokens = prompt;",
    "  *count = prompt_count;",
    "  return UDONONNX_STATUS_SUCCESS;",
    "}",
    "udononnxStatus_t tinyllamaConversationGenerated(const int32_t **tokens, uint32_t *count) {",
    "  if (!completed) return UDONONNX_STATUS_INVALID_STATE;",
    "  *tokens = generated;",
    "  *count = generated_count;",
    "  return UDONONNX_STATUS_SUCCESS;",
    "}",
    "",
    "int main(void) {",
    "  int32_t read[TINYLLAMA_CONTEXT_CAPACITY];",
    "  udon_int32_array_native_t read_array = {read, TINYLLAMA_CONTEXT_CAPACITY};",
    "  int32_t text[512];",
    "  udon_int32_array_native_t text_array = {text, 512};",
    "  udon_int32_array_native_t short_array = {text, 3};",
    "  uint32_t index;",
]
for case_index, (system, user) in enumerate(cases):
    expected = chat_tokens(system, user)
    system_units = utf16_units(system)
    user_units = utf16_units(user)
    lines.extend(
        [
            "  {",
            f"    static const uint16_t system_units[] = {{{c_values(system_units)}}};",
            f"    static const uint16_t user_units[] = {{{c_values(user_units)}}};",
            f"    static const int32_t expected[] = {{{c_values(expected)}}};",
            f"    udon_string_native_t system = {{system_units, {len(system_units)}}};",
            f"    udon_string_native_t user = {{user_units, {len(user_units)}}};",
            "    EXPECT(tinyllama_set_chat_prompt(&system, &user) == UDONONNX_STATUS_SUCCESS);",
            f"    EXPECT(tinyllama_prompt_token_count() == {len(expected)});",
            "    EXPECT(tinyllama_read_prompt_tokens(&read_array) == UDONONNX_STATUS_SUCCESS);",
            f"    for (index = 0; index < {len(expected)}; ++index)",
            "      EXPECT(read[index] == expected[index]);",
            "  }",
        ]
    )
long_units = utf16_units(long_user)
lines.extend(
    [
        "  {",
        f"    static const uint16_t long_units[] = {{{c_values(long_units)}}};",
        f"    udon_string_native_t empty = {{long_units, 0}};",
        f"    udon_string_native_t user = {{long_units, {len(long_units)}}};",
        "    uint32_t before = committed;",
        "    EXPECT(tinyllama_set_chat_prompt(&empty, &user) == UDONONNX_STATUS_CAPACITY_EXCEEDED);",
        "    EXPECT(tinyllama_last_tokenizer_status() == 4);",
        "    EXPECT(committed == before && tinyllama_prompt_token_count() == 0);",
        "  }",
        "  EXPECT(tinyllama_generated_text_length() == 0);",
        "  EXPECT(tinyllama_read_generated_text(&text_array) == UDONONNX_STATUS_INVALID_STATE);",
        "  {",
        f"    static const int32_t reply[] = {{{c_values(reply_ids)}}};",
        f"    static const int32_t expected[] = {{{c_values(reply_units)}}};",
        f"    for (index = 0; index < {len(reply_ids)}; ++index)",
        "      generated[index] = reply[index];",
        f"    generated_count = {len(reply_ids)};",
        "    completed = 1;",
        f"    EXPECT(tinyllama_generated_text_length() == {len(reply_units)});",
        "    EXPECT(tinyllama_read_generated_text(&short_array) == UDONONNX_STATUS_INVALID_VALUE);",
        "    EXPECT(tinyllama_read_generated_text(&text_array) == UDONONNX_STATUS_SUCCESS);",
        f"    for (index = 0; index < {len(reply_units)}; ++index)",
        "      EXPECT(text[index] == expected[index]);",
        "    generated[0] = 32000;",
        "    EXPECT(tinyllama_read_generated_text(&text_array) == UDONONNX_STATUS_EXECUTION_FAILURE);",
        "    EXPECT(tinyllama_last_tokenizer_status() == 5);",
        "  }",
        "  return 0;",
        "}",
        "",
    ]
)
output.mkdir(parents=True, exist_ok=True)
harness = output / "conversation_text_test.c"
harness.write_text("\n".join(lines), encoding="utf-8", newline="\n")
executable = output / ("conversation_text_test.exe" if os.name == "nt" else "conversation_text_test")
sources = [
    harness,
    root / "examples/onnx-transformer-tiny/conversation_text.c",
    generated / "tinyllama_tokenizer.c",
    root / "libraries/udontokenizers/src/udontokenizers.c",
]
includes = [
    root / "third_party/OpenCL-Headers",
    root / "include",
    root / "libraries/udontokenizers/include",
    root / "examples/onnx-transformer-tiny",
    generated,
]
if compiler_path.name.lower() in ("cl", "cl.exe"):
    command = [compiler_path, "/nologo", "/std:c11", "/DCL_TARGET_OPENCL_VERSION=120"]
    command += [f"/I{path}" for path in includes]
    command += sources + [f"/Fe:{executable}"]
else:
    command = [compiler_path, "-std=c11", "-DCL_TARGET_OPENCL_VERSION=120"]
    for path in includes:
        command += ["-I", path]
    command += sources + ["-o", executable]
subprocess.run([str(item) for item in command], check=True, cwd=output)
result = subprocess.run([str(executable)])
assert result.returncode == 0, f"conversation text layer failed: {result.returncode}"
print(
    "tinyllama_conversation_text=passed "
    f"chat_cases={len(cases)} reply_units={len(reply_units)}"
)
