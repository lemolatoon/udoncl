"""Native test of the TinyLlama conversation text layer against Hugging Face.

The token-array conversation Host is replaced by a small fake of its internal
C interface, so this test isolates conversation_text.c: chat-template prompt
construction into the bounded prompt workspace (in one call and in bounded
slices), capacity failure, prompt readback, and streamed decoding of a running
generation to UTF-16. After every generated token the streamed text must be
exactly what Hugging Face decodes from the tokens that can no longer change.
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
CONTEXT_CAPACITY = 64
GENERATED_CAPACITY = 32


def chat_tokens(system: str, user: str) -> list[int]:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    rendered = template.render(
        messages=messages, add_generation_prompt=True, eos_token="</s>", bos_token="<s>"
    )
    return tokenizer.encode(rendered).ids


byte = {value: tokenizer.token_to_id(f"<0x{value:02X}>") for value in range(256)}
byte_ids = set(byte.values())
special_ids = {0, 1, 2}


def stable_units(ids: list[int], count: int) -> list[int]:
    """Hugging Face text of the first `count` tokens that cannot change.

    A trailing byte-fallback run (special tokens are skipped and do not end
    it) is still open: Hugging Face decodes it as a whole once it ends.
    """
    stable = count
    while stable > 0 and (ids[stable - 1] in byte_ids or ids[stable - 1] in special_ids):
        stable -= 1
    return utf16_units(tokenizer.decode(ids[:stable], skip_special_tokens=True))


cases = [
    ("", "What is your name?"),
    ("You are a friendly chatbot.", "こんにちは \U0001f35c"),
    ("", "a</s>b <s> é"),
]
for system, user in cases:
    assert len(chat_tokens(system, user)) <= CONTEXT_CAPACITY
long_user = "word " * 80
assert len(chat_tokens("", long_user)) > CONTEXT_CAPACITY
# More pre-merge symbols than any prompt of the Context Capacity can have.
huge_user = "x y " * 600

replies = [
    tokenizer.encode("My name is Sarah. 日本 \U0001f35c é!", add_special_tokens=False).ids + [2],
    # Byte fallback split by a special token, an invalid run, and a run that
    # is still incomplete when the generation ends.
    [byte[0xE3], 2, byte[0x81], byte[0x82], 15043, byte[0xE3], byte[0x81], byte[0xFF], 29871, byte[0xF0], byte[0x9F]],
    [byte[0x20], 15043, 2],
]
for reply in replies:
    assert 0 < len(reply) <= GENERATED_CAPACITY, len(reply)

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
    "udononnxStatus_t tinyllama_begin_chat_prompt(void);",
    "udononnxStatus_t tinyllama_continue_chat_prompt(udon_string_t, udon_string_t, uint32_t);",
    "uint32_t tinyllama_chat_prompt_pending(void);",
    "uint32_t tinyllama_prompt_token_count(void);",
    "udononnxStatus_t tinyllama_read_prompt_tokens(udon_int32_array_t);",
    "uint32_t tinyllama_streamed_text_length(void);",
    "udononnxStatus_t tinyllama_read_streamed_text(udon_int32_array_t, uint32_t, uint32_t);",
    "uint32_t tinyllama_generated_text_length(void);",
    "udononnxStatus_t tinyllama_read_generated_text(udon_int32_array_t);",
    "int32_t tinyllama_last_tokenizer_status(void);",
    "",
    "static int32_t prompt[TINYLLAMA_CONTEXT_CAPACITY];",
    "static uint32_t prompt_count, committed;",
    "static int32_t generated[TINYLLAMA_GENERATED_CAPACITY];",
    "static uint32_t generated_count, serial;",
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
    "udononnxStatus_t tinyllamaConversationGenerationProgress(",
    "    const int32_t **tokens, uint32_t *count, uint32_t *current, int32_t *is_completed) {",
    "  *tokens = generated;",
    "  *count = generated_count;",
    "  *current = serial;",
    "  *is_completed = completed;",
    "  return UDONONNX_STATUS_SUCCESS;",
    "}",
    "",
    "static int32_t read[TINYLLAMA_CONTEXT_CAPACITY];",
    "static udon_int32_array_native_t read_array = {read, TINYLLAMA_CONTEXT_CAPACITY};",
    "static int32_t text[512];",
    "static int32_t chunk[512];",
    "static udon_int32_array_native_t text_array = {text, 512};",
    "static udon_int32_array_native_t chunk_array = {chunk, 512};",
    "static udon_int32_array_native_t short_array = {text, 3};",
    "",
    "/* Polls the stream once and appends the newly final units to `text`. */",
    "static int poll(uint32_t *streamed, const int32_t *expected, uint32_t expected_count) {",
    "  uint32_t length = tinyllama_streamed_text_length();",
    "  uint32_t index;",
    "  EXPECT(length == expected_count);",
    "  EXPECT(length >= *streamed);",
    "  EXPECT(tinyllama_read_streamed_text(&chunk_array, *streamed, length - *streamed) == UDONONNX_STATUS_SUCCESS);",
    "  for (index = *streamed; index < length; ++index)",
    "    text[index] = chunk[index - *streamed];",
    "  for (index = 0; index < length; ++index)",
    "    EXPECT(text[index] == expected[index]);",
    "  *streamed = length;",
    "  return 0;",
    "}",
    "",
    "int main(void) {",
    "  uint32_t index;",
    "  uint32_t streamed;",
    "  EXPECT(tinyllama_chat_prompt_pending() == 0);",
    "  EXPECT(tinyllama_streamed_text_length() == 0);",
    "  EXPECT(tinyllama_generated_text_length() == 0);",
    "  EXPECT(tinyllama_read_generated_text(&text_array) == UDONONNX_STATUS_INVALID_STATE);",
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
            "    uint32_t calls = 0;",
            "    EXPECT(tinyllama_set_chat_prompt(&system, &user) == UDONONNX_STATUS_SUCCESS);",
            f"    EXPECT(tinyllama_prompt_token_count() == {len(expected)});",
            "    EXPECT(tinyllama_read_prompt_tokens(&read_array) == UDONONNX_STATUS_SUCCESS);",
            f"    for (index = 0; index < {len(expected)}; ++index)",
            "      EXPECT(read[index] == expected[index]);",
            "    /* The same prompt in slices of five work units. */",
            "    EXPECT(tinyllama_continue_chat_prompt(&system, &user, 5) == UDONONNX_STATUS_INVALID_STATE);",
            "    EXPECT(tinyllama_begin_chat_prompt() == UDONONNX_STATUS_SUCCESS);",
            "    EXPECT(tinyllama_chat_prompt_pending() == 1);",
            "    EXPECT(tinyllama_prompt_token_count() == 0);",
            "    EXPECT(tinyllama_continue_chat_prompt(&system, &user, 0) == UDONONNX_STATUS_INVALID_VALUE);",
            "    while (tinyllama_chat_prompt_pending()) {",
            "      EXPECT(tinyllama_continue_chat_prompt(&system, &user, 5) == UDONONNX_STATUS_SUCCESS);",
            "      ++calls;",
            "    }",
            f"    EXPECT(calls > {(len(system_units) + len(user_units)) // 5});",
            f"    EXPECT(tinyllama_prompt_token_count() == {len(expected)});",
            "    EXPECT(tinyllama_read_prompt_tokens(&read_array) == UDONONNX_STATUS_SUCCESS);",
            f"    for (index = 0; index < {len(expected)}; ++index)",
            "      EXPECT(read[index] == expected[index]);",
            "  }",
        ]
    )
long_units = utf16_units(long_user)
huge_units = utf16_units(huge_user)
lines.extend(
    [
        "  {",
        f"    static const uint16_t long_units[] = {{{c_values(long_units)}}};",
        f"    static const uint16_t huge_units[] = {{{c_values(huge_units)}}};",
        "    udon_string_native_t empty = {long_units, 0};",
        f"    udon_string_native_t user = {{long_units, {len(long_units)}}};",
        f"    udon_string_native_t huge = {{huge_units, {len(huge_units)}}};",
        "    uint32_t before = committed;",
        "    EXPECT(tinyllama_set_chat_prompt(&empty, &user) == UDONONNX_STATUS_CAPACITY_EXCEEDED);",
        "    EXPECT(tinyllama_last_tokenizer_status() == 4);",
        "    EXPECT(committed == before && tinyllama_prompt_token_count() == 0);",
        "    EXPECT(tinyllama_chat_prompt_pending() == 0);",
        "    EXPECT(tinyllama_set_chat_prompt(&empty, &huge) == UDONONNX_STATUS_CAPACITY_EXCEEDED);",
        "    EXPECT(tinyllama_last_tokenizer_status() == 4);",
        "    EXPECT(committed == before && tinyllama_prompt_token_count() == 0);",
        "    EXPECT(tinyllama_chat_prompt_pending() == 0);",
        "  }",
    ]
)
for reply_index, reply in enumerate(replies):
    final = utf16_units(tokenizer.decode(reply, skip_special_tokens=True))
    lines.extend(
        [
            "  {",
            f"    static const int32_t reply[] = {{{c_values(reply)}}};",
            f"    static const int32_t final[] = {{{c_values(final)}}};",
            "    /* A new generation starts: nothing is final yet. */",
            "    ++serial;",
            "    completed = 0;",
            "    generated_count = 0;",
            "    streamed = 0;",
            "    EXPECT(tinyllama_streamed_text_length() == 0);",
        ]
    )
    for count in range(1, len(reply) + 1):
        stable = stable_units(reply, count)
        assert final[: len(stable)] == stable
        lines.extend(
            [
                "    {",
                f"      static const int32_t stable[] = {{{c_values(stable)}}};",
                f"      generated[{count - 1}] = reply[{count - 1}];",
                f"      generated_count = {count};",
                f"      if (poll(&streamed, stable, {len(stable)})) return 1;",
                "      /* Polling again without a new token changes nothing. */",
                f"      if (poll(&streamed, stable, {len(stable)})) return 1;",
                "      EXPECT(tinyllama_generated_text_length() == 0);",
                "      EXPECT(tinyllama_read_generated_text(&text_array) == UDONONNX_STATUS_INVALID_STATE);",
                "    }",
            ]
        )
    lines.extend(
        [
            "    completed = 1;",
            f"    if (poll(&streamed, final, {len(final)})) return 1;",
            f"    EXPECT(tinyllama_generated_text_length() == {len(final)});",
            "    EXPECT(tinyllama_read_streamed_text(&chunk_array, streamed + 1, 0) == UDONONNX_STATUS_INVALID_VALUE);",
            "    EXPECT(tinyllama_read_streamed_text(&chunk_array, 0, streamed + 1) == UDONONNX_STATUS_INVALID_VALUE);",
            "    EXPECT(tinyllama_read_streamed_text(&chunk_array, streamed, 0) == UDONONNX_STATUS_SUCCESS);",
            f"    EXPECT(tinyllama_read_generated_text(&short_array) == {'UDONONNX_STATUS_INVALID_VALUE' if len(final) > 3 else 'UDONONNX_STATUS_SUCCESS'});",
            "    EXPECT(tinyllama_read_generated_text(&chunk_array) == UDONONNX_STATUS_SUCCESS);",
            f"    for (index = 0; index < {len(final)}; ++index)",
            "      EXPECT(chunk[index] == final[index]);",
            "    EXPECT(tinyllama_last_tokenizer_status() == 0);",
            "  }",
        ]
    )
lines.extend(
    [
        "  /* A reset leaves no text. */",
        "  ++serial;",
        "  completed = 0;",
        "  generated_count = 0;",
        "  EXPECT(tinyllama_streamed_text_length() == 0);",
        "  EXPECT(tinyllama_generated_text_length() == 0);",
        "  /* A token outside the vocabulary fails the text, not the tokens",
        "   * decoded before it. */",
        "  ++serial;",
        "  generated[0] = 15043;",
        "  generated[1] = 32000;",
        "  generated[2] = 15043;",
        "  generated_count = 3;",
        "  completed = 1;",
        "  EXPECT(tinyllama_streamed_text_length() == 5);",
        "  EXPECT(tinyllama_last_tokenizer_status() == 5);",
        "  EXPECT(tinyllama_generated_text_length() == 0);",
        "  EXPECT(tinyllama_read_generated_text(&text_array) == UDONONNX_STATUS_EXECUTION_FAILURE);",
        "  EXPECT(tinyllama_last_tokenizer_status() == 5);",
        "  EXPECT(tinyllama_read_streamed_text(&text_array, 0, 0) == UDONONNX_STATUS_EXECUTION_FAILURE);",
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
    f"chat_cases={len(cases)} replies={len(replies)} "
    f"streamed_tokens={sum(len(reply) for reply in replies)}"
)
