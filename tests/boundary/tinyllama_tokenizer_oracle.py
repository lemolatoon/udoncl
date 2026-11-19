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
config = json.loads((bundle / "tokenizer_config.json").read_text(encoding="utf-8"))
cases = [
    "",
    "Hello",
    "日本語 \U0001f35c",
    " café\nline",
    "é",
    "  ",
    # Added tokens split normalized segments, also inside user text.
    "a</s>b",
    "<s>b",
    "x<unk>y</s>",
    "</s",
    "<|user|>\nHello</s>\n<|assistant|>\n",
]
oracle = [(utf16_units(text), tokenizer.encode(text).ids) for text in cases]

# Chat conversations: the streamed per-message encoding must equal Hugging
# Face's tokenization of the rendered template, including its BOS.
environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
template = environment.from_string(config["chat_template"])
roles = {"system": 0, "user": 1, "assistant": 2}
conversations = [
    [("user", "Hello")],
    [("user", "What is your name?")],
    [("system", "You are a friendly chatbot."), ("user", "こんにちは \U0001f35c")],
    [("user", "Hi"), ("assistant", "Hello! How can I help?"), ("user", "a</s>b  c")],
    [("user", "")],
]
chat_oracle = []
for conversation in conversations:
    rendered = template.render(
        messages=[{"role": role, "content": content} for role, content in conversation],
        add_generation_prompt=True,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    chat_oracle.append((conversation, tokenizer.encode(rendered).ids))

# Decoding: ordinary text, byte fallback split across tokens, invalid and
# truncated UTF-8 byte runs, and special tokens kept or skipped.
decode_cases = []
for text in (
    "Hello world",
    " leading",
    "日本語 \U0001f35c café",
    "line\nbreak",
    "é",
):
    decode_cases.append(tokenizer.encode(text).ids)
byte = {value: tokenizer.token_to_id(f"<0x{value:02X}>") for value in range(256)}
decode_cases += [
    [1, byte[0x20], 15043, 2],
    [byte[0xE3], 15043, byte[0x81], byte[0x82]],
    [byte[0xF0], byte[0x9F], byte[0x8D], byte[0x9C], 2, byte[0xC3]],
    [byte[0xE0], byte[0x80], byte[0x80], 13, byte[0xED], byte[0xA0], byte[0x80]],
    [1701, 29916, 432, 17204],
    [3421, 1024, 338, 19122, 29889, 2],
]
decode_oracle = []
for ids in decode_cases:
    for skip in (True, False):
        decode_oracle.append(
            (ids, skip, utf16_units(tokenizer.decode(ids, skip_special_tokens=skip)))
        )


def c_array(ctype: str, name: str, values) -> str:
    body = ", ".join(str(value) for value in values) or "0"
    return f"    static const {ctype} {name}[] = {{{body}}};"


lines = [
    "#include <stdint.h>",
    "#include <stdio.h>",
    "#include <udoncl/array.h>",
    "#include <udoncl/string.h>",
    '#include "tinyllama_tokenizer.h"',
    "",
    "static int check(const char *label, int index, udontokenizersStatus_t status,",
    "                 const int32_t *actual, int32_t actual_count,",
    "                 const int32_t *expected, int32_t expected_count) {",
    "  int32_t position;",
    "  if (status != UDONTOKENIZERS_STATUS_SUCCESS) {",
    '    fprintf(stderr, "%s %d: status %d\\n", label, index, (int)status);',
    "    return 1;",
    "  }",
    "  if (actual_count != expected_count) {",
    '    fprintf(stderr, "%s %d: count %d != %d\\n", label, index, (int)actual_count, (int)expected_count);',
    "    return 1;",
    "  }",
    "  for (position = 0; position < actual_count; ++position)",
    "    if (actual[position] != expected[position]) {",
    '      fprintf(stderr, "%s %d: element %d: %d != %d\\n", label, index, (int)position, (int)actual[position], (int)expected[position]);',
    "      return 1;",
    "    }",
    "  return 0;",
    "}",
    "",
    "int main(void) {",
    "  udontokenizers_model_t model;",
    "  udontokenizers_chat_template_t chat;",
    "  int failures = 0;",
    "  tinyllama_tokenizerInitializeModel(&model);",
    "  tinyllama_tokenizerInitializeChatTemplate(&chat);",
]
for case_index, (units, expected) in enumerate(oracle):
    lines.extend(
        [
            "  {",
            c_array("uint16_t", "input", units),
            c_array("int32_t", "expected", expected),
            f"    udon_string_native_t string = {{input, {len(units)}}};",
            "    int32_t values[128] = {0};",
            "    int32_t raw[128] = {0};",
            "    udon_int32_array_native_t output = {values, 128};",
            "    int32_t count = -1;",
            "    udontokenizersStatus_t status = udontokenizersEncode(&model, &string, &output, 128, &count);",
            f'    failures += check("encode", {case_index}, status, values, count, expected, {len(expected)});',
            "    status = udontokenizersEncodeRaw(&model, &string, raw, 128, &count);",
            f'    failures += check("encode-raw", {case_index}, status, raw, count, expected, {len(expected)});',
            "  }",
        ]
    )
for case_index, (conversation, expected) in enumerate(chat_oracle):
    lines.extend(
        [
            "  {",
            c_array("int32_t", "expected", expected),
            "    int32_t tokens[256];",
            "    int32_t count = -1;",
            "    udontokenizers_encoder_t encoder;",
            "    udontokenizersStatus_t status = udontokenizersEncoderBeginRaw(&model, &encoder, tokens, 256, 1);",
        ]
    )
    for message_index, (role, content) in enumerate(conversation):
        units = utf16_units(content)
        lines.extend(
            [
                c_array("uint16_t", f"content_{message_index}", units),
                f"    udon_string_native_t string_{message_index} = {{content_{message_index}, {len(units)}}};",
                "    if (status == UDONTOKENIZERS_STATUS_SUCCESS)",
                f"      status = udontokenizersEncoderAppendChatMessageRaw(&model, &chat, &encoder, {roles[role]}, &string_{message_index}, tokens);",
            ]
        )
    lines.extend(
        [
            "    if (status == UDONTOKENIZERS_STATUS_SUCCESS)",
            "      status = udontokenizersEncoderAppendChatGenerationPromptRaw(&model, &chat, &encoder, tokens);",
            "    if (status == UDONTOKENIZERS_STATUS_SUCCESS)",
            "      status = udontokenizersEncoderFinishRaw(&model, &encoder, tokens, &count);",
            f'    failures += check("chat", {case_index}, status, tokens, count, expected, {len(expected)});',
            "  }",
        ]
    )
for case_index, (ids, skip, expected) in enumerate(decode_oracle):
    lines.extend(
        [
            "  {",
            c_array("int32_t", "ids", ids),
            c_array("int32_t", "expected", expected),
            "    int32_t units[256];",
            "    int32_t count = -1;",
            "    int32_t index;",
            "    udontokenizers_decoder_t decoder;",
            f"    udontokenizersStatus_t status = udontokenizersDecoderBeginRaw(&model, &decoder, units, 256, {int(skip)});",
            f"    for (index = 0; status == UDONTOKENIZERS_STATUS_SUCCESS && index < {len(ids)}; ++index)",
            "      status = udontokenizersDecoderAppendRaw(&model, &decoder, ids[index], units);",
            "    if (status == UDONTOKENIZERS_STATUS_SUCCESS)",
            "      status = udontokenizersDecoderFinishRaw(&model, &decoder, units, &count);",
            f'    failures += check("decode", {case_index}, status, units, count, expected, {len(expected)});',
            "  }",
        ]
    )
lines.extend(["  return failures;", "}", ""])
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
subprocess.run([str(item) for item in command], check=True, cwd=output)
result = subprocess.run([str(executable)])
assert result.returncode == 0, f"TinyLlama tokenizer oracle failures: {result.returncode}"
print(
    "tinyllama_tokenizer_oracle=passed "
    f"encode={len(oracle)} chat={len(chat_oracle)} decode={len(decode_oracle)}"
)
