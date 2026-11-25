"""Native test of the minimal TinyLlama chat Host (examples/tinyllama-chat).

tests/runtime/tinyllama_chat_host.c runs tinyllama_chat.c against a fake
OpenCL implementation and a fake UdonONNX session, with the real tokenizer
tables compiled from the pinned tokenizer files. This script computes what
the Host must do from Hugging Face `tokenizers` and the chat template, as
tests/boundary/tinyllama_chat_oracle.py does for the model:

* the first message is BOS + the chat template's user turn + the
  generation prompt; a later message is the template text after the
  previous reply, without BOS;
* a message runs after the tokens that close the previous reply (its last
  token, never run by the generation, then EOS when the reply was cut off);
* every run binds the next position; a reply stops at EOS, at the reply
  limit or when the capacity is full, and its last token is not run;
* the streamed text after every token is what Hugging Face decodes from the
  tokens that can no longer change, and the whole reply at the end;
* a message beyond the capacity fails with CAPACITY_EXCEEDED and changes
  nothing, and Dispose + Initialize starts a new conversation at position 0.

usage: tinyllama_chat_host.py ROOT OUTPUT C_COMPILER
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

# The Host is compiled at a small capacity and reply limit so that the test
# reaches both.
CAPACITY = 160
MAX_NEW_TOKENS = 8


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
EOS = tokenizer.token_to_id(config["eos_token"])
BOS = tokenizer.token_to_id(config["bos_token"])


def render(messages: list[dict[str, str]]) -> str:
    return template.render(
        messages=messages,
        add_generation_prompt=True,
        bos_token=config["bos_token"],
        eos_token=config["eos_token"],
        unk_token=config["unk_token"],
    )


def turn_text(user: str) -> str:
    """The template text after the EOS that ends a reply, through the
    generation prompt after the next user message (as the chat oracle)."""
    sentinel = "udoncl-reply"
    rendered = render(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": sentinel},
            {"role": "user", "content": user},
        ]
    )
    tail = rendered[rendered.index(sentinel) + len(sentinel) :]
    assert tail.startswith(config["eos_token"]), tail
    return tail[len(config["eos_token"]) :]


def text_tokens(text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


byte_ids = {tokenizer.token_to_id(f"<0x{value:02X}>") for value in range(256)}
special_ids = {0, BOS, EOS}


def stable_units(ids: list[int], count: int) -> list[int]:
    """Hugging Face text of the first `count` tokens that cannot change: a
    trailing byte-fallback run is still open (special tokens do not end
    it)."""
    stable = count
    while stable > 0 and (ids[stable - 1] in byte_ids or ids[stable - 1] in special_ids):
        stable -= 1
    return utf16_units(tokenizer.decode(ids[:stable], skip_special_tokens=True))


def simulate(turns: list[dict]) -> list[dict]:
    """What the Host runs and replies per turn, from position 0."""
    position, last = 0, None
    results = []
    for index, turn in enumerate(turns):
        if index == 0:
            message = tokenizer.encode(render([{"role": "user", "content": turn["user"]}])).ids
            assert message[0] == BOS
            closing = []
        else:
            message = text_tokens(turn_text(turn["user"]))
            assert message and message[0] != BOS
            closing = [last] + ([EOS] if last != EOS else [])
        runs = closing + message
        assert position + len(runs) <= CAPACITY, (turn["user"], position, len(runs))
        start = position
        position += len(runs)
        limit = min(MAX_NEW_TOKENS, CAPACITY - position + 1)
        reply = []
        for token in turn["script"]:
            reply.append(token)
            if token == EOS or len(reply) == limit:
                break
        assert reply and (reply[-1] == EOS or len(reply) == limit), turn
        runs += reply[:-1]
        position += len(reply) - 1
        last = reply[-1]
        stable = [stable_units(reply, count) for count in range(len(reply))]
        final = utf16_units(tokenizer.decode(reply, skip_special_tokens=True))
        for units in stable:
            assert final[: len(units)] == units
        results.append(
            {
                **turn,
                "message": message,
                "runs": runs,
                "first_position": start,
                "limit": limit,
                "reply": reply,
                "stable": stable,
                "final": final,
                "position": position,
            }
        )
    return results


reply_text = text_tokens("My name is Sarah.")
long_reply = text_tokens("日本 \U0001f35c é and then a few more words")
assert len(long_reply) > MAX_NEW_TOKENS and EOS not in long_reply[:MAX_NEW_TOKENS]
too_long = "word " * 200
assert len(text_tokens(turn_text(too_long))) > CAPACITY

# Conversation 1: a reply that ends with EOS, one cut off by the reply limit
# (byte-fallback text in it), one after a message that spells added tokens,
# and one cut off by the capacity. Then no message fits any more.
first_turns = [
    {"user": "What is your name?", "script": reply_text + [EOS, 15043]},
    {"user": "こんにちは \U0001f35c", "script": long_reply},
    {"user": "a</s>b <s> é", "script": [15043, EOS]},
]
simulated = simulate(first_turns)
# A filler message that leaves the next reply fewer positions than its limit.
position = simulated[-1]["position"]
closing = 1 if simulated[-1]["reply"][-1] == EOS else 2
filler = None
for words in range(1, 400):
    user = " ".join(["word"] * words)
    used = position + closing + len(text_tokens(turn_text(user)))
    if CAPACITY - 2 <= used <= CAPACITY - 1:
        filler = user
        break
assert filler, "no filler message reaches the capacity"
first_turns.append({"user": filler, "script": long_reply})
first = simulate(first_turns)
assert first[-1]["limit"] < MAX_NEW_TOKENS and len(first[-1]["reply"]) == first[-1]["limit"]
assert first[1]["reply"] == long_reply[:MAX_NEW_TOKENS]

# Conversation 2, after Dispose + Initialize: a message beyond the capacity
# fails first, then a new conversation starts at position 0.
second = simulate([{"user": "Hi", "script": [15043, 29991, EOS], "too_long": too_long}])

conversations = [(first, "Hi"), (second, None)]


def c_values(values) -> str:
    return ", ".join(str(value) for value in values) or "0"


lines = [
    "/* Generated by tests/boundary/tinyllama_chat_host.py. */",
    "#include <stdint.h>",
    "",
    "struct turn {",
    "  const uint16_t *user, *too_long;",
    "  int32_t user_length, too_long_length;",
    "  const int32_t *runs;",
    "  int run_count;",
    "  int64_t first_position;",
    "  const int32_t *script;",
    "  int script_count, reply_count;",
    "  const int32_t *stable_units, *stable_offsets, *stable_lengths;",
    "  const int32_t *final_units;",
    "  int32_t final_count;",
    "};",
    "struct conversation {",
    "  const struct turn *turns;",
    "  int turn_count;",
    "  const struct turn *overflow;",
    "};",
    "",
]
conversation_entries = []
for conversation_index, (turns, overflow) in enumerate(conversations):
    turn_entries = []
    for turn_index, turn in enumerate(turns):
        name = f"c{conversation_index}t{turn_index}"
        stable_offsets, stable_lengths, stable_data = [], [], []
        for units in turn["stable"]:
            stable_offsets.append(len(stable_data))
            stable_lengths.append(len(units))
            stable_data += units
        user = utf16_units(turn["user"])
        too_long_units = utf16_units(turn.get("too_long", ""))
        lines += [
            f"static const uint16_t {name}_user[] = {{{c_values(user)}}};",
            f"static const uint16_t {name}_too_long[] = {{{c_values(too_long_units)}}};",
            f"static const int32_t {name}_runs[] = {{{c_values(turn['runs'])}}};",
            f"static const int32_t {name}_script[] = {{{c_values(turn['script'])}}};",
            f"static const int32_t {name}_stable_units[] = {{{c_values(stable_data)}}};",
            f"static const int32_t {name}_stable_offsets[] = {{{c_values(stable_offsets)}}};",
            f"static const int32_t {name}_stable_lengths[] = {{{c_values(stable_lengths)}}};",
            f"static const int32_t {name}_final[] = {{{c_values(turn['final'])}}};",
        ]
        turn_entries.append(
            f"  {{{name}_user, {name}_too_long, {len(user)}, {len(too_long_units)}, "
            f"{name}_runs, {len(turn['runs'])}, {turn['first_position']}, "
            f"{name}_script, {len(turn['script'])}, {len(turn['reply'])}, "
            f"{name}_stable_units, {name}_stable_offsets, {name}_stable_lengths, "
            f"{name}_final, {len(turn['final'])}}},"
        )
    lines += [f"static const struct turn c{conversation_index}_turns[] = {{"] + turn_entries + ["};"]
    overflow_entry = "0"
    if overflow:
        units = utf16_units(overflow)
        lines += [
            f"static const uint16_t c{conversation_index}_overflow_user[] = {{{c_values(units)}}};",
            f"static const struct turn c{conversation_index}_overflow = "
            f"{{c{conversation_index}_overflow_user, 0, {len(units)}, 0}};",
        ]
        overflow_entry = f"&c{conversation_index}_overflow"
    conversation_entries.append(
        f"  {{c{conversation_index}_turns, {len(turns)}, {overflow_entry}}},"
    )
lines += [
    f"#define CONVERSATIONS {len(conversations)}",
    "static const struct conversation conversations[CONVERSATIONS] = {",
    *conversation_entries,
    "};",
    "",
]
output.mkdir(parents=True, exist_ok=True)
(output / "tinyllama_chat_expected.h").write_text("\n".join(lines), encoding="utf-8", newline="\n")

executable = output / ("tinyllama_chat_host.exe" if os.name == "nt" else "tinyllama_chat_host")
sources = [
    root / "tests/runtime/tinyllama_chat_host.c",
    root / "examples/tinyllama-chat/tinyllama_chat.c",
    generated / "tinyllama_tokenizer.c",
    root / "libraries/udontokenizers/src/udontokenizers.c",
]
includes = [
    output,
    generated,
    root / "third_party/OpenCL-Headers",
    root / "include",
    root / "libraries/udontokenizers/include",
    root / "libraries/udononnxruntime/include",
]
definitions = [
    "CL_TARGET_OPENCL_VERSION=120",
    f"TINYLLAMA_CAPACITY={CAPACITY}u",
    f"TINYLLAMA_MAX_NEW_TOKENS={MAX_NEW_TOKENS}u",
]
if compiler_path.name.lower() in ("cl", "cl.exe"):
    command = [compiler_path, "/nologo", "/std:c11", "/W3"]
    command += [f"/D{item}" for item in definitions]
    command += [f"/I{path}" for path in includes]
    command += sources + [f"/Fe:{executable}"]
else:
    command = [compiler_path, "-std=c11"]
    command += [f"-D{item}" for item in definitions]
    for path in includes:
        command += ["-I", path]
    command += sources + ["-o", executable]
subprocess.run([str(item) for item in command], check=True, cwd=output)
result = subprocess.run([str(executable)])
assert result.returncode == 0, f"TinyLlama chat Host failed: {result.returncode}"
print(
    "tinyllama_chat_host=passed "
    f"turns={sum(len(turns) for turns, _ in conversations)} "
    f"runs={sum(len(turn['runs']) for turns, _ in conversations for turn in turns)} "
    f"capacity_limit={first[-1]['limit']}"
)
