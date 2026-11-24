"""Greedy chat-templated TinyLlama oracle: text -> tokens -> ORT -> text.

The prompt is rendered from the pinned tokenizer_config.json chat template with
the Hugging Face template settings and tokenized by the pinned
``tokenizers`` implementation with its post-processor (BOS). Generation uses
ONNX Runtime with the profile of the Conversation Artifact, one position per
run on a past of exactly the positions before it (the bounded past of the
model library), and stops at EOS or the requested limit. Every run is checked
against the fixed-capacity, left-padded ``M=1`` profile of the earlier
artifacts: both must select the same tokens. The decoded reply uses
``skip_special_tokens=True``.

Multi-turn (``--next-user``, repeatable): each later user message continues
the same sequence state, as the Chat Artifact does. A reply stays in the
context as its generated tokens: its last token, which no generation step
runs, is run now, followed by EOS when the reply was cut off by the limit.
Then the template text from the end of the reply through the next generation
prompt is tokenized without BOS. A later turn requests at most
``--max-new-tokens``, reduced to what the Context Capacity still holds. The
top-level fields describe the first turn; ``turns`` lists the later ones.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


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


def render_chat(config: dict, messages: list[dict[str, str]]) -> str:
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = environment.from_string(config["chat_template"])
    return template.render(
        messages=messages,
        add_generation_prompt=True,
        bos_token=config["bos_token"],
        eos_token=config["eos_token"],
        unk_token=config["unk_token"],
    )


def turn_text(config: dict, messages: list[dict[str, str]], user: str) -> str:
    """The template text after the EOS that ends the next assistant message,
    through the generation prompt after the next user message."""
    sentinel = "udoncl-reply"
    rendered = render_chat(
        config,
        messages
        + [
            {"role": "assistant", "content": sentinel},
            {"role": "user", "content": user},
        ],
    )
    tail = rendered[rendered.index(sentinel) + len(sentinel) :]
    if not tail.startswith(config["eos_token"]):
        raise SystemExit("the chat template does not end a reply with EOS")
    return tail[len(config["eos_token"]) :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--user", default="Hello")
    parser.add_argument("--system", default="")
    parser.add_argument(
        "--next-user",
        action="append",
        default=[],
        help="a later user message of the same conversation (repeatable)",
    )
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()

    fetch_tokenizer = load(
        "fetch_tinyllama_tokenizer", root / "scripts/fetch_tinyllama_tokenizer.py"
    )
    fetch_model = load("fetch_tinyllama_onnx", root / "scripts/fetch_tinyllama_onnx.py")
    bundle = fetch_tokenizer.ensure(fetch_tokenizer.default_cache())
    config = json.loads((bundle / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    eos = tokenizer.token_to_id(config["eos_token"])

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.user})
    prompt_text = render_chat(config, messages)
    prompt_tokens = tokenizer.encode(prompt_text).ids
    capacity = args.capacity
    if len(prompt_tokens) + args.max_new_tokens - 1 > capacity:
        raise SystemExit(
            f"prompt of {len(prompt_tokens)} tokens plus {args.max_new_tokens} "
            f"new tokens exceeds capacity {capacity}"
        )

    model = fetch_model.ensure(fetch_model.default_cache())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    past_inputs = session.get_inputs()[3:]
    # The Conversation Artifact runs one position per run on a past of exactly
    # the positions before it (examples/onnx-transformer-tiny/conversation_host.c
    # binds the bounded past of its model library). The fixed-capacity,
    # left-padded profile of the earlier artifacts runs alongside: every run
    # must select the same token in both, so a token the Artifact gets wrong
    # is a compiler bug rather than a difference of profiles.
    variable_past = [
        np.zeros((1, 4, 0, 64), dtype=np.float32) for _ in past_inputs
    ]
    fixed_past = [
        np.zeros((1, 4, capacity - 1, 64), dtype=np.float32) for _ in past_inputs
    ]

    def step(token: int, position: int) -> np.ndarray:
        nonlocal variable_past, fixed_past
        ids = np.array([[token]], dtype=np.int64)
        positions = np.array([[position]], dtype=np.int64)
        variable_feed = {
            "input_ids": ids,
            "attention_mask": np.ones((1, position + 1), dtype=np.int64),
            "position_ids": positions,
        }
        fixed_mask = np.zeros((1, capacity), dtype=np.int64)
        fixed_mask[:, capacity - position - 1 :] = 1
        fixed_feed = {
            "input_ids": ids,
            "attention_mask": fixed_mask,
            "position_ids": positions,
        }
        for item, variable, fixed in zip(past_inputs, variable_past, fixed_past):
            variable_feed[item.name] = variable
            fixed_feed[item.name] = fixed
        variable_results = session.run(None, variable_feed)
        fixed_results = session.run(None, fixed_feed)
        variable_past = variable_results[1:]
        fixed_past = [value[:, :, 1:, :].copy() for value in fixed_results[1:]]
        logits = variable_results[0][0, 0]
        if not np.isfinite(logits).all():
            raise SystemExit(f"non-finite logits at position {position}")
        if int(np.argmax(logits)) != int(np.argmax(fixed_results[0][0, 0])):
            raise SystemExit(
                f"the bounded and the fixed-capacity profile select different "
                f"tokens at position {position}"
            )
        return logits

    def select(logits: np.ndarray) -> tuple[int, float]:
        # Lowest index wins exact ties, matching the C Host's Device ArgMax.
        best = int(np.argmax(logits))
        others = np.delete(logits, best)
        return best, float(logits[best] - np.max(others))

    position = 0

    def feed(tokens: list[int]) -> np.ndarray:
        nonlocal position
        for token in tokens:
            if position >= capacity:
                raise SystemExit(f"position {position} reaches capacity {capacity}")
            logits = step(token, position)
            position += 1
        return logits

    def generate(logits: np.ndarray, max_new_tokens: int) -> dict:
        nonlocal position
        generated: list[int] = []
        margins: list[float] = []
        while True:
            token, margin = select(logits)
            generated.append(token)
            margins.append(margin)
            if token == eos or len(generated) == max_new_tokens:
                break
            logits = step(token, position)
            position += 1
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "generated_tokens": generated,
            "stopped_at_eos": generated[-1] == eos,
            "generated_text": text,
            "generated_text_utf16": utf16_units(text),
            "min_greedy_margin": min(margins),
            "greedy_margins": margins,
        }

    started = time.perf_counter()
    first = generate(feed(prompt_tokens), args.max_new_tokens)
    previous, previous_limit = first["generated_tokens"], args.max_new_tokens
    turns = []
    for user in args.next_user:
        last = previous[-1]
        # The Host already ran an EOS selected before the limit; this oracle
        # runs no selected token at the end of a generation, so it runs the
        # last one now. A reply cut off by the limit is closed with EOS.
        closing = [] if last == eos and len(previous) < previous_limit else [last]
        if last != eos:
            closing.append(eos)
        text = turn_text(config, messages, user)
        appended = tokenizer.encode(text, add_special_tokens=False).ids
        if not appended or appended[0] == tokenizer.token_to_id(config["bos_token"]):
            raise SystemExit("a turn must be non-empty and must not start with BOS")
        messages += [
            {"role": "assistant", "content": tokenizer.decode(previous, skip_special_tokens=True)},
            {"role": "user", "content": user},
        ]
        run = [last] + ([eos] if last != eos else []) + appended
        if position + len(run) > capacity:
            raise SystemExit(
                f"turn {user!r} needs {position + len(run)} positions, more than "
                f"capacity {capacity}; the Chat Artifact rejects it as CAPACITY_EXCEEDED"
            )
        logits = feed(run)
        # As the Host, keep every run position below the Context Capacity.
        limit = min(args.max_new_tokens, capacity - position + 1)
        if limit < 1:
            raise SystemExit(f"turn {user!r} does not fit capacity {capacity}")
        context = position
        reply = generate(logits, limit)
        turns.append(
            {
                "user_text": user,
                "turn_text": text,
                "max_new_tokens": limit,
                "closing_tokens": closing,
                "turn_tokens": appended,
                # What ReadPromptTokens reports once ContinueChatTurn is done.
                "prompt_tokens": closing + appended,
                "context_tokens": context,
                **reply,
            }
        )
        previous, previous_limit = reply["generated_tokens"], limit
    elapsed = time.perf_counter() - started

    result = {
        "schema": "udoncl.tinyllama-chat-oracle.v1",
        "model_sha256": fetch_model.MODEL_SHA256,
        "tokenizer_bundle": bundle.name,
        "capacity": capacity,
        "max_new_tokens": args.max_new_tokens,
        "system_text": args.system,
        "user_text": args.user,
        "prompt_text": prompt_text,
        "prompt_tokens": prompt_tokens,
        **first,
        "turns": turns,
        "seconds": elapsed,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "chat-oracle.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"tinyllama_chat_oracle prompt={len(prompt_tokens)} "
        f"generated={len(first['generated_tokens'])} eos={first['stopped_at_eos']} "
        f"min_margin={first['min_greedy_margin']:.4f} text={first['generated_text']!r}"
    )
    for index, turn in enumerate(turns, 2):
        print(
            f"tinyllama_chat_oracle turn={index} prompt={len(turn['prompt_tokens'])} "
            f"context={turn['context_tokens']} max_new_tokens={turn['max_new_tokens']} "
            f"generated={len(turn['generated_tokens'])} eos={turn['stopped_at_eos']} "
            f"min_margin={turn['min_greedy_margin']:.4f} text={turn['generated_text']!r}"
        )


if __name__ == "__main__":
    main()
