"""Greedy chat-templated TinyLlama oracle: text -> tokens -> ORT -> text.

The prompt is rendered from the pinned tokenizer_config.json chat template with
the Hugging Face template settings and tokenized by the pinned
``tokenizers`` implementation with its post-processor (BOS). Generation uses
ONNX Runtime with the same fixed-capacity, left-padded ``M=1`` profile as the
Conversation Artifact and stops at EOS or the requested limit. The decoded
reply uses ``skip_special_tokens=True``.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--user", default="Hello")
    parser.add_argument("--system", default="")
    parser.add_argument("--capacity", type=int, default=64)
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
    rolling_past = [
        np.zeros((1, 4, capacity - 1, 64), dtype=np.float32) for _ in past_inputs
    ]

    def step(token: int, position: int) -> np.ndarray:
        nonlocal rolling_past
        valid = position + 1
        mask = np.zeros((1, capacity), dtype=np.int64)
        mask[:, capacity - valid :] = 1
        feed = {
            "input_ids": np.array([[token]], dtype=np.int64),
            "attention_mask": mask,
            "position_ids": np.array([[position]], dtype=np.int64),
        }
        for item, value in zip(past_inputs, rolling_past):
            feed[item.name] = value
        results = session.run(None, feed)
        rolling_past = [value[:, :, 1:, :].copy() for value in results[1:]]
        logits = results[0][0, 0]
        if not np.isfinite(logits).all():
            raise SystemExit(f"non-finite logits at position {position}")
        return logits

    def select(logits: np.ndarray) -> tuple[int, float]:
        # Lowest index wins exact ties, matching the C Host's Device ArgMax.
        best = int(np.argmax(logits))
        others = np.delete(logits, best)
        return best, float(logits[best] - np.max(others))

    started = time.perf_counter()
    for position, token in enumerate(prompt_tokens):
        logits = step(token, position)
    generated: list[int] = []
    margins: list[float] = []
    position = len(prompt_tokens)
    while True:
        token, margin = select(logits)
        generated.append(token)
        margins.append(margin)
        if token == eos or len(generated) == args.max_new_tokens:
            break
        logits = step(token, position)
        position += 1
    elapsed = time.perf_counter() - started

    text = tokenizer.decode(generated, skip_special_tokens=True)
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
        "generated_tokens": generated,
        "stopped_at_eos": generated[-1] == eos,
        "generated_text": text,
        "generated_text_utf16": utf16_units(text),
        "min_greedy_margin": min(margins),
        "greedy_margins": margins,
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
        f"generated={len(generated)} eos={result['stopped_at_eos']} "
        f"min_margin={result['min_greedy_margin']:.4f} text={text!r}"
    )


if __name__ == "__main__":
    main()
