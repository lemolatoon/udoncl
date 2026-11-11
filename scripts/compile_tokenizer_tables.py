"""Compile a supported Hugging Face byte-fallback BPE tokenizer to C tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


COMPILER_ABI = 2
REQUIRED_ASSETS = (
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read tokenizer asset {path}: {error}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _token_content(value, name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    raise ValueError(f"{name} must name one token")


def _parse_normalizer(tokenizer: dict) -> dict[str, str]:
    normalizer = tokenizer.get("normalizer")
    _require(
        isinstance(normalizer, dict) and normalizer.get("type") == "Sequence",
        "unsupported tokenizer normalizer: expected Sequence",
    )
    steps = normalizer.get("normalizers")
    _require(
        isinstance(steps, list) and len(steps) == 2,
        "unsupported tokenizer normalizer sequence",
    )
    prepend, replace = steps
    _require(
        isinstance(prepend, dict)
        and prepend.get("type") == "Prepend"
        and isinstance(prepend.get("prepend"), str),
        "unsupported tokenizer normalizer prepend",
    )
    pattern = replace.get("pattern") if isinstance(replace, dict) else None
    _require(
        isinstance(replace, dict)
        and replace.get("type") == "Replace"
        and isinstance(pattern, dict)
        and isinstance(pattern.get("String"), str)
        and isinstance(replace.get("content"), str),
        "unsupported tokenizer normalizer replacement",
    )
    result = {
        "prepend": prepend["prepend"],
        "replace_from": pattern["String"],
        "replace_to": replace["content"],
    }
    _require(
        all(len(value) == 1 for value in result.values()),
        "unsupported tokenizer normalizer: values must be one Unicode scalar",
    )
    return result


def _validate_decoder(tokenizer: dict) -> None:
    decoder = tokenizer.get("decoder")
    _require(
        isinstance(decoder, dict) and decoder.get("type") == "Sequence",
        "unsupported tokenizer decoder: expected Sequence",
    )
    steps = decoder.get("decoders")
    expected_types = ["Replace", "ByteFallback", "Fuse", "Strip"]
    _require(
        isinstance(steps, list)
        and [step.get("type") for step in steps if isinstance(step, dict)]
        == expected_types,
        "unsupported tokenizer decoder sequence",
    )


def _special_ids(tokenizer: dict, config: dict, special_map: dict) -> dict[str, int]:
    vocab = tokenizer["model"]["vocab"]
    result = {}
    for short, field in (("unk", "unk_token"), ("bos", "bos_token"), ("eos", "eos_token")):
        value = config.get(field, special_map.get(field))
        token = _token_content(value, field)
        _require(token in vocab, f"{field} is absent from tokenizer vocabulary")
        result[short] = int(vocab[token])
    _require(config.get("add_bos_token") is True, "unsupported tokenizer: BOS must be enabled")
    _require(config.get("add_eos_token") is False, "unsupported tokenizer: EOS must not be automatic")
    return result


def _validate_post_processor(tokenizer: dict, bos_id: int) -> None:
    processor = tokenizer.get("post_processor")
    _require(
        isinstance(processor, dict) and processor.get("type") == "TemplateProcessing",
        "unsupported tokenizer post processor",
    )
    tokens = processor.get("special_tokens")
    _require(isinstance(tokens, dict), "tokenizer post processor lacks special tokens")
    found = False
    for item in tokens.values():
        if isinstance(item, dict) and item.get("ids") == [bos_id]:
            found = True
    _require(found, "tokenizer post processor does not add the configured BOS token")


def _parse_merge(value, rank: int, vocab: dict[str, int]) -> tuple[int, int, int, int]:
    if isinstance(value, list) and len(value) == 2 and all(isinstance(x, str) for x in value):
        left, right = value
    elif isinstance(value, str):
        candidates = []
        for index, character in enumerate(value):
            if character == " ":
                left, right = value[:index], value[index + 1 :]
                if left in vocab and right in vocab and left + right in vocab:
                    candidates.append((left, right))
        _require(len(candidates) == 1, f"cannot resolve BPE merge {value!r}")
        left, right = candidates[0]
    else:
        raise ValueError(f"unsupported BPE merge at rank {rank}")
    result = left + right
    _require(left in vocab and right in vocab, f"BPE merge references unknown token at rank {rank}")
    _require(result in vocab, f"BPE merge result is absent at rank {rank}: {result!r}")
    return int(vocab[left]), int(vocab[right]), int(vocab[result]), rank


def _identifier(value: str) -> str:
    _require(bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)), "symbol is not a C identifier")
    return value


def _write_header(path: Path, symbol: str) -> None:
    guard = f"{symbol.upper()}_H"
    path.write_text(
        f"""#ifndef {guard}
#define {guard}

#include <stdint.h>
#include <udontokenizers.h>

extern const uint32_t {symbol}_scalar_values[];
extern const int32_t {symbol}_scalar_token_ids[];
extern const int32_t {symbol}_byte_token_ids[256];
extern const udontokenizers_merge_t {symbol}_merges[];
extern const udontokenizers_model_t {symbol}_model;

#endif
""",
        encoding="utf-8",
        newline="\n",
    )


def _write_source(
    path: Path,
    symbol: str,
    scalars: list[tuple[int, int]],
    byte_ids: list[int],
    merges: list[tuple[int, int, int, int]],
    normalizer: dict[str, str],
    special_ids: dict[str, int],
) -> None:
    lines = [f'#include "{symbol}.h"', ""]
    lines.append(f"const uint32_t {symbol}_scalar_values[] = {{")
    lines.extend(f"  0x{scalar:08x}u," for scalar, _token in scalars)
    lines.extend(["};", ""])
    lines.append(f"const int32_t {symbol}_scalar_token_ids[] = {{")
    lines.extend(f"  {token}," for _scalar, token in scalars)
    lines.extend(["};", ""])
    lines.append(f"const int32_t {symbol}_byte_token_ids[256] = {{")
    for offset in range(0, 256, 8):
        lines.append("  " + ", ".join(str(value) for value in byte_ids[offset : offset + 8]) + ",")
    lines.extend(["};", ""])
    lines.append(f"const udontokenizers_merge_t {symbol}_merges[] = {{")
    lines.extend(f"  {{{left}, {right}, {result}, {rank}}}," for left, right, result, rank in merges)
    lines.extend(["};", ""])
    lines.extend(
        [
            f"const udontokenizers_model_t {symbol}_model = {{",
            f"  {symbol}_scalar_values,",
            f"  {symbol}_scalar_token_ids,",
            f"  {len(scalars)},",
            f"  {symbol}_merges,",
            f"  {len(merges)},",
            f"  {symbol}_byte_token_ids,",
            f"  {ord(normalizer['prepend'])},",
            f"  {ord(normalizer['replace_from'])},",
            f"  {ord(normalizer['replace_to'])},",
            f"  {special_ids['bos']},",
            "};",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def compile_bundle(bundle: Path, output: Path, *, symbol: str) -> Path:
    bundle = bundle.resolve()
    symbol = _identifier(symbol)
    missing = [name for name in REQUIRED_ASSETS if not (bundle / name).is_file()]
    _require(not missing, f"tokenizer bundle is missing: {', '.join(missing)}")
    tokenizer = _load_json(bundle / "tokenizer.json")
    config = _load_json(bundle / "tokenizer_config.json")
    special_map = _load_json(bundle / "special_tokens_map.json")

    model = tokenizer.get("model")
    _require(isinstance(model, dict) and model.get("type") == "BPE", "unsupported tokenizer model")
    _require(model.get("byte_fallback") is True, "tokenizer must enable byte fallback")
    vocab = model.get("vocab")
    merges_json = model.get("merges")
    _require(isinstance(vocab, dict), "tokenizer BPE vocabulary is missing")
    _require(isinstance(merges_json, list), "tokenizer BPE merges are missing")
    _require(len(set(vocab.values())) == len(vocab), "tokenizer IDs must be unique")

    normalizer = _parse_normalizer(tokenizer)
    _validate_decoder(tokenizer)
    special_ids = _special_ids(tokenizer, config, special_map)
    _validate_post_processor(tokenizer, special_ids["bos"])

    byte_ids = []
    for value in range(256):
        token = f"<0x{value:02X}>"
        _require(token in vocab, f"byte fallback token is missing: {token}")
        byte_ids.append(int(vocab[token]))
    scalars = sorted(
        (ord(piece), int(token_id))
        for piece, token_id in vocab.items()
        if len(piece) == 1 and not 0xD800 <= ord(piece) <= 0xDFFF
    )
    merges = sorted(
        (_parse_merge(value, rank, vocab) for rank, value in enumerate(merges_json)),
        key=lambda merge: (merge[0], merge[1]),
    )

    asset_hashes = {name: _digest(bundle / name) for name in sorted(REQUIRED_ASSETS)}
    cache_key_payload = json.dumps(
        {"compiler_abi": COMPILER_ABI, "assets": asset_hashes},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    manifest = {
        "compiler_abi": COMPILER_ABI,
        "cache_key": hashlib.sha256(cache_key_payload).hexdigest(),
        "assets": asset_hashes,
        "vocab_count": len(vocab),
        "scalar_token_count": len(scalars),
        "byte_token_count": len(byte_ids),
        "merge_count": len(merges),
        "normalizer": normalizer,
        "special_token_ids": special_ids,
        "chat_template_sha256": hashlib.sha256(
            str(config.get("chat_template", "")).encode("utf-8")
        ).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_header(output / f"{symbol}.h", symbol)
    _write_source(
        output / f"{symbol}.c",
        symbol,
        scalars,
        byte_ids,
        merges,
        normalizer,
        special_ids,
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", default="tinyllama_tokenizer")
    args = parser.parse_args()
    try:
        manifest = compile_bundle(args.bundle, args.output, symbol=args.symbol)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(manifest)


if __name__ == "__main__":
    main()
