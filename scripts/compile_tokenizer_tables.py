"""Compile a supported Hugging Face byte-fallback BPE tokenizer to C tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


COMPILER_ABI = 7
# Odd multiplier of the merge index hash; halved for very large token IDs.
MERGE_HASH_MULTIPLIER = 40503
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


def _pattern_string(pattern, message: str) -> str:
    if isinstance(pattern, str):
        return pattern
    _require(
        isinstance(pattern, dict) and isinstance(pattern.get("String"), str),
        message,
    )
    return pattern["String"]


def _parse_decoder(tokenizer: dict) -> dict[str, object]:
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
    replace, _byte_fallback, _fuse, strip = steps
    replace_from = _pattern_string(
        replace.get("pattern"), "unsupported tokenizer decoder replacement"
    )
    replace_to = replace.get("content")
    _require(
        isinstance(replace_to, str)
        and len(replace_from) == 1
        and len(replace_to) == 1,
        "unsupported tokenizer decoder replacement: values must be one scalar",
    )
    content = strip.get("content")
    start = strip.get("start")
    stop = strip.get("stop")
    _require(
        isinstance(content, str)
        and len(content) == 1
        and ord(content) <= 0xFFFF
        and isinstance(start, int)
        and start >= 0
        and stop == 0,
        "unsupported tokenizer decoder strip: only a leading BMP strip is supported",
    )
    return {
        "replace_from": replace_from,
        "replace_to": replace_to,
        "strip": content,
        "strip_start": start,
    }


def _utf16(text: str) -> list[int]:
    encoded = text.encode("utf-16-le", "surrogatepass")
    return [
        int.from_bytes(encoded[index : index + 2], "little")
        for index in range(0, len(encoded), 2)
    ]


def _added_tokens(tokenizer: dict, vocab: dict[str, int]) -> list[dict[str, object]]:
    result = []
    for entry in tokenizer.get("added_tokens") or []:
        _require(isinstance(entry, dict), "unsupported added token entry")
        content = entry.get("content")
        token_id = entry.get("id")
        _require(
            isinstance(content, str) and bool(content) and isinstance(token_id, int),
            "unsupported added token entry",
        )
        _require(
            not entry.get("normalized", False)
            and not entry.get("lstrip", False)
            and not entry.get("rstrip", False)
            and not entry.get("single_word", False),
            f"unsupported added token {content!r}: only raw, unstripped "
            "whole-text matching is supported",
        )
        _require(
            vocab.get(content) == token_id,
            f"added token {content!r} does not match the vocabulary",
        )
        result.append(
            {"id": token_id, "content": content, "special": bool(entry.get("special"))}
        )
    return sorted(result, key=lambda item: (-len(_utf16(item["content"])), item["id"]))


CHAT_ROLES = ("system", "user", "assistant")
_PROBE_CONTENTS = ("", "a b", "x\ny", " lead", "trail ", "日本")


def _compile_chat_template(
    config: dict, added: list[dict[str, object]]
) -> dict[str, object] | None:
    """Decompose a Hugging Face chat template into per-message literals.

    The template is rendered with the Hugging Face settings and sentinel
    contents. It is accepted only when every probe conversation renders as the
    concatenation of `prefix[role] + content + suffix[role]` per message
    followed by the generation prompt; anything else is rejected rather than
    approximated.
    """

    source = config.get("chat_template")
    if source is None:
        return None
    _require(isinstance(source, str), "unsupported chat template: expected one string")
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ValueError("compiling a chat template requires jinja2") from error

    def raise_exception(message):
        raise ValueError(f"chat template raised: {message}")

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.globals["raise_exception"] = raise_exception
    template = environment.from_string(source)
    variables = {}
    for field in ("bos_token", "eos_token", "unk_token", "pad_token"):
        value = config.get(field)
        if value is not None:
            variables[field] = _token_content(value, field)

    def render(messages, generation: bool) -> str:
        return template.render(
            messages=messages, add_generation_prompt=generation, **variables
        )

    sentinel = "UDONCL"
    roles: dict[str, tuple[str, str]] = {}
    for role in CHAT_ROLES:
        try:
            text = render([{"role": role, "content": sentinel}], False)
        except Exception:  # noqa: BLE001 - a rejected role is template data
            continue
        if text.count(sentinel) != 1:
            continue
        prefix, suffix = text.split(sentinel)
        roles[role] = (prefix, suffix)
    _require(bool(roles), "unsupported chat template: no role renders its content")
    first_role = next(iter(roles))
    with_prompt = render([{"role": first_role, "content": sentinel}], True)
    without_prompt = render([{"role": first_role, "content": sentinel}], False)
    _require(
        with_prompt.startswith(without_prompt),
        "unsupported chat template: generation prompt is not a suffix",
    )
    generation = with_prompt[len(without_prompt) :]

    sequences = [[role] for role in roles]
    sequences += [[left, right] for left in roles for right in roles]
    sequences += [[a, b, c] for a in roles for b in roles for c in roles]
    for sequence in sequences:
        for content_index in range(len(_PROBE_CONTENTS)):
            messages = [
                {
                    "role": role,
                    "content": _PROBE_CONTENTS[
                        (content_index + index) % len(_PROBE_CONTENTS)
                    ],
                }
                for index, role in enumerate(sequence)
            ]
            expected = "".join(
                roles[item["role"]][0] + item["content"] + roles[item["role"]][1]
                for item in messages
            )
            for flag in (False, True):
                try:
                    actual = render(messages, flag)
                except Exception as error:  # noqa: BLE001
                    raise ValueError(
                        f"unsupported chat template: roles {sequence} raised {error}"
                    ) from error
                _require(
                    actual == expected + (generation if flag else ""),
                    "unsupported chat template: output is not a per-message "
                    f"concatenation for roles {sequence}",
                )

    # Pieces are streamed into the encoder, so an added token straddling a
    # literal boundary would be matched differently than in the whole text.
    literals = [part for pair in roles.values() for part in pair] + [generation]
    for literal in literals:
        for token in added:
            content = str(token["content"])
            for split in range(1, len(content)):
                _require(
                    not literal.startswith(content[split:])
                    and not literal.endswith(content[:split]),
                    f"unsupported chat template: added token {content!r} can "
                    "straddle a literal boundary",
                )
    return {"roles": roles, "generation": generation}


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


def _write_header(
    path: Path, symbol: str, has_chat_template: bool, max_piece_code_units: int
) -> None:
    guard = f"{symbol.upper()}_H"
    chat = (
        f"void {symbol}InitializeChatTemplate(udontokenizers_chat_template_t *chat);\n"
        if has_chat_template
        else ""
    )
    path.write_text(
        f"""#ifndef {guard}
#define {guard}

#include <stdint.h>
#include <udontokenizers.h>

/* Upper bound of UTF-16 code units one decoded token appends. */
#define {symbol.upper()}_MAX_PIECE_CODE_UNITS {max_piece_code_units}

void {symbol}InitializeModel(udontokenizers_model_t *model);
{chat}
#endif
""",
        encoding="utf-8",
        newline="\n",
    )


def _int_table(lines: list[str], ctype: str, name: str, values, per_line: int = 12) -> None:
    values = list(values)
    lines.append(f"static const {ctype} {name}[{max(1, len(values))}] = {{")
    if not values:
        lines.append("  0,")
    for offset in range(0, len(values), per_line):
        lines.append(
            "  " + ", ".join(str(value) for value in values[offset : offset + per_line]) + ","
        )
    lines.extend(["};", ""])


def _merge_hash(merges: list[tuple[int, int, int, int]]) -> tuple[list[int], int, int]:
    """Open-addressing index of the sorted merge table, at most half full."""
    if not merges:
        return [], 0, 0
    size = 1
    while size < 2 * len(merges):
        size *= 2
    largest = max(max(left, right) for left, right, _result, _rank in merges)
    multiplier = MERGE_HASH_MULTIPLIER
    while multiplier > 1 and largest * multiplier + largest > 0x7FFFFFFF:
        multiplier = (multiplier // 2) | 1
    _require(
        largest * multiplier + largest <= 0x7FFFFFFF,
        "token IDs are too large for the merge index",
    )
    slots = [-1] * size
    for index, (left, right, _result, _rank) in enumerate(merges):
        slot = (left * multiplier + right) & (size - 1)
        while slots[slot] >= 0:
            slot = (slot + 1) & (size - 1)
        slots[slot] = index
    return slots, size - 1, multiplier


def _write_source(
    path: Path,
    symbol: str,
    scalars: list[tuple[int, int]],
    byte_ids: list[int],
    merges: list[tuple[int, int, int, int]],
    normalizer: dict[str, str],
    special_ids: dict[str, int],
    added: list[dict[str, object]],
    pieces: list[list[int]],
    piece_kinds: list[int],
    decoder: dict[str, object],
    chat: dict[str, object] | None,
) -> None:
    lines = [f'#include "{symbol}.h"', ""]
    lines.append(f"static const uint32_t {symbol}_scalar_values[] = {{")
    lines.extend(f"  0x{scalar:08x}u," for scalar, _token in scalars)
    lines.extend(["};", ""])
    lines.append(f"static const int32_t {symbol}_scalar_token_ids[] = {{")
    lines.extend(f"  {token}," for _scalar, token in scalars)
    lines.extend(["};", ""])
    lines.append(f"static const int32_t {symbol}_byte_token_ids[256] = {{")
    for offset in range(0, 256, 8):
        lines.append("  " + ", ".join(str(value) for value in byte_ids[offset : offset + 8]) + ",")
    lines.extend(["};", ""])
    for suffix, value_index in (
        ("merge_left_token_ids", 0),
        ("merge_right_token_ids", 1),
        ("merge_result_token_ids", 2),
        ("merge_ranks", 3),
    ):
        _int_table(
            lines, "int32_t", f"{symbol}_{suffix}", (merge[value_index] for merge in merges)
        )

    hash_slots, hash_mask, hash_multiplier = _merge_hash(merges)
    _int_table(lines, "int32_t", f"{symbol}_merge_hash_slots", hash_slots, 16)

    added_units: list[int] = []
    added_offsets = [0]
    for token in added:
        added_units.extend(_utf16(str(token["content"])))
        added_offsets.append(len(added_units))
    _int_table(lines, "int32_t", f"{symbol}_added_token_ids", (token["id"] for token in added))
    _int_table(lines, "int32_t", f"{symbol}_added_token_offsets", added_offsets)
    _int_table(lines, "uint16_t", f"{symbol}_added_token_code_units", added_units)

    piece_units: list[int] = []
    piece_offsets = [0]
    for piece in pieces:
        piece_units.extend(piece)
        piece_offsets.append(len(piece_units))
    _int_table(lines, "int32_t", f"{symbol}_piece_offsets", piece_offsets)
    _int_table(lines, "uint16_t", f"{symbol}_piece_code_units", piece_units, 16)
    _int_table(lines, "int32_t", f"{symbol}_piece_kinds", piece_kinds, 16)

    lines.extend(
        [
            f"void {symbol}InitializeModel(udontokenizers_model_t *model) {{",
            f"  model->scalar_values = {symbol}_scalar_values;",
            f"  model->scalar_token_ids = {symbol}_scalar_token_ids;",
            f"  model->scalar_count = {len(scalars)};",
            f"  model->merge_left_token_ids = {symbol}_merge_left_token_ids;",
            f"  model->merge_right_token_ids = {symbol}_merge_right_token_ids;",
            f"  model->merge_result_token_ids = {symbol}_merge_result_token_ids;",
            f"  model->merge_ranks = {symbol}_merge_ranks;",
            f"  model->merge_count = {len(merges)};",
            f"  model->byte_token_ids = {symbol}_byte_token_ids;",
            f"  model->normalizer_prepend_scalar = {ord(normalizer['prepend'])};",
            f"  model->normalizer_replace_from_scalar = {ord(normalizer['replace_from'])};",
            f"  model->normalizer_replace_to_scalar = {ord(normalizer['replace_to'])};",
            f"  model->bos_token_id = {special_ids['bos']};",
            f"  model->added_token_ids = {symbol}_added_token_ids;",
            f"  model->added_token_offsets = {symbol}_added_token_offsets;",
            f"  model->added_token_code_units = {symbol}_added_token_code_units;",
            f"  model->added_token_count = {len(added)};",
            f"  model->piece_offsets = {symbol}_piece_offsets;",
            f"  model->piece_code_units = {symbol}_piece_code_units;",
            f"  model->piece_kinds = {symbol}_piece_kinds;",
            f"  model->vocab_count = {len(pieces)};",
            f"  model->decoder_strip_scalar = {ord(str(decoder['strip']))};",
            f"  model->decoder_strip_count = {decoder['strip_start']};",
            f"  model->eos_token_id = {special_ids['eos']};",
            f"  model->merge_hash_slots = {symbol}_merge_hash_slots;"
            if hash_slots
            else "  model->merge_hash_slots = 0;",
            f"  model->merge_hash_mask = {hash_mask};",
            f"  model->merge_hash_multiplier = {hash_multiplier};",
            "}",
            "",
        ]
    )
    if chat is not None:
        roles = chat["roles"]
        units: list[int] = []
        layout = {"prefix_offsets": [], "prefix_lengths": [], "suffix_offsets": [], "suffix_lengths": []}
        for role in CHAT_ROLES:
            if role not in roles:
                for key in layout:
                    layout[key].append(0 if key.endswith("offsets") else -1)
                continue
            for part, key in ((roles[role][0], "prefix"), (roles[role][1], "suffix")):
                encoded = _utf16(part)
                layout[f"{key}_offsets"].append(len(units))
                layout[f"{key}_lengths"].append(len(encoded))
                units.extend(encoded)
        generation = _utf16(str(chat["generation"]))
        generation_offset = len(units)
        units.extend(generation)
        _int_table(lines, "uint16_t", f"{symbol}_chat_code_units", units, 16)
        for key, values in layout.items():
            _int_table(lines, "int32_t", f"{symbol}_chat_role_{key}", values)
        lines.extend(
            [
                f"void {symbol}InitializeChatTemplate(udontokenizers_chat_template_t *chat) {{",
                f"  chat->code_units = {symbol}_chat_code_units;",
                f"  chat->role_prefix_offsets = {symbol}_chat_role_prefix_offsets;",
                f"  chat->role_prefix_lengths = {symbol}_chat_role_prefix_lengths;",
                f"  chat->role_suffix_offsets = {symbol}_chat_role_suffix_offsets;",
                f"  chat->role_suffix_lengths = {symbol}_chat_role_suffix_lengths;",
                f"  chat->generation_prompt_offset = {generation_offset};",
                f"  chat->generation_prompt_length = {len(generation)};",
                "}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


_BYTE_PIECE = re.compile(r"<0x([0-9A-F]{2})>")


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
    _require(
        sorted(vocab.values()) == list(range(len(vocab))),
        "tokenizer IDs must be dense from zero",
    )

    normalizer = _parse_normalizer(tokenizer)
    decoder = _parse_decoder(tokenizer)
    special_ids = _special_ids(tokenizer, config, special_map)
    _validate_post_processor(tokenizer, special_ids["bos"])
    added = _added_tokens(tokenizer, vocab)
    chat = _compile_chat_template(config, added)

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

    special_contents = {str(token["content"]) for token in added if token["special"]}
    by_id = {int(token_id): piece for piece, token_id in vocab.items()}
    pieces: list[list[int]] = []
    piece_kinds: list[int] = []
    for token_id in range(len(vocab)):
        piece = by_id[token_id]
        byte = _BYTE_PIECE.fullmatch(piece)
        if piece in special_contents:
            piece_kinds.append(-2)
        elif byte:
            piece_kinds.append(int(byte.group(1), 16))
        else:
            piece_kinds.append(-1)
        pieces.append(
            [] if byte else _utf16(piece.replace(str(decoder["replace_from"]), str(decoder["replace_to"])))
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
        "decoder": decoder,
        "special_token_ids": special_ids,
        "added_tokens": [
            {"id": token["id"], "content": token["content"], "special": token["special"]}
            for token in added
        ],
        "piece_code_unit_count": sum(len(piece) for piece in pieces),
        "chat_template_sha256": hashlib.sha256(
            str(config.get("chat_template", "")).encode("utf-8")
        ).hexdigest(),
        "chat_template": None
        if chat is None
        else {
            "roles": {role: list(parts) for role, parts in chat["roles"].items()},
            "generation_prompt": chat["generation"],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    max_piece_code_units = max([1] + [len(piece) for piece in pieces])
    manifest["max_piece_code_units"] = max_piece_code_units
    _write_header(output / f"{symbol}.h", symbol, chat is not None, max_piece_code_units)
    _write_source(
        output / f"{symbol}.c",
        symbol,
        scalars,
        byte_ids,
        merges,
        normalizer,
        special_ids,
        added,
        pieces,
        piece_kinds,
        decoder,
        chat,
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
