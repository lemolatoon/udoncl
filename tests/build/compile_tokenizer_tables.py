import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "compile_tokenizer_tables",
    ROOT / "scripts" / "compile_tokenizer_tables.py",
)
COMPILER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPILER)


def write_bundle(root: Path, *, normalizer=None) -> None:
    byte_vocab = {f"<0x{value:02X}>": value + 3 for value in range(256)}
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        **byte_vocab,
        "▁": 259,
        "a": 260,
        "b": 261,
        "ab": 262,
        "▁ab": 263,
    }
    if normalizer is None:
        normalizer = {
            "type": "Sequence",
            "normalizers": [
                {"type": "Prepend", "prepend": "▁"},
                {
                    "type": "Replace",
                    "pattern": {"String": " "},
                    "content": "▁",
                },
            ],
        }
    tokenizer = {
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": True},
            {"id": 1, "content": "<s>", "special": True},
            {"id": 2, "content": "</s>", "special": True},
        ],
        "normalizer": normalizer,
        "pre_tokenizer": None,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}
            },
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": "▁", "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0},
            ],
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": True,
            "vocab": vocab,
            "merges": ["a b", "▁ ab"],
        },
    }
    root.mkdir(parents=True)
    (root / "tokenizer.json").write_text(
        json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8"
    )
    (root / "tokenizer.model").write_bytes(b"synthetic sentencepiece model")
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "add_bos_token": True,
                "add_eos_token": False,
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            }
        ),
        encoding="utf-8",
    )
    (root / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}),
        encoding="utf-8",
    )


class TokenizerTableCompilerTests(unittest.TestCase):
    def test_compiles_deterministic_model_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            write_bundle(bundle)
            first = root / "first"
            second = root / "second"
            COMPILER.compile_bundle(bundle, first, symbol="tinyllama_tokenizer")
            COMPILER.compile_bundle(bundle, second, symbol="tinyllama_tokenizer")

            for name in ("tinyllama_tokenizer.h", "tinyllama_tokenizer.c", "manifest.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["compiler_abi"], COMPILER.COMPILER_ABI)
            self.assertEqual(manifest["vocab_count"], 264)
            self.assertEqual(manifest["merge_count"], 2)
            self.assertEqual(manifest["byte_token_count"], 256)
            self.assertEqual(manifest["normalizer"], {
                "prepend": "▁",
                "replace_from": " ",
                "replace_to": "▁",
            })
            self.assertEqual(manifest["special_token_ids"], {
                "bos": 1,
                "eos": 2,
                "unk": 0,
            })
            source = (first / "tinyllama_tokenizer.c").read_text(encoding="utf-8")
            self.assertIn("tinyllama_tokenizer_byte_token_ids[256]", source)
            self.assertIn("tinyllama_tokenizer_merge_left_token_ids[]", source)
            self.assertIn("tinyllama_tokenizer_merge_ranks[]", source)

    def test_rejects_unsupported_normalizer_instead_of_approximating(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            write_bundle(bundle, normalizer={"type": "Lowercase"})
            with self.assertRaisesRegex(ValueError, "normalizer"):
                COMPILER.compile_bundle(bundle, root / "output", symbol="bad")


if __name__ == "__main__":
    unittest.main()
