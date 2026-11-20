"""Unit tests for the batched tool invocations of scripts/format.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("format_script", ROOT / "scripts/format.py")
FORMAT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FORMAT)


def line(command):
    return len(subprocess.list2cmdline([str(value) for value in command]))


class FormatBatchTests(unittest.TestCase):
    def test_batches_keep_every_file_once_in_order_under_the_limit(self):
        prefix = ["C:/Program Files/LLVM/bin/clang-format.exe", "--style=file", "-i"]
        files = [
            Path("C:/Users/someone/a very long/worktree/path/agent-123")
            / f"lib/Conversion/Some Directory/File{index:04d}.cpp"
            for index in range(600)
        ]
        groups = list(FORMAT.batches(prefix, files, limit=2000))
        self.assertGreater(len(groups), 1)
        self.assertEqual([path for group in groups for path in group], files)
        for group in groups:
            self.assertLessEqual(line([*prefix, *group]), 2000)

    def test_an_oversized_file_gets_its_own_batch(self):
        prefix = ["tool"]
        long_path = "x" * 300
        groups = list(FORMAT.batches(prefix, ["a", long_path, "b"], limit=100))
        self.assertEqual(groups, [["a"], [long_path], ["b"]])

    def test_no_files_runs_nothing(self):
        self.assertEqual(list(FORMAT.batches(["tool"], [])), [])

    def test_default_limit_is_far_below_the_windows_command_line_limit(self):
        self.assertLessEqual(FORMAT.MAX_COMMAND_LINE, 32767 // 2)

    def test_source_files_skip_excluded_directories(self):
        self.assertIn(".claude", FORMAT.EXCLUDED_DIRECTORIES)
        files = FORMAT.source_files(FORMAT.CPP_SUFFIXES)
        self.assertIn(ROOT / "scripts" / "format.py", FORMAT.source_files({".py"}))
        self.assertTrue(files)
        for path in files:
            parts = path.relative_to(ROOT).parts
            self.assertFalse(set(parts) & FORMAT.EXCLUDED_DIRECTORIES, path)

    def test_run_batched_stops_at_the_first_failing_batch(self):
        calls = []

        def fake_run(command):
            calls.append(command)
            if len(calls) == 2:
                raise subprocess.CalledProcessError(1, command)

        files = [f"file{index}.cpp" for index in range(30)]
        with mock.patch.object(FORMAT, "run", fake_run), mock.patch.object(
            FORMAT, "MAX_COMMAND_LINE", 60
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                FORMAT.run_batched(["tool", "--flag"], files)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(command[:2] == ["tool", "--flag"] for command in calls))


if __name__ == "__main__":
    unittest.main()
