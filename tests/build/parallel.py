"""Check process CPU limits, portable fallbacks and the CMake build command."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('builder', Path(__file__).resolve().parents[2] / 'scripts/build.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ParallelTests(unittest.TestCase):
    def test_process_count_takes_priority(self):
        with patch.object(builder.os, 'process_cpu_count', return_value=3, create=True):
            self.assertEqual(builder.available_cpu_count(), 3)

    def test_affinity_fallback(self):
        with patch.object(builder.os, 'process_cpu_count', None, create=True), \
             patch.object(builder.os, 'sched_getaffinity', return_value={1, 3}, create=True):
            self.assertEqual(builder.available_cpu_count(), 2)

    def test_windows_fallback(self):
        with patch.object(builder.os, 'process_cpu_count', None, create=True), \
             patch.object(builder.os, 'sched_getaffinity', None, create=True), \
             patch.object(builder.os, 'cpu_count', return_value=8):
            self.assertEqual(builder.available_cpu_count(), 8)

    def test_unknown_count_is_one(self):
        with patch.object(builder.os, 'process_cpu_count', None, create=True), \
             patch.object(builder.os, 'sched_getaffinity', side_effect=OSError, create=True), \
             patch.object(builder.os, 'cpu_count', return_value=None):
            self.assertEqual(builder.available_cpu_count(), 1)

    def test_build_command(self):
        with patch('sys.argv', ['build.py', '--preset', 'udoncl', '--target', 'vrccl']), \
             patch.object(builder, 'available_cpu_count', return_value=6), \
             patch.object(builder.subprocess, 'run') as run:
            builder.main()
            run.assert_called_once_with(['cmake', '--build', '--preset', 'udoncl', '--parallel', '6', '--target', 'vrccl'], check=True)


if __name__ == '__main__':
    unittest.main()
