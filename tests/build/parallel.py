"""Check process CPU limits, portable fallbacks and the CMake build command."""
import importlib.util
import os
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

    def test_preset_initializes_msvc_environment_before_build(self):
        order = []
        with patch.object(builder.sys, 'platform', 'win32'), \
             patch.dict(builder.os.environ, {}, clear=True), \
             patch('sys.argv', ['build.py', '--preset', 'custom', '--target', 'vrccl']), \
             patch.object(builder, 'available_cpu_count', return_value=6), \
             patch.object(builder, 'ensure_msvc_environment',
                          side_effect=lambda directory: order.append(
                              ('environment', Path(directory)))) as environment, \
             patch.object(builder.subprocess, 'run',
                          side_effect=lambda command, check: order.append(
                              ('build', command))) as run:
            builder.main()
        environment.assert_called_once_with(builder.ROOT / 'build')
        run.assert_called_once_with(
            ['cmake', '--build', '--preset', 'custom', '--parallel', '6',
             '--target', 'vrccl'], check=True)
        self.assertEqual(order[0], ('environment', builder.ROOT / 'build'))
        self.assertEqual(order[1][0], 'build')

    def test_explicit_build_directory_initializes_before_build(self):
        order = []
        build_dir = Path('out') / 'explicit'
        with patch.object(builder.sys, 'platform', 'win32'), \
             patch.dict(builder.os.environ, {}, clear=True), \
             patch('sys.argv', ['build.py', '--build-dir', str(build_dir)]), \
             patch.object(builder, 'available_cpu_count', return_value=2), \
             patch.object(builder, 'ensure_msvc_environment',
                          side_effect=lambda directory: order.append(
                              ('environment', Path(directory)))) as environment, \
             patch.object(builder.subprocess, 'run',
                          side_effect=lambda command, check: order.append(
                              ('build', command))) as run:
            builder.main()
        environment.assert_called_once_with(build_dir)
        run.assert_called_once_with(
            ['cmake', '--build', str(build_dir), '--parallel', '2'], check=True)
        self.assertEqual([entry[0] for entry in order], ['environment', 'build'])

    def test_non_windows_preset_build_does_not_initialize_msvc_process(self):
        with patch.object(builder.sys, 'platform', 'linux'), \
             patch.dict(os.environ, {}, clear=True), \
             patch('sys.argv', ['build.py']), \
             patch.object(builder, 'available_cpu_count', return_value=4), \
             patch.object(builder.subprocess, 'run') as run:
            builder.main()
        run.assert_called_once_with(
            ['cmake', '--build', '--preset', 'udoncl', '--parallel', '4'],
            check=True)


if __name__ == '__main__':
    unittest.main()
