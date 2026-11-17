"""Build with CMake using the number of CPUs available to this process."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

# Keep this script importable by focused tests which load it through
# importlib.util.spec_from_file_location instead of executing it as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from msvc_environment import ensure_msvc_environment

ROOT = Path(__file__).resolve().parents[1]


def available_cpu_count():
    # Python 3.13+ accounts for process affinity directly. On Python 3.12,
    # sched_getaffinity does that on Linux; Windows uses os.cpu_count.
    process_count = getattr(os, 'process_cpu_count', None)
    if process_count is not None:
        count = process_count()
        if count:
            return count
    affinity = getattr(os, 'sched_getaffinity', None)
    if affinity is not None:
        try:
            count = len(affinity(0))
            if count:
                return count
        except OSError:
            pass
    return os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--preset', help='CMake build preset (default: udoncl)')
    selection.add_argument('--build-dir', help='Use a build directory instead of a preset')
    parser.add_argument('--target', nargs='+', help='Build selected targets')
    parser.add_argument('--config', help='Configuration for multi-config generators')
    args = parser.parse_args()
    # Presets own their binary directory, so use the repository build root as
    # the environment bootstrap directory without assuming a preset name or
    # duplicating CMake's preset expansion. Explicit build directories retain
    # the cache-aware Visual Studio discovery in msvc_environment.py.
    ensure_msvc_environment(Path(args.build_dir) if args.build_dir
                            else ROOT / 'build')
    command = ['cmake', '--build']
    command += [args.build_dir] if args.build_dir else ['--preset', args.preset or 'udoncl']
    jobs = available_cpu_count()
    command += ['--parallel', str(jobs)]
    if args.target:
        command += ['--target', *args.target]
    if args.config:
        command += ['--config', args.config]
    print(f'Building with {jobs} parallel jobs (available CPU count)', flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
