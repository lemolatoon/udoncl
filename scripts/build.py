"""Build with CMake using the number of CPUs available to this process."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def ensure_msvc_environment(build_dir):
    if sys.platform != 'win32' or (
        os.environ.get('INCLUDE') and os.environ.get('VCToolsInstallDir')
    ):
        return
    cache = Path(build_dir) / 'CMakeCache.txt'
    if not cache.is_file():
        raise RuntimeError(f'configured CMake build is missing: {cache}')
    compiler_value = next(
        (line.split('=', 1)[1]
         for line in cache.read_text(encoding='utf-8').splitlines()
         if line.startswith('CMAKE_CXX_COMPILER:') and '=' in line),
        None,
    )
    if not compiler_value:
        raise RuntimeError('CMake cache does not identify the MSVC compiler')
    compiler = Path(compiler_value)
    try:
        visual_studio = compiler.parents[7]
    except IndexError as error:
        raise RuntimeError(f'unexpected MSVC compiler path: {compiler}') from error
    vsdevcmd = visual_studio / 'Common7/Tools/VsDevCmd.bat'
    if not vsdevcmd.is_file():
        raise RuntimeError(f'Visual Studio environment script is missing: {vsdevcmd}')
    with tempfile.TemporaryDirectory(prefix='udoncl-vsdevcmd-', dir=cache.parent) as temporary:
        batch = Path(temporary) / 'environment.cmd'
        batch.write_text(
            f'@call "{vsdevcmd}" -no_logo -arch=x64 -host_arch=x64 >nul\n@set\n',
            encoding='utf-8',
        )
        result = subprocess.run(
            ['cmd.exe', '/d', '/c', batch], check=True, text=True,
            capture_output=True)
    for line in result.stdout.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value


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
    if args.build_dir:
        ensure_msvc_environment(args.build_dir)
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
