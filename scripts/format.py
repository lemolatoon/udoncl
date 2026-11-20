"""Format repository C/C++/OpenCL C and C# sources with pinned tools."""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CLANG_FORMAT_VERSION = "22.1.0"
CPP_SUFFIXES = {".c", ".cc", ".cl", ".cpp", ".h", ".hpp"}
CSHARP_SUFFIXES = {".cs"}
# Agent worktrees under .claude/ are separate checkouts with their own hooks.
EXCLUDED_DIRECTORIES = {
    ".claude",
    ".deps",
    ".git",
    ".vs",
    "Library",
    "Temp",
    "build",
    "obj",
    "third_party",
}


def source_files(suffixes):
    # Excluded directories are pruned while walking, so large build trees are
    # never traversed.
    result = []
    for directory, subdirectories, files in os.walk(ROOT):
        subdirectories[:] = [
            name for name in subdirectories if name not in EXCLUDED_DIRECTORIES
        ]
        for name in files:
            path = Path(directory) / name
            if path.suffix.lower() in suffixes and path.is_file():
                result.append(path)
    return sorted(result)


def executable(name, environment_name):
    override = os.environ.get(environment_name)
    if override:
        path = Path(override)
        if not path.is_file():
            raise RuntimeError(f"{environment_name} does not name a file: {path}")
        return str(path)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(
            f"{name} was not found; install scripts/requirements-format.txt"
        )
    return found


def check_clang_format_version(command):
    output = subprocess.check_output([command, "--version"], text=True)
    match = re.search(r"clang-format version ([0-9.]+)", output)
    if not match or match.group(1) != CLANG_FORMAT_VERSION:
        actual = match.group(1) if match else output.strip()
        raise RuntimeError(
            f"clang-format {CLANG_FORMAT_VERSION} is required, found {actual}"
        )


# Windows limits a whole command line to 32,767 characters, and a long
# repository path times every source file exceeds it. File arguments are
# therefore passed in batches whose command lines stay well below the limit.
MAX_COMMAND_LINE = 8000


def run(command):
    print("+ " + subprocess.list2cmdline([str(value) for value in command]), flush=True)
    subprocess.run([str(value) for value in command], cwd=ROOT, check=True)


def batches(prefix, files, limit=None):
    """Split `files` so each `prefix + batch` command line fits in `limit`.

    A file whose own command line is longer still gets a batch of its own.
    """
    limit = MAX_COMMAND_LINE if limit is None else limit
    base = len(subprocess.list2cmdline([str(value) for value in prefix]))
    batch, length = [], base
    for path in files:
        width = len(subprocess.list2cmdline([str(path)])) + 1
        if batch and length + width > limit:
            yield batch
            batch, length = [], base
        batch.append(path)
        length += width
    if batch:
        yield batch


def run_batched(prefix, files):
    """Run `prefix` over `files` in batches; any failing batch fails."""
    for batch in batches(prefix, files):
        run([*prefix, *batch])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if formatting would change files"
    )
    args = parser.parse_args()

    cpp = source_files(CPP_SUFFIXES)
    csharp = source_files(CSHARP_SUFFIXES)
    clang_format = executable("clang-format", "CLANG_FORMAT")
    check_clang_format_version(clang_format)

    if cpp:
        flags = ["--dry-run", "--Werror"] if args.check else ["-i"]
        run_batched([clang_format, "--style=file", *flags], cpp)

    dotnet = executable("dotnet", "DOTNET")
    run(
        [
            dotnet,
            "tool",
            "restore",
            "--add-source",
            "https://api.nuget.org/v3/index.json",
        ]
    )
    if csharp:
        action = "check" if args.check else "format"
        run_batched([dotnet, "tool", "run", "csharpier", action], csharp)

    mode = "check" if args.check else "format"
    print(f"{mode} passed: {len(cpp)} C/C++/OpenCL C and {len(csharp)} C# files")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"format failed: {error}", file=sys.stderr)
        raise SystemExit(1)
