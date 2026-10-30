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
EXCLUDED_DIRECTORIES = {
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
    result = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in EXCLUDED_DIRECTORIES for part in path.relative_to(ROOT).parts):
            continue
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


def run(command):
    print("+ " + subprocess.list2cmdline([str(value) for value in command]), flush=True)
    subprocess.run([str(value) for value in command], cwd=ROOT, check=True)


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
        run([clang_format, "--style=file", *flags, *cpp])

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
        run([dotnet, "tool", "run", "csharpier", action, *csharp])

    mode = "check" if args.check else "format"
    print(f"{mode} passed: {len(cpp)} C/C++/OpenCL C and {len(csharp)} C# files")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"format failed: {error}", file=sys.stderr)
        raise SystemExit(1)
