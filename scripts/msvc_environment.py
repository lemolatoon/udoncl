"""Activate the latest compatible MSVC environment for repository scripts."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def find_vsdevcmd(build_dir):
    cache = Path(build_dir) / "CMakeCache.txt"
    if cache.is_file():
        compiler_value = next(
            (
                line.split("=", 1)[1]
                for line in cache.read_text(encoding="utf-8").splitlines()
                if line.startswith("CMAKE_CXX_COMPILER:") and "=" in line
            ),
            None,
        )
        if compiler_value:
            compiler = Path(compiler_value)
            try:
                candidate = compiler.parents[7] / "Common7/Tools/VsDevCmd.bat"
            except IndexError:
                candidate = None
            if candidate and candidate.is_file():
                return candidate
    program_files_x86 = Path(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    )
    vswhere = program_files_x86 / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.is_file():
        raise RuntimeError(f"Visual Studio locator is missing: {vswhere}")
    installation = subprocess.check_output(
        [
            vswhere,
            "-latest",
            "-prerelease",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
        encoding="utf-8",
    ).strip()
    candidate = Path(installation) / "Common7/Tools/VsDevCmd.bat"
    if not candidate.is_file():
        raise RuntimeError(f"Visual Studio environment script is missing: {candidate}")
    return candidate


def ensure_msvc_environment(build_dir):
    if sys.platform != "win32" or (
        os.environ.get("INCLUDE") and os.environ.get("VCToolsInstallDir")
    ):
        return
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    vsdevcmd = find_vsdevcmd(build_dir)
    with tempfile.TemporaryDirectory(prefix="udoncl-vsdevcmd-", dir=build_dir) as temporary:
        batch = Path(temporary) / "environment.cmd"
        batch.write_text(
            f'@call "{vsdevcmd}" -no_logo -arch=x64 -host_arch=x64 >nul\n@set\n',
            encoding="utf-8",
        )
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", batch],
            check=True,
            text=True,
            capture_output=True,
        )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ[key] = value
