"""Validate the public UdonCL VPM package without requiring Unity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


SEMVER = re.compile(
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
FORBIDDEN_NAMES = {".git", ".DS_Store", "Library", "Logs", "Obj", "Temp"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VPM package validation failed: {message}")


def validate(package: Path) -> dict:
    package = package.resolve()
    require(package.is_dir(), f"package directory does not exist: {package}")
    manifest_path = package / "package.json"
    require(manifest_path.is_file(), "package.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"VPM package validation failed: invalid package.json: {error}")

    expected = {
        "name": "com.lemolatoon.udoncl",
        "displayName": "UdonCL Runtime and ONNX Importer",
        "unity": "2022.3",
        "license": "GPL-3.0-only",
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, f"package.json {key!r} must be {value!r}")
    require(SEMVER.fullmatch(manifest.get("version", "")) is not None, "invalid SemVer")
    require(bool(manifest.get("description")), "description is missing")
    author = manifest.get("author")
    require(isinstance(author, dict), "author object is missing")
    require(author.get("name") == "Lemolatoon", "author.name is incorrect")
    require(author.get("email") == "lemolatoon@gmail.com", "author.email is incorrect")
    require(bool(author.get("url")), "author.url is missing")
    dependencies = manifest.get("vpmDependencies")
    require(isinstance(dependencies, dict), "vpmDependencies is missing")
    require(
        dependencies.get("com.vrchat.worlds") == "3.10.x",
        "com.vrchat.worlds must use the tested 3.10.x compatibility range",
    )

    for relative in (
        "README.md",
        "CHANGELOG.md",
        "LICENSE.md",
        "Editor/Udoncl.Editor.asmdef",
        "Editor/Examples/hello-dispatch.zip.bytes",
        "Editor/Examples/onnx-arena-add.zip.bytes",
        "Editor/Examples/onnx-tiny-llama.zip.bytes",
        "Editor/Examples/saxpy.zip.bytes",
        "Runtime/Udoncl.Runtime.asmdef",
        "Runtime/Udoncl.Runtime.asset",
    ):
        require((package / relative).is_file(), f"required file is missing: {relative}")
    runtime_usharp = (package / "Runtime/Udoncl.Runtime.asset").read_text(
        encoding="utf-8"
    )
    require(
        "guid: 5136146375e9a0a498a72a0091b40cc1" in runtime_usharp
        and "guid: 36e72c97d34c4a489d882db18bbdf480" in runtime_usharp,
        "Runtime assembly is not registered as an UdonSharp assembly",
    )
    license_text = (package / "LICENSE.md").read_text(encoding="utf-8")
    require(
        "GNU GENERAL PUBLIC LICENSE" in license_text and "Version 3" in license_text,
        "LICENSE.md is not GPL version 3",
    )
    changelog = (package / "CHANGELOG.md").read_text(encoding="utf-8")
    require(
        f"[{manifest['version']}]" in changelog,
        "CHANGELOG.md has no entry for the package version",
    )

    for path in package.rglob("*"):
        relative = path.relative_to(package)
        require(not path.is_symlink(), f"symbolic links are not publishable: {relative}")
        require(
            not any(part in FORBIDDEN_NAMES for part in relative.parts),
            f"forbidden generated path: {relative}",
        )
        if path.name.endswith(".meta"):
            require(path.with_name(path.name[:-5]).exists(), f"orphan meta file: {relative}")
        else:
            require(
                path.with_name(path.name + ".meta").is_file(),
                f"Unity meta file is missing: {relative}",
            )
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package",
        nargs="?",
        type=Path,
        default=root / "unity/Packages/com.lemolatoon.udoncl",
    )
    args = parser.parse_args()
    manifest = validate(args.package)
    print(f"vpm_package=valid name={manifest['name']} version={manifest['version']}")


if __name__ == "__main__":
    main()
