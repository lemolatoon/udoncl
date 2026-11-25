"""Validate the public UdonCL VPM package without requiring Unity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import zipfile


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


def archived_binding(archive_name: str):
    """The binding a bundled example archive installs on import."""

    def load(package: Path) -> tuple[str, str, str]:
        archive = package / "Editor/Examples" / archive_name
        require(archive.is_file(), f"packaged example is missing: {archive_name}")
        with zipfile.ZipFile(archive) as bundle:
            program = json.loads(bundle.read("manifest.json").decode("utf-8"))["program"]
            source_name = program.get("binding_source")
            # Without a generated source the importer falls back to its own
            # U# generator, which does not implement the async command ABI.
            require(
                bool(source_name) and source_name in bundle.namelist(),
                f"{archive_name} ships no generated binding source",
            )
            source = bundle.read(source_name).decode("utf-8")
        return program.get("binding_namespace", ""), program.get("binding_class", ""), source

    return load


def fixture_binding(fixture_name: str):
    """The compile-only stand-in a Package Sample uses without its artifact."""

    def load(package: Path) -> tuple[str, str, str]:
        fixture = package / "Tests/Editor/Fixtures" / fixture_name
        require(fixture.is_file(), f"sample binding fixture is missing: {fixture_name}")
        source = fixture.read_text(encoding="utf-8")
        namespace = re.search(r"^namespace ([\w.]+)", source, re.MULTILINE)
        declared = re.search(r"public sealed class (\w+)", source)
        require(
            namespace is not None and declared is not None,
            f"{fixture_name} declares no binding class",
        )
        return namespace.group(1), declared.group(1), source

    return load


# Each Package Sample, its U# script, and the binding it compiles against in
# the Editor gate: the one the bundled example imports, or the stand-in for
# the binding of the TinyLlama chat example, whose model is too large to
# bundle.
SAMPLES = (
    ("C and OpenCL SAXPY", "SaxpyCaller.cs", archived_binding("saxpy.zip.bytes")),
    ("ONNX MNIST", "MnistClassifier.cs", archived_binding("onnx-mnist.zip.bytes")),
    ("TinyLlama Chat", "TinyLlamaChat.cs", fixture_binding("tinyllama-chat-binding.cs.txt")),
)


def require_sample_binding(script: Path, namespace: str, class_name: str, source: str) -> None:
    """Require that a sample references the binding class its source declares."""
    text = script.read_text(encoding="utf-8")
    fields = set(re.findall(r"^\s*private (\w+) binding;", text, re.MULTILINE))
    require(len(fields) == 1, f"{script.name} must declare one binding field")
    (field,) = fields
    require(
        field == class_name,
        f"{script.name} uses {field}, but its binding is {namespace}.{class_name}",
    )
    require(
        f"using {namespace};" in text,
        f"{script.name} does not import the binding namespace {namespace}",
    )
    require(
        re.search(rf"^namespace {re.escape(namespace)}\b", source, re.MULTILINE) is not None
        and f"public sealed class {class_name} " in source,
        f"the binding source does not declare {namespace}.{class_name}",
    )


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
        "Editor/Examples/onnx-mnist.zip.bytes",
        "Editor/Examples/saxpy.zip.bytes",
    ):
        require((package / relative).is_file(), f"required file is missing: {relative}")
    samples = {sample.get("path") for sample in manifest.get("samples", [])}
    for sample, script, binding_source in SAMPLES:
        relative = f"Samples~/{sample}"
        require(relative in samples, f"package.json does not declare the {sample} sample")
        require((package / relative / script).is_file(), f"sample script is missing: {script}")
        require(
            (package / relative / "README.md").is_file(),
            f"sample README is missing: {relative}",
        )
        require_sample_binding(
            package / relative / script, *binding_source(package)
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
