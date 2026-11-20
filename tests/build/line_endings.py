"""Tracked text sources are LF in the index and binary assets are never text.

usage: line_endings.py <repository root>
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2])
BINARY_SUFFIXES = {".bytes", ".zip", ".onnx", ".png", ".jpg", ".jpeg", ".bc"}

output = subprocess.run(
    ["git", "ls-files", "--eol"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
    encoding="utf-8",
).stdout
problems = []
for line in output.splitlines():
    metadata, path = line.split("\t", 1)
    index, _, attributes = metadata.split(None, 2)
    if Path(path).suffix.lower() in BINARY_SUFFIXES:
        if "-text" not in attributes or index != "i/-text":
            problems.append(f"binary asset is not marked binary: {path} ({metadata})")
    elif "text eol=lf" in attributes and index != "i/lf":
        problems.append(f"LF text source is not LF in the index: {path} ({metadata})")
    elif "text" not in attributes:
        problems.append(f"tracked file has no line-ending rule: {path}")
if problems:
    print("\n".join(problems))
    raise SystemExit(1)
print("line_endings=valid")
