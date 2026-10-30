"""Run the MNIST CNN before and after the native UdonHL adapter."""

import argparse
from pathlib import Path
import subprocess
import sys


def run(command, *, capture=False):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=capture,
        timeout=120,
    )
    if result.returncode:
        if capture:
            sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout if capture else ""


def main():
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=root / "build/udoncl")
    parser.add_argument("--output", type=Path, default=root / "build/mnist-cnn")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    object_suffix = ".obj" if sys.platform == "win32" else ".o"

    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    llvm_dir = next(
        line.split("=", 1)[1]
        for line in cache.splitlines()
        if line.startswith("LLVM_DIR:")
    )
    sdk = Path(llvm_dir).parents[2]
    clang = sdk / "bin" / ("clang" + suffix)
    translate = build / "tools/vrc-translate" / ("vrc-translate" + suffix)
    opt = build / "tools/vrc-opt" / ("vrc-opt" + suffix)
    mlir_translate = sdk / "bin" / ("mlir-translate" + suffix)
    runtime_flags = ["-fms-runtime-lib=dll"] if sys.platform == "win32" else []

    host_ir = output / "mnist-cnn.ll"
    imported = output / "mnist-cnn.imported.mlir"
    udonhl = output / "mnist-cnn.udonhl.mlir"
    lowered = output / "mnist-cnn.lowered.mlir"
    lowered_ir = output / "mnist-cnn.lowered.ll"
    lowered_object = output / ("mnist-cnn.lowered" + object_suffix)
    original_executable = output / ("mnist-cnn-original" + suffix)
    lowered_executable = output / ("mnist-cnn-lowered" + suffix)

    run(
        [
            clang,
            *runtime_flags,
            "-std=c11",
            "-O0",
            "-Xclang",
            "-disable-O0-optnone",
            "-fomit-frame-pointer",
            "-S",
            "-emit-llvm",
            source / "mnist_cnn.c",
            "-o",
            host_ir,
        ]
    )
    run([translate, "--import-llvm", host_ir, "-o", imported])
    run([opt, "--vrc-raise-native-host-calls", "--verify-each", imported, "-o", udonhl])
    udonhl_text = udonhl.read_text(encoding="utf-8")
    if "udonhl.call @mnist_cnn_infer" not in udonhl_text:
        raise RuntimeError("CNN call graph did not pass through the UdonHL adapter")
    run([opt, "--vrc-lower-native-host-calls", "--verify-each", udonhl, "-o", lowered])
    run([mlir_translate, "--mlir-to-llvmir", lowered, "-o", lowered_ir])
    run([clang, *runtime_flags, "-c", lowered_ir, "-o", lowered_object])
    run(
        [
            clang,
            *runtime_flags,
            "-std=c11",
            source / "main.c",
            source / "mnist_cnn.c",
            "-o",
            original_executable,
        ]
    )
    run(
        [
            clang,
            *runtime_flags,
            "-std=c11",
            source / "main.c",
            lowered_object,
            "-o",
            lowered_executable,
        ]
    )
    original = run([original_executable], capture=True)
    transformed = run([lowered_executable], capture=True)
    if original != transformed:
        raise RuntimeError("native and UdonHL-adapted CNN output differ")
    if "mnist_fixture_accuracy=10/10" not in transformed:
        raise RuntimeError("CNN failed its ten-class MNIST fixture")
    print(transformed, end="")
    print("native_udonhl_roundtrip=passed")


if __name__ == "__main__":
    main()
