"""Compile and execute the MNIST CNN as a real Device fragment kernel."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=root / "build/udoncl")
    parser.add_argument("--output", type=Path, default=root / "build/mnist-cnn-device")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if sys.platform != "win32":
        icds = list(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json"))
        if not icds:
            parser.error("install mesa-vulkan-drivers and libvulkan-dev")
        environment["VK_DRIVER_FILES"] = environment["VK_ICD_FILENAMES"] = str(icds[0])
    commands = []

    def run(command):
        command = [str(value) for value in command]
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=environment,
            timeout=600,
        )
        commands.append(
            {
                "argv": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        (output / "commands.json").write_text(
            json.dumps(commands, indent=2), encoding="utf-8"
        )
        if result.returncode:
            sys.stderr.write(result.stdout + result.stderr)
            raise SystemExit(result.returncode)
        return result.stdout

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
    compiler = build / "tools/vrccl" / ("vrccl" + suffix)
    translate = build / "tools/vrc-translate" / ("vrc-translate" + suffix)
    opt = build / "tools/vrc-opt" / ("vrc-opt" + suffix)
    mlir_translate = sdk / "bin" / ("mlir-translate" + suffix)
    runtime_flags = ["-fms-runtime-lib=dll"] if sys.platform == "win32" else []

    device_text = (source / "device.cl").read_text(encoding="utf-8")
    (output / "device_source.h").write_text(
        "#define UDONCL_DEVICE_SOURCE " + json.dumps(device_text) + "\n",
        encoding="utf-8",
    )
    device_ir = output / "device.ll"
    run(
        [
            clang,
            "-target",
            "spir-unknown-unknown",
            "-x",
            "cl",
            "-cl-std=CL1.2",
            "-cl-kernel-arg-info",
            "-O0",
            "-Xclang",
            "-disable-O0-optnone",
            "-fomit-frame-pointer",
            "-ffp-contract=off",
            "-S",
            "-emit-llvm",
            source / "device.cl",
            "-o",
            device_ir,
        ]
    )
    bundle = output / "device-bundle"
    with tempfile.TemporaryDirectory(prefix="device-", dir=output) as staging:
        staged = Path(staging) / "bundle"
        run(
            [
                compiler,
                "--emit-native-device-bundle",
                source / "device.json",
                "--device-llvm",
                device_ir,
                "-o",
                staged,
            ]
        )
        if bundle.exists():
            previous = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            if previous.get("artifact_stage") != "native-device-prototype":
                raise RuntimeError("refusing to replace a non-example bundle")
            shutil.rmtree(bundle)
        staged.rename(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    kernel = manifest["device"]["kernels"][0]
    if (
        kernel["name"] != "mnist_cnn"
        or kernel["global_size"] != [4096, 1, 1]
        or [argument["required_bytes"] for argument in kernel["args"][:2]]
        != [16384, 16384]
    ):
        raise RuntimeError("MNIST gather manifest does not match its packed ABI")
    fragment = bundle / "kernels/kernel_20.hlsl"
    (output / "Bindings.h").write_text(
        "inline const char *deviceSource() { return "
        + json.dumps(device_text)
        + "; }\n"
        + "inline std::vector<native_opencl::GatherBinding> manifestBindings(const std::string &fragment) { "
        + 'return {{20, 4096, 0, "mnist_cnn", fragment}}; }\n',
        encoding="utf-8",
    )

    host_ir = output / "host.ll"
    imported = output / "host.imported.mlir"
    udonhl = output / "host.udonhl.mlir"
    lowered = output / "host.lowered.mlir"
    native_ir = output / "host.native.ll"
    run(
        [
            clang,
            *runtime_flags,
            "-x",
            "c",
            "-std=c11",
            "-O0",
            "-Xclang",
            "-disable-O0-optnone",
            "-fomit-frame-pointer",
            "-I",
            root / "third_party/OpenCL-Headers",
            "-I",
            output,
            "-S",
            "-emit-llvm",
            source / "host.c",
            "-o",
            host_ir,
        ]
    )
    run([translate, "--import-llvm", host_ir, "-o", imported])
    run([opt, "--vrc-raise-native-host-calls", "--verify-each", imported, "-o", udonhl])
    run([opt, "--vrc-lower-native-host-calls", "--verify-each", udonhl, "-o", lowered])
    run([mlir_translate, "--mlir-to-llvmir", lowered, "-o", native_ir])
    host_object = output / ("host" + object_suffix)
    device_object = output / ("device" + object_suffix)
    run([clang, *runtime_flags, "-c", native_ir, "-o", host_object])
    run(
        [
            clang,
            *runtime_flags,
            "-x",
            "c",
            "-std=c11",
            "-ffp-contract=off",
            "-c",
            source / "device.cl",
            "-o",
            device_object,
        ]
    )

    vertex = output / "fullscreen.hlsl"
    vertex.write_text(
        "float4 vert(uint id : SV_VertexID) : SV_Position { return float4(id == 2 ? 3.0 : -1.0, id == 1 ? 3.0 : -1.0, 0, 1); }\n",
        encoding="utf-8",
    )
    if sys.platform != "win32":
        glslang = shutil.which("glslangValidator")
        if not glslang:
            parser.error("install glslang-tools")
        binaries = []
        for index, (asset, stage, entry) in enumerate(
            [(fragment, "frag", "frag"), (vertex, "vert", "vert")]
        ):
            binary = output / f"shader_{index}.spv"
            run(
                [
                    glslang,
                    "-D",
                    "-V",
                    "-S",
                    stage,
                    "-e",
                    entry,
                    "--auto-map-locations",
                    asset,
                    "-o",
                    binary,
                ]
            )
            binaries.append(binary)
        fragment, vertex = binaries

    runtime_build = output / "runtime-build"
    run(
        [
            "cmake",
            "-S",
            source,
            "-B",
            runtime_build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DUDONCL_HOST_OBJECT=" + str(host_object),
            "-DUDONCL_DEVICE_OBJECT=" + str(device_object),
            "-DUDONCL_BINDINGS_DIR=" + str(output),
        ]
    )
    run([sys.executable, root / "scripts/build.py", "--build-dir", runtime_build])
    executable = runtime_build / ("mnist-cnn-device" + suffix)
    stdout = run([executable, fragment, vertex])
    print(stdout, end="")


if __name__ == "__main__":
    main()
