"""Compile and execute the MNIST CNN as a real Device fragment kernel."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile


def read_weight_array(header, name, expected):
    match = re.search(
        rf"static const float {name}\[{expected}\] = \{{(.*?)\}};",
        header,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"could not find {name}[{expected}] in weights.h")
    values = [float(token.rstrip("f")) for token in match.group(1).split(",") if token.strip()]
    if len(values) != expected:
        raise RuntimeError(f"{name} contains {len(values)} values; expected {expected}")
    return values


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
    kernels = [
        ("mnist_conv", 20, "MNIST_CONV"),
        ("mnist_pool", 21, "MNIST_POOL"),
        ("mnist_dense", 22, "MNIST_DENSE"),
    ]
    bundles = {}
    fragments = []
    for name, identifier, define in kernels:
        device_ir = output / f"{name}.ll"
        run(
            [
                clang,
                "-target",
                "spir-unknown-unknown",
                "-x",
                "cl",
                "-cl-std=CL1.2",
                "-cl-kernel-arg-info",
                f"-D{define}=1",
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
        config = output / f"{name}.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": "gather-f32",
                    "words": 10800,
                    "kernel": name,
                    "kernel_id": identifier,
                    "global_size": [10800, 1, 1],
                    "local_size": [1, 1, 1],
                    "global_offset": [0, 0, 0],
                }
            ),
            encoding="utf-8",
        )
        bundle = output / f"{name}-bundle"
        with tempfile.TemporaryDirectory(prefix=name + "-", dir=output) as staging:
            staged = Path(staging) / "bundle"
            run(
                [
                    compiler,
                    "--emit-native-device-bundle",
                    config,
                    "--device-llvm",
                    device_ir,
                    "-o",
                    staged,
                ]
            )
            if bundle.exists():
                previous = json.loads(
                    (bundle / "manifest.json").read_text(encoding="utf-8")
                )
                if previous.get("artifact_stage") != "native-device-prototype":
                    raise RuntimeError("refusing to replace a non-example bundle")
                shutil.rmtree(bundle)
            staged.rename(bundle)
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        kernel = manifest["device"]["kernels"][0]
        if (
            kernel["name"] != name
            or kernel["global_size"] != [10800, 1, 1]
            or [argument["required_bytes"] for argument in kernel["args"][:2]]
            != [43200, 43200]
        ):
            raise RuntimeError("MNIST gather manifest does not match its packed ABI")
        bundles[identifier] = bundle
        fragments.append(bundle / f"kernels/kernel_{identifier}.hlsl")
    (output / "Bindings.h").write_text(
        "inline const char *deviceSource() { return "
        + json.dumps(device_text)
        + "; }\n"
        + "inline std::vector<native_opencl::GatherBinding> manifestBindings("
        + "const std::string &conv, const std::string &pool, const std::string &dense) { "
        + 'return {{20, 10800, 0, "mnist_conv", conv}, '
        + '{21, 10800, 0, "mnist_pool", pool}, '
        + '{22, 10800, 0, "mnist_dense", dense}}; }\n',
        encoding="utf-8",
    )

    unity_bundle = output / "unity-bundle"
    with tempfile.TemporaryDirectory(prefix="unity-", dir=output) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "host.uasm", staged / "host.uasm")
        shutil.copy2(source / "display.shader", staged / "display.shader")
        shutil.copy2(source / "pack.shader", staged / "pack.shader")
        shutil.copy2(source / "brush.shader", staged / "brush.shader")
        for name, identifier, _ in kernels:
            shutil.copy2(
                bundles[identifier] / f"kernels/kernel_{identifier}.shader",
                staged / f"kernels/kernel_{identifier}.shader",
            )
        header = (source / "weights.h").read_text(encoding="utf-8")
        packed = [0.0] * 10800
        packed[784:1184] = read_weight_array(header, "kMnistConvWeight", 400)
        packed[1184:1200] = read_weight_array(header, "kMnistConvBias", 16)
        packed[1200:2640] = read_weight_array(header, "kMnistDenseWeight", 1440)
        packed[2640:2650] = read_weight_array(header, "kMnistDenseBias", 10)
        (staged / "weights.bytes").write_bytes(struct.pack("<10800f", *packed))
        unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-mnist-prototype",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "mnist_cnn",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "mnist_gpu_predict",
                    "udon_event": "_interact",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {"id": identifier, "name": name, "shader": f"kernels/kernel_{identifier}.shader"}
                    for name, identifier, _ in kernels
                ]
            },
            "resources": [
                {"name": "pack_input", "kind": "shader", "path": "pack.shader"},
                {"name": "brush", "kind": "shader", "path": "brush.shader"},
                {
                    "name": "packed_weights",
                    "kind": "bytes",
                    "path": "weights.bytes",
                },
            ],
        }
        (staged / "manifest.json").write_text(
            json.dumps(unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_bundle.exists():
            previous = json.loads((unity_bundle / "manifest.json").read_text(encoding="utf-8"))
            if previous.get("artifact_stage") != "unity-mnist-prototype":
                raise RuntimeError("refusing to replace a non-example Unity bundle")
            shutil.rmtree(unity_bundle)
        staged.rename(unity_bundle)

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
        shader_inputs = [(asset, "frag", "frag") for asset in fragments]
        shader_inputs.append((vertex, "vert", "vert"))
        for index, (asset, stage, entry) in enumerate(shader_inputs):
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
        fragments, vertex = binaries[:-1], binaries[-1]

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
    stdout = run([executable, *fragments, vertex])
    print(stdout, end="")


if __name__ == "__main__":
    main()
