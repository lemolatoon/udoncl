"""Compile and execute the MNIST CNN as a real Device fragment kernel."""

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time


def publish_bundle(staged, destination):
    """Publish a generated directory after transient Windows scanners release it."""
    for attempt in range(40):
        try:
            staged.rename(destination)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)

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

    onnx_model = output / "mnist.onnx"
    onnx_plan_path = output / "onnx-plan.json"
    run([sys.executable, source / "generate_onnx.py", "-o", onnx_model])
    run(
        [
            sys.executable,
            root / "runtime/onnx/udon_onnx.py",
            onnx_model,
            "-o",
            onnx_plan_path,
        ]
    )
    onnx_plan = json.loads(onnx_plan_path.read_text(encoding="utf-8"))
    expected_operators = [
        ("Conv", "udondnnConvolutionForward"),
        ("Add", "udonatenAdd"),
        ("Relu", "udonatenRelu"),
        ("AveragePool", "udondnnPoolingForward"),
        ("Flatten", "view"),
        ("Gemm", "udonblasSgemm"),
        ("Add", "udonatenAdd"),
    ]
    if [
        (node["op_type"], node["provider"]) for node in onnx_plan["nodes"]
    ] != expected_operators:
        raise RuntimeError(
            "MNIST ONNX graph was not assigned to UdonDNN/UdonBLAS/UdonATen"
        )
    expected_passes = [
        (20, "mnist_conv", 9216),
        (21, "mnist_pool", 144),
        (22, "mnist_dense", 10),
    ]
    if [
        (device_pass["id"], device_pass["kernel"], device_pass["fragments"])
        for device_pass in onnx_plan["passes"]
    ] != expected_passes:
        raise RuntimeError("MNIST ONNX execution pass plan is not Device-compatible")
    sys.path.insert(0, str(root / "runtime/onnx"))
    from udon_onnx import parse_model

    onnx_graph = parse_model(onnx_model.read_bytes())["graph"]

    def onnx_initializer(name, shape):
        tensor = onnx_graph["initializers"].get(name)
        if tensor is None or tensor["shape"] != shape:
            raise RuntimeError(
                f"ONNX initializer {name!r} has shape "
                f"{None if tensor is None else tensor['shape']}; expected {shape}"
            )
        return list(struct.unpack(f"<{len(tensor['raw']) // 4}f", tensor["raw"]))

    device_text = (source / "device.cl").read_text(encoding="utf-8")
    (output / "device_source.h").write_text(
        "#define UDONCL_DEVICE_SOURCE " + json.dumps(device_text) + "\n",
        encoding="utf-8",
    )
    defines = {
        "mnist_conv": "MNIST_CONV",
        "mnist_pool": "MNIST_POOL",
        "mnist_dense": "MNIST_DENSE",
    }
    kernels = [
        (device_pass["kernel"], device_pass["id"], defines[device_pass["kernel"]])
        for device_pass in onnx_plan["passes"]
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
            publish_bundle(staged, bundle)
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
        shutil.copy2(onnx_model, staged / "model.onnx.bytes")
        shutil.copy2(onnx_plan_path, staged / "onnx-plan.bytes")
        for name, identifier, _ in kernels:
            shutil.copy2(
                bundles[identifier] / f"kernels/kernel_{identifier}.shader",
                staged / f"kernels/kernel_{identifier}.shader",
            )
        conv_weights = onnx_initializer("conv.weight", [16, 1, 5, 5])
        conv_bias = onnx_initializer("conv.bias", [1, 16, 1, 1])
        onnx_dense_weights = onnx_initializer("dense.weight", [144, 10])
        dense_bias = onnx_initializer("dense.bias", [10])
        dense_weights = [0.0] * 1440
        for channel in range(16):
            for pool_pixel in range(9):
                onnx_feature = channel * 9 + pool_pixel
                runtime_feature = pool_pixel * 16 + channel
                for digit in range(10):
                    dense_weights[digit * 144 + runtime_feature] = (
                        onnx_dense_weights[onnx_feature * 10 + digit]
                    )
        packed = [0.0] * 10800
        packed[784:1184] = conv_weights
        packed[1184:1200] = conv_bias
        packed[1200:2640] = dense_weights
        packed[2640:2650] = dense_bias
        (staged / "weights.bytes").write_bytes(struct.pack("<10800f", *packed))
        blank_pooled = [max(conv_bias[feature % 16], 0.0) for feature in range(144)]
        blank_logits = [
            dense_bias[digit]
            + sum(
                dense_weights[digit * 144 + feature] * blank_pooled[feature]
                for feature in range(144)
            )
            for digit in range(10)
        ]
        (staged / "reference_blank_logits.bytes").write_bytes(
            struct.pack("<10f", *blank_logits)
        )
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
                {
                    "name": "reference_blank_logits",
                    "kind": "bytes",
                    "path": "reference_blank_logits.bytes",
                },
                {
                    "name": "onnx_model",
                    "kind": "bytes",
                    "path": "model.onnx.bytes",
                },
                {
                    "name": "onnx_execution_plan",
                    "kind": "bytes",
                    "path": "onnx-plan.bytes",
                },
            ],
            "onnx": {
                "ir_version": onnx_plan["ir_version"],
                "opset": onnx_plan["opset"],
                "nodes": len(onnx_plan["nodes"]),
                "device_passes": len(onnx_plan["passes"]),
                "providers": sorted(
                    {node["provider"] for node in onnx_plan["nodes"]}
                ),
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_bundle.exists():
            previous = json.loads((unity_bundle / "manifest.json").read_text(encoding="utf-8"))
            if previous.get("artifact_stage") != "unity-mnist-prototype":
                raise RuntimeError("refusing to replace a non-example Unity bundle")
            shutil.rmtree(unity_bundle)
        publish_bundle(staged, unity_bundle)

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
            "-DUDONCL_HOST_OBJECT=" + host_object.as_posix(),
            "-DUDONCL_DEVICE_OBJECT=" + device_object.as_posix(),
            "-DUDONCL_BINDINGS_DIR=" + output.as_posix(),
        ]
    )
    run([sys.executable, root / "scripts/build.py", "--build-dir", runtime_build])
    executable = runtime_build / ("mnist-cnn-device" + suffix)
    stdout = run([executable, *fragments, vertex])
    print(stdout, end="")


if __name__ == "__main__":
    main()
