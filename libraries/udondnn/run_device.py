"""Compile and execute UdonDNN primitives through the fragment backend."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=root / "build/udoncl")
    parser.add_argument("--output", type=Path, default=root / "build/udondnn-device")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if sys.platform != "win32":
        icds = list(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json"))
        if not icds:
            parser.error("install mesa-vulkan-drivers and libvulkan-dev")
        environment["VK_DRIVER_FILES"] = environment["VK_ICD_FILENAMES"] = str(
            icds[0]
        )
    commands = []

    def run(command):
        command = [str(value) for value in command]
        result = subprocess.run(
            command, text=True, capture_output=True, env=environment, timeout=600
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
        line.split("=", 1)[1] for line in cache.splitlines() if line.startswith("LLVM_DIR:")
    )
    sdk = Path(llvm_dir).parents[2]
    clang = sdk / "bin" / ("clang" + suffix)
    compiler = build / "tools/vrccl" / ("vrccl" + suffix)
    translate = build / "tools/vrc-translate" / ("vrc-translate" + suffix)
    opt = build / "tools/vrc-opt" / ("vrc-opt" + suffix)
    mlir_translate = sdk / "bin" / ("mlir-translate" + suffix)
    runtime_flags = ["-fms-runtime-lib=dll"] if sys.platform == "win32" else []

    convolution_device = source / "device/convolution.cl"
    activation_device = source / "device/activation.cl"
    convolution_text = convolution_device.read_text(encoding="utf-8")
    activation_text = activation_device.read_text(encoding="utf-8")
    device_text = convolution_text + "\n" + activation_text
    (output / "udondnn_device_source.h").write_text(
        "#define UDONDNN_DEVICE_SOURCE " + json.dumps(device_text) + "\n",
        encoding="utf-8",
    )

    def emit_device(name, device, kernel_name, kernel_id, words):
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
                "-O0",
                "-Xclang",
                "-disable-O0-optnone",
                "-fomit-frame-pointer",
                "-ffp-contract=off",
                "-S",
                "-emit-llvm",
                device,
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
                    "words": words,
                    "kernel": kernel_name,
                    "kernel_id": kernel_id,
                    "global_size": [words, 1, 1],
                    "local_size": [1, 1, 1],
                    "global_offset": [0, 0, 0],
                }
            ),
            encoding="utf-8",
        )
        bundle = output / f"{name}-bundle"
        with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=output) as staging:
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
                    raise RuntimeError("refusing to replace a non-UdonDNN bundle")
                shutil.rmtree(bundle)
            staged.rename(bundle)
        return bundle

    bundle = emit_device(
        "convolution", convolution_device, "udondnn_convolution_forward", 200, 2048
    )
    activation_bundle = emit_device(
        "activation", activation_device, "udondnn_activation_forward", 201, 2048
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    kernel = manifest["device"]["kernels"][0]
    if (
        kernel["name"] != "udondnn_convolution_forward"
        or kernel["global_size"] != [2048, 1, 1]
        or [argument["required_bytes"] for argument in kernel["args"][:2]]
        != [8192, 8192]
        or len(kernel["args"]) != 9
    ):
        raise RuntimeError("UdonDNN convolution manifest does not match its packed ABI")
    fragment = bundle / "kernels/kernel_200.hlsl"
    fragment_text = fragment.read_text(encoding="utf-8")
    shaderlab = (bundle / "kernels/kernel_200.shader").read_text(encoding="utf-8")
    if '_VRC_Arg6 ("Arg6", Int)' not in shaderlab or "Integer)" in shaderlab:
        raise RuntimeError("UdonDNN integer uniforms are not Unity SetInt-compatible")
    if (
        "mode ==" in convolution_text
        or "? ky" in convolution_text
        or "? kx" in convolution_text
    ):
        raise RuntimeError("UdonDNN convolution mode must not lower through select")
    if "(_VRC_Arg4 >>" not in fragment_text or "(_VRC_Arg5 &" not in fragment_text:
        raise RuntimeError("UdonDNN generated shader lost packed shape or separate mode")
    activation_manifest = json.loads(
        (activation_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    activation_kernel = activation_manifest["device"]["kernels"][0]
    if (
        activation_kernel["name"] != "udondnn_activation_forward"
        or activation_kernel["global_size"] != [2048, 1, 1]
        or [argument["required_bytes"] for argument in activation_kernel["args"][:2]]
        != [8192, 8192]
        or len(activation_kernel["args"]) != 8
    ):
        raise RuntimeError("UdonDNN activation manifest does not match its packed ABI")
    activation_fragment = activation_bundle / "kernels/kernel_201.hlsl"
    activation_shaderlab = (
        activation_bundle / "kernels/kernel_201.shader"
    ).read_text(encoding="utf-8")
    if '_VRC_Arg4 ("Arg4", Int)' not in activation_shaderlab:
        raise RuntimeError("UdonDNN activation integer uniforms are not Unity-compatible")
    (output / "Bindings.h").write_text(
        "inline const char *deviceSource() { return "
        + json.dumps(device_text)
        + "; }\n"
        + "inline std::vector<native_opencl::GatherBinding> manifestBindings("
        + "const std::string &convolution, const std::string &activation) { return {"
        + '{200, 2048, 7, "udondnn_convolution_forward", convolution}, '
        + '{201, 2048, 6, "udondnn_activation_forward", activation}}; }\n',
        encoding="utf-8",
    )

    fixture_input_shape = (1, 2, 4, 4)
    fixture_filter_shape = (3, 2, 3, 3)
    fixture_output_shape = (1, 3, 4, 4)
    fixture_alpha, fixture_beta = 0.75, -0.25
    fixture_input = [
        ((index + 5) * 18 % 31 - 15) * 0.0625
        for index in range(1 * 2 * 4 * 4)
    ]
    fixture_filter = [
        ((index + 5) * 84 % 31 - 15) * 0.0625
        for index in range(3 * 2 * 3 * 3)
    ]
    fixture_previous = [
        ((index + 5) * 150 % 31 - 15) * 0.0625
        for index in range(1 * 3 * 4 * 4)
    ]
    fixture_reference = list(fixture_previous)
    for output_channel in range(3):
        for output_y in range(4):
            for output_x in range(4):
                product = 0.0
                for input_channel in range(2):
                    for kernel_y in range(3):
                        for kernel_x in range(3):
                            input_y = output_y + kernel_y - 1
                            input_x = output_x + kernel_x - 1
                            if 0 <= input_y < 4 and 0 <= input_x < 4:
                                input_index = (
                                    (input_channel * 4 + input_y) * 4 + input_x
                                )
                                filter_index = (
                                    (output_channel * 2 + input_channel) * 3
                                    + kernel_y
                                ) * 3 + kernel_x
                                product += (
                                    fixture_input[input_index]
                                    * fixture_filter[filter_index]
                                )
                output_index = (output_channel * 4 + output_y) * 4 + output_x
                fixture_reference[output_index] = (
                    fixture_alpha * product
                    + fixture_beta * fixture_previous[output_index]
                )
    fixture_packed = [0.0] * 2048
    fixture_packed[0 : len(fixture_input)] = fixture_input
    fixture_packed[512 : 512 + len(fixture_filter)] = fixture_filter
    fixture_packed[1024 : 1024 + len(fixture_previous)] = fixture_previous
    unity_bundle = output / "unity-bundle"
    with tempfile.TemporaryDirectory(prefix="unity-udondnn-", dir=output) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/host.uasm", staged / "host.uasm")
        shutil.copy2(source / "unity/display.shader", staged / "display.shader")
        shutil.copy2(
            bundle / "kernels/kernel_200.shader",
            staged / "kernels/kernel_200.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<2048f", *fixture_packed)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<48f", *fixture_reference)
        )
        unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udondnn-convolution",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udondnn_convolution",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "udondnnConvolutionForward",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 200,
                        "name": "udondnn_convolution_forward",
                        "shader": "kernels/kernel_200.shader",
                    }
                ]
            },
            "resources": [
                {"name": "fixture_input", "kind": "bytes", "path": "fixture_input.bytes"},
                {
                    "name": "fixture_reference",
                    "kind": "bytes",
                    "path": "fixture_reference.bytes",
                },
            ],
            "fixture": {
                "input_shape": fixture_input_shape,
                "filter_shape": fixture_filter_shape,
                "output_shape": fixture_output_shape,
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_bundle.exists():
            previous = json.loads(
                (unity_bundle / "manifest.json").read_text(encoding="utf-8")
            )
            if previous.get("artifact_stage") != "unity-udondnn-convolution":
                raise RuntimeError("refusing to replace a non-UdonDNN Unity bundle")
            shutil.rmtree(unity_bundle)
        staged.rename(unity_bundle)

    activation_words = 64
    activation_input = [
        ((index + 5) * (19 * 11 + 7) % 31 - 15) * 0.25
        for index in range(activation_words)
    ]
    activation_input[7] = math.nan
    activation_previous = [
        ((index + 5) * (31 * 11 + 7) % 31 - 15) * 0.0625
        for index in range(activation_words)
    ]
    activation_cases = [
        (0, 0, 0.0, 1.0, 0.0),
        (1, 0, 0.0, 0.75, -0.25),
        (2, 0, 0.0, -1.25, 0.5),
        (3, 0, 0.8, 1.0, 0.125),
        (4, 0, 1.2, 0.5, -0.5),
        (5, 0, 0.0, -0.75, 1.0),
        (1, 0, 0.0, 1.0, 0.0),
        (1, 1, 0.0, 1.0, 0.0),
    ]

    def activation_reference(value, mode, nan_propagation, coefficient):
        if math.isnan(value):
            if nan_propagation:
                return value
            value = 0.0
        if mode == 0:
            return 1.0 / (1.0 + math.exp(-value))
        if mode == 1:
            return value if value > 0.0 else 0.0
        if mode == 2:
            return 2.0 / (1.0 + math.exp(-2.0 * value)) - 1.0
        if mode == 3:
            return min(coefficient, value if value > 0.0 else 0.0)
        if mode == 4:
            return value if value > 0.0 else coefficient * (math.exp(value) - 1.0)
        if mode == 5:
            return value
        raise RuntimeError(f"unknown activation mode {mode}")

    activation_reference_values = []
    for mode, nan_propagation, coefficient, alpha, beta in activation_cases:
        activation_reference_values.extend(
            alpha
            * activation_reference(value, mode, nan_propagation, coefficient)
            + beta * previous
            for value, previous in zip(activation_input, activation_previous)
        )
    activation_packed = [0.0] * 2048
    activation_packed[:activation_words] = activation_input
    activation_packed[512 : 512 + activation_words] = activation_previous
    unity_activation_bundle = output / "unity-activation-bundle"
    with tempfile.TemporaryDirectory(
        prefix="unity-udondnn-activation-", dir=output
    ) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/activation-host.uasm", staged / "host.uasm")
        shutil.copy2(
            source / "unity/activation-display.shader", staged / "display.shader"
        )
        shutil.copy2(
            activation_bundle / "kernels/kernel_201.shader",
            staged / "kernels/kernel_201.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<2048f", *activation_packed)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<512f", *activation_reference_values)
        )
        activation_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udondnn-activation",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udondnn_activation",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "udondnnActivationForward",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 201,
                        "name": "udondnn_activation_forward",
                        "shader": "kernels/kernel_201.shader",
                    }
                ]
            },
            "resources": [
                {"name": "fixture_input", "kind": "bytes", "path": "fixture_input.bytes"},
                {
                    "name": "fixture_reference",
                    "kind": "bytes",
                    "path": "fixture_reference.bytes",
                },
            ],
            "fixture": {
                "tensor_shape": [1, 2, 4, 8],
                "cases": len(activation_cases),
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(activation_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_activation_bundle.exists():
            previous = json.loads(
                (unity_activation_bundle / "manifest.json").read_text(encoding="utf-8")
            )
            if previous.get("artifact_stage") != "unity-udondnn-activation":
                raise RuntimeError(
                    "refusing to replace a non-UdonDNN activation Unity bundle"
                )
            shutil.rmtree(unity_activation_bundle)
        staged.rename(unity_activation_bundle)

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
            source / "include",
            "-I",
            output,
            "-S",
            "-emit-llvm",
            source / "src/udondnn.c",
            "-o",
            host_ir,
        ]
    )
    run([translate, "--import-llvm", host_ir, "-o", imported])
    run([opt, "--vrc-raise-native-host-calls", "--verify-each", imported, "-o", udonhl])
    run([opt, "--vrc-lower-native-host-calls", "--verify-each", udonhl, "-o", lowered])
    run([mlir_translate, "--mlir-to-llvmir", lowered, "-o", native_ir])
    host_object = output / ("host" + object_suffix)
    device_object = output / ("convolution-device" + object_suffix)
    activation_object = output / ("activation-device" + object_suffix)
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
            convolution_device,
            "-o",
            device_object,
        ]
    )
    run(
        [
            clang,
            *runtime_flags,
            "-x",
            "c",
            "-std=c11",
            "-ffp-contract=off",
            "-c",
            activation_device,
            "-o",
            activation_object,
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
        compiled = []
        for index, (asset, stage, entry) in enumerate(
            [
                (fragment, "frag", "frag"),
                (activation_fragment, "frag", "frag"),
                (vertex, "vert", "vert"),
            ]
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
            compiled.append(binary)
        fragment, activation_fragment, vertex = compiled

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
            "-DUDONDNN_HOST_OBJECT=" + str(host_object),
            "-DUDONDNN_DEVICE_OBJECT=" + str(device_object),
            "-DUDONDNN_ACTIVATION_OBJECT=" + str(activation_object),
            "-DUDONDNN_BINDINGS_DIR=" + str(output),
        ]
    )
    run([sys.executable, root / "scripts/build.py", "--build-dir", runtime_build])
    executable = runtime_build / ("udondnn-device" + suffix)
    print(run([executable, fragment, activation_fragment, vertex]), end="")


if __name__ == "__main__":
    main()
