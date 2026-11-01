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
    pooling_device = source / "device/pooling.cl"
    softmax_device = source / "device/softmax.cl"
    batchnorm_device = source / "device/batchnorm.cl"
    op_tensor_device = source / "device/op_tensor.cl"
    rms_norm_device = source / "device/rms_norm.cl"
    gather_device = source / "device/gather.cl"
    int4_device = root / "libraries/udonaten/device/weight_int4pack_mm.cl"
    convolution_text = convolution_device.read_text(encoding="utf-8")
    activation_text = activation_device.read_text(encoding="utf-8")
    pooling_text = pooling_device.read_text(encoding="utf-8")
    softmax_text = softmax_device.read_text(encoding="utf-8")
    batchnorm_text = batchnorm_device.read_text(encoding="utf-8")
    op_tensor_text = op_tensor_device.read_text(encoding="utf-8")
    rms_norm_text = rms_norm_device.read_text(encoding="utf-8")
    gather_text = gather_device.read_text(encoding="utf-8")
    int4_text = int4_device.read_text(encoding="utf-8")
    device_text = (
        convolution_text
        + "\n"
        + activation_text
        + "\n"
        + pooling_text
        + "\n"
        + softmax_text
        + "\n"
        + batchnorm_text
        + "\n"
        + op_tensor_text
        + "\n"
        + rms_norm_text
        + "\n"
        + gather_text
        + "\n"
        + int4_text
    )
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
            publish_bundle(staged, bundle)
        return bundle

    bundle = emit_device(
        "convolution", convolution_device, "udondnn_convolution_forward", 200, 2048
    )
    activation_bundle = emit_device(
        "activation", activation_device, "udondnn_activation_forward", 201, 2048
    )
    pooling_bundle = emit_device(
        "pooling", pooling_device, "udondnn_pooling_forward", 202, 2048
    )
    softmax_bundle = emit_device(
        "softmax", softmax_device, "udondnn_softmax_forward", 203, 2048
    )
    batchnorm_bundle = emit_device(
        "batchnorm", batchnorm_device, "udondnn_batchnorm_inference", 204, 4096
    )
    op_tensor_bundle = emit_device(
        "op-tensor", op_tensor_device, "udondnn_op_tensor", 205, 2048
    )
    rms_norm_bundle = emit_device(
        "rms-norm",
        rms_norm_device,
        "udondnn_ext_rms_normalization_forward",
        206,
        2048,
    )
    gather_bundle = emit_device(
        "gather", gather_device, "udondnn_ext_gather_forward", 207, 2048
    )
    int4_bundle = emit_device(
        "weight-int4pack-mm",
        int4_device,
        "udonaten_weight_int4pack_mm",
        208,
        8192,
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
    pooling_manifest = json.loads(
        (pooling_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    pooling_kernel = pooling_manifest["device"]["kernels"][0]
    if (
        pooling_kernel["name"] != "udondnn_pooling_forward"
        or pooling_kernel["global_size"] != [2048, 1, 1]
        or [argument["required_bytes"] for argument in pooling_kernel["args"][:2]]
        != [8192, 8192]
        or len(pooling_kernel["args"]) != 9
    ):
        raise RuntimeError("UdonDNN pooling manifest does not match its packed ABI")
    pooling_fragment = pooling_bundle / "kernels/kernel_202.hlsl"
    pooling_shaderlab = (pooling_bundle / "kernels/kernel_202.shader").read_text(
        encoding="utf-8"
    )
    if '_VRC_Arg5 ("Arg5", Int)' not in pooling_shaderlab:
        raise RuntimeError("UdonDNN pooling integer uniforms are not Unity-compatible")
    softmax_manifest = json.loads(
        (softmax_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    softmax_kernel = softmax_manifest["device"]["kernels"][0]
    if (
        softmax_kernel["name"] != "udondnn_softmax_forward"
        or softmax_kernel["global_size"] != [2048, 1, 1]
        or [argument["required_bytes"] for argument in softmax_kernel["args"][:2]]
        != [8192, 8192]
        or len(softmax_kernel["args"]) != 7
    ):
        raise RuntimeError("UdonDNN softmax manifest does not match its packed ABI")
    softmax_fragment = softmax_bundle / "kernels/kernel_203.hlsl"
    softmax_fragment_text = softmax_fragment.read_text(encoding="utf-8")
    if "log(" not in softmax_fragment_text:
        raise RuntimeError("UdonDNN softmax LOG algorithm lost generic log lowering")
    softmax_shaderlab = (softmax_bundle / "kernels/kernel_203.shader").read_text(
        encoding="utf-8"
    )
    if '_VRC_Arg4 ("Arg4", Int)' not in softmax_shaderlab:
        raise RuntimeError("UdonDNN softmax integer uniforms are not Unity-compatible")
    batchnorm_manifest = json.loads(
        (batchnorm_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    batchnorm_kernel = batchnorm_manifest["device"]["kernels"][0]
    if (
        batchnorm_kernel["name"] != "udondnn_batchnorm_inference"
        or batchnorm_kernel["global_size"] != [4096, 1, 1]
        or [argument["required_bytes"] for argument in batchnorm_kernel["args"][:2]]
        != [16384, 16384]
        or len(batchnorm_kernel["args"]) != 7
    ):
        raise RuntimeError("UdonDNN batchnorm manifest does not match its packed ABI")
    batchnorm_fragment = batchnorm_bundle / "kernels/kernel_204.hlsl"
    batchnorm_shaderlab = (
        batchnorm_bundle / "kernels/kernel_204.shader"
    ).read_text(encoding="utf-8")
    if '_VRC_Arg3 ("Arg3", Int)' not in batchnorm_shaderlab:
        raise RuntimeError("UdonDNN batchnorm integer uniforms are not Unity-compatible")
    op_tensor_manifest = json.loads(
        (op_tensor_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    op_tensor_kernel = op_tensor_manifest["device"]["kernels"][0]
    if (
        op_tensor_kernel["name"] != "udondnn_op_tensor"
        or op_tensor_kernel["global_size"] != [2048, 1, 1]
        or [argument["required_bytes"] for argument in op_tensor_kernel["args"][:2]]
        != [8192, 8192]
        or len(op_tensor_kernel["args"]) != 10
    ):
        raise RuntimeError("UdonDNN OpTensor manifest does not match its packed ABI")
    op_tensor_fragment = op_tensor_bundle / "kernels/kernel_205.hlsl"
    op_tensor_shaderlab = (
        op_tensor_bundle / "kernels/kernel_205.shader"
    ).read_text(encoding="utf-8")
    if '_VRC_Arg6 ("Arg6", Int)' not in op_tensor_shaderlab:
        raise RuntimeError("UdonDNN OpTensor integer uniforms are not Unity-compatible")
    rms_norm_manifest = json.loads(
        (rms_norm_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    rms_norm_kernel = rms_norm_manifest["device"]["kernels"][0]
    if (
        rms_norm_kernel["name"] != "udondnn_ext_rms_normalization_forward"
        or rms_norm_kernel["global_size"] != [2048, 1, 1]
        or [argument["required_bytes"] for argument in rms_norm_kernel["args"][:2]]
        != [8192, 8192]
        or len(rms_norm_kernel["args"]) != 7
    ):
        raise RuntimeError("UdonDNN RMSNorm manifest does not match its packed ABI")
    rms_norm_fragment = rms_norm_bundle / "kernels/kernel_206.hlsl"
    rms_norm_shaderlab = (rms_norm_bundle / "kernels/kernel_206.shader").read_text(
        encoding="utf-8"
    )
    if '_VRC_Arg3 ("Arg3", Int)' not in rms_norm_shaderlab:
        raise RuntimeError("UdonDNN RMSNorm integer uniforms are not Unity-compatible")
    gather_manifest = json.loads(
        (gather_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    gather_kernel = gather_manifest["device"]["kernels"][0]
    if (
        gather_kernel["name"] != "udondnn_ext_gather_forward"
        or gather_kernel["global_size"] != [2048, 1, 1]
        or [argument["required_bytes"] for argument in gather_kernel["args"][:2]]
        != [8192, 8192]
        or len(gather_kernel["args"]) != 8
    ):
        raise RuntimeError("UdonDNN Gather manifest does not match its packed ABI")
    gather_fragment = gather_bundle / "kernels/kernel_207.hlsl"
    gather_shaderlab = (gather_bundle / "kernels/kernel_207.shader").read_text(
        encoding="utf-8"
    )
    if '_VRC_Arg2 ("Arg2", Int)' not in gather_shaderlab:
        raise RuntimeError("UdonDNN Gather integer uniforms are not Unity-compatible")
    int4_manifest = json.loads(
        (int4_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    int4_kernel = int4_manifest["device"]["kernels"][0]
    if (
        int4_kernel["name"] != "udonaten_weight_int4pack_mm"
        or int4_kernel["global_size"] != [8192, 1, 1]
        or [argument["required_bytes"] for argument in int4_kernel["args"][:2]]
        != [32768, 32768]
        or len(int4_kernel["args"]) != 6
    ):
        raise RuntimeError("UdonATen INT4 manifest does not match its packed ABI")
    int4_fragment = int4_bundle / "kernels/kernel_208.hlsl"
    int4_shaderlab = (int4_bundle / "kernels/kernel_208.shader").read_text(
        encoding="utf-8"
    )
    if '_VRC_Arg5 ("Arg5", Int)' not in int4_shaderlab:
        raise RuntimeError("UdonATen INT4 integer uniforms are not Unity-compatible")
    (output / "Bindings.h").write_text(
        "inline const char *deviceSource() { return "
        + json.dumps(device_text)
        + "; }\n"
        + "inline std::vector<native_opencl::GatherBinding> manifestBindings("
        + "const std::string &convolution, const std::string &activation, "
        + "const std::string &pooling, const std::string &softmax, "
        + "const std::string &batchnorm, const std::string &opTensor, "
        + "const std::string &rmsNorm, const std::string &gather, "
        + "const std::string &int4) { return {"
        + '{200, 2048, 7, "udondnn_convolution_forward", convolution}, '
        + '{201, 2048, 6, "udondnn_activation_forward", activation}, '
        + '{202, 2048, 7, "udondnn_pooling_forward", pooling}, '
        + '{203, 2048, 5, "udondnn_softmax_forward", softmax}, '
        + '{204, 4096, 5, "udondnn_batchnorm_inference", batchnorm}, '
        + '{205, 2048, 8, "udondnn_op_tensor", opTensor}, '
        + '{206, 2048, 5, "udondnn_ext_rms_normalization_forward", rmsNorm}, '
        + '{207, 2048, 6, "udondnn_ext_gather_forward", gather}, '
        + '{208, 8192, 4, "udonaten_weight_int4pack_mm", int4}}; }\n',
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
        publish_bundle(staged, unity_bundle)

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
        publish_bundle(staged, unity_activation_bundle)

    pooling_input_shape = (1, 2, 4, 4)
    pooling_output_shape = (1, 2, 2, 2)
    pooling_window = (3, 3)
    pooling_padding = (1, 1)
    pooling_stride = (2, 2)
    pooling_input = [
        ((index + 5) * (61 * 11 + 7) % 31 - 15) * 0.125
        for index in range(32)
    ]
    pooling_input[7] = math.nan
    pooling_previous = [
        ((index + 5) * (67 * 11 + 7) % 31 - 15) * 0.0625
        for index in range(8)
    ]
    pooling_cases = [
        (0, 0, 1.0, 0.0),
        (3, 0, 0.75, -0.25),
        (1, 0, -1.0, 0.5),
        (2, 0, 1.25, 0.125),
        (0, 1, 1.0, 0.0),
        (0, 0, 0.5, -0.5),
    ]

    def pooling_reference(mode, nan_propagation, alpha, beta):
        values = []
        for channel in range(2):
            for output_y in range(2):
                for output_x in range(2):
                    samples = []
                    has_nan = False
                    for kernel_y in range(3):
                        for kernel_x in range(3):
                            input_y = output_y * 2 + kernel_y - 1
                            input_x = output_x * 2 + kernel_x - 1
                            if 0 <= input_y < 4 and 0 <= input_x < 4:
                                sample = pooling_input[
                                    (channel * 4 + input_y) * 4 + input_x
                                ]
                                if math.isnan(sample):
                                    if nan_propagation:
                                        has_nan = True
                                    sample = 0.0
                                samples.append(sample)
                    if has_nan:
                        pooled = math.nan
                    elif mode == 1:
                        pooled = sum(samples) / 9.0
                    elif mode == 2:
                        pooled = sum(samples) / len(samples)
                    else:
                        pooled = max(samples) if samples else 0.0
                    element = (channel * 2 + output_y) * 2 + output_x
                    values.append(alpha * pooled + beta * pooling_previous[element])
        return values

    pooling_reference_values = []
    for pooling_case in pooling_cases:
        pooling_reference_values.extend(pooling_reference(*pooling_case))
    pooling_packed = [0.0] * 2048
    pooling_packed[: len(pooling_input)] = pooling_input
    pooling_packed[512 : 512 + len(pooling_previous)] = pooling_previous
    unity_pooling_bundle = output / "unity-pooling-bundle"
    with tempfile.TemporaryDirectory(
        prefix="unity-udondnn-pooling-", dir=output
    ) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/pooling-host.uasm", staged / "host.uasm")
        shutil.copy2(
            source / "unity/pooling-display.shader", staged / "display.shader"
        )
        shutil.copy2(
            pooling_bundle / "kernels/kernel_202.shader",
            staged / "kernels/kernel_202.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<2048f", *pooling_packed)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<48f", *pooling_reference_values)
        )
        pooling_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udondnn-pooling",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udondnn_pooling",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "udondnnPoolingForward",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 202,
                        "name": "udondnn_pooling_forward",
                        "shader": "kernels/kernel_202.shader",
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
                "input_shape": pooling_input_shape,
                "output_shape": pooling_output_shape,
                "window": pooling_window,
                "padding": pooling_padding,
                "stride": pooling_stride,
                "cases": len(pooling_cases),
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(pooling_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_pooling_bundle.exists():
            previous = json.loads(
                (unity_pooling_bundle / "manifest.json").read_text(encoding="utf-8")
            )
            if previous.get("artifact_stage") != "unity-udondnn-pooling":
                raise RuntimeError(
                    "refusing to replace a non-UdonDNN pooling Unity bundle"
                )
            shutil.rmtree(unity_pooling_bundle)
        publish_bundle(staged, unity_pooling_bundle)

    softmax_input_shape = (1, 3, 2, 2)
    softmax_input = [
        ((index + 5) * (83 * 11 + 7) % 31 - 15) * 0.5 for index in range(12)
    ]
    softmax_previous = [
        ((index + 5) * (89 * 11 + 7) % 31 - 15) * 0.0625
        for index in range(12)
    ]
    softmax_cases = [
        (0, 0, 1.0, 0.0),
        (1, 0, 0.75, -0.25),
        (2, 0, -1.0, 0.5),
        (0, 1, 1.25, 0.125),
        (1, 1, 0.5, -0.5),
        (2, 1, -0.75, 1.0),
    ]

    def softmax_reference(algorithm, mode, alpha, beta):
        values = []
        for element in range(12):
            x = element % 2
            y = (element // 2) % 2
            if mode == 0:
                group = softmax_input
            else:
                group = [softmax_input[(channel * 2 + y) * 2 + x] for channel in range(3)]
            maximum = max(group)
            denominator = sum(math.exp(value - maximum) for value in group)
            normalized = math.exp(softmax_input[element] - maximum) / denominator
            if algorithm == 2:
                normalized = softmax_input[element] - maximum - math.log(denominator)
            values.append(alpha * normalized + beta * softmax_previous[element])
        return values

    softmax_reference_values = []
    for softmax_case in softmax_cases:
        softmax_reference_values.extend(softmax_reference(*softmax_case))
    softmax_packed = [0.0] * 2048
    softmax_packed[: len(softmax_input)] = softmax_input
    softmax_packed[512 : 512 + len(softmax_previous)] = softmax_previous
    unity_softmax_bundle = output / "unity-softmax-bundle"
    with tempfile.TemporaryDirectory(
        prefix="unity-udondnn-softmax-", dir=output
    ) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/softmax-host.uasm", staged / "host.uasm")
        shutil.copy2(
            source / "unity/softmax-display.shader", staged / "display.shader"
        )
        shutil.copy2(
            softmax_bundle / "kernels/kernel_203.shader",
            staged / "kernels/kernel_203.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<2048f", *softmax_packed)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<72f", *softmax_reference_values)
        )
        softmax_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udondnn-softmax",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udondnn_softmax",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "udondnnSoftmaxForward",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 203,
                        "name": "udondnn_softmax_forward",
                        "shader": "kernels/kernel_203.shader",
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
                "input_shape": softmax_input_shape,
                "cases": len(softmax_cases),
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(softmax_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_softmax_bundle.exists():
            previous = json.loads(
                (unity_softmax_bundle / "manifest.json").read_text(encoding="utf-8")
            )
            if previous.get("artifact_stage") != "unity-udondnn-softmax":
                raise RuntimeError(
                    "refusing to replace a non-UdonDNN softmax Unity bundle"
                )
            shutil.rmtree(unity_softmax_bundle)
        publish_bundle(staged, unity_softmax_bundle)

    batchnorm_input_shape = (1, 3, 2, 2)
    batchnorm_input = [
        ((index + 5) * (97 * 11 + 7) % 31 - 15) * 0.25 for index in range(12)
    ]
    batchnorm_previous = [
        ((index + 5) * (101 * 11 + 7) % 31 - 15) * 0.0625
        for index in range(12)
    ]
    batchnorm_scale = [
        ((index + 3) * 17 % 19 - 9) * 0.125 for index in range(12)
    ]
    batchnorm_bias = [
        ((index + 7) * 23 % 17 - 8) * 0.0625 for index in range(12)
    ]
    batchnorm_mean = [
        ((index + 11) * 29 % 23 - 11) * 0.125 for index in range(12)
    ]
    batchnorm_variance = [0.25 + ((index * 13 + 5) % 17) * 0.125 for index in range(12)]
    batchnorm_cases = [
        (0, 1.0e-5, 1.0, 0.0),
        (1, 1.0e-4, 0.75, -0.25),
        (2, 1.0e-3, -1.25, 0.5),
        (0, 1.0e-3, 0.5, 0.125),
        (1, 1.0e-5, -0.75, 1.0),
        (2, 1.0e-4, 1.25, -0.5),
    ]

    def batchnorm_reference(mode, epsilon, alpha, beta):
        values = []
        for element, input_value in enumerate(batchnorm_input):
            parameter = element if mode == 0 else element // 4
            normalized = (input_value - batchnorm_mean[parameter]) / math.sqrt(
                batchnorm_variance[parameter] + epsilon
            )
            transformed = (
                normalized * batchnorm_scale[parameter] + batchnorm_bias[parameter]
            )
            values.append(
                alpha * transformed + beta * batchnorm_previous[element]
            )
        return values

    batchnorm_reference_values = []
    for batchnorm_case in batchnorm_cases:
        batchnorm_reference_values.extend(batchnorm_reference(*batchnorm_case))
    batchnorm_packed = [0.0] * 4096
    batchnorm_packed[: len(batchnorm_input)] = batchnorm_input
    batchnorm_packed[512 : 512 + len(batchnorm_previous)] = batchnorm_previous
    batchnorm_packed[1024 : 1024 + len(batchnorm_scale)] = batchnorm_scale
    batchnorm_packed[1536 : 1536 + len(batchnorm_bias)] = batchnorm_bias
    batchnorm_packed[2048 : 2048 + len(batchnorm_mean)] = batchnorm_mean
    batchnorm_packed[2560 : 2560 + len(batchnorm_variance)] = batchnorm_variance
    unity_batchnorm_bundle = output / "unity-batchnorm-bundle"
    with tempfile.TemporaryDirectory(
        prefix="unity-udondnn-batchnorm-", dir=output
    ) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/batchnorm-host.uasm", staged / "host.uasm")
        shutil.copy2(
            source / "unity/batchnorm-display.shader", staged / "display.shader"
        )
        shutil.copy2(
            batchnorm_bundle / "kernels/kernel_204.shader",
            staged / "kernels/kernel_204.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<4096f", *batchnorm_packed)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<72f", *batchnorm_reference_values)
        )
        batchnorm_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udondnn-batchnorm",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udondnn_batchnorm",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "udondnnBatchNormalizationForwardInference",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 204,
                        "name": "udondnn_batchnorm_inference",
                        "shader": "kernels/kernel_204.shader",
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
                "input_shape": batchnorm_input_shape,
                "cases": len(batchnorm_cases),
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(batchnorm_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_batchnorm_bundle.exists():
            previous = json.loads(
                (unity_batchnorm_bundle / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            if previous.get("artifact_stage") != "unity-udondnn-batchnorm":
                raise RuntimeError(
                    "refusing to replace a non-UdonDNN batchnorm Unity bundle"
                )
            shutil.rmtree(unity_batchnorm_bundle)
        publish_bundle(staged, unity_batchnorm_bundle)

    op_tensor_output_shape = (1, 3, 2, 2)
    op_tensor_a = [
        ((index + 5) * (109 * 11 + 7) % 31 - 15) * 0.25
        for index in range(12)
    ]
    op_tensor_b = [
        ((index + 5) * (113 * 11 + 7) % 31 - 15) * 0.1875
        for index in range(12)
    ]
    op_tensor_previous = [
        ((index + 5) * (127 * 11 + 7) % 31 - 15) * 0.0625
        for index in range(12)
    ]
    op_tensor_a[10] = math.nan
    op_tensor_cases = [
        ((1, 3, 2, 2), (1, 3, 1, 1), op_tensor_output_shape, 0, 0, 1.0, 0.75, -0.25),
        ((1, 1, 2, 2), (1, 3, 2, 2), op_tensor_output_shape, 1, 0, -1.0, 0.5, 0.25),
        ((1, 1, 1, 2), (1, 1, 2, 1), op_tensor_output_shape, 2, 0, 1.25, -0.75, 0.0),
        ((1, 1, 1, 1), (1, 3, 2, 2), op_tensor_output_shape, 3, 1, 0.5, 1.5, -0.5),
        ((1, 3, 2, 2), (1, 3, 2, 2), op_tensor_output_shape, 2, 1, 0.75, 1.25, 0.125),
        ((1, 3, 2, 2), (1, 3, 2, 2), op_tensor_output_shape, 3, 0, 1.0, -1.0, 0.5),
    ]

    def tensor_words(shape):
        return math.prod(shape)

    def packed_shape(shape):
        return shape[0] | (shape[1] << 4) | (shape[2] << 8) | (shape[3] << 12)

    def broadcast_index(element, source_shape, output_shape):
        x = element % output_shape[3]
        y = (element // output_shape[3]) % output_shape[2]
        channel = (element // (output_shape[3] * output_shape[2])) % output_shape[1]
        batch = element // (output_shape[3] * output_shape[2] * output_shape[1])
        if source_shape[0] == 1:
            batch = 0
        if source_shape[1] == 1:
            channel = 0
        if source_shape[2] == 1:
            y = 0
        if source_shape[3] == 1:
            x = 0
        return ((batch * source_shape[1] + channel) * source_shape[2] + y) * source_shape[3] + x

    def op_tensor_reference(case):
        a_shape, b_shape, c_shape, operation, nan_propagation, alpha_a, alpha_b, beta = case
        values = []
        for element in range(tensor_words(c_shape)):
            left = alpha_a * op_tensor_a[broadcast_index(element, a_shape, c_shape)]
            right = alpha_b * op_tensor_b[broadcast_index(element, b_shape, c_shape)]
            if operation == 1:
                combined = left * right
            elif operation in (2, 3):
                if nan_propagation and (math.isnan(left) or math.isnan(right)):
                    combined = math.nan
                elif math.isnan(left):
                    combined = 0.0 if math.isnan(right) else right
                elif math.isnan(right):
                    combined = left
                elif operation == 2:
                    combined = min(left, right)
                else:
                    combined = max(left, right)
            else:
                combined = left + right
            values.append(combined + beta * op_tensor_previous[element])
        return values

    op_tensor_reference_values = []
    for op_tensor_case in op_tensor_cases:
        op_tensor_reference_values.extend(op_tensor_reference(op_tensor_case))
    op_tensor_packed = [0.0] * 2048
    op_tensor_packed[: len(op_tensor_a)] = op_tensor_a
    op_tensor_packed[512 : 512 + len(op_tensor_b)] = op_tensor_b
    op_tensor_packed[1024 : 1024 + len(op_tensor_previous)] = op_tensor_previous
    unity_op_tensor_bundle = output / "unity-op-tensor-bundle"
    with tempfile.TemporaryDirectory(
        prefix="unity-udondnn-op-tensor-", dir=output
    ) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/op-tensor-host.uasm", staged / "host.uasm")
        shutil.copy2(
            source / "unity/op-tensor-display.shader", staged / "display.shader"
        )
        shutil.copy2(
            op_tensor_bundle / "kernels/kernel_205.shader",
            staged / "kernels/kernel_205.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<2048f", *op_tensor_packed)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<72f", *op_tensor_reference_values)
        )
        op_tensor_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udondnn-op-tensor",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udondnn_op_tensor",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "udondnnOpTensor",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 205,
                        "name": "udondnn_op_tensor",
                        "shader": "kernels/kernel_205.shader",
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
                "output_shape": op_tensor_output_shape,
                "cases": len(op_tensor_cases),
                "shapes": [
                    {
                        "a": case[0],
                        "b": case[1],
                        "c": case[2],
                        "a_packed": packed_shape(case[0]),
                        "b_packed": packed_shape(case[1]),
                        "c_packed": packed_shape(case[2]),
                    }
                    for case in op_tensor_cases
                ],
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(op_tensor_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_op_tensor_bundle.exists():
            previous = json.loads(
                (unity_op_tensor_bundle / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            if previous.get("artifact_stage") != "unity-udondnn-op-tensor":
                raise RuntimeError(
                    "refusing to replace a non-UdonDNN OpTensor Unity bundle"
                )
            shutil.rmtree(unity_op_tensor_bundle)
        publish_bundle(staged, unity_op_tensor_bundle)

    rms_input = [((index * 17 + 5) % 29 - 14) * 0.125 for index in range(16)]
    rms_scale = [0.5 + 0.125 * index for index in range(8)]
    rms_previous = [((index * 11 + 3) % 19 - 9) * 0.0625 for index in range(16)]
    rms_cases = [
        (1.0e-5, 1.0, 0.0),
        (1.0e-4, 0.75, -0.25),
        (1.0e-3, -1.25, 0.5),
        (0.0, 0.5, 1.0),
    ]

    def rms_reference(epsilon, alpha, beta):
        values = []
        for row in range(2):
            square_sum = sum(rms_input[row * 8 + column] ** 2 for column in range(8))
            inverse_rms = 1.0 / math.sqrt(square_sum / 8.0 + epsilon)
            for column in range(8):
                element = row * 8 + column
                values.append(
                    alpha * rms_input[element] * inverse_rms * rms_scale[column]
                    + beta * rms_previous[element]
                )
        return values

    rms_reference_values = []
    for rms_case in rms_cases:
        rms_reference_values.extend(rms_reference(*rms_case))
    rms_packed = [0.0] * 2048
    rms_packed[: len(rms_input)] = rms_input
    rms_packed[512 : 512 + len(rms_scale)] = rms_scale
    rms_packed[1024 : 1024 + len(rms_previous)] = rms_previous
    unity_rms_bundle = output / "unity-rms-norm-bundle"
    with tempfile.TemporaryDirectory(prefix="unity-udondnn-rms-", dir=output) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/rms-norm-host.uasm", staged / "host.uasm")
        shutil.copy2(
            source / "unity/rms-norm-display.shader", staged / "display.shader"
        )
        shutil.copy2(
            rms_norm_bundle / "kernels/kernel_206.shader",
            staged / "kernels/kernel_206.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(struct.pack("<2048f", *rms_packed))
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<64f", *rms_reference_values)
        )
        rms_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udondnn-rms-norm",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udondnn_rms_norm",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "udondnnExtRmsNormalizationForward",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 206,
                        "name": "udondnn_ext_rms_normalization_forward",
                        "shader": "kernels/kernel_206.shader",
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
            "fixture": {"input_shape": [2, 1, 1, 8], "cases": len(rms_cases)},
        }
        (staged / "manifest.json").write_text(
            json.dumps(rms_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_rms_bundle.exists():
            previous = json.loads(
                (unity_rms_bundle / "manifest.json").read_text(encoding="utf-8")
            )
            if previous.get("artifact_stage") != "unity-udondnn-rms-norm":
                raise RuntimeError("refusing to replace a non-UdonDNN RMSNorm Unity bundle")
            shutil.rmtree(unity_rms_bundle)
        publish_bundle(staged, unity_rms_bundle)

    # Package the public ATen-shaped INT4 tile separately from the legacy
    # UdonBLAS Q4 backend. Build semantic packed-weight and [group,N,2]
    # scale/zero inputs first, then expand only the active Device tile.
    int4_n, int4_k, int4_group = 8, 64, 32
    int4_row_stride = (int4_k // int4_group) * 18
    int4_input = [
        ((inner * 13 + 5) % 29 - 14) * 0.0625 for inner in range(int4_k)
    ]
    int4_packed = [0] * (int4_n * int4_k // 2)
    int4_qparams = [0.0] * ((int4_k // int4_group) * int4_n * 2)
    for block in range(int4_k // int4_group):
        for row in range(int4_n):
            qparam = (block * int4_n + row) * 2
            int4_qparams[qparam] = 0.03125 * (row + block + 1)
            int4_qparams[qparam + 1] = 0.015625 * (row - block)
    for row in range(int4_n):
        for inner in range(0, int4_k, 2):
            low = (row * 5 + inner * 3 + 1) % 16 - 8
            high = (row * 7 + (inner + 1) * 5 + 3) % 16 - 8
            int4_packed[row * (int4_k // 2) + inner // 2] = (
                (low & 15) | ((high & 15) << 4)
            )
    int4_reference = [0.0] * int4_n
    for row in range(int4_n):
        for inner in range(int4_k):
            packed = int4_packed[row * (int4_k // 2) + inner // 2]
            nibble = packed >> 4 if inner % 2 else packed & 15
            quantized = nibble - 16 if nibble >= 8 else nibble
            block = inner // int4_group
            qparam = (block * int4_n + row) * 2
            int4_reference[row] += (
                quantized * int4_qparams[qparam] + int4_qparams[qparam + 1]
            ) * int4_input[inner]
    int4_state = [0.0] * 8192
    for row in range(int4_n):
        for block in range(int4_k // int4_group):
            state_base = row * int4_row_stride + block * 18
            packed_base = row * (int4_k // 2) + block * 16
            for byte in range(16):
                int4_state[state_base + byte] = float(
                    int4_packed[packed_base + byte]
                )
            qparam = (block * int4_n + row) * 2
            int4_state[state_base + 16] = int4_qparams[qparam]
            int4_state[state_base + 17] = int4_qparams[qparam + 1]
    int4_state[6144 : 6144 + int4_k] = int4_input
    int4_accumulator = [
        ((row * 11 + 3) % 17 - 8) * 0.125 for row in range(int4_n)
    ]
    int4_accumulate_state = list(int4_state)
    int4_accumulate_state[7168 : 7168 + int4_n] = int4_accumulator
    int4_reference_values = int4_reference + [
        value + accumulator
        for value, accumulator in zip(int4_reference, int4_accumulator)
    ]

    unity_int4_bundle = output / "unity-udonaten-int4-mm-bundle"
    with tempfile.TemporaryDirectory(
        prefix="unity-udonaten-int4-mm-", dir=output
    ) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(
            root / "libraries/udonaten/unity/weight-int4pack-mm-host.uasm",
            staged / "host.uasm",
        )
        shutil.copy2(
            root / "libraries/udonaten/unity/weight-int4pack-mm-display.shader",
            staged / "display.shader",
        )
        shutil.copy2(
            int4_bundle / "kernels/kernel_208.shader",
            staged / "kernels/kernel_208.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<8192f", *int4_state)
        )
        (staged / "fixture_accumulate_input.bytes").write_bytes(
            struct.pack("<8192f", *int4_accumulate_state)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack(f"<{len(int4_reference_values)}f", *int4_reference_values)
        )
        int4_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udonaten-int4-mm",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udonaten_weight_int4pack_mm",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "udonatenWeightInt4packMm",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 208,
                        "name": "udonaten_weight_int4pack_mm",
                        "shader": "kernels/kernel_208.shader",
                    }
                ]
            },
            "resources": [
                {"name": "fixture_input", "kind": "bytes", "path": "fixture_input.bytes"},
                {
                    "name": "fixture_accumulate_input",
                    "kind": "bytes",
                    "path": "fixture_accumulate_input.bytes",
                },
                {
                    "name": "fixture_reference",
                    "kind": "bytes",
                    "path": "fixture_reference.bytes",
                },
            ],
            "fixture": {
                "input_shape": [1, int4_k],
                "packed_weight_shape": [int4_n, int4_k // 2],
                "scales_and_zeros_shape": [int4_k // int4_group, int4_n, 2],
                "output_shape": [1, int4_n],
                "group_size": int4_group,
                "row_stride": int4_row_stride,
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(int4_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_int4_bundle.exists():
            previous = json.loads(
                (unity_int4_bundle / "manifest.json").read_text(encoding="utf-8")
            )
            if previous.get("artifact_stage") != "unity-udonaten-int4-mm":
                raise RuntimeError("refusing to replace a non-UdonATen INT4 Unity bundle")
            shutil.rmtree(unity_int4_bundle)
        publish_bundle(staged, unity_int4_bundle)

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
    udonaten_object = output / ("udonaten" + object_suffix)
    device_object = output / ("convolution-device" + object_suffix)
    activation_object = output / ("activation-device" + object_suffix)
    pooling_object = output / ("pooling-device" + object_suffix)
    softmax_object = output / ("softmax-device" + object_suffix)
    batchnorm_object = output / ("batchnorm-device" + object_suffix)
    op_tensor_object = output / ("op-tensor-device" + object_suffix)
    rms_norm_object = output / ("rms-norm-device" + object_suffix)
    gather_object = output / ("gather-device" + object_suffix)
    int4_object = output / ("weight-int4pack-mm-device" + object_suffix)
    run([clang, *runtime_flags, "-c", native_ir, "-o", host_object])
    run(
        [
            clang,
            *runtime_flags,
            "-std=c11",
            "-I",
            source / "include",
            "-I",
            root / "libraries/udonaten/include",
            "-c",
            root / "libraries/udonaten/src/udonaten.c",
            "-o",
            udonaten_object,
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
    run(
        [
            clang,
            *runtime_flags,
            "-x",
            "c",
            "-std=c11",
            "-ffp-contract=off",
            "-c",
            pooling_device,
            "-o",
            pooling_object,
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
            softmax_device,
            "-o",
            softmax_object,
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
            batchnorm_device,
            "-o",
            batchnorm_object,
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
            op_tensor_device,
            "-o",
            op_tensor_object,
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
            rms_norm_device,
            "-o",
            rms_norm_object,
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
            gather_device,
            "-o",
            gather_object,
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
            int4_device,
            "-o",
            int4_object,
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
                (pooling_fragment, "frag", "frag"),
                (softmax_fragment, "frag", "frag"),
                (batchnorm_fragment, "frag", "frag"),
                (op_tensor_fragment, "frag", "frag"),
                (rms_norm_fragment, "frag", "frag"),
                (gather_fragment, "frag", "frag"),
                (int4_fragment, "frag", "frag"),
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
        (
            fragment,
            activation_fragment,
            pooling_fragment,
            softmax_fragment,
            batchnorm_fragment,
            op_tensor_fragment,
            rms_norm_fragment,
            gather_fragment,
            int4_fragment,
            vertex,
        ) = compiled

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
            "-DUDONDNN_HOST_OBJECT=" + host_object.as_posix(),
            "-DUDONATEN_OBJECT=" + udonaten_object.as_posix(),
            "-DUDONDNN_DEVICE_OBJECT=" + device_object.as_posix(),
            "-DUDONDNN_ACTIVATION_OBJECT=" + activation_object.as_posix(),
            "-DUDONDNN_POOLING_OBJECT=" + pooling_object.as_posix(),
            "-DUDONDNN_SOFTMAX_OBJECT=" + softmax_object.as_posix(),
            "-DUDONDNN_BATCHNORM_OBJECT=" + batchnorm_object.as_posix(),
            "-DUDONDNN_OP_TENSOR_OBJECT=" + op_tensor_object.as_posix(),
            "-DUDONDNN_RMS_NORM_OBJECT=" + rms_norm_object.as_posix(),
            "-DUDONDNN_GATHER_OBJECT=" + gather_object.as_posix(),
            "-DUDONATEN_INT4_OBJECT=" + int4_object.as_posix(),
            "-DUDONDNN_BINDINGS_DIR=" + output.as_posix(),
        ]
    )
    run([sys.executable, root / "scripts/build.py", "--build-dir", runtime_build])
    executable = runtime_build / ("udondnn-device" + suffix)
    print(
        run(
            [
                executable,
                fragment,
                activation_fragment,
                pooling_fragment,
                softmax_fragment,
                batchnorm_fragment,
                op_tensor_fragment,
                rms_norm_fragment,
                gather_fragment,
                int4_fragment,
                vertex,
            ]
        ),
        end="",
    )


if __name__ == "__main__":
    main()
