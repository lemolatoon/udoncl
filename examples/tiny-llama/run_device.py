"""Compile and execute Tiny LLaMA as parallel Device fragment passes."""

import argparse
import importlib.util
import json
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
    parser.add_argument("--output", type=Path, default=root / "build/tiny-llama-device")
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
        line.split("=", 1)[1] for line in cache.splitlines() if line.startswith("LLVM_DIR:")
    )
    sdk = Path(llvm_dir).parents[2]
    clang = sdk / "bin" / ("clang" + suffix)
    compiler = build / "tools/vrccl" / ("vrccl" + suffix)
    translate = build / "tools/vrc-translate" / ("vrc-translate" + suffix)
    opt = build / "tools/vrc-opt" / ("vrc-opt" + suffix)
    mlir_translate = sdk / "bin" / ("mlir-translate" + suffix)
    runtime_flags = ["-fms-runtime-lib=dll"] if sys.platform == "win32" else []

    onnx_model = output / "tiny-llama.onnx"
    onnx_plan_path = output / "onnx-plan.json"
    run(
        [
            sys.executable,
            source / "generate_onnx.py",
            "-o",
            onnx_model,
            "--plan",
            onnx_plan_path,
        ]
    )
    onnx_plan = json.loads(onnx_plan_path.read_text(encoding="utf-8"))
    expected_kernels = [
        "tiny_llama_embed",
        "tiny_llama_qkv",
        "tiny_llama_attention",
        "tiny_llama_project",
        "tiny_llama_ffn_up",
        "tiny_llama_ffn_down",
        "tiny_llama_logits",
    ]
    if (
        onnx_plan.get("profile") != "udon-tiny-llama-f32"
        or [device_pass["kernel"] for device_pass in onnx_plan["passes"]]
        != expected_kernels
    ):
        raise RuntimeError("Tiny Llama ONNX graph was not assigned to Device passes")

    device_text = (source / "device.cl").read_text(encoding="utf-8")
    (output / "device_source.h").write_text(
        "#define UDONCL_DEVICE_SOURCE " + json.dumps(device_text) + "\n",
        encoding="utf-8",
    )
    defines = {
        "tiny_llama_embed": "TINY_LLAMA_EMBED",
        "tiny_llama_qkv": "TINY_LLAMA_QKV",
        "tiny_llama_attention": "TINY_LLAMA_ATTENTION",
        "tiny_llama_project": "TINY_LLAMA_PROJECT",
        "tiny_llama_ffn_up": "TINY_LLAMA_FFN_UP",
        "tiny_llama_ffn_down": "TINY_LLAMA_FFN_DOWN",
        "tiny_llama_logits": "TINY_LLAMA_LOGITS",
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
                    "words": 2048,
                    "kernel": name,
                    "kernel_id": identifier,
                    "global_size": [2048, 1, 1],
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
            or kernel["global_size"] != [2048, 1, 1]
            or [argument["required_bytes"] for argument in kernel["args"][:2]]
            != [8192, 8192]
        ):
            raise RuntimeError("Tiny LLaMA gather manifest does not match its packed ABI")
        bundles[identifier] = bundle
        fragments.append(bundle / f"kernels/kernel_{identifier}.hlsl")

    binding_rows = ", ".join(
        f'{{{identifier}, 2048, 0, "{name}", paths[{index}]}}'
        for index, (name, identifier, _) in enumerate(kernels)
    )
    (output / "Bindings.h").write_text(
        "inline const char *deviceSource() { return "
        + json.dumps(device_text)
        + "; }\n"
        + "inline std::vector<native_opencl::GatherBinding> manifestBindings(char **paths) { "
        + f"return {{{binding_rows}}}; }}\n",
        encoding="utf-8",
    )

    reference_spec = importlib.util.spec_from_file_location(
        "tiny_llama_reference", source / "run.py"
    )
    reference = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference)
    weights = reference.make_weights(0x4C4C414D)
    packed = [0.0] * 2048
    for position, token in enumerate([1, 5, 9, 3]):
        packed[position * 16 + token] = 1.0
    weight_layout = [
        ("embedding", 512),
        ("attention_norm", 640),
        ("wq", 648),
        ("wk", 712),
        ("wv", 776),
        ("wo", 840),
        ("ffn_norm", 904),
        ("w1", 912),
        ("w2", 1040),
        ("w3", 1168),
        ("final_norm", 1296),
        ("output", 1304),
    ]
    for name, offset in weight_layout:
        values = weights[name]
        packed[offset : offset + len(values)] = values
    expected_logits = reference.forward(weights, [1, 5, 9, 3])

    unity_bundle = output / "unity-bundle"
    with tempfile.TemporaryDirectory(prefix="unity-", dir=output) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "host.uasm", staged / "host.uasm")
        shutil.copy2(source / "display.shader", staged / "display.shader")
        shutil.copy2(source / "prompt-pack.shader", staged / "prompt-pack.shader")
        shutil.copy2(source / "argmax-readback.shader", staged / "argmax-readback.shader")
        shutil.copy2(onnx_model, staged / "model.onnx.bytes")
        shutil.copy2(onnx_plan_path, staged / "onnx-plan.bytes")
        for name, identifier, _ in kernels:
            shutil.copy2(
                bundles[identifier] / f"kernels/kernel_{identifier}.shader",
                staged / f"kernels/kernel_{identifier}.shader",
            )
        (staged / "packed_state.bytes").write_bytes(struct.pack("<2048f", *packed))
        (staged / "reference_logits.bytes").write_bytes(
            struct.pack("<16f", *expected_logits)
        )
        unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-tiny-llama-prototype",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "tiny_llama",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "tiny_llama_gpu_predict",
                    "udon_event": "_interact",
                    "status_slot": "",
                    "async": True,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": identifier,
                        "name": name,
                        "shader": f"kernels/kernel_{identifier}.shader",
                    }
                    for name, identifier, _ in kernels
                ]
            },
            "resources": [
                {
                    "name": "prompt_pack",
                    "kind": "shader",
                    "path": "prompt-pack.shader",
                },
                {
                    "name": "argmax_readback",
                    "kind": "shader",
                    "path": "argmax-readback.shader",
                },
                {"name": "packed_state", "kind": "bytes", "path": "packed_state.bytes"},
                {
                    "name": "reference_logits",
                    "kind": "bytes",
                    "path": "reference_logits.bytes",
                },
                {"name": "onnx_model", "kind": "bytes", "path": "model.onnx.bytes"},
                {
                    "name": "onnx_execution_plan",
                    "kind": "bytes",
                    "path": "onnx-plan.bytes",
                },
            ],
            "onnx": {
                "profile": onnx_plan["profile"],
                "ir_version": onnx_plan["ir_version"],
                "opset": onnx_plan["opset"],
                "custom_opset": onnx_plan["custom_opset"],
                "device_passes": len(onnx_plan["passes"]),
                "provider": onnx_plan["nodes"][0]["provider"],
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_bundle.exists():
            previous = json.loads((unity_bundle / "manifest.json").read_text(encoding="utf-8"))
            if previous.get("artifact_stage") != "unity-tiny-llama-prototype":
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
            source / "device_host.c",
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
    executable = runtime_build / ("tiny-llama-device" + suffix)
    stdout = run([executable, *fragments, vertex])
    print(stdout, end="")


if __name__ == "__main__":
    main()
