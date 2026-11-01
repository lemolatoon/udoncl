"""Compile and execute UdonBLAS kernels through the UdonCL fragment backend."""

import argparse
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
    parser.add_argument("--output", type=Path, default=root / "build/udonblas-device")
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

    sgemm_device = source / "device/sgemm.cl"
    q4_device = source / "device/q4_gemv.cl"
    saxpy_device = source / "device/saxpy.cl"
    device_text = (
        sgemm_device.read_text(encoding="utf-8")
        + "\n"
        + q4_device.read_text(encoding="utf-8")
        + "\n"
        + saxpy_device.read_text(encoding="utf-8")
    )
    (output / "udonblas_device_source.h").write_text(
        "#define UDONBLAS_DEVICE_SOURCE " + json.dumps(device_text) + "\n",
        encoding="utf-8",
    )
    def compile_device_ir(device, name):
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
        return device_ir

    sgemm_ir = compile_device_ir(sgemm_device, "sgemm")
    q4_ir = compile_device_ir(q4_device, "q4_gemv")
    saxpy_ir = compile_device_ir(saxpy_device, "saxpy")

    def emit_kernel(name, kernel_id, words, scalar_count, device_ir):
        config = output / f"{name}.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": "gather-f32",
                    "words": words,
                    "kernel": name,
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
                    raise RuntimeError("refusing to replace a non-UdonBLAS bundle")
                shutil.rmtree(bundle)
            publish_bundle(staged, bundle)
        manifest = json.loads(
            (bundle / "manifest.json").read_text(encoding="utf-8")
        )
        kernel = manifest["device"]["kernels"][0]
        if (
            kernel["name"] != name
            or kernel["global_size"] != [words, 1, 1]
            or [argument["required_bytes"] for argument in kernel["args"][:2]]
            != [words * 4, words * 4]
            or len(kernel["args"]) != scalar_count + 2
        ):
            raise RuntimeError(f"UdonBLAS {name} manifest does not match its ABI")
        return bundle

    bundle = emit_kernel("udonblas_sgemm", 100, 1024, 10, sgemm_ir)
    q4_bundle = emit_kernel("udonblas_sgemv_q4", 101, 8192, 7, q4_ir)
    saxpy_bundle = emit_kernel("udonblas_saxpy", 102, 1024, 4, saxpy_ir)
    fragment = bundle / "kernels/kernel_100.hlsl"
    q4_fragment = q4_bundle / "kernels/kernel_101.hlsl"
    saxpy_fragment = saxpy_bundle / "kernels/kernel_102.hlsl"
    shaderlab = (bundle / "kernels/kernel_100.shader").read_text(encoding="utf-8")
    q4_shaderlab = (q4_bundle / "kernels/kernel_101.shader").read_text(
        encoding="utf-8"
    )
    saxpy_shaderlab = (saxpy_bundle / "kernels/kernel_102.shader").read_text(
        encoding="utf-8"
    )
    if (
        "_VRC_Arg9 (\"Arg9\", Int)" not in shaderlab
        or "_VRC_Arg6 (\"Arg6\", Int)" not in q4_shaderlab
        or "_VRC_Arg4 (\"Arg4\", Int)" not in saxpy_shaderlab
        or "Integer)" in shaderlab
        or "Integer)" in q4_shaderlab
        or "Integer)" in saxpy_shaderlab
    ):
        raise RuntimeError("UdonBLAS integer uniforms are not Unity SetInt-compatible")
    (output / "Bindings.h").write_text(
        "inline const char *deviceSource() { return "
        + json.dumps(device_text)
        + "; }\n"
        + "inline std::vector<native_opencl::GatherBinding> manifestBindings("
        + "const std::string &sgemm, const std::string &q4, "
        + "const std::string &saxpy) { return {"
        + '{100, 1024, 10, "udonblas_sgemm", sgemm}, '
        + '{101, 8192, 7, "udonblas_sgemv_q4", q4}, '
        + '{102, 1024, 4, "udonblas_saxpy", saxpy}}; }\n',
        encoding="utf-8",
    )

    fixture_m, fixture_n, fixture_k = 4, 5, 3
    fixture_lda, fixture_ldb, fixture_ldc = 4, 3, 4
    fixture_alpha, fixture_beta = 0.75, -0.25
    fixture_a = [((index * 13 + 5) % 19 - 9) * 0.125 for index in range(12)]
    fixture_b = [((index * 7 + 3) % 17 - 8) * 0.125 for index in range(15)]
    fixture_c = [((index * 5 + 1) % 11 - 5) * 0.0625 for index in range(20)]
    fixture_expected = list(fixture_c)
    for column in range(fixture_n):
        for row in range(fixture_m):
            product = sum(
                fixture_a[row + inner * fixture_lda]
                * fixture_b[inner + column * fixture_ldb]
                for inner in range(fixture_k)
            )
            index = row + column * fixture_ldc
            fixture_expected[index] = (
                fixture_alpha * product + fixture_beta * fixture_expected[index]
            )
    fixture_packed = [0.0] * 1024
    fixture_packed[0 : len(fixture_a)] = fixture_a
    fixture_packed[256 : 256 + len(fixture_b)] = fixture_b
    fixture_packed[512 : 512 + len(fixture_c)] = fixture_c
    unity_bundle = output / "unity-bundle"
    with tempfile.TemporaryDirectory(prefix="unity-udonblas-", dir=output) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/host.uasm", staged / "host.uasm")
        shutil.copy2(source / "unity/display.shader", staged / "display.shader")
        shutil.copy2(
            bundle / "kernels/kernel_100.shader",
            staged / "kernels/kernel_100.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<1024f", *fixture_packed)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<20f", *fixture_expected)
        )
        unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udonblas-sgemm",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udonblas_sgemm",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "udonblasSgemm",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 100,
                        "name": "udonblas_sgemm",
                        "shader": "kernels/kernel_100.shader",
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
        }
        (staged / "manifest.json").write_text(
            json.dumps(unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_bundle.exists():
            previous = json.loads((unity_bundle / "manifest.json").read_text(encoding="utf-8"))
            if previous.get("artifact_stage") != "unity-udonblas-sgemm":
                raise RuntimeError("refusing to replace a non-UdonBLAS Unity bundle")
            shutil.rmtree(unity_bundle)
        publish_bundle(staged, unity_bundle)

    q4_m, q4_k, q4_lda, q4_incx, q4_incy = 5, 64, 36, 2, 3
    q4_alpha, q4_beta = 0.75, -0.5
    q4_matrix = [123.0] * (q4_m * q4_lda)
    q4_x = [0.0] * (q4_k * q4_incx)
    q4_y = [0.0] * (q4_m * q4_incy)

    def q4_value(row, inner):
        return ((row * 19 + inner * 7 + 5) % 16) - 8

    for row in range(q4_m):
        for block in range(q4_k // 32):
            base = row * q4_lda + block * 17
            for pair in range(16):
                low = q4_value(row, block * 32 + pair * 2) & 15
                high = q4_value(row, block * 32 + pair * 2 + 1) & 15
                q4_matrix[base + pair] = float(low | (high << 4))
            q4_matrix[base + 16] = 0.03125 * (block + row + 1)
    for inner in range(q4_k):
        q4_x[inner * q4_incx] = ((inner + 3) * (23 * 7 + 11) % 29 - 14) * 0.0625
    for row in range(q4_m):
        q4_y[row * q4_incy] = ((row + 3) * (29 * 7 + 11) % 29 - 14) * 0.0625
    q4_expected = []
    for row in range(q4_m):
        product = 0.0
        for inner in range(q4_k):
            block = inner // 32
            scale = q4_matrix[row * q4_lda + block * 17 + 16]
            product += q4_value(row, inner) * scale * q4_x[inner * q4_incx]
        q4_expected.append(q4_alpha * product + q4_beta * q4_y[row * q4_incy])
    q4_packed = [0.0] * 8192
    q4_packed[0 : len(q4_matrix)] = q4_matrix
    for inner in range(q4_k):
        q4_packed[6144 + inner * q4_incx] = q4_x[inner * q4_incx]
    for row in range(q4_m):
        q4_packed[6656 + row * q4_incy] = q4_y[row * q4_incy]

    unity_q4_bundle = output / "unity-q4-bundle"
    with tempfile.TemporaryDirectory(prefix="unity-udonblas-q4-", dir=output) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/q4-host.uasm", staged / "host.uasm")
        shutil.copy2(source / "unity/q4-display.shader", staged / "display.shader")
        shutil.copy2(
            q4_bundle / "kernels/kernel_101.shader",
            staged / "kernels/kernel_101.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<8192f", *q4_packed)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<5f", *q4_expected)
        )
        q4_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udonblas-q4-gemv",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udonblas_q4_gemv",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 2,
                    "c_symbol": "udonblasSgemvQ4",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 101,
                        "name": "udonblas_sgemv_q4",
                        "shader": "kernels/kernel_101.shader",
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
        }
        (staged / "manifest.json").write_text(
            json.dumps(q4_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_q4_bundle.exists():
            previous = json.loads(
                (unity_q4_bundle / "manifest.json").read_text(encoding="utf-8")
            )
            if previous.get("artifact_stage") != "unity-udonblas-q4-gemv":
                raise RuntimeError("refusing to replace a non-UdonBLAS Q4 Unity bundle")
            shutil.rmtree(unity_q4_bundle)
        publish_bundle(staged, unity_q4_bundle)

    saxpy_n, saxpy_incx, saxpy_incy = 16, -2, 3
    saxpy_alpha = -0.75
    saxpy_x_span = 1 + (saxpy_n - 1) * abs(saxpy_incx)
    saxpy_y_span = 1 + (saxpy_n - 1) * abs(saxpy_incy)
    saxpy_x_start = (1 - saxpy_n) * saxpy_incx
    saxpy_y_start = 0
    saxpy_x = [((index * 11 + 3) % 23 - 11) * 0.0625 for index in range(saxpy_x_span)]
    saxpy_y = [((index * 7 + 5) % 19 - 9) * 0.125 for index in range(saxpy_y_span)]
    saxpy_expected = []
    for element in range(saxpy_n):
        x_index = saxpy_x_start + element * saxpy_incx
        y_index = saxpy_y_start + element * saxpy_incy
        saxpy_expected.append(saxpy_alpha * saxpy_x[x_index] + saxpy_y[y_index])
    saxpy_packed = [0.0] * 1024
    saxpy_packed[0:saxpy_x_span] = saxpy_x
    saxpy_packed[256 : 256 + saxpy_y_span] = saxpy_y

    unity_saxpy_bundle = output / "unity-saxpy-bundle"
    with tempfile.TemporaryDirectory(prefix="unity-udonblas-saxpy-", dir=output) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/saxpy-host.uasm", staged / "host.uasm")
        shutil.copy2(source / "unity/saxpy-display.shader", staged / "display.shader")
        shutil.copy2(
            saxpy_bundle / "kernels/kernel_102.shader",
            staged / "kernels/kernel_102.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<1024f", *saxpy_packed)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<16f", *saxpy_expected)
        )
        saxpy_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udonblas-saxpy",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udonblas_saxpy",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 3,
                    "c_symbol": "udonblasSaxpy",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 102,
                        "name": "udonblas_saxpy",
                        "shader": "kernels/kernel_102.shader",
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
        }
        (staged / "manifest.json").write_text(
            json.dumps(saxpy_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_saxpy_bundle.exists():
            previous = json.loads(
                (unity_saxpy_bundle / "manifest.json").read_text(encoding="utf-8")
            )
            if previous.get("artifact_stage") != "unity-udonblas-saxpy":
                raise RuntimeError("refusing to replace a non-UdonBLAS SAXPY Unity bundle")
            shutil.rmtree(unity_saxpy_bundle)
        publish_bundle(staged, unity_saxpy_bundle)

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
            source / "src/udonblas.c",
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
    combined_device = output / "device.cl"
    combined_device.write_text(device_text, encoding="utf-8")
    run(
        [
            clang,
            *runtime_flags,
            "-x",
            "c",
            "-std=c11",
            "-ffp-contract=off",
            "-c",
            combined_device,
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
        compiled = []
        for index, (asset, stage, entry) in enumerate(
            [
                (fragment, "frag", "frag"),
                (q4_fragment, "frag", "frag"),
                (saxpy_fragment, "frag", "frag"),
                (vertex, "vert", "vert"),
            ]
        ):
            binary = output / f"shader_{index}.spv"
            run([glslang, "-D", "-V", "-S", stage, "-e", entry, "--auto-map-locations", asset, "-o", binary])
            compiled.append(binary)
        fragment, q4_fragment, saxpy_fragment, vertex = compiled

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
            "-DUDONBLAS_HOST_OBJECT=" + host_object.as_posix(),
            "-DUDONBLAS_DEVICE_OBJECT=" + device_object.as_posix(),
            "-DUDONBLAS_BINDINGS_DIR=" + output.as_posix(),
        ]
    )
    run([sys.executable, root / "scripts/build.py", "--build-dir", runtime_build])
    executable = runtime_build / ("udonblas-device" + suffix)
    print(run([executable, fragment, q4_fragment, saxpy_fragment, vertex]), end="")


if __name__ == "__main__":
    main()
