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
    sgemv_device = source / "device/sgemv.cl"
    saxpy_device = source / "device/saxpy.cl"
    sdot_device = source / "device/sdot.cl"
    device_text = (
        sgemm_device.read_text(encoding="utf-8")
        + "\n"
        + sgemv_device.read_text(encoding="utf-8")
        + "\n"
        + saxpy_device.read_text(encoding="utf-8")
        + "\n"
        + sdot_device.read_text(encoding="utf-8")
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
    sgemv_ir = compile_device_ir(sgemv_device, "sgemv")
    saxpy_ir = compile_device_ir(saxpy_device, "saxpy")
    sdot_ir = compile_device_ir(sdot_device, "sdot")

    def emit_kernel(name, kernel_id, words, scalar_count, device_ir, shape=None):
        global_size = [words, 1, 1] if shape is None else [*shape, 1]
        config = output / f"{name}.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": "gather-f32",
                    "words": words,
                    "kernel": name,
                    "kernel_id": kernel_id,
                    "global_size": global_size,
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
            or kernel["global_size"] != global_size
            or [argument["required_bytes"] for argument in kernel["args"][:2]]
            != [words * 4, words * 4]
            or len(kernel["args"]) != scalar_count + 2
        ):
            raise RuntimeError(f"UdonBLAS {name} manifest does not match its ABI")
        return bundle

    bundle = emit_kernel("udonblas_sgemm", 100, 1024, 10, sgemm_ir, (32, 32))
    sgemv_bundle = emit_kernel("udonblas_sgemv", 101, 1024, 8, sgemv_ir)
    saxpy_bundle = emit_kernel("udonblas_saxpy", 102, 1024, 4, saxpy_ir)
    sdot_product_bundle = emit_kernel(
        "udonblas_sdot_product", 103, 1024, 3, sdot_ir
    )
    sdot_reduce_bundle = emit_kernel("udonblas_sdot_reduce", 104, 1024, 2, sdot_ir)
    fragment = bundle / "kernels/kernel_100.hlsl"
    sgemv_fragment = sgemv_bundle / "kernels/kernel_101.hlsl"
    saxpy_fragment = saxpy_bundle / "kernels/kernel_102.hlsl"
    sdot_product_fragment = sdot_product_bundle / "kernels/kernel_103.hlsl"
    sdot_reduce_fragment = sdot_reduce_bundle / "kernels/kernel_104.hlsl"
    shaderlab = (bundle / "kernels/kernel_100.shader").read_text(encoding="utf-8")
    sgemv_shaderlab = (sgemv_bundle / "kernels/kernel_101.shader").read_text(
        encoding="utf-8"
    )
    saxpy_shaderlab = (saxpy_bundle / "kernels/kernel_102.shader").read_text(
        encoding="utf-8"
    )
    sdot_product_shaderlab = (
        sdot_product_bundle / "kernels/kernel_103.shader"
    ).read_text(encoding="utf-8")
    sdot_reduce_shaderlab = (
        sdot_reduce_bundle / "kernels/kernel_104.shader"
    ).read_text(encoding="utf-8")
    if (
        "_VRC_Arg9 (\"Arg9\", Integer)" not in shaderlab
        or "int _VRC_Arg9;" not in fragment.read_text(encoding="utf-8")
        or "asuint(_VRC_Arg9)" not in fragment.read_text(encoding="utf-8")
        or "_VRC_Arg7 (\"Arg7\", Integer)" not in sgemv_shaderlab
        or "_VRC_Arg4 (\"Arg4\", Integer)" not in saxpy_shaderlab
        or "_VRC_Arg4 (\"Arg4\", Integer)" not in sdot_product_shaderlab
        or "_VRC_Arg3 (\"Arg3\", Integer)" not in sdot_reduce_shaderlab
    ):
        raise RuntimeError("UdonBLAS integer uniforms are not Unity SetInteger-compatible")
    (output / "Bindings.h").write_text(
        "inline const char *deviceSource() { return "
        + json.dumps(device_text)
        + "; }\n"
        + "inline std::vector<native_opencl::GatherBinding> manifestBindings("
        + "const std::string &sgemm, const std::string &sgemv, "
        + "const std::string &saxpy, const std::string &sdotProduct, "
        + "const std::string &sdotReduce) { return {"
        + '{100, 1024, 10, "udonblas_sgemm", sgemm, 32, 32}, '
        + '{101, 1024, 8, "udonblas_sgemv", sgemv}, '
        + '{102, 1024, 4, "udonblas_saxpy", saxpy}, '
        + '{103, 1024, 3, "udonblas_sdot_product", sdotProduct}, '
        + '{104, 1024, 2, "udonblas_sdot_reduce", sdotReduce}}; }\n',
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

    sgemv_trans, sgemv_m, sgemv_n = 2, 3, 5
    sgemv_lda, sgemv_incx, sgemv_incy = 4, -2, -3
    sgemv_alpha, sgemv_beta = 0.5, 1.0
    sgemv_a = [((index * 17 + 3) % 23 - 11) * 0.0625 for index in range(20)]
    sgemv_x = [((index * 11 + 5) % 19 - 9) * 0.125 for index in range(5)]
    sgemv_y = [((index * 7 + 1) % 17 - 8) * 0.0625 for index in range(13)]
    sgemv_x_start = (1 - sgemv_m) * sgemv_incx
    sgemv_y_start = (1 - sgemv_n) * sgemv_incy
    sgemv_expected = []
    for row in range(sgemv_n):
        product = sum(
            sgemv_a[inner + row * sgemv_lda]
            * sgemv_x[sgemv_x_start + inner * sgemv_incx]
            for inner in range(sgemv_m)
        )
        sgemv_expected.append(
            sgemv_alpha * product
            + sgemv_beta * sgemv_y[sgemv_y_start + row * sgemv_incy]
        )
    sgemv_packed = [0.0] * 1024
    sgemv_packed[0 : len(sgemv_a)] = sgemv_a
    sgemv_packed[256 : 256 + len(sgemv_x)] = sgemv_x
    sgemv_packed[512 : 512 + len(sgemv_y)] = sgemv_y

    unity_sgemv_bundle = output / "unity-sgemv-bundle"
    with tempfile.TemporaryDirectory(prefix="unity-udonblas-sgemv-", dir=output) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/sgemv-host.uasm", staged / "host.uasm")
        shutil.copy2(source / "unity/sgemv-display.shader", staged / "display.shader")
        shutil.copy2(
            sgemv_bundle / "kernels/kernel_101.shader",
            staged / "kernels/kernel_101.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<1024f", *sgemv_packed)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<5f", *sgemv_expected)
        )
        sgemv_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udonblas-sgemv",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udonblas_sgemv",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 2,
                    "c_symbol": "udonblasSgemv",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 101,
                        "name": "udonblas_sgemv",
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
            json.dumps(sgemv_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_sgemv_bundle.exists():
            previous = json.loads(
                (unity_sgemv_bundle / "manifest.json").read_text(encoding="utf-8")
            )
            if previous.get("artifact_stage") != "unity-udonblas-sgemv":
                raise RuntimeError("refusing to replace a non-UdonBLAS SGEMV Unity bundle")
            shutil.rmtree(unity_sgemv_bundle)
        publish_bundle(staged, unity_sgemv_bundle)

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

    sdot_n, sdot_incx, sdot_incy = 23, -2, 3
    sdot_x_span = 1 + (sdot_n - 1) * abs(sdot_incx)
    sdot_y_span = 1 + (sdot_n - 1) * abs(sdot_incy)
    sdot_x_start = (1 - sdot_n) * sdot_incx
    sdot_y_start = 0
    sdot_x = [
        ((index * 13 + 7) % 31 - 15) * 0.0625 for index in range(sdot_x_span)
    ]
    sdot_y = [
        ((index * 17 + 3) % 29 - 14) * 0.0625 for index in range(sdot_y_span)
    ]
    sdot_expected = sum(
        sdot_x[sdot_x_start + element * sdot_incx]
        * sdot_y[sdot_y_start + element * sdot_incy]
        for element in range(sdot_n)
    )
    sdot_packed = [0.0] * 1024
    sdot_packed[0:sdot_x_span] = sdot_x
    sdot_packed[256 : 256 + sdot_y_span] = sdot_y

    unity_sdot_bundle = output / "unity-sdot-bundle"
    with tempfile.TemporaryDirectory(
        prefix="unity-udonblas-sdot-", dir=output
    ) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(source / "unity/sdot-host.uasm", staged / "host.uasm")
        shutil.copy2(source / "unity/sdot-display.shader", staged / "display.shader")
        shutil.copy2(
            sdot_product_bundle / "kernels/kernel_103.shader",
            staged / "kernels/kernel_103.shader",
        )
        shutil.copy2(
            sdot_reduce_bundle / "kernels/kernel_104.shader",
            staged / "kernels/kernel_104.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<1024f", *sdot_packed)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack("<f", sdot_expected)
        )
        sdot_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udonblas-sdot",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udonblas_sdot",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 4,
                    "c_symbol": "udonblasSdot",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 103,
                        "name": "udonblas_sdot_product",
                        "shader": "kernels/kernel_103.shader",
                    },
                    {
                        "id": 104,
                        "name": "udonblas_sdot_reduce",
                        "shader": "kernels/kernel_104.shader",
                    },
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
            json.dumps(sdot_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_sdot_bundle.exists():
            previous = json.loads(
                (unity_sdot_bundle / "manifest.json").read_text(encoding="utf-8")
            )
            if previous.get("artifact_stage") != "unity-udonblas-sdot":
                raise RuntimeError("refusing to replace a non-UdonBLAS SDOT Unity bundle")
            shutil.rmtree(unity_sdot_bundle)
        publish_bundle(staged, unity_sdot_bundle)

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
                (sgemv_fragment, "frag", "frag"),
                (saxpy_fragment, "frag", "frag"),
                (sdot_product_fragment, "frag", "frag"),
                (sdot_reduce_fragment, "frag", "frag"),
                (vertex, "vert", "vert"),
            ]
        ):
            binary = output / f"shader_{index}.spv"
            run([glslang, "-D", "-V", "-S", stage, "-e", entry, "--auto-map-locations", asset, "-o", binary])
            compiled.append(binary)
        (
            fragment,
            sgemv_fragment,
            saxpy_fragment,
            sdot_product_fragment,
            sdot_reduce_fragment,
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
            "-DUDONBLAS_HOST_OBJECT=" + host_object.as_posix(),
            "-DUDONBLAS_DEVICE_OBJECT=" + device_object.as_posix(),
            "-DUDONBLAS_BINDINGS_DIR=" + output.as_posix(),
        ]
    )
    run([sys.executable, root / "scripts/build.py", "--build-dir", runtime_build])
    executable = runtime_build / ("udonblas-device" + suffix)
    print(
        run(
            [
                executable,
                fragment,
                sgemv_fragment,
                saxpy_fragment,
                sdot_product_fragment,
                sdot_reduce_fragment,
                vertex,
            ]
        ),
        end="",
    )


if __name__ == "__main__":
    main()
