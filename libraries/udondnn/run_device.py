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
    compiler = build / "tools/vrccl" / ("udonclc" + suffix)
    translate = build / "tools/vrc-translate" / ("udoncl-translate" + suffix)
    opt = build / "tools/vrc-opt" / ("udoncl-opt" + suffix)
    mlir_translate = sdk / "bin" / ("mlir-translate" + suffix)
    runtime_flags = ["-fms-runtime-lib=dll"] if sys.platform == "win32" else []

    convolution_device = source / "device/convolution.cl"
    activation_device = source / "device/activation.cl"
    pooling_device = source / "device/pooling.cl"
    softmax_device = source / "device/softmax.cl"
    batchnorm_device = source / "device/batchnorm.cl"
    op_tensor_device = source / "device/op_tensor.cl"
    rms_norm_small_device = root / "libraries/udonaten/device/rms_norm_small.cl"
    embedding_device = root / "libraries/udonaten/device/embedding.cl"
    tile_device = root / "libraries/udonaten/device/tile.cl"
    aten_rms_device = root / "libraries/udonaten/device/rms_norm.cl"
    arena_pointwise_device = root / "libraries/udonaten/device/arena_pointwise.cl"
    arena_broadcast_device = root / "libraries/udonaten/device/arena_broadcast_pointwise.cl"
    arena_tile_device = root / "libraries/udonaten/device/arena_tile.cl"
    arena_rms_device = root / "libraries/udonaten/device/arena_rms_norm.cl"
    arena_embedding_device = root / "libraries/udonaten/device/arena_embedding.cl"
    arena_sgemm_device = root / "libraries/udonblas/device/arena_sgemm.cl"
    arena_sgemv_device = root / "libraries/udonblas/device/arena_sgemv.cl"
    arena_softmax_device = source / "device/arena_softmax.cl"
    arena_convolution_device = source / "device/arena_convolution.cl"
    arena_pooling_device = source / "device/arena_pooling.cl"
    sgemm_device = root / "libraries/udonblas/device/sgemm.cl"
    sgemv_device = root / "libraries/udonblas/device/sgemv.cl"
    saxpy_device = root / "libraries/udonblas/device/saxpy.cl"
    sdot_device = root / "libraries/udonblas/device/sdot.cl"
    convolution_text = convolution_device.read_text(encoding="utf-8")
    activation_text = activation_device.read_text(encoding="utf-8")
    pooling_text = pooling_device.read_text(encoding="utf-8")
    softmax_text = softmax_device.read_text(encoding="utf-8")
    batchnorm_text = batchnorm_device.read_text(encoding="utf-8")
    op_tensor_text = op_tensor_device.read_text(encoding="utf-8")
    rms_norm_small_text = rms_norm_small_device.read_text(encoding="utf-8")
    embedding_text = embedding_device.read_text(encoding="utf-8")
    tile_text = tile_device.read_text(encoding="utf-8")
    aten_rms_text = aten_rms_device.read_text(encoding="utf-8")
    arena_pointwise_text = arena_pointwise_device.read_text(encoding="utf-8")
    arena_broadcast_text = arena_broadcast_device.read_text(encoding="utf-8")
    arena_tile_text = arena_tile_device.read_text(encoding="utf-8")
    arena_rms_text = arena_rms_device.read_text(encoding="utf-8")
    arena_embedding_text = arena_embedding_device.read_text(encoding="utf-8")
    arena_sgemm_text = arena_sgemm_device.read_text(encoding="utf-8")
    arena_sgemv_text = arena_sgemv_device.read_text(encoding="utf-8")
    arena_softmax_text = arena_softmax_device.read_text(encoding="utf-8")
    arena_convolution_text = arena_convolution_device.read_text(encoding="utf-8")
    arena_pooling_text = arena_pooling_device.read_text(encoding="utf-8")
    sgemm_text = sgemm_device.read_text(encoding="utf-8")
    sgemv_text = sgemv_device.read_text(encoding="utf-8")
    saxpy_text = saxpy_device.read_text(encoding="utf-8")
    sdot_text = sdot_device.read_text(encoding="utf-8")
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
        + rms_norm_small_text
        + "\n"
        + embedding_text
        + "\n"
        + tile_text
        + "\n"
        + aten_rms_text
        + "\n"
        + arena_pointwise_text
        + "\n"
        + arena_broadcast_text
        + "\n"
        + arena_tile_text
        + "\n"
        + arena_rms_text
        + "\n"
        + arena_embedding_text
        + "\n"
        + arena_sgemm_text
        + "\n"
        + arena_sgemv_text
        + "\n"
        + arena_softmax_text
        + "\n"
        + arena_convolution_text
        + "\n"
        + arena_pooling_text
        + "\n"
        + sgemm_text
        + "\n"
        + sgemv_text
        + "\n"
        + saxpy_text
        + "\n"
        + sdot_text
    )
    (output / "udondnn_device_source.h").write_text(
        "#define UDONDNN_DEVICE_SOURCE " + json.dumps(device_text) + "\n",
        encoding="utf-8",
    )
    (output / "udonblas_device_source.h").write_text(
        "#define UDONBLAS_DEVICE_SOURCE " + json.dumps(device_text) + "\n",
        encoding="utf-8",
    )

    def emit_device(
        name, device, kernel_name, kernel_id, words, shape=None, config_extra=None
    ):
        global_size = [words, 1, 1] if shape is None else [*shape, 1]
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
        config_data = {
            "schema_version": 1,
            "profile": "gather-f32",
            "words": words,
            "kernel": kernel_name,
            "kernel_id": kernel_id,
            "global_size": global_size,
            "local_size": [1, 1, 1],
            "global_offset": [0, 0, 0],
        }
        if config_extra:
            config_data.update(config_extra)
        config.write_text(
            json.dumps(config_data),
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
    rms_norm_small_bundle = emit_device(
        "aten-rms-norm-small",
        rms_norm_small_device,
        "udonaten_rms_norm_small",
        206,
        2048,
    )
    embedding_bundle = emit_device(
        "embedding", embedding_device, "udonaten_embedding", 207, 2048
    )
    tile_bundle = emit_device(
        "tile",
        tile_device,
        "udonaten_tile",
        209,
        8192,
    )
    aten_rms_reduce_bundle = emit_device(
        "aten-rms-norm-reduce",
        aten_rms_device,
        "udonaten_rms_norm_reduce",
        210,
        8192,
    )
    aten_rms_finalize_bundle = emit_device(
        "aten-rms-norm-finalize",
        aten_rms_device,
        "udonaten_rms_norm_finalize",
        211,
        8192,
    )
    sgemm_bundle = emit_device(
        "sgemm", sgemm_device, "udonblas_sgemm", 212, 1024, (32, 32)
    )
    sgemv_bundle = emit_device(
        "sgemv", sgemv_device, "udonblas_sgemv", 213, 1024
    )
    saxpy_bundle = emit_device(
        "saxpy", saxpy_device, "udonblas_saxpy", 214, 1024
    )
    sdot_product_bundle = emit_device(
        "sdot-product", sdot_device, "udonblas_sdot_product", 219, 1024
    )
    sdot_reduce_bundle = emit_device(
        "sdot-reduce", sdot_device, "udonblas_sdot_reduce", 220, 1024
    )
    arena_pointwise_bundle = emit_device(
        "aten-arena-pointwise",
        arena_pointwise_device,
        "udonaten_arena_pointwise",
        215,
        16384,
        (8192, 2),
    )
    arena_tile_bundle = emit_device(
        "aten-arena-tile",
        arena_tile_device,
        "udonaten_arena_tile",
        216,
        16384,
        (8192, 2),
    )
    arena_rms_reduce_bundle = emit_device(
        "aten-arena-rms-reduce",
        arena_rms_device,
        "udonaten_arena_rms_norm_reduce",
        217,
        16384,
        (8192, 2),
        config_extra={
            "argument_bounds": [
                {"index": 2, "min": 1, "max": 16384},
                {"index": 3, "min": 1, "max": 4096},
            ]
        },
    )
    arena_rms_finalize_bundle = emit_device(
        "aten-arena-rms-finalize",
        arena_rms_device,
        "udonaten_arena_rms_norm_finalize",
        218,
        16384,
        (8192, 2),
        config_extra={
            "input_words": [16384, 16384, 4096],
            "argument_bounds": [
                {"index": 4, "min": 1, "max": 16384},
                {"index": 5, "min": 1, "max": 4096},
                {"index": 7, "min": 1, "max": 1},
            ],
        },
    )
    arena_sgemm_bundle = emit_device(
        "udonblas-arena-sgemm",
        arena_sgemm_device,
        "udonblas_arena_sgemm",
        221,
        16384,
        (8192, 2),
    )
    arena_sgemv_product_bundle = emit_device(
        "udonblas-arena-sgemv-product",
        arena_sgemv_device,
        "udonblas_arena_sgemv_product",
        227,
        16384,
        (8192, 2),
    )
    arena_sgemv_reduce_bundle = emit_device(
        "udonblas-arena-sgemv-reduce",
        arena_sgemv_device,
        "udonblas_arena_sgemv_reduce",
        228,
        16384,
        (8192, 2),
    )
    arena_sgemv_finalize_bundle = emit_device(
        "udonblas-arena-sgemv-finalize",
        arena_sgemv_device,
        "udonblas_arena_sgemv_finalize",
        229,
        16384,
        (8192, 2),
    )
    arena_softmax_bundle = emit_device(
        "udondnn-arena-softmax",
        arena_softmax_device,
        "udondnn_arena_softmax_forward",
        222,
        16384,
        (8192, 2),
        config_extra={
            "argument_bounds": [
                {"index": 2, "min": 1, "max": 16384},
                {"index": 3, "min": 1, "max": 512},
                {"index": 4, "min": 1, "max": 16384},
                {"index": 5, "min": 1, "max": 16384},
                {"index": 6, "min": 0, "max": 2},
                {"index": 7, "min": 0, "max": 1},
            ]
        },
    )
    arena_broadcast_bundle = emit_device(
        "udonaten-arena-broadcast-pointwise",
        arena_broadcast_device,
        "udonaten_arena_broadcast_pointwise",
        223,
        16384,
        (8192, 2),
    )
    arena_convolution_bundle = emit_device(
        "udondnn-arena-convolution",
        arena_convolution_device,
        "udondnn_arena_convolution_forward",
        224,
        16384,
        (8192, 2),
    )
    arena_pooling_bundle = emit_device(
        "udondnn-arena-pooling",
        arena_pooling_device,
        "udondnn_arena_pooling_forward",
        225,
        16384,
        (8192, 2),
    )
    arena_embedding_bundle = emit_device(
        "udonaten-arena-embedding",
        arena_embedding_device,
        "udonaten_arena_embedding",
        226,
        16384,
        (8192, 2),
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
    if (
        '_VRC_Arg6 ("Arg6", Integer)' not in shaderlab
        or "int _VRC_Arg6;" not in fragment_text
        or "asuint(_VRC_Arg6)" not in fragment_text
    ):
        raise RuntimeError("UdonDNN integer uniforms are not Unity SetInteger-compatible")
    if (
        "mode ==" in convolution_text
        or "? ky" in convolution_text
        or "? kx" in convolution_text
    ):
        raise RuntimeError("UdonDNN convolution mode must not lower through select")
    if (
        "(asuint(_VRC_Arg4) >>" not in fragment_text
        or "(asuint(_VRC_Arg5) &" not in fragment_text
    ):
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
    if '_VRC_Arg4 ("Arg4", Integer)' not in activation_shaderlab:
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
    if '_VRC_Arg5 ("Arg5", Integer)' not in pooling_shaderlab:
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
    if '_VRC_Arg4 ("Arg4", Integer)' not in softmax_shaderlab:
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
    if '_VRC_Arg3 ("Arg3", Integer)' not in batchnorm_shaderlab:
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
    if '_VRC_Arg6 ("Arg6", Integer)' not in op_tensor_shaderlab:
        raise RuntimeError("UdonDNN OpTensor integer uniforms are not Unity-compatible")
    rms_norm_small_manifest = json.loads(
        (rms_norm_small_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    rms_norm_small_kernel = rms_norm_small_manifest["device"]["kernels"][0]
    if (
        rms_norm_small_kernel["name"] != "udonaten_rms_norm_small"
        or rms_norm_small_kernel["global_size"] != [2048, 1, 1]
        or [
            argument["required_bytes"]
            for argument in rms_norm_small_kernel["args"][:2]
        ]
        != [8192, 8192]
        or len(rms_norm_small_kernel["args"]) != 7
    ):
        raise RuntimeError("UdonATen RMSNorm manifest does not match its packed ABI")
    rms_norm_small_fragment = rms_norm_small_bundle / "kernels/kernel_206.hlsl"
    rms_norm_small_shaderlab = (
        rms_norm_small_bundle / "kernels/kernel_206.shader"
    ).read_text(encoding="utf-8")
    if '_VRC_Arg3 ("Arg3", Integer)' not in rms_norm_small_shaderlab:
        raise RuntimeError(
            "UdonATen RMSNorm integer uniforms are not Unity-compatible"
        )
    aten_rms_reduce_manifest = json.loads(
        (aten_rms_reduce_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    aten_rms_reduce_kernel = aten_rms_reduce_manifest["device"]["kernels"][0]
    if (
        aten_rms_reduce_kernel["name"] != "udonaten_rms_norm_reduce"
        or aten_rms_reduce_kernel["global_size"] != [8192, 1, 1]
        or len(aten_rms_reduce_kernel["args"]) != 4
    ):
        raise RuntimeError("UdonATen RMSNorm reduction manifest is invalid")
    aten_rms_finalize_manifest = json.loads(
        (aten_rms_finalize_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    aten_rms_finalize_kernel = aten_rms_finalize_manifest["device"]["kernels"][0]
    if (
        aten_rms_finalize_kernel["name"] != "udonaten_rms_norm_finalize"
        or aten_rms_finalize_kernel["global_size"] != [8192, 1, 1]
        or [argument["access"] for argument in aten_rms_finalize_kernel["args"][:3]]
        != ["read", "read", "write"]
        or len(aten_rms_finalize_kernel["args"]) != 7
    ):
        raise RuntimeError("UdonATen RMSNorm finalization manifest is invalid")
    aten_rms_reduce_fragment = (
        aten_rms_reduce_bundle / "kernels/kernel_210.hlsl"
    )
    aten_rms_finalize_fragment = (
        aten_rms_finalize_bundle / "kernels/kernel_211.hlsl"
    )
    if "_VRC_Input1.Load" not in aten_rms_finalize_fragment.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("UdonATen RMSNorm did not bind its reduction texture")
    arena_pointwise_manifest = json.loads(
        (arena_pointwise_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    arena_pointwise_kernel = arena_pointwise_manifest["device"]["kernels"][0]
    if (
        arena_pointwise_kernel["name"] != "udonaten_arena_pointwise"
        or arena_pointwise_kernel["global_size"] != [8192, 2, 1]
        or [argument["access"] for argument in arena_pointwise_kernel["args"][:3]]
        != ["read", "read", "write"]
        or len(arena_pointwise_kernel["args"]) != 7
    ):
        raise RuntimeError("UdonATen arena pointwise manifest is invalid")
    arena_pointwise_fragment = (
        arena_pointwise_bundle / "kernels/kernel_215.hlsl"
    )
    arena_pointwise_hlsl = arena_pointwise_fragment.read_text(encoding="utf-8")
    if "_VRC_Input1.Load" not in arena_pointwise_hlsl or "uint(position.y)" not in arena_pointwise_hlsl:
        raise RuntimeError("UdonATen arena pointwise lost its multi-row texture ABI")
    arena_tile_manifest = json.loads(
        (arena_tile_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    arena_tile_kernel = arena_tile_manifest["device"]["kernels"][0]
    if (
        arena_tile_kernel["name"] != "udonaten_arena_tile"
        or arena_tile_kernel["global_size"] != [8192, 2, 1]
        or [argument["access"] for argument in arena_tile_kernel["args"][:2]]
        != ["read", "write"]
        or len(arena_tile_kernel["args"]) != 11
    ):
        raise RuntimeError("UdonATen arena Tile manifest is invalid")
    arena_tile_fragment = arena_tile_bundle / "kernels/kernel_216.hlsl"
    if "uint(position.y)" not in arena_tile_fragment.read_text(encoding="utf-8"):
        raise RuntimeError("UdonATen arena Tile lost its multi-row texture ABI")
    arena_rms_reduce_manifest = json.loads(
        (arena_rms_reduce_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    arena_rms_finalize_manifest = json.loads(
        (arena_rms_finalize_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    arena_rms_reduce_kernel = arena_rms_reduce_manifest["device"]["kernels"][0]
    arena_rms_finalize_kernel = arena_rms_finalize_manifest["device"]["kernels"][0]
    if (
        arena_rms_reduce_kernel["name"] != "udonaten_arena_rms_norm_reduce"
        or arena_rms_reduce_kernel["global_size"] != [8192, 2, 1]
        or len(arena_rms_reduce_kernel["args"]) != 4
        or arena_rms_finalize_kernel["name"]
        != "udonaten_arena_rms_norm_finalize"
        or arena_rms_finalize_kernel["global_size"] != [8192, 2, 1]
        or [
            argument["access"]
            for argument in arena_rms_finalize_kernel["args"][:4]
        ]
        != ["read", "read", "read", "write"]
        or len(arena_rms_finalize_kernel["args"]) != 8
    ):
        raise RuntimeError("UdonATen arena RMSNorm manifest is invalid")
    arena_rms_reduce_fragment = (
        arena_rms_reduce_bundle / "kernels/kernel_217.hlsl"
    )
    arena_rms_finalize_fragment = (
        arena_rms_finalize_bundle / "kernels/kernel_218.hlsl"
    )
    arena_rms_finalize_hlsl = arena_rms_finalize_fragment.read_text(encoding="utf-8")
    if (
        "_VRC_Input1.Load" not in arena_rms_finalize_hlsl
        or "_VRC_Input2.Load" not in arena_rms_finalize_hlsl
        or "uint(position.y)" not in arena_rms_finalize_hlsl
    ):
        raise RuntimeError("UdonATen arena RMSNorm lost its three-input texture ABI")
    arena_sgemm_manifest = json.loads(
        (arena_sgemm_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    arena_sgemm_kernel = arena_sgemm_manifest["device"]["kernels"][0]
    if (
        arena_sgemm_kernel["name"] != "udonblas_arena_sgemm"
        or arena_sgemm_kernel["global_size"] != [8192, 2, 1]
        or [argument["access"] for argument in arena_sgemm_kernel["args"][:3]]
        != ["read", "read", "write"]
        or len(arena_sgemm_kernel["args"]) != 9
    ):
        raise RuntimeError("UdonBLAS arena SGEMM manifest is invalid")
    arena_sgemm_fragment = arena_sgemm_bundle / "kernels/kernel_221.hlsl"
    arena_softmax_manifest = json.loads(
        (arena_softmax_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    arena_softmax_kernel = arena_softmax_manifest["device"]["kernels"][0]
    if (
        arena_softmax_kernel["name"] != "udondnn_arena_softmax_forward"
        or arena_softmax_kernel["global_size"] != [8192, 2, 1]
        or [argument["access"] for argument in arena_softmax_kernel["args"][:2]]
        != ["read", "write"]
        or len(arena_softmax_kernel["args"]) != 8
    ):
        raise RuntimeError("UdonDNN arena Softmax manifest is invalid")
    arena_softmax_fragment = arena_softmax_bundle / "kernels/kernel_222.hlsl"
    arena_broadcast_manifest = json.loads(
        (arena_broadcast_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    arena_broadcast_kernel = arena_broadcast_manifest["device"]["kernels"][0]
    if (
        arena_broadcast_kernel["name"] != "udonaten_arena_broadcast_pointwise"
        or arena_broadcast_kernel["global_size"] != [8192, 2, 1]
        or [argument["access"] for argument in arena_broadcast_kernel["args"][:3]]
        != ["read", "read", "write"]
        or len(arena_broadcast_kernel["args"]) != 16
    ):
        raise RuntimeError("UdonATen arena broadcast pointwise manifest is invalid")
    arena_broadcast_fragment = arena_broadcast_bundle / "kernels/kernel_223.hlsl"
    arena_convolution_manifest = json.loads(
        (arena_convolution_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    arena_convolution_kernel = arena_convolution_manifest["device"]["kernels"][0]
    if (
        arena_convolution_kernel["name"] != "udondnn_arena_convolution_forward"
        or arena_convolution_kernel["global_size"] != [8192, 2, 1]
        or [argument["access"] for argument in arena_convolution_kernel["args"][:3]]
        != ["read", "read", "write"]
        or len(arena_convolution_kernel["args"]) != 16
    ):
        raise RuntimeError("UdonDNN arena convolution manifest is invalid")
    arena_convolution_fragment = arena_convolution_bundle / "kernels/kernel_224.hlsl"
    arena_pooling_manifest = json.loads(
        (arena_pooling_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    arena_pooling_kernel = arena_pooling_manifest["device"]["kernels"][0]
    if (
        arena_pooling_kernel["name"] != "udondnn_arena_pooling_forward"
        or arena_pooling_kernel["global_size"] != [8192, 2, 1]
        or [argument["access"] for argument in arena_pooling_kernel["args"][:2]]
        != ["read", "write"]
        or len(arena_pooling_kernel["args"]) != 13
    ):
        raise RuntimeError("UdonDNN arena pooling manifest is invalid")
    arena_pooling_fragment = arena_pooling_bundle / "kernels/kernel_225.hlsl"
    arena_sgemv_product_manifest = json.loads(
        (arena_sgemv_product_bundle / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    arena_sgemv_product_kernel = arena_sgemv_product_manifest["device"][
        "kernels"
    ][0]
    if (
        arena_sgemv_product_kernel["name"] != "udonblas_arena_sgemv_product"
        or arena_sgemv_product_kernel["global_size"] != [8192, 2, 1]
        or len(arena_sgemv_product_kernel["args"]) != 9
    ):
        raise RuntimeError("UdonBLAS arena SGEMV product manifest is invalid")
    arena_sgemv_reduce_manifest = json.loads(
        (arena_sgemv_reduce_bundle / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    arena_sgemv_reduce_kernel = arena_sgemv_reduce_manifest["device"][
        "kernels"
    ][0]
    if (
        arena_sgemv_reduce_kernel["name"] != "udonblas_arena_sgemv_reduce"
        or arena_sgemv_reduce_kernel["global_size"] != [8192, 2, 1]
        or len(arena_sgemv_reduce_kernel["args"]) != 4
    ):
        raise RuntimeError("UdonBLAS arena SGEMV reduce manifest is invalid")
    arena_sgemv_finalize_manifest = json.loads(
        (arena_sgemv_finalize_bundle / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    arena_sgemv_finalize_kernel = arena_sgemv_finalize_manifest["device"][
        "kernels"
    ][0]
    if (
        arena_sgemv_finalize_kernel["name"] != "udonblas_arena_sgemv_finalize"
        or arena_sgemv_finalize_kernel["global_size"] != [8192, 2, 1]
        or len(arena_sgemv_finalize_kernel["args"]) != 4
    ):
        raise RuntimeError("UdonBLAS arena SGEMV finalize manifest is invalid")
    arena_embedding_manifest = json.loads(
        (arena_embedding_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    arena_embedding_kernel = arena_embedding_manifest["device"]["kernels"][0]
    if (
        arena_embedding_kernel["name"] != "udonaten_arena_embedding"
        or arena_embedding_kernel["global_size"] != [8192, 2, 1]
        or [argument["access"] for argument in arena_embedding_kernel["args"][:3]]
        != ["read", "read", "write"]
        or len(arena_embedding_kernel["args"]) != 7
    ):
        raise RuntimeError("UdonATen arena embedding manifest is invalid")
    arena_embedding_fragment = (
        arena_embedding_bundle / "kernels/kernel_226.hlsl"
    )
    blas_bundles = [
        (sgemm_bundle, "udonblas_sgemm", 212, 1024, 10),
        (sgemv_bundle, "udonblas_sgemv", 213, 1024, 8),
        (saxpy_bundle, "udonblas_saxpy", 214, 1024, 4),
        (sdot_product_bundle, "udonblas_sdot_product", 219, 1024, 3),
        (sdot_reduce_bundle, "udonblas_sdot_reduce", 220, 1024, 2),
    ]
    blas_fragments = []
    for blas_bundle, name, identifier, words, scalar_count in blas_bundles:
        blas_manifest = json.loads(
            (blas_bundle / "manifest.json").read_text(encoding="utf-8")
        )
        blas_kernel = blas_manifest["device"]["kernels"][0]
        expected_global_size = [32, 32, 1] if name == "udonblas_sgemm" else [words, 1, 1]
        if (
            blas_kernel["name"] != name
            or blas_kernel["global_size"] != expected_global_size
            or len(blas_kernel["args"]) != scalar_count + 2
            or [argument["required_bytes"] for argument in blas_kernel["args"][:2]]
            != [words * 4, words * 4]
        ):
            raise RuntimeError(f"{name} manifest does not match its UdonBLAS ABI")
        blas_fragments.append(
            blas_bundle / f"kernels/kernel_{identifier}.hlsl"
        )
    embedding_manifest = json.loads(
        (embedding_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    embedding_kernel = embedding_manifest["device"]["kernels"][0]
    if (
        embedding_kernel["name"] != "udonaten_embedding"
        or embedding_kernel["global_size"] != [2048, 1, 1]
        or [
            argument["required_bytes"]
            for argument in embedding_kernel["args"][:2]
        ]
        != [8192, 8192]
        or len(embedding_kernel["args"]) != 8
    ):
        raise RuntimeError("UdonATen Embedding manifest does not match its packed ABI")
    embedding_fragment = embedding_bundle / "kernels/kernel_207.hlsl"
    embedding_shaderlab = (
        embedding_bundle / "kernels/kernel_207.shader"
    ).read_text(encoding="utf-8")
    if '_VRC_Arg2 ("Arg2", Integer)' not in embedding_shaderlab:
        raise RuntimeError(
            "UdonATen Embedding integer uniforms are not Unity-compatible"
        )
    tile_manifest = json.loads((tile_bundle / "manifest.json").read_text(encoding="utf-8"))
    tile_kernel = tile_manifest["device"]["kernels"][0]
    if (
        tile_kernel["name"] != "udonaten_tile"
        or tile_kernel["global_size"] != [8192, 1, 1]
        or [argument["required_bytes"] for argument in tile_kernel["args"][:2]]
        != [32768, 32768]
        or len(tile_kernel["args"]) != 11
    ):
        raise RuntimeError("UdonATen Tile manifest does not match its ATen ABI")
    tile_fragment = tile_bundle / "kernels/kernel_209.hlsl"
    tile_shaderlab = (tile_bundle / "kernels/kernel_209.shader").read_text(
        encoding="utf-8"
    )
    if '_VRC_Arg10 ("Arg10", Integer)' not in tile_shaderlab:
        raise RuntimeError("UdonATen Tile integer uniforms are not Unity-compatible")
    (output / "Bindings.h").write_text(
        "inline const char *deviceSource() { return "
        + json.dumps(device_text)
        + "; }\n"
        + "inline std::vector<native_opencl::GatherBinding> manifestBindings("
        + "const std::string &convolution, const std::string &activation, "
        + "const std::string &pooling, const std::string &softmax, "
        + "const std::string &batchnorm, const std::string &opTensor, "
        + "const std::string &rmsNorm, const std::string &embedding, "
        + "const std::string &tile, "
        + "const std::string &atenRmsReduce, const std::string &atenRmsFinalize, "
        + "const std::string &sgemm, const std::string &sgemv, "
        + "const std::string &saxpy, const std::string &sdotProduct, "
        + "const std::string &sdotReduce, "
        + "const std::string &arenaPointwise, "
        + "const std::string &arenaTile, const std::string &arenaRmsReduce, "
        + "const std::string &arenaRmsFinalize) { return {"
        + '{200, 2048, 7, "udondnn_convolution_forward", convolution}, '
        + '{201, 2048, 6, "udondnn_activation_forward", activation}, '
        + '{202, 2048, 7, "udondnn_pooling_forward", pooling}, '
        + '{203, 2048, 5, "udondnn_softmax_forward", softmax}, '
        + '{204, 4096, 5, "udondnn_batchnorm_inference", batchnorm}, '
        + '{205, 2048, 8, "udondnn_op_tensor", opTensor}, '
        + '{206, 2048, 5, "udonaten_rms_norm_small", rmsNorm}, '
        + '{207, 2048, 6, "udonaten_embedding", embedding}, '
        + '{209, 8192, 9, "udonaten_tile", tile}, '
        + '{210, 8192, 2, "udonaten_rms_norm_reduce", atenRmsReduce}, '
        + '{211, 8192, 4, "udonaten_rms_norm_finalize", atenRmsFinalize, 0, 0, 2}, '
        + '{212, 1024, 10, "udonblas_sgemm", sgemm, 32, 32}, '
        + '{213, 1024, 8, "udonblas_sgemv", sgemv}, '
        + '{214, 1024, 4, "udonblas_saxpy", saxpy}, '
        + '{219, 1024, 3, "udonblas_sdot_product", sdotProduct}, '
        + '{220, 1024, 2, "udonblas_sdot_reduce", sdotReduce}, '
        + '{215, 16384, 4, "udonaten_arena_pointwise", arenaPointwise, 8192, 2, 2}, '
        + '{216, 16384, 9, "udonaten_arena_tile", arenaTile, 8192, 2}, '
        + '{217, 16384, 2, "udonaten_arena_rms_norm_reduce", arenaRmsReduce, 8192, 2}, '
        + '{218, 16384, 4, "udonaten_arena_rms_norm_finalize", arenaRmsFinalize, 8192, 2, 3}}; }\n',
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
    with tempfile.TemporaryDirectory(prefix="unity-udonaten-rms-", dir=output) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(
            root / "libraries/udonaten/unity/rms-norm-host.uasm",
            staged / "host.uasm",
        )
        shutil.copy2(
            root / "libraries/udonaten/unity/rms-norm-display.shader",
            staged / "display.shader",
        )
        shutil.copy2(
            rms_norm_small_bundle / "kernels/kernel_206.shader",
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
            "artifact_stage": "unity-udonaten-rms-norm",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udonaten_rms_norm",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "udonatenRmsNorm",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 206,
                        "name": "udonaten_rms_norm_small",
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
            if previous.get("artifact_stage") not in (
                "unity-udondnn-rms-norm",
                "unity-udonaten-rms-norm",
            ):
                raise RuntimeError(
                    "refusing to replace a non-UdonATen RMSNorm Unity bundle"
                )
            shutil.rmtree(unity_rms_bundle)
        publish_bundle(staged, unity_rms_bundle)

    tile_input_shape = (1, 2, 2, 3)
    tile_repeats = (1, 1, 2, 3)
    tile_output_shape = tuple(
        dimension * repeat
        for dimension, repeat in zip(tile_input_shape, tile_repeats)
    )
    tile_input = [((index * 23 + 7) % 31 - 15) * 0.125 for index in range(12)]
    tile_reference = []
    for n in range(tile_output_shape[0]):
        for c in range(tile_output_shape[1]):
            for h in range(tile_output_shape[2]):
                for w in range(tile_output_shape[3]):
                    source_index = (
                        ((n % tile_input_shape[0]) * tile_input_shape[1]
                         + c % tile_input_shape[1])
                        * tile_input_shape[2]
                        + h % tile_input_shape[2]
                    ) * tile_input_shape[3] + w % tile_input_shape[3]
                    tile_reference.append(tile_input[source_index])
    tile_state = [0.0] * 8192
    tile_state[: len(tile_input)] = tile_input
    unity_tile_bundle = output / "unity-udonaten-tile-bundle"
    with tempfile.TemporaryDirectory(
        prefix="unity-udonaten-tile-", dir=output
    ) as staging:
        staged = Path(staging) / "bundle"
        (staged / "kernels").mkdir(parents=True)
        shutil.copy2(
            root / "libraries/udonaten/unity/tile-host.uasm",
            staged / "host.uasm",
        )
        shutil.copy2(
            root / "libraries/udonaten/unity/tile-display.shader",
            staged / "display.shader",
        )
        shutil.copy2(
            tile_bundle / "kernels/kernel_209.shader",
            staged / "kernels/kernel_209.shader",
        )
        (staged / "fixture_input.bytes").write_bytes(
            struct.pack("<8192f", *tile_state)
        )
        (staged / "fixture_reference.bytes").write_bytes(
            struct.pack(f"<{len(tile_reference)}f", *tile_reference)
        )
        tile_unity_manifest = {
            "schema_version": 1,
            "compiler_version": "0.1.0-dev",
            "runtime_abi_version": 1,
            "artifact_stage": "unity-udonaten-tile",
            "ready_for_unity": True,
            "sdk_compatibility": {"sdk_version": "3.10.5", "extern_db_hash": None},
            "target": {
                "host_pointer_bits": 32,
                "device_pointer_bits": 32,
                "endianness": "little",
                "platform": "VRChat-PC",
            },
            "program": {
                "name": "udonaten_tile",
                "assembly": "host.uasm",
                "display_shader": "display.shader",
            },
            "exports": [
                {
                    "id": 1,
                    "c_symbol": "udonatenTile",
                    "udon_event": "_run",
                    "status_slot": "",
                    "async": False,
                }
            ],
            "device": {
                "kernels": [
                    {
                        "id": 209,
                        "name": "udonaten_tile",
                        "shader": "kernels/kernel_209.shader",
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
                "input_shape": list(tile_input_shape),
                "repeats": list(tile_repeats),
                "output_shape": list(tile_output_shape),
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(tile_unity_manifest, indent=2), encoding="utf-8"
        )
        if unity_tile_bundle.exists():
            previous = json.loads(
                (unity_tile_bundle / "manifest.json").read_text(encoding="utf-8")
            )
            if previous.get("artifact_stage") != "unity-udonaten-tile":
                raise RuntimeError("refusing to replace a non-UdonATen Tile Unity bundle")
            shutil.rmtree(unity_tile_bundle)
        publish_bundle(staged, unity_tile_bundle)

    rl_model = output / "rl-policy.onnx"
    run(
        [
            sys.executable,
            root / "examples/reinforcement-learning/generate_onnx.py",
            "-o",
            rl_model,
        ]
    )
    sys.path.insert(0, str(root / "runtime/onnx"))
    from udon_onnx import compile_operators, emit_c_host, parse_model

    rl_plan = compile_operators(parse_model(rl_model.read_bytes()))
    expected_rl_calls = [
        "udonblasSgemv",
        "udonatenAdd",
        "udonatenTanh",
        "udonatenSigmoid",
        "udonatenMul",
        "udonblasSgemv",
        "udonatenAdd",
        "udonatenSub",
        "udonatenMul",
        "udondnnSoftmaxForward",
    ]
    if (
        rl_plan["execution_provider"] != "UdonOperatorExecutionProvider"
        or [call["function"] for call in rl_plan["aot"]["calls"]]
        != expected_rl_calls
    ):
        raise RuntimeError("RL policy did not use the generic ONNX operator provider")
    (output / "rl-plan.json").write_text(
        json.dumps(rl_plan, indent=2), encoding="utf-8"
    )
    rl_host_text = emit_c_host(rl_plan)
    rl_host_source = output / "rl-host.c"
    rl_host_source.write_text(rl_host_text, encoding="utf-8")
    rl_declarations = rl_host_text.split("int udon_onnx_run", 1)[0]
    (output / "rl_host.h").write_text(
        rl_declarations
        + "int udon_onnx_run(udon_onnx_buffers_t *buffers);\n",
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
    blas_host_ir = output / "blas-host.ll"
    blas_imported = output / "blas-host.imported.mlir"
    blas_udonhl = output / "blas-host.udonhl.mlir"
    blas_lowered = output / "blas-host.lowered.mlir"
    blas_native_ir = output / "blas-host.native.ll"
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
            root / "libraries/udonblas/include",
            "-I",
            output,
            "-S",
            "-emit-llvm",
            root / "libraries/udonblas/src/udonblas.c",
            "-o",
            blas_host_ir,
        ]
    )
    run([translate, "--import-llvm", blas_host_ir, "-o", blas_imported])
    run(
        [
            opt,
            "--vrc-raise-native-host-calls",
            "--verify-each",
            blas_imported,
            "-o",
            blas_udonhl,
        ]
    )
    run(
        [
            opt,
            "--vrc-lower-native-host-calls",
            "--verify-each",
            blas_udonhl,
            "-o",
            blas_lowered,
        ]
    )
    run([mlir_translate, "--mlir-to-llvmir", blas_lowered, "-o", blas_native_ir])
    blas_host_object = output / ("blas-host" + object_suffix)
    rl_host_object = output / ("rl-host" + object_suffix)
    udonaten_object = output / ("udonaten" + object_suffix)
    device_object = output / ("convolution-device" + object_suffix)
    activation_object = output / ("activation-device" + object_suffix)
    pooling_object = output / ("pooling-device" + object_suffix)
    softmax_object = output / ("softmax-device" + object_suffix)
    batchnorm_object = output / ("batchnorm-device" + object_suffix)
    op_tensor_object = output / ("op-tensor-device" + object_suffix)
    rms_norm_small_object = output / ("rms-norm-small-device" + object_suffix)
    embedding_object = output / ("embedding-device" + object_suffix)
    tile_object = output / ("tile-device" + object_suffix)
    sgemm_object = output / ("sgemm-device" + object_suffix)
    sgemv_object = output / ("sgemv-device" + object_suffix)
    saxpy_object = output / ("saxpy-device" + object_suffix)
    sdot_object = output / ("sdot-device" + object_suffix)
    arena_pointwise_object = output / ("arena-pointwise-device" + object_suffix)
    arena_tile_object = output / ("arena-tile-device" + object_suffix)
    arena_rms_object = output / ("arena-rms-device" + object_suffix)
    run([clang, *runtime_flags, "-c", native_ir, "-o", host_object])
    run([clang, *runtime_flags, "-c", blas_native_ir, "-o", blas_host_object])
    run(
        [
            clang,
            *runtime_flags,
            "-std=c11",
            "-I",
            source / "include",
            "-I",
            root / "libraries/udonaten/include",
            "-I",
            root / "libraries/udonblas/include",
            "-c",
            rl_host_source,
            "-o",
            rl_host_object,
        ]
    )
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
            rms_norm_small_device,
            "-o",
            rms_norm_small_object,
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
            embedding_device,
            "-o",
            embedding_object,
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
            tile_device,
            "-o",
            tile_object,
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
            arena_pointwise_device,
            "-o",
            arena_pointwise_object,
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
            arena_tile_device,
            "-o",
            arena_tile_object,
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
            arena_rms_device,
            "-o",
            arena_rms_object,
        ]
    )
    for blas_device, blas_object in (
        (sgemm_device, sgemm_object),
        (sgemv_device, sgemv_object),
        (saxpy_device, saxpy_object),
        (sdot_device, sdot_object),
    ):
        run(
            [
                clang,
                *runtime_flags,
                "-x",
                "c",
                "-std=c11",
                "-ffp-contract=off",
                "-c",
                blas_device,
                "-o",
                blas_object,
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
                (rms_norm_small_fragment, "frag", "frag"),
                (embedding_fragment, "frag", "frag"),
                (tile_fragment, "frag", "frag"),
                (aten_rms_reduce_fragment, "frag", "frag"),
                (aten_rms_finalize_fragment, "frag", "frag"),
                (blas_fragments[0], "frag", "frag"),
                (blas_fragments[1], "frag", "frag"),
                (blas_fragments[2], "frag", "frag"),
                (blas_fragments[3], "frag", "frag"),
                (blas_fragments[4], "frag", "frag"),
                (arena_pointwise_fragment, "frag", "frag"),
                (arena_tile_fragment, "frag", "frag"),
                (arena_rms_reduce_fragment, "frag", "frag"),
                (arena_rms_finalize_fragment, "frag", "frag"),
                (arena_sgemm_fragment, "frag", "frag"),
                (arena_softmax_fragment, "frag", "frag"),
                (arena_broadcast_fragment, "frag", "frag"),
                (arena_convolution_fragment, "frag", "frag"),
                (arena_pooling_fragment, "frag", "frag"),
                (arena_embedding_fragment, "frag", "frag"),
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
            rms_norm_small_fragment,
            embedding_fragment,
            tile_fragment,
            aten_rms_reduce_fragment,
            aten_rms_finalize_fragment,
            *blas_fragments,
            arena_pointwise_fragment,
            arena_tile_fragment,
            arena_rms_reduce_fragment,
            arena_rms_finalize_fragment,
            arena_sgemm_fragment,
            arena_softmax_fragment,
            arena_broadcast_fragment,
            arena_convolution_fragment,
            arena_pooling_fragment,
            arena_embedding_fragment,
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
            "-DUDONBLAS_HOST_OBJECT=" + blas_host_object.as_posix(),
            "-DUDON_ONNX_RL_HOST_OBJECT=" + rl_host_object.as_posix(),
            "-DUDONATEN_OBJECT=" + udonaten_object.as_posix(),
            "-DUDONDNN_DEVICE_OBJECT=" + device_object.as_posix(),
            "-DUDONDNN_ACTIVATION_OBJECT=" + activation_object.as_posix(),
            "-DUDONDNN_POOLING_OBJECT=" + pooling_object.as_posix(),
            "-DUDONDNN_SOFTMAX_OBJECT=" + softmax_object.as_posix(),
            "-DUDONDNN_BATCHNORM_OBJECT=" + batchnorm_object.as_posix(),
            "-DUDONDNN_OP_TENSOR_OBJECT=" + op_tensor_object.as_posix(),
            "-DUDONATEN_RMS_NORM_SMALL_OBJECT=" + rms_norm_small_object.as_posix(),
            "-DUDONATEN_EMBEDDING_OBJECT=" + embedding_object.as_posix(),
            "-DUDONATEN_TILE_OBJECT=" + tile_object.as_posix(),
            "-DUDONBLAS_SGEMM_OBJECT=" + sgemm_object.as_posix(),
            "-DUDONBLAS_SGEMV_OBJECT=" + sgemv_object.as_posix(),
            "-DUDONBLAS_SAXPY_OBJECT=" + saxpy_object.as_posix(),
            "-DUDONBLAS_SDOT_OBJECT=" + sdot_object.as_posix(),
            "-DUDONATEN_ARENA_POINTWISE_OBJECT=" + arena_pointwise_object.as_posix(),
            "-DUDONATEN_ARENA_TILE_OBJECT=" + arena_tile_object.as_posix(),
            "-DUDONATEN_ARENA_RMS_OBJECT=" + arena_rms_object.as_posix(),
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
                rms_norm_small_fragment,
                embedding_fragment,
                tile_fragment,
                aten_rms_reduce_fragment,
                aten_rms_finalize_fragment,
                *blas_fragments,
                arena_pointwise_fragment,
                arena_tile_fragment,
                arena_rms_reduce_fragment,
                arena_rms_finalize_fragment,
                vertex,
            ]
        ),
        end="",
    )


if __name__ == "__main__":
    main()
