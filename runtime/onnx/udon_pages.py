"""Materialize bounded UdonATen Device pages from ONNX external weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import tempfile
import time


def _safe_external_path(directory: Path, location: str) -> Path:
    if not isinstance(location, str) or not location or "\\" in location:
        raise ValueError("external page location must be a non-empty POSIX path")
    relative = PurePosixPath(location)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("external page location escapes its data directory")
    directory = directory.resolve()
    candidate = (directory / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(directory)
    except ValueError as error:
        raise ValueError("external page location escapes its data directory") from error
    if not candidate.is_file():
        raise ValueError(f"external page file does not exist: {location}")
    return candidate


def _read_external(
    directory: Path, external: dict, relative_offset: int, length: int
) -> bytes:
    if not isinstance(external, dict):
        raise ValueError("external page descriptor is missing")
    base = external.get("offset")
    extent = external.get("length")
    if (
        not isinstance(base, int)
        or not isinstance(extent, int)
        or base < 0
        or extent < 0
        or relative_offset < 0
        or length < 0
        or relative_offset + length > extent
    ):
        raise ValueError("external page read exceeds its tensor range")
    path = _safe_external_path(directory, external.get("location"))
    if base + extent > path.stat().st_size:
        raise ValueError("external tensor range exceeds its file")
    with path.open("rb") as stream:
        stream.seek(base + relative_offset)
        payload = stream.read(length)
    if len(payload) != length:
        raise ValueError("external page read was truncated")
    return payload


def _bfloat16(raw: bytes) -> float:
    if len(raw) != 2:
        raise ValueError("bfloat16 value must occupy two bytes")
    bits = struct.unpack("<H", raw)[0] << 16
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def linear_texture_layout(words: int) -> dict:
    """Map a positive linear word count to the shared 8,192-wide texture ABI."""
    if not isinstance(words, int) or words <= 0:
        raise ValueError("linear texture word count must be positive")
    return {"width": min(8192, words), "height": math.ceil(words / 8192)}


def _replace_path(source: Path, destination: Path):
    """Retry an atomic rename while Windows releases short-lived scanner handles."""
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def _int4_call_contract(plan: dict, call_index: int) -> dict:
    """Validate and return one tiled ATen INT4 call and its fixed paging ABI."""
    if plan.get("runtime") != "udon-onnx":
        raise ValueError("page source is not a Udon ONNX execution plan")
    calls = (plan.get("aot") or {}).get("calls") or []
    matches = [call for call in calls if call.get("index") == call_index]
    if len(matches) != 1:
        raise ValueError(f"execution plan has no unique call index {call_index}")
    call = matches[0]
    if (
        call.get("kind") != "tiled_library"
        or call.get("function") != "udonatenWeightInt4packMm"
        or call.get("group_size") != 32
        or not isinstance(call.get("inputs"), list)
        or len(call["inputs"]) < 3
        or not all(isinstance(name, str) and name for name in call["inputs"][:3])
        or not isinstance(call.get("output"), str)
        or not call["output"]
    ):
        raise ValueError("execution-plan call is not a supported ATen INT4 tile")
    grid = call.get("tile_grid") or {}
    paging = call.get("paging") or {}
    if (
        grid.get("m") != 64
        or grid.get("k") != 128
        or not isinstance(grid.get("m_tiles"), int)
        or not isinstance(grid.get("k_tiles"), int)
        or grid["m_tiles"] <= 0
        or grid["k_tiles"] <= 0
        or paging.get("state_words") != 8192
        or paging.get("input_offset_words") != 6144
        or paging.get("result_offset_words") != 7168
        or paging.get("row_stride_words") != 72
        or paging.get("accumulate_argument") != 5
    ):
        raise ValueError("execution-plan INT4 paging ABI is unsupported")
    attributes = call.get("attributes") or {}
    output_features = (attributes.get("output_features") or {}).get("i")
    input_features = (attributes.get("input_features") or {}).get("i")
    if (
        not isinstance(output_features, int)
        or not isinstance(input_features, int)
        or output_features <= 0
        or input_features <= 0
        or input_features % 32
        or grid["m_tiles"] != math.ceil(output_features / 64)
        or grid["k_tiles"] != math.ceil(input_features / 128)
    ):
        raise ValueError("INT4 call has invalid feature dimensions")
    return {
        "call": call,
        "grid": grid,
        "paging": paging,
        "output_features": output_features,
        "input_features": input_features,
    }


def materialize_int4_page(
    plan: dict,
    call_index: int,
    m_tile: int,
    k_tile: int,
    external_data_directory: Path,
    activation=None,
    accumulator=None,
) -> dict:
    """Gather one ATen INT4 tile into the kernel-208 active-state layout."""
    contract = _int4_call_contract(plan, call_index)
    call = contract["call"]
    grid = contract["grid"]
    paging = contract["paging"]
    if (
        not isinstance(m_tile, int)
        or not isinstance(k_tile, int)
        or not 0 <= m_tile < grid.get("m_tiles", 0)
        or not 0 <= k_tile < grid.get("k_tiles", 0)
    ):
        raise ValueError("INT4 tile coordinates are outside the planned grid")

    output_features = contract["output_features"]
    input_features = contract["input_features"]
    m_offset = m_tile * 64
    k_offset = k_tile * 128
    n = min(64, output_features - m_offset)
    k = min(128, input_features - k_offset)
    groups = math.ceil(k / 32)
    packed = paging.get("packed_weight") or {}
    qparams = paging.get("scales_and_zeros") or {}
    if (
        packed.get("source_row_stride_bytes") != input_features // 2
        or packed.get("tile_row_bytes") != 64
        or qparams.get("source_group_stride_bytes") != output_features * 4
        or qparams.get("source_row_stride_bytes") != 4
        or qparams.get("tile_groups") != 4
        or qparams.get("destination_group_stride_words") != 18
        or qparams.get("scale_word") != 16
        or qparams.get("zero_word") != 17
    ):
        raise ValueError("INT4 call paging strides do not match its tensor shapes")

    state = [0.0] * 8192
    row_stride = paging["row_stride_words"]
    for row in range(n):
        source_row = m_offset + row
        packed_relative = source_row * (input_features // 2) + k_offset // 2
        packed_row = _read_external(
            external_data_directory,
            packed.get("external"),
            packed_relative,
            k // 2,
        )
        for block in range(groups):
            destination = row * row_stride + block * 18
            block_bytes = packed_row[block * 16 : (block + 1) * 16]
            for byte, value in enumerate(block_bytes):
                state[destination + byte] = float(value)
            source_group = k_offset // 32 + block
            qparam_relative = (source_group * output_features + source_row) * 4
            scale_zero = _read_external(
                external_data_directory,
                qparams.get("external"),
                qparam_relative,
                4,
            )
            state[destination + 16] = _bfloat16(scale_zero[:2])
            state[destination + 17] = _bfloat16(scale_zero[2:])

    if activation is not None:
        activation = list(activation)
        if len(activation) != input_features:
            raise ValueError("activation length does not match INT4 input features")
        for index, value in enumerate(activation[k_offset : k_offset + k]):
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("activation page values must be finite")
            state[6144 + index] = value
    if accumulator is not None:
        accumulator = list(accumulator)
        if len(accumulator) != output_features:
            raise ValueError("accumulator length does not match INT4 output features")
        for index, value in enumerate(accumulator[m_offset : m_offset + n]):
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("accumulator page values must be finite")
            state[7168 + index] = value
    return {
        "call_index": call_index,
        "m_tile": m_tile,
        "k_tile": k_tile,
        "m_offset": m_offset,
        "k_offset": k_offset,
        "n": n,
        "k": k,
        "row_stride": row_stride,
        "state": state,
    }


def encode_int4_page(page: dict) -> bytes:
    state = page.get("state")
    if not isinstance(state, list) or len(state) != 8192:
        raise ValueError("INT4 page state must contain 8,192 float words")
    return struct.pack("<8192f", *state)


def emit_int4_call_uasm(plan: dict, call_index: int) -> str:
    """Emit a static Udon Host dispatcher for every M/K page of one INT4 call."""
    contract = _int4_call_contract(plan, call_index)
    grid = contract["grid"]
    output_features = contract["output_features"]
    input_features = contract["input_features"]
    pages = [
        (m_tile, k_tile, f"pageM{m_tile}K{k_tile}")
        for m_tile in range(grid["m_tiles"])
        for k_tile in range(grid["k_tiles"])
    ]
    lines = [
        ".data_start",
        "    .export activationState",
        "    .export composeState",
        "    .export accumulatorState",
        "    .export resultA",
        "    .export resultB",
        "    .export resultState",
        "    .export composeMaterial",
        "    .export int4Material",
        "    .export scatterMaterial",
    ]
    lines.extend(f"    .export {name}" for _, _, name in pages)
    lines.extend(
        [
            "",
            "    activationState: %UnityEngineTexture, null",
            "    composeState: %UnityEngineRenderTexture, null",
            "    accumulatorState: %UnityEngineRenderTexture, null",
            "    resultA: %UnityEngineRenderTexture, null",
            "    resultB: %UnityEngineRenderTexture, null",
            "    resultState: %UnityEngineRenderTexture, null",
            "    composeMaterial: %UnityEngineMaterial, null",
            "    int4Material: %UnityEngineMaterial, null",
            "    scatterMaterial: %UnityEngineMaterial, null",
        ]
    )
    lines.extend(f"    {name}: %UnityEngineTexture, null" for _, _, name in pages)
    constants = {
        "__zero": 0,
        "__one": 1,
        "__rowStride": 72,
        "__resultOffset": 7168,
        "__outputFeatures": output_features,
        "__activationWidth": linear_texture_layout(input_features)["width"],
        "__outputWidth": linear_texture_layout(output_features)["width"],
    }
    for m_tile in range(grid["m_tiles"]):
        constants[f"__mOffset{m_tile}"] = m_tile * 64
        constants[f"__n{m_tile}"] = min(64, output_features - m_tile * 64)
    for k_tile in range(grid["k_tiles"]):
        constants[f"__kOffset{k_tile}"] = k_tile * 128
        constants[f"__k{k_tile}"] = min(128, input_features - k_tile * 128)
    lines.extend(f"    {name}: %SystemInt32, {value}" for name, value in constants.items())
    strings = {
        "__weightPageName": "_WeightPage",
        "__activationName": "_Activation",
        "__accumulatorName": "_Accumulator",
        "__activationOffsetName": "_ActivationOffset",
        "__activationWidthName": "_ActivationWidth",
        "__accumulatorOffsetName": "_AccumulatorOffset",
        "__kOffsetName": "_KOffset",
        "__composeNName": "_N",
        "__composeKName": "_K",
        "__composeAccumulateName": "_Accumulate",
        "__inputName": "_VRC_Input",
        "__nName": "_VRC_Arg2",
        "__kName": "_VRC_Arg3",
        "__rowStrideName": "_VRC_Arg4",
        "__accumulateName": "_VRC_Arg5",
        "__scatterSourceName": "_TileState",
        "__scatterPriorName": "_PriorResult",
        "__scatterOffsetName": "_OutputOffset",
        "__scatterNName": "_N",
        "__scatterWidthName": "_OutputWidth",
        "__scatterFeaturesName": "_OutputFeatures",
        "__scatterPriorFlagName": "_HasPrior",
    }
    lines.extend(f'    {name}: %SystemString, "{value}"' for name, value in strings.items())
    lines.extend(
        [
            ".data_end",
            "",
            ".code_start",
            "    .export _start",
            "    _start:",
            "        JUMP, 0xFFFFFFFC",
            "",
            "    .export _run",
            "    _run:",
        ]
    )

    def set_texture(material: str, property_name: str, texture: str):
        lines.extend(
            [
                f"        PUSH, {material}",
                f"        PUSH, {property_name}",
                f"        PUSH, {texture}",
                '        EXTERN, "UnityEngineMaterial.__SetTexture__SystemString_UnityEngineTexture__SystemVoid"',
            ]
        )

    def set_int(material: str, property_name: str, value: str):
        lines.extend(
            [
                f"        PUSH, {material}",
                f"        PUSH, {property_name}",
                f"        PUSH, {value}",
                '        EXTERN, "UnityEngineMaterial.__SetInt__SystemString_SystemInt32__SystemVoid"',
            ]
        )

    def blit(source: str, target: str, material: str):
        lines.extend(
            [
                f"        PUSH, {source}",
                f"        PUSH, {target}",
                f"        PUSH, {material}",
                '        EXTERN, "VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid"',
            ]
        )

    set_texture("composeMaterial", "__activationName", "activationState")
    set_int("composeMaterial", "__activationOffsetName", "__zero")
    set_int("composeMaterial", "__activationWidthName", "__activationWidth")
    set_int("composeMaterial", "__accumulatorOffsetName", "__resultOffset")
    set_int("int4Material", "__rowStrideName", "__rowStride")
    set_int("scatterMaterial", "__scatterWidthName", "__outputWidth")
    set_int("scatterMaterial", "__scatterFeaturesName", "__outputFeatures")
    previous_result = None
    for m_tile in range(grid["m_tiles"]):
        for k_tile in range(grid["k_tiles"]):
            page = f"pageM{m_tile}K{k_tile}"
            accumulate = "__zero" if k_tile == 0 else "__one"
            set_texture("composeMaterial", "__weightPageName", page)
            set_texture("composeMaterial", "__accumulatorName", "accumulatorState")
            set_int("composeMaterial", "__kOffsetName", f"__kOffset{k_tile}")
            set_int("composeMaterial", "__composeNName", f"__n{m_tile}")
            set_int("composeMaterial", "__composeKName", f"__k{k_tile}")
            set_int("composeMaterial", "__composeAccumulateName", accumulate)
            blit(page, "composeState", "composeMaterial")
            set_texture("int4Material", "__inputName", "composeState")
            set_int("int4Material", "__nName", f"__n{m_tile}")
            set_int("int4Material", "__kName", f"__k{k_tile}")
            set_int("int4Material", "__accumulateName", accumulate)
            blit("composeState", "accumulatorState", "int4Material")
        target_result = "resultA" if m_tile % 2 == 0 else "resultB"
        set_texture("scatterMaterial", "__scatterSourceName", "accumulatorState")
        set_texture(
            "scatterMaterial",
            "__scatterPriorName",
            previous_result or "accumulatorState",
        )
        set_int("scatterMaterial", "__scatterOffsetName", f"__mOffset{m_tile}")
        set_int("scatterMaterial", "__scatterNName", f"__n{m_tile}")
        set_int(
            "scatterMaterial",
            "__scatterPriorFlagName",
            "__zero" if previous_result is None else "__one",
        )
        blit("accumulatorState", target_result, "scatterMaterial")
        previous_result = target_result
    lines.extend(
        [
            f"        PUSH, {previous_result}",
            "        PUSH, resultState",
            "        COPY",
            "        JUMP, 0xFFFFFFFC",
            ".code_end",
            "",
        ]
    )
    return "\n".join(lines)


def materialize_int4_call(
    plan: dict,
    call_index: int,
    external_data_directory: Path,
    output_directory: Path,
) -> dict:
    """Atomically materialize every static weight page and dispatcher for one call."""
    contract = _int4_call_contract(plan, call_index)
    grid = contract["grid"]
    external_data_directory = external_data_directory.resolve()
    output_directory = output_directory.resolve()
    try:
        external_data_directory.relative_to(output_directory)
    except ValueError:
        pass
    else:
        raise ValueError("INT4 call output cannot contain its external data directory")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output_directory.name + "-", dir=output_directory.parent)
    )
    try:
        pages_directory = staging / "pages"
        pages_directory.mkdir()
        resources = []
        for m_tile in range(grid["m_tiles"]):
            for k_tile in range(grid["k_tiles"]):
                page = materialize_int4_page(
                    plan,
                    call_index,
                    m_tile,
                    k_tile,
                    external_data_directory,
                )
                relative = f"pages/m{m_tile:04d}-k{k_tile:04d}.bytes"
                payload = encode_int4_page(page)
                (staging / relative).write_bytes(payload)
                resources.append(
                    {
                        "name": f"pageM{m_tile}K{k_tile}",
                        "kind": "bytes",
                        "path": relative,
                        "m_tile": m_tile,
                        "k_tile": k_tile,
                        "n": page["n"],
                        "k": page["k"],
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
        (staging / "host.uasm").write_text(
            emit_int4_call_uasm(plan, call_index), encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "artifact_stage": "udon-onnx-int4-call",
            "call_index": call_index,
            "function": "udonatenWeightInt4packMm",
            "input": contract["call"]["inputs"][0],
            "output": contract["call"]["output"],
            "input_features": contract["input_features"],
            "output_features": contract["output_features"],
            "activation_layout": linear_texture_layout(contract["input_features"]),
            "result_layout": linear_texture_layout(contract["output_features"]),
            "tile_grid": dict(grid),
            "assembly": "host.uasm",
            "required_shaders": [
                "weight-int4pack-mm-compose.shader",
                "weight-int4pack-mm-scatter.shader",
                "kernel_208.shader",
            ],
            "resources": resources,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        backup = None
        if output_directory.exists():
            previous_manifest = output_directory / "manifest.json"
            if not previous_manifest.is_file() or json.loads(
                previous_manifest.read_text(encoding="utf-8")
            ).get("artifact_stage") != "udon-onnx-int4-call":
                raise ValueError("refusing to replace a non-INT4-call output directory")
            backup = Path(
                tempfile.mkdtemp(
                    prefix=output_directory.name + "-backup-",
                    dir=output_directory.parent,
                )
            )
            backup.rmdir()
            _replace_path(output_directory, backup)
        try:
            _replace_path(staging, output_directory)
        except BaseException:
            if backup is not None:
                _replace_path(backup, output_directory)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("external_data_directory", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--call", type=int, required=True)
    parser.add_argument("--m-tile", type=int)
    parser.add_argument("--k-tile", type=int)
    parser.add_argument(
        "--all",
        action="store_true",
        help="materialize every M/K page plus the static Udon Host dispatcher",
    )
    parser.add_argument(
        "--activation-f32",
        type=Path,
        help="optional complete float32 activation vector to copy into the page",
    )
    parser.add_argument(
        "--accumulator-f32",
        type=Path,
        help="optional complete float32 output accumulator for a later K tile",
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.all:
        if args.m_tile is not None or args.k_tile is not None:
            parser.error("--all cannot be combined with --m-tile or --k-tile")
        if args.activation_f32 or args.accumulator_f32:
            parser.error("--all accepts runtime activations and accumulators through Udon")
        manifest = materialize_int4_call(
            plan,
            args.call,
            args.external_data_directory,
            args.output,
        )
        print(
            f"udon_onnx_int4_call={args.output} call={args.call} "
            f"pages={len(manifest['resources'])}"
        )
        return
    if args.m_tile is None or args.k_tile is None:
        parser.error("single-page mode requires --m-tile and --k-tile")
    activation = None
    if args.activation_f32:
        raw = args.activation_f32.read_bytes()
        if len(raw) % 4:
            parser.error("activation file size is not a multiple of four bytes")
        activation = struct.unpack(f"<{len(raw) // 4}f", raw)
    accumulator = None
    if args.accumulator_f32:
        raw = args.accumulator_f32.read_bytes()
        if len(raw) % 4:
            parser.error("accumulator file size is not a multiple of four bytes")
        accumulator = struct.unpack(f"<{len(raw) // 4}f", raw)
    page = materialize_int4_page(
        plan,
        args.call,
        args.m_tile,
        args.k_tile,
        args.external_data_directory,
        activation,
        accumulator,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encode_int4_page(page))
    print(
        f"udon_onnx_int4_page={args.output} call={page['call_index']} "
        f"tile={page['m_tile']},{page['k_tile']} n={page['n']} k={page['k']}"
    )


if __name__ == "__main__":
    main()
