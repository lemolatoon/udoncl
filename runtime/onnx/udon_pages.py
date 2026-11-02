"""Materialize Udon ONNX model bundles and bounded external-weight pages."""

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


INT4_COMPACT_ROW_BYTES = 80
INT4_COMPACT_PAGE_BYTES = 64 * INT4_COMPACT_ROW_BYTES
INT4_COMPACT_ATLAS_WIDTH = INT4_COMPACT_PAGE_BYTES // 4
INT4_ATLAS_MAX_PAGES = 8192


def _uasm_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Udon assembly float literal must be finite")
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return text if "." in text else text + ".0"


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


def int4_atlas_location(page_index: int) -> dict:
    """Map a non-negative linear page index to a bounded atlas and row."""
    if not isinstance(page_index, int) or page_index < 0:
        raise ValueError("INT4 page index must be non-negative")
    return {
        "atlas": page_index // INT4_ATLAS_MAX_PAGES,
        "row": page_index % INT4_ATLAS_MAX_PAGES,
    }


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


def _publish_artifact(staging: Path, output_directory: Path, artifact_stage: str):
    """Publish a staged artifact with validation, retry, and rollback."""
    backup = None
    if output_directory.exists():
        previous_manifest = output_directory / "manifest.json"
        if not previous_manifest.is_file() or json.loads(
            previous_manifest.read_text(encoding="utf-8")
        ).get("artifact_stage") != artifact_stage:
            raise ValueError(f"refusing to replace a non-{artifact_stage} output directory")
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


def materialize_int4_compact_page(
    plan: dict,
    call_index: int,
    m_tile: int,
    k_tile: int,
    external_data_directory: Path,
) -> dict:
    """Gather one tile as 64 compact 80-byte Q4/BF16 rows."""
    contract = _int4_call_contract(plan, call_index)
    grid = contract["grid"]
    if (
        not isinstance(m_tile, int)
        or not isinstance(k_tile, int)
        or not 0 <= m_tile < grid["m_tiles"]
        or not 0 <= k_tile < grid["k_tiles"]
    ):
        raise ValueError("INT4 tile coordinates are outside the planned grid")
    output_features = contract["output_features"]
    input_features = contract["input_features"]
    m_offset = m_tile * 64
    k_offset = k_tile * 128
    n = min(64, output_features - m_offset)
    k = min(128, input_features - k_offset)
    groups = math.ceil(k / 32)
    paging = contract["paging"]
    packed = paging.get("packed_weight") or {}
    qparams = paging.get("scales_and_zeros") or {}
    if (
        packed.get("source_row_stride_bytes") != input_features // 2
        or qparams.get("source_group_stride_bytes") != output_features * 4
        or qparams.get("source_row_stride_bytes") != 4
    ):
        raise ValueError("INT4 call paging strides do not match its tensor shapes")
    payload = bytearray(INT4_COMPACT_PAGE_BYTES)
    for row in range(n):
        source_row = m_offset + row
        packed_relative = source_row * (input_features // 2) + k_offset // 2
        packed_row = _read_external(
            external_data_directory,
            packed.get("external"),
            packed_relative,
            k // 2,
        )
        destination = row * INT4_COMPACT_ROW_BYTES
        payload[destination : destination + len(packed_row)] = packed_row
        for block in range(groups):
            source_group = k_offset // 32 + block
            qparam_relative = (source_group * output_features + source_row) * 4
            qparam = _read_external(
                external_data_directory,
                qparams.get("external"),
                qparam_relative,
                4,
            )
            qparam_destination = destination + 64 + block * 4
            payload[qparam_destination : qparam_destination + 4] = qparam
    return {
        "call_index": call_index,
        "m_tile": m_tile,
        "k_tile": k_tile,
        "m_offset": m_offset,
        "k_offset": k_offset,
        "n": n,
        "k": k,
        "data": bytes(payload),
    }


def decode_int4_compact_page(page: dict) -> list[float]:
    """Decode compact bytes to the kernel-208 weight area for validation."""
    payload = page.get("data")
    if not isinstance(payload, bytes) or len(payload) != INT4_COMPACT_PAGE_BYTES:
        raise ValueError("compact INT4 page must contain 5,120 bytes")
    state = [0.0] * 8192
    for row in range(64):
        source = row * INT4_COMPACT_ROW_BYTES
        destination = row * 72
        for block in range(4):
            for byte in range(16):
                state[destination + block * 18 + byte] = float(
                    payload[source + block * 16 + byte]
                )
            qparam = source + 64 + block * 4
            state[destination + block * 18 + 16] = _bfloat16(
                payload[qparam : qparam + 2]
            )
            state[destination + block * 18 + 17] = _bfloat16(
                payload[qparam + 2 : qparam + 4]
            )
    return state


def _emit_unrolled_int4_call_uasm(plan: dict, call_index: int) -> str:
    """Reference emitter used to compare the bounded loop dispatcher."""
    contract = _int4_call_contract(plan, call_index)
    grid = contract["grid"]
    output_features = contract["output_features"]
    input_features = contract["input_features"]
    page_count = grid["m_tiles"] * grid["k_tiles"]
    atlases = [
        f"pageAtlas{index}"
        for index in range(math.ceil(page_count / INT4_ATLAS_MAX_PAGES))
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
    lines.extend(f"    .export {name}" for name in atlases)
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
    lines.extend(f"    {name}: %UnityEngineTexture, null" for name in atlases)
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
    for page in range(page_count):
        constants[f"__pageRow{page}"] = int4_atlas_location(page)["row"]
    lines.extend(f"    {name}: %SystemInt32, {value}" for name, value in constants.items())
    strings = {
        "__weightPageName": "_WeightPage",
        "__compactPageName": "_CompactPage",
        "__pageRowName": "_PageRow",
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
                '        EXTERN, "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"',
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
    set_int("composeMaterial", "__compactPageName", "__one")
    set_int("composeMaterial", "__activationOffsetName", "__zero")
    set_int("composeMaterial", "__activationWidthName", "__activationWidth")
    set_int("composeMaterial", "__accumulatorOffsetName", "__resultOffset")
    set_int("int4Material", "__rowStrideName", "__rowStride")
    set_int("scatterMaterial", "__scatterWidthName", "__outputWidth")
    set_int("scatterMaterial", "__scatterFeaturesName", "__outputFeatures")
    previous_result = None
    for m_tile in range(grid["m_tiles"]):
        for k_tile in range(grid["k_tiles"]):
            page = m_tile * grid["k_tiles"] + k_tile
            atlas = f"pageAtlas{int4_atlas_location(page)['atlas']}"
            accumulate = "__zero" if k_tile == 0 else "__one"
            set_texture("composeMaterial", "__weightPageName", atlas)
            set_int("composeMaterial", "__pageRowName", f"__pageRow{page}")
            set_texture("composeMaterial", "__accumulatorName", "accumulatorState")
            set_int("composeMaterial", "__kOffsetName", f"__kOffset{k_tile}")
            set_int("composeMaterial", "__composeNName", f"__n{m_tile}")
            set_int("composeMaterial", "__composeKName", f"__k{k_tile}")
            set_int("composeMaterial", "__composeAccumulateName", accumulate)
            blit(atlas, "composeState", "composeMaterial")
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


def emit_int4_call_uasm(
    plan: dict,
    call_index: int,
    *,
    activation_storage_width: int | None = None,
    result_storage_width: int | None = None,
) -> str:
    """Emit a bounded counter-based Udon Host dispatcher for one INT4 call."""
    contract = _int4_call_contract(plan, call_index)
    grid = contract["grid"]
    output_features = contract["output_features"]
    input_features = contract["input_features"]
    activation_storage_width = activation_storage_width or linear_texture_layout(
        input_features
    )["width"]
    result_storage_width = result_storage_width or linear_texture_layout(output_features)[
        "width"
    ]
    for label, width in (
        ("activation", activation_storage_width),
        ("result", result_storage_width),
    ):
        if not isinstance(width, int) or not 1 <= width <= 8192:
            raise ValueError(f"INT4 {label} storage width must be between 1 and 8,192")
    page_count = grid["m_tiles"] * grid["k_tiles"]
    atlas_count = math.ceil(page_count / INT4_ATLAS_MAX_PAGES)
    atlas_names = [f"pageAtlas{index}" for index in range(atlas_count)]
    last_n = output_features - (grid["m_tiles"] - 1) * 64
    last_k = input_features - (grid["k_tiles"] - 1) * 128
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
    lines.extend(f"    .export {name}" for name in atlas_names)
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
    lines.extend(f"    {name}: %UnityEngineTexture, null" for name in atlas_names)
    lines.extend(
        [
            "    __selectedAtlas: %UnityEngineTexture, null",
            "    __previousResult: %UnityEngineTexture, null",
            "    __mTile: %SystemInt32, null",
            "    __kTile: %SystemInt32, null",
            "    __mOffset: %SystemInt32, null",
            "    __kOffset: %SystemInt32, null",
            "    __pageRow: %SystemInt32, null",
            "    __atlasIndex: %SystemInt32, null",
            "    __currentN: %SystemInt32, null",
            "    __currentK: %SystemInt32, null",
            "    __resultSelector: %SystemInt32, null",
            "    __hasPrior: %SystemInt32, null",
            "    __condition: %SystemBoolean, null",
            "    __zero: %SystemInt32, 0",
            "    __one: %SystemInt32, 1",
            "    __tileN: %SystemInt32, 64",
            "    __tileK: %SystemInt32, 128",
            "    __rowStride: %SystemInt32, 72",
            "    __resultOffset: %SystemInt32, 7168",
            f"    __mTiles: %SystemInt32, {grid['m_tiles']}",
            f"    __kTiles: %SystemInt32, {grid['k_tiles']}",
            f"    __lastMTile: %SystemInt32, {grid['m_tiles'] - 1}",
            f"    __lastKTile: %SystemInt32, {grid['k_tiles'] - 1}",
            f"    __lastN: %SystemInt32, {last_n}",
            f"    __lastK: %SystemInt32, {last_k}",
            f"    __atlasMaxPages: %SystemInt32, {INT4_ATLAS_MAX_PAGES}",
            f"    __outputFeatures: %SystemInt32, {output_features}",
            f"    __activationWidth: %SystemInt32, {activation_storage_width}",
            f"    __outputWidth: %SystemInt32, {result_storage_width}",
        ]
    )
    lines.extend(
        f"    __atlas{index}: %SystemInt32, {index}" for index in range(atlas_count)
    )
    strings = {
        "__weightPageName": "_WeightPage",
        "__compactPageName": "_CompactPage",
        "__pageRowName": "_PageRow",
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

    def copy(destination: str, source: str):
        lines.extend([f"        PUSH, {source}", f"        PUSH, {destination}", "        COPY"])

    def binary(destination: str, left: str, right: str, operation: str):
        lines.extend(
            [
                f"        PUSH, {left}",
                f"        PUSH, {right}",
                f"        PUSH, {destination}",
                f'        EXTERN, "SystemInt32.__op_{operation}__SystemInt32_SystemInt32__SystemInt32"',
            ]
        )

    def compare(left: str, right: str, operation: str):
        lines.extend(
            [
                f"        PUSH, {left}",
                f"        PUSH, {right}",
                "        PUSH, __condition",
                f'        EXTERN, "SystemInt32.__op_{operation}__SystemInt32_SystemInt32__SystemBoolean"',
            ]
        )

    def jump_if_false(label: str):
        lines.extend(["        PUSH, __condition", f"        JUMP_IF_FALSE, {label}"])

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
                '        EXTERN, "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"',
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

    for variable in (
        "__mTile",
        "__kTile",
        "__mOffset",
        "__kOffset",
        "__pageRow",
        "__atlasIndex",
        "__resultSelector",
        "__hasPrior",
    ):
        copy(variable, "__zero")
    copy("__previousResult", "accumulatorState")
    set_texture("composeMaterial", "__activationName", "activationState")
    set_int("composeMaterial", "__compactPageName", "__one")
    set_int("composeMaterial", "__activationOffsetName", "__zero")
    set_int("composeMaterial", "__activationWidthName", "__activationWidth")
    set_int("composeMaterial", "__accumulatorOffsetName", "__resultOffset")
    set_int("int4Material", "__rowStrideName", "__rowStride")
    set_int("scatterMaterial", "__scatterWidthName", "__outputWidth")
    set_int("scatterMaterial", "__scatterFeaturesName", "__outputFeatures")
    lines.append("    __mLoop:")
    compare("__mTile", "__lastMTile", "Equality")
    jump_if_false("__fullN")
    copy("__currentN", "__lastN")
    lines.append("        JUMP, __kLoop")
    lines.append("    __fullN:")
    copy("__currentN", "__tileN")
    lines.append("    __kLoop:")
    compare("__kTile", "__lastKTile", "Equality")
    jump_if_false("__fullK")
    copy("__currentK", "__lastK")
    lines.append("        JUMP, __kReady")
    lines.append("    __fullK:")
    copy("__currentK", "__tileK")
    lines.append("    __kReady:")
    for index, atlas in enumerate(atlas_names):
        next_label = f"__atlasNext{index}"
        compare("__atlasIndex", f"__atlas{index}", "Equality")
        jump_if_false(next_label)
        copy("__selectedAtlas", atlas)
        lines.append("        JUMP, __atlasReady")
        lines.append(f"    {next_label}:")
    lines.append("        JUMP, __halt")
    lines.append("    __atlasReady:")
    set_texture("composeMaterial", "__weightPageName", "__selectedAtlas")
    set_texture("composeMaterial", "__accumulatorName", "accumulatorState")
    set_int("composeMaterial", "__pageRowName", "__pageRow")
    set_int("composeMaterial", "__kOffsetName", "__kOffset")
    set_int("composeMaterial", "__composeNName", "__currentN")
    set_int("composeMaterial", "__composeKName", "__currentK")
    set_int("composeMaterial", "__composeAccumulateName", "__kTile")
    blit("__selectedAtlas", "composeState", "composeMaterial")
    set_texture("int4Material", "__inputName", "composeState")
    set_int("int4Material", "__nName", "__currentN")
    set_int("int4Material", "__kName", "__currentK")
    set_int("int4Material", "__accumulateName", "__kTile")
    blit("composeState", "accumulatorState", "int4Material")
    binary("__pageRow", "__pageRow", "__one", "Addition")
    compare("__pageRow", "__atlasMaxPages", "Equality")
    jump_if_false("__pageReady")
    copy("__pageRow", "__zero")
    binary("__atlasIndex", "__atlasIndex", "__one", "Addition")
    lines.append("    __pageReady:")
    binary("__kTile", "__kTile", "__one", "Addition")
    binary("__kOffset", "__kOffset", "__tileK", "Addition")
    compare("__kTile", "__kTiles", "LessThan")
    jump_if_false("__scatter")
    lines.append("        JUMP, __kLoop")
    lines.append("    __scatter:")
    set_texture("scatterMaterial", "__scatterSourceName", "accumulatorState")
    set_texture("scatterMaterial", "__scatterPriorName", "__previousResult")
    set_int("scatterMaterial", "__scatterOffsetName", "__mOffset")
    set_int("scatterMaterial", "__scatterNName", "__currentN")
    set_int("scatterMaterial", "__scatterPriorFlagName", "__hasPrior")
    compare("__resultSelector", "__zero", "Equality")
    jump_if_false("__scatterB")
    blit("accumulatorState", "resultA", "scatterMaterial")
    copy("__previousResult", "resultA")
    copy("__resultSelector", "__one")
    lines.append("        JUMP, __scatterDone")
    lines.append("    __scatterB:")
    blit("accumulatorState", "resultB", "scatterMaterial")
    copy("__previousResult", "resultB")
    copy("__resultSelector", "__zero")
    lines.append("    __scatterDone:")
    copy("__hasPrior", "__one")
    binary("__mTile", "__mTile", "__one", "Addition")
    binary("__mOffset", "__mOffset", "__tileN", "Addition")
    copy("__kTile", "__zero")
    copy("__kOffset", "__zero")
    compare("__mTile", "__mTiles", "LessThan")
    jump_if_false("__complete")
    lines.append("        JUMP, __mLoop")
    lines.append("    __complete:")
    copy("resultState", "__previousResult")
    lines.extend(["    __halt:", "        JUMP, 0xFFFFFFFC", ".code_end", ""])
    return "\n".join(lines)


def _model_arena_binding(plan: dict, name: str) -> dict:
    arena = plan.get("texture_arena") or {}
    bindings = arena.get("bindings") or {}
    binding = bindings.get(name)
    if (
        not isinstance(binding, dict)
        or not isinstance(binding.get("slot"), int)
        or not 0 <= binding["slot"] < arena.get("slot_count", 0)
        or binding.get("offset_words") != 0
    ):
        raise ValueError(f"AOT value {name!r} has no whole-slot texture arena binding")
    return binding


def emit_model_uasm(plan: dict) -> str:
    """Emit one arena-aware Host for a public-library AOT schedule."""
    if plan.get("runtime") != "udon-onnx" or not isinstance(plan.get("aot"), dict):
        raise ValueError("model source is not a Udon ONNX AOT execution plan")
    calls = plan["aot"].get("calls") or []
    passes = plan.get("passes") or []
    arena = plan.get("texture_arena") or {}
    summary = _texture_arena_summary(plan)
    if [call.get("index") for call in calls] != list(range(len(calls))):
        raise ValueError("AOT model calls must have contiguous topological indexes")
    calls_by_index = {call["index"]: call for call in calls}
    passes_by_call = {index: [] for index in range(len(calls))}
    for device_pass in passes:
        call_index = device_pass.get("call_index")
        if call_index not in calls_by_index:
            raise ValueError("Device pass references an unknown AOT call")
        passes_by_call[call_index].append(device_pass)
    int4_calls = [
        call
        for call in calls
        if call.get("kind") == "tiled_library"
        and call.get("function") == "udonatenWeightInt4packMm"
    ]
    direct_passes = [
        device_pass
        for device_pass in passes
        if calls_by_index[device_pass["call_index"]] not in int4_calls
    ]
    for call in calls:
        if call.get("kind") == "view":
            if passes_by_call[call["index"]]:
                raise ValueError("zero-copy view call unexpectedly owns a Device pass")
        elif call not in int4_calls and not passes_by_call[call["index"]]:
            raise ValueError("public-library AOT call has no Device pass")

    slot_names = [f"arenaSlot{index}" for index in range(summary["slot_count"])]
    max_inputs = 0
    dynamic_inputs = {}
    for device_pass in direct_passes:
        call = calls_by_index[device_pass["call_index"]]
        names = device_pass.get("inputs")
        if not isinstance(names, list):
            names = call.get("inputs") or []
        names = [name for name in names if name in (arena.get("bindings") or {})]
        if not names:
            raise ValueError("Device pass has no texture-arena activation input")
        dynamic_inputs[device_pass["id"]] = names
        max_inputs = max(max_inputs, len(names))
        _model_arena_binding(plan, device_pass.get("output"))
    material_int_properties = sorted(
        {
            property_name
            for device_pass in direct_passes
            for property_name in (device_pass.get("material_ints") or {})
        }
    )
    if any(not name.startswith("_VRC_Arg") for name in material_int_properties):
        raise ValueError("Device pass has an unsupported material integer property")
    material_int_names = {
        name: f"__materialIntName{name.removeprefix('_VRC_Arg')}"
        for name in material_int_properties
    }
    material_float_properties = sorted(
        {
            property_name
            for device_pass in direct_passes
            for property_name in (device_pass.get("material_floats") or {})
        }
    )
    if any(not name.startswith("_VRC_Arg") for name in material_float_properties):
        raise ValueError("Device pass has an unsupported material float property")
    material_float_names = {
        name: f"__materialFloatName{name.removeprefix('_VRC_Arg')}"
        for name in material_float_properties
    }

    lines = [".data_start"]
    lines.extend(f"    .export {name}" for name in slot_names)
    lines.extend(
        [
            "    .export int4Runners",
            "    .export passMaterials",
            "    .export resultState",
            "",
        ]
    )
    lines.extend(f"    {name}: %UnityEngineRenderTexture, null" for name in slot_names)
    lines.extend(
        [
            "    int4Runners: %VRCUdonCommonInterfacesIUdonEventReceiverArray, null",
            "    passMaterials: %UnityEngineMaterialArray, null",
            "    resultState: %UnityEngineRenderTexture, null",
            "    __runner: %VRCUdonCommonInterfacesIUdonEventReceiver, null",
            "    __material: %UnityEngineMaterial, null",
            "    __runnerIndex: %SystemInt32, null",
            "    __passIndex: %SystemInt32, null",
            "    __zero: %SystemInt32, 0",
            "    __one: %SystemInt32, 1",
            '    __runName: %SystemString, "_run"',
            '    __activationStateName: %SystemString, "activationState"',
            '    __resultAName: %SystemString, "resultA"',
            '    __resultBName: %SystemString, "resultB"',
        ]
    )
    for index in range(max_inputs):
        suffix = "" if index == 0 else str(index)
        lines.append(f'    __inputName{index}: %SystemString, "_VRC_Input{suffix}"')
    lines.extend(
        f'    {material_int_names[name]}: %SystemString, "{name}"'
        for name in material_int_properties
    )
    lines.extend(
        f'    {material_float_names[name]}: %SystemString, "{name}"'
        for name in material_float_properties
    )
    lines.extend([".data_end", "", ".code_start", "    .export _start", "    _start:"])
    lines.extend(
        [
            "        JUMP, 0xFFFFFFFC",
            "",
            "    .export _run",
            "    _run:",
            "        PUSH, __zero",
            "        PUSH, __runnerIndex",
            "        COPY",
            "        PUSH, __zero",
            "        PUSH, __passIndex",
            "        COPY",
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

    def set_program_variable(runner: str, variable: str, value: str):
        lines.extend(
            [
                f"        PUSH, {runner}",
                f"        PUSH, {variable}",
                f"        PUSH, {value}",
                '        EXTERN, "VRCUdonCommonInterfacesIUdonEventReceiver.__SetProgramVariable__SystemString_SystemObject__SystemVoid"',
            ]
        )

    def set_int(material: str, property_name: str, value: int):
        value_name = f"__int{value}"
        if not any(line.startswith(f"    {value_name}:") for line in lines):
            raise ValueError("Device pass references an undeclared material integer")
        lines.extend(
            [
                f"        PUSH, {material}",
                f"        PUSH, {property_name}",
                f"        PUSH, {value_name}",
                '        EXTERN, "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"',
            ]
        )

    def set_float(material: str, property_name: str, value_name: str):
        lines.extend(
            [
                f"        PUSH, {material}",
                f"        PUSH, {property_name}",
                f"        PUSH, {value_name}",
                '        EXTERN, "UnityEngineMaterial.__SetFloat__SystemString_SystemSingle__SystemVoid"',
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

    def increment(variable: str):
        lines.extend(
            [
                f"        PUSH, {variable}",
                "        PUSH, __one",
                f"        PUSH, {variable}",
                '        EXTERN, "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"',
            ]
        )

    def get_runner():
        lines.extend(
            [
                "        PUSH, int4Runners",
                "        PUSH, __runnerIndex",
                "        PUSH, __runner",
                '        EXTERN, "VRCUdonCommonInterfacesIUdonEventReceiverArray.__Get__SystemInt32__VRCUdonCommonInterfacesIUdonEventReceiver"',
            ]
        )
        increment("__runnerIndex")

    def get_material():
        lines.extend(
            [
                "        PUSH, passMaterials",
                "        PUSH, __passIndex",
                "        PUSH, __material",
                '        EXTERN, "UnityEngineMaterialArray.__Get__SystemInt32__UnityEngineMaterial"',
            ]
        )
        increment("__passIndex")

    material_int_values = sorted(
        {
            value
            for device_pass in direct_passes
            for value in (device_pass.get("material_ints") or {}).values()
        }
    )
    data_end = lines.index(".data_end")
    lines[data_end:data_end] = [
        f"    __int{value}: %SystemInt32, {value}" for value in material_int_values
    ]
    material_float_values = sorted(
        {
            value
            for device_pass in direct_passes
            for value in (device_pass.get("material_floats") or {}).values()
        }
    )
    float_value_names = {
        value: f"__float{index}" for index, value in enumerate(material_float_values)
    }
    lines[data_end:data_end] = [
        f"    {float_value_names[value]}: %SystemSingle, {_uasm_float(value)}"
        for value in material_float_values
    ]

    for call in calls:
        call_index = call["index"]
        if call.get("kind") == "view":
            continue
        if call in int4_calls:
            get_runner()
            runner = "__runner"
            activation = _model_arena_binding(plan, call["inputs"][0])
            output = _model_arena_binding(plan, call["output"])
            result_name = "__resultAName" if call["tile_grid"]["m_tiles"] % 2 else "__resultBName"
            set_program_variable(runner, "__activationStateName", slot_names[activation["slot"]])
            set_program_variable(runner, result_name, slot_names[output["slot"]])
            lines.extend(
                [
                    f"        PUSH, {runner}",
                    "        PUSH, __runName",
                    '        EXTERN, "VRCUdonCommonInterfacesIUdonEventReceiver.__SendCustomEvent__SystemString__SystemVoid"',
                ]
            )
            continue
        for device_pass in passes_by_call[call_index]:
            get_material()
            material = "__material"
            inputs = dynamic_inputs[device_pass["id"]]
            for input_index, name in enumerate(inputs):
                binding = _model_arena_binding(plan, name)
                set_texture(material, f"__inputName{input_index}", slot_names[binding["slot"]])
            for property_name, value in (device_pass.get("material_ints") or {}).items():
                set_int(material, material_int_names[property_name], value)
            for property_name, value in (
                device_pass.get("material_floats") or {}
            ).items():
                set_float(
                    material,
                    material_float_names[property_name],
                    float_value_names[value],
                )
            output = _model_arena_binding(plan, device_pass["output"])
            source = _model_arena_binding(plan, inputs[0])
            blit(slot_names[source["slot"]], slot_names[output["slot"]], material)
    outputs = plan.get("outputs") or {}
    if len(outputs) != 1:
        raise ValueError("mixed INT4 model Host currently requires exactly one graph output")
    result = _model_arena_binding(plan, next(iter(outputs)))
    lines.extend(
        [
            f"        PUSH, {slot_names[result['slot']]}",
            "        PUSH, resultState",
            "        COPY",
            "        JUMP, 0xFFFFFFFC",
            ".code_end",
            "",
        ]
    )
    return "\n".join(lines)


def emit_int4_model_uasm(plan: dict) -> str:
    """Compatibility alias for the general arena-aware model Host emitter."""
    return emit_model_uasm(plan)


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
        atlases_directory = staging / "atlases"
        atlases_directory.mkdir()
        resources = []
        tiles = []
        atlas_payload = bytearray()
        atlas_index = 0
        atlas_first_page = 0
        for m_tile in range(grid["m_tiles"]):
            for k_tile in range(grid["k_tiles"]):
                page = materialize_int4_compact_page(
                    plan,
                    call_index,
                    m_tile,
                    k_tile,
                    external_data_directory,
                )
                page_index = m_tile * grid["k_tiles"] + k_tile
                location = int4_atlas_location(page_index)
                atlas_row = location["row"]
                atlas_payload.extend(page["data"])
                tiles.append(
                    {
                        "m_tile": m_tile,
                        "k_tile": k_tile,
                        "atlas": location["atlas"],
                        "row": atlas_row,
                        "n": page["n"],
                        "k": page["k"],
                    }
                )
                last_page = page_index + 1 == grid["m_tiles"] * grid["k_tiles"]
                if atlas_row + 1 == INT4_ATLAS_MAX_PAGES or last_page:
                    relative = f"atlases/atlas-{atlas_index:04d}.rgba"
                    payload = bytes(atlas_payload)
                    (staging / relative).write_bytes(payload)
                    resources.append(
                        {
                            "name": f"pageAtlas{atlas_index}",
                            "kind": "bytes",
                            "path": relative,
                            "encoding": "q4-bf16-row80-rgba32",
                            "width": INT4_COMPACT_ATLAS_WIDTH,
                            "height": atlas_row + 1,
                            "first_page": atlas_first_page,
                            "page_count": atlas_row + 1,
                            "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    )
                    atlas_index += 1
                    atlas_first_page = page_index + 1
                    atlas_payload = bytearray()
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
            "page_encoding": {
                "name": "q4-bf16-row80-rgba32",
                "row_bytes": INT4_COMPACT_ROW_BYTES,
                "page_bytes": INT4_COMPACT_PAGE_BYTES,
                "atlas_width": INT4_COMPACT_ATLAS_WIDTH,
                "max_pages_per_atlas": INT4_ATLAS_MAX_PAGES,
            },
            "assembly": "host.uasm",
            "required_shaders": [
                "weight-int4pack-mm-compose.shader",
                "weight-int4pack-mm-scatter.shader",
                "kernel_208.shader",
            ],
            "resources": resources,
            "tiles": tiles,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _publish_artifact(staging, output_directory, "udon-onnx-int4-call")
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _materialize_model(
    plan: dict,
    external_data_directory: Path,
    output_directory: Path,
    *,
    unity_template_directory: Path | None = None,
    model_path: Path | None = None,
    program_name: str = "udon_onnx_model",
    legacy_int4_manifest: bool = False,
) -> dict:
    """Bundle a complete arena-aware AOT model schedule."""
    if plan.get("runtime") != "udon-onnx" or not isinstance(plan.get("aot"), dict):
        raise ValueError("model source is not a Udon ONNX AOT execution plan")
    calls = plan["aot"].get("calls") or []
    int4_calls = [
        call
        for call in calls
        if call.get("kind") == "tiled_library"
        and call.get("function") == "udonatenWeightInt4packMm"
    ]
    int4_call_indexes = {call["index"] for call in int4_calls}
    direct_passes = [
        device_pass
        for device_pass in plan.get("passes") or []
        if device_pass.get("call_index") not in int4_call_indexes
    ]
    missing_shader_passes = [
        device_pass["id"]
        for device_pass in direct_passes
        if not isinstance(device_pass.get("shader"), str)
        or not device_pass["shader"]
    ]
    if legacy_int4_manifest and not int4_calls:
        raise ValueError("execution plan contains no tiled ATen INT4 calls")
    if not legacy_int4_manifest and missing_shader_passes:
        raise ValueError(
            "Udon ONNX model has Device passes without public-library shaders: "
            + ", ".join(map(str, missing_shader_passes))
        )
    external_data_directory = external_data_directory.resolve()
    output_directory = output_directory.resolve()
    try:
        external_data_directory.relative_to(output_directory)
    except ValueError:
        pass
    else:
        raise ValueError("INT4 model output cannot contain its external data directory")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output_directory.name + "-", dir=output_directory.parent)
    )
    try:
        calls_directory = staging / "calls"
        calls_directory.mkdir()
        producers = {}
        for call in calls:
            output = call.get("output")
            if not isinstance(output, str):
                continue
            if output in producers:
                raise ValueError(f"AOT schedule has duplicate producer for {output!r}")
            producers[output] = call.get("index")
        artifacts = []
        for call in int4_calls:
            call_index = call.get("index")
            if not isinstance(call_index, int):
                raise ValueError("INT4 model call index must be an integer")
            relative = f"calls/call-{call_index:04d}"
            child = materialize_int4_call(
                plan,
                call_index,
                external_data_directory,
                staging / relative,
            )
            activation = call["inputs"][0]
            activation_binding = _model_arena_binding(plan, activation)
            output_binding = _model_arena_binding(plan, call["output"])
            slots = plan["texture_arena"]["slots"]
            activation_width = slots[activation_binding["slot"]]["width"]
            result_width = slots[output_binding["slot"]]["width"]
            child["model_arena_binding"] = {
                "activation_slot": activation_binding["slot"],
                "activation_storage_width": activation_width,
                "result_slot": output_binding["slot"],
                "result_storage_width": result_width,
            }
            child_directory = staging / relative
            (child_directory / "host.uasm").write_text(
                emit_int4_call_uasm(
                    plan,
                    call_index,
                    activation_storage_width=activation_width,
                    result_storage_width=result_width,
                ),
                encoding="utf-8",
            )
            (child_directory / "manifest.json").write_text(
                json.dumps(child, indent=2) + "\n", encoding="utf-8"
            )
            producer = producers.get(activation)
            if producer is not None and (
                not isinstance(producer, int) or producer >= call_index
            ):
                raise ValueError("INT4 activation producer is not topologically earlier")
            artifacts.append(
                {
                    "call_index": call_index,
                    "path": relative,
                    "input": activation,
                    "input_producer_call": producer,
                    "output": call["output"],
                    "arena_binding": dict(child["model_arena_binding"]),
                    "tile_grid": dict(child["tile_grid"]),
                    "atlases": len(child["resources"]),
                    "pages": len(child["tiles"]),
                    "atlas_bytes": sum(
                        resource["bytes"] for resource in child["resources"]
                    ),
                }
            )
        static_inputs = materialize_static_inputs(
            plan, external_data_directory, staging, model_path=model_path
        )
        serialized_plan = json.dumps(plan, indent=2) + "\n"
        plan_name = (
            "execution-plan.json.bytes"
            if unity_template_directory is not None
            else "execution-plan.json"
        )
        (staging / plan_name).write_text(serialized_plan, encoding="utf-8")
        (staging / "host.uasm").write_text(emit_model_uasm(plan), encoding="utf-8")
        scheduled_call_indexes = {
            call["index"] for call in calls if call.get("kind") == "view"
        }
        scheduled_call_indexes.update(call["index"] for call in int4_calls)
        scheduled_call_indexes.update(
            device_pass["call_index"] for device_pass in plan.get("passes") or []
        )
        unmaterialized_calls = [
            {
                "index": call.get("index"),
                "kind": call.get("kind"),
                "function": call.get("function"),
                "output": call.get("output"),
            }
            for call in calls
            if call.get("index") not in scheduled_call_indexes
        ]
        legacy_unmaterialized_calls = [
            {
                "index": call.get("index"),
                "kind": call.get("kind"),
                "function": call.get("function"),
                "output": call.get("output"),
            }
            for call in calls
            if call not in int4_calls
        ]
        artifact_stage = (
            "udon-onnx-int4-model" if legacy_int4_manifest else "udon-onnx-model"
        )
        manifest = {
            "schema_version": 1,
            "artifact_stage": artifact_stage,
            "runtime": "udon-onnx",
            "graph": plan.get("graph"),
            "execution_provider": plan.get("execution_provider"),
            "execution_plan": plan_name,
            "assembly": "host.uasm",
            "execution_plan_sha256": hashlib.sha256(
                serialized_plan.encode("utf-8")
            ).hexdigest(),
            "aot_call_count": len(calls),
            "int4_call_count": len(artifacts),
            "int4_atlas_count": sum(item["atlases"] for item in artifacts),
            "int4_page_count": sum(item["pages"] for item in artifacts),
            "int4_atlas_bytes": sum(item["atlas_bytes"] for item in artifacts),
            "texture_arena": _texture_arena_summary(plan),
            "host_schedule": _model_host_summary(plan),
            "static_inputs": static_inputs,
            "complete_graph": (
                len(int4_calls) == len(calls)
                if legacy_int4_manifest
                else not unmaterialized_calls
            ),
            "call_artifacts": artifacts,
            "unmaterialized_calls": (
                legacy_unmaterialized_calls
                if legacy_int4_manifest
                else unmaterialized_calls
            ),
        }
        publish_stage = artifact_stage
        if unity_template_directory is not None:
            if model_path is None:
                raise ValueError("Unity ONNX bundle requires model_path")
            unity_template_directory = unity_template_directory.resolve()
            model_path = model_path.resolve()
            if missing_shader_passes:
                raise ValueError(
                    "Unity ONNX bundle has Device passes without public-library "
                    "shaders: " + ", ".join(map(str, missing_shader_passes))
                )
            direct_kernel_passes = {}
            for device_pass in direct_passes:
                direct_kernel_passes.setdefault(device_pass["shader"], device_pass)
            required = {
                "display.shader": unity_template_directory / "display.shader",
            }
            if int4_calls:
                required.update(
                    {
                        "compose.shader": unity_template_directory / "compose.shader",
                        "scatter.shader": unity_template_directory / "scatter.shader",
                        "kernels/kernel_208.shader": unity_template_directory
                        / "kernels/kernel_208.shader",
                    }
                )
            for shader in sorted(direct_kernel_passes):
                required[f"kernels/{shader}"] = (
                    unity_template_directory / "kernels" / shader
                )
            for relative, source in required.items():
                if not source.is_file():
                    raise ValueError(f"Unity ONNX template is missing {relative}")
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            if not model_path.is_file():
                raise ValueError("Unity ONNX model file does not exist")
            shutil.copyfile(model_path, staging / "model.onnx.bytes")
            publish_stage = "unity-udon-onnx-model"
            manifest.update(
                {
                    "compiler_version": "0.1.0-dev",
                    "runtime_abi_version": 1,
                    "artifact_stage": publish_stage,
                    "ready_for_unity": True,
                    "sdk_compatibility": {
                        "sdk_version": "3.10.5",
                        "extern_db_hash": None,
                    },
                    "target": {
                        "host_pointer_bits": 32,
                        "device_pointer_bits": 32,
                        "endianness": "little",
                        "platform": "VRChat-PC",
                    },
                    "program": {
                        "name": program_name,
                        "assembly": "host.uasm",
                        "display_shader": "display.shader",
                    },
                    "device": {
                        "kernels": (
                            [
                                {
                                    "id": 208,
                                    "name": "udonaten_weight_int4pack_mm",
                                    "shader": "kernels/kernel_208.shader",
                                }
                            ]
                            if int4_calls
                            else []
                        )
                        + [
                            {
                                "id": int(
                                    Path(shader).stem.removeprefix("kernel_")
                                ),
                                "name": device_pass["kernel"],
                                "shader": f"kernels/{shader}",
                            }
                            for shader, device_pass in direct_kernel_passes.items()
                        ]
                    },
                    "resources": [
                        {
                            "name": "onnx_model",
                            "kind": "bytes",
                            "path": "model.onnx.bytes",
                        },
                        {
                            "name": "onnx_execution_plan",
                            "kind": "bytes",
                            "path": plan_name,
                        },
                    ]
                    + (
                        [
                            {
                                "name": "page_compose",
                                "kind": "shader",
                                "path": "compose.shader",
                            },
                            {
                                "name": "result_scatter",
                                "kind": "shader",
                                "path": "scatter.shader",
                            },
                        ]
                        if int4_calls
                        else []
                    ),
                }
            )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _publish_artifact(staging, output_directory, publish_stage)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def materialize_model(
    plan: dict,
    external_data_directory: Path,
    output_directory: Path,
    *,
    unity_template_directory: Path | None = None,
    model_path: Path | None = None,
    program_name: str = "udon_onnx_model",
) -> dict:
    """Bundle any complete arena-aware Udon ONNX AOT schedule."""
    return _materialize_model(
        plan,
        external_data_directory,
        output_directory,
        unity_template_directory=unity_template_directory,
        model_path=model_path,
        program_name=program_name,
    )


def materialize_int4_model(
    plan: dict,
    external_data_directory: Path,
    output_directory: Path,
    *,
    unity_template_directory: Path | None = None,
    model_path: Path | None = None,
    program_name: str = "udon_onnx_int4_model",
) -> dict:
    """Compatibility wrapper for the historical INT4 model artifact contract."""
    return _materialize_model(
        plan,
        external_data_directory,
        output_directory,
        unity_template_directory=unity_template_directory,
        model_path=model_path,
        program_name=program_name,
        legacy_int4_manifest=True,
    )


def _texture_arena_summary(plan: dict) -> dict:
    arena = plan.get("texture_arena")
    if not isinstance(arena, dict) or arena.get("abi") != "linear-rgba32-words":
        raise ValueError("execution plan has no supported Device texture arena")
    slots = arena.get("slots")
    if not isinstance(slots, list) or arena.get("slot_count") != len(slots):
        raise ValueError("execution plan texture arena slot count is invalid")
    summary_slots = []
    total_words = 0
    for index, slot in enumerate(slots):
        if (
            not isinstance(slot, dict)
            or slot.get("index") != index
            or not isinstance(slot.get("capacity_words"), int)
            or slot["capacity_words"] <= 0
            or slot.get("width") != min(8192, slot["capacity_words"])
            or slot.get("height") != math.ceil(slot["capacity_words"] / 8192)
        ):
            raise ValueError("execution plan texture arena slot layout is invalid")
        total_words += slot["capacity_words"]
        summary_slots.append(
            {
                "index": index,
                "capacity_words": slot["capacity_words"],
                "width": slot["width"],
                "height": slot["height"],
            }
        )
    if (
        arena.get("total_words") != total_words
        or arena.get("total_bytes") != total_words * 4
    ):
        raise ValueError("execution plan texture arena capacity is inconsistent")
    return {
        "abi": arena["abi"],
        "slot_count": len(slots),
        "total_words": total_words,
        "total_bytes": total_words * 4,
        "slots": summary_slots,
    }


def _model_host_summary(plan: dict) -> dict:
    calls = (plan.get("aot") or {}).get("calls") or []
    int4_calls = [
        call
        for call in calls
        if call.get("kind") == "tiled_library"
        and call.get("function") == "udonatenWeightInt4packMm"
    ]
    int4_indexes = {call.get("index") for call in int4_calls}
    direct_passes = [
        device_pass
        for device_pass in plan.get("passes") or []
        if device_pass.get("call_index") not in int4_indexes
    ]
    static_initializers = {
        item["initializer"]
        for device_pass in direct_passes
        for item in device_pass.get("static_inputs") or []
    }
    return {
        "assembly": "host.uasm",
        "arena_slots": (plan.get("texture_arena") or {}).get("slot_count"),
        "int4_runners": len(int4_calls),
        "direct_device_passes": len(direct_passes),
        "static_initializer_textures": len(static_initializers),
        "zero_copy_views": sum(call.get("kind") == "view" for call in calls),
    }


def _model_static_inputs(plan: dict) -> list[dict]:
    uses = {}
    for device_pass in plan.get("passes") or []:
        for item in device_pass.get("static_inputs") or []:
            name = item.get("initializer")
            property_name = item.get("property")
            if not isinstance(name, str) or not isinstance(property_name, str):
                raise ValueError("Device pass static input contract is invalid")
            uses.setdefault(name, set()).add(property_name)
    initializers = plan.get("initializers") or {}
    result = []
    for name in sorted(uses):
        initializer = initializers.get(name)
        if not isinstance(initializer, dict):
            raise ValueError(f"static input {name!r} is not an initializer")
        words = math.prod(initializer.get("shape") or [])
        result.append(
            {
                "name": name,
                "properties": sorted(uses[name]),
                "shape": initializer.get("shape"),
                "data_type": initializer.get("data_type"),
                "source_bytes": initializer.get("bytes"),
                "storage": initializer.get("storage"),
                "external": initializer.get("external"),
                "texture": {
                    "abi": "linear-rgba32-words",
                    "words": words,
                    "width": min(8192, words),
                    "height": math.ceil(words / 8192),
                },
            }
        )
    return result


def materialize_static_inputs(
    plan: dict,
    external_data_directory: Path,
    output_directory: Path,
    *,
    model_path: Path | None = None,
) -> list[dict]:
    """Decode ATen constants into the RGBA32 word-texture payload."""
    output_directory.mkdir(parents=True, exist_ok=True)
    embedded = None
    artifacts = []
    for index, item in enumerate(_model_static_inputs(plan)):
        words = item["texture"]["words"]
        if item["storage"] == "external":
            raw = _read_external(
                external_data_directory, item["external"], 0, item["source_bytes"]
            )
        elif item["storage"] == "embedded":
            if model_path is None or not model_path.is_file():
                raise ValueError(
                    "embedded static Device inputs require the original ONNX model"
                )
            if embedded is None:
                from udon_onnx import parse_model

                embedded = parse_model(model_path.read_bytes())["graph"]["initializers"]
            tensor = embedded.get(item["name"])
            if not isinstance(tensor, dict) or tensor.get("data_location") != 0:
                raise ValueError(f"embedded static input {item['name']!r} is missing")
            raw = tensor["raw"]
            expected_sha256 = (plan.get("initializers") or {}).get(item["name"], {}).get(
                "sha256"
            )
            if (
                len(raw) != item["source_bytes"]
                or hashlib.sha256(raw).hexdigest() != expected_sha256
            ):
                raise ValueError(
                    f"embedded static input {item['name']!r} does not match the plan"
                )
        else:
            raise ValueError("static Device input storage is unsupported")
        if item["data_type"] == "bfloat16":
            if len(raw) != words * 2:
                raise ValueError("bfloat16 static input byte size is inconsistent")
            values = struct.unpack(f"<{words}H", raw)
            payload = struct.pack(f"<{words}I", *(value << 16 for value in values))
        elif item["data_type"] == "float32":
            if len(raw) != words * 4:
                raise ValueError("float32 static input byte size is inconsistent")
            payload = raw
        else:
            raise ValueError(
                f"static Device input type {item['data_type']!r} is unsupported"
            )
        relative = f"static-inputs/input-{index:04d}.bytes"
        destination = output_directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        texture_bytes = item["texture"]["width"] * item["texture"]["height"] * 4
        payload = payload.ljust(texture_bytes, b"\0")
        destination.write_bytes(payload)
        artifacts.append(
            {
                **item,
                "resource": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return artifacts


def plan_int4_model_bundle(plan: dict) -> dict:
    """Plan call artifacts and compact-atlas capacity without reading weights."""
    if plan.get("runtime") != "udon-onnx" or not isinstance(plan.get("aot"), dict):
        raise ValueError("model source is not a Udon ONNX AOT execution plan")
    calls = plan["aot"].get("calls") or []
    producers = {}
    for call in calls:
        output = call.get("output")
        if not isinstance(output, str):
            continue
        if output in producers:
            raise ValueError(f"AOT schedule has duplicate producer for {output!r}")
        producers[output] = call.get("index")
    artifacts = []
    materialized_indexes = set()
    for call in calls:
        if not (
            call.get("kind") == "tiled_library"
            and call.get("function") == "udonatenWeightInt4packMm"
        ):
            continue
        call_index = call.get("index")
        if not isinstance(call_index, int) or call_index in materialized_indexes:
            raise ValueError("INT4 model call indexes must be unique integers")
        contract = _int4_call_contract(plan, call_index)
        activation = call["inputs"][0]
        producer = producers.get(activation)
        if producer is not None and (
            not isinstance(producer, int) or producer >= call_index
        ):
            raise ValueError("INT4 activation producer is not topologically earlier")
        pages = contract["grid"]["m_tiles"] * contract["grid"]["k_tiles"]
        artifacts.append(
            {
                "call_index": call_index,
                "input": activation,
                "input_producer_call": producer,
                "output": call["output"],
                "tile_grid": dict(contract["grid"]),
                "atlases": math.ceil(pages / INT4_ATLAS_MAX_PAGES),
                "pages": pages,
                "atlas_bytes": pages * INT4_COMPACT_PAGE_BYTES,
            }
        )
        materialized_indexes.add(call_index)
    if not artifacts:
        raise ValueError("execution plan contains no tiled ATen INT4 calls")
    return {
        "aot_call_count": len(calls),
        "int4_call_count": len(artifacts),
        "int4_atlas_count": sum(item["atlases"] for item in artifacts),
        "int4_page_count": sum(item["pages"] for item in artifacts),
        "int4_atlas_bytes": sum(item["atlas_bytes"] for item in artifacts),
        "texture_arena": _texture_arena_summary(plan),
        "host_schedule": _model_host_summary(plan),
        "static_inputs": _model_static_inputs(plan),
        "complete_graph": len(artifacts) == len(calls),
        "call_artifacts": artifacts,
        "unmaterialized_calls": [
            {
                "index": call.get("index"),
                "kind": call.get("kind"),
                "function": call.get("function"),
                "output": call.get("output"),
            }
            for call in calls
            if call.get("index") not in materialized_indexes
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("external_data_directory", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--call", type=int)
    parser.add_argument("--m-tile", type=int)
    parser.add_argument("--k-tile", type=int)
    parser.add_argument(
        "--all",
        action="store_true",
        help="materialize every M/K page plus the static Udon Host dispatcher",
    )
    parser.add_argument(
        "--all-calls",
        action="store_true",
        help="bundle all tiled ATen INT4 calls plus the complete AOT schedule",
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="bundle the complete general Udon ONNX AOT schedule",
    )
    parser.add_argument(
        "--unity-template-directory",
        type=Path,
        help="copy Unity display and planned Device shaders from this directory",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="original ONNX model required with --unity-template-directory",
    )
    parser.add_argument(
        "--program-name",
        default="udon_onnx_model",
        help="Unity artifact program and generated asset directory name",
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
    if args.model:
        if (
            args.call is not None
            or args.all
            or args.all_calls
            or args.m_tile is not None
            or args.k_tile is not None
            or args.activation_f32
            or args.accumulator_f32
        ):
            parser.error("--model cannot be combined with call or page options")
        manifest = materialize_model(
            plan,
            args.external_data_directory,
            args.output,
            unity_template_directory=args.unity_template_directory,
            model_path=args.model_path,
            program_name=args.program_name,
        )
        print(
            f"udon_onnx_model={args.output} calls={manifest['aot_call_count']} "
            f"complete={manifest['complete_graph']}"
        )
        return
    if args.all_calls:
        if (
            args.call is not None
            or args.all
            or args.m_tile is not None
            or args.k_tile is not None
            or args.activation_f32
            or args.accumulator_f32
        ):
            parser.error("--all-calls cannot be combined with call or page options")
        manifest = materialize_int4_model(
            plan,
            args.external_data_directory,
            args.output,
        )
        print(
            f"udon_onnx_int4_model={args.output} "
            f"calls={manifest['int4_call_count']} complete={manifest['complete_graph']}"
        )
        return
    if args.unity_template_directory or args.model_path:
        parser.error("Unity model options require --model")
    if args.all:
        if args.call is None:
            parser.error("--all requires --call")
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
            f"pages={len(manifest['tiles'])} atlases={len(manifest['resources'])}"
        )
        return
    if args.m_tile is None or args.k_tile is None:
        parser.error("single-page mode requires --m-tile and --k-tile")
    if args.call is None:
        parser.error("single-page mode requires --call")
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
