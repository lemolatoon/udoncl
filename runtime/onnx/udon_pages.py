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


INT4_COMPACT_ROW_BYTES = 80
INT4_COMPACT_PAGE_BYTES = 64 * INT4_COMPACT_ROW_BYTES
INT4_COMPACT_ATLAS_WIDTH = INT4_COMPACT_PAGE_BYTES // 4
INT4_ATLAS_MAX_PAGES = 8192


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


def emit_int4_call_uasm(plan: dict, call_index: int) -> str:
    """Emit a bounded counter-based Udon Host dispatcher for one INT4 call."""
    contract = _int4_call_contract(plan, call_index)
    grid = contract["grid"]
    output_features = contract["output_features"]
    input_features = contract["input_features"]
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
            f"    __activationWidth: %SystemInt32, {linear_texture_layout(input_features)['width']}",
            f"    __outputWidth: %SystemInt32, {linear_texture_layout(output_features)['width']}",
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
        lines.extend([f"        PUSH, {destination}", f"        PUSH, {source}", "        COPY"])

    def binary(destination: str, left: str, right: str, operation: str):
        lines.extend(
            [
                f"        PUSH, {destination}",
                f"        PUSH, {left}",
                f"        PUSH, {right}",
                f'        EXTERN, "SystemInt32.__op_{operation}__SystemInt32_SystemInt32__SystemInt32"',
            ]
        )

    def compare(left: str, right: str, operation: str):
        lines.extend(
            [
                "        PUSH, __condition",
                f"        PUSH, {left}",
                f"        PUSH, {right}",
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


def materialize_int4_model(
    plan: dict,
    external_data_directory: Path,
    output_directory: Path,
) -> dict:
    """Bundle every ATen INT4 call with the complete AOT model schedule."""
    if plan.get("runtime") != "udon-onnx" or not isinstance(plan.get("aot"), dict):
        raise ValueError("model source is not a Udon ONNX AOT execution plan")
    calls = plan["aot"].get("calls") or []
    int4_calls = [
        call
        for call in calls
        if call.get("kind") == "tiled_library"
        and call.get("function") == "udonatenWeightInt4packMm"
    ]
    if not int4_calls:
        raise ValueError("execution plan contains no tiled ATen INT4 calls")
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
                    "tile_grid": dict(child["tile_grid"]),
                    "atlases": len(child["resources"]),
                    "pages": len(child["tiles"]),
                    "atlas_bytes": sum(
                        resource["bytes"] for resource in child["resources"]
                    ),
                }
            )
        serialized_plan = json.dumps(plan, indent=2) + "\n"
        (staging / "execution-plan.json").write_text(serialized_plan, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "artifact_stage": "udon-onnx-int4-model",
            "runtime": "udon-onnx",
            "graph": plan.get("graph"),
            "execution_provider": plan.get("execution_provider"),
            "execution_plan": "execution-plan.json",
            "execution_plan_sha256": hashlib.sha256(
                serialized_plan.encode("utf-8")
            ).hexdigest(),
            "aot_call_count": len(calls),
            "int4_call_count": len(artifacts),
            "int4_atlas_count": sum(item["atlases"] for item in artifacts),
            "int4_page_count": sum(item["pages"] for item in artifacts),
            "int4_atlas_bytes": sum(item["atlas_bytes"] for item in artifacts),
            "texture_arena": _texture_arena_summary(plan),
            "complete_graph": len(int4_calls) == len(calls),
            "call_artifacts": artifacts,
            "unmaterialized_calls": [
                {
                    "index": call.get("index"),
                    "kind": call.get("kind"),
                    "function": call.get("function"),
                    "output": call.get("output"),
                }
                for call in calls
                if call not in int4_calls
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _publish_artifact(staging, output_directory, "udon-onnx-int4-model")
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


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
