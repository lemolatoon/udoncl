"""Materialize bounded UdonATen Device pages from ONNX external weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import struct


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
    ):
        raise ValueError("execution-plan call is not a supported ATen INT4 tile")
    grid = call.get("tile_grid") or {}
    paging = call.get("paging") or {}
    if (
        grid.get("m") != 64
        or grid.get("k") != 128
        or paging.get("state_words") != 8192
        or paging.get("input_offset_words") != 6144
        or paging.get("result_offset_words") != 7168
        or paging.get("row_stride_words") != 72
        or paging.get("accumulate_argument") != 5
    ):
        raise ValueError("execution-plan INT4 paging ABI is unsupported")
    if (
        not isinstance(m_tile, int)
        or not isinstance(k_tile, int)
        or not 0 <= m_tile < grid.get("m_tiles", 0)
        or not 0 <= k_tile < grid.get("k_tiles", 0)
    ):
        raise ValueError("INT4 tile coordinates are outside the planned grid")

    attributes = call.get("attributes") or {}
    output_features = (attributes.get("output_features") or {}).get("i")
    input_features = (attributes.get("input_features") or {}).get("i")
    if (
        not isinstance(output_features, int)
        or not isinstance(input_features, int)
        or output_features <= 0
        or input_features <= 0
        or input_features % 32
    ):
        raise ValueError("INT4 call has invalid feature dimensions")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("external_data_directory", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--call", type=int, required=True)
    parser.add_argument("--m-tile", type=int, required=True)
    parser.add_argument("--k-tile", type=int, required=True)
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
