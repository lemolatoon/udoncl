"""Pack safetensors checkpoints into Udon ONNX external-weight files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile

import numpy as np


SOURCE_DTYPES = {
    "F32": (np.dtype("<f4"), 4),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
}


class SafeTensorFile:
    def __init__(self, path: Path):
        self.path = path.resolve()
        with self.path.open("rb") as stream:
            encoded_length = stream.read(8)
            if len(encoded_length) != 8:
                raise ValueError(f"truncated safetensors header in {self.path}")
            header_length = struct.unpack("<Q", encoded_length)[0]
            if header_length == 0 or header_length > self.path.stat().st_size - 8:
                raise ValueError(f"invalid safetensors header length in {self.path}")
            try:
                self.header = json.loads(stream.read(header_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid safetensors JSON header in {self.path}") from error
        self.data_offset = 8 + header_length
        self.data_bytes = self.path.stat().st_size - self.data_offset
        self.tensors = {}
        intervals = []
        for name, tensor in self.header.items():
            if name == "__metadata__":
                continue
            if not isinstance(tensor, dict) or tensor.get("dtype") not in SOURCE_DTYPES:
                raise ValueError(f"unsupported safetensors entry {name!r}")
            shape = tensor.get("shape")
            offsets = tensor.get("data_offsets")
            if (
                not isinstance(shape, list)
                or not shape
                or any(not isinstance(value, int) or value <= 0 for value in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(value, int) for value in offsets)
            ):
                raise ValueError(f"invalid safetensors metadata for {name!r}")
            start, end = offsets
            item_size = SOURCE_DTYPES[tensor["dtype"]][1]
            elements = int(np.prod(shape, dtype=np.int64))
            if start < 0 or end < start or end > self.data_bytes or end - start != elements * item_size:
                raise ValueError(f"invalid safetensors byte range for {name!r}")
            intervals.append((start, end, name))
            self.tensors[name] = {
                "dtype": tensor["dtype"],
                "shape": shape,
                "elements": elements,
                "offset": self.data_offset + start,
                "bytes": end - start,
            }
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            if left[1] > right[0]:
                raise ValueError(
                    f"overlapping safetensors entries {left[2]!r} and {right[2]!r}"
                )

    def array(self, name: str, start: int = 0, end: int | None = None) -> np.ndarray:
        tensor = self.tensors[name]
        end = tensor["elements"] if end is None else end
        if start < 0 or end < start or end > tensor["elements"]:
            raise ValueError(f"invalid tensor slice for {name!r}")
        dtype = SOURCE_DTYPES[tensor["dtype"]][0]
        raw = np.memmap(
            self.path,
            mode="r",
            dtype=dtype,
            offset=tensor["offset"],
            shape=(tensor["elements"],),
        )
        raw = raw[start:end]
        if tensor["dtype"] == "BF16":
            bits = np.asarray(raw, dtype=np.uint32) << np.uint32(16)
            return bits.view(np.float32)
        return np.asarray(raw, dtype=np.float32)


def _checkpoint_files(checkpoint: Path) -> tuple[dict[str, str], dict[str, SafeTensorFile]]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index has no weight_map")
    else:
        candidates = list(checkpoint.glob("*.safetensors"))
        if len(candidates) != 1:
            raise ValueError("checkpoint needs one safetensors file or an index")
        reader = SafeTensorFile(candidates[0])
        weight_map = {name: candidates[0].name for name in reader.tensors}
    readers = {}
    for relative in sorted(set(weight_map.values())):
        if not isinstance(relative, str):
            raise ValueError("safetensors index file names must be strings")
        path = (checkpoint / relative).resolve()
        try:
            path.relative_to(checkpoint)
        except ValueError as error:
            raise ValueError("safetensors index path escapes checkpoint directory") from error
        readers[relative] = SafeTensorFile(path)
    for name, relative in weight_map.items():
        if name not in readers[relative].tensors:
            raise ValueError(f"safetensors index points {name!r} to the wrong shard")
    return weight_map, readers


def _q4_row_chunks(
    reader: SafeTensorFile,
    name: str,
    block_values: int = 32,
    rows_per_chunk: int = 128,
):
    shape = reader.tensors[name]["shape"]
    if len(shape) != 2 or shape[1] % block_values or shape[1] % 2:
        raise ValueError("ATen INT4 weights must be [N,K] with K divisible by 32")
    groups_per_row = shape[1] // block_values
    for row_start in range(0, shape[0], rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, shape[0])
        values = np.asarray(
            reader.array(name, row_start * shape[1], row_end * shape[1]),
            dtype=np.float32,
        ).reshape(row_end - row_start, groups_per_row, block_values)
        maximum = np.max(np.abs(values), axis=2)
        scales = np.where(maximum == 0.0, 1.0, maximum / 7.0).astype(np.float32)
        quantized = np.clip(
            np.rint(values / scales[:, :, None]), -8, 7
        ).astype(np.int8)
        encoded = np.bitwise_and(quantized.astype(np.int16), 15).astype(np.uint8)
        packed = encoded[:, :, 0::2] | (encoded[:, :, 1::2] << np.uint8(4))
        # ATen CUDA indexes qScaleAndZeros as [K/group_size, N, 2] and computes
        # dequantized = signed_int4 * scale + zero.
        scale_zero = np.zeros(
            (groups_per_row, row_end - row_start, 2), dtype=np.float32
        )
        scale_zero[:, :, 0] = scales.T
        yield row_start, row_end, packed, scale_zero


def _bfloat16_bytes(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2").tobytes()


def pack_checkpoint(checkpoint: Path, target_index: Path, output: Path) -> dict:
    layout = json.loads(target_index.read_text(encoding="utf-8"))
    tensors = layout.get("tensors")
    total_bytes = layout.get("storage_bytes_with_alignment")
    if not isinstance(tensors, list) or not tensors or not isinstance(total_bytes, int):
        raise ValueError("target weight index is malformed")
    weight_map, readers = _checkpoint_files(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = []
    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.truncate(total_bytes)
            previous_end = 0
            for target in tensors:
                name = target.get("name")
                source_name = target.get("source_name", name)
                if source_name not in weight_map:
                    raise ValueError(f"checkpoint is missing tensor {source_name!r}")
                reader = readers[weight_map[source_name]]
                source = reader.tensors[source_name]
                if source["shape"] != target.get("logical_shape"):
                    raise ValueError(
                        f"shape mismatch for {name!r}: {source['shape']} != {target.get('logical_shape')}"
                    )
                offset = target.get("offset")
                length = target.get("length")
                if (
                    not isinstance(offset, int)
                    or not isinstance(length, int)
                    or offset < previous_end
                    or offset + length > total_bytes
                ):
                    raise ValueError(f"invalid target range for {name!r}")
                stream.seek(offset)
                digest = hashlib.sha256()
                written = 0
                if target.get("format") == "q4_int4pack":
                    chunks = (
                        packed.tobytes()
                        for _, _, packed, _ in _q4_row_chunks(reader, source_name)
                    )
                elif target.get("format") == "q4_scale_zero_bf16":
                    rows, columns = source["shape"]
                    groups_per_row = columns // 32
                    for row_start, row_end, _, scale_zero in _q4_row_chunks(
                        reader, source_name
                    ):
                        for group in range(groups_per_row):
                            stream.seek(offset + (group * rows + row_start) * 4)
                            stream.write(_bfloat16_bytes(scale_zero[group]))
                    stream.seek(offset)
                    remaining = length
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError(
                                f"failed to read packed qparams for {name!r}"
                            )
                        digest.update(chunk)
                        remaining -= len(chunk)
                    written = length
                    chunks = ()
                elif target.get("format") == "bf16":
                    chunks = [_bfloat16_bytes(reader.array(source_name))]
                elif target.get("format") == "float32":
                    chunks = [
                        np.asarray(reader.array(source_name), dtype="<f4").tobytes()
                    ]
                else:
                    raise ValueError(f"unsupported target format for {name!r}")
                for chunk in chunks:
                    stream.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                if written != length:
                    raise ValueError(f"packed length mismatch for {name!r}: {written} != {length}")
                previous_end = offset + length
                report.append({"name": name, "bytes": written, "sha256": digest.hexdigest()})
            stream.flush()
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(output)
    manifest = {
        "schema_version": 1,
        "source_checkpoint": str(checkpoint.resolve()),
        "target_index": str(target_index.resolve()),
        "output": output.name,
        "bytes": output.stat().st_size,
        "tensors": report,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("target_index", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = pack_checkpoint(args.checkpoint, args.target_index, args.output)
    print(
        f"udon_onnx_weights={manifest['bytes']} tensors={len(manifest['tensors'])} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
