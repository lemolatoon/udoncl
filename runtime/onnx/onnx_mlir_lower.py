"""Lower canonical ONNX MLIR directly to public Udon library Host calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from onnx_mlir_graph import Graph, Operator, parse
from udon_onnx import PROVIDERS, emit_c_host, parse_model, validate_external_data


def _static_shape(type_, context: str) -> list[int]:
    if type_ is None or type_.shape is None or any(item is None for item in type_.shape):
        raise ValueError(f"{context} requires a static tensor shape after specialization")
    if type_.element_type != "f32":
        raise ValueError(f"{context} has unsupported element type {type_.element_type!r}")
    return list(type_.shape)


def _attribute(value: object) -> dict:
    if isinstance(value, bool) or isinstance(value, int):
        return {"i": int(value)}
    if isinstance(value, float):
        return {"f": value}
    if isinstance(value, tuple) and all(isinstance(item, int) for item in value):
        return {"ints": list(value)}
    if isinstance(value, str):
        return {"s": value}
    raise ValueError(f"unsupported canonical ONNX attribute value {value!r}")


def _value_name(operator: Operator, index: int, used: set[str]) -> str:
    stem = operator.node_name or operator.op_type.lower()
    stem = "".join(character if character.isalnum() or character in "._" else "_" for character in stem)
    candidate = stem if len(operator.outputs) == 1 else f"{stem}.{index}"
    suffix = 1
    unique = candidate
    while unique in used:
        suffix += 1
        unique = f"{candidate}.{suffix}"
    used.add(unique)
    return unique


def host_graph(graph: Graph, model: dict, dimension_bounds=None) -> dict:
    """Build the ephemeral graph consumed by Host C emission.

    The returned object is never serialized and contains no Device-pass
    schedule; each canonical ONNX operation maps directly to one public
    UdonBLAS, UdonDNN, or UdonATen call site.
    """
    dimension_bounds = dimension_bounds or {}
    model_graph = model["graph"]
    model_initializers = model_graph["initializers"]
    ssa_names: dict[str, str] = {}
    values: dict[str, list[int]] = {}
    value_types: dict[str, str] = {}
    used: set[str] = set()
    for value in (*graph.inputs, *graph.initializers):
        if value.ssa_name is None:
            raise ValueError(f"canonical ONNX value {value.name!r} has no SSA identity")
        if value.name in used:
            raise ValueError(f"duplicate canonical ONNX value name {value.name!r}")
        used.add(value.name)
        ssa_names[value.ssa_name] = value.name
        if (
            value in graph.inputs
            and value.type is not None
            and value.type.shape is not None
            and any(dimension is None for dimension in value.type.shape)
        ):
            model_shape = model_graph.get("inputs", {}).get(value.name, {}).get("shape", ())
            specialized = []
            for axis, dimension in enumerate(value.type.shape):
                if dimension is not None:
                    specialized.append(dimension)
                    continue
                symbol = model_shape[axis] if axis < len(model_shape) else None
                if not isinstance(symbol, str) or symbol not in dimension_bounds:
                    raise ValueError(
                        f"input {value.name!r} axis {axis} requires DIMENSION_BOUNDS for its ONNX symbol"
                    )
                specialized.append(dimension_bounds[symbol])
            values[value.name] = specialized
        else:
            values[value.name] = _static_shape(value.type, value.name)
        value_types[value.name] = "float32"
    canonical_initializers = {value.name for value in graph.initializers}
    missing = sorted(canonical_initializers - set(model_initializers))
    if missing:
        raise ValueError(f"canonical ONNX initializer is absent from model: {missing[0]}")

    nodes = []
    for index, operator in enumerate(graph.operators):
        provider = PROVIDERS.get(operator.op_type)
        if provider is None:
            place = operator.location or operator.node_name or f"operator {index}"
            raise ValueError(
                f"unsupported canonical ONNX operator {operator.op_type!r} at {place}"
            )
        if len(operator.outputs) != len(operator.output_types):
            raise ValueError(f"ONNX {operator.op_type} result and type counts differ")
        inputs = []
        for ssa in operator.inputs:
            if ssa not in ssa_names:
                # onnx.NoValue is structural and is not an API operand.
                continue
            inputs.append(ssa_names[ssa])
        outputs = []
        for output_index, (ssa, type_) in enumerate(
            zip(operator.outputs, operator.output_types)
        ):
            name = _value_name(operator, output_index, used)
            ssa_names[ssa] = name
            values[name] = _static_shape(type_, f"ONNX {operator.op_type} output")
            value_types[name] = "float32"
            outputs.append(name)
        if len(outputs) != 1:
            raise ValueError(
                f"ONNX {operator.op_type} multi-result lowering is not implemented"
            )
        nodes.append(
            {
                "domain": "",
                "op_type": operator.op_type,
                "name": operator.node_name or operator.location or f"node_{index}",
                "inputs": inputs,
                "output": outputs[0],
                "shape": values[outputs[0]],
                "attributes": {
                    name: _attribute(value)
                    for name, value in operator.attributes.items()
                },
                "provider": provider,
            }
        )
    for output in graph.outputs:
        if output.ssa_name not in ssa_names:
            raise ValueError(f"canonical ONNX output {output.name!r} is not produced")
    return {
        "nodes": nodes,
        "values": values,
        "value_types": value_types,
        "inputs": {value.name: values[value.name] for value in graph.inputs},
        "outputs": {
            value.name: values[ssa_names[value.ssa_name]] for value in graph.outputs
        },
        "initializers": {
            name: model_initializers[name] for name in canonical_initializers
        },
        "session": {
            "inputs": {
                value.name: ssa_names[value.ssa_name] for value in graph.inputs
            },
            "outputs": {
                value.name: ssa_names[value.ssa_name] for value in graph.outputs
            },
            "initializers": {
                name: model_initializers[name] for name in canonical_initializers
            },
        },
    }


def emit_host(model: dict, canonical_mlir: str, dimension_bounds=None) -> str:
    return emit_c_host(
        host_graph(parse(canonical_mlir), model, dimension_bounds)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("canonical_mlir", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--dimension-bound", action="append", default=[])
    args = parser.parse_args()
    bounds = {}
    for item in args.dimension_bound:
        name, separator, raw_value = item.partition("=")
        if not separator or not name or not raw_value.isdigit() or int(raw_value) <= 0:
            parser.error(f"invalid --dimension-bound {item!r}; expected name=positive_integer")
        if name in bounds:
            parser.error(f"duplicate --dimension-bound {name!r}")
        bounds[name] = int(raw_value)
    model = parse_model(args.model.read_bytes())
    if any(
        tensor["data_location"] == 1
        for tensor in model["graph"]["initializers"].values()
    ):
        validate_external_data(model, args.model.parent)
    source = emit_host(
        model,
        args.canonical_mlir.read_text(encoding="utf-8"),
        bounds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")
    print(f"onnx_mlir_host_bytes={len(source.encode())}")


if __name__ == "__main__":
    main()
