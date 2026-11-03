"""Build and optionally materialize the Llama 3.1 8B Unity ONNX artifact."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / "runtime/onnx"))

from generate_onnx import build_model, build_operator_model  # noqa: E402
from udon_onnx import (  # noqa: E402
    compile_model,
    compile_operators,
    parse_model,
    validate_external_data,
)
from udon_pages import materialize_model, plan_int4_model_bundle  # noqa: E402


REQUIRED_TEMPLATE_FILES = [
    "manifest.json",
    "display.shader",
    "compose.shader",
    "scatter.shader",
    "kernels/kernel_208.shader",
    "kernels/kernel_215.shader",
    "kernels/kernel_216.shader",
    "kernels/kernel_217.shader",
    "kernels/kernel_218.shader",
]


def default_template(build_dir: Path) -> Path:
    return build_dir.parent / "udondnn-device-run" / "unity-udonaten-int4-mm-bundle"


def ensure_template(build_dir: Path, template: Path) -> None:
    required = [template / relative for relative in REQUIRED_TEMPLATE_FILES]
    if all(path.is_file() for path in required):
        return
    subprocess.run(
        [
            sys.executable,
            ROOT / "libraries/udondnn/run_device.py",
            "--build-dir",
            build_dir,
            "--output",
            template.parent,
        ],
        cwd=ROOT,
        check=True,
    )
    missing = [
        path.relative_to(template).as_posix()
        for path in required
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"UdonDNN Unity template is missing: {missing}")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/udoncl")
    weight_source = parser.add_mutually_exclusive_group()
    weight_source.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Pack a licensed safetensors checkpoint into weights.q4, then "
            "materialize the Unity artifact."
        ),
    )
    weight_source.add_argument(
        "--external-data-directory",
        type=Path,
        help="Reuse a directory that already contains the generated weights.q4.",
    )
    parser.add_argument("--unity-template-directory", type=Path)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Read weights.q4 and emit the multi-gigabyte importable Unity artifact.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fused_model, index, storage_bytes = build_model()
    operator_model, operator_index, operator_storage_bytes = build_operator_model()
    if index != operator_index or storage_bytes != operator_storage_bytes:
        raise RuntimeError("fused and operator models disagree about external weights")

    fused_model_path = output / "model.onnx"
    fused_plan_path = output / "execution-plan.json"
    operator_model_path = output / "operators.onnx"
    operator_plan_path = output / "operators-plan.json"
    index_path = output / "weights-index.json"
    fused_model_path.write_bytes(fused_model)
    operator_model_path.write_bytes(operator_model)
    fused_plan = compile_model(parse_model(fused_model))
    operator_plan = compile_operators(parse_model(operator_model))
    write_json(fused_plan_path, fused_plan)
    write_json(operator_plan_path, operator_plan)
    write_json(
        index_path,
        {
            "schema_version": 1,
            "architecture": "llama-3.1-8b",
            "config": fused_plan["config"],
            "storage_bytes_with_alignment": storage_bytes,
            "tensors": index,
        },
    )

    materialization = plan_int4_model_bundle(operator_plan)
    calls = operator_plan["aot"]["calls"]
    int4_indexes = {
        call["index"]
        for call in calls
        if call.get("kind") == "tiled_library"
        and call.get("function") == "udonatenWeightInt4packMm"
    }
    scheduled_indexes = {
        call["index"] for call in calls if call.get("kind") == "view"
    }
    scheduled_indexes.update(int4_indexes)
    scheduled_indexes.update(
        device_pass["call_index"] for device_pass in operator_plan["passes"]
    )
    materialization["int4_only_complete_graph"] = materialization.pop("complete_graph")
    materialization["int4_only_unmaterialized_calls"] = materialization.pop(
        "unmaterialized_calls"
    )
    materialization["unmaterialized_calls"] = [
        {
            "index": call.get("index"),
            "kind": call.get("kind"),
            "function": call.get("function"),
            "output": call.get("output"),
        }
        for call in calls
        if call.get("index") not in scheduled_indexes
    ]
    materialization["complete_graph"] = not materialization["unmaterialized_calls"]
    write_json(output / "materialization-plan.json", materialization)
    if (
        not materialization["complete_graph"]
        or len(operator_plan["passes"]) != 547
        or materialization["int4_call_count"] != 225
        or materialization["host_schedule"]["direct_device_passes"] != 322
    ):
        raise RuntimeError("Llama 8B operator graph is not fully scheduled")

    materialize = args.materialize or args.checkpoint is not None
    if not materialize:
        print(
            "llama_8b_preflight=passed "
            f"operators={len(operator_plan['nodes'])} "
            f"int4_calls={materialization['int4_call_count']} "
            f"direct_passes={materialization['host_schedule']['direct_device_passes']} "
            f"artifact_bytes={materialization['int4_atlas_bytes']}"
        )
        return

    external_data_directory = args.external_data_directory
    if args.checkpoint is not None:
        from udon_weights import pack_checkpoint

        packed = output / "weights.q4"
        weight_manifest = pack_checkpoint(args.checkpoint, index_path, packed)
        if weight_manifest["bytes"] != storage_bytes:
            raise RuntimeError("packed Llama 8B external weights have the wrong byte size")
        external_data_directory = output
        print(
            "llama_8b_weights=packed "
            f"tensors={len(weight_manifest['tensors'])} bytes={weight_manifest['bytes']}"
        )
    if external_data_directory is None:
        parser.error(
            "--materialize requires --checkpoint or --external-data-directory "
            "containing weights.q4"
        )
    external_data_directory = external_data_directory.resolve()
    weight_summary = validate_external_data(
        parse_model(operator_model), external_data_directory
    )
    if weight_summary["tensors"] != 517:
        raise RuntimeError("Llama 8B external weight set is incomplete")

    artifact = output / "unity-artifact"
    free_bytes = shutil.disk_usage(artifact.parent).free
    required_bytes = materialization["int4_atlas_bytes"]
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Unity artifact needs at least {required_bytes} free bytes; found {free_bytes}"
        )
    build_dir = args.build_dir.resolve()
    template = (
        args.unity_template_directory.resolve()
        if args.unity_template_directory is not None
        else default_template(build_dir)
    )
    ensure_template(build_dir, template)
    manifest = materialize_model(
        operator_plan,
        external_data_directory,
        artifact,
        unity_template_directory=template,
        model_path=operator_model_path,
        program_name="llama_3_1_8b",
    )
    if not manifest["complete_graph"] or manifest["int4_call_count"] != 225:
        raise RuntimeError("materialized Llama 8B Unity graph is incomplete")
    print(
        f"llama_8b_artifact={artifact} calls={manifest['aot_call_count']} "
        f"atlas_bytes={manifest['int4_atlas_bytes']} complete=True"
    )


if __name__ == "__main__":
    main()
