"""Build the standard-operator Tiny Llama Unity ONNX artifact."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime/onnx"))

from udon_onnx import compile_operators, parse_model  # noqa: E402
from udon_pages import materialize_model  # noqa: E402


def default_template(build_dir: Path) -> Path:
    return build_dir.parent / "udondnn-device-run" / "unity-udonaten-int4-mm-bundle"


def ensure_template(build_dir: Path, template: Path) -> None:
    required = [
        template / "manifest.json",
        template / "kernels/kernel_215.shader",
        template / "kernels/kernel_217.shader",
        template / "kernels/kernel_218.shader",
        template / "kernels/kernel_221.shader",
        template / "kernels/kernel_226.shader",
    ]
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
    missing = [path.relative_to(template) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"UdonDNN Unity template is missing: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/udoncl")
    parser.add_argument("--unity-template-directory", type=Path)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Stop after generating and validating the standard ONNX plan.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    custom_model = output / "custom-model.onnx"
    custom_plan = output / "custom-plan.json"
    model = output / "model.onnx"
    plan_path = output / "execution-plan.json"
    subprocess.run(
        [
            sys.executable,
            SOURCE / "generate_onnx.py",
            "--output",
            custom_model,
            "--plan",
            custom_plan,
            "--standard-output",
            model,
            "--standard-plan",
            plan_path,
        ],
        cwd=ROOT,
        check=True,
    )
    plan = compile_operators(parse_model(model.read_bytes()))
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if len(plan["passes"]) != 18 or any(
        not device_pass.get("shader") for device_pass in plan["passes"]
    ):
        raise RuntimeError("Tiny Llama does not have a complete Device schedule")
    if args.plan_only:
        print("tiny_llama_onnx_plan=passed nodes=16 passes=18")
        return

    build_dir = args.build_dir.resolve()
    template = (
        args.unity_template_directory.resolve()
        if args.unity_template_directory is not None
        else default_template(build_dir)
    )
    ensure_template(build_dir, template)
    artifact = output / "unity-artifact"
    manifest = materialize_model(
        plan,
        output,
        artifact,
        unity_template_directory=template,
        model_path=model,
        program_name="onnx_tiny_llama_example",
    )
    if not manifest["complete_graph"] or manifest["unmaterialized_calls"]:
        raise RuntimeError("Tiny Llama Unity artifact is incomplete")
    print(
        f"tiny_llama_onnx_artifact={artifact} "
        f"nodes={manifest['aot_call_count']} passes=18"
    )


if __name__ == "__main__":
    main()
