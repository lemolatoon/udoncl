"""Build the arena-add ONNX plan and optional Unity artifact."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / "runtime/onnx"))

from generate_onnx import WORDS, build_model  # noqa: E402
from udon_onnx import compile_operators, parse_model  # noqa: E402
from udon_pages import materialize_model  # noqa: E402


def default_template(build_dir: Path) -> Path:
    return build_dir.parent / "udondnn-device-run" / "unity-udonaten-int4-mm-bundle"


def ensure_template(build_dir: Path, template: Path) -> None:
    required = [template / "manifest.json", template / "kernels/kernel_215.shader"]
    if all(path.is_file() for path in required):
        return
    output = template.parent
    subprocess.run(
        [
            sys.executable,
            ROOT / "libraries/udondnn/run_device.py",
            "--build-dir",
            build_dir,
            "--output",
            output,
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
        help="Stop after generating and validating the ONNX execution plan.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    model_path = output / "model.onnx"
    model_path.write_bytes(build_model())
    plan = compile_operators(parse_model(model_path.read_bytes()))
    plan_path = output / "execution-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    if plan["execution_provider"] != "UdonOperatorExecutionProvider":
        raise RuntimeError("arena-add selected the wrong execution provider")
    if len(plan["passes"]) != 1 or plan["passes"][0]["shader"] != "kernel_215.shader":
        raise RuntimeError("arena-add did not lower to the expected pointwise Device pass")
    if args.plan_only:
        print(f"onnx_arena_add_plan=passed words={WORDS} passes=1")
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
        model_path=model_path,
        program_name="onnx_arena_add_example",
    )
    if not manifest["complete_graph"]:
        raise RuntimeError("arena-add Unity artifact is incomplete")
    if [item["words"] for item in manifest["model_inputs"]] != [WORDS, WORDS]:
        raise RuntimeError("arena-add Unity artifact has invalid inputs")
    if [item["words"] for item in manifest["model_outputs"]] != [WORDS]:
        raise RuntimeError("arena-add Unity artifact has invalid outputs")
    print(f"onnx_arena_add_artifact={artifact} words={WORDS} passes=1")


if __name__ == "__main__":
    main()
