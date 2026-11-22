"""Execute the user OpenCL C kernels of Unity library artifacts natively.

A Host C on the standard OpenCL API launches OpenCL C kernels of its own
DEVICE_LIBRARIES through the linked OpenCL runtime. This builds two such
Unity libraries through the source-tree UdonCL package, the SAXPY example and
tests/build/opencl_user_kernel, and runs the kernel shaders of their artifacts
on the native fragment renderer, which must reproduce the Host reference
bit for bit.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cmake", required=True)
parser.add_argument("--generator", required=True)
parser.add_argument("--make-program", required=True)
parser.add_argument("--compiler", required=True)
parser.add_argument("--clang", required=True)
parser.add_argument("--llvm-link", required=True)
parser.add_argument("--runner", required=True)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
output = args.output.resolve()
if output.exists():
    shutil.rmtree(output)
output.mkdir(parents=True)
environment = dict(os.environ)
if sys.platform != "win32":
    icds = list(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json"))
    assert icds, "install mesa-vulkan-drivers and libvulkan-dev"
    environment["VK_DRIVER_FILES"] = environment["VK_ICD_FILENAMES"] = str(icds[0])


def run(command):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        env=environment,
        timeout=600,
    )
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    return result


def build(source, target, binary):
    run(
        [
            args.cmake,
            "-S",
            source,
            "-B",
            binary,
            "-G",
            args.generator,
            f"-DCMAKE_MAKE_PROGRAM={args.make_program}",
            f"-DUdonCL_DIR={root / 'cmake'}",
            f"-DUDONCLC_EXECUTABLE={Path(args.compiler).resolve()}",
            f"-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}",
            f"-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}",
        ]
    )
    run([args.cmake, "--build", binary, "--target", target])
    artifact = binary / "udoncl" / target / "artifact"
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ready_for_unity"] is True
    return artifact, {
        kernel["name"]: (kernel["id"], artifact / f"kernels/kernel_{kernel['id']}.hlsl")
        for kernel in manifest["device"]["kernels"]
    }


_, example = build(root / "examples/saxpy", "saxpy-unity-library", output / "saxpy")
_, user = build(
    root / "tests/build/opencl_user_kernel",
    "opencl-user-kernel",
    output / "opencl-user-kernel",
)
# The kernels nobody describes get the user kernel ids, in name order.
assert {name: kernel[0] for name, kernel in example.items()} == {"saxpy": 1024}
assert {name: kernel[0] for name, kernel in user.items()} == {
    "user_saxpy": 1024,
    "user_scale": 1025,
}

vertex = output / "fullscreen.hlsl"
vertex.write_text(
    "float4 vert(uint id : SV_VertexID) : SV_Position { return float4(id == 2 ? 3.0 : -1.0, id == 1 ? 3.0 : -1.0, 0, 1); }\n",
    encoding="utf-8",
)
shaders = [example["saxpy"][1], user["user_saxpy"][1], user["user_scale"][1], vertex]
if sys.platform != "win32":
    glslang = shutil.which("glslangValidator")
    assert glslang, "install glslang-tools"
    compiled = []
    for index, shader in enumerate(shaders):
        stage = "vert" if shader == vertex else "frag"
        binary = output / f"shader_{index}.spv"
        run(
            [
                glslang,
                "-D",
                "-V",
                "-S",
                stage,
                "-e",
                stage,
                "--auto-map-locations",
                shader,
                "-o",
                binary,
            ]
        )
        compiled.append(binary)
    shaders = compiled

result = run([args.runner, *shaders])
for counter in ["example_saxpy", "user_saxpy", "user_scale"]:
    assert f"{counter}_bitexact=32/32" in result.stdout, result.stdout
assert "draws=3" in result.stdout, result.stdout
print(result.stdout, end="")
