"""Build a Unity library bundle whose Host module carries a synthesized kernel.

The Host MLIR holds the hand-written rotary gather kernel next to a dispatch
declaration that names it, and also dispatches the configured UdonDNN large
OpTensor library kernel. The compiler must assign the synthesized kernel an id
of its own, prove it, emit its shader and manifest entry like a library
kernel, and bind its dispatch site to that kernel's material without anything
being keyed on its name. A declaration that disagrees with the kernel and a
kernel that cannot be proven must fail the build.
"""

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

compiler, clang, output = sys.argv[1:]
output = Path(output).resolve()
output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]

LIBRARY_ID = 234
WORDS = 131072
GRID = [8192, 16, 1]


def run(command, expected=0):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == expected, (command, result.stdout, result.stderr)
    return result


device_ir = output / "large-pointwise.ll"
run(
    [
        clang,
        "-target",
        "spir-unknown-unknown",
        "-x",
        "cl",
        "-cl-std=CL1.2",
        "-cl-kernel-arg-info",
        "-O0",
        "-Xclang",
        "-disable-O0-optnone",
        "-fomit-frame-pointer",
        "-ffp-contract=off",
        "-S",
        "-emit-llvm",
        root / "libraries/udondnn/device/large_pointwise.cl",
        "-o",
        device_ir,
    ]
)

config = {
    "schema_version": 1,
    "program": "synthesized_dispatch",
    "exports": [
        {"symbol": "rotate", "public_name": "Rotate", "id": 1, "parameters": []}
    ],
    "buffers": [
        {
            "id": identifier,
            "name": f"tensor{identifier}",
            "byte_size": WORDS * 4,
            "width": GRID[0],
            "height": GRID[1],
        }
        for identifier in range(1, 8)
    ],
    "device_kernels": [
        {
            "schema_version": 1,
            "profile": "gather-f32",
            "words": WORDS,
            "kernel": "udondnn_large_op_tensor",
            "kernel_id": LIBRARY_ID,
            "global_size": GRID,
            "local_size": [1, 1, 1],
            "global_offset": [0, 0, 0],
        }
    ],
}
kernel_source = (root / "tests/boundary/synthesized_rotary.mlir").read_text(
    encoding="utf-8"
)
host_source = (root / "tests/boundary/unity_synthesized_dispatch.mlir").read_text(
    encoding="utf-8"
)


def build(
    name,
    kernel=kernel_source,
    host=host_source,
    buffers=None,
    library=True,
    expected=0,
):
    variant = {**config, **({"buffers": buffers} if buffers else {})}
    if not library:
        del variant["device_kernels"]
    config_path = output / f"{name}.json"
    config_path.write_text(json.dumps(variant), encoding="utf-8")
    host_path = output / f"{name}.mlir"
    host_path.write_text(kernel + host, encoding="utf-8", newline="\n")
    bundle = output / name
    if bundle.exists():
        shutil.rmtree(bundle)
    result = run(
        [
            compiler,
            "--emit-unity-library-bundle",
            config_path,
            "--host-mlir",
            host_path,
            *(["--device-llvm", device_ir] if library else []),
            "-o",
            bundle,
        ],
        expected,
    )
    return bundle, result


# The kernel also declares its cost: live fragments load x twice and cos and
# sin once, and only the first `count` fragments are live.
cost = "vrc.fragment_cost = {active = [{index = 4, max = 256}], loads = 4}, "
assert kernel_source.count("vrc.profile = ") == 1
bundle, _ = build(
    "bundle", kernel_source.replace("vrc.profile = ", cost + "vrc.profile = ")
)
manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
assert manifest["artifact_stage"] == "unity-udoncl-library"

# The synthesized kernel is an ordinary Device kernel with a compiler-assigned
# id that cannot collide with a library kernel.
kernels = {kernel["name"]: kernel for kernel in manifest["device"]["kernels"]}
assert set(kernels) == {"udondnn_large_op_tensor", "synthesized_rotary"}, kernels
assert kernels["udondnn_large_op_tensor"]["id"] == LIBRARY_ID
synthesized = kernels["synthesized_rotary"]
synthesized_id = synthesized["id"]
assert synthesized_id == 4096, synthesized_id
assert synthesized["global_size"] == GRID
assert [argument["access"] for argument in synthesized["args"][:4]] == [
    "read",
    "read",
    "read",
    "write",
]
assert len(synthesized["args"]) == 5
assert synthesized["descriptor"]["scalar_payload_bytes"] == 4
assert synthesized["shader"] == f"kernels/kernel_{synthesized_id}.shader"
shader = (bundle / synthesized["shader"]).read_text(encoding="utf-8")
for expected in [
    '_VRC_Input ("Input 0", 2D)',
    '_VRC_Input1 ("Input 1", 2D)',
    '_VRC_Input2 ("Input 2", 2D)',
    '_VRC_Arg4 ("Arg4", Integer)',
]:
    assert expected in shader, (expected, shader)
assert "_VRC_Input3" not in shader and "_VRC_Arg3" not in shader
hlsl = (bundle / f"kernels/kernel_{synthesized_id}.hlsl").read_text(encoding="utf-8")
assert len(re.findall(r"\.Load\(", hlsl)) == 4, hlsl
assert (bundle / f"kernels/kernel_{LIBRARY_ID}.shader").is_file()

# Its dispatch site is bound by id like the library site next to it.
sites = {site["kernel_id"]: site for site in manifest["dispatch_sites"]}
assert set(sites) == {LIBRARY_ID, synthesized_id}, sites
site = sites[synthesized_id]
assert site["resource_accesses"] == ["read", "read", "read", "write"], site
assert site["function"] == "rotate" and site["resources"] == 4, site

assembly = (bundle / "host.uasm").read_text(encoding="utf-8")
for kernel_id in (LIBRARY_ID, synthesized_id):
    assert f".export __udoncl_kernel_material_{kernel_id}" in assembly, kernel_id
material = f"__udoncl_kernel_material_{synthesized_id}"
# Everything between the first use of the kernel's material and its Blit
# binds that kernel: three input textures and the i32 count. A constant count
# is bound once, ahead of the arguments every dispatch binds.
binding = assembly[assembly.index(f"PUSH, {material}") :]
binding = binding[: binding.index("VRCSDKBaseVRCGraphics.__Blit__")]
strings = dict(
    re.findall(r'^\s*(\S+): %SystemString, "([^"]*)"', assembly, re.MULTILINE)
)
pushed = re.findall(r"PUSH, (\S+)", binding)
bound = [strings[name] for name in pushed if name in strings]
assert sorted(bound) == ["_VRC_Arg4", "_VRC_Input", "_VRC_Input1", "_VRC_Input2"], bound
assert binding.count("__SetTexture__SystemString_UnityEngineTexture__") == 3, binding
assert binding.count("__SetInteger__SystemString_SystemInt32__") == 1, binding

# The dispatch carries the kernel's declared cost for the Device work
# scheduler, and nothing of the carrier is left in the Host.
host_ir = (bundle / "host.async-runtime.mlir").read_text(encoding="utf-8")
dispatches = re.findall(r"= vrc\.dispatch .*", host_ir)
assert len(dispatches) == 2, dispatches
costed = [line for line in dispatches if "vrc.fragment_cost" in line]
assert len(costed) == 1 and f"kernel_id = {synthesized_id} : i32" in costed[0], costed
assert (
    "vrc.fragment_cost = {active = [{index = 4 : i64, max = 256 : i64}], "
    "loads = 4 : i64}" in costed[0]
), costed
assert "__vrc_opencl_dispatch_rotary" not in assembly + host_ir
assert "gpu.module" not in host_ir and "vrc.synthesized" not in host_ir

# A declaration must repeat the kernel's argument list.
_, mismatch = build(
    "mismatch",
    host=host_source.replace(
        "@__vrc_opencl_dispatch_rotary(i32, i32, i32, i32, i32) -> i32 attributes",
        "@__vrc_opencl_dispatch_rotary(i32, i32, i32, i32, f32) -> i32 attributes",
    )
    .replace(
        "(%x, %cos, %sin, %out, %count) : (i32, i32, i32, i32, i32)",
        "(%x, %cos, %sin, %out, %alpha) : (i32, i32, i32, i32, f32)",
    ),
    expected=1,
)
assert (
    "vrc.synthesized_dispatch declaration does not match kernel "
    "@synthesized_rotary: argument 4 must be 'i32' but is 'f32'" in mismatch.stderr
), mismatch.stderr

# A kernel whose loads cannot be proven in bounds never reaches the Device.
_, unproven = build(
    "unproven",
    kernel=kernel_source.replace("max = 256 : i32", "max = 257 : i32"),
    expected=1,
)
assert "is not proven in bounds" in unproven.stderr, unproven.stderr

# A reference to a kernel the module does not carry is an error.
_, dangling = build(
    "dangling",
    host=host_source.replace(
        "@synthesized::@synthesized_rotary", "@synthesized::@gone"
    ),
    expected=1,
)
assert "does not name a synthesized kernel" in dangling.stderr, dangling.stderr

# The textures the buffers share must hold the kernel's output.
_, small = build(
    "small",
    buffers=[
        {**buffer, "height": 8, "byte_size": WORDS * 2}
        for buffer in config["buffers"]
    ],
    expected=1,
)
assert (
    "synthesized kernel @synthesized_rotary writes 131072 words but the Unity "
    "buffers are 8192x8 textures" in small.stderr
), small.stderr

# Without a call the kernel is dropped and the bundle holds the library alone.
uncalled, _ = build(
    "uncalled",
    host=re.sub(r"\n\s*%rotated = [^\n]*", "", host_source),
)
manifest = json.loads((uncalled / "manifest.json").read_text(encoding="utf-8"))
assert [kernel["id"] for kernel in manifest["device"]["kernels"]] == [LIBRARY_ID]
assert not (uncalled / f"kernels/kernel_{synthesized_id}.shader").exists()

# A Host may bring every kernel itself: no library config, no Device bitcode.
alone, _ = build(
    "alone",
    host=re.sub(r".*udondnn_large_op_tensor.*", "", host_source),
    library=False,
)
manifest = json.loads((alone / "manifest.json").read_text(encoding="utf-8"))
assert [kernel["id"] for kernel in manifest["device"]["kernels"]] == [synthesized_id]
assert [site["kernel_id"] for site in manifest["dispatch_sites"]] == [synthesized_id]
assert (alone / f"kernels/kernel_{synthesized_id}.shader").is_file()

print(f"unity_synthesized_dispatch=ok kernel_id={synthesized_id}")
