"""Provenance analysis of a long chain of conditional releases.

Each conditional region visits its join block from two predecessors. The
block worklist must reach a fixed point in time linear in the chain length
instead of revisiting every later block once per pending branch arm.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


optimizer = Path(sys.argv[1])
count = 192

lines = ["module {"]
for index in range(count):
    lines.append(
        '  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, '
        'initial_value = 0 : i32, linkage = "internal", '
        "storage_type = !udonhl.ptr<0>, "
        f'sym_name = "buffer_{index}"}}> : () -> ()'
    )
lines += [
    "  func.func private @clCreateBuffer() -> !udonhl.ptr<0>",
    "  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32",
    "  func.func @Initialize() attributes {udoncl.export} {",
]
for index in range(count):
    lines += [
        f"    %resource_{index} = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>",
        f"    %address_{index} = udonhl.address_of @buffer_{index} : !udonhl.ptr<0>",
        f'    "udonhl.store"(%resource_{index}, %address_{index}) '
        "<{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()",
    ]
lines += [
    "    return",
    "  }",
    "  func.func @Dispose(%mask: i32) attributes {udoncl.export} {",
    "    %zero = arith.constant 0 : i32",
    "    %four = arith.constant 4 : i32",
    '    %result = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>',
    '    "udonhl.store"(%zero, %result) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()',
    "    cf.br ^check_0",
]
for index in range(count):
    lines += [
        f"  ^check_{index}:",
        f"    %bit_{index} = arith.constant {index} : i32",
        f"    %owned_{index} = arith.cmpi ne, %mask, %bit_{index} : i32",
        f"    cf.cond_br %owned_{index}, ^release_{index}, ^check_{index + 1}",
        f"  ^release_{index}:",
        f"    %address_{index} = udonhl.address_of @buffer_{index} : !udonhl.ptr<0>",
        f'    %loaded_{index} = "udonhl.load"(%address_{index}) '
        "<{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>",
        f"    %status_{index} = func.call @clReleaseMemObject(%loaded_{index}) "
        ": (!udonhl.ptr<0>) -> i32",
        f'    "udonhl.store"(%status_{index}, %result) '
        "<{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()",
        f'    %null_{index} = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>',
        f'    "udonhl.store"(%null_{index}, %address_{index}) '
        "<{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()",
        f"    cf.br ^check_{index + 1}",
    ]
lines += [
    f"  ^check_{count}:",
    "    return",
    "  }",
    "}",
]

with tempfile.TemporaryDirectory() as directory:
    source = Path(directory) / "branch_chain.mlir"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            optimizer,
            "--vrc-discover-static-device-resources",
            "--vrc-propagate-device-resource-provenance",
            source,
        ],
        text=True,
        capture_output=True,
    )
assert result.returncode == 0, result.stderr[-2000:]
releases = [
    line for line in result.stdout.splitlines() if "call @clReleaseMemObject" in line
]
assert len(releases) == count, len(releases)
identities = set()
for release in releases:
    marker = "resource_ids = ["
    assert marker in release, release
    ids = release[release.index(marker) + len(marker) :].split("]", 1)[0]
    assert ids.count(": i32") == 1, release
    identities.add(ids)
assert len(identities) == count, len(identities)
print(f"static_device_resource_branch_chain=passed releases={count}")
