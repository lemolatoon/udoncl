import re
import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2]).read_text()
command = [optimizer, "--vrc-expose-host-backend-literal-uses"]
discover_command = command + ["--vrc-discover-host-rematerialization"]


def run(text, success=True, diagnostic=None):
    result = subprocess.run(command, input=text, text=True, capture_output=True)
    if success:
        assert result.returncode == 0, (result.stdout, result.stderr)
    else:
        assert result.returncode != 0, result.stdout
        assert diagnostic in result.stderr, result.stderr
    return result.stdout


first = run(source)
second = run(first)
assert first == second
assert "vrc.host_backend_literal_use_version = 1 : i32" in first
assert first.count('vrc.host_literal_role = "static_resource_index"') == 2
assert first.count("vrc.host_literal_generated_version = 1 : i32") == 2
assert "static_indices(%c4_i32, %c9_i32)" in first
assert "static_resource_positions = array<i32: 0, 2>" in first
dynamic_dispatch = next(
    line
    for line in first.splitlines()
    if "kernel_id = 8 : i32" in line and "vrc.dispatch " in line
)
assert "static_indices" not in dynamic_dispatch

bound_source = source.replace(
    "func.func @run(%dynamic: !vrc.buffer)",
    "func.func @run(%dynamic: !vrc.buffer, %bound0: !vrc.buffer, "
    "%bound1: !vrc.buffer)",
    1,
).replace(
    "%static0, %dynamic, %static1", "%bound0, %dynamic, %bound1", 1
)
bound = run(bound_source)
assert bound.count('vrc.host_literal_role = "static_resource_index"') == 2
assert "static_indices(%c4_i32, %c9_i32)" in bound
run(
    bound_source.replace(
        ", vrc.unity_static_tensor_ids = [4 : i32, -1 : i32, 9 : i32]",
        "",
        1,
    ),
    success=False,
    diagnostic="neither SSA nor bound provenance",
)
run(
    source.replace(
        "vrc.unity_static_tensor_ids = [4 : i32, -1 : i32, 9 : i32]",
        "vrc.unity_static_tensor_ids = [5 : i32, -1 : i32, 9 : i32]",
        1,
    ),
    success=False,
    diagnostic="does not match bound compatibility identity",
)

discovered = subprocess.run(
    discover_command, input=source, text=True, capture_output=True, check=True
).stdout
assert "vrc.host_literal_use_site_version = 3 : i32" in discovered
assert discovered.count('value_role = "static_resource_index"') == 2
assert discovered.count('kind = "static_resource_index"') == 2
assert discovered.count('origin = "backend"') == 2
assert "resource_index = 0 : i32" in discovered
assert "resource_index = 2 : i32" in discovered
assert ":static_resource:" not in discovered
backend_sites = [
    entry
    for entry in re.findall(r"\{([^{}]+)\}", discovered)
    if 'value_role = "static_resource_index"' in entry
]
assert len(backend_sites) == 2
backend_instances = {
    re.search(r'scratch_instance_id = "([^"]+)"', entry).group(1)
    for entry in backend_sites
}
assert len(backend_instances) == 2
allocated = subprocess.run(
    discover_command + ["--vrc-allocate-host-storage"],
    input=source,
    text=True,
    capture_output=True,
    check=True,
).stdout
scratch_entries = re.findall(r"\{([^{}]+)\}", allocated)
backend_colors = {
    int(re.search(r"color = (\d+) : i32", entry).group(1))
    for entry in scratch_entries
    if "color = " in entry
    and any(f'instance_id = "{instance}"' in entry for instance in backend_instances)
}
assert len(backend_colors) == 2
legacy = subprocess.run(
    [optimizer, "--vrc-discover-host-rematerialization"],
    input=source,
    text=True,
    capture_output=True,
    check=True,
).stdout
assert legacy.count('kind = "static_tensor_index"') == 2
assert 'value_role = "static_resource_index"' not in legacy

run(
    source.replace(
        "resource_id = 4 : i32}], scalars",
        "resource_id = 5 : i32}], scalars",
    ),
    success=False,
    diagnostic="resource id does not match its operand",
)
run(
    source.replace(
        'access = "read", kind = "static_tensor", operand_index = 0',
        'access = "write", kind = "static_tensor", operand_index = 0',
    ).replace(
        'access = "write", kind = "allocation", operand_index = 1',
        'access = "read", kind = "allocation", operand_index = 1',
    ).replace(
        'resource_accesses = ["read", "write", "read"]',
        'resource_accesses = ["write", "read", "read"]',
    ),
    success=False,
    diagnostic="static tensor dispatch resource must be read",
)
run(
    source.replace("scalars = [], schema_version", "schema_version", 1),
    success=False,
    diagnostic="invalid static kernel binding snapshot",
)
unsupported_version = source.replace(
    "module {",
    "module attributes {vrc.host_backend_literal_use_version = 2 : i32} {",
    1,
)
run(
    unsupported_version.rsplit("}", 1)[0] + "}",
    success=False,
    diagnostic="unsupported Host backend literal use version",
)
run(
    first.replace(
        'vrc.host_literal_role = "static_resource_index"} 4 : i32',
        'vrc.host_literal_role = "static_resource_index"} 5 : i32',
        1,
    ),
    success=False,
    diagnostic="generated static resource index literal is stale",
)
unversioned = first.replace(
    "vrc.host_backend_literal_use_version = 1 : i32", "", 1
)
run(
    unversioned,
    success=False,
    diagnostic="unversioned static resource index operands are not allowed",
)
print("host_backend_literal_exposure=passed static_indices=2")
