import re
import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])
SHIFT = "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__SystemInt32"
LOGICAL_OR = "SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32"


source = r'''module attributes {
  vrc.host_stage = "udonhl",
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_output"}
  ]
} {
  func.func @entry() attributes {vrc.export_id = 71 : i32} {
    return
  }
  func.func private @dispatch_static_read() {
    %id = arith.constant 1 : i32
    %bytes = arith.constant 4 : i32
    %seed6 = arith.constant 6 : i32
    %output, %status = vrc.buffer_alloc %id, %bytes {
      format = "rgba8_unorm", usage = "storage"
    } : (i32, i32) -> (!vrc.buffer, i32)
    %weight = "vrc.static_buffer"() {
      byte_size = 4 : i32, resource_id = 22 : i32
    } : () -> !vrc.buffer
    %desc = vrc.dispatch_desc_create {
      kernel_id = 1 : i32, scalar_kinds = []
    } : () -> !vrc.dispatch_desc
    %event = vrc.dispatch %desc(%weight, %output) {
      argument_kinds = ["buffer", "buffer"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 1 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.static_kernel_binding_snapshot = {
        kernel_id = 1 : i32,
        resources = [
          {access = "read", kind = "static_tensor", operand_index = 0 : i32, resource_id = 22 : i32},
          {access = "write", kind = "allocation", operand_index = 1 : i32, resource_id = 1 : i32}
        ],
        scalars = [], schema_version = 1 : i32
      },
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_1",
      vrc.unity_output_resource_index = 1 : i32,
      vrc.unity_resource_heap_slots = ["", "__udoncl_output"],
      vrc.unity_static_tensor_ids = [22 : i32, -1 : i32]
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return
  }
}'''


def run(arguments, text):
    return subprocess.run(
        [optimizer, *arguments], input=text, text=True, capture_output=True
    )


def entries(text, name, next_name):
    start = text.index(f"{name} = [")
    end = text.index(next_name, start)
    return re.findall(r"\{([^{}]+)\}", text[start:end])


allocated = run(
    [
        "--vrc-expose-host-backend-literal-uses",
        "--vrc-discover-host-rematerialization",
        "--vrc-allocate-host-storage",
    ],
    source,
)
assert allocated.returncode == 0, (allocated.stdout, allocated.stderr)

scratch = next(
    entry
    for entry in entries(
        allocated.stdout,
        "vrc.host_literal_scratch_allocations",
        "vrc.host_literal_scratch_instances",
    )
    if 'literal_key = "SystemInt32:0x16"' in entry
)
instance_id = re.search(r'instance_id = "([^"]+)"', scratch).group(1)
color = int(re.search(r"color = (\d+) : i32", scratch).group(1))
site_ids = re.findall(
    r'"([^"]+)"', re.search(r"site_ids = \[([^]]+)\]", scratch).group(1)
)

resident_entries = entries(
    allocated.stdout,
    "vrc.host_literal_allocations",
    "vrc.host_literal_consumer_groups",
)
resident_colors = {
    re.search(r'key_id = "([^"]+)"', entry).group(1): int(
        re.search(r"color = (\d+) : i32", entry).group(1)
    )
    for entry in resident_entries
}
seed_bindings = ", ".join(
    "{color = %d : i32, literal_key = \"SystemInt32:0x%X\", "
    "physical_type = \"SystemInt32\", storage_domain = \"resident\"}"
    % (resident_colors[f"SystemInt32:0x{seed:X}"], seed)
    for seed in [1, 4, 6]
)
sites = ", ".join(f'"{site}"' for site in site_ids)
selection = (
    "{color = %d : i32, generated_version = 1 : i32, "
    "helper_scratch = 0 : i32, instance_id = \"%s\", "
    "literal_key = \"SystemInt32:0x16\", logical_type = \"i32\", "
    "physical_type = \"SystemInt32\", placement = \"rematerialized\", "
    "recipe = \"rematerialize_i32\", required_externs = [\"%s\", \"%s\"], "
    "seed_bindings = [%s], site_ids = [%s], storage_domain = \"writable\"}"
    % (color, instance_id, SHIFT, LOGICAL_OR, seed_bindings, sites)
)
selected = allocated.stdout.replace(
    "module attributes {",
    "module attributes {vrc.host_literal_selected_placement_version = "
    f"1 : i32, vrc.host_literal_selected_placements = [{selection}], ",
    1,
)
materialized = run(["--vrc-materialize-host-storage"], selected)
assert materialized.returncode == 0, (materialized.stdout, materialized.stderr)
assert materialized.stdout.count('"udonhl.literal_materialize"') == 1

lowered = run(["--vrc-lower-udonhl-to-udonasm", "--verify-each"], materialized.stdout)
assert lowered.returncode == 0, (lowered.stdout, lowered.stderr)
scratch_slot = f"__vrc_shared_r_SystemInt32_{color}"
get_call = re.search(
    r'udonasm\.push "__udoncl_static_tensors"\s+'
    r'udonasm\.push "([^"]+)"\s+'
    r'udonasm\.push "[^"]+"\s+'
    r'udonasm\.extern "UnityEngineTexture2DArray\.__Get__',
    lowered.stdout,
)
assert get_call and get_call.group(1) == scratch_slot, lowered.stdout
assert 'initial = "22"' not in lowered.stdout

candidate = run(
    [
        "--vrc-expose-host-backend-literal-uses",
        "--vrc-lower-udonhl-to-udonasm",
        "--verify-each",
    ],
    source,
)
assert candidate.returncode == 0, (candidate.stdout, candidate.stderr)
candidate_get = re.search(
    r'udonasm\.push "__udoncl_static_tensors"\s+'
    r'udonasm\.push "([^"]+)"\s+'
    r'udonasm\.push "[^"]+"\s+'
    r'udonasm\.extern "UnityEngineTexture2DArray\.__Get__',
    candidate.stdout,
)
assert candidate_get
assert re.search(
    r'initial = "22", name = "'
    + re.escape(candidate_get.group(1))
    + r'", type = "SystemInt32"',
    candidate.stdout,
)

legacy = run(["--vrc-lower-udonhl-to-udonasm", "--verify-each"], source)
assert legacy.returncode == 0, (legacy.stdout, legacy.stderr)
assert 'initial = "22"' in legacy.stdout

exposed = run(["--vrc-expose-host-backend-literal-uses"], source)
assert exposed.returncode == 0, (exposed.stdout, exposed.stderr)


def reject(text, diagnostic):
    result = run(["--vrc-lower-udonhl-to-udonasm", "--verify-each"], text)
    assert result.returncode != 0, (result.stdout, result.stderr)
    assert diagnostic in result.stderr, result.stderr


reject(
    exposed.stdout.replace(
        "vrc.host_backend_literal_use_version = 1 : i32, ", "", 1
    ),
    "explicit static-resource indices require Host backend literal exposure version 1",
)
partial = re.sub(r" static_indices\([^)]*\)", "", exposed.stdout, count=1)
partial = re.sub(
    r"static_resource_positions = array<i32: [^>]*>, ", "", partial, count=1
)
partial = partial.replace(
    "!vrc.buffer, !vrc.buffer, i32) -> !vrc.event",
    "!vrc.buffer, !vrc.buffer) -> !vrc.event",
    1,
)
reject(partial, "incomplete static-resource index coverage")
reject(
    exposed.stdout.replace(
        'vrc.host_literal_role = "static_resource_index"} 22 : i32',
        'vrc.host_literal_role = "static_resource_index"} 23 : i32',
        1,
    ),
    "static-resource index literal disagrees with resource identity",
)
reject(
    exposed.stdout.replace(
        ', vrc.host_literal_role = "static_resource_index"', "", 1
    ),
    "static-resource index is not a generated Host backend literal",
)
reject(
    exposed.stdout.replace(
        "vrc.host_literal_generated_version = 1 : i32",
        "vrc.host_literal_generated_version = 2 : i32",
        1,
    ),
    "static-resource index is not a generated Host backend literal",
)
reject(
    exposed.stdout.replace(
        "byte_size = 4 : i32, resource_id = 22 : i32",
        "byte_size = 4 : i32, resource_id = 23 : i32",
        1,
    ),
    "static-resource compatibility identity disagrees with resource provenance",
)
reject(
    exposed.stdout.replace(
        "static_resource_positions = array<i32: 0>",
        "static_resource_positions = array<i32: 1>",
        1,
    ),
    "static-resource index references a non-static resource",
)

print(
    "host_backend_literal_dispatch=passed selected=1 candidate=1 legacy=1 "
    "diagnostics=7"
)
