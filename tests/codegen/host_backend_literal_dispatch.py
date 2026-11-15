import re
import subprocess
import sys
from pathlib import Path


optimizer = Path(sys.argv[1])


source = r'''module attributes {
  vrc.host_stage = "udonhl",
  vrc.host_storage_pressure_request = {
    schema_version = 1 : i32,
    measured_data_slots = 3 : i32,
    data_slot_capacity = 1 : i32,
    required_resident_slot_savings = 2 : i32
  },
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_output"},
    {id = 2 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_output_2"}
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
  func.func private @dispatch_static_read_second() {
    %id = arith.constant 2 : i32
    %bytes = arith.constant 4 : i32
    %output, %status = vrc.buffer_alloc %id, %bytes {
      format = "rgba8_unorm", usage = "storage"
    } : (i32, i32) -> (!vrc.buffer, i32)
    %weight = "vrc.static_buffer"() {
      byte_size = 4 : i32, resource_id = 33 : i32
    } : () -> !vrc.buffer
    %desc = vrc.dispatch_desc_create {
      kernel_id = 2 : i32, scalar_kinds = []
    } : () -> !vrc.dispatch_desc
    %event = vrc.dispatch %desc(%weight, %output) {
      argument_kinds = ["buffer", "buffer"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 2 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.static_kernel_binding_snapshot = {
        kernel_id = 2 : i32,
        resources = [
          {access = "read", kind = "static_tensor", operand_index = 0 : i32, resource_id = 33 : i32},
          {access = "write", kind = "allocation", operand_index = 1 : i32, resource_id = 2 : i32}
        ],
        scalars = [], schema_version = 1 : i32
      },
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_2",
      vrc.unity_output_resource_index = 1 : i32,
      vrc.unity_resource_heap_slots = ["", "__udoncl_output_2"],
      vrc.unity_static_tensor_ids = [33 : i32, -1 : i32]
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return
  }
}'''


def run(arguments, text):
    return subprocess.run(
        [optimizer, *arguments], input=text, text=True, capture_output=True
    )


allocation_prefix = [
    "--vrc-expose-host-backend-literal-uses",
    "--vrc-discover-host-rematerialization",
]


def allocate(text):
    return run([*allocation_prefix, "--vrc-allocate-host-literal-storage"], text)


allocated = allocate(source)
assert allocated.returncode == 0, (allocated.stdout, allocated.stderr)
legacy_allocated = run(
    [*allocation_prefix, "--vrc-allocate-host-storage"], source
)
assert legacy_allocated.returncode == 0, (
    legacy_allocated.stdout,
    legacy_allocated.stderr,
)
assert legacy_allocated.stdout == allocated.stdout
repeated = allocate(source)
assert repeated.returncode == 0
assert repeated.stdout == allocated.stdout
assert "vrc.host_literal_selected_placement_version = 1 : i32" in allocated.stdout
assert "vrc.host_storage_selection" in allocated.stdout
assert "guaranteed_resident_slot_savings = 2 : i32" in allocated.stdout
assert 'literal_key = "SystemInt32:0x16"' in allocated.stdout
assert 'literal_key = "SystemInt32:0x21"' in allocated.stdout
selected_color = re.search(
    r"vrc\.host_literal_selected_placements = \[\{color = (\d+) : i32,.*?"
    r'literal_key = "SystemInt32:0x16"',
    allocated.stdout,
    re.DOTALL,
)
assert selected_color, allocated.stdout
color = int(selected_color.group(1))

without_request = re.sub(
    r"\s*vrc\.host_storage_pressure_request = \{.*?\},\n",
    "\n",
    source,
    count=1,
    flags=re.DOTALL,
)
candidate_only = allocate(without_request)
assert candidate_only.returncode == 0, (
    candidate_only.stdout,
    candidate_only.stderr,
)
assert "vrc.host_literal_selected_placement_version" not in candidate_only.stdout
assert "vrc.host_literal_selected_placements" not in candidate_only.stdout
assert "vrc.host_storage_selection" not in candidate_only.stdout

async_sentinel = r'''module {
  func.func @untouched() attributes {vrc.async_spill_slots = "sentinel"} {
    return
  }
}'''
sentinel_result = run(["--vrc-allocate-host-literal-storage"], async_sentinel)
assert sentinel_result.returncode == 0, (
    sentinel_result.stdout,
    sentinel_result.stderr,
)
assert 'vrc.async_spill_slots = "sentinel"' in sentinel_result.stdout


def reject_allocation(text, diagnostic):
    result = allocate(text)
    assert result.returncode != 0, (result.stdout, result.stderr)
    assert diagnostic in result.stderr, result.stderr


reject_allocation(
    source.replace("schema_version = 1 : i32", "schema_version = 2 : i32", 1),
    "invalid Host storage pressure request",
)
reject_allocation(
    source.replace("schema_version = 1 : i32", "schema_version = 1 : i64", 1),
    "invalid Host storage pressure request",
)
reject_allocation(
    source.replace(
        "required_resident_slot_savings = 2 : i32",
        "required_resident_slot_savings = 1 : i32",
        1,
    ),
    "invalid Host storage pressure request",
)
reject_allocation(
    source.replace("measured_data_slots = 3 : i32", "measured_data_slots = 4 : i32", 1)
    .replace(
        "required_resident_slot_savings = 2 : i32",
        "required_resident_slot_savings = 3 : i32",
        1,
    ),
    "resident slots but only 2 are safely selectable",
)
reject_allocation(
    source.replace(
        "module attributes {",
        "module attributes {vrc.host_literal_selected_placement_version = 1 : i32, ",
        1,
    ),
    "Host storage pressure selection already has an output",
)

materialized = run(["--vrc-materialize-host-storage"], allocated.stdout)
assert materialized.returncode == 0, (materialized.stdout, materialized.stderr)
assert materialized.stdout.count('"udonhl.literal_materialize"') == 2

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
assert 'initial = "33"' not in lowered.stdout

candidate_materialized = run(
    ["--vrc-materialize-host-storage"], candidate_only.stdout
)
assert candidate_materialized.returncode == 0, (
    candidate_materialized.stdout,
    candidate_materialized.stderr,
)
assert '"udonhl.literal_materialize"' not in candidate_materialized.stdout
candidate = run(
    ["--vrc-lower-udonhl-to-udonasm", "--verify-each"],
    candidate_materialized.stdout,
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
assert 'initial = "33"' in candidate.stdout


def address_count(text):
    data_slots = int(re.search(r"data_slot_count = (\d+) : i64", text).group(1))
    externs = set(re.findall(r'udonasm\.extern "([^"]+)"', text))
    return data_slots + len(externs)


selected_addresses = address_count(lowered.stdout)
candidate_addresses = address_count(candidate.stdout)
assert candidate_addresses == selected_addresses + 1, (
    selected_addresses,
    candidate_addresses,
)
reserve = 512 - selected_addresses


def with_reserve(text):
    return text.replace(
        "module attributes {",
        f"module attributes {{vrc.final_data_slot_reserve = {reserve} : i32, ",
        1,
    )


selected_boundary = run(
    ["--vrc-lower-udonhl-to-udonasm", "--verify-each"],
    with_reserve(materialized.stdout),
)
assert selected_boundary.returncode == 0, (
    selected_boundary.stdout,
    selected_boundary.stderr,
)
assert address_count(selected_boundary.stdout) + reserve == 512
candidate_boundary = run(
    ["--vrc-lower-udonhl-to-udonasm", "--verify-each"],
    with_reserve(candidate_materialized.stdout),
)
assert candidate_boundary.returncode != 0, (
    candidate_boundary.stdout,
    candidate_boundary.stderr,
)
assert (
    f"requires {candidate_addresses} heap/extern addresses after constant "
    f"materialization; limit is {selected_addresses}"
    in candidate_boundary.stderr
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
    "host_backend_literal_dispatch=passed selected=2 candidate=2 legacy=1 "
    "diagnostics=7"
)
