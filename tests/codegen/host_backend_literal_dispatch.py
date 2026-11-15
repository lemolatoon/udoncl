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


def address_components(text):
    data_slots = int(re.search(r"data_slot_count = (\d+) : i64", text).group(1))
    externs = set(re.findall(r'udonasm\.extern "([^"]+)"', text))
    return data_slots, len(externs)


selected_data, selected_externs = address_components(lowered.stdout)
candidate_data, candidate_externs = address_components(candidate.stdout)
selected_addresses = selected_data + selected_externs
candidate_addresses = candidate_data + candidate_externs
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


def measure(text):
    result = run(["--vrc-measure-host-storage-capacity"], text)
    assert result.returncode == 0, (result.stdout, result.stderr)
    match = re.search(
        r"vrc\.host_storage_capacity_measurement = \{([^}]*)\}", result.stdout
    )
    assert match, result.stdout
    fields = match.group(1)

    def count(name):
        value = re.search(rf"{name} = (\d+) : i64", fields)
        assert value, fields
        return int(value.group(1))

    fits = re.search(r"fits = (true|false)", fields)
    assert fits, fields
    return result, {
        "addresses": count("address_count"),
        "data": count("data_slot_count"),
        "externs": count("unique_extern_count"),
        "limit": count("address_limit"),
        "capacity": count("data_slot_capacity"),
        "reserve": count("final_data_slot_reserve"),
        "required": count("required_reduction"),
        "fits": fits.group(1) == "true",
    }


measured_selected, selected_measurement = measure(with_reserve(materialized.stdout))
measured_candidate, candidate_measurement = measure(
    with_reserve(candidate_materialized.stdout)
)
assert '"udonhl.literal_materialize"' in measured_selected.stdout
assert "udonasm.data" not in measured_selected.stdout
assert selected_measurement == {
    "addresses": selected_addresses,
    "data": selected_data,
    "externs": selected_externs,
    "limit": selected_addresses,
    "capacity": selected_addresses,
    "reserve": reserve,
    "required": 0,
    "fits": True,
}
assert candidate_measurement["addresses"] == candidate_addresses
assert candidate_measurement["data"] == candidate_data
assert candidate_measurement["externs"] == candidate_externs
assert candidate_measurement["limit"] == selected_addresses
assert candidate_measurement["capacity"] == selected_addresses
assert candidate_measurement["reserve"] == reserve
assert candidate_measurement["required"] == 1
assert not candidate_measurement["fits"]

normal_lowering_ignores_mode_attributes = run(
    ["--vrc-lower-udonhl-to-udonasm", "--verify-each"],
    materialized.stdout.replace(
        "module attributes {",
        "module attributes {vrc.host_storage_measurement_only, ",
        1,
    ),
)
assert normal_lowering_ignores_mode_attributes.returncode == 0, (
    normal_lowering_ignores_mode_attributes.stdout,
    normal_lowering_ignores_mode_attributes.stderr,
)
assert "udonasm.data" in normal_lowering_ignores_mode_attributes.stdout

selected_boundary = run(
    ["--vrc-lower-udonhl-to-udonasm", "--verify-each"],
    with_reserve(materialized.stdout),
)
assert selected_boundary.returncode == 0, (
    selected_boundary.stdout,
    selected_boundary.stderr,
)
assert sum(address_components(selected_boundary.stdout)) + reserve == 512
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


def fit_capacity(text):
    return run(
        [
            "--vrc-expose-host-backend-literal-uses",
            "--vrc-discover-host-rematerialization",
            "--vrc-fit-host-storage-capacity",
        ],
        text,
    )


fit_input = with_reserve(without_request).replace(
    "func.func @entry() attributes {",
    'func.func @entry() attributes {vrc.async_spill_slots = "sentinel", ',
    1,
)
fitted = fit_capacity(fit_input)
assert fitted.returncode == 0, (fitted.stdout, fitted.stderr)
fitted_repeat = fit_capacity(fit_input)
assert fitted_repeat.returncode == 0, (
    fitted_repeat.stdout,
    fitted_repeat.stderr,
)
assert fitted_repeat.stdout == fitted.stdout
assert fitted.stdout.count('"udonhl.literal_materialize"') == 2
assert 'vrc.async_spill_slots = "sentinel"' in fitted.stdout
fit_report = re.search(
    r"vrc\.host_storage_capacity_fit = \{([^}]*)\}", fitted.stdout
)
assert fit_report, fitted.stdout
assert (
    f"baseline_address_count = {candidate_addresses} : i64"
    in fit_report.group(1)
)
assert f"final_address_count = {selected_addresses} : i64" in fit_report.group(1)
assert "attempt_count = 2 : i32" in fit_report.group(1)
assert "requested_resident_slot_savings = 2 : i32" in fit_report.group(1)
fitted_lowered = run(
    ["--vrc-lower-udonhl-to-udonasm", "--verify-each"], fitted.stdout
)
assert fitted_lowered.returncode == 0, (
    fitted_lowered.stdout,
    fitted_lowered.stderr,
)
assert sum(address_components(fitted_lowered.stdout)) == selected_addresses

already_fit_input = fit_input.replace(
    f"vrc.final_data_slot_reserve = {reserve} : i32",
    f"vrc.final_data_slot_reserve = {reserve - 1} : i32",
    1,
)
already_fit = fit_capacity(already_fit_input)
assert already_fit.returncode == 0, (already_fit.stdout, already_fit.stderr)
assert '"udonhl.literal_materialize"' not in already_fit.stdout
already_fit_report = re.search(
    r"vrc\.host_storage_capacity_fit = \{([^}]*)\}", already_fit.stdout
)
assert already_fit_report, already_fit.stdout
assert "attempt_count = 0 : i32" in already_fit_report.group(1)
assert "requested_resident_slot_savings = 0 : i32" in already_fit_report.group(1)
assert "vrc.host_storage_pressure_request" not in already_fit.stdout

exhausted_input = fit_input.replace(
    f"vrc.final_data_slot_reserve = {reserve} : i32",
    f"vrc.final_data_slot_reserve = {reserve + 1} : i32",
    1,
)
exhausted = fit_capacity(exhausted_input)
assert exhausted.returncode != 0, (exhausted.stdout, exhausted.stderr)
assert "only 2 are safely selectable" in exhausted.stderr
assert "failed to prepare Host storage selection attempt" in exhausted.stderr

spill_input = r'''module attributes {vrc.host_stage = "udonhl"} {
  func.func @reuse(%a: i32, %b: i32, %c: i32) attributes {
    vrc.export_id = 72 : i32,
    vrc.async_spill_slots = [
      {value_id = 10 : i32, slot = 0 : i32, storage_key = "SystemInt32",
       placement = "typed_spill_array", interferes_with = array<i32>},
      {value_id = 2 : i32, slot = 1 : i32, storage_key = "SystemInt32",
       placement = "typed_spill_array", interferes_with = array<i32: 10>},
      {value_id = 7 : i32, slot = 2 : i32, storage_key = "SystemInt32",
       placement = "typed_spill_array", interferes_with = array<i32>}
    ]
  } {
    udonhl.spill_store %a[0] {storage_key = "SystemInt32",
      vrc.async_frame_store} : i32
    udonhl.spill_store %b[1] {storage_key = "SystemInt32",
      vrc.async_frame_store} : i32
    udonhl.spill_store %c[2] {storage_key = "SystemInt32",
      vrc.async_frame_store} : i32
    return
  }
}'''
spill_fitted = fit_capacity(spill_input)
assert spill_fitted.returncode == 0, (
    spill_fitted.stdout,
    spill_fitted.stderr,
)
assert spill_fitted.stdout.count("spill_store %arg0[0]") == 1
assert spill_fitted.stdout.count("spill_store %arg1[1]") == 1
assert spill_fitted.stdout.count("spill_store %arg2[2]") == 1
assert spill_fitted.stdout.count("slot = 0 : i32") == 1
assert spill_fitted.stdout.count("slot = 1 : i32") == 1
assert spill_fitted.stdout.count("slot = 2 : i32") == 1

missing_discovery = run(["--vrc-fit-host-storage-capacity"], without_request)
assert missing_discovery.returncode != 0
assert "requires pristine post-discovery IR" in missing_discovery.stderr

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
    "capacity_fit=2 retries=2 diagnostics=7"
)
