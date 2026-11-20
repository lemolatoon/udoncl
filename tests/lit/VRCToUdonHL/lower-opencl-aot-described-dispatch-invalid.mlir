// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-lower-opencl-aot-intrinsics %s

// vrc.device_dispatch must fully describe the launch and agree with the
// declaration that carries it.

// expected-error @below {{invalid vrc.device_dispatch: requires {kernel_id : i32 > 0, buffer_count : i32, global_size : array<i32>}}}
func.func private @not_a_dictionary(i32, i32) -> i32 attributes {vrc.device_dispatch = 5 : i32}

// -----

// expected-error @below {{invalid vrc.device_dispatch: requires {kernel_id : i32 > 0, buffer_count : i32, global_size : array<i32>}}}
func.func private @missing_global_size(i32, i32) -> i32 attributes {vrc.device_dispatch = {kernel_id = 5 : i32, buffer_count = 2 : i32}}

// -----

// expected-error @below {{invalid vrc.device_dispatch: requires {kernel_id : i32 > 0, buffer_count : i32, global_size : array<i32>}}}
func.func private @zero_kernel_id(i32, i32) -> i32 attributes {vrc.device_dispatch = {kernel_id = 0 : i32, buffer_count = 2 : i32, global_size = array<i32: 4, 1, 1>}}

// -----

// expected-error @below {{invalid vrc.device_dispatch: global_size must hold three positive extents}}
func.func private @flat_global_size(i32, i32) -> i32 attributes {vrc.device_dispatch = {kernel_id = 5 : i32, buffer_count = 2 : i32, global_size = array<i32: 4, 0, 1>}}

// -----

// expected-error @below {{invalid vrc.device_dispatch: fragment_cost must be a dictionary}}
func.func private @scalar_cost(i32, i32) -> i32 attributes {vrc.device_dispatch = {kernel_id = 5 : i32, buffer_count = 2 : i32, global_size = array<i32: 4, 1, 1>, fragment_cost = 3 : i64}}

// -----

// expected-error @below {{invalid vrc.device_dispatch: unknown entry}}
func.func private @unknown_entry(i32, i32) -> i32 attributes {vrc.device_dispatch = {kernel_id = 5 : i32, buffer_count = 2 : i32, global_size = array<i32: 4, 1, 1>, local_size = array<i32: 1, 1, 1>}}

// -----

// expected-error @below {{invalid vrc.device_dispatch: requires an external declaration returning i32}}
func.func private @wrong_result(i32, i32) -> f32 attributes {vrc.device_dispatch = {kernel_id = 5 : i32, buffer_count = 2 : i32, global_size = array<i32: 4, 1, 1>}}

// -----

// expected-error @below {{invalid vrc.device_dispatch: requires an external declaration returning i32}}
func.func private @defined(%a: i32, %b: i32) -> i32 attributes {vrc.device_dispatch = {kernel_id = 5 : i32, buffer_count = 2 : i32, global_size = array<i32: 4, 1, 1>}} {
  return %a : i32
}

// -----

// expected-error @below {{invalid vrc.device_dispatch: buffer_count must be between 1 and the number of arguments}}
func.func private @too_many_buffers(i32, i32) -> i32 attributes {vrc.device_dispatch = {kernel_id = 5 : i32, buffer_count = 3 : i32, global_size = array<i32: 4, 1, 1>}}

// -----

// expected-error @below {{invalid vrc.device_dispatch: buffer argument 1 must be an i32 token}}
func.func private @float_buffer(i32, f32) -> i32 attributes {vrc.device_dispatch = {kernel_id = 5 : i32, buffer_count = 2 : i32, global_size = array<i32: 4, 1, 1>}}

// -----

// expected-error @below {{invalid vrc.device_dispatch: scalar argument 2 must be i32 or f32}}
func.func private @wide_scalar(i32, i32, i64) -> i32 attributes {vrc.device_dispatch = {kernel_id = 5 : i32, buffer_count = 2 : i32, global_size = array<i32: 4, 1, 1>}}

// -----

// expected-error @below {{vrc.synthesized_dispatch must be resolved by vrc-assign-synthesized-kernel-ids first}}
func.func private @unresolved(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @carrier::@kernel}
