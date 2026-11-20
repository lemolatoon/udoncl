// RUN: %vrc-opt --vrc-lower-opencl-aot-intrinsics %s | %FileCheck %s

// A dispatch declaration that carries vrc.device_dispatch describes its own
// kernel launch, so it is lowered without a name known to the pass: the
// first buffer_count i32 arguments are buffer tokens in kernel argument
// order with the last one written, and the declared types of the remaining
// arguments are the scalar kinds. The result is the form a named library
// intrinsic produces, shown here by the embedding kernel.

// CHECK-LABEL: func.func @named(
// CHECK: %[[DESC:.+]] = vrc.dispatch_desc_create %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} {kernel_id = 207 : i32, scalar_kinds = ["i32", "i32", "f32", "i32", "i32", "i32"]} : (i32, i32, f32, i32, i32, i32) -> !vrc.dispatch_desc
// CHECK: %[[IN:.+]] = builtin.unrealized_conversion_cast %{{.+}} : i32 to !vrc.buffer
// CHECK: %[[OUT:.+]] = builtin.unrealized_conversion_cast %{{.+}} : i32 to !vrc.buffer
// CHECK: %[[EVENT:.+]] = vrc.dispatch %[[DESC]](%[[IN]], %[[OUT]]) {argument_kinds = ["buffer", "buffer", "i32", "i32", "f32", "i32", "i32", "i32"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1536, 1, 1>, kernel_id = 207 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["read", "write"]} : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
// CHECK: builtin.unrealized_conversion_cast %[[EVENT]] : !vrc.event to i32

// CHECK-LABEL: func.func @described(
// CHECK-SAME: %[[A:.+]]: i32, %[[B:.+]]: i32, %[[C:.+]]: i32, %[[N:.+]]: i32, %[[S:.+]]: f32
// CHECK: %[[DESC:.+]] = vrc.dispatch_desc_create %[[N]], %[[S]], %[[N]] {kernel_id = 4096 : i32, scalar_kinds = ["i32", "f32", "i32"]} : (i32, f32, i32) -> !vrc.dispatch_desc
// CHECK: %[[IN0:.+]] = builtin.unrealized_conversion_cast %[[A]] : i32 to !vrc.buffer
// CHECK: %[[IN1:.+]] = builtin.unrealized_conversion_cast %[[B]] : i32 to !vrc.buffer
// CHECK: %[[OUT:.+]] = builtin.unrealized_conversion_cast %[[C]] : i32 to !vrc.buffer
// CHECK: %[[EVENT:.+]] = vrc.dispatch %[[DESC]](%[[IN0]], %[[IN1]], %[[OUT]]) {argument_kinds = ["buffer", "buffer", "buffer", "i32", "f32", "i32"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 8192, 16, 1>, kernel_id = 4096 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["read", "read", "write"]} : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer, !vrc.buffer) -> !vrc.event
// CHECK: %[[TOKEN:.+]] = builtin.unrealized_conversion_cast %[[EVENT]] : !vrc.event to i32
// CHECK: return %[[TOKEN]]

// A declared fragment cost reaches the dispatch for the Device work
// scheduler, and a kernel without scalars has an empty descriptor.
// CHECK-LABEL: func.func @costed(
// CHECK: vrc.dispatch_desc_create {{ *}}{kernel_id = 7 : i32, scalar_kinds = []} : () -> !vrc.dispatch_desc
// CHECK: vrc.dispatch %{{.+}}(%{{.+}}) {argument_kinds = ["buffer"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 4, 1, 1>, kernel_id = 7 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"], vrc.fragment_cost = {active = [{index = 1 : i64, max = 4 : i64}], loads = 3 : i64}}

// CHECK-NOT: vrc.device_dispatch
module {
  func.func private @__vrc_opencl_dispatch_udonaten_embedding(i32, i32, i32, i32, f32, i32, i32, i32) -> i32
  func.func private @any_name(i32, i32, i32, i32, f32, i32) -> i32 attributes {
    vrc.device_dispatch = {kernel_id = 4096 : i32, buffer_count = 3 : i32, global_size = array<i32: 8192, 16, 1>}
  }
  func.func private @fill(i32) -> i32 attributes {
    vrc.device_dispatch = {kernel_id = 7 : i32, buffer_count = 1 : i32, global_size = array<i32: 4, 1, 1>, fragment_cost = {loads = 3 : i64, active = [{index = 1 : i64, max = 4 : i64}]}}
  }
  func.func @named(%in: i32, %out: i32, %n: i32, %s: f32) -> i32 {
    %event = func.call @__vrc_opencl_dispatch_udonaten_embedding(%in, %out, %n, %n, %s, %n, %n, %n) : (i32, i32, i32, i32, f32, i32, i32, i32) -> i32
    return %event : i32
  }
  func.func @described(%a: i32, %b: i32, %c: i32, %n: i32, %s: f32) -> i32 {
    %event = func.call @any_name(%a, %b, %c, %n, %s, %n) : (i32, i32, i32, i32, f32, i32) -> i32
    return %event : i32
  }
  func.func @costed(%a: i32) -> i32 {
    %event = func.call @fill(%a) : (i32) -> i32
    return %event : i32
  }
}
