// RUN: %vrc-opt --vrc-assign-synthesized-kernel-ids --vrc-lower-opencl-aot-intrinsics --vrc-schedule-udon-device-work="fragment-weight=1" %s | %FileCheck %s

// The cost a synthesized kernel declares travels with its dispatch
// declaration to the dispatch, where the Device work scheduler charges it.
// @costed launches 8 fragments, of which the 3 selected by its count
// argument load 2 texels: 8 + 3 * 2 = 14. @plain declares nothing and is
// charged one load per read resource in every fragment: 8 + 8 = 16.

// CHECK: vrc.device_work_static_cost = {commands = 2 : i64, device = 30 : i64,
// CHECK-LABEL: func.func @run(
// CHECK: vrc.dispatch {{.*}}kernel_id = 1 : i32{{.*}}vrc.fragment_cost = {active = [{index = 2 : i64, max = 4 : i64}], loads = 2 : i64}
// CHECK: vrc.dispatch {{.*}}kernel_id = 2 : i32
// CHECK-NOT: vrc.fragment_cost
module {
  gpu.module @carrier {
    gpu.func @costed(%in: memref<4xf32>, %out: memref<8xf32>, %count: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 8, 1, 1>, vrc.fragment_cost = {active = [{index = 2 : i64, max = 4 : i64}], loads = 2 : i64}, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 8 : i32} {
      gpu.return
    }
    gpu.func @plain(%in: memref<4xf32>, %out: memref<8xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 8, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 8 : i32} {
      gpu.return
    }
  }
  func.func private @dispatch_costed(i32, i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @carrier::@costed}
  func.func private @dispatch_plain(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @carrier::@plain}
  func.func @run(%a: i32, %b: i32) -> i32 attributes {vrc.async_effect = "may_suspend"} {
    %count = arith.constant 3 : i32
    %first = func.call @dispatch_costed(%a, %b, %count) : (i32, i32, i32) -> i32
    %second = func.call @dispatch_plain(%b, %a) : (i32, i32) -> i32
    return %second : i32
  }
}
