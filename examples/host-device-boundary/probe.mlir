module {
  // U# entry ABI: a scalar C library export, not an automatic main().
  func.func @add(%a: i32, %b: i32) -> i32 attributes {
    vrc.export_id = 1 : i32,
    vrc.public_name = "Add",
    vrc.parameter_names = ["left", "right"]
  } {
    %result = func.call @add_impl(%a, %b) : (i32, i32) -> i32
    return %result : i32
  }
  func.func private @add_impl(%a: i32, %b: i32) -> i32 {
    %result = arith.addi %a, %b : i32
    return %result : i32
  }
  // Public primitive arrays share the same Udon array object with generated U#.
  func.func @TouchArrays(%values: memref<4xf32>, %indices: memref<2xi32>) attributes {
    vrc.export_id = 2 : i32,
    vrc.public_name = "TouchArrays",
    vrc.parameter_names = ["values", "indices"]
  } {
    return
  }
  // Internal __vrc_* boundary probe. This is not clEnqueueNDRangeKernel itself;
  // the C OpenCL runtime will own handles, copied arguments, queue and errors.
  func.func private @request_square(%buffer: !vrc.buffer, %desc: !vrc.dispatch_desc) -> !vrc.event {
    %event = func.call @enqueue_square(%buffer, %desc) : (!vrc.buffer, !vrc.dispatch_desc) -> !vrc.event
    return %event : !vrc.event
  }
  func.func private @enqueue_square(%buffer: !vrc.buffer, %desc: !vrc.dispatch_desc) -> !vrc.event {
    %event = func.call @__vrc_dispatch(%desc, %buffer) : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return %event : !vrc.event
  }
  func.func private @__vrc_dispatch(!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event attributes {
    vrc.intrinsic = "dispatch", vrc.kernel_id = 7 : i32,
    vrc.argument_kinds = ["buffer", "f32"],
    vrc.global_size = array<i32: 1, 1, 1>, vrc.local_size = array<i32: 1, 1, 1>,
    vrc.global_offset = array<i32: 0, 0, 0>
  }
  gpu.module @device {
    gpu.func @square(%output: memref<1xf32>, %x: f32) kernel attributes {
      vrc.kernel_id = 7 : i32,
      known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 1, 1, 1>
    } {
      %index = arith.constant 0 : index
      %value = arith.mulf %x, %x : f32
      memref.store %value, %output[%index] : memref<1xf32>
      gpu.return
    }
  }
}
