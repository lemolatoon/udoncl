module attributes {
  udoncl.program = "static_tensor_table",
  udoncl.static_tensors = [{
    alignment = 4 : i64,
    byte_length = 4 : i64,
    byte_strides = array<i64: 4>,
    data = dense<1.0> : tensor<1xf32>,
    data_type = "float32",
    id = 0 : i32,
    layout = "row_major_contiguous",
    name = "weight",
    sha256 = "e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c",
    shape = array<i64: 1>,
    symbol = "__udoncl_static_tensor_0"
  }]
} {
  func.func private @static_buffer(i32, i32) -> !vrc.buffer attributes {
    vrc.intrinsic = "static_buffer"
  }
  func.func @read_static_tensor() -> i32 attributes {
    udoncl.export,
    vrc.async_effect = "may_suspend"
  } {
    %zero = arith.constant 0 : i32
    %pointer = arith.constant 4 : i32
    %bytes = arith.constant 4 : i32
    %weight = func.call @static_buffer(%zero, %bytes) : (i32, i32) -> !vrc.buffer
    %event = vrc.buffer_readback %weight, %zero, %bytes, %pointer : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %status = vrc.await %event : (!vrc.event) -> i32
    return %status : i32
  }
}
