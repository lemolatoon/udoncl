// ONNX Host graph of the fused kernel execution test (onnx_fusion.py). Host
// finalization turns each of its four results into one synthesized gather
// kernel, in this order:
//   rotated = scale * (x * cos + concat(-x[.., 32:64], x[.., 0:32]) * sin)
//   gated   = (x * sigmoid(x)) * y
//   keys    = scale * transpose(reshape(expand(k)))
//   mixed   = tanh(x) * relu(y) + sin(cos(x))
// onnx_fusion.cpp rebuilds every result from the unfused library kernels.

func.func private @clEnqueueMarkerWithWaitList(
    !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

func.func private @__udon_onnx_mul_row(
    tensor<1x4x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x4x1x64xf32>
    attributes {
      udon.onnx.operation = "Mul",
      udon.onnx.provider = "udonatenMul"
    }
func.func private @__udon_onnx_mul_scalar(
    tensor<1x4x1x64xf32>, tensor<f32>) -> tensor<1x4x1x64xf32>
    attributes {
      udon.onnx.operation = "Mul",
      udon.onnx.provider = "udonatenMul"
    }
func.func private @__udon_onnx_mul(
    tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
    attributes {
      udon.onnx.operation = "Mul",
      udon.onnx.provider = "udonatenMul"
    }
func.func private @__udon_onnx_add(
    tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }
func.func private @__udon_onnx_slice(
    tensor<1x4x1x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
    -> tensor<1x4x1x32xf32>
    attributes {
      udon.onnx.operation = "Slice",
      udon.onnx.provider = "udononnxSlice"
    }
func.func private @__udon_onnx_neg(tensor<1x4x1x32xf32>) -> tensor<1x4x1x32xf32>
    attributes {
      udon.onnx.operation = "Neg",
      udon.onnx.provider = "udononnxNeg"
    }
func.func private @__udon_onnx_concat(
    tensor<1x4x1x32xf32>, tensor<1x4x1x32xf32>) -> tensor<1x4x1x64xf32>
    attributes {
      udon.onnx.operation = "Concat",
      udon.onnx.provider = "udononnxConcat"
    }
func.func private @__udon_onnx_sigmoid(
    tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
    attributes {
      udon.onnx.operation = "Sigmoid",
      udon.onnx.provider = "udonatenSigmoid"
    }
func.func private @__udon_onnx_tanh(
    tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
    attributes {
      udon.onnx.operation = "Tanh",
      udon.onnx.provider = "udonatenTanh"
    }
func.func private @__udon_onnx_relu(
    tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
    attributes {
      udon.onnx.operation = "Relu",
      udon.onnx.provider = "udonatenRelu"
    }
func.func private @__udon_onnx_cos(
    tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
    attributes {
      udon.onnx.operation = "Cos",
      udon.onnx.provider = "udononnxCos"
    }
func.func private @__udon_onnx_sin(
    tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
    attributes {
      udon.onnx.operation = "Sin",
      udon.onnx.provider = "udononnxSin"
    }
func.func private @__udon_onnx_expand(
    tensor<1x1x4x8xf32>, tensor<4xi64>) -> tensor<1x3x4x8xf32>
    attributes {
      udon.onnx.operation = "Expand",
      udon.onnx.provider = "udononnxExpand"
    }
func.func private @__udon_onnx_reshape(
    tensor<1x3x4x8xf32>, tensor<3xi64>) -> tensor<1x12x8xf32>
    attributes {
      udon.onnx.operation = "Reshape",
      udon.onnx.provider = "udononnxReshape"
    }
func.func private @__udon_onnx_transpose(
    tensor<1x12x8xf32>) -> tensor<1x8x12xf32>
    attributes {
      udon.onnx.operation = "Transpose",
      udon.onnx.provider = "udononnxTranspose"
    }
func.func private @__udon_onnx_mul_keys(
    tensor<1x8x12xf32>, tensor<f32>) -> tensor<1x8x12xf32>
    attributes {
      udon.onnx.operation = "Mul",
      udon.onnx.provider = "udonatenMul"
    }

func.func private @__static_scale() -> tensor<f32>
    attributes {
      udoncl.static_tensor = {
        byte_length = 4 : i64, id = 0 : i64,
        data = dense<3.000000e-01> : tensor<f32>
      }
    }
func.func private @__static_zero() -> tensor<1xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 8 : i64, id = 1 : i64, data = dense<0> : tensor<1xi64>
      }
    }
func.func private @__static_half() -> tensor<1xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 8 : i64, id = 2 : i64, data = dense<32> : tensor<1xi64>
      }
    }
func.func private @__static_end() -> tensor<1xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 8 : i64, id = 3 : i64, data = dense<64> : tensor<1xi64>
      }
    }
func.func private @__static_axis() -> tensor<1xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 8 : i64, id = 4 : i64, data = dense<3> : tensor<1xi64>
      }
    }
func.func private @__static_expand_shape() -> tensor<4xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 32 : i64, id = 5 : i64,
        data = dense<[1, 3, 4, 8]> : tensor<4xi64>
      }
    }
func.func private @__static_keys_shape() -> tensor<3xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 24 : i64, id = 6 : i64,
        data = dense<[1, 12, 8]> : tensor<3xi64>
      }
    }

func.func @main_graph(
    %x: tensor<1x4x1x64xf32> {onnx.name = "x"},
    %y: tensor<1x4x1x64xf32> {onnx.name = "y"},
    %cos: tensor<1x1x1x64xf32> {onnx.name = "cos"},
    %sin: tensor<1x1x1x64xf32> {onnx.name = "sin"},
    %k: tensor<1x1x4x8xf32> {onnx.name = "k"})
    -> (tensor<1x4x1x64xf32> {onnx.name = "rotated"},
        tensor<1x4x1x64xf32> {onnx.name = "gated"},
        tensor<1x8x12xf32> {onnx.name = "keys"},
        tensor<1x4x1x64xf32> {onnx.name = "mixed"}) {
  %scale = func.call @__static_scale() : () -> tensor<f32>
  %zero = func.call @__static_zero() : () -> tensor<1xi64>
  %half = func.call @__static_half() : () -> tensor<1xi64>
  %end = func.call @__static_end() : () -> tensor<1xi64>
  %axis = func.call @__static_axis() : () -> tensor<1xi64>
  %expand_shape = func.call @__static_expand_shape() : () -> tensor<4xi64>
  %keys_shape = func.call @__static_keys_shape() : () -> tensor<3xi64>

  %x_cos = func.call @__udon_onnx_mul_row(%x, %cos)
      : (tensor<1x4x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x4x1x64xf32>
  %high = func.call @__udon_onnx_slice(%x, %half, %end, %axis)
      : (tensor<1x4x1x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<1x4x1x32xf32>
  %low = func.call @__udon_onnx_slice(%x, %zero, %half, %axis)
      : (tensor<1x4x1x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<1x4x1x32xf32>
  %negated = func.call @__udon_onnx_neg(%high)
      : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x32xf32>
  %turned = func.call @__udon_onnx_concat(%negated, %low)
      {udon.onnx.attributes = {axis = -1 : i64}}
      : (tensor<1x4x1x32xf32>, tensor<1x4x1x32xf32>) -> tensor<1x4x1x64xf32>
  %turned_sin = func.call @__udon_onnx_mul_row(%turned, %sin)
      : (tensor<1x4x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x4x1x64xf32>
  %sum = func.call @__udon_onnx_add(%x_cos, %turned_sin)
      : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
  %rotated = func.call @__udon_onnx_mul_scalar(%sum, %scale)
      : (tensor<1x4x1x64xf32>, tensor<f32>) -> tensor<1x4x1x64xf32>

  %gate = func.call @__udon_onnx_sigmoid(%x)
      : (tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
  %activated = func.call @__udon_onnx_mul(%x, %gate)
      : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
  %gated = func.call @__udon_onnx_mul(%activated, %y)
      : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>

  %expanded = func.call @__udon_onnx_expand(%k, %expand_shape)
      : (tensor<1x1x4x8xf32>, tensor<4xi64>) -> tensor<1x3x4x8xf32>
  %merged = func.call @__udon_onnx_reshape(%expanded, %keys_shape)
      : (tensor<1x3x4x8xf32>, tensor<3xi64>) -> tensor<1x12x8xf32>
  %transposed = func.call @__udon_onnx_transpose(%merged)
      {udon.onnx.attributes = {perm = [0, 2, 1]}}
      : (tensor<1x12x8xf32>) -> tensor<1x8x12xf32>
  %keys = func.call @__udon_onnx_mul_keys(%transposed, %scale)
      : (tensor<1x8x12xf32>, tensor<f32>) -> tensor<1x8x12xf32>

  %squashed = func.call @__udon_onnx_tanh(%x)
      : (tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
  %rectified = func.call @__udon_onnx_relu(%y)
      : (tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
  %product = func.call @__udon_onnx_mul(%squashed, %rectified)
      : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
  %cosine = func.call @__udon_onnx_cos(%x)
      : (tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
  %wave = func.call @__udon_onnx_sin(%cosine)
      : (tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>
  %mixed = func.call @__udon_onnx_add(%product, %wave)
      : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf32>

  return %rotated, %gated, %keys, %mixed
      : tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>, tensor<1x8x12xf32>,
        tensor<1x4x1x64xf32>
}
