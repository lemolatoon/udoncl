// RUN: %vrc-opt --vrc-privatize-onnx-binding-resources %s | %FileCheck %s

// The buffers a generated ONNX binding allocates for its graph inputs and
// outputs lose their public binding; everything else stays.

// CHECK-LABEL: func.func @create
// CHECK: call @udononnxBindingStateAllocateBuffer({{.*}}) {vrc.keep} :
module {
  func.func private @udononnxBindingStateAllocateBuffer(i32, i32, i32, i32) -> i32
  func.func @create(%state: i32, %handle: i32) -> i32 {
    %access = arith.constant 0 : i32
    %bytes = arith.constant 64 : i32
    %status = func.call @udononnxBindingStateAllocateBuffer(%state, %access, %bytes, %handle) {vrc.keep, vrc.static_public_resource_binding = {access = "input", index = 0 : i32, tensor = 0 : i32}} : (i32, i32, i32, i32) -> i32
    return %status : i32
  }
}
