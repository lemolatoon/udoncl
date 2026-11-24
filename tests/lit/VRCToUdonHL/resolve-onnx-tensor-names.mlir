// RUN: %vrc-opt --split-input-file --vrc-resolve-onnx-tensor-names %s | %FileCheck %s

// A bind whose name is a constant string (or a constant offset into one)
// binds the graph input or output of that name by its index.
// CHECK-LABEL: func.func @Initialize
// CHECK: %[[INDEX0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: call @udononnxBindInputAt(%arg0, %[[INDEX0]], %arg1, %{{.*}}, %arg2, %arg2)
// CHECK: %[[INDEX1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: call @udononnxBindInputAt(%arg0, %[[INDEX1]], %arg1,
// CHECK: %[[OUTPUT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: call @udononnxBindOutputAt(%arg0, %[[OUTPUT0]], %arg1,
// CHECK-NOT: call @udononnxBindInput(
// CHECK-NOT: call @udononnxBindOutput(
// CHECK: func.func private @udononnxBindInputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
// CHECK: func.func private @udononnxBindOutputAt(
module attributes {
  udoncl.onnx_binding = {
    inputs = [{index = 0 : i32, name = "input_ids"}, {index = 1 : i32, name = "past_key_values.0.key"}],
    outputs = [{index = 0 : i32, name = "logits"}]
  }
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 1 : i64, constant, initial_value = array<i8: 105, 110, 112, 117, 116, 95, 105, 100, 115, 0>, linkage = "internal", storage_type = !llvm.array<10 x i8>, sym_name = "input_ids"}> : () -> ()
  // "xpast_key_values.0.key": the name starts one byte in.
  "udonhl.global"() <{address_space = 0 : i32, alignment = 1 : i64, constant, initial_value = array<i8: 120, 112, 97, 115, 116, 95, 107, 101, 121, 95, 118, 97, 108, 117, 101, 115, 46, 48, 46, 107, 101, 121, 0>, linkage = "internal", storage_type = !llvm.array<23 x i8>, sym_name = "past"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 1 : i64, constant, initial_value = array<i8: 108, 111, 103, 105, 116, 115, 0>, linkage = "internal", storage_type = !llvm.array<7 x i8>, sym_name = "logits"}> : () -> ()
  func.func private @udononnxBindInput(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udononnxBindOutput(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Initialize(%session: !udonhl.ptr<0>, %memory: !udonhl.ptr<0>, %dims: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %ids = udonhl.address_of @input_ids : <0>
    %0 = func.call @udononnxBindInput(%session, %ids, %memory, %c2, %dims, %dims) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %past_base = udonhl.address_of @past : <0>
    %past = "udonhl.ptr_add"(%past_base, %c1) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %1 = func.call @udononnxBindInput(%session, %past, %memory, %c2, %dims, %dims) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %logits = udonhl.address_of @logits : <0>
    %2 = func.call @udononnxBindOutput(%session, %logits, %memory, %c2, %dims, %dims) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// Without ONNX binding metadata nothing changes.
// CHECK-LABEL: func.func @Initialize
// CHECK: call @udononnxBindInput(
module {
  func.func private @udononnxBindInput(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Initialize(%session: !udonhl.ptr<0>, %c: i32) attributes {udoncl.export} {
    %0 = func.call @udononnxBindInput(%session, %session, %session, %c, %session, %session) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}
