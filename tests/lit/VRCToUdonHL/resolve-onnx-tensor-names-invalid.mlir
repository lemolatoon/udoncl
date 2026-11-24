// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-resolve-onnx-tensor-names %s

// A name only known at run time cannot select a session slot.
module attributes {udoncl.onnx_binding = {inputs = [{index = 0 : i32, name = "x"}], outputs = []}} {
  func.func private @udononnxBindInput(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Initialize(%session: !udonhl.ptr<0>, %name: !udonhl.ptr<0>, %c: i32) attributes {udoncl.export} {
    // expected-error @below {{udononnxBindInput requires a tensor name known at compile time}}
    %0 = func.call @udononnxBindInput(%session, %name, %session, %c, %session, %session) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A string the program may write is not a constant name.
module attributes {udoncl.onnx_binding = {inputs = [{index = 0 : i32, name = "x"}], outputs = []}} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 1 : i64, initial_value = array<i8: 120, 0>, linkage = "internal", storage_type = !llvm.array<2 x i8>, sym_name = "name"}> : () -> ()
  func.func private @udononnxBindInput(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Initialize(%session: !udonhl.ptr<0>, %c: i32) attributes {udoncl.export} {
    %name = udonhl.address_of @name : <0>
    // expected-error @below {{udononnxBindInput requires a tensor name known at compile time}}
    %0 = func.call @udononnxBindInput(%session, %name, %session, %c, %session, %session) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// An unknown name lists the graph inputs.
module attributes {udoncl.onnx_binding = {inputs = [{index = 0 : i32, name = "x"}, {index = 1 : i32, name = "y"}], outputs = []}} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 1 : i64, constant, initial_value = array<i8: 122, 0>, linkage = "internal", storage_type = !llvm.array<2 x i8>, sym_name = "name"}> : () -> ()
  func.func private @udononnxBindInput(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Initialize(%session: !udonhl.ptr<0>, %c: i32) attributes {udoncl.export} {
    %name = udonhl.address_of @name : <0>
    // expected-error @below {{udononnxBindInput: the model has no graph input "z" (graph inputs: "x", "y")}}
    %0 = func.call @udononnxBindInput(%session, %name, %session, %c, %session, %session) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A graph input bound as an output says which bind it needs.
module attributes {udoncl.onnx_binding = {inputs = [{index = 0 : i32, name = "x"}], outputs = [{index = 0 : i32, name = "y"}]}} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 1 : i64, constant, initial_value = array<i8: 120, 0>, linkage = "internal", storage_type = !llvm.array<2 x i8>, sym_name = "name"}> : () -> ()
  func.func private @udononnxBindOutput(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Initialize(%session: !udonhl.ptr<0>, %c: i32) attributes {udoncl.export} {
    %name = udonhl.address_of @name : <0>
    // expected-error @below {{udononnxBindOutput: the model has no graph output "x"; it is a graph input, bound with udononnxBindInput}}
    %0 = func.call @udononnxBindOutput(%session, %name, %session, %c, %session, %session) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A name without its terminating NUL is not a string.
module attributes {udoncl.onnx_binding = {inputs = [{index = 0 : i32, name = "x"}], outputs = []}} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 1 : i64, constant, initial_value = array<i8: 120>, linkage = "internal", storage_type = !llvm.array<1 x i8>, sym_name = "name"}> : () -> ()
  func.func private @udononnxBindInput(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Initialize(%session: !udonhl.ptr<0>, %c: i32) attributes {udoncl.export} {
    %name = udonhl.address_of @name : <0>
    // expected-error @below {{udononnxBindInput requires a tensor name known at compile time}}
    %0 = func.call @udononnxBindInput(%session, %name, %session, %c, %session, %session) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}
