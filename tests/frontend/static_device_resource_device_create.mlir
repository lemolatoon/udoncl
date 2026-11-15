module {
  func.func private @udonclDeviceBufferCreate(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferGetRaw(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @CreateAndResolve(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %flags = arith.constant 1 : i64
    %bytes = arith.constant 96 : i32
    %handle_bytes = arith.constant 8 : i32
    %cell_bytes = arith.constant 4 : i32
    %handle = "udonhl.alloca"(%handle_bytes) {alignment = 4 : i64} : (i32) -> !udonhl.ptr<0>
    %cell = "udonhl.alloca"(%cell_bytes) {alignment = 4 : i64} : (i32) -> !udonhl.ptr<0>
    %created = func.call @udonclDeviceBufferCreate(%context, %flags, %bytes, %handle) {
      vrc.static_public_resource_binding = {capacity_bytes = 96 : i64, data_type = "float32", logical_bytes = 0 : i64, name = "values", public_access = "input", public_index = 0 : i32, resource_source = "call_allocation", schema_version = 1 : i32}
    } : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>) -> i32
    %resolved = func.call @udonclDeviceBufferGetRaw(%handle, %cell) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}
