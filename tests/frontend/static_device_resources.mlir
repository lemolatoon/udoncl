module {
  func.func private @clCreateBuffer() -> i32
  func.func private @udonclCreateBufferForQueue() -> i32
  func.func private @external_log()

  func.func @allocate() -> i32 {
    %buffer = func.call @clCreateBuffer() : () -> i32
    return %buffer : i32
  }

  func.func @non_resource_helper() {
    func.call @external_log() : () -> ()
    return
  }

  func.func @nested_allocate() -> i32 {
    %buffer = func.call @allocate() : () -> i32
    return %buffer : i32
  }

  func.func @unreachable_indirect() {
    %callee = func.constant @external_log : () -> ()
    func.call_indirect %callee() : () -> ()
    return
  }

  func.func @RunRepeat() attributes {vrc.export_id = 4 : i32} {
    %first = func.call @allocate() : () -> i32
    %second = func.call @allocate() : () -> i32
    return
  }

  func.func @RunDirect() attributes {vrc.export_id = 2 : i32} {
    %first = func.call @clCreateBuffer() : () -> i32
    func.call @non_resource_helper() : () -> ()
    %second = func.call @udonclCreateBufferForQueue() : () -> i32
    return
  }

  func.func @RunBranch(%condition: i1) attributes {vrc.export_id = 1 : i32} {
    cf.cond_br %condition, ^left, ^right
  ^left:
    %left_buffer = func.call @allocate() : () -> i32
    cf.br ^exit
  ^right:
    %right_buffer = func.call @allocate() : () -> i32
    cf.br ^exit
  ^exit:
    return
  }

  func.func @RunOther() attributes {vrc.export_id = 3 : i32} {
    %buffer = func.call @allocate() : () -> i32
    return
  }

  func.func @RunNested() attributes {udoncl.export, vrc.export_id = 5 : i32} {
    %buffer = func.call @nested_allocate() : () -> i32
    return
  }
}
