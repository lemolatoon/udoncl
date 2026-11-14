module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "input"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "output"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "switch_zero"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "switch_default"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "loop_result"}> : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32

  func.func @bind(%is_output: i32, %resource: !udonhl.ptr<0>) {
    %zero = arith.constant 0 : i32
    %is_input = arith.cmpi eq, %is_output, %zero : i32
    cf.cond_br %is_input, ^input, ^output
  ^input:
    %input_address = udonhl.address_of @input : !udonhl.ptr<0>
    "udonhl.store"(%resource, %input_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  ^output:
    %output_address = udonhl.address_of @output : !udonhl.ptr<0>
    "udonhl.store"(%resource, %output_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }

  func.func @bind_switch(%selector: i32, %resource: !udonhl.ptr<0>) {
    cf.switch %selector : i32, [
      default: ^default,
      0: ^zero
    ]
  ^zero:
    %zero_address = udonhl.address_of @switch_zero : !udonhl.ptr<0>
    "udonhl.store"(%resource, %zero_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  ^default:
    %default_address = udonhl.address_of @switch_default : !udonhl.ptr<0>
    "udonhl.store"(%resource, %default_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }

  func.func @bind_after_loop(%limit: i32, %resource: !udonhl.ptr<0>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    cf.br ^loop(%zero : i32)
  ^loop(%index: i32):
    %continue = arith.cmpi ult, %index, %limit : i32
    cf.cond_br %continue, ^body, ^exit
  ^body:
    %next = arith.addi %index, %one : i32
    cf.br ^loop(%next : i32)
  ^exit:
    %address = udonhl.address_of @loop_result : !udonhl.ptr<0>
    "udonhl.store"(%resource, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }

  func.func @Initialize() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %input_resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %output_resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %switch_zero_resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %switch_default_resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %loop_resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    func.call @bind(%zero, %input_resource) : (i32, !udonhl.ptr<0>) -> ()
    func.call @bind(%one, %output_resource) : (i32, !udonhl.ptr<0>) -> ()
    func.call @bind_switch(%zero, %switch_zero_resource) : (i32, !udonhl.ptr<0>) -> ()
    func.call @bind_switch(%one, %switch_default_resource) : (i32, !udonhl.ptr<0>) -> ()
    func.call @bind_after_loop(%two, %loop_resource) : (i32, !udonhl.ptr<0>) -> ()
    return
  }

  func.func @Run() attributes {udoncl.export} {
    %input_address = udonhl.address_of @input : !udonhl.ptr<0>
    %output_address = udonhl.address_of @output : !udonhl.ptr<0>
    %switch_zero_address = udonhl.address_of @switch_zero : !udonhl.ptr<0>
    %switch_default_address = udonhl.address_of @switch_default : !udonhl.ptr<0>
    %loop_address = udonhl.address_of @loop_result : !udonhl.ptr<0>
    %input_resource = "udonhl.load"(%input_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %output_resource = "udonhl.load"(%output_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %switch_zero_resource = "udonhl.load"(%switch_zero_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %switch_default_resource = "udonhl.load"(%switch_default_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %loop_resource = "udonhl.load"(%loop_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %input_status = func.call @clReleaseMemObject(%input_resource) : (!udonhl.ptr<0>) -> i32
    %output_status = func.call @clReleaseMemObject(%output_resource) : (!udonhl.ptr<0>) -> i32
    %switch_zero_status = func.call @clReleaseMemObject(%switch_zero_resource) : (!udonhl.ptr<0>) -> i32
    %switch_default_status = func.call @clReleaseMemObject(%switch_default_resource) : (!udonhl.ptr<0>) -> i32
    %loop_status = func.call @clReleaseMemObject(%loop_resource) : (!udonhl.ptr<0>) -> i32
    return
  }
}
