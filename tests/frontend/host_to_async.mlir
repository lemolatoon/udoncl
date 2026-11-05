module {
  func.func private @wait_leaf(%event: !vrc.event) attributes {
    vrc.async_effect = "may_suspend"
  } {
    %status = vrc.await %event : (!vrc.event) -> i32
    return
  }

  func.func private @wait_child(%event: !vrc.event, %value: i32) -> i32 attributes {
    vrc.async_effect = "may_suspend"
  } {
    func.call @wait_leaf(%event) : (!vrc.event) -> ()
    %status = vrc.await %event : (!vrc.event) -> i32
    %result = arith.addi %value, %status : i32
    return %result : i32
  }

  func.func private @wait_middle(%event: !vrc.event, %value: i32) -> i32 {
    %result = func.call @wait_child(%event, %value)
      : (!vrc.event, i32) -> i32
    return %result : i32
  }

  func.func private @wait_colored(%event: !vrc.event, %value: i32) -> i32 attributes {
    vrc.async_effect = "may_suspend"
  } {
    %first = vrc.await %event : (!vrc.event) -> i32
    %between = arith.addi %value, %first : i32
    %second = vrc.await %event : (!vrc.event) -> i32
    %result = arith.addi %between, %second : i32
    return %result : i32
  }

  func.func private @wait_i64(%event: !vrc.event, %value: i64) -> i64 attributes {
    vrc.async_effect = "may_suspend"
  } {
    %status = vrc.await %event : (!vrc.event) -> i32
    return %value : i64
  }

  func.func @run(%event: !vrc.event, %value: i32) -> i32 attributes {
    vrc.export_id = 1 : i32,
    vrc.parameter_names = ["event", "value"],
    vrc.public_name = "Run"
  } {
    %result = func.call @wait_middle(%event, %value)
      : (!vrc.event, i32) -> i32
    return %result : i32
  }
}
