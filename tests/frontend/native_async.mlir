module {
  func.func @add(%left: i32, %right: i32) -> i32 {
    %result = arith.addi %left, %right : i32
    return %result : i32
  }

  func.func private @wait_and_add(%event: !vrc.event, %value: i32) -> i32
      attributes {vrc.async_effect = "may_suspend"} {
    %status = vrc.await %event : (!vrc.event) -> i32
    %result = arith.addi %value, %status : i32
    return %result : i32
  }

  func.func @run(%event: !vrc.event, %value: i32) -> i32 {
    %result = func.call @wait_and_add(%event, %value)
        : (!vrc.event, i32) -> i32
    return %result : i32
  }

  func.func @run_sequential(
      %first_event: !vrc.event,
      %second_event: !vrc.event,
      %between: !async.value<i32>,
      %value: i32) -> i32 attributes {vrc.async_effect = "may_suspend"} {
    %first = vrc.await %first_event : (!vrc.event) -> i32
    "async.runtime.store"(%first, %between)
        : (i32, !async.value<i32>) -> ()
    "async.runtime.set_available"(%between) : (!async.value<i32>) -> ()
    %second = vrc.await %second_event : (!vrc.event) -> i32
    %partial = arith.addi %value, %first : i32
    %result = arith.addi %partial, %second : i32
    return %result : i32
  }
}
