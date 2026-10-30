; A PHI-based counted loop with a conditional inside the body.
; Entry parameters intentionally remain dynamic: CFG structuring must retain them.
define i32 @sum_even(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %sum = phi i32 [0, %entry], [%chosen, %latch]
  %more = icmp slt i32 %i, %n
  br i1 %more, label %body, label %exit
body:
  %bit = and i32 %i, 1
  %even = icmp eq i32 %bit, 0
  br i1 %even, label %take, label %skip
take:
  %added = add i32 %sum, %i
  br label %latch
skip:
  br label %latch
latch:
  %chosen = phi i32 [%added, %take], [%sum, %skip]
  %next = add nsw i32 %i, 1
  br label %loop
exit:
  ret i32 %sum
}
