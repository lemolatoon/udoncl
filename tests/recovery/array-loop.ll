; Fixed local array: initialize all elements, update them, and return their sum.
define i32 @array_sum(i32 %seed, i32 %delta) {
entry:
  %a = alloca [8 x i32], align 16
  br label %init
init:
  %i = phi i32 [0, %entry], [%inext, %init_body]
  %imore = icmp slt i32 %i, 8
  br i1 %imore, label %init_body, label %initialized
init_body:
  %wide_i = sext i32 %i to i64
  %p = getelementptr inbounds [8 x i32], ptr %a, i64 0, i64 %wide_i
  %value = add i32 %seed, %i
  store i32 %value, ptr %p, align 4
  %inext = add nsw i32 %i, 1
  br label %init
initialized:
  %base = getelementptr inbounds [8 x i32], ptr %a, i32 0, i32 0
  br label %sum
sum:
  %j = phi i32 [0, %initialized], [%jnext, %sum_body]
  %acc = phi i32 [0, %initialized], [%accnext, %sum_body]
  %jmore = icmp ult i32 %j, 8
  br i1 %jmore, label %sum_body, label %exit
sum_body:
  %wide_j = zext i32 %j to i64
  %q = getelementptr inbounds i32, ptr %base, i64 %wide_j
  %old = load i32, ptr %q, align 4
  %updated = add i32 %old, %delta
  store i32 %updated, ptr %q, align 4
  %accnext = add i32 %acc, %updated
  %jnext = add nuw nsw i32 %j, 1
  br label %sum
exit:
  ret i32 %acc
}
