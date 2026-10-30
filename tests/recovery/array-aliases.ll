; Different opaque-pointer allocations recover different memref types.
; A one-past pointer is used only to step back into the same allocation.
define i32 @array_alias(i32 %value, i32 %selector) {
entry:
  %a = alloca [4 x i32], align 16
  %b = alloca i16, i32 2, align 2
  %a0 = getelementptr [4 x i32], ptr %a, i32 0, i32 0
  %a1 = getelementptr [4 x i32], ptr %a, i32 0, i32 1
  %a2 = getelementptr [4 x i32], ptr %a, i32 0, i32 2
  %end = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 4
  %a3 = getelementptr inbounds i32, ptr %end, i32 -1
  store i32 %value, ptr %a0, align 16
  %v1 = add i32 %value, 1
  %v2 = add i32 %value, 2
  %v3 = add i32 %value, 3
  store i32 %v1, ptr %a1, align 4
  store i32 %v2, ptr %a2, align 8
  store i32 %v3, ptr %a3, align 4
  %b1 = getelementptr i16, ptr %b, i32 1
  store i16 10, ptr %b, align 2
  store i16 20, ptr %b1, align 2
  %idx = and i32 %selector, 3
  %p = getelementptr inbounds i32, ptr %a0, i32 %idx
  %r = load i32, ptr %p, align 4
  %s = load i16, ptr %b1, align 2
  %w = zext i16 %s to i32
  %out = add i32 %r, %w
  ret i32 %out
}
; Flattened storage preserves LLVM's legal cross-subarray addressing.
define float @array_matrix(float %value) {
entry:
  %a = alloca [2 x [3 x float]], align 8
  %p = getelementptr inbounds [2 x [3 x float]], ptr %a, i32 0, i32 1, i32 2
  store float %value, ptr %p, align 4
  %q = getelementptr inbounds [2 x [3 x float]], ptr %a, i32 0, i32 0, i32 5
  %result = load float, ptr %q, align 4
  ret float %result
}
