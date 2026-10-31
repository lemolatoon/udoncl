define i32 @scalar_ops(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  %mix = xor i32 %sum, %a
  %shifted = lshr i32 %mix, 3
  %small = trunc i32 %shifted to i16
  %extended = zext i16 %small to i32
  %choose = icmp ult i32 %a, %b
  %result = select i1 %choose, i32 %extended, i32 %sum
  ret i32 %result
}
define float @float_ops(float %a, float %b) {
  %product = fmul float %a, %b
  %sum = fadd float %product, 5.000000e-01
  %negative = fneg float %sum
  ret float %negative
}
define i32 @checked_inc(i32 %a) {
  %result = add nsw nuw i32 %a, 1
  ret i32 %result
}
define i32 @unsigned_divrem(i32 %a, i32 %b) {
  %quotient = udiv i32 %a, %b
  %remainder = urem i32 %a, %b
  %result = add i32 %quotient, %remainder
  ret i32 %result
}
