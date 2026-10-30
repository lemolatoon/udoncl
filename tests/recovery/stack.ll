; Non-escaping, initialized local storage may be promoted by upstream mem2reg.
define i32 @stack_add(i32 %value) {
  %slot = alloca i32, align 4
  store i32 %value, ptr %slot, align 4
  %loaded = load i32, ptr %slot, align 4
  %result = add nsw i32 %loaded, 3
  ret i32 %result
}
