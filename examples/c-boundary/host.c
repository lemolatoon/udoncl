// Compiler boundary probe: these are uint32 handles, never host pointers.
typedef unsigned int vrc_buffer;
typedef unsigned int vrc_dispatch_desc;
typedef unsigned int vrc_event;
extern vrc_event __vrc_dispatch(vrc_dispatch_desc desc, vrc_buffer buffer);

static __attribute__((noinline)) int add_impl(int a, int b) { return a + b; }
int add(int a, int b) { return add_impl(a, b); }

// The future C runtime supplies the copied descriptor and allocation handle.
vrc_event request_square(vrc_buffer buffer, vrc_dispatch_desc desc) {
  return __vrc_dispatch(desc, buffer);
}
