#include <udoncl/unity_buffer.h>

void copy_full(void) {
  udoncl_buffer source = udonclBufferAlloc(32772);
  udoncl_buffer destination = udonclBufferAlloc(32772);
  (void)udonclCopyBuffer(source, destination, 0, 0, 32772);
  (void)udonclCopyBuffer(source, destination, 0, 0, 32772);
}
