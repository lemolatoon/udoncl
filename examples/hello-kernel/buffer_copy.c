#include <udoncl/unity_buffer.h>

void copy_full(void) {
  udoncl_buffer source = udonclBufferAlloc(4);
  udoncl_buffer destination = udonclBufferAlloc(4);
  (void)udonclCopyBuffer(source, destination, 0, 0, 4);
}
