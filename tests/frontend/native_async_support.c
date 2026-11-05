#include <stddef.h>
#include <stdlib.h>

#if defined(_WIN32)
__declspec(dllexport) void *aligned_alloc(size_t alignment, size_t size) {
  (void)alignment;
  return malloc(size);
}
#endif
