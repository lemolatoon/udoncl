#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/unity_buffer.h>

_Static_assert(sizeof(cl_mem) == sizeof(udoncl_buffer),
               "UdonCL Host input must use the 32-bit OpenCL handle ABI");
_Static_assert(CL_MEM_READ_WRITE == 1, "unexpected OpenCL 1.2 headers");

void copy_full(void) {
  udoncl_buffer source = udonclBufferAlloc(32772);
  udoncl_buffer intermediate = udonclBufferAlloc(32772);
  udoncl_buffer destination = udonclBufferAlloc(32772);
  (void)udonclCopyBuffer(source, intermediate, 0, 0, 32772);
  (void)udonclCopyBuffer(intermediate, destination, 0, 0, 32772);
}
