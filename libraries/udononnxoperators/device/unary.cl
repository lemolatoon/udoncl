#ifdef __OPENCL_C_VERSION__
#define UDONONNX_UNARY_GID() ((size_t)get_global_id(0))
#else
#include <math.h>
#include <stddef.h>
typedef unsigned int uint;
extern size_t udononnx_unary_cpu_gid;
#define UDONONNX_UNARY_GID() (udononnx_unary_cpu_gid)
#define __kernel
#define __global
#endif

__kernel void udononnx_unary_f32(__global const float *input,
                                 __global float *output, int operation,
                                 int words) {
  uint gid = (uint)UDONONNX_UNARY_GID();
  float value = input[gid];
  if (gid < (uint)words) {
    if (operation == 1)
      value = cos(value);
    else if (operation == 2)
      value = sin(value);
    else if (operation == 3)
      value = -value;
  }
  output[gid] = value;
}
