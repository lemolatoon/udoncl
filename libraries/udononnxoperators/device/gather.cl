#ifdef __OPENCL_C_VERSION__
#define UDONONNX_GATHER_GID() ((size_t)get_global_id(0))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udononnx_gather_cpu_gid;
#define UDONONNX_GATHER_GID() (udononnx_gather_cpu_gid)
#define __kernel
#define __global
#endif

/* Gather selects the source slice on the Host and copies each tile in
 * parallel. The float boundary matches Unity's gather-f32 shader ABI. */
__kernel void udononnx_gather_copy(__global const float *input,
                                   __global float *output, int words) {
  uint gid = (uint)UDONONNX_GATHER_GID();
  (void)words;
  output[gid] = input[gid];
}
