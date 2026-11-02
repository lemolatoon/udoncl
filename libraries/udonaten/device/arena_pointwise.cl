#ifdef __OPENCL_C_VERSION__
#define UDONATEN_ARENA_X() ((size_t)get_global_id(0))
#define UDONATEN_ARENA_Y() ((size_t)get_global_id(1))
#else
#include <math.h>
#include <stddef.h>
extern size_t udonaten_arena_pointwise_cpu_gid_x;
extern size_t udonaten_arena_pointwise_cpu_gid_y;
#define UDONATEN_ARENA_X() (udonaten_arena_pointwise_cpu_gid_x)
#define UDONATEN_ARENA_Y() (udonaten_arena_pointwise_cpu_gid_y)
#define __kernel
#define __global
#define exp expf
#endif

#define UDONATEN_ARENA_WIDTH 8192u

/* Private AOT backend for ATen-owned pointwise operators. Public host calls
 * remain udonatenAdd/udonatenMul/udonatenSigmoid; this kernel only removes the
 * 512-word packed-state limit after ONNX has selected the same ATen operation.
 */
__kernel void udonaten_arena_pointwise(__global const float *self,
                                       __global const float *other,
                                       __global float *output, int words,
                                       int self_words, int other_words,
                                       int operation) {
  size_t gid = UDONATEN_ARENA_Y() * UDONATEN_ARENA_WIDTH + UDONATEN_ARENA_X();
  float value = 0.0f;
  if (gid < (size_t)words) {
    size_t left_index = (gid % (size_t)self_words) & 16383u;
    size_t right_index = (gid % (size_t)other_words) & 16383u;
    float left = self[left_index];
    float right = other[right_index];
    if (operation == 0)
      value = left + right;
    else if (operation == 1)
      value = left * right;
    else if (operation == 2)
      value = 1.0f / (1.0f + exp(-left));
    else if (operation == 3)
      value = left - right;
    else if (operation == 4)
      value = left > 0.0f ? left : 0.0f;
    else if (operation == 5)
      value = 2.0f / (1.0f + exp(-2.0f * left)) - 1.0f;
  }
  output[gid] = value;
}
