#ifdef __OPENCL_C_VERSION__
#define UDONONNX_REINDEX_GID() ((size_t)get_global_id(0))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udononnx_reindex_cpu_gid;
#define UDONONNX_REINDEX_GID() (udononnx_reindex_cpu_gid)
#define __kernel
#define __global
#endif

#define UDONONNX_TRANSPOSE_DIMENSION(axis, shape01, shape23, shape4)           \
  (((((axis) < 2 ? (shape01) : ((axis) < 4 ? (shape23) : (shape4))) >>         \
     ((uint)((axis) & 1) * 16u)) &                                             \
    65535u)                                                                    \
       ? ((((axis) < 2 ? (shape01) : ((axis) < 4 ? (shape23) : (shape4))) >>   \
           ((uint)((axis) & 1) * 16u)) &                                       \
          65535u)                                                              \
       : 1u)

/* Generic contiguous transpose materialization. Shapes use five packed u16
 * dimensions and permutation uses one three-bit input-axis index per output
 * axis. The Host facade validates the complete descriptor before dispatch. */
__kernel void udononnx_transpose_f32(__global const float *input,
                                     __global float *output, int words,
                                     int rank, int shape01_value,
                                     int shape23_value, int shape4_value,
                                     int permutation_value) {
  size_t gid = UDONONNX_REINDEX_GID();
  uint shape01 = (uint)shape01_value;
  uint shape23 = (uint)shape23_value;
  uint shape4 = (uint)shape4_value;
  uint permutation = (uint)permutation_value;
  float value = 0.0f;
  if (gid < (size_t)words) {
    size_t input_index = 0u;
    size_t output_stride = 1u;
    for (int reverse_axis = 0; reverse_axis < 5; ++reverse_axis) {
      int output_axis = 4 - reverse_axis;
      if (output_axis < rank) {
        int input_axis = (int)((permutation >> (output_axis * 3)) & 7u);
        uint output_dimension =
            UDONONNX_TRANSPOSE_DIMENSION(input_axis, shape01, shape23, shape4);
        size_t coordinate = (gid / output_stride) % output_dimension;
        size_t input_stride = 1u;
        for (int candidate = 0; candidate < 5; ++candidate)
          if (candidate > input_axis && candidate < rank)
            input_stride *= UDONONNX_TRANSPOSE_DIMENSION(candidate, shape01,
                                                         shape23, shape4);
        input_index += coordinate * input_stride;
        output_stride *= output_dimension;
      }
    }
    value = input[input_index & 16383u];
  }
  output[gid] = value;
}
