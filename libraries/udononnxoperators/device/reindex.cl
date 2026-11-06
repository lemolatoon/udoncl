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

#define UDONONNX_REINDEX_PACKED(axis, values01, values23, value4)              \
  ((axis) < 2 ? (values01) : ((axis) < 4 ? (values23) : (value4)))
#define UDONONNX_REINDEX_U16(axis, values01, values23, value4)                 \
  ((UDONONNX_REINDEX_PACKED(axis, values01, values23, value4) >>               \
    ((uint)((axis) & 1) * 16u)) &                                              \
   65535u)

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

/* Generic static Slice materialization. The Host facade normalizes ONNX
 * starts/ends/axes/steps and validates that every generated coordinate is in
 * bounds. Normalized nonnegative starts and positive steps use packed u16
 * values. */
__kernel void udononnx_slice_f32(
    __global const float *input, __global float *output, int words, int rank,
    int input_shape01_value, int input_shape23_value, int input_shape4_value,
    int output_shape01_value, int output_shape23_value, int output_shape4_value,
    int starts01_value, int starts23_value, int starts4_value,
    int steps01_value, int steps23_value, int steps4_value) {
  size_t gid = UDONONNX_REINDEX_GID();
  uint input_shape01 = (uint)input_shape01_value;
  uint input_shape23 = (uint)input_shape23_value;
  uint input_shape4 = (uint)input_shape4_value;
  uint output_shape01 = (uint)output_shape01_value;
  uint output_shape23 = (uint)output_shape23_value;
  uint output_shape4 = (uint)output_shape4_value;
  uint starts01 = (uint)starts01_value;
  uint starts23 = (uint)starts23_value;
  uint starts4 = (uint)starts4_value;
  uint steps01 = (uint)steps01_value;
  uint steps23 = (uint)steps23_value;
  uint steps4 = (uint)steps4_value;
  float value = 0.0f;
  if (gid < (size_t)words) {
    size_t input_index = 0u;
    size_t output_stride = 1u;
    for (int reverse_axis = 0; reverse_axis < 5; ++reverse_axis) {
      int axis = 4 - reverse_axis;
      if (axis < rank) {
        uint output_dimension = UDONONNX_REINDEX_U16(
            axis, output_shape01, output_shape23, output_shape4);
        size_t coordinate = (gid / output_stride) % output_dimension;
        size_t input_coordinate =
            (size_t)UDONONNX_REINDEX_U16(axis, starts01, starts23, starts4) +
            coordinate *
                (size_t)UDONONNX_REINDEX_U16(axis, steps01, steps23, steps4);
        size_t input_stride = 1u;
        for (int candidate = 0; candidate < 5; ++candidate)
          if (candidate > axis && candidate < rank)
            input_stride *= UDONONNX_REINDEX_U16(candidate, input_shape01,
                                                 input_shape23, input_shape4);
        input_index += (size_t)input_coordinate * input_stride;
        output_stride *= output_dimension;
      }
    }
    value = input[input_index & 16383u];
  }
  output[gid] = value;
}

/* Generic static Expand materialization. Input dimensions are right-aligned
 * with the output dimensions; an input extent of one broadcasts that axis. */
__kernel void
udononnx_expand_f32(__global const float *input, __global float *output,
                    int words, int input_rank, int input_shape01_value,
                    int input_shape23_value, int input_shape4_value,
                    int output_rank, int output_shape01_value,
                    int output_shape23_value, int output_shape4_value) {
  size_t gid = UDONONNX_REINDEX_GID();
  uint input_shape01 = (uint)input_shape01_value;
  uint input_shape23 = (uint)input_shape23_value;
  uint input_shape4 = (uint)input_shape4_value;
  uint output_shape01 = (uint)output_shape01_value;
  uint output_shape23 = (uint)output_shape23_value;
  uint output_shape4 = (uint)output_shape4_value;
  float value = 0.0f;
  if (gid < (size_t)words) {
    size_t input_index = 0u;
    size_t input_stride = 1u;
    size_t output_stride = 1u;
    int rank_offset = output_rank - input_rank;
    for (int reverse_axis = 0; reverse_axis < 5; ++reverse_axis) {
      int output_axis = 4 - reverse_axis;
      if (output_axis < output_rank) {
        uint output_dimension = UDONONNX_REINDEX_U16(
            output_axis, output_shape01, output_shape23, output_shape4);
        size_t coordinate = (gid / output_stride) % output_dimension;
        int input_axis = output_axis - rank_offset;
        if (input_axis >= 0) {
          uint input_dimension = UDONONNX_REINDEX_U16(
              input_axis, input_shape01, input_shape23, input_shape4);
          if (input_dimension != 1u)
            input_index += coordinate * input_stride;
          input_stride *= input_dimension;
        }
        output_stride *= output_dimension;
      }
    }
    value = input[input_index & 16383u];
  }
  output[gid] = value;
}
