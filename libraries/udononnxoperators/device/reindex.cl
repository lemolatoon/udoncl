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

#define UDONONNX_BROADCAST_INDEX(                                              \
    result, output_index, input_rank, input_shape01, input_shape23,            \
    input_shape4, output_rank, output_shape01, output_shape23, output_shape4)  \
  do {                                                                         \
    size_t broadcast_input_index = 0u;                                         \
    size_t broadcast_input_stride = 1u;                                        \
    size_t broadcast_output_stride = 1u;                                       \
    int broadcast_rank_offset = (output_rank) - (input_rank);                  \
    for (int broadcast_reverse_axis = 0; broadcast_reverse_axis < 5;           \
         ++broadcast_reverse_axis) {                                           \
      int broadcast_output_axis = 4 - broadcast_reverse_axis;                  \
      if (broadcast_output_axis < (output_rank)) {                             \
        uint broadcast_output_dimension =                                      \
            UDONONNX_REINDEX_U16(broadcast_output_axis, output_shape01,        \
                                 output_shape23, output_shape4);               \
        size_t broadcast_coordinate =                                          \
            ((output_index) / broadcast_output_stride) %                       \
            (size_t)broadcast_output_dimension;                                \
        int broadcast_input_axis =                                             \
            broadcast_output_axis - broadcast_rank_offset;                     \
        if (broadcast_input_axis >= 0) {                                       \
          uint broadcast_input_dimension =                                     \
              UDONONNX_REINDEX_U16(broadcast_input_axis, input_shape01,        \
                                   input_shape23, input_shape4);               \
          if (broadcast_input_dimension != 1u)                                 \
            broadcast_input_index +=                                           \
                broadcast_coordinate * broadcast_input_stride;                 \
          broadcast_input_stride *= broadcast_input_dimension;                 \
        }                                                                      \
        broadcast_output_stride *= broadcast_output_dimension;                 \
      }                                                                        \
    }                                                                          \
    (result) = broadcast_input_index;                                          \
  } while (0)

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

/* Generic contiguous two-input Concat. The Host facade folds the static shape
 * into the number of words inside one concat-axis element and both axis
 * extents, so the Device kernel does not need model-specific rank handling. */
__kernel void udononnx_concat2_f32(__global const float *left,
                                   __global const float *right,
                                   __global float *output, int words,
                                   int inner_words, int left_extent,
                                   int right_extent) {
  size_t gid = UDONONNX_REINDEX_GID();
  float value = 0.0f;
  if (gid < (size_t)words) {
    size_t left_block = (size_t)inner_words * (size_t)left_extent;
    size_t right_block = (size_t)inner_words * (size_t)right_extent;
    size_t output_block = left_block + right_block;
    size_t outer = gid / output_block;
    size_t offset = gid % output_block;
    if (offset < left_block)
      value = left[(outer * left_block + offset) & 16383u];
    else
      value = right[(outer * right_block + offset - left_block) & 16383u];
  }
  output[gid] = value;
}

/* Generic axis-zero Gather for immutable FP32 tables. Indices are raw
 * little-endian words so the same kernel accepts ONNX INT32 and INT64 without
 * requiring 64-bit shader arithmetic. Negative indices follow ONNX semantics;
 * out-of-range indices produce zero instead of reading outside the texture. */
__kernel void udononnx_gather_axis0_f32(__global const float *data,
                                        __global const uint *indices,
                                        __global float *output, int words,
                                        int inner_words, int axis_extent,
                                        int index_words) {
  size_t gid = UDONONNX_REINDEX_GID();
  float value = 0.0f;
  if (gid < (size_t)words) {
    size_t index_offset = (gid / (size_t)inner_words) * (size_t)index_words;
    uint low = indices[index_offset];
    uint high = index_words == 2 ? indices[index_offset + 1u] : 0u;
    int selected = (int)low;
    int representable = index_words == 1 || high == 0u || high == 0xffffffffu;
    if (representable && selected < 0)
      selected += axis_extent;
    int valid = representable && selected >= 0 && selected < axis_extent;
    uint safe_selected = (uint)selected % (uint)axis_extent;
    size_t source =
        (size_t)safe_selected * (size_t)inner_words + gid % (size_t)inner_words;
    float gathered = data[source];
    if (valid)
      value = gathered;
  }
  output[gid] = value;
}

/* ONNX INT64-to-FP32 conversion without requiring 64-bit fragment integers.
 * The two's-complement source is reconstructed from little-endian uint32
 * words and converted with the rounding behavior of FP32 arithmetic. */
__kernel void udononnx_cast_i64_f32(__global const uint *input,
                                    __global float *output, int words) {
  size_t gid = UDONONNX_REINDEX_GID();
  float value = 0.0f;
  if (gid < (size_t)words) {
    uint low = input[gid * 2u];
    uint high = input[gid * 2u + 1u];
    if ((high & 0x80000000u) == 0u)
      value = (float)high * 4294967296.0f + (float)low;
    else {
      uint magnitude_low = ~low + 1u;
      uint magnitude_high = ~high;
      if (magnitude_low == 0u)
        magnitude_high += 1u;
      value = -((float)magnitude_high * 4294967296.0f + (float)magnitude_low);
    }
  }
  output[gid] = value;
}

/* Boolean tensors use FP32 0/1 words in Device storage. This keeps fragment
 * shader resources word-addressable while preserving ONNX i1 in the graph. */
__kernel void
udononnx_equal_f32(__global const float *left, __global const float *right,
                   __global float *output, int words, int ranks_value,
                   int left_shape01_value, int left_shape23_value,
                   int right_shape01_value, int right_shape23_value,
                   int output_shape01_value, int output_shape23_value) {
  size_t gid = UDONONNX_REINDEX_GID();
  float value = 0.0f;
  if (gid < (size_t)words) {
    uint ranks = (uint)ranks_value;
    int left_rank = (int)(ranks & 7u);
    int right_rank = (int)((ranks >> 3u) & 7u);
    int output_rank = (int)((ranks >> 6u) & 7u);
    uint output_shape01 = (uint)output_shape01_value;
    uint output_shape23 = (uint)output_shape23_value;
    size_t left_index;
    size_t right_index;
    UDONONNX_BROADCAST_INDEX(left_index, gid, left_rank,
                             (uint)left_shape01_value, (uint)left_shape23_value,
                             1u, output_rank, output_shape01, output_shape23,
                             1u);
    UDONONNX_BROADCAST_INDEX(right_index, gid, right_rank,
                             (uint)right_shape01_value,
                             (uint)right_shape23_value, 1u, output_rank,
                             output_shape01, output_shape23, 1u);
    if (left[left_index & 16383u] == right[right_index & 16383u])
      value = 1.0f;
  }
  output[gid] = value;
}

__kernel void udononnx_where_f32(
    __global const float *condition, __global const float *when_true,
    __global const float *when_false, __global float *output, int words,
    int ranks_value, int condition_shape01_value, int condition_shape23_value,
    int true_shape01_value, int true_shape23_value, int false_shape01_value,
    int false_shape23_value, int output_shape01_value,
    int output_shape23_value) {
  size_t gid = UDONONNX_REINDEX_GID();
  float value = 0.0f;
  if (gid < (size_t)words) {
    uint ranks = (uint)ranks_value;
    int condition_rank = (int)(ranks & 7u);
    int true_rank = (int)((ranks >> 3u) & 7u);
    int false_rank = (int)((ranks >> 6u) & 7u);
    int output_rank = (int)((ranks >> 9u) & 7u);
    uint output_shape01 = (uint)output_shape01_value;
    uint output_shape23 = (uint)output_shape23_value;
    size_t condition_index;
    size_t true_index;
    size_t false_index;
    UDONONNX_BROADCAST_INDEX(condition_index, gid, condition_rank,
                             (uint)condition_shape01_value,
                             (uint)condition_shape23_value, 1u, output_rank,
                             output_shape01, output_shape23, 1u);
    UDONONNX_BROADCAST_INDEX(true_index, gid, true_rank,
                             (uint)true_shape01_value, (uint)true_shape23_value,
                             1u, output_rank, output_shape01, output_shape23,
                             1u);
    UDONONNX_BROADCAST_INDEX(false_index, gid, false_rank,
                             (uint)false_shape01_value,
                             (uint)false_shape23_value, 1u, output_rank,
                             output_shape01, output_shape23, 1u);
    value = when_false[false_index & 16383u];
    if (condition[condition_index & 16383u] != 0.0f)
      value = when_true[true_index & 16383u];
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
