#ifdef __OPENCL_C_VERSION__
#define UDONONNX_GID() ((size_t)get_global_id(0))
#define UDONONNX_LINEAR_GID()                                                  \
  ((size_t)get_global_id(1) * 8192u + (size_t)get_global_id(0))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udononnx_matmul_nbits_cpu_gid;
#define UDONONNX_GID() (udononnx_matmul_nbits_cpu_gid)
#define UDONONNX_LINEAR_GID() (udononnx_matmul_nbits_cpu_gid)
#define __kernel
#define __global
#endif

#define UDONONNX_NBITS_SCALE_OFFSET 4096u
#define UDONONNX_NBITS_ZERO_OFFSET 8192u
#define UDONONNX_NBITS_INPUT_OFFSET 12288u
#define UDONONNX_NBITS_BIAS_OFFSET 12416u
#define UDONONNX_NBITS_RESULT_OFFSET 12544u

/* The Host facade losslessly expands packed integers to numeric q values,
 * while scale and zero selection remains explicit in the Device operation.
 * One fragment owns one output feature of an N tile. */
__kernel void udononnx_matmul_nbits(__global const float *state,
                                    __global float *result, int tile_n,
                                    int tile_k, int accumulate, int finalize) {
  uint gid = (uint)UDONONNX_GID();
  float value = state[gid & 16383u];
  if (gid >= UDONONNX_NBITS_RESULT_OFFSET &&
      gid < UDONONNX_NBITS_RESULT_OFFSET + (uint)tile_n) {
    uint row = gid - UDONONNX_NBITS_RESULT_OFFSET;
    uint base = row * 128u;
    float sum = accumulate != 0 ? state[gid & 16383u] : 0.0f;
    for (uint inner = 0; inner < 128u; ++inner)
      if (inner < (uint)tile_k) {
        uint index = base + inner;
        float weight = (state[index & 16383u] -
                        state[(UDONONNX_NBITS_ZERO_OFFSET + index) & 16383u]) *
                       state[(UDONONNX_NBITS_SCALE_OFFSET + index) & 16383u];
        sum += weight * state[(UDONONNX_NBITS_INPUT_OFFSET + inner) & 16383u];
      }
    if (finalize != 0)
      sum += state[(UDONONNX_NBITS_BIAS_OFFSET + row) & 16383u];
    value = sum;
  }
  result[gid] = value;
}

/* Device-resident Q4 tile used by the generic ONNX AOT path. Packed weights
 * remain little-endian uint32 words in their immutable texture. Every pass
 * rewrites the complete output surface so previous/output may be ping-ponged
 * safely across N and K tiles without an in-place Blit. */
__kernel void udononnx_matmul_nbits_q4(
    __global const float *activation, __global const uint *packed_weight,
    __global const float *scales, __global const float *previous,
    __global float *output, int m, int n, int k, int block_size, int n_base,
    int k_base, int tile_n, int tile_k, int accumulate) {
  uint gid = (uint)UDONONNX_LINEAR_GID();
  uint output_words = (uint)(m * n);
  float value = previous[gid];
  if (gid < output_words) {
    uint row = gid / (uint)n;
    uint column = gid % (uint)n;
    if (column >= (uint)n_base && column < (uint)(n_base + tile_n)) {
      uint blocks = ((uint)k + 31u) / 32u;
      float sum = accumulate != 0 ? previous[gid] : 0.0f;
      for (uint inner = 0; inner < 128u; ++inner) {
        uint inner_index = (uint)k_base + inner;
        if (inner < (uint)tile_k && inner_index < (uint)k) {
          uint block = inner_index / 32u;
          uint in_block = inner_index % 32u;
          uint word_index = (column * blocks + block) * 4u + in_block / 8u;
          uint shift = (in_block % 8u) * 4u;
          uint quantized = (packed_weight[word_index] >> shift) & 15u;
          float weight =
              ((float)quantized - 8.0f) * scales[column * blocks + block];
          sum += weight * activation[row * (uint)k + inner_index];
        }
      }
      value = sum;
    }
  }
  output[gid] = value;
}
