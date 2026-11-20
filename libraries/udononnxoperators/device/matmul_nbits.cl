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

/* Device-resident Q4 (bits=4, block_size=32, implicit zero point 8) used by
 * the generic ONNX AOT path. Packed weights remain little-endian uint32 words
 * in their immutable texture.
 *
 * Split-K schedule. A dispatch writes `splits` planes of m x n words; plane s
 * starts at word s * m * n, so fragment gid = (s * m + row) * n + column owns
 * output word (row, column) of plane s. That fragment
 *  1. adds the `previous_partials` planes of previous at its (row, column),
 *     in ascending plane order, and then
 *  2. accumulates its K slice, the blocks [s * block_count,
 *     min((s + 1) * block_count, ceil(k / 32))), strictly ascending in k.
 * The Host runs a long reduction as one partial dispatch (splits = S,
 * previous_partials = 0) into a scratch surface and one ordered sum dispatch
 * (splits = 1, block_count = 0, previous_partials = S) into the output, so
 * the summation order is fixed at compile time and repeated runs are bit
 * identical. With splits = 1, block_count = ceil(k / 32) and
 * previous_partials = 0 one dispatch computes the serial whole-K sum.
 *
 * The Device lowering accepts only constant loop bounds, so the block loop
 * runs a constant number of chunks of UDONONNX_Q4_CHUNK_BLOCKS blocks and
 * guards the tail: a short slice skips whole chunks at the cost of one
 * compare. block_count is at most UDONONNX_Q4_DISPATCH_BLOCKS and
 * previous_partials at most UDONONNX_Q4_MAX_SPLITS. Fragments outside the
 * splits x m x n planes write 0 without any Load, and every dispatch rewrites
 * the complete output surface. */
#define UDONONNX_Q4_CHUNK_BLOCKS 16u
#define UDONONNX_Q4_DISPATCH_CHUNKS 16u
#define UDONONNX_Q4_DISPATCH_BLOCKS                                            \
  (UDONONNX_Q4_CHUNK_BLOCKS * UDONONNX_Q4_DISPATCH_CHUNKS)
#define UDONONNX_Q4_PARTIAL_CHUNKS 4u
#define UDONONNX_Q4_MAX_SPLITS                                                 \
  (UDONONNX_Q4_CHUNK_BLOCKS * UDONONNX_Q4_PARTIAL_CHUNKS)
#define UDONONNX_Q4_SURFACE_WORDS 131072u

/* All eight nibbles of one packed word; the expression and order match the
 * per-element K tail below, so a full block and a tail block round alike. */
#define UDONONNX_Q4_MAC(PACKED, J)                                             \
  sum += (((float)(((PACKED) >> ((J) * 4u)) & 15u) - 8.0f) * scale) *          \
         activation[a_base + (J)]
#define UDONONNX_Q4_WORD(W)                                                    \
  {                                                                            \
    uint packed = packed_weight[scale_index * 4u + (W)];                       \
    uint a_base = row * (uint)k + block * 32u + (W) * 8u;                      \
    UDONONNX_Q4_MAC(packed, 0u);                                               \
    UDONONNX_Q4_MAC(packed, 1u);                                               \
    UDONONNX_Q4_MAC(packed, 2u);                                               \
    UDONONNX_Q4_MAC(packed, 3u);                                               \
    UDONONNX_Q4_MAC(packed, 4u);                                               \
    UDONONNX_Q4_MAC(packed, 5u);                                               \
    UDONONNX_Q4_MAC(packed, 6u);                                               \
    UDONONNX_Q4_MAC(packed, 7u);                                               \
  }

__kernel void udononnx_matmul_nbits_q4(__global const float *activation,
                                       __global const uint *packed_weight,
                                       __global const float *scales,
                                       __global const float *previous,
                                       __global float *output, int m, int n,
                                       int k, int block_count, int splits,
                                       int previous_partials) {
  uint gid = (uint)UDONONNX_LINEAR_GID();
  float value = 0.0f;
  /* plane_row = s * m + row. Dividing gid by n under the guard
   * gid < (splits * m) * n is what bounds plane_row, and so row, for the
   * Device index proof. */
  if (gid < (uint)(splits * m) * (uint)n) {
    uint plane_row = gid / (uint)n;
    uint row = plane_row % (uint)m;
    uint split = plane_row / (uint)m;
    uint column = gid % (uint)n;
    uint blocks = ((uint)k + 31u) / 32u;
    uint full_blocks = (uint)k / 32u;
    uint block_base = split * (uint)block_count;
    float sum = 0.0f;
    for (uint chunk = 0; chunk < UDONONNX_Q4_PARTIAL_CHUNKS; ++chunk)
      if (chunk * UDONONNX_Q4_CHUNK_BLOCKS < (uint)previous_partials)
        for (uint lane = 0; lane < UDONONNX_Q4_CHUNK_BLOCKS; ++lane) {
          uint plane = chunk * UDONONNX_Q4_CHUNK_BLOCKS + lane;
          uint partial = (plane * (uint)m + row) * (uint)n + column;
          if (plane < (uint)previous_partials &&
              partial < UDONONNX_Q4_SURFACE_WORDS)
            sum += previous[partial];
        }
    for (uint chunk = 0; chunk < UDONONNX_Q4_DISPATCH_CHUNKS; ++chunk)
      if (chunk * UDONONNX_Q4_CHUNK_BLOCKS < (uint)block_count)
        for (uint lane = 0; lane < UDONONNX_Q4_CHUNK_BLOCKS; ++lane) {
          uint step = chunk * UDONONNX_Q4_CHUNK_BLOCKS + lane;
          uint block = block_base + step;
          /* The full-block test comes first so the Device index proof bounds
           * every full-block access by k / 32 rather than ceil(k / 32). */
          if (step < (uint)block_count) {
            if (block < full_blocks) {
              /* 1 scale, 4 packed-word and 32 activation Loads, none of
               * which depends on the accumulation chain. */
              uint scale_index = column * blocks + block;
              float scale = scales[scale_index];
              UDONONNX_Q4_WORD(0u);
              UDONONNX_Q4_WORD(1u);
              UDONONNX_Q4_WORD(2u);
              UDONONNX_Q4_WORD(3u);
            } else if (block < blocks) {
              /* K tail block (k % 32 != 0). */
              uint scale_index = column * blocks + block;
              float scale = scales[scale_index];
              for (uint in_block = 0; in_block < 32u; ++in_block) {
                uint k_index = block * 32u + in_block;
                if (k_index < (uint)k) {
                  uint packed = packed_weight[scale_index * 4u + in_block / 8u];
                  sum += (((float)((packed >> ((in_block % 8u) * 4u)) & 15u) -
                           8.0f) *
                          scale) *
                         activation[row * (uint)k + k_index];
                }
              }
            }
          }
        }
    value = sum;
  }
  output[gid] = value;
}
