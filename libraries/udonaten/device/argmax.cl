#ifndef __OPENCL_C_VERSION__
#include <math.h>
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonaten_argmax_cpu_gid;
#define UDONATEN_ARGMAX_GID() (udonaten_argmax_cpu_gid)
#define __kernel
#define __global
static inline uint udonaten_argmax_bits(float value) {
  union {
    float as_float;
    uint as_uint;
  } bits = {value};
  return bits.as_uint;
}
#define UDONATEN_ARGMAX_BITS(value) udonaten_argmax_bits(value)
#else
#define UDONATEN_ARGMAX_GID() ((size_t)get_global_id(0))
#define UDONATEN_ARGMAX_BITS(value) as_uint(value)
#endif

#define UDONATEN_ARGMAX_CHUNK 256u
#define UDONATEN_ARGMAX_PARTIALS 512u
#define UDONATEN_ARGMAX_INDEX_MASK 0x7fffffffu
#define UDONATEN_ARGMAX_NONFINITE 0x80000000u
#define UDONATEN_ARGMAX_IS_NAN(value)                                          \
  ((UDONATEN_ARGMAX_BITS(value) & 0x7fffffffu) > 0x7f800000u)
#define UDONATEN_ARGMAX_IS_FINITE(value)                                       \
  ((UDONATEN_ARGMAX_BITS(value) & 0x7fffffffu) < 0x7f800000u)

#define UDONATEN_ARGMAX_BETTER(candidate, candidate_index, best, best_index)   \
  ((UDONATEN_ARGMAX_IS_NAN(candidate) != UDONATEN_ARGMAX_IS_NAN(best))         \
       ? UDONATEN_ARGMAX_IS_NAN(candidate)                                     \
       : ((candidate) > (best) ||                                              \
          ((candidate) == (best) && (candidate_index) < (best_index))))

/* ATen-owned one-dimensional argmax reduction.  Each fragment emits one
 * winning relative index plus a flag recording any non-finite input in its
 * chunk.  Inactive fragments still initialize their partial deterministically.
 */
__kernel void udonaten_argmax_reduce(__global const float *input,
                                     __global uint *partials,
                                     int input_word_offset, int words) {
  uint gid = (uint)UDONATEN_ARGMAX_GID();
  uint begin = gid * UDONATEN_ARGMAX_CHUNK;
  uint best_index = UDONATEN_ARGMAX_INDEX_MASK;
  float best = 0.0f;
  uint flags = 0u;
  for (uint lane = 0u; lane < UDONATEN_ARGMAX_CHUNK; ++lane) {
    uint index = begin + lane;
    if (index < (uint)words) {
      float candidate = input[(uint)input_word_offset + index];
      if (!UDONATEN_ARGMAX_IS_FINITE(candidate))
        flags = UDONATEN_ARGMAX_NONFINITE;
      if (best_index == UDONATEN_ARGMAX_INDEX_MASK ||
          UDONATEN_ARGMAX_BETTER(candidate, index, best, best_index)) {
        best = candidate;
        best_index = index;
      }
    }
  }
  partials[gid] = best_index | flags;
}

/* reject_nonfinite is a backend policy, not a change to ATen ArgMax.  With it
 * disabled, NaNs win and ties select the lowest index like PyTorch CUDA.
 */
__kernel void udonaten_argmax_finalize(__global const float *input,
                                       __global const uint *partials,
                                       __global uint *result,
                                       int input_word_offset, int words,
                                       int reject_nonfinite) {
  uint best_index = UDONATEN_ARGMAX_INDEX_MASK;
  float best = 0.0f;
  uint nonfinite = 0u;
  if (UDONATEN_ARGMAX_GID() != 0u)
    return;
  for (uint partial = 0u; partial < UDONATEN_ARGMAX_PARTIALS; ++partial) {
    uint encoded = partials[partial];
    uint index = encoded & UDONATEN_ARGMAX_INDEX_MASK;
    nonfinite |= encoded & UDONATEN_ARGMAX_NONFINITE;
    if (index < (uint)words) {
      float candidate = input[(uint)input_word_offset + index];
      if (best_index == UDONATEN_ARGMAX_INDEX_MASK ||
          UDONATEN_ARGMAX_BETTER(candidate, index, best, best_index)) {
        best = candidate;
        best_index = index;
      }
    }
  }
  result[0] = reject_nonfinite && nonfinite ? 0xffffffffu : best_index;
}
