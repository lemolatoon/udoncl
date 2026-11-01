#ifndef __OPENCL_C_VERSION__
#include <math.h>
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonaten_rms_norm_cpu_gid;
#define UDONATEN_RMS_GID() (udonaten_rms_norm_cpu_gid)
#define __kernel
#define __global
#define sqrt sqrtf
#else
#define UDONATEN_RMS_GID() ((size_t)get_global_id(0))
#endif

#define UDONATEN_RMS_STATE_WORDS 8192u
#define UDONATEN_RMS_WEIGHT_OFFSET 4096u
#define UDONATEN_RMS_REDUCTION_CHUNK 64u
#define UDONATEN_RMS_MAX_PARTIALS 64u

/* Large RMSNorm is an ATen-owned two-pass operation. The first pass emits one
 * partial square sum per 64 normalized elements. Every fragment still writes
 * its output word so the gather profile retains a complete render target. */
__kernel void udonaten_rms_norm_reduce(__global const float *state,
                                       __global float *partials, int words,
                                       int normalized_words) {
  uint gid = (uint)UDONATEN_RMS_GID();
  uint base = gid * UDONATEN_RMS_REDUCTION_CHUNK;
  float square_sum = 0.0f;
  if (gid < UDONATEN_RMS_MAX_PARTIALS && base < (uint)normalized_words &&
      words == normalized_words) {
    for (uint candidate = 0u; candidate < UDONATEN_RMS_REDUCTION_CHUNK;
         ++candidate)
      if (base + candidate < (uint)normalized_words) {
        float sample = state[(base + candidate) & 8191u];
        square_sum += sample * sample;
      }
  }
  partials[gid] = square_sum;
}

/* The second readonly texture is the reduction state produced above. This is
 * the first Device kernel that exercises the generalized multi-input ABI. */
__kernel void udonaten_rms_norm_finalize(__global const float *state,
                                         __global const float *partials,
                                         __global float *output, int words,
                                         int normalized_words, float epsilon,
                                         int has_weight) {
  uint gid = (uint)UDONATEN_RMS_GID();
  float value = state[gid & 8191u];
  if (gid < (uint)words && words == normalized_words) {
    uint partial_count =
        ((uint)normalized_words + UDONATEN_RMS_REDUCTION_CHUNK - 1u) /
        UDONATEN_RMS_REDUCTION_CHUNK;
    float square_sum = 0.0f;
    for (uint candidate = 0u; candidate < UDONATEN_RMS_MAX_PARTIALS;
         ++candidate)
      if (candidate < partial_count)
        square_sum += partials[candidate & 8191u];
    float inverse_rms =
        1.0f / sqrt(square_sum / (float)normalized_words + epsilon);
    float weight = 1.0f;
    if (has_weight)
      weight =
          state[(UDONATEN_RMS_WEIGHT_OFFSET + gid % (uint)normalized_words) &
                8191u];
    value = state[gid & 8191u] * inverse_rms * weight;
  }
  output[gid] = value;
}
