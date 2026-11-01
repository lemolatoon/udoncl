#ifndef __OPENCL_C_VERSION__
#include <math.h>
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonaten_arena_rms_cpu_gid_x;
extern size_t udonaten_arena_rms_cpu_gid_y;
#define UDONATEN_ARENA_RMS_X() (udonaten_arena_rms_cpu_gid_x)
#define UDONATEN_ARENA_RMS_Y() (udonaten_arena_rms_cpu_gid_y)
#define __kernel
#define __global
#define sqrt sqrtf
#else
#define UDONATEN_ARENA_RMS_X() ((size_t)get_global_id(0))
#define UDONATEN_ARENA_RMS_Y() ((size_t)get_global_id(1))
#endif

#define UDONATEN_ARENA_WIDTH 8192u
#define UDONATEN_RMS_REDUCTION_CHUNK 64u
#define UDONATEN_RMS_MAX_PARTIALS_PER_ROW 64u

/* ATen-owned RMSNorm backend. Partials are laid out per normalized row so the
 * same two passes cover one or more rows without CPU readback. */
__kernel void udonaten_arena_rms_norm_reduce(__global const float *input,
                                             __global float *partials,
                                             int words, int normalized_words) {
  uint gid = (uint)(UDONATEN_ARENA_RMS_Y() * UDONATEN_ARENA_WIDTH +
                    UDONATEN_ARENA_RMS_X());
  uint partials_per_row =
      ((uint)normalized_words + UDONATEN_RMS_REDUCTION_CHUNK - 1u) /
      UDONATEN_RMS_REDUCTION_CHUNK;
  uint partial_count =
      ((uint)words / (uint)normalized_words) * partials_per_row;
  float square_sum = 0.0f;
  if (gid < partial_count) {
    uint row = gid / partials_per_row;
    uint chunk = gid % partials_per_row;
    uint base =
        row * (uint)normalized_words + chunk * UDONATEN_RMS_REDUCTION_CHUNK;
    for (uint candidate = 0u; candidate < UDONATEN_RMS_REDUCTION_CHUNK;
         ++candidate)
      if (chunk * UDONATEN_RMS_REDUCTION_CHUNK + candidate <
          (uint)normalized_words) {
        float sample = input[(base + candidate) & 16383u];
        square_sum += sample * sample;
      }
  }
  partials[gid] = square_sum;
}

/* Weight is a separate readonly input, matching ATen's optional weight rather
 * than embedding it in a private packed-state offset. */
__kernel void udonaten_arena_rms_norm_finalize(
    __global const float *input, __global const float *partials,
    __global const float *weight_input, __global float *output, int words,
    int normalized_words, float epsilon, int has_weight) {
  uint gid = (uint)(UDONATEN_ARENA_RMS_Y() * UDONATEN_ARENA_WIDTH +
                    UDONATEN_ARENA_RMS_X());
  float value = 0.0f;
  if (gid < (uint)words) {
    uint partials_per_row =
        ((uint)normalized_words + UDONATEN_RMS_REDUCTION_CHUNK - 1u) /
        UDONATEN_RMS_REDUCTION_CHUNK;
    uint row = gid / (uint)normalized_words;
    float square_sum = 0.0f;
    for (uint candidate = 0u; candidate < UDONATEN_RMS_MAX_PARTIALS_PER_ROW;
         ++candidate)
      if (candidate < partials_per_row)
        square_sum += partials[(row * partials_per_row + candidate) & 16383u];
    float inverse_rms =
        1.0f / sqrt(square_sum / (float)normalized_words + epsilon);
    float weight = 1.0f;
    if (has_weight)
      weight = weight_input[(gid % (uint)normalized_words) & 16383u];
    value = input[gid & 16383u] * inverse_rms * weight;
  }
  output[gid] = value;
}
