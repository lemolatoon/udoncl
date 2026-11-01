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

#define UDONATEN_RMS_WEIGHT_OFFSET 4096u
#define UDONATEN_RMS_RESULT_OFFSET 4096u

/* One fragment owns one output element. The current two-texture Device ABI can
 * hold the complete 4096-word input and optional weight without a CPU readback.
 * A future multi-input ABI can replace the repeated scan with a reduction tree
 * without changing the public ATen call. */
__kernel void udondnn_ext_rms_normalization_forward(
    __global const float *state, __global float *output, int words,
    int normalized_words, float epsilon, int has_weight) {
  uint gid = (uint)UDONATEN_RMS_GID();
  float value = state[gid & 8191u];
  if (gid >= UDONATEN_RMS_RESULT_OFFSET &&
      gid < UDONATEN_RMS_RESULT_OFFSET + (uint)words) {
    uint element = gid - UDONATEN_RMS_RESULT_OFFSET;
    uint row_start = element - element % (uint)normalized_words;
    float square_sum = 0.0f;
    for (uint candidate = 0u; candidate < 4096u; ++candidate)
      if (candidate < (uint)normalized_words) {
        float sample = state[(row_start + candidate) & 8191u];
        square_sum += sample * sample;
      }
    float inverse_rms =
        1.0f / sqrt(square_sum / (float)normalized_words + epsilon);
    float weight = 1.0f;
    if (has_weight)
      weight = state[(UDONATEN_RMS_WEIGHT_OFFSET +
                      element % (uint)normalized_words) &
                     8191u];
    value = state[element & 8191u] * inverse_rms * weight;
  }
  output[gid] = value;
}
