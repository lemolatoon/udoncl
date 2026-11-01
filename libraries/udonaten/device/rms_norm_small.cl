#ifndef __OPENCL_C_VERSION__
#include <math.h>
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonaten_rms_norm_small_cpu_gid;
static inline size_t get_global_id(unsigned dimension) {
  (void)dimension;
  return udonaten_rms_norm_small_cpu_gid;
}
#define __kernel
#define __global
#define sqrt sqrtf
#endif

#define UDONATEN_RMS_SMALL_INPUT_OFFSET 0u
#define UDONATEN_RMS_SMALL_WEIGHT_OFFSET 512u
#define UDONATEN_RMS_SMALL_PREVIOUS_OUTPUT_OFFSET 1024u
#define UDONATEN_RMS_SMALL_RESULT_OFFSET 1536u

__kernel void udonaten_rms_norm_small(__global const float *packed,
                                      __global float *output, int words,
                                      int normalized_words, float epsilon,
                                      float alpha, float beta) {
  uint gid = (uint)get_global_id(0);
  float value = packed[gid];
  if (gid >= UDONATEN_RMS_SMALL_RESULT_OFFSET &&
      gid < UDONATEN_RMS_SMALL_RESULT_OFFSET + (uint)words) {
    uint element = gid - UDONATEN_RMS_SMALL_RESULT_OFFSET;
    uint row = element / (uint)normalized_words;
    uint row_start = row * (uint)normalized_words;
    float square_sum = 0.0f;
    for (uint candidate = 0u; candidate < 512u; ++candidate)
      if (candidate < (uint)normalized_words) {
        float sample = packed[UDONATEN_RMS_SMALL_INPUT_OFFSET +
                              ((row_start + candidate) & 511u)];
        square_sum += sample * sample;
      }
    float inverse_rms =
        1.0f / sqrt(square_sum / (float)normalized_words + epsilon);
    float normalized =
        packed[UDONATEN_RMS_SMALL_INPUT_OFFSET + (element & 511u)] *
        inverse_rms *
        packed[UDONATEN_RMS_SMALL_WEIGHT_OFFSET +
               ((element % (uint)normalized_words) & 511u)];
    value = alpha * normalized +
            beta * packed[UDONATEN_RMS_SMALL_PREVIOUS_OUTPUT_OFFSET +
                          (element & 511u)];
  }
  output[gid] = value;
}
