#ifndef __OPENCL_C_VERSION__
#include <stddef.h>
typedef unsigned int uint;
extern size_t udondnn_pooling_cpu_gid;
static inline size_t get_global_id(unsigned dimension) {
  (void)dimension;
  return udondnn_pooling_cpu_gid;
}
#define __kernel
#define __global
#endif

#define UDONDNN_POOLING_WORDS 2048u
#define UDONDNN_POOLING_PREVIOUS_OUTPUT_OFFSET 512u
#define UDONDNN_POOLING_RESULT_OFFSET 1024u

__kernel void udondnn_pooling_forward(__global const float *packed,
                                      __global float *output, int input_shape,
                                      int pooling_shape, int pooling_mode,
                                      int nan_propagation, int output_shape,
                                      float alpha, float beta) {
  uint gid = (uint)get_global_id(0);
  float value = packed[gid];
  uint output_n = (uint)output_shape & 15u;
  uint output_c = ((uint)output_shape >> 4u) & 15u;
  uint output_h = ((uint)output_shape >> 8u) & 15u;
  uint output_w = ((uint)output_shape >> 12u) & 15u;
  uint output_words = output_n * output_c * output_h * output_w;
  if (gid >= UDONDNN_POOLING_RESULT_OFFSET &&
      gid < UDONDNN_POOLING_RESULT_OFFSET + output_words) {
    uint element = gid - UDONDNN_POOLING_RESULT_OFFSET;
    uint ox = element % output_w;
    uint oy = (element / output_w) % output_h;
    uint channel = (element / (output_w * output_h)) % output_c;
    uint batch = element / (output_w * output_h * output_c);
    uint input_c = ((uint)input_shape >> 4u) & 15u;
    uint input_h = ((uint)input_shape >> 8u) & 15u;
    uint input_w = ((uint)input_shape >> 12u) & 15u;
    uint window_h = (uint)pooling_shape & 15u;
    uint window_w = ((uint)pooling_shape >> 4u) & 15u;
    uint pad_h = ((uint)pooling_shape >> 8u) & 15u;
    uint pad_w = ((uint)pooling_shape >> 12u) & 15u;
    uint stride_h = ((uint)pooling_shape >> 16u) & 15u;
    uint stride_w = ((uint)pooling_shape >> 20u) & 15u;
    float maximum = -3.402823466e+38f;
    float sum = 0.0f;
    float nan_value = 0.0f;
    uint valid = 0u;
    uint numeric = 0u;
    int has_nan = 0;
    for (uint ky = 0; ky < 5u; ++ky)
      for (uint kx = 0; kx < 5u; ++kx) {
        int iy = (int)(oy * stride_h + ky) - (int)pad_h;
        int ix = (int)(ox * stride_w + kx) - (int)pad_w;
        if (ky < window_h && kx < window_w && iy >= 0 && ix >= 0 &&
            iy < (int)input_h && ix < (int)input_w) {
          uint input_index =
              ((batch * input_c + channel) * input_h + (uint)iy) * input_w +
              (uint)ix;
          float sample = packed[input_index & 511u];
          ++valid;
          if (sample != sample) {
            if (nan_propagation != 0) {
              has_nan = 1;
              nan_value = sample;
            }
            sample = 0.0f;
          } else {
            ++numeric;
          }
          sum += sample;
          if (sample > maximum)
            maximum = sample;
        }
      }
    float pooled = maximum;
    if (pooling_mode == 1)
      pooled = sum / (float)(window_h * window_w);
    else if (pooling_mode == 2)
      pooled = sum / (float)valid;
    else if (numeric == 0u)
      pooled = 0.0f;
    if (has_nan != 0)
      pooled = nan_value;
    value =
        alpha * pooled +
        beta *
            packed[UDONDNN_POOLING_PREVIOUS_OUTPUT_OFFSET + (element & 511u)];
  }
  output[gid] = value;
}
