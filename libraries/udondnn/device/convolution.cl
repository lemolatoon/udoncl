#ifndef __OPENCL_C_VERSION__
#include <stddef.h>
typedef unsigned int uint;
extern size_t udondnn_cpu_gid;
static inline size_t get_global_id(unsigned dimension) {
  (void)dimension;
  return udondnn_cpu_gid;
}
#define __kernel
#define __global
#endif

#define UDONDNN_WORDS 2048u
#define UDONDNN_INPUT_OFFSET 0u
#define UDONDNN_FILTER_OFFSET 512u
#define UDONDNN_PREVIOUS_OUTPUT_OFFSET 1024u
#define UDONDNN_RESULT_OFFSET 1536u

__kernel void udondnn_convolution_forward(
    __global const float *packed, __global float *output, int input_shape,
    int filter_shape, int convolution_shape, int convolution_mode,
    int output_shape, float alpha, float beta) {
  uint gid = (uint)get_global_id(0);
  float value = packed[gid];
  uint output_n = (uint)output_shape & 15u;
  uint output_c = ((uint)output_shape >> 4u) & 15u;
  uint output_h = ((uint)output_shape >> 8u) & 15u;
  uint output_w = ((uint)output_shape >> 12u) & 15u;
  uint output_words = output_n * output_c * output_h * output_w;
  if (gid >= UDONDNN_RESULT_OFFSET &&
      gid < UDONDNN_RESULT_OFFSET + output_words) {
    uint element = gid - UDONDNN_RESULT_OFFSET;
    uint ox = element % output_w;
    uint oy = (element / output_w) % output_h;
    uint oc = (element / (output_w * output_h)) % output_c;
    uint batch = element / (output_w * output_h * output_c);
    uint input_c = ((uint)input_shape >> 4u) & 15u;
    uint input_h = ((uint)input_shape >> 8u) & 15u;
    uint input_w = ((uint)input_shape >> 12u) & 15u;
    uint filter_h = ((uint)filter_shape >> 8u) & 15u;
    uint filter_w = ((uint)filter_shape >> 12u) & 15u;
    uint pad_h = (uint)convolution_shape & 15u;
    uint pad_w = ((uint)convolution_shape >> 4u) & 15u;
    uint stride_h = ((uint)convolution_shape >> 8u) & 15u;
    uint stride_w = ((uint)convolution_shape >> 12u) & 15u;
    uint dilation_h = ((uint)convolution_shape >> 16u) & 15u;
    uint dilation_w = ((uint)convolution_shape >> 20u) & 15u;
    uint mode = (uint)convolution_mode & 1u;
    float sum = 0.0f;
    for (uint ic = 0; ic < 8u; ++ic)
      for (uint ky = 0; ky < 5u; ++ky)
        for (uint kx = 0; kx < 5u; ++kx) {
          int iy = (int)(oy * stride_h + ky * dilation_h) - (int)pad_h;
          int ix = (int)(ox * stride_w + kx * dilation_w) - (int)pad_w;
          if (ic < input_c && ky < filter_h && kx < filter_w && iy >= 0 &&
              ix >= 0 && iy < (int)input_h && ix < (int)input_w) {
            uint input_index =
                ((batch * input_c + ic) * input_h + (uint)iy) * input_w +
                (uint)ix;
            uint filter_y = mode * ky + (1u - mode) * (filter_h - 1u - ky);
            uint filter_x = mode * kx + (1u - mode) * (filter_w - 1u - kx);
            uint filter_index =
                ((oc * input_c + ic) * filter_h + filter_y) * filter_w +
                filter_x;
            sum += packed[UDONDNN_INPUT_OFFSET + (input_index & 511u)] *
                   packed[UDONDNN_FILTER_OFFSET + (filter_index & 511u)];
          }
        }
    value = alpha * sum +
            beta * packed[UDONDNN_PREVIOUS_OUTPUT_OFFSET + (element & 511u)];
  }
  output[gid] = value;
}
