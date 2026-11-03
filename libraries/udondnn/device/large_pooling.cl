#ifndef __OPENCL_C_VERSION__
#include <float.h>
#include <stddef.h>
typedef unsigned int uint;
extern size_t udondnn_large_pooling_cpu_gid_x;
extern size_t udondnn_large_pooling_cpu_gid_y;
#define UDONDNN_LARGE_POOL_X() (udondnn_large_pooling_cpu_gid_x)
#define UDONDNN_LARGE_POOL_Y() (udondnn_large_pooling_cpu_gid_y)
#define __kernel
#define __global
#else
#define UDONDNN_LARGE_POOL_X() ((size_t)get_global_id(0))
#define UDONDNN_LARGE_POOL_Y() ((size_t)get_global_id(1))
#endif

#define UDONDNN_LARGE_POOL_WIDTH 8192u

/* Public udondnnPoolingForward backend for tensors that do not fit the small
 * 512-word staging path. It is ordinary OpenCL C: one work-item owns one NCHW
 * output element and reduces a bounded 8x8 window. */
__kernel void udondnn_large_pooling_forward(
    __global const float *input, __global const float *previous,
    __global float *output, int output_words, int input_c, int input_h,
    int input_w, int output_h, int output_w, int window_shape, int pad_shape,
    int stride_shape, int pooling_options, float alpha, float beta) {
  size_t gid = UDONDNN_LARGE_POOL_Y() * UDONDNN_LARGE_POOL_WIDTH +
               UDONDNN_LARGE_POOL_X();
  float result = 0.0f;
  if (gid < (size_t)output_words) {
    int window_h = window_shape & 65535;
    int window_w = (window_shape >> 16) & 65535;
    int pad_h = pad_shape & 65535;
    int pad_w = (pad_shape >> 16) & 65535;
    int stride_h = stride_shape & 65535;
    int stride_w = (stride_shape >> 16) & 65535;
    int pooling_mode = pooling_options & 255;
    int nan_propagation = (pooling_options >> 8) & 255;
    size_t remaining = gid;
    size_t output_x = remaining % (size_t)output_w;
    remaining /= (size_t)output_w;
    size_t output_y = remaining % (size_t)output_h;
    remaining /= (size_t)output_h;
    size_t channel = remaining % (size_t)input_c;
    size_t batch = remaining / (size_t)input_c;
    float sum = 0.0f;
    float maximum = -3.402823466e+38f;
    float nan_value = 0.0f;
    uint valid = 0u;
    uint numeric = 0u;
    int has_nan = 0;
    for (uint kernel_y = 0u; kernel_y < 8u; ++kernel_y)
      for (uint kernel_x = 0u; kernel_x < 8u; ++kernel_x) {
        int input_y = (int)output_y * stride_h + (int)kernel_y - pad_h;
        int input_x = (int)output_x * stride_w + (int)kernel_x - pad_w;
        if (kernel_y < (uint)window_h && kernel_x < (uint)window_w &&
            input_y >= 0 && input_y < input_h && input_x >= 0 &&
            input_x < input_w) {
          size_t input_index =
              ((batch * (size_t)input_c + channel) * (size_t)input_h +
               (size_t)input_y) *
                  (size_t)input_w +
              (size_t)input_x;
          float sample = input[input_index & 16383u];
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
      pooled = valid ? sum / (float)valid : 0.0f;
    else if (numeric == 0u)
      pooled = 0.0f;
    if (has_nan != 0)
      pooled = nan_value;
    result = alpha * pooled + beta * previous[gid];
  }
  output[gid] = result;
}
