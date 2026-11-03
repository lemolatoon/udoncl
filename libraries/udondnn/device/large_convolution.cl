#ifdef __OPENCL_C_VERSION__
#define UDONDNN_LARGE_CONV_X() ((size_t)get_global_id(0))
#define UDONDNN_LARGE_CONV_Y() ((size_t)get_global_id(1))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udondnn_large_convolution_cpu_gid_x;
extern size_t udondnn_large_convolution_cpu_gid_y;
#define UDONDNN_LARGE_CONV_X() (udondnn_large_convolution_cpu_gid_x)
#define UDONDNN_LARGE_CONV_Y() (udondnn_large_convolution_cpu_gid_y)
#define __kernel
#define __global
#endif

#define UDONDNN_LARGE_CONV_WIDTH 8192u

__kernel void udondnn_large_convolution_forward(
    __global const float *input, __global const float *filter,
    __global const float *previous, __global float *output, int output_words,
    int input_ch, int input_w, int output_ch, int output_w, int filter_hw,
    int pad_hw, int stride_hw, int dilation_hw, int mode, float alpha,
    float beta) {
  size_t gid = UDONDNN_LARGE_CONV_Y() * UDONDNN_LARGE_CONV_WIDTH +
               UDONDNN_LARGE_CONV_X();
  float value = 0.0f;
  if (gid < (size_t)output_words) {
    int input_c = input_ch & 65535;
    int input_h = (input_ch >> 16) & 65535;
    int output_c = output_ch & 65535;
    int output_h = (output_ch >> 16) & 65535;
    int filter_h = filter_hw & 65535;
    int filter_w = (filter_hw >> 16) & 65535;
    int pad_h = pad_hw & 65535;
    int pad_w = (pad_hw >> 16) & 65535;
    int stride_h = stride_hw & 65535;
    int stride_w = (stride_hw >> 16) & 65535;
    int dilation_h = dilation_hw & 65535;
    int dilation_w = (dilation_hw >> 16) & 65535;
    size_t remaining = gid;
    size_t output_x = remaining % (size_t)output_w;
    remaining /= (size_t)output_w;
    size_t output_y = remaining % (size_t)output_h;
    remaining /= (size_t)output_h;
    size_t output_channel = remaining % (size_t)output_c;
    size_t batch = remaining / (size_t)output_c;
    float sum = 0.0f;
    uint reduction_words = (uint)(input_c * filter_h * filter_w);
    for (uint inner = 0; inner < 256u; ++inner)
      if (inner < reduction_words) {
        uint kernel_x = inner % (uint)filter_w;
        uint quotient = inner / (uint)filter_w;
        uint kernel_y = quotient % (uint)filter_h;
        uint input_channel = quotient / (uint)filter_h;
        int input_y =
            (int)output_y * stride_h - pad_h + (int)kernel_y * dilation_h;
        int input_x =
            (int)output_x * stride_w - pad_w + (int)kernel_x * dilation_w;
        if (input_y >= 0 && input_y < input_h && input_x >= 0 &&
            input_x < input_w) {
          uint filter_y = mode != 0 ? kernel_y : (uint)filter_h - 1u - kernel_y;
          uint filter_x = mode != 0 ? kernel_x : (uint)filter_w - 1u - kernel_x;
          size_t input_index =
              ((batch * (size_t)input_c + input_channel) * (size_t)input_h +
               (size_t)input_y) *
                  (size_t)input_w +
              (size_t)input_x;
          size_t filter_index =
              ((output_channel * (size_t)input_c + input_channel) *
                   (size_t)filter_h +
               filter_y) *
                  (size_t)filter_w +
              filter_x;
          sum += input[input_index & 16383u] * filter[filter_index & 16383u];
        }
      }
    value = alpha * sum + beta * previous[gid];
  }
  output[gid] = value;
}
