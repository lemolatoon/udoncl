#ifdef __OPENCL_C_VERSION__
#define UDONDNN_ARENA_CONV_X() ((size_t)get_global_id(0))
#define UDONDNN_ARENA_CONV_Y() ((size_t)get_global_id(1))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udondnn_arena_convolution_cpu_gid_x;
extern size_t udondnn_arena_convolution_cpu_gid_y;
#define UDONDNN_ARENA_CONV_X() (udondnn_arena_convolution_cpu_gid_x)
#define UDONDNN_ARENA_CONV_Y() (udondnn_arena_convolution_cpu_gid_y)
#define __kernel
#define __global
#endif

#define UDONDNN_ARENA_CONV_WIDTH 8192u

/* One fragment owns one NCHW output element. This is the arena backend of
 * udondnnConvolutionForward, restricted by the planner to group=1 and at most
 * 256 reduction elements. */
__kernel void udondnn_arena_convolution_forward(
    __global const float *input, __global const float *filter,
    __global float *output, int words, int input_c, int input_h, int input_w,
    int output_c, int output_h, int output_w, int filter_h, int filter_w,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  size_t gid = UDONDNN_ARENA_CONV_Y() * UDONDNN_ARENA_CONV_WIDTH +
               UDONDNN_ARENA_CONV_X();
  float value = 0.0f;
  if (gid < (size_t)words) {
    size_t remaining = gid;
    size_t output_x = remaining % (size_t)output_w;
    remaining /= (size_t)output_w;
    size_t output_y = remaining % (size_t)output_h;
    remaining /= (size_t)output_h;
    size_t output_channel = remaining % (size_t)output_c;
    size_t batch = remaining / (size_t)output_c;
    uint reduction_words = (uint)(input_c * filter_h * filter_w);
    for (uint inner = 0; inner < 256u; ++inner) {
      if (inner < reduction_words) {
        uint kernel_x = inner % (uint)filter_w;
        uint quotient = inner / (uint)filter_w;
        uint kernel_y = quotient % (uint)filter_h;
        uint input_channel = quotient / (uint)filter_h;
        int input_y = (int)output_y * stride_h - pad_h + (int)kernel_y;
        int input_x = (int)output_x * stride_w - pad_w + (int)kernel_x;
        if (input_y >= 0 && input_y < input_h && input_x >= 0 &&
            input_x < input_w) {
          size_t input_index =
              (((batch * (size_t)input_c + (size_t)input_channel) *
                    (size_t)input_h +
                (size_t)input_y) *
                   (size_t)input_w +
               (size_t)input_x) &
              16383u;
          size_t filter_index =
              (((output_channel * (size_t)input_c + (size_t)input_channel) *
                    (size_t)filter_h +
                (size_t)kernel_y) *
                   (size_t)filter_w +
               (size_t)kernel_x) &
              16383u;
          value += input[input_index] * filter[filter_index];
        }
      }
    }
  }
  output[gid] = value;
}
