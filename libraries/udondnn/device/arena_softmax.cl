#ifdef __OPENCL_C_VERSION__
#define UDONDNN_ARENA_X() ((size_t)get_global_id(0))
#define UDONDNN_ARENA_Y() ((size_t)get_global_id(1))
#else
#include <math.h>
#include <stddef.h>
extern size_t udondnn_arena_softmax_cpu_gid_x;
extern size_t udondnn_arena_softmax_cpu_gid_y;
#define UDONDNN_ARENA_X() (udondnn_arena_softmax_cpu_gid_x)
#define UDONDNN_ARENA_Y() (udondnn_arena_softmax_cpu_gid_y)
#define __kernel
#define __global
#define exp expf
#define log logf
#endif

#define UDONDNN_ARENA_WIDTH 8192u

/* Private texture-arena backend for udondnnSoftmaxForward. The scalar
 * arguments retain cuDNN's algorithm/mode distinction; ONNX currently emits
 * alpha=1 and beta=0, so no previous-output texture is required. */
__kernel void udondnn_arena_softmax_forward(__global const float *input,
                                            __global float *output, int input_n,
                                            int input_c, int input_h,
                                            int input_w, int algorithm,
                                            int mode) {
  size_t gid = UDONDNN_ARENA_Y() * UDONDNN_ARENA_WIDTH + UDONDNN_ARENA_X();
  size_t sample_words = (size_t)input_c * (size_t)input_h * (size_t)input_w;
  size_t words = (size_t)input_n * sample_words;
  float value = 0.0f;
  if (gid < words) {
    size_t group_start;
    size_t group_stride;
    int group_words;
    if (mode == 0) {
      group_start = (gid / sample_words) * sample_words;
      group_stride = 1u;
      group_words = (int)sample_words;
    } else {
      size_t spatial = (size_t)input_h * (size_t)input_w;
      size_t batch = gid / sample_words;
      size_t within_sample = gid % sample_words;
      size_t spatial_index = within_sample % spatial;
      group_start = batch * sample_words + spatial_index;
      group_stride = spatial;
      group_words = input_c;
    }
    float maximum = -3.402823466e+38f;
    for (int candidate = 0; candidate < 512; ++candidate) {
      if (candidate < group_words) {
        size_t sample_index =
            (group_start + (size_t)candidate * group_stride) & 16383u;
        float sample = input[sample_index];
        if (sample > maximum)
          maximum = sample;
      }
    }
    float denominator = 0.0f;
    for (int candidate = 0; candidate < 512; ++candidate)
      if (candidate < group_words)
        denominator += exp(
            input[(group_start + (size_t)candidate * group_stride) & 16383u] -
            maximum);
    value = exp(input[gid] - maximum) / denominator;
    if (algorithm == 2)
      value = input[gid] - maximum - log(denominator);
  }
  output[gid] = value;
}
