#ifdef __OPENCL_C_VERSION__
#define UDONDNN_ARENA_POOL_X() ((size_t)get_global_id(0))
#define UDONDNN_ARENA_POOL_Y() ((size_t)get_global_id(1))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udondnn_arena_pooling_cpu_gid_x;
extern size_t udondnn_arena_pooling_cpu_gid_y;
#define UDONDNN_ARENA_POOL_X() (udondnn_arena_pooling_cpu_gid_x)
#define UDONDNN_ARENA_POOL_Y() (udondnn_arena_pooling_cpu_gid_y)
#define __kernel
#define __global
#endif

#define UDONDNN_ARENA_POOL_WIDTH 8192u
#define UDONDNN_ARENA_POOL_ROW(KY)                                             \
  {                                                                            \
    if ((KY) < (size_t)window_h) {                                             \
      size_t row_index =                                                       \
          plane_index +                                                        \
          (output_y * (size_t)stride_h + (KY)) * (size_t)input_w +             \
          output_x * (size_t)stride_w;                                         \
      if (window_w > 0)                                                        \
        value += input[(row_index + 0u) & 16383u];                             \
      if (window_w > 1)                                                        \
        value += input[(row_index + 1u) & 16383u];                             \
      if (window_w > 2)                                                        \
        value += input[(row_index + 2u) & 16383u];                             \
      if (window_w > 3)                                                        \
        value += input[(row_index + 3u) & 16383u];                             \
      if (window_w > 4)                                                        \
        value += input[(row_index + 4u) & 16383u];                             \
      if (window_w > 5)                                                        \
        value += input[(row_index + 5u) & 16383u];                             \
      if (window_w > 6)                                                        \
        value += input[(row_index + 6u) & 16383u];                             \
      if (window_w > 7)                                                        \
        value += input[(row_index + 7u) & 16383u];                             \
    }                                                                          \
  }

/* Average-pooling arena backend of udondnnPoolingForward. The planner selects
 * it only for unpadded exclude-padding average mode with at most an 8x8
 * window; one fragment independently reduces one NCHW output window. */
__kernel void udondnn_arena_pooling_forward(
    __global const float *input, __global float *output, int words, int input_c,
    int input_h, int input_w, int output_c, int output_h, int output_w,
    int window_h, int window_w, int stride_h, int stride_w) {
  size_t gid = UDONDNN_ARENA_POOL_Y() * UDONDNN_ARENA_POOL_WIDTH +
               UDONDNN_ARENA_POOL_X();
  float value = 0.0f;
  if (gid < (size_t)words) {
    size_t remaining = gid;
    size_t output_x = remaining % (size_t)output_w;
    remaining /= (size_t)output_w;
    size_t output_y = remaining % (size_t)output_h;
    remaining /= (size_t)output_h;
    size_t channel = remaining % (size_t)output_c;
    size_t batch = remaining / (size_t)output_c;
    size_t plane_index =
        (batch * (size_t)input_c + channel) * (size_t)input_h * (size_t)input_w;
    UDONDNN_ARENA_POOL_ROW(0u);
    UDONDNN_ARENA_POOL_ROW(1u);
    UDONDNN_ARENA_POOL_ROW(2u);
    UDONDNN_ARENA_POOL_ROW(3u);
    UDONDNN_ARENA_POOL_ROW(4u);
    UDONDNN_ARENA_POOL_ROW(5u);
    UDONDNN_ARENA_POOL_ROW(6u);
    UDONDNN_ARENA_POOL_ROW(7u);
    value /= (float)(window_h * window_w);
  }
  output[gid] = value;
}
