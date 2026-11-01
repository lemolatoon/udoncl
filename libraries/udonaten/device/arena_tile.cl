#ifdef __OPENCL_C_VERSION__
#define UDONATEN_ARENA_TILE_X() ((size_t)get_global_id(0))
#define UDONATEN_ARENA_TILE_Y() ((size_t)get_global_id(1))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonaten_arena_tile_cpu_gid_x;
extern size_t udonaten_arena_tile_cpu_gid_y;
#define UDONATEN_ARENA_TILE_X() (udonaten_arena_tile_cpu_gid_x)
#define UDONATEN_ARENA_TILE_Y() (udonaten_arena_tile_cpu_gid_y)
#define __kernel
#define __global
#endif

#define UDONATEN_ARENA_WIDTH 8192u

/* Private AOT backend for ATen tile/repeat. It writes the logical output at
 * arena offset zero, unlike the packed-state validation kernel. */
__kernel void udonaten_arena_tile(__global const float *input,
                                  __global float *output, int input_n,
                                  int input_c, int input_h, int input_w,
                                  int repeat_n, int repeat_c, int repeat_h,
                                  int repeat_w, int output_words) {
  uint element = (uint)(UDONATEN_ARENA_TILE_Y() * UDONATEN_ARENA_WIDTH +
                        UDONATEN_ARENA_TILE_X());
  float value = 0.0f;
  if (element < (uint)output_words) {
    uint output_w = (uint)input_w * (uint)repeat_w;
    uint output_h = (uint)input_h * (uint)repeat_h;
    uint output_c = (uint)input_c * (uint)repeat_c;
    uint w = element % output_w;
    uint remaining = element / output_w;
    uint h = remaining % output_h;
    remaining /= output_h;
    uint c = remaining % output_c;
    uint n = remaining / output_c;
    uint source = (((n % (uint)input_n) * (uint)input_c + c % (uint)input_c) *
                       (uint)input_h +
                   h % (uint)input_h) *
                      (uint)input_w +
                  w % (uint)input_w;
    value = input[source & 16383u];
  }
  output[element] = value;
}
