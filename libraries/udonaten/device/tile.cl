#ifdef __OPENCL_C_VERSION__
#define UDONATEN_TILE_GID() ((size_t)get_global_id(0))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonaten_tile_cpu_gid;
#define UDONATEN_TILE_GID() (udonaten_tile_cpu_gid)
#define __kernel
#define __global
#endif

#define UDONATEN_TILE_RESULT_OFFSET 4096u

/* ATen tile/repeat copies one output element per fragment. Repetition is
 * aligned to the four trailing tensor dimensions, matching tile_symint's
 * leading-one promotion before it delegates to repeat. */
__kernel void udonaten_tile(__global const float *state, __global float *result,
                            int input_n, int input_c, int input_h, int input_w,
                            int repeat_n, int repeat_c, int repeat_h,
                            int repeat_w, int output_words) {
  uint gid = (uint)UDONATEN_TILE_GID();
  float value = state[gid];
  if (gid >= UDONATEN_TILE_RESULT_OFFSET &&
      gid < UDONATEN_TILE_RESULT_OFFSET + (uint)output_words) {
    uint element = gid - UDONATEN_TILE_RESULT_OFFSET;
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
    value = state[source & 4095u];
  }
  result[gid] = value;
}
