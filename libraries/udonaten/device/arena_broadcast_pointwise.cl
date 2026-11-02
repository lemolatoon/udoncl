#ifdef __OPENCL_C_VERSION__
#define UDONATEN_BROADCAST_X() ((size_t)get_global_id(0))
#define UDONATEN_BROADCAST_Y() ((size_t)get_global_id(1))
#else
#include <math.h>
#include <stddef.h>
extern size_t udonaten_arena_broadcast_cpu_gid_x;
extern size_t udonaten_arena_broadcast_cpu_gid_y;
#define UDONATEN_BROADCAST_X() (udonaten_arena_broadcast_cpu_gid_x)
#define UDONATEN_BROADCAST_Y() (udonaten_arena_broadcast_cpu_gid_y)
#define __kernel
#define __global
#define exp expf
#endif

#define UDONATEN_BROADCAST_WIDTH 8192u

/* Coordinate-aware private backend for ATen pointwise broadcasting. Shapes
 * are right-aligned to NCHW by the ONNX planner. Public ownership remains with
 * the corresponding udonaten operation. */
__kernel void udonaten_arena_broadcast_pointwise(
    __global const float *self, __global const float *other,
    __global float *output, int words, int operation, int out_c, int out_h,
    int out_w, int self_n, int self_c, int self_h, int self_w, int other_n,
    int other_c, int other_h, int other_w) {
  size_t gid = UDONATEN_BROADCAST_Y() * UDONATEN_BROADCAST_WIDTH +
               UDONATEN_BROADCAST_X();
  float value = 0.0f;
  if (gid < (size_t)words) {
    size_t remaining = gid;
    size_t w = remaining % (size_t)out_w;
    remaining /= (size_t)out_w;
    size_t h = remaining % (size_t)out_h;
    remaining /= (size_t)out_h;
    size_t c = remaining % (size_t)out_c;
    size_t n = remaining / (size_t)out_c;
    size_t left_index =
        ((((n % (size_t)self_n) * (size_t)self_c + c % (size_t)self_c) *
              (size_t)self_h +
          h % (size_t)self_h) *
             (size_t)self_w +
         w % (size_t)self_w) &
        16383u;
    size_t right_index =
        ((((n % (size_t)other_n) * (size_t)other_c + c % (size_t)other_c) *
              (size_t)other_h +
          h % (size_t)other_h) *
             (size_t)other_w +
         w % (size_t)other_w) &
        16383u;
    float left = self[left_index];
    float right = other[right_index];
    if (operation == 0)
      value = left + right;
    else if (operation == 1)
      value = left * right;
    else if (operation == 2)
      value = 1.0f / (1.0f + exp(-left));
    else if (operation == 3)
      value = left - right;
    else if (operation == 4)
      value = left > 0.0f ? left : 0.0f;
    else if (operation == 5)
      value = 2.0f / (1.0f + exp(-2.0f * left)) - 1.0f;
  }
  output[gid] = value;
}
