#ifdef __OPENCL_C_VERSION__
#define UDONATEN_INT4_GID() ((size_t)get_global_id(0))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonaten_int4_cpu_gid;
#define UDONATEN_INT4_GID() (udonaten_int4_cpu_gid)
#define __kernel
#define __global
#endif

#define UDONATEN_INT4_INPUT_OFFSET 6144u
#define UDONATEN_INT4_RESULT_OFFSET 7168u

/* One fragment owns one output feature. The active tile expands packed bytes
 * to exact numeric float words while preserving ATen's separate scale/zero
 * contract: dequantized = signed_int4 * scale + zero. */
__kernel void udonaten_weight_int4pack_mm(__global const float *state,
                                          __global float *result, int n, int k,
                                          int row_stride) {
  uint gid = (uint)UDONATEN_INT4_GID();
  float value = state[gid];
  if (gid >= UDONATEN_INT4_RESULT_OFFSET &&
      gid < UDONATEN_INT4_RESULT_OFFSET + (uint)n) {
    uint row = gid - UDONATEN_INT4_RESULT_OFFSET;
    float sum = 0.0f;
    for (uint inner = 0; inner < 128u; ++inner) {
      if (inner < (uint)k) {
        uint block = inner / 32u;
        uint in_block = inner % 32u;
        uint base = row * (uint)row_stride + block * 18u;
        float low = state[(base + in_block / 2u) & 8191u];
        float high = 0.0f;
        if (low >= 128.0f) {
          low -= 128.0f;
          high += 8.0f;
        }
        if (low >= 64.0f) {
          low -= 64.0f;
          high += 4.0f;
        }
        if (low >= 32.0f) {
          low -= 32.0f;
          high += 2.0f;
        }
        if (low >= 16.0f) {
          low -= 16.0f;
          high += 1.0f;
        }
        float nibble = (in_block % 2u) == 0u ? low : high;
        float quantized = nibble >= 8.0f ? nibble - 16.0f : nibble;
        float scale = state[(base + 16u) & 8191u];
        float zero = state[(base + 17u) & 8191u];
        sum += (quantized * scale + zero) *
               state[(UDONATEN_INT4_INPUT_OFFSET + inner) & 8191u];
      }
    }
    value = sum;
  }
  result[gid] = value;
}
