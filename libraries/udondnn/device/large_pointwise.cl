#ifdef __OPENCL_C_VERSION__
#define UDONDNN_LARGE_POINTWISE_X() ((size_t)get_global_id(0))
#define UDONDNN_LARGE_POINTWISE_Y() ((size_t)get_global_id(1))
#define UDONDNN_ALWAYS_INLINE __attribute__((always_inline))
#else
#include <math.h>
#include <stddef.h>
extern size_t udondnn_large_pointwise_cpu_gid_x;
extern size_t udondnn_large_pointwise_cpu_gid_y;
#define UDONDNN_LARGE_POINTWISE_X() (udondnn_large_pointwise_cpu_gid_x)
#define UDONDNN_LARGE_POINTWISE_Y() (udondnn_large_pointwise_cpu_gid_y)
#define __kernel
#define __global
#define exp expf
#define UDONDNN_ALWAYS_INLINE
#endif

#define UDONDNN_LARGE_POINTWISE_WIDTH 8192u

UDONDNN_ALWAYS_INLINE static inline size_t
udondnn_large_broadcast_index(size_t gid, int source_nc, int source_hw,
                              int output_nc, int output_hw) {
  size_t output_w = (size_t)((output_hw >> 16) & 65535);
  size_t output_h = (size_t)(output_hw & 65535);
  size_t output_c = (size_t)((output_nc >> 16) & 65535);
  size_t coordinate_w = gid % output_w;
  size_t coordinate_h = (gid / output_w) % output_h;
  size_t coordinate_c = (gid / (output_w * output_h)) % output_c;
  size_t coordinate_n = gid / (output_w * output_h * output_c);
  size_t source_n = (size_t)(source_nc & 65535);
  size_t source_c = (size_t)((source_nc >> 16) & 65535);
  size_t source_h = (size_t)(source_hw & 65535);
  size_t source_w = (size_t)((source_hw >> 16) & 65535);
  return ((((coordinate_n % source_n) * source_c + coordinate_c % source_c) *
               source_h +
           coordinate_h % source_h) *
              source_w +
          coordinate_w % source_w) &
         131071u;
}

__kernel void udondnn_large_activation_forward(
    __global const float *input, __global const float *previous,
    __global float *output, int words, int mode, int nan_propagation,
    float coefficient, float alpha, float beta) {
  size_t gid = UDONDNN_LARGE_POINTWISE_Y() * UDONDNN_LARGE_POINTWISE_WIDTH +
               UDONDNN_LARGE_POINTWISE_X();
  float value = 0.0f;
  if (gid < (size_t)words) {
    float sample = input[gid];
    float activated = sample;
    if (sample != sample && nan_propagation == 0)
      sample = 0.0f;
    if (sample != sample)
      activated = sample;
    else if (mode == 0)
      activated = 1.0f / (1.0f + exp(-sample));
    else if (mode == 1)
      activated = sample > 0.0f ? sample : 0.0f;
    else if (mode == 2)
      activated = 2.0f / (1.0f + exp(-2.0f * sample)) - 1.0f;
    else if (mode == 3) {
      activated = sample > 0.0f ? sample : 0.0f;
      if (activated > coefficient)
        activated = coefficient;
    } else if (mode == 4)
      activated = sample > 0.0f ? sample : coefficient * (exp(sample) - 1.0f);
    else
      activated = sample;
    value = alpha * activated;
    if (beta != 0.0f)
      value += beta * previous[gid];
  }
  output[gid] = value;
}

__kernel void udondnn_large_op_tensor(
    __global const float *a_input, __global const float *b_input,
    __global const float *previous, __global float *output, int words, int a_nc,
    int a_hw, int b_nc, int b_hw, int c_nc, int c_hw, int options,
    float alpha_a, float alpha_b, float beta) {
  size_t gid = UDONDNN_LARGE_POINTWISE_Y() * UDONDNN_LARGE_POINTWISE_WIDTH +
               UDONDNN_LARGE_POINTWISE_X();
  float value = 0.0f;
  if (gid < (size_t)words) {
    int operation = options & 255;
    int nan_propagation = (options >> 8) & 255;
    size_t a_index = udondnn_large_broadcast_index(gid, a_nc, a_hw, c_nc, c_hw);
    size_t b_index = udondnn_large_broadcast_index(gid, b_nc, b_hw, c_nc, c_hw);
    float a = alpha_a * a_input[a_index];
    float b = alpha_b * b_input[b_index];
    float combined = a + b;
    if (operation == 1)
      combined = a * b;
    if (operation == 2) {
      if (nan_propagation != 0) {
        if (a != a)
          combined = a;
        else if (b != b)
          combined = b;
        else {
          combined = b;
          if (a < b)
            combined = a;
        }
      } else if (a != a) {
        combined = b;
        if (b != b)
          combined = 0.0f;
      } else if (b != b)
        combined = a;
      else {
        combined = b;
        if (a < b)
          combined = a;
      }
    }
    if (operation == 3) {
      if (nan_propagation != 0) {
        if (a != a)
          combined = a;
        else if (b != b)
          combined = b;
        else {
          combined = b;
          if (a > b)
            combined = a;
        }
      } else if (a != a) {
        combined = b;
        if (b != b)
          combined = 0.0f;
      } else if (b != b)
        combined = a;
      else {
        combined = b;
        if (a > b)
          combined = a;
      }
    }
    value = combined;
    if (beta != 0.0f)
      value += beta * previous[gid];
  }
  output[gid] = value;
}
