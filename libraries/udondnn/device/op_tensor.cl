#ifndef __OPENCL_C_VERSION__
#include <stddef.h>
typedef unsigned int uint;
extern size_t udondnn_op_tensor_cpu_gid;
static inline size_t get_global_id(unsigned dimension) {
  (void)dimension;
  return udondnn_op_tensor_cpu_gid;
}
#define __kernel
#define __global
#endif

#define UDONDNN_OP_TENSOR_WORDS 2048u
#define UDONDNN_OP_TENSOR_B_OFFSET 512u
#define UDONDNN_OP_TENSOR_C_OFFSET 1024u
#define UDONDNN_OP_TENSOR_RESULT_OFFSET 1536u

__kernel void udondnn_op_tensor(__global const float *packed,
                                __global float *output, int a_shape,
                                int b_shape, int c_shape, int operation,
                                int nan_propagation, float alpha_a,
                                float alpha_b, float beta) {
  uint gid = (uint)get_global_id(0);
  float value = packed[gid];
  uint output_words = ((uint)c_shape & 15u) * (((uint)c_shape >> 4u) & 15u) *
                      (((uint)c_shape >> 8u) & 15u) *
                      (((uint)c_shape >> 12u) & 15u);
  if (gid >= UDONDNN_OP_TENSOR_RESULT_OFFSET &&
      gid < UDONDNN_OP_TENSOR_RESULT_OFFSET + output_words) {
    uint element = gid - UDONDNN_OP_TENSOR_RESULT_OFFSET;
    uint output_c = ((uint)c_shape >> 4u) & 15u;
    uint output_h = ((uint)c_shape >> 8u) & 15u;
    uint output_w = ((uint)c_shape >> 12u) & 15u;
    uint coordinate_w = element % output_w;
    uint coordinate_h = (element / output_w) % output_h;
    uint coordinate_c = (element / (output_w * output_h)) % output_c;
    uint coordinate_n = element / (output_w * output_h * output_c);
    uint a_n = (uint)a_shape & 15u;
    uint a_c = ((uint)a_shape >> 4u) & 15u;
    uint a_h = ((uint)a_shape >> 8u) & 15u;
    uint a_w = ((uint)a_shape >> 12u) & 15u;
    uint a_coordinate_n = coordinate_n;
    uint a_coordinate_c = coordinate_c;
    uint a_coordinate_h = coordinate_h;
    uint a_coordinate_w = coordinate_w;
    if (a_n == 1u)
      a_coordinate_n = 0u;
    if (a_c == 1u)
      a_coordinate_c = 0u;
    if (a_h == 1u)
      a_coordinate_h = 0u;
    if (a_w == 1u)
      a_coordinate_w = 0u;
    uint a_index =
        ((a_coordinate_n * a_c + a_coordinate_c) * a_h + a_coordinate_h) * a_w +
        a_coordinate_w;
    uint b_n = (uint)b_shape & 15u;
    uint b_c = ((uint)b_shape >> 4u) & 15u;
    uint b_h = ((uint)b_shape >> 8u) & 15u;
    uint b_w = ((uint)b_shape >> 12u) & 15u;
    uint b_coordinate_n = coordinate_n;
    uint b_coordinate_c = coordinate_c;
    uint b_coordinate_h = coordinate_h;
    uint b_coordinate_w = coordinate_w;
    if (b_n == 1u)
      b_coordinate_n = 0u;
    if (b_c == 1u)
      b_coordinate_c = 0u;
    if (b_h == 1u)
      b_coordinate_h = 0u;
    if (b_w == 1u)
      b_coordinate_w = 0u;
    uint b_index =
        ((b_coordinate_n * b_c + b_coordinate_c) * b_h + b_coordinate_h) * b_w +
        b_coordinate_w;
    float a = alpha_a * packed[a_index & 511u];
    float b = alpha_b * packed[UDONDNN_OP_TENSOR_B_OFFSET + (b_index & 511u)];
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
    value =
        combined + beta * packed[UDONDNN_OP_TENSOR_C_OFFSET + (element & 511u)];
  }
  output[gid] = value;
}
