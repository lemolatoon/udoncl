#ifndef __OPENCL_C_VERSION__
#include <math.h>
#include <stddef.h>
typedef unsigned int uint;
extern size_t udondnn_softmax_cpu_gid;
static inline size_t get_global_id(unsigned dimension) {
  (void)dimension;
  return udondnn_softmax_cpu_gid;
}
#define __kernel
#define __global
#define exp expf
#define log logf
#endif

#define UDONDNN_SOFTMAX_WORDS 2048u
#define UDONDNN_SOFTMAX_PREVIOUS_OUTPUT_OFFSET 512u
#define UDONDNN_SOFTMAX_RESULT_OFFSET 1024u

__kernel void udondnn_softmax_forward(__global const float *packed,
                                      __global float *output, int input_shape,
                                      int algorithm, int mode, float alpha,
                                      float beta) {
  uint gid = (uint)get_global_id(0);
  float value = packed[gid];
  uint input_n = (uint)input_shape & 15u;
  uint input_c = ((uint)input_shape >> 4u) & 15u;
  uint input_h = ((uint)input_shape >> 8u) & 15u;
  uint input_w = ((uint)input_shape >> 12u) & 15u;
  uint sample_words = input_c * input_h * input_w;
  uint words = input_n * sample_words;
  if (gid >= UDONDNN_SOFTMAX_RESULT_OFFSET &&
      gid < UDONDNN_SOFTMAX_RESULT_OFFSET + words) {
    uint element = gid - UDONDNN_SOFTMAX_RESULT_OFFSET;
    uint x = element % input_w;
    uint y = (element / input_w) % input_h;
    uint batch = element / sample_words;
    float maximum = -3.402823466e+38f;
    if (mode == 0) {
      for (uint candidate = 0u; candidate < 512u; ++candidate) {
        if (candidate < sample_words) {
          uint candidate_index = batch * sample_words + candidate;
          float sample = packed[candidate_index & 511u];
          if (sample > maximum)
            maximum = sample;
        }
      }
    } else {
      for (uint candidate = 0u; candidate < 15u; ++candidate) {
        if (candidate < input_c) {
          uint candidate_index =
              ((batch * input_c + candidate) * input_h + y) * input_w + x;
          float sample = packed[candidate_index & 511u];
          if (sample > maximum)
            maximum = sample;
        }
      }
    }
    float denominator = 0.0f;
    if (mode == 0) {
      for (uint candidate = 0u; candidate < 512u; ++candidate) {
        if (candidate < sample_words) {
          uint candidate_index = batch * sample_words + candidate;
          denominator += exp(packed[candidate_index & 511u] - maximum);
        }
      }
    } else {
      for (uint candidate = 0u; candidate < 15u; ++candidate) {
        if (candidate < input_c) {
          uint candidate_index =
              ((batch * input_c + candidate) * input_h + y) * input_w + x;
          denominator += exp(packed[candidate_index & 511u] - maximum);
        }
      }
    }
    float normalized = exp(packed[element & 511u] - maximum) / denominator;
    if (algorithm == 2)
      normalized = packed[element & 511u] - maximum - log(denominator);
    value =
        alpha * normalized +
        beta *
            packed[UDONDNN_SOFTMAX_PREVIOUS_OUTPUT_OFFSET + (element & 511u)];
  }
  output[gid] = value;
}
