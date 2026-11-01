#ifndef __OPENCL_C_VERSION__
#include <math.h>
#include <stddef.h>
typedef unsigned int uint;
extern size_t udondnn_batchnorm_cpu_gid;
static inline size_t get_global_id(unsigned dimension) {
  (void)dimension;
  return udondnn_batchnorm_cpu_gid;
}
#define __kernel
#define __global
#define sqrt sqrtf
#endif

#define UDONDNN_BATCHNORM_WORDS 4096u
#define UDONDNN_BATCHNORM_PREVIOUS_OUTPUT_OFFSET 512u
#define UDONDNN_BATCHNORM_SCALE_OFFSET 1024u
#define UDONDNN_BATCHNORM_BIAS_OFFSET 1536u
#define UDONDNN_BATCHNORM_MEAN_OFFSET 2048u
#define UDONDNN_BATCHNORM_VARIANCE_OFFSET 2560u
#define UDONDNN_BATCHNORM_RESULT_OFFSET 3072u

__kernel void udondnn_batchnorm_inference(__global const float *packed,
                                          __global float *output,
                                          int input_shape, int mode,
                                          float epsilon, float alpha,
                                          float beta) {
  uint gid = (uint)get_global_id(0);
  float value = packed[gid];
  uint input_n = (uint)input_shape & 15u;
  uint input_c = ((uint)input_shape >> 4u) & 15u;
  uint input_h = ((uint)input_shape >> 8u) & 15u;
  uint input_w = ((uint)input_shape >> 12u) & 15u;
  uint sample_words = input_c * input_h * input_w;
  uint words = input_n * sample_words;
  if (gid >= UDONDNN_BATCHNORM_RESULT_OFFSET &&
      gid < UDONDNN_BATCHNORM_RESULT_OFFSET + words) {
    uint element = gid - UDONDNN_BATCHNORM_RESULT_OFFSET;
    uint parameter = element % sample_words;
    if (mode != 0)
      parameter = (element / (input_h * input_w)) % input_c;
    float normalized =
        (packed[element & 511u] -
         packed[UDONDNN_BATCHNORM_MEAN_OFFSET + (parameter & 511u)]) /
        sqrt(packed[UDONDNN_BATCHNORM_VARIANCE_OFFSET + (parameter & 511u)] +
             epsilon);
    normalized =
        normalized *
            packed[UDONDNN_BATCHNORM_SCALE_OFFSET + (parameter & 511u)] +
        packed[UDONDNN_BATCHNORM_BIAS_OFFSET + (parameter & 511u)];
    value =
        alpha * normalized +
        beta *
            packed[UDONDNN_BATCHNORM_PREVIOUS_OUTPUT_OFFSET + (element & 511u)];
  }
  output[gid] = value;
}
