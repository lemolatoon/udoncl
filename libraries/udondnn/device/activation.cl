#ifndef __OPENCL_C_VERSION__
#include <math.h>
#include <stddef.h>
typedef unsigned int uint;
extern size_t udondnn_activation_cpu_gid;
static inline size_t get_global_id(unsigned dimension) {
  (void)dimension;
  return udondnn_activation_cpu_gid;
}
#define __kernel
#define __global
#define exp expf
#endif

#define UDONDNN_ACTIVATION_WORDS 2048u
#define UDONDNN_ACTIVATION_PREVIOUS_OUTPUT_OFFSET 512u
#define UDONDNN_ACTIVATION_RESULT_OFFSET 1024u

__kernel void udondnn_activation_forward(__global const float *packed,
                                         __global float *output, int words,
                                         int mode, int nan_propagation,
                                         float coefficient, float alpha,
                                         float beta) {
  uint gid = (uint)get_global_id(0);
  float value = packed[gid];
  if (gid >= UDONDNN_ACTIVATION_RESULT_OFFSET &&
      gid < UDONDNN_ACTIVATION_RESULT_OFFSET + (uint)words) {
    uint element = gid - UDONDNN_ACTIVATION_RESULT_OFFSET;
    float input = packed[element & 511u];
    float activated = input;
    if (input != input && nan_propagation == 0)
      input = 0.0f;
    if (input != input) {
      activated = input;
    } else if (mode == 0) {
      activated = 1.0f / (1.0f + exp(-input));
    } else if (mode == 1) {
      activated = 0.0f;
      if (input > 0.0f)
        activated = input;
    } else if (mode == 2) {
      activated = 2.0f / (1.0f + exp(-2.0f * input)) - 1.0f;
    } else if (mode == 3) {
      activated = 0.0f;
      if (input > 0.0f)
        activated = input;
      if (activated > coefficient)
        activated = coefficient;
    } else if (mode == 4) {
      if (input > 0.0f)
        activated = input;
      else
        activated = coefficient * (exp(input) - 1.0f);
    } else {
      activated = input;
    }
    value = alpha * activated +
            beta * packed[UDONDNN_ACTIVATION_PREVIOUS_OUTPUT_OFFSET +
                          (element & 511u)];
  }
  output[gid] = value;
}
