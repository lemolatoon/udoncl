#ifndef __OPENCL_C_VERSION__
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonaten_embedding_cpu_gid;
static inline size_t get_global_id(unsigned dimension) {
  (void)dimension;
  return udonaten_embedding_cpu_gid;
}
#define __kernel
#define __global
#endif

#define UDONATEN_EMBEDDING_WEIGHT_OFFSET 0u
#define UDONATEN_EMBEDDING_INDICES_OFFSET 512u
#define UDONATEN_EMBEDDING_RESULT_OFFSET 1024u

__kernel void udonaten_embedding(__global const float *packed,
                                 __global float *output, int outer, int rows,
                                 float rows_float, int inner, int indices_words,
                                 int output_words) {
  uint gid = (uint)get_global_id(0);
  float value = packed[gid];
  if (gid >= UDONATEN_EMBEDDING_RESULT_OFFSET &&
      gid < UDONATEN_EMBEDDING_RESULT_OFFSET + (uint)output_words) {
    uint element = gid - UDONATEN_EMBEDDING_RESULT_OFFSET;
    uint inner_index = element % (uint)inner;
    uint group = element / (uint)inner;
    uint index_position = group % (uint)indices_words;
    uint outer_index = group / (uint)indices_words;
    float target =
        packed[UDONATEN_EMBEDDING_INDICES_OFFSET + (index_position & 511u)];
    uint selected = 0u;
    float candidate_value = 0.0f;
    if (target < 0.0f)
      target += rows_float;
    for (uint candidate = 0u; candidate < 512u; ++candidate) {
      if (candidate < (uint)rows && candidate_value == target)
        selected = candidate;
      candidate_value += 1.0f;
    }
    uint source =
        ((outer_index * (uint)rows + selected) * (uint)inner + inner_index) &
        511u;
    value = packed[UDONATEN_EMBEDDING_WEIGHT_OFFSET + source];
  }
  output[gid] = value;
}
