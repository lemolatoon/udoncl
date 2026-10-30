#ifndef __OPENCL_C_VERSION__
#include <stddef.h>
typedef unsigned int uint;
extern size_t mnist_cpu_gid;
static inline size_t get_global_id(unsigned dimension) {
  (void)dimension;
  return mnist_cpu_gid;
}
#define __kernel
#define __global
#endif

#define MNIST_INPUT_OFFSET 0u
#define MNIST_CONV_WEIGHT_OFFSET 784u
#define MNIST_CONV_BIAS_OFFSET 1184u
#define MNIST_DENSE_WEIGHT_OFFSET 1200u
#define MNIST_DENSE_BIAS_OFFSET 3504u

// One fragment owns one output word. The first sixteen fragments evaluate a
// class; weights for classes 10-15 are zero-padded by the Host. Remaining
// fragments write zero so the current symmetric 4096-word gather profile can
// be used without reading an uninitialized output region.
__kernel void mnist_cnn(__global const float *packed, __global float *output) {
  uint gid = (uint)get_global_id(0);
  uint digit = gid & 15u;
  float logit = 0.0f;
  if (gid < 16u) {
    logit = packed[MNIST_DENSE_BIAS_OFFSET + digit];
    for (uint pool_y = 0; pool_y < 3u; ++pool_y) {
      for (uint pool_x = 0; pool_x < 3u; ++pool_x) {
        for (uint channel = 0; channel < 16u; ++channel) {
          float pooled = 0.0f;
          for (uint offset_y = 0; offset_y < 8u; ++offset_y) {
            for (uint offset_x = 0; offset_x < 8u; ++offset_x) {
              float activation = packed[MNIST_CONV_BIAS_OFFSET + channel];
              for (uint kernel_y = 0; kernel_y < 5u; ++kernel_y) {
                for (uint kernel_x = 0; kernel_x < 5u; ++kernel_x) {
                  uint image_y = pool_y * 8u + offset_y + kernel_y;
                  uint image_x = pool_x * 8u + offset_x + kernel_x;
                  uint image_index = image_y * 28u + image_x;
                  uint weight_index = (channel * 5u + kernel_y) * 5u + kernel_x;
                  activation += packed[MNIST_INPUT_OFFSET + image_index] *
                                packed[MNIST_CONV_WEIGHT_OFFSET + weight_index];
                }
              }
              pooled += activation > 0.0f ? activation : 0.0f;
            }
          }
          uint feature = (pool_y * 3u + pool_x) * 16u + channel;
          uint dense_index = digit * 144u + feature;
          logit += pooled * (1.0f / 64.0f) *
                   packed[MNIST_DENSE_WEIGHT_OFFSET + dense_index];
        }
      }
    }
  }
  output[gid] = logit;
}
