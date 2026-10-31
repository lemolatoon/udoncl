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

#define MNIST_WORDS 10800u
#define MNIST_IMAGE_OFFSET 0u
#define MNIST_CONV_WEIGHT_OFFSET 784u
#define MNIST_CONV_BIAS_OFFSET 1184u
#define MNIST_DENSE_WEIGHT_OFFSET 1200u
#define MNIST_DENSE_BIAS_OFFSET 2640u
#define MNIST_CONV_OUTPUT_WORDS 9216u
#define MNIST_CARRIED_DENSE_WEIGHT_OFFSET 9216u
#define MNIST_CARRIED_DENSE_BIAS_OFFSET 10656u
#define MNIST_POOL_OUTPUT_WORDS 144u
#define MNIST_POOL_DENSE_WEIGHT_OFFSET 144u
#define MNIST_POOL_DENSE_BIAS_OFFSET 1584u

#if !defined(MNIST_CONV) && !defined(MNIST_POOL) && !defined(MNIST_DENSE)
#define MNIST_CONV 1
#define MNIST_POOL 1
#define MNIST_DENSE 1
#endif

// CUDA-style output ownership: 9,216 fragments independently evaluate one
// convolution activation. Remaining fragments carry immutable parameters.
#if MNIST_CONV
__kernel void mnist_conv(__global const float *packed, __global float *output) {
  uint gid = (uint)get_global_id(0);
  float value = 0.0f;
  if (gid < MNIST_CONV_OUTPUT_WORDS) {
    uint channel = gid % 16u;
    uint pixel = (gid / 16u) % 576u;
    uint output_x = pixel % 24u;
    uint output_y = pixel / 24u;
    value = packed[MNIST_CONV_BIAS_OFFSET + channel];
    for (uint kernel_y = 0; kernel_y < 5u; ++kernel_y) {
      for (uint kernel_x = 0; kernel_x < 5u; ++kernel_x) {
        uint image_index = (output_y + kernel_y) * 28u + output_x + kernel_x;
        uint weight_index = (channel * 5u + kernel_y) * 5u + kernel_x;
        value += packed[MNIST_IMAGE_OFFSET + image_index] *
                 packed[MNIST_CONV_WEIGHT_OFFSET + weight_index];
      }
    }
    value = value > 0.0f ? value : 0.0f;
  } else if (gid < MNIST_CARRIED_DENSE_BIAS_OFFSET) {
    value = packed[MNIST_DENSE_WEIGHT_OFFSET + (gid + 864u) % 1440u];
  } else if (gid < MNIST_CARRIED_DENSE_BIAS_OFFSET + 10u) {
    value = packed[MNIST_DENSE_BIAS_OFFSET + (gid + 4u) % 10u];
  }
  output[gid] = value;
}
#endif

// 144 fragments independently reduce one 8x8 pooling window.
#if MNIST_POOL
__kernel void mnist_pool(__global const float *convolution,
                         __global float *output) {
  uint gid = (uint)get_global_id(0);
  float value = 0.0f;
  if (gid < MNIST_POOL_OUTPUT_WORDS) {
    uint channel = gid % 16u;
    uint pool_pixel = (gid / 16u) % 9u;
    uint pool_x = pool_pixel % 3u;
    uint pool_y = pool_pixel / 3u;
    for (uint offset_y = 0; offset_y < 8u; ++offset_y) {
      for (uint offset_x = 0; offset_x < 8u; ++offset_x) {
        uint conv_x = pool_x * 8u + offset_x;
        uint conv_y = pool_y * 8u + offset_y;
        value += convolution[(conv_y * 24u + conv_x) * 16u + channel];
      }
    }
    value *= 1.0f / 64.0f;
  } else if (gid < MNIST_POOL_DENSE_BIAS_OFFSET) {
    value =
        convolution[MNIST_CARRIED_DENSE_WEIGHT_OFFSET + (gid + 1296u) % 1440u];
  } else if (gid < MNIST_POOL_DENSE_BIAS_OFFSET + 10u) {
    value = convolution[MNIST_CARRIED_DENSE_BIAS_OFFSET + (gid + 6u) % 10u];
  }
  output[gid] = value;
}
#endif

// Ten fragments independently evaluate the class logits.
#if MNIST_DENSE
__kernel void mnist_dense(__global const float *pooled,
                          __global float *output) {
  uint gid = (uint)get_global_id(0);
  float value = 0.0f;
  if (gid < 10u) {
    uint digit = gid % 10u;
    value = pooled[MNIST_POOL_DENSE_BIAS_OFFSET + digit];
    for (uint feature = 0; feature < MNIST_POOL_OUTPUT_WORDS; ++feature)
      value += pooled[feature] *
               pooled[MNIST_POOL_DENSE_WEIGHT_OFFSET + digit * 144u + feature];
  }
  output[gid] = value;
}
#endif
