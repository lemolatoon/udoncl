#ifdef __OPENCL_C_VERSION__
#define UDONBLAS_SDOT_PRODUCT_GID() ((size_t)get_global_id(0))
#define UDONBLAS_SDOT_REDUCE_GID() ((size_t)get_global_id(0))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonblas_sdot_product_cpu_gid;
extern size_t udonblas_sdot_reduce_cpu_gid;
#define UDONBLAS_SDOT_PRODUCT_GID() (udonblas_sdot_product_cpu_gid)
#define UDONBLAS_SDOT_REDUCE_GID() (udonblas_sdot_reduce_cpu_gid)
#define __kernel
#define __global
#endif

#define UDONBLAS_SDOT_X_OFFSET 0u
#define UDONBLAS_SDOT_Y_OFFSET 256u
#define UDONBLAS_SDOT_RESULT_OFFSET 768u
#define UDONBLAS_SDOT_CAPACITY 256u

/* The product stage exposes one independent fragment per logical vector
 * element. Negative increments start at the last physical element, matching
 * the cuBLAS vector traversal contract. */
__kernel void udonblas_sdot_product(__global const float *input,
                                    __global float *output, int n, int incx,
                                    int incy) {
  uint gid = (uint)UDONBLAS_SDOT_PRODUCT_GID();
  float value = input[gid];
  if (gid >= UDONBLAS_SDOT_RESULT_OFFSET &&
      gid < UDONBLAS_SDOT_RESULT_OFFSET + UDONBLAS_SDOT_CAPACITY) {
    uint element = gid - UDONBLAS_SDOT_RESULT_OFFSET;
    value = 0.0f;
    if (element < (uint)n) {
      int x_start = incx < 0 ? (1 - n) * incx : 0;
      int y_start = incy < 0 ? (1 - n) * incy : 0;
      int x_index = x_start + (int)element * incx;
      int y_index = y_start + (int)element * incy;
      value = input[UDONBLAS_SDOT_X_OFFSET + ((uint)x_index & 255u)] *
              input[UDONBLAS_SDOT_Y_OFFSET + ((uint)y_index & 255u)];
    }
  }
  output[gid] = value;
}

/* Eight ping-pong passes reduce up to 256 products as a balanced binary tree.
 * Pass numbers are one based, so each fragment combines two adjacent partials
 * from the preceding pass. */
__kernel void udonblas_sdot_reduce(__global const float *input,
                                   __global float *output, int n, int pass) {
  uint gid = (uint)UDONBLAS_SDOT_REDUCE_GID();
  float value = input[gid];
  if (gid >= UDONBLAS_SDOT_RESULT_OFFSET &&
      gid < UDONBLAS_SDOT_RESULT_OFFSET + UDONBLAS_SDOT_CAPACITY) {
    uint element = gid - UDONBLAS_SDOT_RESULT_OFFSET;
    uint divisor = 1u << (uint)pass;
    uint previous_divisor = divisor >> 1u;
    uint active = ((uint)n + divisor - 1u) / divisor;
    uint previous_active = ((uint)n + previous_divisor - 1u) / previous_divisor;
    value = 0.0f;
    if (element < active) {
      uint first = element * 2u;
      value = input[UDONBLAS_SDOT_RESULT_OFFSET + (first & 255u)];
      if (first + 1u < previous_active)
        value += input[UDONBLAS_SDOT_RESULT_OFFSET + ((first + 1u) & 255u)];
    }
  }
  output[gid] = value;
}
