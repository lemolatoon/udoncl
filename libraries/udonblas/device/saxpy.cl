#ifdef __OPENCL_C_VERSION__
#define UDONBLAS_SAXPY_GID() ((size_t)get_global_id(0))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonblas_saxpy_cpu_gid;
#define UDONBLAS_SAXPY_GID() (udonblas_saxpy_cpu_gid)
#define __kernel
#define __global
#endif

#define UDONBLAS_SAXPY_WORDS 1024u
#define UDONBLAS_SAXPY_X_OFFSET 0u
#define UDONBLAS_SAXPY_Y_OFFSET 256u
#define UDONBLAS_SAXPY_RESULT_OFFSET 768u

/* One fragment owns one logical vector element. Negative increments begin at
 * the last physical element, matching the BLAS vector traversal contract. */
__kernel void udonblas_saxpy(__global const float *input,
                             __global float *output, int n, int incx, int incy,
                             float alpha) {
  uint gid = (uint)UDONBLAS_SAXPY_GID();
  float value = input[gid];
  if (gid >= UDONBLAS_SAXPY_RESULT_OFFSET &&
      gid < UDONBLAS_SAXPY_RESULT_OFFSET + (uint)n) {
    uint element = gid - UDONBLAS_SAXPY_RESULT_OFFSET;
    int x_start = incx < 0 ? (1 - n) * incx : 0;
    int y_start = incy < 0 ? (1 - n) * incy : 0;
    int x_index = x_start + (int)element * incx;
    int y_index = y_start + (int)element * incy;
    value = alpha * input[UDONBLAS_SAXPY_X_OFFSET + ((uint)x_index & 255u)] +
            input[UDONBLAS_SAXPY_Y_OFFSET + ((uint)y_index & 255u)];
  }
  output[gid] = value;
}
