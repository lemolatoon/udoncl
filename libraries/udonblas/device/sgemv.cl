#ifdef __OPENCL_C_VERSION__
#define UDONBLAS_SGEMV_GID() ((size_t)get_global_id(0))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonblas_sgemv_cpu_gid;
#define UDONBLAS_SGEMV_GID() (udonblas_sgemv_cpu_gid)
#define __kernel
#define __global
#endif

#define UDONBLAS_SGEMV_A_OFFSET 0u
#define UDONBLAS_SGEMV_X_OFFSET 256u
#define UDONBLAS_SGEMV_Y_OFFSET 512u
#define UDONBLAS_SGEMV_RESULT_OFFSET 768u

/* One fragment owns one output vector element. OP_C is equivalent to OP_T for
 * this real-valued entry point. Negative increments follow BLAS traversal. */
__kernel void udonblas_sgemv(__global const float *input,
                             __global float *output, int trans, int m, int n,
                             int lda, int incx, int incy, float alpha,
                             float beta) {
  uint gid = (uint)UDONBLAS_SGEMV_GID();
  float value = input[gid];
  uint output_length = trans == 0 ? (uint)m : (uint)n;
  uint inner_length = trans == 0 ? (uint)n : (uint)m;
  if (gid >= UDONBLAS_SGEMV_RESULT_OFFSET &&
      gid < UDONBLAS_SGEMV_RESULT_OFFSET + output_length) {
    uint row = gid - UDONBLAS_SGEMV_RESULT_OFFSET;
    int x_start = incx < 0 ? (1 - (int)inner_length) * incx : 0;
    int y_start = incy < 0 ? (1 - (int)output_length) * incy : 0;
    float sum = 0.0f;
    for (uint inner = 0; inner < 16u; ++inner) {
      uint a_index =
          trans == 0 ? row + inner * (uint)lda : inner + row * (uint)lda;
      int x_index = x_start + (int)inner * incx;
      if (inner < inner_length)
        sum += input[UDONBLAS_SGEMV_A_OFFSET + (a_index & 255u)] *
               input[UDONBLAS_SGEMV_X_OFFSET + ((uint)x_index & 255u)];
    }
    int y_index = y_start + (int)row * incy;
    value = alpha * sum +
            beta * input[UDONBLAS_SGEMV_Y_OFFSET + ((uint)y_index & 255u)];
  }
  output[gid] = value;
}
