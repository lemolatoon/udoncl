#ifndef UDONBLAS_H
#define UDONBLAS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t udonblasHandle_t;

typedef enum udonblasStatus_t {
  UDONBLAS_STATUS_SUCCESS = 0,
  UDONBLAS_STATUS_NOT_INITIALIZED = 1,
  UDONBLAS_STATUS_ALLOC_FAILED = 3,
  UDONBLAS_STATUS_INVALID_VALUE = 7,
  UDONBLAS_STATUS_EXECUTION_FAILED = 13,
  UDONBLAS_STATUS_NOT_SUPPORTED = 15
} udonblasStatus_t;

typedef enum udonblasOperation_t {
  UDONBLAS_OP_N = 0,
  UDONBLAS_OP_T = 1,
  UDONBLAS_OP_C = 2
} udonblasOperation_t;

enum {
  UDONBLAS_SGEMM_MAX_M = 16,
  UDONBLAS_SGEMM_MAX_N = 16,
  UDONBLAS_SGEMM_MAX_K = 16,
  UDONBLAS_SGEMV_MAX_M = 16,
  UDONBLAS_SGEMV_MAX_N = 16,
  UDONBLAS_SAXPY_MAX_N = 256,
  UDONBLAS_SDOT_MAX_N = 256
};

udonblasStatus_t udonblasCreate(udonblasHandle_t *handle);
udonblasStatus_t udonblasDestroy(udonblasHandle_t handle);

/* Level-1 vector update with the cuBLAS contract: y = alpha * x + y.
 * Positive and negative non-zero increments are supported. */
udonblasStatus_t udonblasSaxpy(udonblasHandle_t handle, int n,
                               const float *alpha, const float *x, int incx,
                               float *y, int incy);

/* CUDA 13.0 cublasSdot contract. Computes result = sum(x[i] * y[i]) and
 * writes zero for n <= 0. Positive and negative non-zero increments are
 * supported by the current 256-element Device reduction profile. */
udonblasStatus_t udonblasSdot(udonblasHandle_t handle, int n, const float *x,
                              int incx, const float *y, int incy,
                              float *result);

/* CUDA 13.0 cublasSgemv contract. A is column-major m by n and
 * y = alpha * op(A) * x + beta * y. For real data, OP_C equals OP_T. */
udonblasStatus_t udonblasSgemv(udonblasHandle_t handle,
                               udonblasOperation_t trans, int m, int n,
                               const float *alpha, const float *a, int lda,
                               const float *x, int incx, const float *beta,
                               float *y, int incy);

/* Column-major SGEMM with the cuBLAS operand contract:
 * C = alpha * op(A) * op(B) + beta * C.
 *
 * The initial UdonCL gather profile supports dimensions up to 16 and one
 * serialized handle per UdonBehaviour instance. The stable API is intended to
 * remain unchanged when tiled multi-pass kernels lift those profile limits. */
udonblasStatus_t udonblasSgemm(udonblasHandle_t handle,
                               udonblasOperation_t transa,
                               udonblasOperation_t transb, int m, int n, int k,
                               const float *alpha, const float *a, int lda,
                               const float *b, int ldb, const float *beta,
                               float *c, int ldc);

#ifdef __cplusplus
}
#endif

#endif
