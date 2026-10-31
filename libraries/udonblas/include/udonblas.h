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
  UDONBLAS_STATUS_EXECUTION_FAILED = 13
} udonblasStatus_t;

typedef enum udonblasOperation_t {
  UDONBLAS_OP_N = 0,
  UDONBLAS_OP_T = 1
} udonblasOperation_t;

enum {
  UDONBLAS_SGEMM_MAX_M = 16,
  UDONBLAS_SGEMM_MAX_N = 16,
  UDONBLAS_SGEMM_MAX_K = 16
};

udonblasStatus_t udonblasCreate(udonblasHandle_t *handle);
udonblasStatus_t udonblasDestroy(udonblasHandle_t handle);

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
