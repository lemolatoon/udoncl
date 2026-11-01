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
  UDONBLAS_SGEMM_MAX_K = 16,
  UDONBLAS_Q4_BLOCK_VALUES = 32,
  UDONBLAS_Q4_EXPANDED_BLOCK_WORDS = 17,
  UDONBLAS_Q4_GEMV_MAX_M = 64,
  UDONBLAS_Q4_GEMV_MAX_K = 128
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

/* Row-major matrix-vector multiply for streamed signed-Q4 weight tiles:
 * y = alpha * dequantize(A) * x + beta * y.
 *
 * Each group of 32 weights occupies 17 float words in A: 16 words containing
 * the numeric packed-byte values (0..255), followed by one float32 scale. Two
 * signed four-bit values use two's-complement nibbles in each byte, low nibble
 * first. This expanded Device layout is produced from the compact 18-byte ONNX
 * external-data block only for the active tile; the full model stays compact.
 * lda is the expanded row stride in float words. k must be a multiple of 32. */
udonblasStatus_t udonblasSgemvQ4(udonblasHandle_t handle, int m, int k,
                                 const float *alpha, const float *expanded_a,
                                 int lda, const float *x, int incx,
                                 const float *beta, float *y, int incy);

#ifdef __cplusplus
}
#endif

#endif
