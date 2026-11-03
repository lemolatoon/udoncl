#include <udonblas.h>

static float matrix_a[144] = {2.0f};
static float matrix_b[1440] = {3.0f};
static float matrix_c[10];

void run_udonblas_sgemm(void) {
  udonblasHandle_t handle = 0;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  (void)udonblasCreate(&handle);
  (void)udonblasSgemm(handle, UDONBLAS_OP_N, UDONBLAS_OP_N, 1, 10, 144, &alpha,
                      matrix_a, 1, matrix_b, 144, &beta, matrix_c, 1);
  (void)udonblasDestroy(handle);
}
