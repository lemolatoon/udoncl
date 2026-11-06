#include <udonblas.h>

#define REPEAT_2(value) value, value
#define REPEAT_4(value) REPEAT_2(value), REPEAT_2(value)
#define REPEAT_8(value) REPEAT_4(value), REPEAT_4(value)
#define REPEAT_16(value) REPEAT_8(value), REPEAT_8(value)
#define REPEAT_32(value) REPEAT_16(value), REPEAT_16(value)
#define REPEAT_64(value) REPEAT_32(value), REPEAT_32(value)
#define REPEAT_128(value) REPEAT_64(value), REPEAT_64(value)
#define REPEAT_144(value) REPEAT_128(value), REPEAT_16(value)

#define A_ROW                                                                  \
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,   \
      13.0f, 14.0f, 15.0f, 16.0f

static float matrix_a[144] = {A_ROW, A_ROW, A_ROW, A_ROW, A_ROW,
                              A_ROW, A_ROW, A_ROW, A_ROW};
static float matrix_b[1440] = {
    REPEAT_144(1.0f), REPEAT_144(2.0f), REPEAT_144(3.0f), REPEAT_144(4.0f),
    REPEAT_144(5.0f), REPEAT_144(6.0f), REPEAT_144(7.0f), REPEAT_144(8.0f),
    REPEAT_144(9.0f), REPEAT_144(10.0f)};
static float matrix_c[10];
static int last_phase;

int run_udonblas_sgemm(void) {
  udonblasHandle_t handle = 0;
  cl_command_queue stream = 0;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  udonblasStatus_t status;

  last_phase = 1;
  status = udonblasCreate(&handle);
  if (status != UDONBLAS_STATUS_SUCCESS)
    return status;
  status = udonblasGetStream(handle, &stream);
  if (status != UDONBLAS_STATUS_SUCCESS || !stream)
    return status != UDONBLAS_STATUS_SUCCESS ? status
                                             : UDONBLAS_STATUS_NOT_INITIALIZED;
  status = udonblasSetStream(handle, stream);
  if (status != UDONBLAS_STATUS_SUCCESS)
    return status;
  last_phase = 2;
  status =
      udonblasSgemm(handle, UDONBLAS_OP_N, UDONBLAS_OP_N, 1, 10, 144, &alpha,
                    matrix_a, 1, matrix_b, 144, &beta, matrix_c, 1);
  if (status != UDONBLAS_STATUS_SUCCESS)
    return status;
  last_phase = 3;
  return udonblasDestroy(handle);
}

float get_udonblas_sgemm_result(int column) { return matrix_c[column]; }

int get_udonblas_sgemm_phase(void) { return last_phase; }
