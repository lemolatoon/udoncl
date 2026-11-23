#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udonblas.h>
#include <udoncl/export.h>

UDONCL_PROGRAM("udonblas_sgemm_m1n10k144");

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

static float host_a[144] = {A_ROW, A_ROW, A_ROW, A_ROW, A_ROW,
                            A_ROW, A_ROW, A_ROW, A_ROW};
static float host_b[1440] = {
    REPEAT_144(1.0f), REPEAT_144(2.0f), REPEAT_144(3.0f), REPEAT_144(4.0f),
    REPEAT_144(5.0f), REPEAT_144(6.0f), REPEAT_144(7.0f), REPEAT_144(8.0f),
    REPEAT_144(9.0f), REPEAT_144(10.0f)};
static float host_c[10];
static int last_phase;

/* The Device buffers of A, B and C are public: U# may write matrix_a and
 * matrix_b (CL_MEM_READ_ONLY) and read matrix_c (CL_MEM_WRITE_ONLY). The
 * export uploads the fixture itself, so a caller only has to run it. */
UDONCL_PUBLIC(float) cl_mem matrix_a, matrix_b, matrix_c;

UDONCL_EXPORT int run_udon_blas_sgemm(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context context = 0;
  cl_command_queue queue = 0;
  udonblasHandle_t handle = 0;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  udonblasStatus_t status = UDONBLAS_STATUS_EXECUTION_FAILED;
  cl_int error = CL_SUCCESS;

  last_phase = 1;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    goto cleanup;
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (!context || error != CL_SUCCESS)
    goto cleanup;
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (!queue || error != CL_SUCCESS)
    goto cleanup;
  matrix_a =
      clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(host_a), 0, &error);
  if (!matrix_a || error != CL_SUCCESS)
    goto cleanup;
  matrix_b =
      clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(host_b), 0, &error);
  if (!matrix_b || error != CL_SUCCESS)
    goto cleanup;
  matrix_c =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(host_c), 0, &error);
  if (!matrix_c || error != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, matrix_a, CL_FALSE, 0, sizeof(host_a), host_a,
                           0, 0, 0) != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, matrix_b, CL_FALSE, 0, sizeof(host_b), host_b,
                           0, 0, 0) != CL_SUCCESS)
    goto cleanup;
  status = udonblasCreate(&handle);
  if (status != UDONBLAS_STATUS_SUCCESS)
    goto cleanup;
  status = udonblasSetStream(handle, queue);
  if (status != UDONBLAS_STATUS_SUCCESS)
    goto cleanup;
  last_phase = 2;
  status =
      udonblasSgemm(handle, UDONBLAS_OP_N, UDONBLAS_OP_N, 1, 10, 144, &alpha,
                    (const float *)matrix_a, 1, (const float *)matrix_b, 144,
                    &beta, (float *)matrix_c, 1);
  if (status != UDONBLAS_STATUS_SUCCESS ||
      clEnqueueReadBuffer(queue, matrix_c, CL_TRUE, 0, sizeof(host_c), host_c,
                          0, 0, 0) != CL_SUCCESS)
    goto cleanup;
  last_phase = 3;
cleanup:
  if (handle && udonblasDestroy(handle) != UDONBLAS_STATUS_SUCCESS)
    status = UDONBLAS_STATUS_EXECUTION_FAILED;
  if ((matrix_c && clReleaseMemObject(matrix_c) != CL_SUCCESS) ||
      (matrix_b && clReleaseMemObject(matrix_b) != CL_SUCCESS) ||
      (matrix_a && clReleaseMemObject(matrix_a) != CL_SUCCESS) ||
      (queue && clReleaseCommandQueue(queue) != CL_SUCCESS) ||
      (context && clReleaseContext(context) != CL_SUCCESS))
    status = UDONBLAS_STATUS_EXECUTION_FAILED;
  return status;
}

UDONCL_EXPORT float get_udon_blas_sgemm_result(int column) {
  return host_c[column];
}

/* The phase is polled while RunUdonBlasSgemm is suspended. */
UDONCL_EXPORT_WHILE_BUSY int get_udon_blas_sgemm_phase(void) {
  return last_phase;
}
