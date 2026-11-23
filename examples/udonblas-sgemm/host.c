#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
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

int run_udon_blas_sgemm(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context context = 0;
  cl_command_queue queue = 0;
  cl_mem device_a = 0;
  cl_mem device_b = 0;
  cl_mem device_c = 0;
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
  device_a =
      clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(matrix_a), 0, &error);
  if (!device_a || error != CL_SUCCESS)
    goto cleanup;
  device_b =
      clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(matrix_b), 0, &error);
  if (!device_b || error != CL_SUCCESS)
    goto cleanup;
  device_c =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(matrix_c), 0, &error);
  if (!device_c || error != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, device_a, CL_FALSE, 0, sizeof(matrix_a),
                           matrix_a, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, device_b, CL_FALSE, 0, sizeof(matrix_b),
                           matrix_b, 0, 0, 0) != CL_SUCCESS)
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
                    (const float *)device_a, 1, (const float *)device_b, 144,
                    &beta, (float *)device_c, 1);
  if (status != UDONBLAS_STATUS_SUCCESS ||
      clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, sizeof(matrix_c),
                          matrix_c, 0, 0, 0) != CL_SUCCESS)
    goto cleanup;
  last_phase = 3;
cleanup:
  if (handle && udonblasDestroy(handle) != UDONBLAS_STATUS_SUCCESS)
    status = UDONBLAS_STATUS_EXECUTION_FAILED;
  if ((device_c && clReleaseMemObject(device_c) != CL_SUCCESS) ||
      (device_b && clReleaseMemObject(device_b) != CL_SUCCESS) ||
      (device_a && clReleaseMemObject(device_a) != CL_SUCCESS) ||
      (queue && clReleaseCommandQueue(queue) != CL_SUCCESS) ||
      (context && clReleaseContext(context) != CL_SUCCESS))
    status = UDONBLAS_STATUS_EXECUTION_FAILED;
  return status;
}

float get_udon_blas_sgemm_result(int column) { return matrix_c[column]; }

int get_udon_blas_sgemm_phase(void) { return last_phase; }
