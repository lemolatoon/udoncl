#include "udonblas.h"
#include "udonblas_device_source.h"
#include <CL/cl.h>

#define UDONBLAS_WORDS 1024
#define UDONBLAS_MATRIX_WORDS 256
#define UDONBLAS_A_OFFSET 0
#define UDONBLAS_B_OFFSET 256
#define UDONBLAS_C_OFFSET 512
#define UDONBLAS_RESULT_OFFSET 768
#define UDONBLAS_HANDLE 1u
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel sgemm_kernel;
static cl_kernel sgemv_kernel;
static cl_kernel saxpy_kernel;
static cl_kernel sdot_product_kernel;
static cl_kernel sdot_reduce_kernel;
static cl_mem input_buffer;
static cl_mem output_buffer;
static float packed[UDONBLAS_WORDS];
static float result[UDONBLAS_MATRIX_WORDS];

static void clear_objects(void) {
  if (sdot_reduce_kernel)
    clReleaseKernel(sdot_reduce_kernel);
  if (sdot_product_kernel)
    clReleaseKernel(sdot_product_kernel);
  if (saxpy_kernel)
    clReleaseKernel(saxpy_kernel);
  if (sgemv_kernel)
    clReleaseKernel(sgemv_kernel);
  if (sgemm_kernel)
    clReleaseKernel(sgemm_kernel);
  if (program)
    clReleaseProgram(program);
  if (output_buffer)
    clReleaseMemObject(output_buffer);
  if (input_buffer)
    clReleaseMemObject(input_buffer);
  if (queue)
    clReleaseCommandQueue(queue);
  if (context)
    clReleaseContext(context);
  sgemm_kernel = 0;
  sgemv_kernel = 0;
  saxpy_kernel = 0;
  sdot_product_kernel = 0;
  sdot_reduce_kernel = 0;
  program = 0;
  output_buffer = input_buffer = 0;
  queue = 0;
  context = 0;
}

static udonblasStatus_t execution_failure(void) {
  clear_objects();
  return UDONBLAS_STATUS_EXECUTION_FAILED;
}

udonblasStatus_t udonblasCreate(udonblasHandle_t *handle) {
  cl_platform_id platform;
  cl_device_id device;
  const char *source = UDONBLAS_DEVICE_SOURCE;
  cl_int status;
  if (!handle)
    return UDONBLAS_STATUS_INVALID_VALUE;
  clear_objects();
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, 0) !=
          CL_SUCCESS)
    return execution_failure();
  context = clCreateContext(0, 1, &device, 0, 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  queue = clCreateCommandQueue(context, device, 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  input_buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(packed), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  output_buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(packed), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  program = clCreateProgramWithSource(context, 1, &source, 0, &status);
  if (status != CL_SUCCESS ||
      clBuildProgram(program, 0, 0, "-cl-std=CL1.2", 0, 0) != CL_SUCCESS)
    return execution_failure();
  sgemm_kernel = clCreateKernel(program, "udonblas_sgemm", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(sgemm_kernel, 0, sizeof(input_buffer), &input_buffer) !=
          CL_SUCCESS ||
      clSetKernelArg(sgemm_kernel, 1, sizeof(output_buffer), &output_buffer) !=
          CL_SUCCESS)
    return execution_failure();
  sgemv_kernel = clCreateKernel(program, "udonblas_sgemv", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(sgemv_kernel, 0, sizeof(input_buffer), &input_buffer) !=
          CL_SUCCESS ||
      clSetKernelArg(sgemv_kernel, 1, sizeof(output_buffer), &output_buffer) !=
          CL_SUCCESS)
    return execution_failure();
  saxpy_kernel = clCreateKernel(program, "udonblas_saxpy", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(saxpy_kernel, 0, sizeof(input_buffer), &input_buffer) !=
          CL_SUCCESS ||
      clSetKernelArg(saxpy_kernel, 1, sizeof(output_buffer), &output_buffer) !=
          CL_SUCCESS)
    return execution_failure();
  sdot_product_kernel =
      clCreateKernel(program, "udonblas_sdot_product", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(sdot_product_kernel, 0, sizeof(input_buffer),
                     &input_buffer) != CL_SUCCESS ||
      clSetKernelArg(sdot_product_kernel, 1, sizeof(output_buffer),
                     &output_buffer) != CL_SUCCESS)
    return execution_failure();
  sdot_reduce_kernel = clCreateKernel(program, "udonblas_sdot_reduce", &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  *handle = UDONBLAS_HANDLE;
  return UDONBLAS_STATUS_SUCCESS;
}

udonblasStatus_t udonblasSdot(udonblasHandle_t handle, int n, const float *x,
                              int incx, const float *y, int incy,
                              float *dot_result) {
  int x_span, y_span, index, pass;
  int64_t x_span64, y_span64;
  size_t global = UDONBLAS_WORDS, local = 1;
  cl_int status;
  if (handle != UDONBLAS_HANDLE || !context)
    return UDONBLAS_STATUS_NOT_INITIALIZED;
  if (!dot_result)
    return UDONBLAS_STATUS_INVALID_VALUE;
  if (n <= 0) {
    *dot_result = 0.0f;
    return UDONBLAS_STATUS_SUCCESS;
  }
  if (!x || !y || incx == 0 || incy == 0)
    return UDONBLAS_STATUS_INVALID_VALUE;
  if (n > UDONBLAS_SDOT_MAX_N)
    return UDONBLAS_STATUS_NOT_SUPPORTED;
  x_span64 = 1 + (int64_t)(n - 1) * (incx < 0 ? -(int64_t)incx : incx);
  y_span64 = 1 + (int64_t)(n - 1) * (incy < 0 ? -(int64_t)incy : incy);
  if (x_span64 > UDONBLAS_MATRIX_WORDS || y_span64 > UDONBLAS_MATRIX_WORDS)
    return UDONBLAS_STATUS_NOT_SUPPORTED;
  x_span = (int)x_span64;
  y_span = (int)y_span64;
  for (index = 0; index < UDONBLAS_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < x_span; ++index)
    packed[UDONBLAS_A_OFFSET + index] = x[index];
  for (index = 0; index < y_span; ++index)
    packed[UDONBLAS_B_OFFSET + index] = y[index];

#define SET_SDOT_ARGUMENT(kernel, slot, value)                                 \
  do {                                                                         \
    status = clSetKernelArg(kernel, slot, sizeof(value), &(value));            \
    if (status != CL_SUCCESS)                                                  \
      return UDONBLAS_STATUS_EXECUTION_FAILED;                                 \
  } while (0)
  SET_SDOT_ARGUMENT(sdot_product_kernel, 2, n);
  SET_SDOT_ARGUMENT(sdot_product_kernel, 3, incx);
  SET_SDOT_ARGUMENT(sdot_product_kernel, 4, incy);
  if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(packed),
                           packed, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, sdot_product_kernel, 1, 0, &global, &local,
                             0, 0, 0) != CL_SUCCESS)
    return UDONBLAS_STATUS_EXECUTION_FAILED;
  for (pass = 1; pass <= 8; ++pass) {
    cl_mem source = pass & 1 ? output_buffer : input_buffer;
    cl_mem destination = pass & 1 ? input_buffer : output_buffer;
    if (clSetKernelArg(sdot_reduce_kernel, 0, sizeof(source), &source) !=
            CL_SUCCESS ||
        clSetKernelArg(sdot_reduce_kernel, 1, sizeof(destination),
                       &destination) != CL_SUCCESS)
      return UDONBLAS_STATUS_EXECUTION_FAILED;
    SET_SDOT_ARGUMENT(sdot_reduce_kernel, 2, n);
    SET_SDOT_ARGUMENT(sdot_reduce_kernel, 3, pass);
    if (clEnqueueNDRangeKernel(queue, sdot_reduce_kernel, 1, 0, &global, &local,
                               0, 0, 0) != CL_SUCCESS)
      return UDONBLAS_STATUS_EXECUTION_FAILED;
  }
#undef SET_SDOT_ARGUMENT
  if (clEnqueueReadBuffer(queue, output_buffer, CL_TRUE,
                          UDONBLAS_RESULT_OFFSET * sizeof(float), sizeof(float),
                          dot_result, 0, 0, 0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONBLAS_STATUS_EXECUTION_FAILED;
  return UDONBLAS_STATUS_SUCCESS;
}

udonblasStatus_t udonblasSaxpy(udonblasHandle_t handle, int n,
                               const float *alpha, const float *x, int incx,
                               float *y, int incy) {
  int x_span, y_span, y_start, index;
  int64_t x_span64, y_span64;
  size_t global = UDONBLAS_WORDS, local = 1;
  cl_int status;
  if (handle != UDONBLAS_HANDLE || !context)
    return UDONBLAS_STATUS_NOT_INITIALIZED;
  if (n < 0 || !alpha || !x || !y || incx == 0 || incy == 0)
    return UDONBLAS_STATUS_INVALID_VALUE;
  if (n > UDONBLAS_SAXPY_MAX_N)
    return UDONBLAS_STATUS_NOT_SUPPORTED;
  if (n == 0)
    return UDONBLAS_STATUS_SUCCESS;
  x_span64 = 1 + (int64_t)(n - 1) * (incx < 0 ? -(int64_t)incx : incx);
  y_span64 = 1 + (int64_t)(n - 1) * (incy < 0 ? -(int64_t)incy : incy);
  if (x_span64 > UDONBLAS_MATRIX_WORDS || y_span64 > UDONBLAS_MATRIX_WORDS)
    return UDONBLAS_STATUS_NOT_SUPPORTED;
  x_span = (int)x_span64;
  y_span = (int)y_span64;
  y_start = incy < 0 ? (1 - n) * incy : 0;
  for (index = 0; index < UDONBLAS_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < x_span; ++index)
    packed[UDONBLAS_A_OFFSET + index] = x[index];
  for (index = 0; index < y_span; ++index)
    packed[UDONBLAS_B_OFFSET + index] = y[index];

#define SET_SAXPY_ARGUMENT(slot, value)                                        \
  do {                                                                         \
    status = clSetKernelArg(saxpy_kernel, slot, sizeof(value), &(value));      \
    if (status != CL_SUCCESS)                                                  \
      return UDONBLAS_STATUS_EXECUTION_FAILED;                                 \
  } while (0)
  SET_SAXPY_ARGUMENT(2, n);
  SET_SAXPY_ARGUMENT(3, incx);
  SET_SAXPY_ARGUMENT(4, incy);
  SET_SAXPY_ARGUMENT(5, *alpha);
#undef SET_SAXPY_ARGUMENT

  if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(packed),
                           packed, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, saxpy_kernel, 1, 0, &global, &local, 0, 0,
                             0) != CL_SUCCESS ||
      clEnqueueReadBuffer(queue, output_buffer, CL_TRUE,
                          UDONBLAS_RESULT_OFFSET * sizeof(float),
                          n * sizeof(float), result, 0, 0, 0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONBLAS_STATUS_EXECUTION_FAILED;
  for (index = 0; index < n; ++index)
    y[y_start + index * incy] = result[index];
  return UDONBLAS_STATUS_SUCCESS;
}

udonblasStatus_t udonblasSgemv(udonblasHandle_t handle,
                               udonblasOperation_t trans, int m, int n,
                               const float *alpha, const float *a, int lda,
                               const float *x, int incx, const float *beta,
                               float *y, int incy) {
  int input_length, output_length, a_words, x_span, y_span, y_start, index;
  int64_t a_words64, x_span64, y_span64;
  size_t global = UDONBLAS_WORDS, local = 1;
  cl_int status;
  if (handle != UDONBLAS_HANDLE || !context)
    return UDONBLAS_STATUS_NOT_INITIALIZED;
  if ((trans != UDONBLAS_OP_N && trans != UDONBLAS_OP_T &&
       trans != UDONBLAS_OP_C) ||
      m < 0 || n < 0 || !alpha || !a || !x || !beta || !y ||
      lda < (m > 1 ? m : 1) || incx == 0 || incy == 0)
    return UDONBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0)
    return UDONBLAS_STATUS_SUCCESS;
  if (m > UDONBLAS_SGEMV_MAX_M || n > UDONBLAS_SGEMV_MAX_N)
    return UDONBLAS_STATUS_NOT_SUPPORTED;
  input_length = trans == UDONBLAS_OP_N ? n : m;
  output_length = trans == UDONBLAS_OP_N ? m : n;
  a_words64 = (int64_t)lda * n;
  x_span64 =
      1 + (int64_t)(input_length - 1) * (incx < 0 ? -(int64_t)incx : incx);
  y_span64 =
      1 + (int64_t)(output_length - 1) * (incy < 0 ? -(int64_t)incy : incy);
  if (a_words64 > UDONBLAS_MATRIX_WORDS || x_span64 > UDONBLAS_MATRIX_WORDS ||
      y_span64 > UDONBLAS_MATRIX_WORDS)
    return UDONBLAS_STATUS_NOT_SUPPORTED;
  a_words = (int)a_words64;
  x_span = (int)x_span64;
  y_span = (int)y_span64;
  y_start = incy < 0 ? (1 - output_length) * incy : 0;
  for (index = 0; index < UDONBLAS_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < a_words; ++index)
    packed[UDONBLAS_A_OFFSET + index] = a[index];
  for (index = 0; index < x_span; ++index)
    packed[UDONBLAS_B_OFFSET + index] = x[index];
  for (index = 0; index < y_span; ++index)
    packed[UDONBLAS_C_OFFSET + index] = y[index];

#define SET_SGEMV_ARGUMENT(slot, value)                                        \
  do {                                                                         \
    status = clSetKernelArg(sgemv_kernel, slot, sizeof(value), &(value));      \
    if (status != CL_SUCCESS)                                                  \
      return UDONBLAS_STATUS_EXECUTION_FAILED;                                 \
  } while (0)
  SET_SGEMV_ARGUMENT(2, trans);
  SET_SGEMV_ARGUMENT(3, m);
  SET_SGEMV_ARGUMENT(4, n);
  SET_SGEMV_ARGUMENT(5, lda);
  SET_SGEMV_ARGUMENT(6, incx);
  SET_SGEMV_ARGUMENT(7, incy);
  SET_SGEMV_ARGUMENT(8, *alpha);
  SET_SGEMV_ARGUMENT(9, *beta);
#undef SET_SGEMV_ARGUMENT

  if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(packed),
                           packed, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, sgemv_kernel, 1, 0, &global, &local, 0, 0,
                             0) != CL_SUCCESS ||
      clEnqueueReadBuffer(
          queue, output_buffer, CL_TRUE, UDONBLAS_RESULT_OFFSET * sizeof(float),
          output_length * sizeof(float), result, 0, 0, 0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONBLAS_STATUS_EXECUTION_FAILED;
  for (index = 0; index < output_length; ++index)
    y[y_start + index * incy] = result[index];
  return UDONBLAS_STATUS_SUCCESS;
}

udonblasStatus_t udonblasDestroy(udonblasHandle_t handle) {
  if (handle != UDONBLAS_HANDLE || !context)
    return UDONBLAS_STATUS_NOT_INITIALIZED;
  clear_objects();
  return UDONBLAS_STATUS_SUCCESS;
}

static int matrix_words(udonblasOperation_t operation, int rows, int columns,
                        int leading_dimension) {
  int stored_columns = operation == UDONBLAS_OP_N ? columns : rows;
  return leading_dimension * stored_columns;
}

udonblasStatus_t udonblasSgemm(udonblasHandle_t handle,
                               udonblasOperation_t transa,
                               udonblasOperation_t transb, int m, int n, int k,
                               const float *alpha, const float *a, int lda,
                               const float *b, int ldb, const float *beta,
                               float *c, int ldc) {
  int a_rows = transa == UDONBLAS_OP_N ? m : k;
  int b_rows = transb == UDONBLAS_OP_N ? k : n;
  int a_words, b_words, c_words, index, row, column;
  size_t global[2] = {32, 32}, local[2] = {1, 1};
  cl_int status;
  if (handle != UDONBLAS_HANDLE || !context)
    return UDONBLAS_STATUS_NOT_INITIALIZED;
  if ((transa != UDONBLAS_OP_N && transa != UDONBLAS_OP_T &&
       transa != UDONBLAS_OP_C) ||
      (transb != UDONBLAS_OP_N && transb != UDONBLAS_OP_T &&
       transb != UDONBLAS_OP_C) ||
      m < 0 || n < 0 || k < 0 || !alpha || !beta || !a || !b || !c ||
      lda < (a_rows > 1 ? a_rows : 1) || ldb < (b_rows > 1 ? b_rows : 1) ||
      ldc < (m > 1 ? m : 1))
    return UDONBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0)
    return UDONBLAS_STATUS_SUCCESS;
  if (m > UDONBLAS_SGEMM_MAX_M || n > UDONBLAS_SGEMM_MAX_N ||
      k > UDONBLAS_SGEMM_MAX_K)
    return UDONBLAS_STATUS_NOT_SUPPORTED;
  a_words = matrix_words(transa, m, k, lda);
  b_words = matrix_words(transb, k, n, ldb);
  c_words = ldc * n;
  if (a_words > UDONBLAS_MATRIX_WORDS || b_words > UDONBLAS_MATRIX_WORDS ||
      c_words > UDONBLAS_MATRIX_WORDS)
    return UDONBLAS_STATUS_NOT_SUPPORTED;
  for (index = 0; index < UDONBLAS_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < a_words; ++index)
    packed[UDONBLAS_A_OFFSET + index] = a[index];
  for (index = 0; index < b_words; ++index)
    packed[UDONBLAS_B_OFFSET + index] = b[index];
  for (index = 0; index < c_words; ++index)
    packed[UDONBLAS_C_OFFSET + index] = c[index];

#define SET_ARGUMENT(slot, value)                                              \
  do {                                                                         \
    status = clSetKernelArg(sgemm_kernel, slot, sizeof(value), &(value));      \
    if (status != CL_SUCCESS)                                                  \
      return UDONBLAS_STATUS_EXECUTION_FAILED;                                 \
  } while (0)
  SET_ARGUMENT(2, m);
  SET_ARGUMENT(3, n);
  SET_ARGUMENT(4, k);
  SET_ARGUMENT(5, transa);
  SET_ARGUMENT(6, transb);
  SET_ARGUMENT(7, lda);
  SET_ARGUMENT(8, ldb);
  SET_ARGUMENT(9, ldc);
  SET_ARGUMENT(10, *alpha);
  SET_ARGUMENT(11, *beta);
#undef SET_ARGUMENT

  if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(packed),
                           packed, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, sgemm_kernel, 2, 0, global, local, 0, 0,
                             0) != CL_SUCCESS ||
      clEnqueueReadBuffer(
          queue, output_buffer, CL_TRUE, UDONBLAS_RESULT_OFFSET * sizeof(float),
          m * n * sizeof(float), result, 0, 0, 0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONBLAS_STATUS_EXECUTION_FAILED;
  for (column = 0; column < n; ++column)
    for (row = 0; row < m; ++row)
      c[row + column * ldc] = result[row + column * m];
  return UDONBLAS_STATUS_SUCCESS;
}
