#include <CL/cl.h>
#include <stddef.h>

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition))                                                          \
      return __LINE__;                                                         \
  } while (0)

static cl_uint allocated_logical_id;
static size_t allocated_bytes;
static cl_uint uploaded_token;
static cl_uint released_token;
static cl_uint dispatched_input_token;
static cl_uint dispatched_output_token;
static cl_int dispatched_m;
static cl_float dispatched_alpha;

cl_uint __vrc_opencl_buffer_alloc(cl_uint logical_id, size_t bytes) {
  allocated_logical_id = logical_id;
  allocated_bytes = bytes;
  return logical_id + 100;
}

cl_int __vrc_opencl_buffer_release(cl_uint physical_token) {
  released_token = physical_token;
  return CL_SUCCESS;
}

cl_int __vrc_opencl_buffer_upload(cl_uint physical_token, size_t offset,
                                  size_t bytes, const void *source) {
  (void)offset;
  (void)bytes;
  (void)source;
  uploaded_token = physical_token;
  return 7;
}

cl_int __vrc_opencl_buffer_readback(cl_uint physical_token, size_t offset,
                                    size_t bytes, void *destination) {
  (void)physical_token;
  (void)offset;
  (void)bytes;
  (void)destination;
  return 8;
}

cl_int __vrc_opencl_dispatch_udonblas_sgemm(cl_uint input_token,
                                            cl_uint output_token, cl_int m,
                                            cl_int n, cl_int k, cl_int transa,
                                            cl_int transb, cl_int lda,
                                            cl_int ldb, cl_int ldc,
                                            cl_float alpha, cl_float beta) {
  (void)n;
  (void)k;
  (void)transa;
  (void)transb;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)beta;
  dispatched_input_token = input_token;
  dispatched_output_token = output_token;
  dispatched_m = m;
  dispatched_alpha = alpha;
  return 10;
}

cl_int __vrc_opencl_await(cl_int event) { return event; }

int main(void) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem first, second;
  cl_int error;
  cl_int integer_arguments[8] = {1, 10, 16, 0, 0, 1, 16, 1};
  cl_float float_arguments[2] = {1.5f, 0.0f};
  const char *source_text = "known AOT program";
  size_t global[2] = {32, 32}, local[2] = {1, 1};
  cl_uint index;
  int source = 42;

  EXPECT(clGetPlatformIDs(1, &platform, 0) == CL_SUCCESS);
  EXPECT(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) ==
         CL_SUCCESS);
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  EXPECT(context && error == CL_SUCCESS);
  queue = clCreateCommandQueue(context, device, 0, &error);
  EXPECT(queue && error == CL_SUCCESS);

  first = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(source), 0, &error);
  EXPECT(first && error == CL_SUCCESS);
  EXPECT(allocated_logical_id == 1 && allocated_bytes == sizeof(source));
  EXPECT(clEnqueueWriteBuffer(queue, first, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, 0) == CL_SUCCESS);
  EXPECT(uploaded_token == 101);
  EXPECT(clReleaseMemObject(first) == CL_SUCCESS);
  EXPECT(released_token == 101);

  /* Reusing the logical table slot must request a new physical token rather
   * than retaining the released backing-resource identity. */
  second =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(source), 0, &error);
  EXPECT(second && error == CL_SUCCESS);
  EXPECT(allocated_logical_id == 1);

  first = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(source), 0, &error);
  EXPECT(first && error == CL_SUCCESS);
  program = clCreateProgramWithSource(context, 1, &source_text, 0, &error);
  EXPECT(program && error == CL_SUCCESS);
  EXPECT(clBuildProgram(program, 0, 0, 0, 0, 0) == CL_SUCCESS);
  kernel = clCreateKernel(program, "udonblas_sgemm", &error);
  EXPECT(kernel && error == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 0, sizeof(first), &first) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 1, sizeof(second), &second) == CL_SUCCESS);
  for (index = 0; index < 8; ++index)
    EXPECT(clSetKernelArg(kernel, index + 2, sizeof(integer_arguments[index]),
                          &integer_arguments[index]) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 10, sizeof(float_arguments[0]),
                        &float_arguments[0]) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 11, sizeof(float_arguments[1]),
                        &float_arguments[1]) == CL_SUCCESS);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 2, 0, global, local, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(dispatched_input_token == 102 && dispatched_output_token == 101);
  EXPECT(dispatched_m == 1 && dispatched_alpha == 1.5f);

  /* Replacing copied arguments for later K tiles must not consume an append-
   * only byte arena or retain the previous scalar value. */
  integer_arguments[0] = 2;
  float_arguments[0] = 2.5f;
  EXPECT(clSetKernelArg(kernel, 2, sizeof(integer_arguments[0]),
                        &integer_arguments[0]) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 10, sizeof(float_arguments[0]),
                        &float_arguments[0]) == CL_SUCCESS);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 2, 0, global, local, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(dispatched_m == 2 && dispatched_alpha == 2.5f);

  EXPECT(clReleaseKernel(kernel) == CL_SUCCESS);
  EXPECT(clReleaseProgram(program) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(first) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(second) == CL_SUCCESS);
  EXPECT(released_token == 101);
  return 0;
}
