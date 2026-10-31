#include "device_source.h"
#include "weights.h"
#include <CL/cl.h>

#define MNIST_WORDS 10800
#define MNIST_LOGITS 10
#define MNIST_CONV_WEIGHT_OFFSET 784
#define MNIST_CONV_BIAS_OFFSET 1184
#define MNIST_DENSE_WEIGHT_OFFSET 1200
#define MNIST_DENSE_BIAS_OFFSET 2640

static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel conv_kernel, pool_kernel, dense_kernel;
static cl_mem state_a, state_b;
static float packed[MNIST_WORDS];
static cl_int status;

#define CHECK(call)                                                            \
  do {                                                                         \
    status = (call);                                                           \
    if (status != CL_SUCCESS)                                                  \
      return status;                                                           \
  } while (0)

int mnist_gpu_close(void) {
  if (dense_kernel)
    clReleaseKernel(dense_kernel);
  if (pool_kernel)
    clReleaseKernel(pool_kernel);
  if (conv_kernel)
    clReleaseKernel(conv_kernel);
  if (program)
    clReleaseProgram(program);
  if (state_b)
    clReleaseMemObject(state_b);
  if (state_a)
    clReleaseMemObject(state_a);
  if (queue)
    clReleaseCommandQueue(queue);
  if (context)
    clReleaseContext(context);
  dense_kernel = pool_kernel = conv_kernel = 0;
  program = 0;
  state_b = state_a = 0;
  queue = 0;
  context = 0;
  return CL_SUCCESS;
}

int mnist_gpu_init(void) {
  cl_platform_id platform;
  cl_device_id device;
  const char *source = UDONCL_DEVICE_SOURCE;
  mnist_gpu_close();
  CHECK(clGetPlatformIDs(1, &platform, 0));
  CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, 0));
  context = clCreateContext(0, 1, &device, 0, 0, &status);
  CHECK(status);
  queue = clCreateCommandQueue(context, device, 0, &status);
  CHECK(status);
  state_a = clCreateBuffer(context, CL_MEM_READ_WRITE,
                           MNIST_WORDS * sizeof(float), 0, &status);
  CHECK(status);
  state_b = clCreateBuffer(context, CL_MEM_READ_WRITE,
                           MNIST_WORDS * sizeof(float), 0, &status);
  CHECK(status);
  program = clCreateProgramWithSource(context, 1, &source, 0, &status);
  CHECK(status);
  CHECK(clBuildProgram(program, 0, 0, "-cl-std=CL1.2", 0, 0));
  conv_kernel = clCreateKernel(program, "mnist_conv", &status);
  CHECK(status);
  pool_kernel = clCreateKernel(program, "mnist_pool", &status);
  CHECK(status);
  dense_kernel = clCreateKernel(program, "mnist_dense", &status);
  CHECK(status);
  CHECK(clSetKernelArg(conv_kernel, 0, sizeof(state_a), &state_a));
  CHECK(clSetKernelArg(conv_kernel, 1, sizeof(state_b), &state_b));
  CHECK(clSetKernelArg(pool_kernel, 0, sizeof(state_b), &state_b));
  CHECK(clSetKernelArg(pool_kernel, 1, sizeof(state_a), &state_a));
  CHECK(clSetKernelArg(dense_kernel, 0, sizeof(state_a), &state_a));
  CHECK(clSetKernelArg(dense_kernel, 1, sizeof(state_b), &state_b));
  return CL_SUCCESS;
}

int mnist_gpu_predict(const float *image, float *logits) {
  size_t global = MNIST_WORDS, local = 1;
  unsigned index;
  for (index = 0; index < MNIST_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < 784; ++index)
    packed[index] = image[index];
  for (index = 0; index < 400; ++index)
    packed[MNIST_CONV_WEIGHT_OFFSET + index] = kMnistConvWeight[index];
  for (index = 0; index < 16; ++index)
    packed[MNIST_CONV_BIAS_OFFSET + index] = kMnistConvBias[index];
  for (index = 0; index < 1440; ++index)
    packed[MNIST_DENSE_WEIGHT_OFFSET + index] = kMnistDenseWeight[index];
  for (index = 0; index < MNIST_LOGITS; ++index)
    packed[MNIST_DENSE_BIAS_OFFSET + index] = kMnistDenseBias[index];
  CHECK(clEnqueueWriteBuffer(queue, state_a, CL_TRUE, 0, sizeof(packed), packed,
                             0, 0, 0));
  CHECK(clEnqueueNDRangeKernel(queue, conv_kernel, 1, 0, &global, &local, 0, 0,
                               0));
  CHECK(clEnqueueNDRangeKernel(queue, pool_kernel, 1, 0, &global, &local, 0, 0,
                               0));
  CHECK(clEnqueueNDRangeKernel(queue, dense_kernel, 1, 0, &global, &local, 0, 0,
                               0));
  CHECK(clEnqueueReadBuffer(queue, state_b, CL_TRUE, 0,
                            MNIST_LOGITS * sizeof(float), logits, 0, 0, 0));
  return clFinish(queue);
}
