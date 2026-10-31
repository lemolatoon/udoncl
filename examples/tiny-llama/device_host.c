#include "device_source.h"
#include <CL/cl.h>

#define TINY_LLAMA_WORDS 2048
#define TINY_LLAMA_LOGITS 16
#define TINY_LLAMA_LOGITS_OFFSET 352

static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernels[7];
static cl_mem state_a, state_b;
static float packed[TINY_LLAMA_WORDS];
static cl_int status;

#define CHECK(call)                                                            \
  do {                                                                         \
    status = (call);                                                           \
    if (status != CL_SUCCESS)                                                  \
      return status;                                                           \
  } while (0)

int tiny_llama_gpu_close(void) {
  int index;
  for (index = 6; index >= 0; --index)
    if (kernels[index])
      clReleaseKernel(kernels[index]);
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
  for (index = 0; index < 7; ++index)
    kernels[index] = 0;
  program = 0;
  state_b = state_a = 0;
  queue = 0;
  context = 0;
  return CL_SUCCESS;
}

int tiny_llama_gpu_init(void) {
  static const char *names[7] = {
      "tiny_llama_embed",   "tiny_llama_qkv",    "tiny_llama_attention",
      "tiny_llama_project", "tiny_llama_ffn_up", "tiny_llama_ffn_down",
      "tiny_llama_logits",
  };
  cl_platform_id platform;
  cl_device_id device;
  const char *source = UDONCL_DEVICE_SOURCE;
  int index;
  tiny_llama_gpu_close();
  CHECK(clGetPlatformIDs(1, &platform, 0));
  CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, 0));
  context = clCreateContext(0, 1, &device, 0, 0, &status);
  CHECK(status);
  queue = clCreateCommandQueue(context, device, 0, &status);
  CHECK(status);
  state_a = clCreateBuffer(context, CL_MEM_READ_WRITE,
                           TINY_LLAMA_WORDS * sizeof(float), 0, &status);
  CHECK(status);
  state_b = clCreateBuffer(context, CL_MEM_READ_WRITE,
                           TINY_LLAMA_WORDS * sizeof(float), 0, &status);
  CHECK(status);
  program = clCreateProgramWithSource(context, 1, &source, 0, &status);
  CHECK(status);
  CHECK(clBuildProgram(program, 0, 0, "-cl-std=CL1.2", 0, 0));
  for (index = 0; index < 7; ++index) {
    kernels[index] = clCreateKernel(program, names[index], &status);
    CHECK(status);
    CHECK(clSetKernelArg(kernels[index], 0, sizeof(state_a), &state_a));
    CHECK(clSetKernelArg(kernels[index], 1, sizeof(state_b), &state_b));
    if (index != 6) {
      cl_mem swap = state_a;
      state_a = state_b;
      state_b = swap;
    }
  }
  return CL_SUCCESS;
}

int tiny_llama_gpu_predict(const float *input, float *logits) {
  size_t global = TINY_LLAMA_WORDS, local = 1;
  int index;
  for (index = 0; index < TINY_LLAMA_WORDS; ++index)
    packed[index] = input[index];
  CHECK(clEnqueueWriteBuffer(queue, state_a, CL_TRUE, 0, sizeof(packed), packed,
                             0, 0, 0));
  for (index = 0; index < 7; ++index)
    CHECK(clEnqueueNDRangeKernel(queue, kernels[index], 1, 0, &global, &local,
                                 0, 0, 0));
  CHECK(clEnqueueReadBuffer(
      queue, state_b, CL_TRUE, TINY_LLAMA_LOGITS_OFFSET * sizeof(float),
      TINY_LLAMA_LOGITS * sizeof(float), logits, 0, 0, 0));
  return clFinish(queue);
}
