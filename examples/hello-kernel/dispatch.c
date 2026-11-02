#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

static const char kernel_source[] =
    "__kernel void square_plus_one(__global float *output, float x) {"
    "output[0] = x * x + 1.0f; }";

static const char kernel_source_i32[] =
    "__kernel void square_plus_one_i32(__global int *output, int x) {"
    "output[0] = x * x + 1; }";

void dispatch_square(float value) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_mem output;
  cl_program program;
  cl_kernel kernel;
  const char *source = kernel_source;
  size_t global = 1;
  size_t local = 1;

  (void)clGetPlatformIDs(1, &platform, 0);
  (void)clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
  context = clCreateContext(0, 1, &device, 0, 0, 0);
  queue = clCreateCommandQueue(context, device, 0, 0);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), 0, 0);
  program = clCreateProgramWithSource(context, 1, &source, 0, 0);
  (void)clBuildProgram(program, 0, 0, 0, 0, 0);
  kernel = clCreateKernel(program, "square_plus_one", 0);
  (void)clSetKernelArg(kernel, 0, sizeof(output), &output);
  (void)clSetKernelArg(kernel, 1, sizeof(value), &value);
  (void)clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global, &local, 0, 0, 0);
  (void)clReleaseKernel(kernel);
  (void)clReleaseProgram(program);
  (void)clReleaseMemObject(output);
  (void)clReleaseCommandQueue(queue);
  (void)clReleaseContext(context);
}

void dispatch_square_i32(int value) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_mem output;
  cl_program program;
  cl_kernel kernel;
  const char *source = kernel_source_i32;
  size_t global = 1;
  size_t local = 1;

  (void)clGetPlatformIDs(1, &platform, 0);
  (void)clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
  context = clCreateContext(0, 1, &device, 0, 0, 0);
  queue = clCreateCommandQueue(context, device, 0, 0);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), 0, 0);
  program = clCreateProgramWithSource(context, 1, &source, 0, 0);
  (void)clBuildProgram(program, 0, 0, 0, 0, 0);
  kernel = clCreateKernel(program, "square_plus_one_i32", 0);
  (void)clSetKernelArg(kernel, 0, sizeof(output), &output);
  (void)clSetKernelArg(kernel, 1, sizeof(value), &value);
  (void)clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global, &local, 0, 0, 0);
  (void)clReleaseKernel(kernel);
  (void)clReleaseProgram(program);
  (void)clReleaseMemObject(output);
  (void)clReleaseCommandQueue(queue);
  (void)clReleaseContext(context);
}
