#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

static const char kernel_source[] =
    "__kernel void saxpy_demo(__global const float *input, "
    "__global float *output, int n, float alpha) { }";

void run_saxpy(int n, float alpha) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_mem input;
  cl_mem output;
  cl_program program;
  cl_kernel kernel;
  const char *source = kernel_source;
  size_t global = 32;
  size_t local = 1;

  (void)clGetPlatformIDs(1, &platform, 0);
  (void)clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
  context = clCreateContext(0, 1, &device, 0, 0, 0);
  queue = clCreateCommandQueue(context, device, 0, 0);
  input = clCreateBuffer(context, CL_MEM_READ_ONLY, 32 * sizeof(float), 0, 0);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 32 * sizeof(float), 0, 0);
  program = clCreateProgramWithSource(context, 1, &source, 0, 0);
  (void)clBuildProgram(program, 0, 0, 0, 0, 0);
  kernel = clCreateKernel(program, "saxpy_demo", 0);
  (void)clSetKernelArg(kernel, 0, sizeof(input), &input);
  (void)clSetKernelArg(kernel, 1, sizeof(output), &output);
  (void)clSetKernelArg(kernel, 2, sizeof(n), &n);
  (void)clSetKernelArg(kernel, 3, sizeof(alpha), &alpha);
  (void)clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global, &local, 0, 0, 0);
  (void)clReleaseKernel(kernel);
  (void)clReleaseProgram(program);
  (void)clReleaseMemObject(input);
  (void)clReleaseMemObject(output);
  (void)clReleaseCommandQueue(queue);
  (void)clReleaseContext(context);
}
