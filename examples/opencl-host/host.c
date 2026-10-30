/* Ordinary OpenCL 1.2 Host code. device_source.h is generated from device.cl.
 */
#define CL_TARGET_OPENCL_VERSION 120
#include "device_source.h"
#include <CL/cl.h>

cl_int run_kernel(float input, float *result) {
  cl_int status;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem output = NULL;
  cl_event done = NULL;
  const char *source = UDONCL_DEVICE_SOURCE;
  size_t global = 1, local = 1;
  float value = 0;
#define CHECK(call)                                                            \
  do {                                                                         \
    status = (call);                                                           \
    if (status != CL_SUCCESS)                                                  \
      goto cleanup;                                                            \
  } while (0)
  if (!result)
    return CL_INVALID_VALUE;
  CHECK(clGetPlatformIDs(1, &platform, NULL));
  CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  if (status)
    goto cleanup;
  queue = clCreateCommandQueue(context, device, 0, &status);
  if (status)
    goto cleanup;
  program = clCreateProgramWithSource(context, 1, &source, NULL, &status);
  if (status)
    goto cleanup;
  CHECK(clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL));
  kernel = clCreateKernel(program, UDONCL_KERNEL_ENTRY, &status);
  if (status)
    goto cleanup;
  output =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(value), NULL, &status);
  if (status)
    goto cleanup;
  CHECK(clSetKernelArg(kernel, 0, sizeof(output), &output));
  CHECK(clSetKernelArg(kernel, 1, sizeof(input), &input));
  input = -999.0f; /* clSetKernelArg must already own a copy. */
  CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL,
                               &done));
  CHECK(clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(value), &value, 1,
                            &done, NULL));
  CHECK(clFinish(queue));
  *result = value + 1.0f; /* Use readback in generated Host code. */
cleanup:
  if (done)
    clReleaseEvent(done);
  if (output)
    clReleaseMemObject(output);
  if (kernel)
    clReleaseKernel(kernel);
  if (program)
    clReleaseProgram(program);
  if (queue)
    clReleaseCommandQueue(queue);
  if (context)
    clReleaseContext(context);
  return status;
}
