#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/export.h>

#define SAXPY_WORDS 32

/* Unity imports the library under Assets/udoncl-generated/saxpy_demo and
 * names its U# binding UdonclSaxpyDemoBinding. */
UDONCL_PROGRAM("saxpy_demo");

/* The kernel is compiled ahead of time from saxpy.cl, the Device library of
 * this example, so the program source is only a placeholder. */
static const char kernel_source[] = "saxpy.cl";

/* The public Device buffers of the library, named after the variables: U#
 * writes x and y (CL_MEM_READ_ONLY) with TryWriteX and TryWriteY before Run
 * and reads result (CL_MEM_WRITE_ONLY) with StartReadResult after it. Each
 * holds 32 floats. */
UDONCL_PUBLIC(float) cl_mem x, y, result;

/* The OpenCL objects live from initialize to dispose. */
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel;

/* Creates the context, the queue, the three buffers and the saxpy kernel.
 * Returns CL_SUCCESS or the first OpenCL error. */
UDONCL_EXPORT int initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  const char *source = kernel_source;
  cl_int status;
  status = clGetPlatformIDs(1, &platform, 0);
  if (status != CL_SUCCESS)
    return status;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
  if (status != CL_SUCCESS)
    return status;
  context = clCreateContext(0, 1, &device, 0, 0, &status);
  if (status != CL_SUCCESS)
    return status;
  queue = clCreateCommandQueue(context, device, 0, &status);
  if (status != CL_SUCCESS)
    return status;
  x = clCreateBuffer(context, CL_MEM_READ_ONLY, SAXPY_WORDS * sizeof(float), 0,
                     &status);
  if (status != CL_SUCCESS)
    return status;
  y = clCreateBuffer(context, CL_MEM_READ_ONLY, SAXPY_WORDS * sizeof(float), 0,
                     &status);
  if (status != CL_SUCCESS)
    return status;
  result = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                          SAXPY_WORDS * sizeof(float), 0, &status);
  if (status != CL_SUCCESS)
    return status;
  program = clCreateProgramWithSource(context, 1, &source, 0, &status);
  if (status != CL_SUCCESS)
    return status;
  status = clBuildProgram(program, 0, 0, 0, 0, 0);
  if (status != CL_SUCCESS)
    return status;
  kernel = clCreateKernel(program, "saxpy", &status);
  return status;
}

/* result = alpha * x + y over the first n words; the other words of result
 * are y. Returns CL_SUCCESS or the first OpenCL error. */
UDONCL_EXPORT int run(int n, float alpha) {
  size_t global = SAXPY_WORDS;
  size_t local = 1;
  cl_int status;
  if ((status = clSetKernelArg(kernel, 0, sizeof(x), &x)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 1, sizeof(y), &y)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 2, sizeof(result), &result)) !=
          CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 3, sizeof(n), &n)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 4, sizeof(alpha), &alpha)) != CL_SUCCESS)
    return status;
  status =
      clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global, &local, 0, 0, 0);
  if (status != CL_SUCCESS)
    return status;
  return clFinish(queue);
}

/* Releases everything initialize created; initialize may run again. The
 * variables keep their stale handles until then: the kernel launched by run
 * is the one of initialize. */
UDONCL_EXPORT int dispose(void) {
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(result);
  clReleaseMemObject(y);
  clReleaseMemObject(x);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return CL_SUCCESS;
}
