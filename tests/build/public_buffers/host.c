#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/export.h>

UDONCL_PROGRAM("public_buffers");

#define SAXPY_WORDS 32

/* The kernel is compiled ahead of time from saxpy.cl. */
static const char kernel_source[] = "saxpy.cl";

/* U# writes x and y and reads result: their buffers are created once, with
 * CL_MEM_READ_ONLY and CL_MEM_WRITE_ONLY, and live until shutdown. */
UDONCL_PUBLIC(float) cl_mem x, y;
UDONCL_PUBLIC(float) cl_mem result;

static cl_context context;
static cl_command_queue queue;

static cl_mem create(cl_mem_flags flags, cl_int *status) {
  return clCreateBuffer(context, flags, SAXPY_WORDS * sizeof(float), 0, status);
}

UDONCL_EXPORT int initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
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
  x = create(CL_MEM_READ_ONLY, &status);
  if (status != CL_SUCCESS)
    return status;
  y = create(CL_MEM_READ_ONLY, &status);
  if (status != CL_SUCCESS)
    return status;
  result = create(CL_MEM_WRITE_ONLY, &status);
  return status;
}

/* result = alpha * x + y over the first n words. */
UDONCL_EXPORT int run(int n, float alpha) {
  cl_program program = 0;
  cl_kernel kernel = 0;
  const char *source = kernel_source;
  size_t global = SAXPY_WORDS;
  size_t local = 1;
  cl_int status;
  program = clCreateProgramWithSource(context, 1, &source, 0, &status);
  if (status != CL_SUCCESS)
    goto cleanup;
  status = clBuildProgram(program, 0, 0, 0, 0, 0);
  if (status != CL_SUCCESS)
    goto cleanup;
  kernel = clCreateKernel(program, "saxpy", &status);
  if (status != CL_SUCCESS)
    goto cleanup;
  if ((status = clSetKernelArg(kernel, 0, sizeof(x), &x)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 1, sizeof(y), &y)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 2, sizeof(result), &result)) !=
          CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 3, sizeof(n), &n)) != CL_SUCCESS ||
      (status = clSetKernelArg(kernel, 4, sizeof(alpha), &alpha)) != CL_SUCCESS)
    goto cleanup;
  status =
      clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global, &local, 0, 0, 0);
  if (status != CL_SUCCESS)
    goto cleanup;
  status = clFinish(queue);
cleanup:
  if (kernel)
    clReleaseKernel(kernel);
  if (program)
    clReleaseProgram(program);
  return status;
}

UDONCL_EXPORT int shutdown(void) {
  if (result)
    clReleaseMemObject(result);
  if (y)
    clReleaseMemObject(y);
  if (x)
    clReleaseMemObject(x);
  if (queue)
    clReleaseCommandQueue(queue);
  if (context)
    clReleaseContext(context);
  result = y = x = 0;
  queue = 0;
  context = 0;
  return CL_SUCCESS;
}
