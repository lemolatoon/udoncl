#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/export.h>

#define SAXPY_WORDS 32

/* The kernel is compiled ahead of time from saxpy.cl, the Device library of
 * this example, so the program source is only a placeholder. */
static const char kernel_source[] = "saxpy.cl";

/* Device buffers whose address is passed to the OpenCL runtime live in Host
 * globals. U# writes x and y before RunSaxpy and reads result after it. */
static cl_mem x;
static cl_mem y;
static cl_mem result;

/* result = alpha * x + y over the first n words. Returns CL_SUCCESS or the
 * first OpenCL error. */
UDONCL_EXPORT int run_saxpy(int n, float alpha) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context context = 0;
  cl_command_queue queue = 0;
  cl_program program = 0;
  cl_kernel kernel = 0;
  const char *source = kernel_source;
  size_t global = SAXPY_WORDS;
  size_t local = 1;
  cl_int status;

  x = 0;
  y = 0;
  result = 0;
  status = clGetPlatformIDs(1, &platform, 0);
  if (status != CL_SUCCESS)
    goto cleanup;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
  if (status != CL_SUCCESS)
    goto cleanup;
  context = clCreateContext(0, 1, &device, 0, 0, &status);
  if (status != CL_SUCCESS)
    goto cleanup;
  queue = clCreateCommandQueue(context, device, 0, &status);
  if (status != CL_SUCCESS)
    goto cleanup;
  x = clCreateBuffer(context, CL_MEM_READ_ONLY, SAXPY_WORDS * sizeof(float), 0,
                     &status);
  if (status != CL_SUCCESS)
    goto cleanup;
  y = clCreateBuffer(context, CL_MEM_READ_ONLY, SAXPY_WORDS * sizeof(float), 0,
                     &status);
  if (status != CL_SUCCESS)
    goto cleanup;
  result = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                          SAXPY_WORDS * sizeof(float), 0, &status);
  if (status != CL_SUCCESS)
    goto cleanup;
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
  return status;
}
