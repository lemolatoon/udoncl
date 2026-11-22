#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/export.h>

static const char kernel_source[] = "user kernels of the Device library";
static float result[32];
static cl_mem input;
static cl_mem output;

/* Launches user_scale over a run-time number of work items: its NDRange
 * cannot be derived, so the build must ask for a device_kernels entry. */
UDONCL_EXPORT int run_dynamic_scale(int words, float factor) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel scale;
  const char *source = kernel_source;
  size_t global = (size_t)words;
  size_t local = 1;

  (void)clGetPlatformIDs(1, &platform, 0);
  (void)clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
  context = clCreateContext(0, 1, &device, 0, 0, 0);
  queue = clCreateCommandQueue(context, device, 0, 0);
  input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(result), 0, 0);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(result), 0, 0);
  program = clCreateProgramWithSource(context, 1, &source, 0, 0);
  (void)clBuildProgram(program, 0, 0, 0, 0, 0);
  scale = clCreateKernel(program, "user_scale", 0);
  (void)clSetKernelArg(scale, 0, sizeof(input), &input);
  (void)clSetKernelArg(scale, 1, sizeof(output), &output);
  (void)clSetKernelArg(scale, 2, sizeof(factor), &factor);
  (void)clEnqueueNDRangeKernel(queue, scale, 1, 0, &global, &local, 0, 0, 0);
  (void)clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(result), result,
                            0, 0, 0);
  (void)clReleaseKernel(scale);
  (void)clReleaseProgram(program);
  (void)clReleaseMemObject(output);
  (void)clReleaseMemObject(input);
  (void)clReleaseCommandQueue(queue);
  (void)clReleaseContext(context);
  return 0;
}
