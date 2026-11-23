#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/export.h>

#define USER_WORDS 32

UDONCL_PROGRAM("opencl_user_kernel");

static const char kernel_source[] = "user kernels of the Device library";
static float x_values[USER_WORDS];
static float y_values[USER_WORDS];
static float result[USER_WORDS];
/* Private Device buffers: they live in Host globals, as the static Device
 * resource analysis requires of a handle whose address is passed to the
 * runtime, and each resolves at compile time to its own Unity Texture. */
static cl_mem x;
static cl_mem y;
static cl_mem sum;
static cl_mem scaled;

/* result = factor * (alpha * x + y) over the first n words and 0 beyond,
 * with x[i] = i and y[i] = 1: two user kernels in sequence. */
UDONCL_EXPORT int run_user_kernels(int n, float alpha, float factor) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel saxpy;
  cl_kernel scale;
  const char *source = kernel_source;
  size_t global = USER_WORDS;
  size_t local = 1;
  int index;

  for (index = 0; index < USER_WORDS; ++index) {
    x_values[index] = (float)index;
    y_values[index] = 1.0f;
  }
  (void)clGetPlatformIDs(1, &platform, 0);
  (void)clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
  context = clCreateContext(0, 1, &device, 0, 0, 0);
  queue = clCreateCommandQueue(context, device, 0, 0);
  x = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(x_values), 0, 0);
  y = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(y_values), 0, 0);
  sum = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(result), 0, 0);
  scaled = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(result), 0, 0);
  (void)clEnqueueWriteBuffer(queue, x, CL_FALSE, 0, sizeof(x_values), x_values,
                             0, 0, 0);
  (void)clEnqueueWriteBuffer(queue, y, CL_FALSE, 0, sizeof(y_values), y_values,
                             0, 0, 0);
  program = clCreateProgramWithSource(context, 1, &source, 0, 0);
  (void)clBuildProgram(program, 0, 0, 0, 0, 0);
  saxpy = clCreateKernel(program, "user_saxpy", 0);
  (void)clSetKernelArg(saxpy, 0, sizeof(x), &x);
  (void)clSetKernelArg(saxpy, 1, sizeof(y), &y);
  (void)clSetKernelArg(saxpy, 2, sizeof(sum), &sum);
  (void)clSetKernelArg(saxpy, 3, sizeof(n), &n);
  (void)clSetKernelArg(saxpy, 4, sizeof(alpha), &alpha);
  (void)clEnqueueNDRangeKernel(queue, saxpy, 1, 0, &global, &local, 0, 0, 0);
  scale = clCreateKernel(program, "user_scale", 0);
  (void)clSetKernelArg(scale, 0, sizeof(sum), &sum);
  (void)clSetKernelArg(scale, 1, sizeof(scaled), &scaled);
  (void)clSetKernelArg(scale, 2, sizeof(factor), &factor);
  (void)clEnqueueNDRangeKernel(queue, scale, 1, 0, &global, &local, 0, 0, 0);
  (void)clEnqueueReadBuffer(queue, scaled, CL_TRUE, 0, sizeof(result), result,
                            0, 0, 0);
  (void)clReleaseKernel(scale);
  (void)clReleaseKernel(saxpy);
  (void)clReleaseProgram(program);
  (void)clReleaseMemObject(scaled);
  (void)clReleaseMemObject(sum);
  (void)clReleaseMemObject(y);
  (void)clReleaseMemObject(x);
  (void)clReleaseCommandQueue(queue);
  (void)clReleaseContext(context);
  return 0;
}

UDONCL_EXPORT float user_kernel_result(int index) { return result[index]; }
