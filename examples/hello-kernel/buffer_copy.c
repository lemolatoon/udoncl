#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

_Static_assert(sizeof(cl_mem) == 4,
               "UdonCL Host input must use the 32-bit OpenCL handle ABI");
_Static_assert(CL_MEM_READ_WRITE == 1, "unexpected OpenCL 1.2 headers");

void copy_full(void) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_mem source;
  cl_mem intermediate;
  cl_mem destination;

  (void)clGetPlatformIDs(1, &platform, 0);
  (void)clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
  context = clCreateContext(0, 1, &device, 0, 0, 0);
  queue = clCreateCommandQueue(context, device, 0, 0);
  source = clCreateBuffer(context, CL_MEM_READ_WRITE, 32772, 0, 0);
  intermediate = clCreateBuffer(context, CL_MEM_READ_WRITE, 32772, 0, 0);
  destination = clCreateBuffer(context, CL_MEM_READ_WRITE, 32772, 0, 0);
  (void)clEnqueueCopyBuffer(queue, source, intermediate, 0, 0, 32772, 0, 0, 0);
  (void)clEnqueueCopyBuffer(queue, intermediate, destination, 0, 0, 32772, 0, 0,
                            0);
  (void)clReleaseMemObject(destination);
  (void)clReleaseMemObject(intermediate);
  (void)clReleaseMemObject(source);
  (void)clReleaseCommandQueue(queue);
  (void)clReleaseContext(context);
}
