/* Ordinary OpenCL 1.2 Host C. */
#include "device_source.h"
#include <CL/cl.h>
#define MD_WORDS 256
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel drift, kick;
static cl_mem state, predicted;
static float frame;
static cl_int status;
#define CHECK(call)                                                            \
  do {                                                                         \
    status = (call);                                                           \
    if (status != CL_SUCCESS)                                                  \
      return status;                                                           \
  } while (0)
int md_close(void) {
  if (kick)
    clReleaseKernel(kick);
  if (drift)
    clReleaseKernel(drift);
  if (program)
    clReleaseProgram(program);
  if (predicted)
    clReleaseMemObject(predicted);
  if (state)
    clReleaseMemObject(state);
  if (queue)
    clReleaseCommandQueue(queue);
  if (context)
    clReleaseContext(context);
  kick = drift = 0;
  program = 0;
  predicted = state = 0;
  queue = 0;
  context = 0;
  return 0;
}
int md_init(const float *initial) {
  cl_platform_id platform;
  cl_device_id device;
  const char *source = UDONCL_DEVICE_SOURCE;
  md_close();
  frame = 0;
  CHECK(clGetPlatformIDs(1, &platform, 0));
  CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, 0));
  context = clCreateContext(0, 1, &device, 0, 0, &status);
  CHECK(status);
  queue = clCreateCommandQueue(context, device, 0, &status);
  CHECK(status);
  state = clCreateBuffer(context, CL_MEM_READ_WRITE, MD_WORDS * sizeof(float),
                         0, &status);
  CHECK(status);
  CHECK(clEnqueueWriteBuffer(queue, state, CL_TRUE, 0, MD_WORDS * sizeof(float),
                             initial, 0, 0, 0));
  predicted = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             MD_WORDS * sizeof(float), 0, &status);
  CHECK(status);
  program = clCreateProgramWithSource(context, 1, &source, 0, &status);
  CHECK(status);
  CHECK(clBuildProgram(program, 0, 0, "-cl-std=CL1.2", 0, 0));
  drift = clCreateKernel(program, "md_drift", &status);
  CHECK(status);
  kick = clCreateKernel(program, "md_kick", &status);
  CHECK(status);
  return CL_SUCCESS;
}
int md_step(unsigned steps, float dt, float bx, float by, float bz, float h0x,
            float h0y, float h0z, float h1x, float h1y, float h1z, float radius,
            float amplitude, float cooling) {
  float scalars[13] = {dt,  bx,  by,  bz,     h0x,       h0y,    h0z,
                       h1x, h1y, h1z, radius, amplitude, cooling};
  size_t global = MD_WORDS, local = 1;
  unsigned it, phase, arg;
  for (it = 0; it < steps; ++it)
    for (phase = 0; phase < 2; ++phase) {
      cl_kernel kernel = phase ? kick : drift;
      cl_mem input = phase ? predicted : state,
             output = phase ? state : predicted;
      CHECK(clSetKernelArg(kernel, 0, sizeof(input), &input));
      CHECK(clSetKernelArg(kernel, 1, sizeof(output), &output));
      for (arg = 0; arg < 13; ++arg)
        CHECK(clSetKernelArg(kernel, arg + 2, sizeof(float), scalars + arg));
      CHECK(clSetKernelArg(kernel, 15, sizeof(frame), &frame));
      CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global, &local, 0, 0,
                                   0));
      if (phase)
        ++frame;
    }
  return clFinish(queue);
}
int md_read(float *output) {
  return clEnqueueReadBuffer(queue, state, CL_TRUE, 0, MD_WORDS * sizeof(float),
                             output, 0, 0, 0);
}
int md_contract_tests(void) {
  float scalar = 0, unread[MD_WORDS], step = 0;
  size_t global = MD_WORDS, local = 1;
  unsigned arg;
  cl_mem small;
  for (arg = 2; arg < 15; ++arg)
    CHECK(clSetKernelArg(drift, arg, sizeof(scalar), &scalar));
  CHECK(clSetKernelArg(drift, 15, sizeof(step), &step));
  CHECK(clSetKernelArg(drift, 0, sizeof(state), &state));
  CHECK(clSetKernelArg(drift, 1, sizeof(state), &state));
  if (clEnqueueNDRangeKernel(queue, drift, 1, 0, &global, &local, 0, 0, 0) !=
      CL_INVALID_OPERATION)
    return CL_INVALID_VALUE;
  CHECK(clSetKernelArg(drift, 0, sizeof(predicted), &predicted));
  if (clEnqueueNDRangeKernel(queue, drift, 1, 0, &global, &local, 0, 0, 0) !=
      CL_INVALID_OPERATION)
    return CL_INVALID_VALUE;
  if (clEnqueueReadBuffer(queue, predicted, CL_TRUE, 0, sizeof(unread), unread,
                          0, 0, 0) != CL_INVALID_OPERATION)
    return CL_INVALID_VALUE;
  CHECK(clSetKernelArg(drift, 0, sizeof(state), &state));
  CHECK(clSetKernelArg(drift, 1, sizeof(predicted), &predicted));
  global = MD_WORDS - 1;
  if (clEnqueueNDRangeKernel(queue, drift, 1, 0, &global, &local, 0, 0, 0) !=
      CL_INVALID_GLOBAL_WORK_SIZE)
    return CL_INVALID_VALUE;
  global = MD_WORDS;
  small = clCreateBuffer(context, CL_MEM_READ_WRITE, 4, 0, &status);
  CHECK(status);
  CHECK(clSetKernelArg(drift, 1, sizeof(small), &small));
  status = clEnqueueNDRangeKernel(queue, drift, 1, 0, &global, &local, 0, 0, 0);
  clReleaseMemObject(small);
  if (status != CL_INVALID_MEM_OBJECT)
    return CL_INVALID_VALUE;
  if (clEnqueueNDRangeKernel(queue, drift, 1, 0, &global, &local, 0, 0, 0) !=
      CL_INVALID_MEM_OBJECT)
    return CL_INVALID_VALUE;
  return CL_SUCCESS;
}
