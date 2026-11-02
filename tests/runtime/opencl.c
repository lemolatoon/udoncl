#include "device_source.h"
#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>
void UDONCL_ORACLE(float *, float);
#define EXPECT(value)                                                          \
  do {                                                                         \
    if (!(value))                                                              \
      return __LINE__;                                                         \
  } while (0)
#define OK(call) EXPECT((call) == CL_SUCCESS)
int run_opencl_runtime_tests(void) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context, other;
  cl_command_queue queue, other_queue;
  cl_program program, bad;
  cl_kernel kernel;
  cl_mem buffer, copy_buffer, foreign, saved;
  cl_event copied_event, done, read, foreign_event;
  cl_int error, status;
  cl_uint count;
  float scalar = 2.5f, expected = 0, actual = 0;
  size_t one = 1, two = 2, offset = 1, required;
  const char *text = UDONCL_DEVICE_SOURCE,
             *unknown = "__kernel void unknown(){}";
  char *copy = (char *)malloc(strlen(text) + 1);
  unsigned char bytes[4] = {1, 2, 3, 4}, copied[4] = {0};
  EXPECT(copy != NULL);
  strcpy(copy, text);
  EXPECT(clGetPlatformIDs(0, NULL, NULL) == CL_INVALID_VALUE);
  OK(clGetPlatformIDs(1, &platform, &count));
  EXPECT(count == 1);
  EXPECT(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL) ==
         CL_DEVICE_NOT_FOUND);
  OK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
  EXPECT(context && !error);
  other = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
  EXPECT(other && !error);
  EXPECT(!clCreateBuffer(NULL, 0, 4, NULL, &error) &&
         error == CL_INVALID_CONTEXT);
  EXPECT(!clCreateBuffer(context, 0, 0, NULL, &error) &&
         error == CL_INVALID_BUFFER_SIZE);
  EXPECT(!clCreateBuffer(context, 0, 4, bytes, &error) &&
         error == CL_INVALID_HOST_PTR);
  EXPECT(!clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, 4, bytes, &error) &&
         error == CL_INVALID_VALUE);
  EXPECT(!clCreateCommandQueue(
             context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error) &&
         error == CL_INVALID_QUEUE_PROPERTIES);
  queue = clCreateCommandQueue(context, device, 0, &error);
  EXPECT(queue && !error);
  other_queue = clCreateCommandQueue(other, device, 0, &error);
  EXPECT(other_queue && !error);
  {
    const char *pieces[2] = {copy, copy + 3};
    size_t lengths[2] = {3, strlen(copy) - 3};
    program = clCreateProgramWithSource(context, 2, pieces, lengths, &error);
    EXPECT(program && !error);
  }
  copy[0] = '?';
  free(copy); /* Program owns a copy of all input strings. */
  EXPECT(!clCreateKernel(program, UDONCL_KERNEL_ENTRY, &error) &&
         error == CL_INVALID_PROGRAM_EXECUTABLE);
  EXPECT(clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL,
                        NULL) == CL_INVALID_BUILD_OPTIONS);
  OK(clBuildProgram(program, 1, &device, NULL, NULL, NULL));
  EXPECT(!clCreateKernel(program, "not_the_kernel", &error) &&
         error == CL_INVALID_KERNEL_NAME);
  bad = clCreateProgramWithSource(context, 1, &unknown, NULL, &error);
  EXPECT(bad && !error);
  EXPECT(clBuildProgram(bad, 0, NULL, NULL, NULL, NULL) ==
         CL_BUILD_PROGRAM_FAILURE);
  OK(clGetProgramBuildInfo(bad, device, CL_PROGRAM_BUILD_STATUS, sizeof(status),
                           &status, NULL));
  EXPECT(status == CL_BUILD_ERROR);
  OK(clGetProgramBuildInfo(bad, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                           &required));
  EXPECT(required > 1);
  OK(clReleaseProgram(bad));
  kernel = clCreateKernel(program, UDONCL_KERNEL_ENTRY, &error);
  EXPECT(kernel && !error);
  EXPECT(clBuildProgram(program, 0, NULL, NULL, NULL, NULL) ==
         CL_INVALID_OPERATION);
  buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 4, NULL, &error);
  EXPECT(buffer && !error);
  copy_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 4, NULL, &error);
  EXPECT(copy_buffer && !error);
  foreign = clCreateBuffer(other, CL_MEM_WRITE_ONLY, 4, NULL, &error);
  EXPECT(foreign && !error);
  EXPECT(!clCreateSubBuffer(buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, NULL,
                            &error) &&
         error == CL_INVALID_OPERATION);
  EXPECT(!clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0, 4, 0, NULL,
                             NULL, &error) &&
         error == CL_INVALID_OPERATION);
  EXPECT(clEnqueueUnmapMemObject(queue, buffer, bytes, 0, NULL, NULL) ==
         CL_INVALID_OPERATION);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &one, &one, 0, NULL,
                                NULL) == CL_INVALID_KERNEL_ARGS);
  EXPECT(clSetKernelArg(kernel, 2, 4, &scalar) == CL_INVALID_ARG_INDEX);
  EXPECT(clSetKernelArg(kernel, 1, 2, &scalar) == CL_INVALID_ARG_SIZE);
  EXPECT(clSetKernelArg(kernel, 1, 4, NULL) == CL_INVALID_ARG_VALUE);
  EXPECT(clSetKernelArg(kernel, 0, sizeof(buffer) - 1, &buffer) ==
         CL_INVALID_ARG_SIZE);
  OK(clSetKernelArg(kernel, 0, sizeof(buffer), &foreign));
  OK(clSetKernelArg(kernel, 1, 4, &scalar));
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &one, &one, 0, NULL,
                                NULL) == CL_INVALID_CONTEXT);
  OK(clSetKernelArg(kernel, 0, sizeof(buffer), &buffer));
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 0, NULL, &one, &one, 0, NULL,
                                NULL) == CL_INVALID_WORK_DIMENSION);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &two, &one, 0, NULL,
                                NULL) == CL_INVALID_GLOBAL_WORK_SIZE);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, &offset, &one, &one, 0, NULL,
                                NULL) == CL_INVALID_GLOBAL_OFFSET);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &one, &two, 0, NULL,
                                NULL) == CL_INVALID_WORK_GROUP_SIZE);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &one, &one, 1, NULL,
                                NULL) == CL_INVALID_EVENT_WAIT_LIST);
  EXPECT(clEnqueueReadBuffer(queue, foreign, CL_TRUE, 0, 4, &actual, 0, NULL,
                             NULL) == CL_INVALID_CONTEXT);
  EXPECT(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 3, 2, &actual, 0, NULL,
                             NULL) == CL_INVALID_VALUE);
  OK(clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, 2, bytes, 0, NULL, NULL));
  OK(clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 2, 2, bytes + 2, 0, NULL,
                          NULL));
  OK(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, 4, copied, 0, NULL, NULL));
  EXPECT(!memcmp(bytes, copied, 4));
  EXPECT(clEnqueueCopyBuffer(queue, copy_buffer, buffer, 0, 0, 4, 0, NULL,
                             NULL) == CL_INVALID_OPERATION);
  EXPECT(clEnqueueCopyBuffer(queue, buffer, buffer, 0, 1, 3, 0, NULL, NULL) ==
         CL_MEM_COPY_OVERLAP);
  EXPECT(clEnqueueCopyBuffer(queue, buffer, foreign, 0, 0, 4, 0, NULL, NULL) ==
         CL_INVALID_CONTEXT);
  EXPECT(clEnqueueCopyBufferRect(queue, buffer, copy_buffer, NULL, NULL, NULL,
                                 0, 0, 0, 0, 0, NULL,
                                 NULL) == CL_INVALID_OPERATION);
  OK(clEnqueueCopyBuffer(queue, buffer, copy_buffer, 0, 0, 4, 0, NULL,
                         &copied_event));
  memset(copied, 0, sizeof(copied));
  OK(clEnqueueReadBuffer(queue, copy_buffer, CL_TRUE, 0, 4, copied, 1,
                         &copied_event, NULL));
  EXPECT(!memcmp(bytes, copied, 4));
  OK(clReleaseEvent(copied_event));
  OK(clEnqueueWriteBuffer(other_queue, foreign, CL_TRUE, 0, 4, bytes, 0, NULL,
                          &foreign_event));
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &one, &one, 1,
                                &foreign_event, NULL) == CL_INVALID_CONTEXT);
  OK(clReleaseEvent(foreign_event));
  UDONCL_ORACLE(&expected, scalar);
  saved = buffer;
  buffer = NULL;
  scalar = -999; /* Both argument values were copied. */
  OK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &one, NULL, 0, NULL,
                            &done));
  OK(clGetEventInfo(done, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status),
                    &status, NULL));
  EXPECT(status == CL_COMPLETE);
  OK(clWaitForEvents(1, &done));
  OK(clEnqueueReadBuffer(queue, saved, CL_TRUE, 0, 4, &actual, 1, &done,
                         &read));
  EXPECT(!memcmp(&expected, &actual, 4));
  EXPECT(clWaitForEvents(0, NULL) == CL_INVALID_VALUE);
  OK(clFlush(queue));
  OK(clFinish(queue));
  OK(clReleaseContext(context));
  OK(clReleaseProgram(program)); /* Children keep parents alive. */
  OK(clReleaseMemObject(saved));
  OK(clReleaseMemObject(copy_buffer));
  buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 4, NULL, &error);
  EXPECT(buffer && !error);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &one, NULL, 0, NULL,
                                NULL) == CL_INVALID_MEM_OBJECT);
  OK(clReleaseMemObject(buffer));
  OK(clReleaseKernel(kernel));
  OK(clReleaseCommandQueue(queue)); /* Events keep queue alive. */
  OK(clWaitForEvents(1, &read));
  OK(clReleaseEvent(done));
  OK(clReleaseEvent(read));
  OK(clReleaseMemObject(foreign));
  OK(clReleaseCommandQueue(other_queue));
  OK(clReleaseContext(other));
  return 0;
}
