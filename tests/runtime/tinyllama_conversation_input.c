#include <CL/cl.h>
#include <stdint.h>
#include <string.h>
#include <udoncl/onnx_types.h>
#include <udoncl/string.h>
#include <udontokenizers.h>
#include <VRC/Runtime/OpenCL/Backend.h>

#include "conversation_input.h"

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition))                                                          \
      return __LINE__;                                                         \
  } while (0)

size_t udoncl_cl_buffer_limit(void) { return 1024 * 1024; }

int udoncl_cl_gather_profile(const char *name,
                             struct udoncl_cl_gather_profile *profile) {
  (void)name;
  (void)profile;
  return 0;
}

int udoncl_cl_draw_gather(uint32_t kernel, const unsigned char *const *inputs,
                          size_t input_count, size_t bytes,
                          const unsigned char *scalars, size_t scalar_bytes,
                          unsigned char *output) {
  (void)kernel;
  (void)inputs;
  (void)input_count;
  (void)bytes;
  (void)scalars;
  (void)scalar_bytes;
  (void)output;
  return 0;
}

int udoncl_cl_source_matches(const char *source, size_t length) {
  (void)source;
  (void)length;
  return 0;
}

int udoncl_cl_kernel_matches(const char *name) {
  (void)name;
  return 0;
}

int udoncl_cl_draw(uint32_t buffer, uint32_t generation,
                   const unsigned char scalar[4], unsigned char result[4]) {
  (void)buffer;
  (void)generation;
  (void)scalar;
  (void)result;
  return 0;
}

static cl_mem input_buffers[3];

int main(void) {
  static const uint32_t scalar_values[] = {0x61};
  static const int32_t scalar_tokens[] = {7};
  static const int32_t byte_tokens[256] = {0};
  static const uint16_t text_data[] = {0x61, 0x61, 0x61, 0x61};
  static const uint16_t long_text_data[] = {0x61, 0x61, 0x61, 0x61, 0x61};
  udontokenizers_model_t tokenizer = {scalar_values,
                                      scalar_tokens,
                                      1,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                      byte_tokens,
                                      -1,
                                      -1,
                                      -1,
                                      -1};
  udon_string_native_t text = {text_data, 4};
  udon_string_native_t long_text = {long_text_data, 5};
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context context = 0;
  cl_command_queue queue = 0;
  cl_int error = CL_SUCCESS;
  int64_t values[4];
  int32_t index;

  EXPECT(clGetPlatformIDs(1, &platform, 0) == CL_SUCCESS);
  EXPECT(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) ==
         CL_SUCCESS);
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  EXPECT(context && error == CL_SUCCESS);
  queue = clCreateCommandQueue(context, device, 0, &error);
  EXPECT(queue && error == CL_SUCCESS);
  for (index = 0; index < 3; ++index) {
    int64_t sentinel[4] = {-1, -1, -1, -1};
    input_buffers[index] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(sentinel), 0, &error);
    EXPECT(input_buffers[index] && error == CL_SUCCESS);
    EXPECT(clEnqueueWriteBuffer(queue, input_buffers[index], CL_TRUE, 0,
                                sizeof(sentinel), sentinel, 0, 0,
                                0) == CL_SUCCESS);
  }

  /* A text of too many tokens writes nothing. */
  EXPECT(tinyllamaPrepareFixedPrefillInputs(
             &tokenizer, &long_text, queue, input_buffers[0], input_buffers[1],
             input_buffers[2]) == UDONONNX_STATUS_CAPACITY_EXCEEDED);
  EXPECT(clFinish(queue) == CL_SUCCESS);
  for (index = 0; index < 3; ++index) {
    int32_t element;
    EXPECT(clEnqueueReadBuffer(queue, input_buffers[index], CL_TRUE, 0,
                               sizeof(values), values, 0, 0, 0) == CL_SUCCESS);
    for (element = 0; element < 4; ++element)
      EXPECT(values[element] == -1);
  }
  EXPECT(tinyllamaPrepareFixedPrefillInputs(
             &tokenizer, &text, queue, 0, input_buffers[1], input_buffers[2]) ==
         UDONONNX_STATUS_INVALID_VALUE);

  EXPECT(tinyllamaPrepareFixedPrefillInputs(
             &tokenizer, &text, queue, input_buffers[0], input_buffers[1],
             input_buffers[2]) == UDONONNX_STATUS_SUCCESS);
  EXPECT(clFinish(queue) == CL_SUCCESS);
  EXPECT(clEnqueueReadBuffer(queue, input_buffers[0], CL_TRUE, 0,
                             sizeof(values), values, 0, 0, 0) == CL_SUCCESS);
  for (index = 0; index < 4; ++index)
    EXPECT(values[index] == 7);
  EXPECT(clEnqueueReadBuffer(queue, input_buffers[1], CL_TRUE, 0,
                             sizeof(values), values, 0, 0, 0) == CL_SUCCESS);
  for (index = 0; index < 4; ++index)
    EXPECT(values[index] == 1);
  EXPECT(clEnqueueReadBuffer(queue, input_buffers[2], CL_TRUE, 0,
                             sizeof(values), values, 0, 0, 0) == CL_SUCCESS);
  for (index = 0; index < 4; ++index)
    EXPECT(values[index] == index);

  for (index = 0; index < 3; ++index)
    EXPECT(clReleaseMemObject(input_buffers[index]) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(queue) == CL_SUCCESS);
  EXPECT(clReleaseContext(context) == CL_SUCCESS);
  return 0;
}
