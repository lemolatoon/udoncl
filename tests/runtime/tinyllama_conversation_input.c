#include <CL/cl.h>
#include <stdint.h>
#include <string.h>
#include <udoncl/onnx_generated_binding.h>
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
static int32_t resolve_counts[3];

udononnxStatus_t udon_onnx_binding_resolve_input(uint32_t input_index,
                                                 cl_mem *memory,
                                                 size_t *byte_offset,
                                                 size_t *byte_length) {
  if (input_index >= 3 || !memory || !byte_offset || !byte_length)
    return UDONONNX_STATUS_INVALID_VALUE;
  ++resolve_counts[input_index];
  *memory = input_buffers[input_index];
  *byte_offset = 8;
  *byte_length = 4 * sizeof(int64_t);
  return UDONONNX_STATUS_SUCCESS;
}

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
    input_buffers[index] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, 48, 0, &error);
    EXPECT(input_buffers[index] && error == CL_SUCCESS);
  }

  EXPECT(tinyllamaPrepareFixedPrefillInputs(&tokenizer, &long_text, queue) ==
         UDONONNX_STATUS_CAPACITY_EXCEEDED);
  for (index = 0; index < 3; ++index)
    EXPECT(resolve_counts[index] == 0);

  EXPECT(tinyllamaPrepareFixedPrefillInputs(&tokenizer, &text, queue) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(clFinish(queue) == CL_SUCCESS);
  EXPECT(clEnqueueReadBuffer(queue, input_buffers[0], CL_TRUE, 8,
                             sizeof(values), values, 0, 0, 0) == CL_SUCCESS);
  for (index = 0; index < 4; ++index)
    EXPECT(values[index] == 7);
  EXPECT(clEnqueueReadBuffer(queue, input_buffers[1], CL_TRUE, 8,
                             sizeof(values), values, 0, 0, 0) == CL_SUCCESS);
  for (index = 0; index < 4; ++index)
    EXPECT(values[index] == 1);
  EXPECT(clEnqueueReadBuffer(queue, input_buffers[2], CL_TRUE, 8,
                             sizeof(values), values, 0, 0, 0) == CL_SUCCESS);
  for (index = 0; index < 4; ++index)
    EXPECT(values[index] == index);
  for (index = 0; index < 3; ++index)
    EXPECT(resolve_counts[index] == 1);

  for (index = 0; index < 3; ++index)
    EXPECT(clReleaseMemObject(input_buffers[index]) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(queue) == CL_SUCCESS);
  EXPECT(clReleaseContext(context) == CL_SUCCESS);
  return 0;
}
