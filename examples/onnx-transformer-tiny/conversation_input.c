#define CL_TARGET_OPENCL_VERSION 120
#include "conversation_input.h"

#include <udoncl/onnx_generated_binding.h>

static udononnxStatus_t upload_input(cl_command_queue queue,
                                     uint32_t input_index,
                                     const int64_t *values) {
  cl_mem memory = 0;
  size_t byte_offset = 0;
  size_t byte_length = 0;
  udononnxStatus_t status = udon_onnx_binding_resolve_input(
      input_index, &memory, &byte_offset, &byte_length);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!memory ||
      byte_length != TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT * sizeof(int64_t))
    return UDONONNX_STATUS_TYPE_MISMATCH;
  return clEnqueueWriteBuffer(queue, memory, CL_FALSE, byte_offset, byte_length,
                              values, 0, 0, 0) == CL_SUCCESS
             ? UDONONNX_STATUS_SUCCESS
             : UDONONNX_STATUS_OPENCL_ERROR;
}

udononnxStatus_t
tinyllamaPrepareFixedPrefillInputs(const udontokenizers_model_t *tokenizer,
                                   udon_string_t text, cl_command_queue queue) {
  int32_t token_ids[TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT];
  int64_t input_ids[TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT];
  int64_t attention_mask[TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT];
  int64_t position_ids[TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT];
  int32_t token_count = 0;
  int32_t index;
  udontokenizersStatus_t tokenizer_status;
  udononnxStatus_t status;
  if (!tokenizer || !queue)
    return UDONONNX_STATUS_INVALID_VALUE;
  tokenizer_status = udontokenizersEncodeRaw(
      tokenizer, text, token_ids, TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT,
      &token_count);
  if (tokenizer_status == UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL)
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  if (tokenizer_status != UDONTOKENIZERS_STATUS_SUCCESS ||
      token_count != TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT; ++index) {
    input_ids[index] = token_ids[index];
    attention_mask[index] = 1;
    position_ids[index] = index;
  }
  status = upload_input(queue, 0, input_ids);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  status = upload_input(queue, 1, attention_mask);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return upload_input(queue, 2, position_ids);
}
