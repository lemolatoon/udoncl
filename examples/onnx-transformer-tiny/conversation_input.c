#define CL_TARGET_OPENCL_VERSION 120
#include "conversation_input.h"

/* Writes the TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT values of one int64 graph
 * input into the Device buffer the Host bound to it. */
static udononnxStatus_t upload_input(cl_command_queue queue, cl_mem memory,
                                     const int64_t *values) {
  if (!memory)
    return UDONONNX_STATUS_INVALID_VALUE;
  return clEnqueueWriteBuffer(queue, memory, CL_FALSE, 0,
                              TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT *
                                  sizeof(int64_t),
                              values, 0, 0, 0) == CL_SUCCESS
             ? UDONONNX_STATUS_SUCCESS
             : UDONONNX_STATUS_OPENCL_ERROR;
}

udononnxStatus_t tinyllamaPrepareFixedPrefillInputs(
    const udontokenizers_model_t *tokenizer, udon_string_t text,
    cl_command_queue queue, cl_mem input_ids_memory,
    cl_mem attention_mask_memory, cl_mem position_ids_memory) {
  int32_t token_ids[TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT];
  int32_t workspace[UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(
      TINYLLAMA_FIXED_PREFILL_SYMBOL_CAPACITY)];
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
      &token_count, workspace,
      UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(
          TINYLLAMA_FIXED_PREFILL_SYMBOL_CAPACITY));
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
  status = upload_input(queue, input_ids_memory, input_ids);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  status = upload_input(queue, attention_mask_memory, attention_mask);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return upload_input(queue, position_ids_memory, position_ids);
}
