/* Optional text layer of the TinyLlama conversation example.
 *
 * The token-array conversation Host stays independent of any text service;
 * this file composes it with the generic UdonTokenizers runtime and the
 * project-local tables compiled from the pinned tokenizer files. The chat
 * template is compiled at build time, so no Jinja or JSON is interpreted at
 * run time. Decoded text is returned as UTF-16 code units, one per int32
 * element of a caller-owned array. */
#include <stdint.h>
#include <udoncl/array.h>
#include <udoncl/export.h>
#include <udoncl/onnx_types.h>
#include <udoncl/string.h>
#include <udontokenizers.h>

#include <tinyllama_tokenizer.h>

#include "conversation_host.h"

#define TINYLLAMA_TEXT_CAPACITY                                                \
  (TINYLLAMA_GENERATED_CAPACITY * TINYLLAMA_TOKENIZER_MAX_PIECE_CODE_UNITS)

static int32_t text_units[TINYLLAMA_TEXT_CAPACITY];
static int32_t last_tokenizer_status;

static udononnxStatus_t map_tokenizer_status(udontokenizersStatus_t status) {
  last_tokenizer_status = (int32_t)status;
  switch (status) {
  case UDONTOKENIZERS_STATUS_SUCCESS:
    return UDONONNX_STATUS_SUCCESS;
  case UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL:
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  case UDONTOKENIZERS_STATUS_INVALID_UTF16:
  case UDONTOKENIZERS_STATUS_INVALID_ARGUMENT:
    return UDONONNX_STATUS_INVALID_VALUE;
  default:
    return UDONONNX_STATUS_EXECUTION_FAILURE;
  }
}

/* Replaces the prompt with BOS + the chat-templated conversation
 * [system (omitted when empty), user] + the generation prompt. */
UDONCL_EXPORT udononnxStatus_t
tinyllama_set_chat_prompt(udon_string_t system_text, udon_string_t user_text) {
  udontokenizers_model_t model;
  udontokenizers_chat_template_t chat;
  udontokenizers_encoder_t encoder;
  int32_t *workspace = 0;
  int32_t count = 0;
  udontokenizersStatus_t status;
  udononnxStatus_t host = tinyllamaConversationPreparePrompt(&workspace);
  last_tokenizer_status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (host != UDONONNX_STATUS_SUCCESS)
    return host;
  tinyllama_tokenizerInitializeModel(&model);
  tinyllama_tokenizerInitializeChatTemplate(&chat);
  status = udontokenizersEncoderBeginRaw(
      &model, &encoder, workspace, (int32_t)TINYLLAMA_CONTEXT_CAPACITY, 1);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS && udonStringLength(system_text))
    status = udontokenizersEncoderAppendChatMessageRaw(
        &model, &chat, &encoder, UDONTOKENIZERS_CHAT_ROLE_SYSTEM, system_text,
        workspace);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status = udontokenizersEncoderAppendChatMessageRaw(
        &model, &chat, &encoder, UDONTOKENIZERS_CHAT_ROLE_USER, user_text,
        workspace);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status = udontokenizersEncoderAppendChatGenerationPromptRaw(
        &model, &chat, &encoder, workspace);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status =
        udontokenizersEncoderFinishRaw(&model, &encoder, workspace, &count);
  host = map_tokenizer_status(status);
  if (host != UDONONNX_STATUS_SUCCESS)
    return host;
  return tinyllamaConversationCommitPrompt((uint32_t)count);
}

UDONCL_EXPORT uint32_t tinyllama_prompt_token_count(void) {
  const int32_t *tokens = 0;
  uint32_t count = 0;
  if (tinyllamaConversationPrompt(&tokens, &count) != UDONONNX_STATUS_SUCCESS)
    return 0;
  return count;
}

UDONCL_EXPORT udononnxStatus_t
tinyllama_read_prompt_tokens(udon_int32_array_t destination) {
  const int32_t *tokens = 0;
  uint32_t count = 0;
  uint32_t index;
  udononnxStatus_t status = tinyllamaConversationPrompt(&tokens, &count);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!destination || udonInt32ArrayLength(destination) < (int32_t)count)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < count; ++index)
    udonInt32ArraySet(destination, (int32_t)index, tokens[index]);
  return UDONONNX_STATUS_SUCCESS;
}

/* Decodes the completed generation, skipping special tokens such as EOS. */
static udononnxStatus_t decode_generated(int32_t *length) {
  udontokenizers_model_t model;
  udontokenizers_decoder_t decoder;
  const int32_t *tokens = 0;
  uint32_t count = 0;
  uint32_t index;
  udontokenizersStatus_t status;
  udononnxStatus_t host = tinyllamaConversationGenerated(&tokens, &count);
  *length = 0;
  last_tokenizer_status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (host != UDONONNX_STATUS_SUCCESS)
    return host;
  tinyllama_tokenizerInitializeModel(&model);
  status = udontokenizersDecoderBeginRaw(&model, &decoder, text_units,
                                         TINYLLAMA_TEXT_CAPACITY, 1);
  for (index = 0; status == UDONTOKENIZERS_STATUS_SUCCESS && index < count;
       ++index)
    status = udontokenizersDecoderAppendRaw(&model, &decoder, tokens[index],
                                            text_units);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status =
        udontokenizersDecoderFinishRaw(&model, &decoder, text_units, length);
  return map_tokenizer_status(status);
}

UDONCL_EXPORT uint32_t tinyllama_generated_text_length(void) {
  int32_t length = 0;
  if (decode_generated(&length) != UDONONNX_STATUS_SUCCESS)
    return 0;
  return (uint32_t)length;
}

UDONCL_EXPORT udononnxStatus_t
tinyllama_read_generated_text(udon_int32_array_t destination) {
  int32_t length = 0;
  int32_t index;
  udononnxStatus_t status = decode_generated(&length);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!destination || udonInt32ArrayLength(destination) < length)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < length; ++index)
    udonInt32ArraySet(destination, index, text_units[index]);
  return UDONONNX_STATUS_SUCCESS;
}

/* The UdonTokenizers status of the last text operation, for diagnostics. */
UDONCL_EXPORT int32_t tinyllama_last_tokenizer_status(void) {
  return last_tokenizer_status;
}
