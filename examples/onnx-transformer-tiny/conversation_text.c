/* Optional text layer of the TinyLlama conversation example.
 *
 * The token-array conversation Host stays independent of any text service;
 * this file composes it with the generic UdonTokenizers runtime and the
 * project-local tables compiled from the pinned tokenizer files. The chat
 * template is compiled at build time, so no Jinja or JSON is interpreted at
 * run time. Decoded text is returned as UTF-16 code units, one per int32
 * element of a caller-owned array.
 *
 * Text in: a prompt is tokenized either in one call or in bounded slices
 * (begin, then continue with a work budget until nothing is pending), so a
 * long message never has to fit in one Udon event.
 *
 * Multi-turn: a chat prompt starts a conversation. After a reply, the next
 * user message is appended as a turn (begin, then continue, the same way):
 * the reply enters the context as its generated tokens, and only the template
 * text from the end of the reply through the next generation prompt is
 * encoded, without BOS. A conversation that would overflow the Context
 * Capacity is rejected unchanged; ResetChat starts a new one.
 *
 * Text out: generated tokens are decoded incrementally. The streamed text is
 * the prefix of the reply that can no longer change; it only grows, and once
 * the generation completed it is the whole reply. Each generation streams its
 * own reply from the start. Polling costs time proportional to the tokens
 * generated since the previous poll. */
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
/* A token covers at most its piece length in symbols, so a text with more
 * symbols than this cannot fit the Context Capacity either. */
#define TINYLLAMA_PROMPT_SYMBOL_CAPACITY                                       \
  (TINYLLAMA_CONTEXT_CAPACITY * TINYLLAMA_TOKENIZER_MAX_PIECE_CODE_UNITS)
#define TINYLLAMA_PROMPT_WORKSPACE_WORDS                                       \
  UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(TINYLLAMA_PROMPT_SYMBOL_CAPACITY)
#define TINYLLAMA_UNBOUNDED_WORK 0x7fffffffu

/* Stages of building a chat prompt or turn. */
#define PROMPT_IDLE 0
#define PROMPT_SYSTEM 1
#define PROMPT_USER 2
#define PROMPT_TURN 3
#define PROMPT_MERGE 4

static int32_t prompt_workspace[TINYLLAMA_PROMPT_WORKSPACE_WORDS];
static udontokenizers_encoder_t prompt_encoder;
static int32_t prompt_stage;
static int32_t prompt_position;
/* 1 while the text being built is a turn appended to the conversation. */
static int32_t prompt_is_turn;

static int32_t text_units[TINYLLAMA_TEXT_CAPACITY];
static udontokenizers_decoder_t stream;
static uint32_t stream_serial;
static uint32_t stream_tokens;
static int32_t stream_started;
static int32_t stream_finished;

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

/* Starts replacing the prompt with BOS + the chat-templated conversation
 * [system (omitted when empty), user] + the generation prompt. The previous
 * prompt is cleared; the new one exists once no work is pending. */
static udononnxStatus_t begin_text(udononnxStatus_t host, int32_t is_turn) {
  udontokenizers_model_t model;
  prompt_stage = PROMPT_IDLE;
  last_tokenizer_status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (host != UDONONNX_STATUS_SUCCESS)
    return host;
  tinyllama_tokenizerInitializeModel(&model);
  /* A turn continues the conversation, so only a new prompt starts with
   * BOS. */
  host = map_tokenizer_status(udontokenizersEncoderBeginRaw(
      &model, &prompt_encoder, prompt_workspace,
      TINYLLAMA_PROMPT_WORKSPACE_WORDS, !is_turn));
  if (host != UDONONNX_STATUS_SUCCESS)
    return host;
  prompt_stage = is_turn ? PROMPT_TURN : PROMPT_SYSTEM;
  prompt_is_turn = is_turn;
  prompt_position = 0;
  return UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t tinyllama_begin_chat_prompt(void) {
  int32_t *tokens = 0;
  return begin_text(tinyllamaConversationPreparePrompt(&tokens), 0);
}

/* Starts appending the next user message to the conversation, after a
 * completed reply: the reply's end-of-turn text, the user message and the
 * generation prompt. A turn that was committed but not generated yet is
 * replaced. */
UDONCL_EXPORT udononnxStatus_t tinyllama_begin_chat_turn(void) {
  int32_t *tokens = 0;
  uint32_t capacity = 0;
  return begin_text(tinyllamaConversationPrepareTurn(&tokens, &capacity), 1);
}

static udononnxStatus_t commit_text(void) {
  int32_t *tokens = 0;
  uint32_t capacity = TINYLLAMA_CONTEXT_CAPACITY;
  int32_t count = prompt_encoder.count;
  int32_t index;
  udononnxStatus_t host =
      prompt_is_turn ? tinyllamaConversationPrepareTurn(&tokens, &capacity)
                     : tinyllamaConversationPreparePrompt(&tokens);
  if (host != UDONONNX_STATUS_SUCCESS)
    return host;
  if (count > (int32_t)capacity)
    return map_tokenizer_status(UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);
  for (index = 0; index < count; ++index)
    tokens[index] = prompt_workspace[index];
  return prompt_is_turn ? tinyllamaConversationCommitTurn((uint32_t)count)
                        : tinyllamaConversationCommitPrompt((uint32_t)count);
}

/* The bounded work of a begun prompt or turn; see continue_chat_prompt. */
static udononnxStatus_t continue_text(udon_string_t system_text,
                                      udon_string_t user_text,
                                      uint32_t max_work) {
  udontokenizers_model_t model;
  udontokenizers_chat_template_t chat;
  udontokenizersStatus_t status = UDONTOKENIZERS_STATUS_SUCCESS;
  int32_t remaining =
      max_work > TINYLLAMA_UNBOUNDED_WORK ? 0x7fffffff : (int32_t)max_work;
  int32_t finished = 0;
  if (remaining <= 0)
    return UDONONNX_STATUS_INVALID_VALUE;
  tinyllama_tokenizerInitializeModel(&model);
  tinyllama_tokenizerInitializeChatTemplate(&chat);
  if (prompt_stage == PROMPT_SYSTEM && !udonStringLength(system_text))
    prompt_stage = PROMPT_USER;
  while (status == UDONTOKENIZERS_STATUS_SUCCESS && remaining > 0 &&
         prompt_stage != PROMPT_MERGE) {
    int32_t before = prompt_position;
    if (prompt_stage == PROMPT_SYSTEM)
      status = udontokenizersEncoderAppendChatMessagePartRaw(
          &model, &chat, &prompt_encoder, UDONTOKENIZERS_CHAT_ROLE_SYSTEM,
          system_text, &prompt_position, remaining, &finished,
          prompt_workspace);
    else if (prompt_stage == PROMPT_USER)
      status = udontokenizersEncoderAppendChatMessagePartRaw(
          &model, &chat, &prompt_encoder, UDONTOKENIZERS_CHAT_ROLE_USER,
          user_text, &prompt_position, remaining, &finished, prompt_workspace);
    else
      status = udontokenizersEncoderAppendChatTurnPartRaw(
          &model, &chat, &prompt_encoder, UDONTOKENIZERS_CHAT_ROLE_ASSISTANT,
          UDONTOKENIZERS_CHAT_ROLE_USER, user_text, &prompt_position, remaining,
          &finished, prompt_workspace);
    if (status != UDONTOKENIZERS_STATUS_SUCCESS)
      break;
    remaining -= prompt_position - before;
    if (!finished)
      continue;
    prompt_position = 0;
    if (prompt_stage == PROMPT_SYSTEM) {
      prompt_stage = PROMPT_USER;
    } else {
      status = udontokenizersEncoderAppendChatGenerationPromptRaw(
          &model, &chat, &prompt_encoder, prompt_workspace);
      prompt_stage = PROMPT_MERGE;
    }
  }
  finished = 0;
  if (status == UDONTOKENIZERS_STATUS_SUCCESS && remaining > 0)
    status = udontokenizersEncoderStepRaw(
        &model, &prompt_encoder, prompt_workspace, remaining, &finished);
  if (status != UDONTOKENIZERS_STATUS_SUCCESS) {
    prompt_stage = PROMPT_IDLE;
    return map_tokenizer_status(status);
  }
  if (!finished)
    return UDONONNX_STATUS_SUCCESS;
  prompt_stage = PROMPT_IDLE;
  return commit_text();
}

/* Does at most `max_work` units of prompt work: one unit consumes about one
 * UTF-16 code unit of the template and messages or does one unit of merge
 * work. The caller passes the same texts on every call of one prompt. The
 * prompt is committed by the call that leaves no work pending; any failure
 * abandons it. */
UDONCL_EXPORT udononnxStatus_t tinyllama_continue_chat_prompt(
    udon_string_t system_text, udon_string_t user_text, uint32_t max_work) {
  if (prompt_stage == PROMPT_IDLE || prompt_is_turn)
    return UDONONNX_STATUS_INVALID_STATE;
  return continue_text(system_text, user_text, max_work);
}

/* The bounded work of a begun turn, as ContinueChatPrompt. The turn is
 * committed by the call that leaves no work pending. A turn that does not
 * fit the Context Capacity fails with CAPACITY_EXCEEDED and leaves the
 * conversation unchanged. */
UDONCL_EXPORT udononnxStatus_t
tinyllama_continue_chat_turn(udon_string_t user_text, uint32_t max_work) {
  if (prompt_stage == PROMPT_IDLE || !prompt_is_turn)
    return UDONONNX_STATUS_INVALID_STATE;
  return continue_text(user_text, user_text, max_work);
}

/* Discards the conversation, including a prompt or turn being built, and
 * starts a new one; the Host Reset. */
UDONCL_EXPORT udononnxStatus_t tinyllama_reset_chat(void) {
  udononnxStatus_t status = tinyllamaConversationReset();
  if (status == UDONONNX_STATUS_SUCCESS) {
    prompt_stage = PROMPT_IDLE;
    prompt_is_turn = 0;
    last_tokenizer_status = UDONTOKENIZERS_STATUS_SUCCESS;
  }
  return status;
}

/* 1 while a begun chat prompt or turn still has work pending, else 0. */
UDONCL_EXPORT uint32_t tinyllama_chat_prompt_pending(void) {
  return prompt_stage != PROMPT_IDLE;
}

/* Replaces the prompt in one call. */
UDONCL_EXPORT udononnxStatus_t
tinyllama_set_chat_prompt(udon_string_t system_text, udon_string_t user_text) {
  udononnxStatus_t status = tinyllama_begin_chat_prompt();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return tinyllama_continue_chat_prompt(system_text, user_text,
                                        TINYLLAMA_UNBOUNDED_WORK);
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

/* Decodes the tokens generated since the previous call, skipping special
 * tokens such as EOS, and ends the text once the generation completed. */
static udononnxStatus_t advance_stream(int32_t *is_completed) {
  udontokenizers_model_t model;
  const int32_t *tokens = 0;
  uint32_t count = 0;
  uint32_t serial = 0;
  int32_t length = 0;
  udontokenizersStatus_t status = UDONTOKENIZERS_STATUS_SUCCESS;
  udononnxStatus_t host = tinyllamaConversationGenerationProgress(
      &tokens, &count, &serial, is_completed);
  if (host != UDONONNX_STATUS_SUCCESS)
    return host;
  tinyllama_tokenizerInitializeModel(&model);
  if (!stream_started || serial != stream_serial || count < stream_tokens) {
    stream_started = 1;
    stream_serial = serial;
    stream_tokens = 0;
    stream_finished = 0;
    status = udontokenizersDecoderBeginRaw(&model, &stream, text_units,
                                           TINYLLAMA_TEXT_CAPACITY, 1);
  }
  while (status == UDONTOKENIZERS_STATUS_SUCCESS && stream_tokens < count) {
    status = udontokenizersDecoderAppendRaw(&model, &stream,
                                            tokens[stream_tokens], text_units);
    ++stream_tokens;
  }
  if (status == UDONTOKENIZERS_STATUS_SUCCESS && *is_completed &&
      !stream_finished) {
    status =
        udontokenizersDecoderFinishRaw(&model, &stream, text_units, &length);
    stream_finished = 1;
  }
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status = (udontokenizersStatus_t)stream.status;
  return map_tokenizer_status(status);
}

/* The number of UTF-16 code units of the reply that are final so far. It
 * never decreases during one generation and is valid while the generation
 * is still running; incomplete characters are not counted. */
UDONCL_EXPORT uint32_t tinyllama_streamed_text_length(void) {
  int32_t is_completed = 0;
  udononnxStatus_t status = advance_stream(&is_completed);
  if (status == UDONONNX_STATUS_DISPOSED ||
      status == UDONONNX_STATUS_NOT_INITIALIZED || !stream_started)
    return 0;
  return (uint32_t)stream.final_count;
}

/* Copies the final code units [offset, offset + count) of the reply to the
 * start of `destination`. */
UDONCL_EXPORT udononnxStatus_t tinyllama_read_streamed_text(
    udon_int32_array_t destination, uint32_t offset, uint32_t count) {
  int32_t is_completed = 0;
  uint32_t index;
  udononnxStatus_t status = advance_stream(&is_completed);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!destination || offset > (uint32_t)stream.final_count ||
      count > (uint32_t)stream.final_count - offset ||
      (uint32_t)udonInt32ArrayLength(destination) < count)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < count; ++index)
    udonInt32ArraySet(destination, (int32_t)index, text_units[offset + index]);
  return UDONONNX_STATUS_SUCCESS;
}

/* The length of the whole reply; zero until the generation completed. */
UDONCL_EXPORT uint32_t tinyllama_generated_text_length(void) {
  int32_t is_completed = 0;
  if (advance_stream(&is_completed) != UDONONNX_STATUS_SUCCESS || !is_completed)
    return 0;
  return (uint32_t)stream.final_count;
}

/* Copies the whole reply; requires a completed generation. */
UDONCL_EXPORT udononnxStatus_t
tinyllama_read_generated_text(udon_int32_array_t destination) {
  const int32_t *tokens = 0;
  uint32_t count = 0;
  int32_t is_completed = 0;
  int32_t index;
  udononnxStatus_t status = tinyllamaConversationGenerated(&tokens, &count);
  last_tokenizer_status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (status == UDONONNX_STATUS_SUCCESS)
    status = advance_stream(&is_completed);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!is_completed)
    return UDONONNX_STATUS_INVALID_STATE;
  if (!destination || udonInt32ArrayLength(destination) < stream.final_count)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < stream.final_count; ++index)
    udonInt32ArraySet(destination, index, text_units[index]);
  return UDONONNX_STATUS_SUCCESS;
}

/* The UdonTokenizers status of the last text operation, for diagnostics. */
UDONCL_EXPORT int32_t tinyllama_last_tokenizer_status(void) {
  return last_tokenizer_status;
}
