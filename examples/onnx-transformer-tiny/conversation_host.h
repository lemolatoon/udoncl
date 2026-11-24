#ifndef UDONCL_TINYLLAMA_CONVERSATION_HOST_H
#define UDONCL_TINYLLAMA_CONVERSATION_HOST_H

#include <stdint.h>
#include <udoncl/onnx_types.h>

/* Context Capacity is a build parameter of the example. The ONNX graph's past
 * is bounded by the same CMake setting (DIMENSION_BOUNDS past_sequence_length
 * = capacity - 1), and one run computes one position. */
#ifndef TINYLLAMA_CONTEXT_CAPACITY
#define TINYLLAMA_CONTEXT_CAPACITY 64u
#endif
#ifndef TINYLLAMA_GENERATED_CAPACITY
#define TINYLLAMA_GENERATED_CAPACITY 32u
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Internal C interface between the token-array conversation Host and the
 * optional text layer. These functions are not exported to U#. */

/* Starts replacing the prompt: on success the bounded prompt workspace of
 * TINYLLAMA_CONTEXT_CAPACITY tokens is returned for direct writes and the
 * previous prompt is cleared. */
udononnxStatus_t tinyllamaConversationPreparePrompt(int32_t **workspace);
/* Validates and accepts the first token_count workspace tokens. */
udononnxStatus_t tinyllamaConversationCommitPrompt(uint32_t token_count);
/* Starts appending the next turn to the conversation after a completed
 * generation. The previous reply enters the context as its generated tokens;
 * a reply cut off by the token limit is closed with its never-run last token
 * and EOS. On success the workspace for the turn tokens and its capacity (the
 * Context Capacity left after the conversation and those closing tokens) are
 * returned and any previously committed turn is cleared; the conversation
 * itself is unchanged until the next generation. */
udononnxStatus_t tinyllamaConversationPrepareTurn(int32_t **workspace,
                                                  uint32_t *capacity);
/* Validates and accepts the first token_count turn workspace tokens; the
 * next generation feeds the closing tokens and the turn. */
udononnxStatus_t tinyllamaConversationCommitTurn(uint32_t token_count);
/* The exported Reset of the Conversation Host; an export is not called from
 * another one. */
udononnxStatus_t tinyllamaConversationReset(void);
udononnxStatus_t tinyllamaConversationPrompt(const int32_t **tokens,
                                             uint32_t *token_count);
/* Requires a successfully completed generation. */
udononnxStatus_t tinyllamaConversationGenerated(const int32_t **tokens,
                                                uint32_t *token_count);
/* The tokens generated so far, also while a generation is running. *serial
 * changes whenever a generation starts or the conversation is reset, so a
 * consumer can tell a continued token sequence from a new one; a failed
 * generation reports no tokens. *is_completed is set once the generation
 * finished successfully. */
udononnxStatus_t tinyllamaConversationGenerationProgress(const int32_t **tokens,
                                                         uint32_t *token_count,
                                                         uint32_t *serial,
                                                         int32_t *is_completed);

#ifdef __cplusplus
}
#endif

#endif
