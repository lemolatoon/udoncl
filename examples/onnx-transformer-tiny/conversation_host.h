#ifndef UDONCL_TINYLLAMA_CONVERSATION_HOST_H
#define UDONCL_TINYLLAMA_CONVERSATION_HOST_H

#include <stdint.h>
#include <udoncl/onnx_types.h>

/* Context Capacity is a build parameter of the example. The ONNX graph is
 * specialized to past_sequence_length = capacity - 1 and an attention mask of
 * length capacity by the same CMake setting. */
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
