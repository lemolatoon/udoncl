/* The Conversation Host of the 4096-token Conversation and Chat Artifacts
 * (onnx-transformer-tiny-conversation-4k and onnx-transformer-tiny-chat-4k).
 *
 * It is the token-array Conversation Host of conversation_host.c at a Context
 * Capacity of 4096 tokens: the same exports, generation protocol and
 * multi-turn contract, with a 4096-token prompt workspace. CMake specializes
 * the M=1 graph of these artifacts to past_sequence_length = 4095 and a
 * 4096-long attention mask and compiles the text layer with the same
 * capacity; this file pins it so that the three cannot drift apart.
 *
 * TinyLlama-1.1B was trained on sequences of 2048 positions
 * (max_position_embeddings). Its rotary embedding is computed from the
 * position, so positions beyond 2048 run, but the model has not learned
 * them and replies may degrade there. */
#ifndef TINYLLAMA_CONTEXT_CAPACITY
#define TINYLLAMA_CONTEXT_CAPACITY 4096u
#endif
#if TINYLLAMA_CONTEXT_CAPACITY != 4096u
#error "conversation_host_4k.c is the Host of a 4096-token Context Capacity"
#endif

#include "conversation_host.c"
