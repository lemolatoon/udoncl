#ifndef UDONCL_TRANSFORMER_TENSORS_H
#define UDONCL_TRANSFORMER_TENSORS_H

/* The graph inputs and outputs of the models host.c binds, as X-macro lists
 * in ONNX order. Each entry is
 *
 *   X(variable, onnx_name, element_type, element_bytes, elements, rank, dims)
 *
 * `variable` is the UDONCL_PUBLIC cl_mem Host variable, named after the ONNX
 * tensor with every character outside [A-Za-z0-9_] replaced by '_' (the U#
 * methods and Unity buffers are named after it), `element_type` is the
 * UDONCL_PUBLIC element type (float or long), `elements` the element count
 * and `dims` the parenthesized dimensions. */

/* The synthetic transformer (generate_onnx.py) of four positions: no KV
 * concatenation, the presents are the pasts plus this run's projections. */
#define TRANSFORMER_TINY_INPUTS(X)                                             \
  X(input_ids, "input_ids", long, 8, 4, 2, (1, 4))                             \
  X(past_key, "past_key", float, 4, 32, 3, (1, 4, 8))                          \
  X(past_value, "past_value", float, 4, 32, 3, (1, 4, 8))
#define TRANSFORMER_TINY_OUTPUTS(X)                                            \
  X(logits, "logits", float, 4, 32, 3, (1, 4, 8))                              \
  X(present_key, "present_key", float, 4, 32, 3, (1, 4, 8))                    \
  X(present_value, "present_value", float, 4, 32, 3, (1, 4, 8))

/* The 22 decoder layers of TinyLlama-1.1B: L(layer, ...) for each. */
#define TINYLLAMA_LAYERS(L, ...)                                               \
  L(0, __VA_ARGS__)                                                            \
  L(1, __VA_ARGS__)                                                            \
  L(2, __VA_ARGS__)                                                            \
  L(3, __VA_ARGS__)                                                            \
  L(4, __VA_ARGS__)                                                            \
  L(5, __VA_ARGS__)                                                            \
  L(6, __VA_ARGS__)                                                            \
  L(7, __VA_ARGS__)                                                            \
  L(8, __VA_ARGS__)                                                            \
  L(9, __VA_ARGS__)                                                            \
  L(10, __VA_ARGS__)                                                           \
  L(11, __VA_ARGS__)                                                           \
  L(12, __VA_ARGS__)                                                           \
  L(13, __VA_ARGS__)                                                           \
  L(14, __VA_ARGS__)                                                           \
  L(15, __VA_ARGS__)                                                           \
  L(16, __VA_ARGS__)                                                           \
  L(17, __VA_ARGS__)                                                           \
  L(18, __VA_ARGS__)                                                           \
  L(19, __VA_ARGS__)                                                           \
  L(20, __VA_ARGS__)                                                           \
  L(21, __VA_ARGS__)

/* The past (input) and present (output) K and V of one layer over `rows`
 * positions: [1, 4, rows, 64] floats. */
#define TINYLLAMA_PAST_KV(layer, X, rows)                                      \
  X(past_key_values_##layer##_key, "past_key_values." #layer ".key", float, 4, \
    256 * (rows), 4, (1, 4, rows, 64))                                         \
  X(past_key_values_##layer##_value, "past_key_values." #layer ".value",       \
    float, 4, 256 * (rows), 4, (1, 4, rows, 64))
#define TINYLLAMA_PRESENT_KV(layer, X, rows)                                   \
  X(present_##layer##_key, "present." #layer ".key", float, 4, 256 * (rows),   \
    4, (1, 4, rows, 64))                                                       \
  X(present_##layer##_value, "present." #layer ".value", float, 4,             \
    256 * (rows), 4, (1, 4, rows, 64))

/* The real TinyLlama prefill validation graph: four positions on an empty
 * past. Its 44 past K/V inputs ([1, 4, 0, 64]) hold no element; they are
 * listed apart (TINYLLAMA_PREFILL_EMPTY_INPUTS) because host.c binds them to
 * one private buffer instead of public ones. */
#define TINYLLAMA_PREFILL_INPUTS(X)                                            \
  X(input_ids, "input_ids", long, 8, 4, 2, (1, 4))                             \
  X(attention_mask, "attention_mask", long, 8, 4, 2, (1, 4))                   \
  X(position_ids, "position_ids", long, 8, 4, 2, (1, 4))
#define TINYLLAMA_PREFILL_EMPTY_INPUTS(X)                                      \
  TINYLLAMA_LAYERS(TINYLLAMA_PAST_KV, X, 0)
#define TINYLLAMA_PREFILL_OUTPUTS(X)                                           \
  X(logits, "logits", float, 4, 4 * 32000, 3, (1, 4, 32000))                   \
  TINYLLAMA_LAYERS(TINYLLAMA_PRESENT_KV, X, 4)

/* The real TinyLlama M=1 step graph of a Context Capacity of `capacity`
 * tokens: one position on a past of capacity - 1 rows. */
#define TINYLLAMA_STEP_INPUTS(X, capacity)                                     \
  X(input_ids, "input_ids", long, 8, 1, 2, (1, 1))                             \
  X(attention_mask, "attention_mask", long, 8, capacity, 2, (1, capacity))     \
  X(position_ids, "position_ids", long, 8, 1, 2, (1, 1))                       \
  TINYLLAMA_LAYERS(TINYLLAMA_PAST_KV, X, (capacity) - 1)
#define TINYLLAMA_STEP_OUTPUTS(X, capacity)                                    \
  X(logits, "logits", float, 4, 32000, 3, (1, 1, 32000))                       \
  TINYLLAMA_LAYERS(TINYLLAMA_PRESENT_KV, X, capacity)

#endif
