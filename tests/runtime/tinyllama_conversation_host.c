#include <CL/cl.h>
#include <stdint.h>
#include <string.h>
#include <udoncl/array.h>
#include <udoncl/onnx_generated_binding.h>
#include <VRC/Runtime/OpenCL/Backend.h>

#define EXPECT(x)                                                              \
  do {                                                                         \
    if (!(x))                                                                  \
      return __LINE__;                                                         \
  } while (0)

extern udononnxStatus_t tinyllama_initialize(void);
extern udononnxStatus_t tinyllama_reset(void);
extern udononnxStatus_t tinyllama_set_prompt_tokens(udon_int32_array_t,
                                                    uint32_t);
extern udononnxStatus_t tinyllama_generate(uint32_t);
extern uint32_t tinyllama_generated_token_count(void);
extern udononnxStatus_t tinyllama_read_generated_tokens(udon_int32_array_t);
extern udononnxStatus_t tinyllama_dispose(void);
extern udononnxStatus_t
tinyllamaConversationGenerationProgress(const int32_t **tokens,
                                        uint32_t *token_count, uint32_t *serial,
                                        int32_t *is_completed);
extern udononnxStatus_t tinyllamaConversationPreparePrompt(int32_t **);
extern udononnxStatus_t tinyllamaConversationPrepareTurn(int32_t **,
                                                         uint32_t *);
extern udononnxStatus_t tinyllamaConversationCommitTurn(uint32_t);
extern udononnxStatus_t tinyllamaConversationPrompt(const int32_t **,
                                                    uint32_t *);

static cl_context binding_context;
static cl_command_queue binding_queue;
static cl_mem input[3], output;
static int64_t seen_tokens[128], seen_positions[128];
static uint32_t scheduled_tokens[32];
static char trace[512];
static int trace_size, mask_valid = 1;
/* The generated binding's sequence-state protocol: a run needs no pending
 * entry and leaves one, an advance needs one, and a reset clears it. */
static int pending, protocol_valid = 1;
static int runs, advances, resets, reduce_calls, finalize_calls;
static int fail_advance_call = -1, fail_finalize_call = -1;
/* Generation progress observed while each token is being selected. */
static uint32_t progress_counts[32], progress_serials[32];
static int32_t progress_last_token[32];
static int progress_valid = 1;

static void record(char operation) { trace[trace_size++] = operation; }

static void clear_trace(void) {
  runs = advances = reduce_calls = finalize_calls = trace_size = 0;
  mask_valid = 1;
  fail_advance_call = fail_finalize_call = -1;
}

size_t udoncl_cl_buffer_limit(void) { return 1024 * 1024; }
int udoncl_cl_gather_profile(const char *n,
                             struct udoncl_cl_gather_profile *p) {
  (void)n;
  (void)p;
  return 0;
}
int udoncl_cl_draw_gather(uint32_t k, const unsigned char *const *i, size_t n,
                          size_t b, const unsigned char *s, size_t z,
                          unsigned char *o) {
  (void)k;
  (void)i;
  (void)n;
  (void)b;
  (void)s;
  (void)z;
  (void)o;
  return 0;
}
int udoncl_cl_source_matches(const char *s, size_t n) {
  (void)s;
  (void)n;
  return 0;
}
int udoncl_cl_kernel_matches(const char *n) {
  (void)n;
  return 0;
}
int udoncl_cl_draw(uint32_t b, uint32_t g, const unsigned char s[4],
                   unsigned char r[4]) {
  (void)b;
  (void)g;
  (void)s;
  (void)r;
  return 0;
}

udononnxStatus_t udon_onnx_binding_create(cl_context c, cl_command_queue q) {
  cl_int e;
  binding_context = c;
  binding_queue = q;
  input[0] = clCreateBuffer(c, CL_MEM_READ_WRITE, 24, 0, &e);
  input[1] = clCreateBuffer(c, CL_MEM_READ_WRITE, 528, 0, &e);
  input[2] = clCreateBuffer(c, CL_MEM_READ_WRITE, 24, 0, &e);
  output = clCreateBuffer(c, CL_MEM_READ_WRITE, 131088, 0, &e);
  return input[0] && input[1] && input[2] && output
             ? UDONONNX_STATUS_SUCCESS
             : UDONONNX_STATUS_OPENCL_ERROR;
}
udononnxStatus_t udon_onnx_binding_enqueue_initialize(void) { return 0; }
udononnxStatus_t udon_onnx_binding_bind_all(void) { return 0; }
udononnxStatus_t udon_onnx_binding_resolve_input(uint32_t i, cl_mem *m,
                                                 size_t *o, size_t *n) {
  static const size_t lengths[] = {8, 512, 8};
  if (i >= 3)
    return UDONONNX_STATUS_INVALID_VALUE;
  *m = input[i];
  *o = 8;
  *n = lengths[i];
  return 0;
}
udononnxStatus_t udon_onnx_binding_resolve_output(uint32_t i, cl_mem *m,
                                                  size_t *o, size_t *n) {
  if (i)
    return UDONONNX_STATUS_INVALID_VALUE;
  *m = output;
  *o = 16;
  *n = 128000;
  return 0;
}
udononnxStatus_t udon_onnx_binding_enqueue_reset_sequence_state(void) {
  ++resets;
  pending = 0;
  return 0;
}
udononnxStatus_t udon_onnx_binding_enqueue_advance_sequence_state(void) {
  record('A');
  if (advances == fail_advance_call)
    return UDONONNX_STATUS_EXECUTION_FAILURE;
  if (pending != 1)
    protocol_valid = 0;
  pending = 0;
  ++advances;
  return 0;
}
udononnxStatus_t udon_onnx_binding_enqueue_run(void) {
  int64_t mask[64];
  int index;
  record('R');
  if (pending != 0)
    protocol_valid = 0;
  pending = 1;
  clEnqueueReadBuffer(binding_queue, input[0], CL_TRUE, 8, 8,
                      &seen_tokens[runs], 0, 0, 0);
  clEnqueueReadBuffer(binding_queue, input[2], CL_TRUE, 8, 8,
                      &seen_positions[runs], 0, 0, 0);
  clEnqueueReadBuffer(binding_queue, input[1], CL_TRUE, 8, sizeof(mask), mask,
                      0, 0, 0);
  for (index = 0; index < 64; ++index) {
    int64_t expected = index + seen_positions[runs] >= 63 ? 1 : 0;
    if (mask[index] != expected)
      mask_valid = 0;
  }
  ++runs;
  return 0;
}
udononnxStatus_t udon_onnx_binding_dispose(void) {
  int i;
  for (i = 0; i < 3; ++i)
    if (input[i])
      clReleaseMemObject(input[i]);
  if (output)
    clReleaseMemObject(output);
  return 0;
}
cl_int __vrc_enqueue_udonaten_argmax_reduce_device_args(
    cl_command_queue q, cl_mem a, cl_mem p, cl_int o, cl_int w, cl_uint n,
    const cl_event *l, cl_event *e) {
  (void)q;
  (void)p;
  (void)n;
  (void)l;
  (void)e;
  record('D');
  ++reduce_calls;
  return a == output && o == 4 && w == 32000 ? CL_SUCCESS : CL_INVALID_VALUE;
}
cl_int __vrc_enqueue_udonaten_argmax_finalize_device_args(
    cl_command_queue q, cl_mem a, cl_mem p, cl_mem r, cl_int o, cl_int w,
    cl_int reject, cl_uint n, const cl_event *l, cl_event *e) {
  uint32_t *token = &scheduled_tokens[finalize_calls];
  (void)a;
  (void)p;
  (void)o;
  (void)w;
  (void)n;
  (void)l;
  (void)e;
  record('F');
  {
    const int32_t *tokens = 0;
    uint32_t count = 0, serial = 0;
    int32_t is_completed = 1;
    if (tinyllamaConversationGenerationProgress(&tokens, &count, &serial,
                                                &is_completed) !=
            UDONONNX_STATUS_SUCCESS ||
        is_completed)
      progress_valid = 0;
    progress_counts[finalize_calls] = count;
    progress_serials[finalize_calls] = serial;
    progress_last_token[finalize_calls] = count ? tokens[count - 1] : -1;
  }
  if (finalize_calls == fail_finalize_call)
    return CL_INVALID_OPERATION;
  ++finalize_calls;
  return reject == 1
             ? clEnqueueWriteBuffer(q, r, CL_FALSE, 0, 4, token, 0, 0, 0)
             : CL_INVALID_VALUE;
}

/* Writes a turn of `count` tokens 100, 101, ... into the prepared workspace
 * and commits it. */
static int commit_turn(uint32_t count, uint32_t expected_closing,
                       uint32_t expected_capacity) {
  int32_t *workspace = 0;
  const int32_t *tokens = 0;
  uint32_t capacity = 0, index, token_count = 9;
  EXPECT(tinyllamaConversationPrepareTurn(&workspace, &capacity) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllamaConversationPrompt(&tokens, &token_count) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(token_count == 0 && workspace == tokens + expected_closing);
  EXPECT(capacity == expected_capacity);
  for (index = 0; index < count; ++index)
    workspace[index] = (int32_t)(100 + index);
  EXPECT(tinyllamaConversationCommitTurn(count) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllamaConversationPrompt(&tokens, &token_count) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(token_count == expected_closing + count);
  return 0;
}

/* Multi-turn: every turn continues the sequence state at the position where
 * the previous reply ended, closing a reply cut off by the token limit. */
static int test_turns(void) {
  int32_t prompt_data[] = {1, 450, 4996, 17354}, result_data[32];
  udon_int32_array_native_t prompt = {prompt_data, 4},
                            result = {result_data, 32};
  int32_t *workspace = 0;
  const int32_t *tokens = 0;
  uint32_t capacity = 0, count = 0, serial = 0, previous_serial;
  int32_t is_completed = 0;
  int i, failure;

  EXPECT(tinyllama_reset() == UDONONNX_STATUS_SUCCESS);
  protocol_valid = 1;
  /* No reply to continue yet. */
  EXPECT(tinyllamaConversationPrepareTurn(&workspace, &capacity) ==
         UDONONNX_STATUS_INVALID_STATE);
  EXPECT(tinyllamaConversationCommitTurn(1) == UDONONNX_STATUS_INVALID_STATE);
  clear_trace();
  scheduled_tokens[0] = 1701;
  scheduled_tokens[1] = 2;
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generate(5) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generated_token_count() == 2);
  /* Positions 0..5 hold the prompt, 1701 and the EOS that ended the reply. */
  EXPECT(seen_tokens[5] == 2 && seen_positions[5] == 5);
  EXPECT(tinyllama_generate(1) == UDONONNX_STATUS_INVALID_STATE);
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) ==
         UDONONNX_STATUS_INVALID_STATE);
  EXPECT(tinyllamaConversationPreparePrompt(&workspace) ==
         UDONONNX_STATUS_INVALID_STATE);
  EXPECT(tinyllamaConversationGenerationProgress(
             &tokens, &count, &previous_serial, &is_completed) ==
         UDONONNX_STATUS_SUCCESS);

  /* An EOS-ended reply already ran its EOS: nothing to close. A committed
   * turn may be replaced before it is generated. */
  if ((failure = commit_turn(5, 0, 58)) != 0)
    return failure;
  if ((failure = commit_turn(3, 0, 58)) != 0)
    return failure;
  EXPECT(tinyllama_generated_token_count() == 2);
  clear_trace();
  scheduled_tokens[0] = 555;
  scheduled_tokens[1] = 2;
  EXPECT(tinyllama_generate(4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(trace_size == 14 &&
         !memcmp(trace, "ARARARDFARDFAR", (size_t)trace_size));
  for (i = 0; i < 3; ++i)
    EXPECT(seen_tokens[i] == 100 + i && seen_positions[i] == 6 + i);
  EXPECT(seen_tokens[3] == 555 && seen_positions[3] == 9);
  EXPECT(seen_tokens[4] == 2 && seen_positions[4] == 10);
  EXPECT(mask_valid && protocol_valid);
  /* Each turn is its own generation with its own tokens. */
  EXPECT(tinyllama_generated_token_count() == 2);
  EXPECT(tinyllama_read_generated_tokens(&result) == UDONONNX_STATUS_SUCCESS);
  EXPECT(result_data[0] == 555 && result_data[1] == 2);
  EXPECT(tinyllamaConversationGenerationProgress(&tokens, &count, &serial,
                                                 &is_completed) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(count == 2 && is_completed && serial != previous_serial);

  /* A reply that reaches the token limit never ran its last token. */
  if ((failure = commit_turn(1, 0, 53)) != 0)
    return failure;
  clear_trace();
  scheduled_tokens[0] = 777;
  scheduled_tokens[1] = 778;
  EXPECT(tinyllama_generate(2) == UDONONNX_STATUS_SUCCESS);
  EXPECT(trace_size == 8 && !memcmp(trace, "ARDFARDF", (size_t)trace_size));
  EXPECT(seen_tokens[0] == 100 && seen_positions[0] == 11);
  EXPECT(seen_tokens[1] == 777 && seen_positions[1] == 12);
  /* The next turn first runs 778 at 13, then the EOS the reply lacks. */
  if ((failure = commit_turn(1, 2, 49)) != 0)
    return failure;
  EXPECT(tinyllamaConversationPrompt(&tokens, &count) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(count == 3 && tokens[0] == 778 && tokens[1] == 2 && tokens[2] == 100);
  clear_trace();
  scheduled_tokens[0] = 2;
  EXPECT(tinyllama_generate(1) == UDONONNX_STATUS_SUCCESS);
  EXPECT(trace_size == 8 && !memcmp(trace, "ARARARDF", (size_t)trace_size));
  EXPECT(seen_tokens[0] == 778 && seen_positions[0] == 13);
  EXPECT(seen_tokens[1] == 2 && seen_positions[1] == 14);
  EXPECT(seen_tokens[2] == 100 && seen_positions[2] == 15);
  EXPECT(mask_valid && protocol_valid);
  /* An EOS selected as the last requested token is run, but not doubled. */
  if ((failure = commit_turn(2, 1, 47)) != 0)
    return failure;
  EXPECT(tinyllamaConversationPrompt(&tokens, &count) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(count == 3 && tokens[0] == 2 && tokens[1] == 100);

  /* Overflow leaves the conversation unchanged: a turn that does not fit is
   * rejected, and so is a generation whose token budget does not fit. */
  EXPECT(tinyllamaConversationPrepareTurn(&workspace, &capacity) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(capacity == 47);
  EXPECT(tinyllamaConversationCommitTurn(48) ==
         UDONONNX_STATUS_CAPACITY_EXCEEDED);
  EXPECT(tinyllama_generate(1) == UDONONNX_STATUS_INVALID_STATE);
  if ((failure = commit_turn(47, 1, 47)) != 0)
    return failure;
  clear_trace();
  EXPECT(tinyllama_generate(2) == UDONONNX_STATUS_CAPACITY_EXCEEDED);
  EXPECT(trace_size == 0);
  scheduled_tokens[0] = 2;
  EXPECT(tinyllama_generate(1) == UDONONNX_STATUS_SUCCESS);
  /* The last prompt step runs at the final position of the capacity. */
  EXPECT(runs == 48 && seen_positions[0] == 16 && seen_positions[47] == 63);
  EXPECT(mask_valid && protocol_valid);
  EXPECT(tinyllamaConversationPrepareTurn(&workspace, &capacity) ==
         UDONONNX_STATUS_CAPACITY_EXCEEDED);

  /* Reset starts a new conversation at position 0. */
  EXPECT(tinyllama_reset() == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllamaConversationPrepareTurn(&workspace, &capacity) ==
         UDONONNX_STATUS_INVALID_STATE);
  clear_trace();
  scheduled_tokens[0] = 1701;
  scheduled_tokens[1] = 1702;
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generate(2) == UDONONNX_STATUS_SUCCESS);
  EXPECT(seen_positions[0] == 0 && trace[0] == 'R');
  EXPECT(mask_valid && protocol_valid);

  /* A failed turn needs Reset like any failed generation. */
  if ((failure = commit_turn(2, 2, 57)) != 0)
    return failure;
  clear_trace();
  fail_finalize_call = 0;
  EXPECT(tinyllama_generate(1) == UDONONNX_STATUS_OPENCL_ERROR);
  EXPECT(tinyllamaConversationPrepareTurn(&workspace, &capacity) ==
         UDONONNX_STATUS_INVALID_STATE);
  EXPECT(tinyllama_reset() == UDONONNX_STATUS_SUCCESS);
  return 0;
}

int main(void) {
  static const uint32_t expected_tokens[] = {1701, 29916, 432, 17204};
  int32_t prompt_data[] = {1, 450, 4996, 17354};
  int32_t long_prompt_data[64], result_data[32] = {0};
  udon_int32_array_native_t prompt = {prompt_data, 4},
                            result = {result_data, 32},
                            long_prompt = {long_prompt_data, 64};
  const int32_t *progress_tokens = 0;
  uint32_t progress_count = 9, serial = 0, previous_serial;
  int32_t is_completed = 1;
  int i;
  for (i = 0; i < 64; ++i)
    long_prompt_data[i] = i;
  EXPECT(tinyllamaConversationGenerationProgress(
             &progress_tokens, &progress_count, &serial, &is_completed) ==
         UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(tinyllama_initialize() == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllamaConversationGenerationProgress(
             &progress_tokens, &progress_count, &serial, &is_completed) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(progress_count == 0 && !is_completed);
  previous_serial = serial;

  clear_trace();
  for (i = 0; i < 4; ++i)
    scheduled_tokens[i] = expected_tokens[i];
  prompt_data[0] = -1;
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) ==
         UDONONNX_STATUS_INVALID_VALUE);
  prompt_data[0] = 32000;
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) ==
         UDONONNX_STATUS_INVALID_VALUE);
  prompt_data[0] = 1;
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generate(4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(runs == 7 && advances == 6 && reduce_calls == 4 &&
         finalize_calls == 4 && mask_valid);
  EXPECT(trace_size == 21 &&
         !memcmp(trace, "RARARARDFARDFARDFARDF", (size_t)trace_size));
  for (i = 0; i < 4; ++i)
    EXPECT(seen_tokens[i] == prompt_data[i] && seen_positions[i] == i);
  for (i = 0; i < 3; ++i)
    EXPECT(seen_tokens[i + 4] == (int64_t)expected_tokens[i] &&
           seen_positions[i + 4] == i + 4);
  EXPECT(tinyllama_generated_token_count() == 4);
  EXPECT(tinyllama_read_generated_tokens(&result) == UDONONNX_STATUS_SUCCESS);
  for (i = 0; i < 4; ++i)
    EXPECT(result_data[i] == (int32_t)expected_tokens[i]);
  EXPECT(tinyllama_generate(1) == UDONONNX_STATUS_INVALID_STATE);
  /* While token i was being selected, exactly the i earlier tokens of this
   * generation were visible, and the generation was not completed. */
  EXPECT(progress_valid);
  for (i = 0; i < 4; ++i) {
    EXPECT(progress_counts[i] == (uint32_t)i);
    EXPECT(progress_serials[i] == progress_serials[0]);
    EXPECT(progress_last_token[i] ==
           (i ? (int32_t)expected_tokens[i - 1] : -1));
  }
  EXPECT(progress_serials[0] != previous_serial);
  EXPECT(tinyllamaConversationGenerationProgress(
             &progress_tokens, &progress_count, &serial, &is_completed) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(progress_count == 4 && is_completed && serial == progress_serials[0]);
  for (i = 0; i < 4; ++i)
    EXPECT(progress_tokens[i] == (int32_t)expected_tokens[i]);
  previous_serial = serial;

  EXPECT(tinyllama_reset() == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllamaConversationGenerationProgress(
             &progress_tokens, &progress_count, &serial, &is_completed) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(progress_count == 0 && !is_completed && serial != previous_serial);
  previous_serial = serial;
  clear_trace();
  scheduled_tokens[0] = 1701;
  scheduled_tokens[1] = 2;
  scheduled_tokens[2] = 999;
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generate(3) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generated_token_count() == 2);
  EXPECT(tinyllama_read_generated_tokens(&result) == UDONONNX_STATUS_SUCCESS);
  EXPECT(result_data[0] == 1701 && result_data[1] == 2);
  /* Each decode step is enqueued from the Device-resident token before the
   * Host reads it, so the step fed by EOS has already been issued; it is
   * discarded. */
  EXPECT(runs == 6 && advances == 5 && finalize_calls == 2 && mask_valid);
  EXPECT(trace_size == 15 &&
         !memcmp(trace, "RARARARDFARDFAR", (size_t)trace_size));
  EXPECT(seen_tokens[4] == 1701 && seen_tokens[5] == 2 &&
         seen_positions[5] == 5);

  EXPECT(tinyllama_reset() == UDONONNX_STATUS_SUCCESS);
  clear_trace();
  scheduled_tokens[0] = 2;
  EXPECT(tinyllama_set_prompt_tokens(&long_prompt, 64) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generate(2) == UDONONNX_STATUS_CAPACITY_EXCEEDED);
  EXPECT(tinyllama_generate(33) == UDONONNX_STATUS_CAPACITY_EXCEEDED);
  EXPECT(tinyllama_generate(1) == UDONONNX_STATUS_SUCCESS);
  EXPECT(runs == 64 && advances == 63 && mask_valid);

  EXPECT(tinyllama_reset() == UDONONNX_STATUS_SUCCESS);
  long_prompt.length = 33;
  clear_trace();
  scheduled_tokens[0] = 2;
  EXPECT(tinyllama_set_prompt_tokens(&long_prompt, 33) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generate(32) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_reset() == UDONONNX_STATUS_SUCCESS);
  long_prompt.length = 34;
  EXPECT(tinyllama_set_prompt_tokens(&long_prompt, 34) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generate(32) == UDONONNX_STATUS_CAPACITY_EXCEEDED);

  EXPECT(tinyllama_reset() == UDONONNX_STATUS_SUCCESS);
  clear_trace();
  scheduled_tokens[0] = 1701;
  fail_finalize_call = 0;
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generate(2) == UDONONNX_STATUS_OPENCL_ERROR);
  EXPECT(tinyllama_generated_token_count() == 0);
  EXPECT(tinyllamaConversationGenerationProgress(
             &progress_tokens, &progress_count, &serial, &is_completed) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(progress_count == 0 && !is_completed);
  EXPECT(tinyllama_read_generated_tokens(&result) ==
         UDONONNX_STATUS_INVALID_STATE);
  EXPECT(tinyllama_generate(1) == UDONONNX_STATUS_INVALID_STATE);
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) ==
         UDONONNX_STATUS_INVALID_STATE);
  EXPECT(tinyllama_reset() == UDONONNX_STATUS_SUCCESS);

  clear_trace();
  scheduled_tokens[0] = 1701;
  fail_advance_call = 3;
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generate(2) == UDONONNX_STATUS_EXECUTION_FAILURE);
  EXPECT(tinyllama_generated_token_count() == 0);
  EXPECT(tinyllama_generate(1) == UDONONNX_STATUS_INVALID_STATE);
  EXPECT(tinyllama_reset() == UDONONNX_STATUS_SUCCESS);

  clear_trace();
  scheduled_tokens[0] = 2;
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generate(1) == UDONONNX_STATUS_SUCCESS);
  EXPECT(protocol_valid);

  i = test_turns();
  if (i)
    return i;

  EXPECT(tinyllama_dispose() == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllamaConversationGenerationProgress(
             &progress_tokens, &progress_count, &serial, &is_completed) ==
         UDONONNX_STATUS_DISPOSED);
  EXPECT(tinyllama_initialize() == UDONONNX_STATUS_DISPOSED);
  return 0;
}
