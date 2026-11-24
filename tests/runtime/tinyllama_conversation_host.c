#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <udoncl/array.h>
#include <udoncl/onnx_types.h>
#include <udononnxruntime.h>
#include <VRC/Runtime/OpenCL/Backend.h>

#define EXPECT(x)                                                              \
  do {                                                                         \
    if (!(x))                                                                  \
      return __LINE__;                                                         \
  } while (0)

extern udononnxStatus_t initialize(void);
extern udononnxStatus_t reset(void);
extern udononnxStatus_t set_prompt_tokens(udon_int32_array_t, uint32_t);
extern udononnxStatus_t generate(uint32_t);
extern uint32_t generated_token_count(void);
extern udononnxStatus_t read_generated_tokens(udon_int32_array_t);
extern udononnxStatus_t shutdown(void);
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

/* The default Context Capacity of conversation_host.h. */
#define CAPACITY 64
#define LAYERS 22

/* A fake UdonONNX session of the TinyLlama graph. It keeps the binding of
 * every graph input and output by name, as the session does, and every run
 * checks that the Host bound the tensors of one position on a past of
 * `position` rows: the token and the position (1 x 1), the attention mask of
 * position + 1 ones in its buffer of the capacity, and each layer's past of
 * `position` rows and present of position + 1 rows in one buffer of the
 * capacity, the past the leading block of the present. */
struct udononnxSession {
  int live;
};
struct binding {
  char name[32];
  cl_mem memory;
  int rank;
  int64_t dims[4], buffer_dims[4];
  int has_buffer_dims;
};
static struct udononnxSession fake_session;
static struct binding bindings[96];
static int binding_count, sessions;
static cl_command_queue session_queue;
static int64_t seen_tokens[128], seen_positions[128], seen_extents[128];
static uint32_t scheduled_tokens[32];
static char trace[512];
static int trace_size, binding_valid = 1;
static int runs, binds, reduce_calls, finalize_calls;
static int fail_run_call = -1, fail_finalize_call = -1;
/* The bound logits buffer, which ArgMax must read. */
static cl_mem bound_logits;
/* Generation progress observed while each token is being selected. */
static uint32_t progress_counts[32], progress_serials[32];
static int32_t progress_last_token[32];
static int progress_valid = 1;

static void record(char operation) { trace[trace_size++] = operation; }

static void clear_trace(void) {
  runs = binds = reduce_calls = finalize_calls = trace_size = 0;
  binding_valid = 1;
  fail_run_call = fail_finalize_call = -1;
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

udononnxStatus_t udononnxCreateSession(cl_context context,
                                       cl_command_queue queue,
                                       udononnxSession_t **session) {
  (void)context;
  if (fake_session.live)
    return UDONONNX_STATUS_INVALID_STATE;
  fake_session.live = 1;
  ++sessions;
  session_queue = queue;
  binding_count = 0;
  *session = &fake_session;
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t bind(udononnxSession_t *session, const char *name,
                             cl_mem memory, int rank, const int64_t *dims,
                             const int64_t *buffer_dims) {
  struct binding *entry = 0;
  int index;
  if (session != &fake_session || !fake_session.live || !name || !memory ||
      rank < 1 || rank > 4 || !dims || strlen(name) >= sizeof(entry->name))
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < binding_count; ++index)
    if (!strcmp(bindings[index].name, name))
      entry = &bindings[index];
  if (!entry) {
    entry = &bindings[binding_count++];
    strcpy(entry->name, name);
  }
  entry->memory = memory;
  entry->rank = rank;
  memcpy(entry->dims, dims, sizeof(int64_t) * (size_t)rank);
  entry->has_buffer_dims = buffer_dims != 0;
  if (buffer_dims)
    memcpy(entry->buffer_dims, buffer_dims, sizeof(int64_t) * (size_t)rank);
  ++binds;
  return UDONONNX_STATUS_SUCCESS;
}
udononnxStatus_t udononnxBindInput(udononnxSession_t *session, const char *name,
                                   cl_mem memory, int rank, const int64_t *dims,
                                   const int64_t *buffer_dims) {
  return bind(session, name, memory, rank, dims, buffer_dims);
}
udononnxStatus_t udononnxBindOutput(udononnxSession_t *session,
                                    const char *name, cl_mem memory, int rank,
                                    const int64_t *dims,
                                    const int64_t *buffer_dims) {
  return bind(session, name, memory, rank, dims, buffer_dims);
}

static const struct binding *bound(const char *name) {
  int index;
  for (index = 0; index < binding_count; ++index)
    if (!strcmp(bindings[index].name, name))
      return &bindings[index];
  return 0;
}

/* Whether `name` is bound with `rank` and the dimensions `dims`, and in a
 * buffer laid out as `layout` (0: as dims). */
static int bound_as(const char *name, int rank, const int64_t *dims,
                    const int64_t *layout) {
  const struct binding *entry = bound(name);
  if (!entry || entry->rank != rank ||
      memcmp(entry->dims, dims, sizeof(int64_t) * (size_t)rank))
    return 0;
  if (!layout)
    return !entry->has_buffer_dims ||
           !memcmp(entry->buffer_dims, dims, sizeof(int64_t) * (size_t)rank);
  return entry->has_buffer_dims &&
         !memcmp(entry->buffer_dims, layout, sizeof(int64_t) * (size_t)rank);
}

static int check_run_bindings(int64_t *position) {
  static const int64_t token[2] = {1, 1}, logits[3] = {1, 1, 32000},
                       mask_layout[2] = {1, CAPACITY},
                       cache_layout[4] = {1, 4, CAPACITY, 64};
  const struct binding *mask = bound("attention_mask");
  int64_t mask_dims[2], past[4], present[4], ones[CAPACITY];
  char name[32];
  int layer, kind, index;
  if (!mask || mask->rank != 2 || mask->dims[1] < 1 || mask->dims[1] > CAPACITY)
    return 0;
  *position = mask->dims[1] - 1;
  mask_dims[0] = 1;
  mask_dims[1] = *position + 1;
  past[0] = present[0] = 1;
  past[1] = present[1] = 4;
  past[2] = *position;
  present[2] = *position + 1;
  past[3] = present[3] = 64;
  if (!bound_as("input_ids", 2, token, 0) ||
      !bound_as("position_ids", 2, token, 0) ||
      !bound_as("logits", 3, logits, 0) ||
      !bound_as("attention_mask", 2, mask_dims, mask_layout))
    return 0;
  bound_logits = bound("logits")->memory;
  for (layer = 0; layer < LAYERS; ++layer)
    for (kind = 0; kind < 2; ++kind) {
      const struct binding *input, *output;
      snprintf(name, sizeof(name), "past_key_values.%d.%s", layer,
               kind ? "value" : "key");
      if (!bound_as(name, 4, past, cache_layout))
        return 0;
      input = bound(name);
      snprintf(name, sizeof(name), "present.%d.%s", layer,
               kind ? "value" : "key");
      if (!bound_as(name, 4, present, cache_layout))
        return 0;
      output = bound(name);
      /* The past is the leading block of the present's buffer. */
      if (input->memory != output->memory)
        return 0;
    }
  if (binding_count != 3 + 1 + 2 * 2 * LAYERS)
    return 0;
  /* The mask buffer holds a one for every position of the run. */
  if (clEnqueueReadBuffer(session_queue, mask->memory, CL_TRUE, 0,
                          sizeof(int64_t) * (size_t)(*position + 1), ones, 0, 0,
                          0) != CL_SUCCESS)
    return 0;
  for (index = 0; index <= *position; ++index)
    if (ones[index] != 1)
      return 0;
  return 1;
}

udononnxStatus_t udononnxRun(udononnxSession_t *session) {
  int64_t position = -1;
  record('R');
  if (session != &fake_session || !fake_session.live)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (runs == fail_run_call)
    return UDONONNX_STATUS_EXECUTION_FAILURE;
  if (!check_run_bindings(&position))
    binding_valid = 0;
  clEnqueueReadBuffer(session_queue, bound("input_ids")->memory, CL_TRUE, 0, 8,
                      &seen_tokens[runs], 0, 0, 0);
  clEnqueueReadBuffer(session_queue, bound("position_ids")->memory, CL_TRUE, 0,
                      8, &seen_positions[runs], 0, 0, 0);
  /* A run at position p binds a past of p rows. */
  seen_extents[runs] = position;
  if (seen_positions[runs] != position)
    binding_valid = 0;
  ++runs;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxReleaseSession(udononnxSession_t *session) {
  if (session != &fake_session || !fake_session.live)
    return UDONONNX_STATUS_INVALID_VALUE;
  fake_session.live = 0;
  binding_count = 0;
  return UDONONNX_STATUS_SUCCESS;
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
  /* The logits of the one position of the last run. */
  return a == bound_logits && o == 0 && w == 32000 ? CL_SUCCESS
                                                   : CL_INVALID_VALUE;
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

/* Every run of the last trace ran at the past extent of its position, from
 * `first_position` on. */
static int runs_at(int64_t first_position) {
  int i;
  for (i = 0; i < runs; ++i)
    EXPECT(seen_extents[i] == first_position + i);
  return 0;
}

/* Multi-turn: every turn continues the conversation at the position where
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

  EXPECT(reset() == UDONONNX_STATUS_SUCCESS);
  /* No reply to continue yet. */
  EXPECT(tinyllamaConversationPrepareTurn(&workspace, &capacity) ==
         UDONONNX_STATUS_INVALID_STATE);
  EXPECT(tinyllamaConversationCommitTurn(1) == UDONONNX_STATUS_INVALID_STATE);
  clear_trace();
  scheduled_tokens[0] = 1701;
  scheduled_tokens[1] = 2;
  EXPECT(set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generate(5) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generated_token_count() == 2);
  /* Positions 0..5 hold the prompt, 1701 and the EOS that ended the reply. */
  EXPECT(seen_tokens[5] == 2 && seen_positions[5] == 5);
  if ((failure = runs_at(0)) != 0)
    return failure;
  EXPECT(generate(1) == UDONONNX_STATUS_INVALID_STATE);
  EXPECT(set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_INVALID_STATE);
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
  EXPECT(generated_token_count() == 2);
  clear_trace();
  scheduled_tokens[0] = 555;
  scheduled_tokens[1] = 2;
  EXPECT(generate(4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(trace_size == 9 && !memcmp(trace, "RRRDFRDFR", (size_t)trace_size));
  for (i = 0; i < 3; ++i)
    EXPECT(seen_tokens[i] == 100 + i && seen_positions[i] == 6 + i);
  EXPECT(seen_tokens[3] == 555 && seen_positions[3] == 9);
  EXPECT(seen_tokens[4] == 2 && seen_positions[4] == 10);
  /* The second turn continues the cache at a past of six rows. */
  if ((failure = runs_at(6)) != 0)
    return failure;
  EXPECT(binding_valid);
  /* Each turn is its own generation with its own tokens. */
  EXPECT(generated_token_count() == 2);
  EXPECT(read_generated_tokens(&result) == UDONONNX_STATUS_SUCCESS);
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
  EXPECT(generate(2) == UDONONNX_STATUS_SUCCESS);
  EXPECT(trace_size == 6 && !memcmp(trace, "RDFRDF", (size_t)trace_size));
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
  EXPECT(generate(1) == UDONONNX_STATUS_SUCCESS);
  EXPECT(trace_size == 5 && !memcmp(trace, "RRRDF", (size_t)trace_size));
  EXPECT(seen_tokens[0] == 778 && seen_positions[0] == 13);
  EXPECT(seen_tokens[1] == 2 && seen_positions[1] == 14);
  EXPECT(seen_tokens[2] == 100 && seen_positions[2] == 15);
  if ((failure = runs_at(13)) != 0)
    return failure;
  EXPECT(binding_valid);
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
  EXPECT(generate(1) == UDONONNX_STATUS_INVALID_STATE);
  if ((failure = commit_turn(47, 1, 47)) != 0)
    return failure;
  clear_trace();
  EXPECT(generate(2) == UDONONNX_STATUS_CAPACITY_EXCEEDED);
  EXPECT(trace_size == 0);
  scheduled_tokens[0] = 2;
  EXPECT(generate(1) == UDONONNX_STATUS_SUCCESS);
  /* The last prompt step runs at the final position of the capacity: a past
   * of capacity - 1 rows and a present that fills the cache. */
  EXPECT(runs == 48 && seen_positions[0] == 16 && seen_positions[47] == 63);
  if ((failure = runs_at(16)) != 0)
    return failure;
  EXPECT(binding_valid);
  EXPECT(tinyllamaConversationPrepareTurn(&workspace, &capacity) ==
         UDONONNX_STATUS_CAPACITY_EXCEEDED);

  /* Reset starts a new conversation at position 0, a past of no rows. */
  EXPECT(reset() == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllamaConversationPrepareTurn(&workspace, &capacity) ==
         UDONONNX_STATUS_INVALID_STATE);
  clear_trace();
  scheduled_tokens[0] = 1701;
  scheduled_tokens[1] = 1702;
  EXPECT(set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generate(2) == UDONONNX_STATUS_SUCCESS);
  EXPECT(seen_positions[0] == 0 && trace[0] == 'R');
  if ((failure = runs_at(0)) != 0)
    return failure;
  EXPECT(binding_valid);

  /* A failed turn needs Reset like any failed generation. */
  if ((failure = commit_turn(2, 2, 57)) != 0)
    return failure;
  clear_trace();
  fail_finalize_call = 0;
  EXPECT(generate(1) == UDONONNX_STATUS_OPENCL_ERROR);
  EXPECT(tinyllamaConversationPrepareTurn(&workspace, &capacity) ==
         UDONONNX_STATUS_INVALID_STATE);
  EXPECT(reset() == UDONONNX_STATUS_SUCCESS);
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
  int i, failure;
  for (i = 0; i < 64; ++i)
    long_prompt_data[i] = i;
  EXPECT(tinyllamaConversationGenerationProgress(
             &progress_tokens, &progress_count, &serial, &is_completed) ==
         UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(initialize() == UDONONNX_STATUS_SUCCESS);
  EXPECT(sessions == 1 && fake_session.live);
  EXPECT(tinyllamaConversationGenerationProgress(
             &progress_tokens, &progress_count, &serial, &is_completed) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(progress_count == 0 && !is_completed);
  previous_serial = serial;

  clear_trace();
  for (i = 0; i < 4; ++i)
    scheduled_tokens[i] = expected_tokens[i];
  prompt_data[0] = -1;
  EXPECT(set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_INVALID_VALUE);
  prompt_data[0] = 32000;
  EXPECT(set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_INVALID_VALUE);
  prompt_data[0] = 1;
  EXPECT(set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generate(4) == UDONONNX_STATUS_SUCCESS);
  /* One run per position, each binding the mask and the 44 past and present
   * K/V again: 89 binds. */
  EXPECT(runs == 7 && binds == 7 * 89 && reduce_calls == 4 &&
         finalize_calls == 4 && binding_valid);
  EXPECT(trace_size == 15 &&
         !memcmp(trace, "RRRRDFRDFRDFRDF", (size_t)trace_size));
  if ((failure = runs_at(0)) != 0)
    return failure;
  for (i = 0; i < 4; ++i)
    EXPECT(seen_tokens[i] == prompt_data[i] && seen_positions[i] == i);
  for (i = 0; i < 3; ++i)
    EXPECT(seen_tokens[i + 4] == (int64_t)expected_tokens[i] &&
           seen_positions[i + 4] == i + 4);
  EXPECT(generated_token_count() == 4);
  EXPECT(read_generated_tokens(&result) == UDONONNX_STATUS_SUCCESS);
  for (i = 0; i < 4; ++i)
    EXPECT(result_data[i] == (int32_t)expected_tokens[i]);
  EXPECT(generate(1) == UDONONNX_STATUS_INVALID_STATE);
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

  /* Generate, reset, generate: the second conversation starts again at a
   * past of no rows, without clearing anything. */
  EXPECT(reset() == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllamaConversationGenerationProgress(
             &progress_tokens, &progress_count, &serial, &is_completed) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(progress_count == 0 && !is_completed && serial != previous_serial);
  previous_serial = serial;
  clear_trace();
  scheduled_tokens[0] = 1701;
  scheduled_tokens[1] = 2;
  scheduled_tokens[2] = 999;
  EXPECT(set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generate(3) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generated_token_count() == 2);
  EXPECT(read_generated_tokens(&result) == UDONONNX_STATUS_SUCCESS);
  EXPECT(result_data[0] == 1701 && result_data[1] == 2);
  /* Each decode step is enqueued from the Device-resident token before the
   * Host reads it, so the step fed by EOS has already been issued; it is
   * discarded. */
  EXPECT(runs == 6 && finalize_calls == 2 && binding_valid);
  EXPECT(trace_size == 10 && !memcmp(trace, "RRRRDFRDFR", (size_t)trace_size));
  EXPECT(seen_tokens[4] == 1701 && seen_tokens[5] == 2 &&
         seen_positions[5] == 5);
  if ((failure = runs_at(0)) != 0)
    return failure;

  EXPECT(reset() == UDONONNX_STATUS_SUCCESS);
  clear_trace();
  scheduled_tokens[0] = 2;
  EXPECT(set_prompt_tokens(&long_prompt, 64) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generate(2) == UDONONNX_STATUS_CAPACITY_EXCEEDED);
  EXPECT(generate(33) == UDONONNX_STATUS_CAPACITY_EXCEEDED);
  EXPECT(generate(1) == UDONONNX_STATUS_SUCCESS);
  EXPECT(runs == 64 && binding_valid);
  if ((failure = runs_at(0)) != 0)
    return failure;

  EXPECT(reset() == UDONONNX_STATUS_SUCCESS);
  long_prompt.length = 33;
  clear_trace();
  scheduled_tokens[0] = 2;
  EXPECT(set_prompt_tokens(&long_prompt, 33) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generate(32) == UDONONNX_STATUS_SUCCESS);
  EXPECT(reset() == UDONONNX_STATUS_SUCCESS);
  long_prompt.length = 34;
  EXPECT(set_prompt_tokens(&long_prompt, 34) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generate(32) == UDONONNX_STATUS_CAPACITY_EXCEEDED);

  EXPECT(reset() == UDONONNX_STATUS_SUCCESS);
  clear_trace();
  scheduled_tokens[0] = 1701;
  fail_finalize_call = 0;
  EXPECT(set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generate(2) == UDONONNX_STATUS_OPENCL_ERROR);
  EXPECT(generated_token_count() == 0);
  EXPECT(tinyllamaConversationGenerationProgress(
             &progress_tokens, &progress_count, &serial, &is_completed) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(progress_count == 0 && !is_completed);
  EXPECT(read_generated_tokens(&result) == UDONONNX_STATUS_INVALID_STATE);
  EXPECT(generate(1) == UDONONNX_STATUS_INVALID_STATE);
  EXPECT(set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_INVALID_STATE);
  EXPECT(reset() == UDONONNX_STATUS_SUCCESS);

  /* A failed run fails the generation. */
  clear_trace();
  scheduled_tokens[0] = 1701;
  fail_run_call = 3;
  EXPECT(set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generate(2) == UDONONNX_STATUS_EXECUTION_FAILURE);
  EXPECT(generated_token_count() == 0);
  EXPECT(generate(1) == UDONONNX_STATUS_INVALID_STATE);
  EXPECT(reset() == UDONONNX_STATUS_SUCCESS);

  clear_trace();
  scheduled_tokens[0] = 2;
  EXPECT(set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(generate(1) == UDONONNX_STATUS_SUCCESS);
  EXPECT(binding_valid);

  i = test_turns();
  if (i)
    return i;

  EXPECT(shutdown() == UDONONNX_STATUS_SUCCESS);
  EXPECT(!fake_session.live);
  EXPECT(tinyllamaConversationGenerationProgress(
             &progress_tokens, &progress_count, &serial, &is_completed) ==
         UDONONNX_STATUS_DISPOSED);
  EXPECT(initialize() == UDONONNX_STATUS_DISPOSED);
  return 0;
}
