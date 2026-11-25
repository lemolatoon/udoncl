/* Native test of examples/tinyllama-chat/tinyllama_chat.c.
 *
 * The Host runs against a fake OpenCL implementation and a fake UdonONNX
 * session defined here. The session keeps the binding of every graph input
 * and output, as the real one does, and every run checks that the Host bound
 * one position t: the token and its position (1 x 1), the attention mask of
 * t + 1 ones in its buffer of the capacity, and each layer's past of t rows
 * and present of t + 1 rows in one buffer of the capacity. The ArgMax launch
 * replays a token script of the test. tests/boundary/tinyllama_chat_host.py
 * generates the expectations (tinyllama_chat_expected.h) from the pinned
 * tokenizer with Hugging Face `tokenizers`, compiles this file with the Host,
 * the tokenizer tables and UdonTokenizers, and runs it. */
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <udoncl/array.h>
#include <udoncl/onnx_types.h>
#include <udoncl/string.h>
#include <udononnxruntime.h>

#include "tinyllama_chat_expected.h"

#define EXPECT(x)                                                              \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x);                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define LAYERS 22
#define VOCABULARY 32000

udononnxStatus_t initialize(void);
udononnxStatus_t begin_message(udon_string_t text);
int32_t continue_message(void);
udononnxStatus_t generate(void);
int32_t read_streamed_text(udon_int32_array_t destination, uint32_t offset);
int32_t read_generated_tokens(udon_int32_array_t destination);
udononnxStatus_t dispose(void);

/* ------------------------------------------------------------------------ */
/* Fake OpenCL: every object counts as live until released. Commands run    */
/* when they are enqueued.                                                  */

struct _cl_platform_id {
  int unused;
};
struct _cl_device_id {
  int unused;
};
struct _cl_context {
  int live;
};
struct _cl_command_queue {
  int live;
};
struct _cl_mem {
  int live;
  size_t size;
  cl_mem_flags flags;
  unsigned char *data;
};
struct _cl_program {
  int live, built;
};
struct _cl_kernel {
  int live;
  cl_mem args[2];
  int set[2];
};

static struct _cl_platform_id platform;
static struct _cl_device_id device;
static int live_objects, created_buffers;
static struct _cl_command_queue *the_queue;

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(cl_uint count,
                                                 cl_platform_id *platforms,
                                                 cl_uint *available) {
  if (count && platforms)
    platforms[0] = &platform;
  if (available)
    *available = 1;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id p,
                                               cl_device_type type,
                                               cl_uint count,
                                               cl_device_id *devices,
                                               cl_uint *available) {
  if (p != &platform || type != CL_DEVICE_TYPE_GPU)
    return CL_INVALID_VALUE;
  if (count && devices)
    devices[0] = &device;
  if (available)
    *available = 1;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_context CL_API_CALL clCreateContext(
    const cl_context_properties *properties, cl_uint count,
    const cl_device_id *devices,
    void(CL_CALLBACK *notify)(const char *, const void *, size_t, void *),
    void *user, cl_int *error) {
  struct _cl_context *context = calloc(1, sizeof(*context));
  (void)properties;
  (void)notify;
  (void)user;
  EXPECT(count == 1 && devices[0] == &device);
  context->live = 1;
  ++live_objects;
  *error = CL_SUCCESS;
  return context;
}

CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context context, cl_device_id d,
                     cl_command_queue_properties properties, cl_int *error) {
  struct _cl_command_queue *queue = calloc(1, sizeof(*queue));
  EXPECT(context->live && d == &device && !properties);
  queue->live = 1;
  ++live_objects;
  the_queue = queue;
  *error = CL_SUCCESS;
  return queue;
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context context,
                                               cl_mem_flags flags, size_t size,
                                               void *host, cl_int *error) {
  struct _cl_mem *memory = calloc(1, sizeof(*memory));
  EXPECT(context->live && !host && size);
  memory->live = 1;
  memory->size = size;
  memory->flags = flags;
  memory->data = calloc(1, size);
  ++live_objects;
  ++created_buffers;
  *error = CL_SUCCESS;
  return memory;
}

static void check_transfer(cl_command_queue queue, cl_mem memory, size_t offset,
                           size_t size, const void *pointer, cl_uint waits) {
  EXPECT(queue == the_queue && queue->live && memory && memory->live);
  EXPECT(pointer && size && offset + size <= memory->size && !waits);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue queue, cl_mem memory, cl_bool blocking,
                     size_t offset, size_t size, const void *source,
                     cl_uint waits, const cl_event *list, cl_event *event) {
  (void)blocking;
  (void)list;
  check_transfer(queue, memory, offset, size, source, waits);
  EXPECT(!event);
  memcpy(memory->data + offset, source, size);
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue queue, cl_mem memory, cl_bool blocking,
                    size_t offset, size_t size, void *destination,
                    cl_uint waits, const cl_event *list, cl_event *event) {
  (void)list;
  check_transfer(queue, memory, offset, size, destination, waits);
  /* The Host reads only the selected token, and waits for it. */
  EXPECT(blocking == CL_TRUE && !event && size == 4);
  memcpy(destination, memory->data + offset, size);
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueFillBuffer(cl_command_queue queue, cl_mem memory, const void *pattern,
                    size_t pattern_size, size_t offset, size_t size,
                    cl_uint waits, const cl_event *list, cl_event *event) {
  size_t at;
  (void)list;
  check_transfer(queue, memory, offset, size, pattern, waits);
  EXPECT(!event && pattern_size && size % pattern_size == 0);
  for (at = offset; at < offset + size; at += pattern_size)
    memcpy(memory->data + at, pattern, pattern_size);
  return CL_SUCCESS;
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(
    cl_context context, cl_uint count, const char **strings,
    const size_t *lengths, cl_int *error) {
  struct _cl_program *program = calloc(1, sizeof(*program));
  EXPECT(context->live && count == 1 && !lengths);
  EXPECT(!strcmp(strings[0], "argmax.cl"));
  program->live = 1;
  ++live_objects;
  *error = CL_SUCCESS;
  return program;
}

CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program program, cl_uint count, const cl_device_id *devices,
               const char *options,
               void(CL_CALLBACK *notify)(cl_program, void *), void *user) {
  EXPECT(program->live && !count && !devices && !options && !notify && !user);
  program->built = 1;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program program,
                                                  const char *name,
                                                  cl_int *error) {
  struct _cl_kernel *kernel = calloc(1, sizeof(*kernel));
  EXPECT(program->live && program->built);
  EXPECT(!strcmp(name, "tinyllama_argmax"));
  kernel->live = 1;
  ++live_objects;
  *error = CL_SUCCESS;
  return kernel;
}

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel, cl_uint index,
                                               size_t size, const void *value) {
  EXPECT(kernel->live && index < 2 && size == sizeof(cl_mem) && value);
  memcpy(&kernel->args[index], value, sizeof(cl_mem));
  kernel->set[index] = 1;
  return CL_SUCCESS;
}

static cl_mem bound_memory(const char *name);
static void argmax_launched(cl_mem token);

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(
    cl_command_queue queue, cl_kernel kernel, cl_uint dimensions,
    const size_t *offset, const size_t *global, const size_t *local,
    cl_uint waits, const cl_event *list, cl_event *event) {
  (void)list;
  EXPECT(queue == the_queue && kernel->live && dimensions == 1 && !offset);
  EXPECT(global[0] == 1 && local && local[0] == 1 && !waits && !event);
  EXPECT(kernel->set[0] && kernel->set[1]);
  /* ArgMax reads the logits the last run wrote. */
  EXPECT(kernel->args[0] == bound_memory("logits"));
  EXPECT(kernel->args[1]->live && kernel->args[1]->size == 4);
  argmax_launched(kernel->args[1]);
  return CL_SUCCESS;
}

#define RELEASE(type, object)                                                  \
  CL_API_ENTRY cl_int CL_API_CALL clRelease##type(object value) {              \
    EXPECT(value && value->live);                                              \
    value->live = 0;                                                           \
    --live_objects;                                                            \
    return CL_SUCCESS;                                                         \
  }
RELEASE(Context, cl_context)
RELEASE(CommandQueue, cl_command_queue)
RELEASE(Program, cl_program)
RELEASE(Kernel, cl_kernel)
RELEASE(MemObject, cl_mem)

/* ------------------------------------------------------------------------ */
/* Fake UdonONNX session.                                                   */

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

/* What every run saw: its token and position. */
static int32_t run_tokens[4 * TINYLLAMA_CAPACITY];
static int64_t run_positions[4 * TINYLLAMA_CAPACITY];
static int runs;

udononnxStatus_t udononnxCreateSession(cl_context context,
                                       cl_command_queue queue,
                                       udononnxSession_t **session) {
  EXPECT(context->live && queue == the_queue);
  if (fake_session.live)
    return UDONONNX_STATUS_INVALID_STATE;
  fake_session.live = 1;
  ++sessions;
  binding_count = 0;
  *session = &fake_session;
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t bind(udononnxSession_t *session, const char *name,
                             cl_mem memory, int rank, const int64_t *dims,
                             const int64_t *buffer_dims) {
  struct binding *entry = 0;
  int index;
  EXPECT(session == &fake_session && fake_session.live);
  EXPECT(name && memory && memory->live && rank >= 1 && rank <= 4 && dims);
  EXPECT(strlen(name) < sizeof(entry->name));
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

static cl_mem bound_memory(const char *name) {
  const struct binding *entry = bound(name);
  EXPECT(entry);
  return entry->memory;
}

/* Whether `name` is bound with `rank` and `dims` in a buffer laid out as
 * `layout` (0: as dims). */
static int bound_as(const char *name, int rank, const int64_t *dims,
                    const int64_t *layout) {
  const struct binding *entry = bound(name);
  if (!entry || entry->rank != rank ||
      memcmp(entry->dims, dims, sizeof(int64_t) * (size_t)rank))
    return 0;
  if (!layout)
    return !entry->has_buffer_dims;
  return entry->has_buffer_dims &&
         !memcmp(entry->buffer_dims, layout, sizeof(int64_t) * (size_t)rank);
}

/* Checks the bindings of a run and returns its position. */
static int64_t check_run_bindings(void) {
  static const int64_t token[2] = {1, 1}, logits[3] = {1, 1, VOCABULARY},
                       mask_layout[2] = {1, TINYLLAMA_CAPACITY},
                       cache_layout[4] = {1, 4, TINYLLAMA_CAPACITY, 64};
  const struct binding *mask = bound("attention_mask");
  int64_t position, mask_dims[2], past[4], present[4], one;
  char name[32];
  int layer, kind;
  EXPECT(mask && mask->rank == 2 && mask->dims[1] >= 1 &&
         mask->dims[1] <= TINYLLAMA_CAPACITY);
  position = mask->dims[1] - 1;
  mask_dims[0] = 1;
  mask_dims[1] = position + 1;
  past[0] = present[0] = 1;
  past[1] = present[1] = 4;
  past[2] = position;
  present[2] = position + 1;
  past[3] = present[3] = 64;
  EXPECT(bound_as("input_ids", 2, token, 0));
  EXPECT(bound_as("position_ids", 2, token, 0));
  EXPECT(bound_as("logits", 3, logits, 0));
  EXPECT(bound_as("attention_mask", 2, mask_dims, mask_layout));
  EXPECT(bound("logits")->memory->size == VOCABULARY * sizeof(float));
  EXPECT(mask->memory->size == TINYLLAMA_CAPACITY * sizeof(int64_t));
  for (layer = 0; layer < LAYERS; ++layer)
    for (kind = 0; kind < 2; ++kind) {
      const struct binding *input, *output;
      snprintf(name, sizeof(name), "past_key_values.%d.%s", layer,
               kind ? "value" : "key");
      EXPECT(bound_as(name, 4, past, cache_layout));
      input = bound(name);
      snprintf(name, sizeof(name), "present.%d.%s", layer,
               kind ? "value" : "key");
      EXPECT(bound_as(name, 4, present, cache_layout));
      output = bound(name);
      /* The past is the leading block of the present's buffer, a buffer of
       * the whole capacity of its own. */
      EXPECT(input->memory == output->memory);
      EXPECT(input->memory->size ==
             (size_t)TINYLLAMA_CAPACITY * 4 * 64 * sizeof(float));
    }
  EXPECT(binding_count == 3 + 1 + 2 * 2 * LAYERS);
  /* The mask holds a one for every position of the run. */
  for (one = 0; one <= position; ++one) {
    int64_t value;
    memcpy(&value, mask->memory->data + one * sizeof(int64_t), sizeof(value));
    EXPECT(value == 1);
  }
  return position;
}

udononnxStatus_t udononnxRun(udononnxSession_t *session) {
  int64_t position, token, uploaded;
  EXPECT(session == &fake_session && fake_session.live);
  position = check_run_bindings();
  memcpy(&token, bound_memory("input_ids")->data, sizeof(token));
  memcpy(&uploaded, bound_memory("position_ids")->data, sizeof(uploaded));
  EXPECT(uploaded == position);
  EXPECT(token >= 0 && token < VOCABULARY);
  EXPECT(runs < 4 * TINYLLAMA_CAPACITY);
  run_tokens[runs] = (int32_t)token;
  run_positions[runs] = position;
  ++runs;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxReleaseSession(udononnxSession_t *session) {
  EXPECT(session == &fake_session && fake_session.live);
  fake_session.live = 0;
  binding_count = 0;
  return UDONONNX_STATUS_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* The ArgMax script and the streamed text it observes.                     */

static const struct turn *current;
static int launches;
static int32_t chunk[512];
static udon_int32_array_native_t chunk_array = {chunk, 512};

/* While Generate runs, the text streamed so far is exactly what Hugging Face
 * decodes from the tokens that can no longer change. */
static void argmax_launched(cl_mem token) {
  int32_t count, index;
  uint32_t value;
  EXPECT(current && launches < current->script_count);
  count = read_streamed_text(&chunk_array, 0);
  EXPECT(count == current->stable_lengths[launches]);
  for (index = 0; index < count; ++index)
    EXPECT(chunk[index] ==
           current->stable_units[current->stable_offsets[launches] + index]);
  value = (uint32_t)current->script[launches++];
  memcpy(token->data, &value, sizeof(value));
}

/* ------------------------------------------------------------------------ */

static int32_t tokens[64];
static udon_int32_array_native_t tokens_array = {tokens, 64};
static udon_int32_array_native_t short_array = {tokens, 1};

/* Tokenizes a message; returns the number of ContinueMessage calls or
 * -status. */
static int32_t tokenize_units(const uint16_t *units, int32_t length) {
  static udon_string_native_t text;
  int32_t calls = 0, result;
  text.data = units;
  text.length = length;
  EXPECT(begin_message(&text) == UDONONNX_STATUS_SUCCESS);
  /* Nothing is ready until the message was tokenized. */
  EXPECT(generate() == UDONONNX_STATUS_INVALID_STATE);
  do {
    result = continue_message();
    ++calls;
  } while (result == 1);
  return result ? result : calls;
}

static int32_t tokenize(const struct turn *turn) {
  return tokenize_units(turn->user, turn->user_length);
}

static void run_turn(const struct turn *turn) {
  int32_t calls, index, count;
  int first = runs;
  if (turn->too_long_length) {
    /* A message beyond the capacity fails and changes nothing. */
    EXPECT(tokenize_units(turn->too_long, turn->too_long_length) ==
           -UDONONNX_STATUS_CAPACITY_EXCEEDED);
    EXPECT(generate() == UDONONNX_STATUS_INVALID_STATE);
    EXPECT(runs == first);
  }
  calls = tokenize(turn);
  EXPECT(calls > 1);
  current = turn;
  launches = 0;
  EXPECT(generate() == UDONONNX_STATUS_SUCCESS);
  current = 0;
  /* The ArgMax ran once per reply token. */
  EXPECT(launches == turn->reply_count);
  /* The closing tokens, the message, then every reply token but the last,
   * each at the next position. */
  EXPECT(runs - first == turn->run_count);
  for (index = 0; index < turn->run_count; ++index) {
    EXPECT(run_tokens[first + index] == turn->runs[index]);
    EXPECT(run_positions[first + index] == turn->first_position + index);
  }
  EXPECT(read_generated_tokens(&short_array) ==
         (turn->reply_count > 1 ? -UDONONNX_STATUS_INVALID_VALUE : 1));
  EXPECT(read_generated_tokens(&tokens_array) == turn->reply_count);
  for (index = 0; index < turn->reply_count; ++index)
    EXPECT(tokens[index] == turn->script[index]);
  /* The streamed text is the whole reply now. */
  count = read_streamed_text(&chunk_array, 0);
  EXPECT(count == turn->final_count);
  for (index = 0; index < count; ++index)
    EXPECT(chunk[index] == turn->final_units[index]);
  EXPECT(read_streamed_text(&chunk_array, (uint32_t)count) == 0);
  EXPECT(read_streamed_text(&chunk_array, (uint32_t)count + 1) ==
         -UDONONNX_STATUS_INVALID_VALUE);
  if (count > 1) {
    EXPECT(read_streamed_text(&chunk_array, 1) == count - 1);
    EXPECT(chunk[0] == turn->final_units[1]);
    EXPECT(read_streamed_text(&short_array, 1) == 1);
  }
}

int main(void) {
  int index;
  /* Nothing works before Initialize. */
  EXPECT(begin_message(0) == UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(continue_message() == -UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(generate() == UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(read_streamed_text(&chunk_array, 0) ==
         -UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(read_generated_tokens(&tokens_array) ==
         -UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(dispose() == UDONONNX_STATUS_NOT_INITIALIZED);

  for (index = 0; index < CONVERSATIONS; ++index) {
    const struct conversation *conversation = &conversations[index];
    int turn;
    EXPECT(initialize() == UDONONNX_STATUS_SUCCESS);
    EXPECT(initialize() == UDONONNX_STATUS_INVALID_STATE);
    /* The context, the queue, 5 + 44 buffers, the program and the kernel. */
    EXPECT(live_objects == 2 + 5 + 2 * LAYERS + 2);
    EXPECT(sessions == index + 1 && binding_count == 3);
    EXPECT(continue_message() == -UDONONNX_STATUS_INVALID_STATE);
    EXPECT(generate() == UDONONNX_STATUS_INVALID_STATE);
    EXPECT(read_streamed_text(&chunk_array, 0) == 0);
    EXPECT(read_generated_tokens(&tokens_array) == 0);
    for (turn = 0; turn < conversation->turn_count; ++turn)
      run_turn(&conversation->turns[turn]);
    if (conversation->overflow) {
      /* A message beyond the capacity fails and changes nothing. */
      int before = runs;
      EXPECT(tokenize(conversation->overflow) ==
             -UDONONNX_STATUS_CAPACITY_EXCEEDED);
      EXPECT(generate() == UDONONNX_STATUS_INVALID_STATE);
      EXPECT(runs == before);
      EXPECT(read_generated_tokens(&tokens_array) ==
             conversation->turns[conversation->turn_count - 1].reply_count);
    }
    EXPECT(dispose() == UDONONNX_STATUS_SUCCESS);
    EXPECT(live_objects == 0 && !fake_session.live);
    EXPECT(dispose() == UDONONNX_STATUS_NOT_INITIALIZED);
    EXPECT(begin_message(0) == UDONONNX_STATUS_NOT_INITIALIZED);
  }
  printf("tinyllama_chat_host=passed conversations=%d runs=%d buffers=%d\n",
         CONVERSATIONS, runs, created_buffers);
  return 0;
}
