#include <CL/cl.h>
#include <stdint.h>
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

static cl_context binding_context;
static cl_command_queue binding_queue;
static cl_mem input[3], output;
static int64_t seen_tokens[4], seen_positions[4];
static int runs, advances, resets, reduce_calls, finalize_calls;

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
  return 0;
}
udononnxStatus_t udon_onnx_binding_enqueue_advance_sequence_state(void) {
  ++advances;
  return 0;
}
udononnxStatus_t udon_onnx_binding_enqueue_run(void) {
  clEnqueueReadBuffer(binding_queue, input[0], CL_TRUE, 8, 8,
                      &seen_tokens[runs], 0, 0, 0);
  clEnqueueReadBuffer(binding_queue, input[2], CL_TRUE, 8, 8,
                      &seen_positions[runs], 0, 0, 0);
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
  ++reduce_calls;
  return a == output && o == 4 && w == 32000 ? CL_SUCCESS : CL_INVALID_VALUE;
}
cl_int __vrc_enqueue_udonaten_argmax_finalize_device_args(
    cl_command_queue q, cl_mem a, cl_mem p, cl_mem r, cl_int o, cl_int w,
    cl_int reject, cl_uint n, const cl_event *l, cl_event *e) {
  uint32_t token = 1701;
  (void)a;
  (void)p;
  (void)o;
  (void)w;
  (void)n;
  (void)l;
  (void)e;
  ++finalize_calls;
  return reject == 1
             ? clEnqueueWriteBuffer(q, r, CL_FALSE, 0, 4, &token, 0, 0, 0)
             : CL_INVALID_VALUE;
}

int main(void) {
  int32_t prompt_data[] = {1, 450, 4996, 17354}, result_data[] = {0};
  udon_int32_array_native_t prompt = {prompt_data, 4},
                            result = {result_data, 1};
  int i;
  EXPECT(tinyllama_initialize() == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_set_prompt_tokens(&prompt, 4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_generate(2) == UDONONNX_STATUS_NOT_SUPPORTED);
  EXPECT(tinyllama_generate(1) == UDONONNX_STATUS_SUCCESS);
  EXPECT(runs == 4 && advances == 3 && reduce_calls == 1 &&
         finalize_calls == 1);
  for (i = 0; i < 4; ++i)
    EXPECT(seen_tokens[i] == prompt_data[i] && seen_positions[i] == i);
  EXPECT(tinyllama_generated_token_count() == 1);
  EXPECT(tinyllama_read_generated_tokens(&result) == UDONONNX_STATUS_SUCCESS);
  EXPECT(result_data[0] == 1701 && resets == 1);
  EXPECT(tinyllama_dispose() == UDONONNX_STATUS_SUCCESS);
  EXPECT(tinyllama_initialize() == UDONONNX_STATUS_DISPOSED);
  return 0;
}
