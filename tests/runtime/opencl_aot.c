#include <CL/cl.h>
#include <stddef.h>
#include <string.h>
#include <udoncl/device_memory.h>
#include <udonblas.h>
#include <udondnn.h>
#include <udononnxoperators.h>
#include <udononnxruntime.h>

size_t udononnx_matmul_nbits_cpu_gid;
#include "../../libraries/udononnxoperators/device/matmul_nbits.cl"
size_t udononnx_gather_cpu_gid;
#include "../../libraries/udononnxoperators/device/gather.cl"
size_t udononnx_unary_cpu_gid;
#include "../../libraries/udononnxoperators/device/unary.cl"
size_t udononnx_reindex_cpu_gid;
#include "../../libraries/udononnxoperators/device/reindex.cl"
size_t udonaten_argmax_cpu_gid;
#include "../../libraries/udonaten/device/argmax.cl"

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition))                                                          \
      return __LINE__;                                                         \
  } while (0)

size_t udononnxGeneratedGetModelCount(void) { return 0; }

udononnxStatus_t udononnxGeneratedGetModel(size_t index,
                                           const udononnxModel_t **model) {
  (void)index;
  if (model)
    *model = 0;
  return UDONONNX_STATUS_INVALID_VALUE;
}

udononnxStatus_t udononnxGeneratedEnqueueInitialize(
    uint32_t graph_id, udononnxSession_t *session, cl_command_queue queue,
    cl_uint wait_count, const cl_event *wait_list, cl_event *completion) {
  (void)graph_id;
  (void)session;
  (void)queue;
  (void)wait_count;
  (void)wait_list;
  (void)completion;
  return UDONONNX_STATUS_NOT_SUPPORTED;
}

udononnxStatus_t
udononnxGeneratedEnqueueRun(uint32_t graph_id, udononnxSession_t *session,
                            cl_command_queue queue, cl_uint wait_count,
                            const cl_event *wait_list, cl_event *completion) {
  (void)graph_id;
  (void)session;
  (void)queue;
  (void)wait_count;
  (void)wait_list;
  (void)completion;
  return UDONONNX_STATUS_NOT_SUPPORTED;
}

udononnxStatus_t udononnxGeneratedDispose(uint32_t graph_id,
                                          udononnxSession_t *session) {
  (void)graph_id;
  (void)session;
  return UDONONNX_STATUS_SUCCESS;
}

static cl_uint allocated_logical_id;
static size_t allocated_bytes;
static cl_uint static_resource_id;
static size_t static_resource_bytes;
static cl_uint uploaded_token;
static cl_uint released_token;
static cl_uint dispatched_input_token;
static cl_uint dispatched_output_token;
static cl_uint dispatched_arena_a_token;
static cl_uint dispatched_arena_b_token;
static cl_uint dispatched_arena_output_token;
static cl_int dispatched_arena_stride_a;
static cl_int dispatched_arena_stride_b;
static cl_int dispatched_arena_batch_count;
static cl_int dispatched_m;
static cl_float dispatched_alpha;
static cl_int dispatched_sgemv_trans;
static cl_int dispatched_mnist_kernel;
static cl_int dispatched_nbits_tile_n;
static cl_int dispatched_nbits_finalize;
static cl_int dispatched_q4_block_count;
static cl_int dispatched_q4_splits;
static cl_int dispatched_q4_previous_partials;
static cl_int dispatched_gather_words;
static cl_int dispatched_dnn_operation;
static cl_uint dispatched_dnn_input_token;
static cl_uint dispatched_dnn_other_token;
static cl_uint dispatched_dnn_previous_token;
static cl_uint dispatched_dnn_output_token;
static cl_uint await_count;
/* The Udon binding owns one host-transfer command slot that an await flushes.
 * A transfer issued while another one is unawaited would overwrite it. */
static cl_uint transfer_outstanding;
static cl_uint transfer_overlaps;
static cl_uint await_failure_call;
static cl_int awaited_events[8];
static unsigned char physical_buffers[33][524288];

cl_uint __vrc_opencl_buffer_alloc(cl_uint logical_id, size_t bytes) {
  allocated_logical_id = logical_id;
  allocated_bytes = bytes;
  if (!logical_id || logical_id > 32 || bytes > sizeof(physical_buffers[0]))
    return 0;
  return logical_id + 100;
}

cl_uint __vrc_opencl_static_tensor(cl_uint resource_id, size_t bytes) {
  static_resource_id = resource_id;
  static_resource_bytes = bytes;
  return 0u - resource_id - 1u;
}

cl_int __vrc_opencl_buffer_release(cl_uint physical_token) {
  released_token = physical_token;
  return CL_SUCCESS;
}

cl_int __vrc_opencl_buffer_upload(cl_uint physical_token, size_t offset,
                                  size_t bytes, const void *source) {
  (void)offset;
  (void)bytes;
  (void)source;
  uploaded_token = physical_token;
  transfer_overlaps += transfer_outstanding;
  transfer_outstanding = 1;
  memcpy(physical_buffers[physical_token - 100] + offset, source, bytes);
  return 7;
}

cl_int __vrc_opencl_buffer_readback(cl_uint physical_token, size_t offset,
                                    size_t bytes, void *destination) {
  (void)physical_token;
  (void)offset;
  (void)bytes;
  transfer_overlaps += transfer_outstanding;
  transfer_outstanding = 1;
  memcpy(destination, physical_buffers[physical_token - 100] + offset, bytes);
  return 8;
}

static cl_uint copy_count;

cl_int __vrc_opencl_buffer_copy(cl_uint source_token, cl_uint destination_token,
                                size_t source_offset, size_t destination_offset,
                                size_t bytes) {
  ++copy_count;
  memmove(physical_buffers[destination_token - 100] + destination_offset,
          physical_buffers[source_token - 100] + source_offset, bytes);
  return 19;
}

static cl_uint fill_count;
static size_t filled_offset;
static size_t filled_bytes;
static size_t filled_pattern_bytes;
static uint32_t filled_pattern[32];

/* Like the Udon backend: word fill, complete at issue. */
cl_int __vrc_opencl_buffer_fill(cl_uint physical_token, size_t offset,
                                size_t bytes, const void *pattern,
                                size_t pattern_bytes) {
  size_t word;
  ++fill_count;
  filled_offset = offset;
  filled_bytes = bytes;
  filled_pattern_bytes = pattern_bytes;
  if (((size_t)pattern & 3u) || (offset & 3u) || (bytes & 3u) ||
      (pattern_bytes & 3u) || !pattern_bytes || pattern_bytes > 128u)
    return CL_INVALID_VALUE;
  memcpy(filled_pattern, pattern, pattern_bytes);
  for (word = 0; word < bytes / 4u; ++word)
    memcpy(physical_buffers[physical_token - 100] + offset + 4u * word,
           filled_pattern + word % (pattern_bytes / 4u), 4u);
  return CL_SUCCESS;
}

cl_int __vrc_opencl_dispatch_udonblas_sgemm(cl_uint input_token,
                                            cl_uint output_token, cl_int m,
                                            cl_int n, cl_int k, cl_int transa,
                                            cl_int transb, cl_int lda,
                                            cl_int ldb, cl_int ldc,
                                            cl_float alpha, cl_float beta) {
  (void)n;
  (void)k;
  (void)transa;
  (void)transb;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)beta;
  dispatched_input_token = input_token;
  dispatched_output_token = output_token;
  dispatched_m = m;
  dispatched_alpha = alpha;
  return 10;
}

cl_int __vrc_opencl_dispatch_udonblas_arena_sgemm(
    cl_uint a_token, cl_uint b_token, cl_uint output_token, cl_int m, cl_int n,
    cl_int k, cl_int transa, cl_int transb, cl_float alpha) {
  (void)m;
  (void)n;
  (void)k;
  (void)transa;
  (void)transb;
  (void)alpha;
  dispatched_arena_a_token = a_token;
  dispatched_arena_b_token = b_token;
  dispatched_arena_output_token = output_token;
  return 11;
}

cl_int __vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched(
    cl_uint a_token, cl_uint b_token, cl_uint output_token, cl_int m, cl_int n,
    cl_int k, cl_int stride_a, cl_int stride_b, cl_int stride_c,
    cl_int batch_count, cl_int transa, cl_int transb, cl_float alpha) {
  (void)m;
  (void)n;
  (void)k;
  (void)stride_c;
  (void)transa;
  (void)transb;
  (void)alpha;
  dispatched_arena_a_token = a_token;
  dispatched_arena_b_token = b_token;
  dispatched_arena_output_token = output_token;
  dispatched_arena_stride_a = stride_a;
  dispatched_arena_stride_b = stride_b;
  dispatched_arena_batch_count = batch_count;
  return 20;
}

cl_int __vrc_opencl_dispatch_udondnn_large_convolution(
    cl_uint input_token, cl_uint filter_token, cl_uint previous_token,
    cl_uint output_token, cl_int output_words, cl_int input_ch, cl_int input_w,
    cl_int output_ch, cl_int output_w, cl_int filter_hw, cl_int pad_hw,
    cl_int stride_hw, cl_int dilation_hw, cl_int mode, cl_float alpha,
    cl_float beta) {
  (void)output_words;
  (void)input_ch;
  (void)input_w;
  (void)output_ch;
  (void)output_w;
  (void)filter_hw;
  (void)pad_hw;
  (void)stride_hw;
  (void)dilation_hw;
  (void)mode;
  (void)alpha;
  (void)beta;
  dispatched_dnn_operation = 1;
  dispatched_dnn_input_token = input_token;
  dispatched_dnn_other_token = filter_token;
  dispatched_dnn_previous_token = previous_token;
  dispatched_dnn_output_token = output_token;
  return 30;
}

cl_int __vrc_opencl_dispatch_udondnn_large_activation(
    cl_uint input_token, cl_uint previous_token, cl_uint output_token,
    cl_int words, cl_int mode, cl_int nan_propagation, cl_float coefficient,
    cl_float alpha, cl_float beta) {
  (void)words;
  (void)mode;
  (void)nan_propagation;
  (void)coefficient;
  (void)alpha;
  (void)beta;
  dispatched_dnn_operation = 2;
  dispatched_dnn_input_token = input_token;
  dispatched_dnn_other_token = 0;
  dispatched_dnn_previous_token = previous_token;
  dispatched_dnn_output_token = output_token;
  return 31;
}

cl_int __vrc_opencl_dispatch_udondnn_large_pooling(
    cl_uint input_token, cl_uint previous_token, cl_uint output_token,
    cl_int output_words, cl_int input_c, cl_int input_h, cl_int input_w,
    cl_int output_h, cl_int output_w, cl_int window_shape, cl_int pad_shape,
    cl_int stride_shape, cl_int options, cl_float alpha, cl_float beta) {
  (void)output_words;
  (void)input_c;
  (void)input_h;
  (void)input_w;
  (void)output_h;
  (void)output_w;
  (void)window_shape;
  (void)pad_shape;
  (void)stride_shape;
  (void)options;
  (void)alpha;
  (void)beta;
  dispatched_dnn_operation = 3;
  dispatched_dnn_input_token = input_token;
  dispatched_dnn_other_token = 0;
  dispatched_dnn_previous_token = previous_token;
  dispatched_dnn_output_token = output_token;
  return 32;
}

cl_int __vrc_opencl_dispatch_udondnn_large_op_tensor(
    cl_uint a_token, cl_uint b_token, cl_uint previous_token,
    cl_uint output_token, cl_int output_words, cl_int a_nc, cl_int a_hw,
    cl_int b_nc, cl_int b_hw, cl_int output_nc, cl_int output_hw,
    cl_int options, cl_float alpha_a, cl_float alpha_b, cl_float beta) {
  (void)output_words;
  (void)a_nc;
  (void)a_hw;
  (void)b_nc;
  (void)b_hw;
  (void)output_nc;
  (void)output_hw;
  (void)options;
  (void)alpha_a;
  (void)alpha_b;
  (void)beta;
  dispatched_dnn_operation = 4;
  dispatched_dnn_input_token = a_token;
  dispatched_dnn_other_token = b_token;
  dispatched_dnn_previous_token = previous_token;
  dispatched_dnn_output_token = output_token;
  return 33;
}

cl_int __vrc_opencl_dispatch_udonaten_embedding(
    cl_uint input_token, cl_uint output_token, cl_int outer, cl_int rows,
    cl_float rows_float, cl_int inner, cl_int indices_words,
    cl_int output_words) {
  (void)input_token;
  (void)output_token;
  (void)outer;
  (void)rows;
  (void)rows_float;
  (void)inner;
  (void)indices_words;
  (void)output_words;
  return 34;
}

cl_int __vrc_opencl_dispatch_udonaten_arena_rms_norm_reduce(
    cl_uint input_token, cl_uint partials_token, cl_int words,
    cl_int normalized_words) {
  (void)input_token;
  (void)partials_token;
  (void)words;
  (void)normalized_words;
  return 35;
}

cl_int __vrc_opencl_dispatch_udonaten_arena_rms_norm_finalize(
    cl_uint input_token, cl_uint partials_token, cl_uint weight_token,
    cl_uint output_token, cl_int words, cl_int normalized_words,
    cl_float epsilon, cl_int has_weight) {
  (void)input_token;
  (void)partials_token;
  (void)weight_token;
  (void)output_token;
  (void)words;
  (void)normalized_words;
  (void)epsilon;
  (void)has_weight;
  return 36;
}

cl_int __vrc_opencl_dispatch_udonaten_argmax_reduce(cl_uint input_token,
                                                    cl_uint partials_token,
                                                    cl_int input_word_offset,
                                                    cl_int words) {
  for (udonaten_argmax_cpu_gid = 0; udonaten_argmax_cpu_gid < 512;
       ++udonaten_argmax_cpu_gid)
    udonaten_argmax_reduce((const float *)physical_buffers[input_token - 100],
                           (cl_uint *)physical_buffers[partials_token - 100],
                           input_word_offset, words);
  return 40;
}

cl_int __vrc_opencl_dispatch_udonaten_argmax_finalize(
    cl_uint input_token, cl_uint partials_token, cl_uint result_token,
    cl_int input_word_offset, cl_int words, cl_int reject_nonfinite) {
  udonaten_argmax_cpu_gid = 0;
  udonaten_argmax_finalize(
      (const float *)physical_buffers[input_token - 100],
      (const cl_uint *)physical_buffers[partials_token - 100],
      (cl_uint *)physical_buffers[result_token - 100], input_word_offset, words,
      reject_nonfinite);
  return 41;
}

cl_int __vrc_opencl_dispatch_udondnn_arena_softmax(
    cl_uint input_token, cl_uint output_token, cl_int input_n, cl_int input_c,
    cl_int input_h, cl_int input_w, cl_int algorithm, cl_int mode) {
  (void)input_token;
  (void)output_token;
  (void)input_n;
  (void)input_c;
  (void)input_h;
  (void)input_w;
  (void)algorithm;
  (void)mode;
  return 37;
}

extern cl_int __vrc_enqueue_udonblas_arena_sgemm_args(
    cl_command_queue queue, cl_mem a, cl_mem b, cl_mem output, cl_int m,
    cl_int n, cl_int k, cl_int transa, cl_int transb, cl_float alpha);
extern cl_int __vrc_enqueue_udonblas_arena_sgemm_device_args(
    cl_command_queue queue, cl_mem a, cl_mem b, cl_mem output, cl_int m,
    cl_int n, cl_int k, cl_int transa, cl_int transb, cl_float alpha,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event);
extern cl_int __vrc_enqueue_udonblas_arena_sgemm_strided_batched_device_args(
    cl_command_queue queue, cl_mem a, cl_mem b, cl_mem output, cl_int m,
    cl_int n, cl_int k, cl_int stride_a, cl_int stride_b, cl_int stride_c,
    cl_int batch_count, cl_int transa, cl_int transb, cl_float alpha,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event);
extern cl_int __vrc_enqueue_udononnx_unary_device_args(
    cl_command_queue queue, cl_mem input, cl_mem output, cl_int operation,
    cl_int words, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event);
extern cl_int __vrc_enqueue_udononnx_transpose_device_args(
    cl_command_queue queue, cl_mem input, cl_mem output, cl_int words,
    cl_int rank, cl_int shape01, cl_int shape23, cl_int shape4,
    cl_int permutation, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event);
extern cl_int __vrc_enqueue_udononnx_slice_device_args(
    cl_command_queue queue, cl_mem input, cl_mem output, cl_int words,
    cl_int rank, cl_int input_shape01, cl_int input_shape23,
    cl_int input_shape4, cl_int output_shape01, cl_int output_shape23,
    cl_int output_shape4, cl_int starts01, cl_int starts23, cl_int starts4,
    cl_int steps01, cl_int steps23, cl_int steps4, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event);
extern cl_int __vrc_enqueue_udononnx_expand_device_args(
    cl_command_queue queue, cl_mem input, cl_mem output, cl_int words,
    cl_int input_rank, cl_int input_shape01, cl_int input_shape23,
    cl_int input_shape4, cl_int output_rank, cl_int output_shape01,
    cl_int output_shape23, cl_int output_shape4, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event);
extern cl_int __vrc_enqueue_udondnn_large_op_tensor_device_args(
    cl_command_queue queue, cl_mem a, cl_mem b, cl_mem previous, cl_mem output,
    cl_int words, cl_int a_nc, cl_int a_hw, cl_int b_nc, cl_int b_hw,
    cl_int c_nc, cl_int c_hw, cl_int options, cl_float alpha_a,
    cl_float alpha_b, cl_float beta, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event);
extern cl_int __vrc_enqueue_udononnx_concat2_device_args(
    cl_command_queue queue, cl_mem left, cl_mem right, cl_mem output,
    cl_int words, cl_int inner_words, cl_int left_extent, cl_int right_extent,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event);
extern cl_int __vrc_enqueue_udononnx_gather_axis0_device_args(
    cl_command_queue queue, cl_mem data, cl_mem indices, cl_mem output,
    cl_int words, cl_int inner_words, cl_int axis_extent, cl_int index_words,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event);
extern cl_int __vrc_enqueue_udononnx_cast_i64_f32_device_args(
    cl_command_queue queue, cl_mem input, cl_mem output, cl_int words,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event);
extern cl_int __vrc_enqueue_udononnx_equal_f32_device_args(
    cl_command_queue queue, cl_mem left, cl_mem right, cl_mem output,
    cl_int words, cl_int ranks, cl_int left_shape01, cl_int left_shape23,
    cl_int right_shape01, cl_int right_shape23, cl_int output_shape01,
    cl_int output_shape23, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event);
extern cl_int __vrc_enqueue_udononnx_where_f32_device_args(
    cl_command_queue queue, cl_mem condition, cl_mem when_true,
    cl_mem when_false, cl_mem output, cl_int words, cl_int ranks,
    cl_int condition_shape01, cl_int condition_shape23, cl_int true_shape01,
    cl_int true_shape23, cl_int false_shape01, cl_int false_shape23,
    cl_int output_shape01, cl_int output_shape23, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event);
extern cl_int __vrc_enqueue_udononnx_scatter_nd_f32_device_args(
    cl_command_queue queue, cl_mem data, cl_mem indices, cl_mem updates,
    cl_mem output, cl_int words, cl_int rank, cl_int shape01, cl_int shape23,
    cl_int tuple_rank, cl_int update_count, cl_int slice_words,
    cl_int index_words, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event);
extern cl_int __vrc_enqueue_udonaten_argmax_reduce_device_args(
    cl_command_queue queue, cl_mem input, cl_mem partials,
    cl_int input_word_offset, cl_int words, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event);
extern cl_int __vrc_enqueue_udonaten_argmax_finalize_device_args(
    cl_command_queue queue, cl_mem input, cl_mem partials, cl_mem result,
    cl_int input_word_offset, cl_int words, cl_int reject_nonfinite,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event);
extern cl_int __vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
    cl_command_queue queue, cl_mem activation, cl_mem packed_weight,
    cl_mem scales, cl_mem previous, cl_mem output, cl_int m, cl_int n, cl_int k,
    cl_int block_count, cl_int splits, cl_int previous_partials,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event);
extern cl_mem udonclAdoptStaticBuffer(cl_command_queue queue,
                                      cl_uint resource_id, size_t bytes,
                                      cl_uint physical_token, cl_int *error);
extern cl_mem udonclCreateBufferForQueue(cl_command_queue queue,
                                         cl_mem_flags flags, size_t bytes,
                                         cl_int *error);

cl_int __vrc_opencl_dispatch_udonblas_sgemv(cl_uint input_token,
                                            cl_uint output_token, cl_int trans,
                                            cl_int m, cl_int n, cl_int lda,
                                            cl_int incx, cl_int incy,
                                            cl_float alpha, cl_float beta) {
  (void)m;
  (void)n;
  (void)lda;
  (void)incx;
  (void)incy;
  (void)beta;
  dispatched_input_token = input_token;
  dispatched_output_token = output_token;
  dispatched_sgemv_trans = trans;
  dispatched_alpha = alpha;
  return 11;
}

cl_int __vrc_opencl_dispatch_mnist_conv(cl_uint input_token,
                                        cl_uint output_token) {
  dispatched_input_token = input_token;
  dispatched_output_token = output_token;
  dispatched_mnist_kernel = 20;
  return 12;
}

cl_int __vrc_opencl_dispatch_mnist_pool(cl_uint input_token,
                                        cl_uint output_token) {
  dispatched_input_token = input_token;
  dispatched_output_token = output_token;
  dispatched_mnist_kernel = 21;
  return 13;
}

cl_int __vrc_opencl_dispatch_mnist_dense(cl_uint input_token,
                                         cl_uint output_token) {
  dispatched_input_token = input_token;
  dispatched_output_token = output_token;
  dispatched_mnist_kernel = 22;
  return 14;
}

cl_int __vrc_opencl_dispatch_udononnx_matmul_nbits(cl_uint input_token,
                                                   cl_uint output_token,
                                                   cl_int tile_n, cl_int tile_k,
                                                   cl_int accumulate,
                                                   cl_int finalize) {
  (void)tile_k;
  (void)accumulate;
  dispatched_input_token = input_token;
  dispatched_output_token = output_token;
  dispatched_nbits_tile_n = tile_n;
  dispatched_nbits_finalize = finalize;
  for (udononnx_matmul_nbits_cpu_gid = 0; udononnx_matmul_nbits_cpu_gid < 16384;
       ++udononnx_matmul_nbits_cpu_gid)
    udononnx_matmul_nbits((const float *)physical_buffers[input_token - 100],
                          (float *)physical_buffers[output_token - 100], tile_n,
                          tile_k, accumulate, finalize);
  return 15;
}

cl_int __vrc_opencl_dispatch_udononnx_matmul_nbits_q4(
    cl_uint activation_token, cl_uint packed_weight_token, cl_uint scales_token,
    cl_uint previous_token, cl_uint output_token, cl_int m, cl_int n, cl_int k,
    cl_int block_count, cl_int splits, cl_int previous_partials) {
  (void)activation_token;
  (void)packed_weight_token;
  (void)scales_token;
  (void)previous_token;
  (void)output_token;
  (void)m;
  (void)n;
  (void)k;
  dispatched_q4_block_count = block_count;
  dispatched_q4_splits = splits;
  dispatched_q4_previous_partials = previous_partials;
  return CL_SUCCESS;
}

cl_int __vrc_opencl_dispatch_udononnx_gather_copy(cl_uint input_token,
                                                  cl_uint output_token,
                                                  cl_int words) {
  dispatched_input_token = input_token;
  dispatched_output_token = output_token;
  dispatched_gather_words = words;
  for (udononnx_gather_cpu_gid = 0; udononnx_gather_cpu_gid < 16384;
       ++udononnx_gather_cpu_gid)
    udononnx_gather_copy((const float *)physical_buffers[input_token - 100],
                         (float *)physical_buffers[output_token - 100], words);
  return 16;
}

cl_int __vrc_opencl_dispatch_udononnx_unary_f32(cl_uint input_token,
                                                cl_uint output_token,
                                                cl_int operation,
                                                cl_int words) {
  for (udononnx_unary_cpu_gid = 0; udononnx_unary_cpu_gid < 16384;
       ++udononnx_unary_cpu_gid)
    udononnx_unary_f32((const float *)physical_buffers[input_token - 100],
                       (float *)physical_buffers[output_token - 100], operation,
                       words);
  return 17;
}

cl_int __vrc_opencl_dispatch_udononnx_transpose_f32(
    cl_uint input_token, cl_uint output_token, cl_int words, cl_int rank,
    cl_int shape01, cl_int shape23, cl_int shape4, cl_int permutation) {
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < 131072;
       ++udononnx_reindex_cpu_gid)
    udononnx_transpose_f32((const float *)physical_buffers[input_token - 100],
                           (float *)physical_buffers[output_token - 100], words,
                           rank, shape01, shape23, shape4, permutation);
  return 21;
}

cl_int __vrc_opencl_dispatch_udononnx_slice_f32(
    cl_uint input_token, cl_uint output_token, cl_int words, cl_int rank,
    cl_int input_shape01, cl_int input_shape23, cl_int input_shape4,
    cl_int output_shape01, cl_int output_shape23, cl_int output_shape4,
    cl_int starts01, cl_int starts23, cl_int starts4, cl_int steps01,
    cl_int steps23, cl_int steps4) {
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < 16384;
       ++udononnx_reindex_cpu_gid)
    udononnx_slice_f32((const float *)physical_buffers[input_token - 100],
                       (float *)physical_buffers[output_token - 100], words,
                       rank, input_shape01, input_shape23, input_shape4,
                       output_shape01, output_shape23, output_shape4, starts01,
                       starts23, starts4, steps01, steps23, steps4);
  return 22;
}

cl_int __vrc_opencl_dispatch_udononnx_expand_f32(
    cl_uint input_token, cl_uint output_token, cl_int words, cl_int input_rank,
    cl_int input_shape01, cl_int input_shape23, cl_int input_shape4,
    cl_int output_rank, cl_int output_shape01, cl_int output_shape23,
    cl_int output_shape4) {
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < 131072;
       ++udononnx_reindex_cpu_gid)
    udononnx_expand_f32((const float *)physical_buffers[input_token - 100],
                        (float *)physical_buffers[output_token - 100], words,
                        input_rank, input_shape01, input_shape23, input_shape4,
                        output_rank, output_shape01, output_shape23,
                        output_shape4);
  return 23;
}

cl_int __vrc_opencl_dispatch_udononnx_concat2_f32(
    cl_uint left_token, cl_uint right_token, cl_uint output_token, cl_int words,
    cl_int inner_words, cl_int left_extent, cl_int right_extent) {
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < 16384;
       ++udononnx_reindex_cpu_gid)
    udononnx_concat2_f32((const float *)physical_buffers[left_token - 100],
                         (const float *)physical_buffers[right_token - 100],
                         (float *)physical_buffers[output_token - 100], words,
                         inner_words, left_extent, right_extent);
  return 24;
}

cl_int __vrc_opencl_dispatch_udononnx_gather_axis0_f32(
    cl_uint data_token, cl_uint indices_token, cl_uint output_token,
    cl_int words, cl_int inner_words, cl_int axis_extent, cl_int index_words) {
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < 16384;
       ++udononnx_reindex_cpu_gid)
    udononnx_gather_axis0_f32(
        (const float *)physical_buffers[data_token - 100],
        (const uint *)physical_buffers[indices_token - 100],
        (float *)physical_buffers[output_token - 100], words, inner_words,
        axis_extent, index_words);
  return 25;
}

cl_int __vrc_opencl_dispatch_udononnx_cast_i64_f32(cl_uint input_token,
                                                   cl_uint output_token,
                                                   cl_int words) {
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < 16384;
       ++udononnx_reindex_cpu_gid)
    udononnx_cast_i64_f32((const uint *)physical_buffers[input_token - 100],
                          (float *)physical_buffers[output_token - 100], words);
  return 26;
}

cl_int __vrc_opencl_dispatch_udononnx_equal_f32(
    cl_uint left_token, cl_uint right_token, cl_uint output_token, cl_int words,
    cl_int ranks, cl_int left_shape01, cl_int left_shape23,
    cl_int right_shape01, cl_int right_shape23, cl_int output_shape01,
    cl_int output_shape23) {
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < 16384;
       ++udononnx_reindex_cpu_gid)
    udononnx_equal_f32((const float *)physical_buffers[left_token - 100],
                       (const float *)physical_buffers[right_token - 100],
                       (float *)physical_buffers[output_token - 100], words,
                       ranks, left_shape01, left_shape23, right_shape01,
                       right_shape23, output_shape01, output_shape23);
  return 27;
}

cl_int __vrc_opencl_dispatch_udononnx_where_f32(
    cl_uint condition_token, cl_uint true_token, cl_uint false_token,
    cl_uint output_token, cl_int words, cl_int ranks, cl_int condition_shape01,
    cl_int condition_shape23, cl_int true_shape01, cl_int true_shape23,
    cl_int false_shape01, cl_int false_shape23, cl_int output_shape01,
    cl_int output_shape23) {
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < 16384;
       ++udononnx_reindex_cpu_gid)
    udononnx_where_f32((const float *)physical_buffers[condition_token - 100],
                       (const float *)physical_buffers[true_token - 100],
                       (const float *)physical_buffers[false_token - 100],
                       (float *)physical_buffers[output_token - 100], words,
                       ranks, condition_shape01, condition_shape23,
                       true_shape01, true_shape23, false_shape01, false_shape23,
                       output_shape01, output_shape23);
  return 28;
}

cl_int __vrc_opencl_dispatch_udononnx_scatter_nd_f32(
    cl_uint data_token, cl_uint indices_token, cl_uint updates_token,
    cl_uint output_token, cl_int words, cl_int rank, cl_int shape01,
    cl_int shape23, cl_int tuple_rank, cl_int update_count, cl_int slice_words,
    cl_int index_words) {
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < 16384;
       ++udononnx_reindex_cpu_gid)
    udononnx_scatter_nd_f32(
        (const float *)physical_buffers[data_token - 100],
        (const uint *)physical_buffers[indices_token - 100],
        (const float *)physical_buffers[updates_token - 100],
        (float *)physical_buffers[output_token - 100], words, rank, shape01,
        shape23, tuple_rank, update_count, slice_words, index_words);
  return 29;
}

static cl_uint flush_count;

/* Like the Udon backend: the binding submits the published transfer and the
 * Host resumes without waiting for it. */
cl_int __vrc_opencl_flush(void) {
  ++flush_count;
  return CL_SUCCESS;
}

cl_int __vrc_opencl_await(cl_int event) {
  transfer_outstanding = 0;
  if (await_count < sizeof(awaited_events) / sizeof(awaited_events[0]))
    awaited_events[await_count] = event;
  ++await_count;
  if (await_failure_call && await_count == await_failure_call)
    return CL_INVALID_VALUE;
  return CL_SUCCESS;
}

static void pack_nbits(uint8_t *destination, size_t bytes, size_t index,
                       unsigned value, int bits) {
  size_t bit = index * (size_t)bits;
  size_t byte = bit >> 3u;
  unsigned shift = (unsigned)(bit & 7u);
  unsigned word = value << shift;
  destination[byte] |= (uint8_t)word;
  if (shift + (unsigned)bits > 8u && byte + 1u < bytes)
    destination[byte + 1u] |= (uint8_t)(word >> 8u);
}

/* clEnqueueFillBuffer replicates 1- and 2-byte patterns to one word, keeps
 * aligned word patterns in place, copies unaligned ones, drains an
 * outstanding binding transfer first, and completes at issue. */
static int test_fill_buffer(cl_context context, cl_command_queue queue,
                            cl_mem static_buffer) {
  static const unsigned char expected[24] = {
      1, 2, 3, 4, 0xA5, 0x5A, 0xA5, 0x5A, 0xA5, 0x5A, 0xA5, 0x5A,
      1, 2, 3, 4, 9,    8,    7,    6,    5,    4,    3,    2};
  unsigned char initial[24], readback[24];
  unsigned char halfword[2] = {0xA5, 0x5A};
  unsigned char unaligned[12] = {0, 9, 8, 7, 6, 5, 4, 3, 2};
  uint32_t word_pattern[2] = {0x04030201u, 0x08070605u};
  cl_event fill_event = 0;
  cl_int error, status;
  size_t byte;
  cl_mem buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(initial), 0, &error);
  EXPECT(buffer && error == CL_SUCCESS);
  for (byte = 0; byte < sizeof(initial); ++byte)
    initial[byte] = (unsigned char)(byte + 1u);
  EXPECT(clEnqueueFillBuffer(queue, buffer, 0, 4, 0, 4, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueFillBuffer(queue, buffer, halfword, 3, 0, 6, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueFillBuffer(queue, buffer, word_pattern, 256, 0, 0, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueFillBuffer(queue, buffer, word_pattern, 8, 4, 8, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueFillBuffer(queue, buffer, word_pattern, 8, 0, 12, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueFillBuffer(queue, buffer, word_pattern, 4, 20, 8, 0, 0, 0) ==
         CL_INVALID_VALUE);
  /* OpenCL permits these 2-byte fills; the word Device does not. */
  EXPECT(clEnqueueFillBuffer(queue, buffer, halfword, 2, 2, 4, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueFillBuffer(queue, buffer, halfword, 2, 4, 2, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueFillBuffer(queue, static_buffer, word_pattern, 4, 0, 4, 0, 0,
                             0) == CL_INVALID_MEM_OBJECT);
  EXPECT(clEnqueueFillBuffer(0, buffer, word_pattern, 4, 0, 4, 0, 0, 0) ==
         CL_INVALID_COMMAND_QUEUE);
  EXPECT(clEnqueueFillBuffer(queue, buffer, word_pattern, 4, 0, 4, 1, 0, 0) ==
         CL_INVALID_EVENT_WAIT_LIST);
  EXPECT(fill_count == 0);

  /* A non-blocking write still in the command slot is drained first. */
  await_count = 0;
  EXPECT(clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, sizeof(initial),
                              initial, 0, 0, 0) == CL_SUCCESS);
  EXPECT(transfer_outstanding && await_count == 0);
  EXPECT(clEnqueueFillBuffer(queue, buffer, halfword, 2, 4, 8, 0, 0,
                             &fill_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && !transfer_outstanding);
  EXPECT(fill_count == 1 && filled_offset == 4 && filled_bytes == 8 &&
         filled_pattern_bytes == 4 && filled_pattern[0] == 0x5AA55AA5u);
  EXPECT(clGetEventInfo(fill_event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(status), &status, 0) == CL_SUCCESS &&
         status == CL_COMPLETE);
  EXPECT(clReleaseEvent(fill_event) == CL_SUCCESS);
  EXPECT(clEnqueueFillBuffer(queue, buffer, unaligned + 1, 8, 16, 8, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(fill_count == 2 && filled_pattern_bytes == 8 &&
         filled_pattern[0] == 0x06070809u && filled_pattern[1] == 0x02030405u);
  EXPECT(clEnqueueFillBuffer(queue, buffer, word_pattern, 4, 12, 4, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(fill_count == 3 && filled_pattern[0] == 0x04030201u);
  EXPECT(clEnqueueFillBuffer(queue, buffer, word_pattern, 8, 0, 0, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(fill_count == 3);
  EXPECT(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(readback),
                             readback, 0, 0, 0) == CL_SUCCESS);
  EXPECT(!memcmp(readback, expected, sizeof(expected)));
  EXPECT(clReleaseMemObject(buffer) == CL_SUCCESS);
  return 0;
}

/* clEnqueueCopyBuffer copies any word-aligned range between two physical
 * buffers as one Device command that is complete at issue; it never reads the
 * destination back. */
static int test_copy_buffer(cl_context context, cl_command_queue queue,
                            cl_mem static_buffer) {
  static const unsigned char expected[16] = {21, 22, 23, 24, 9,  10, 11, 12,
                                             29, 30, 31, 32, 33, 34, 35, 36};
  unsigned char source_bytes[16], destination_bytes[16], readback[16];
  cl_buffer_region region = {8, 4};
  cl_event copy_event = 0;
  cl_command_type command;
  cl_int error;
  size_t byte;
  cl_mem source = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(source_bytes), 0, &error);
  cl_mem destination, part;
  EXPECT(source && error == CL_SUCCESS);
  /* CL_MEM_READ_ONLY constrains kernel access, not Host transfers: a model
   * input is such a buffer and still receives Device feedback copies. */
  destination = clCreateBuffer(context, CL_MEM_READ_ONLY,
                               sizeof(destination_bytes), 0, &error);
  EXPECT(destination && error == CL_SUCCESS);
  part = clCreateSubBuffer(source, CL_MEM_READ_WRITE,
                           CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
  EXPECT(part && error == CL_SUCCESS);
  for (byte = 0; byte < sizeof(source_bytes); ++byte) {
    source_bytes[byte] = (unsigned char)(byte + 1u);
    destination_bytes[byte] = (unsigned char)(byte + 21u);
  }
  EXPECT(clEnqueueWriteBuffer(queue, source, CL_TRUE, 0, sizeof(source_bytes),
                              source_bytes, 0, 0, 0) == CL_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, destination, CL_TRUE, 0,
                              sizeof(destination_bytes), destination_bytes, 0,
                              0, 0) == CL_SUCCESS);
  copy_count = 0;
  /* OpenCL permits byte ranges; the word Device does not. */
  EXPECT(clEnqueueCopyBuffer(queue, source, destination, 2, 0, 4, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueCopyBuffer(queue, source, destination, 0, 2, 4, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueCopyBuffer(queue, source, destination, 0, 0, 6, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueCopyBuffer(queue, source, destination, 12, 0, 8, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueCopyBuffer(queue, source, destination, 0, 12, 8, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(clEnqueueCopyBuffer(queue, source, source, 0, 4, 8, 0, 0, 0) ==
         CL_MEM_COPY_OVERLAP);
  /* One Blit cannot sample the surface it renders to. */
  EXPECT(clEnqueueCopyBuffer(queue, source, source, 0, 8, 4, 0, 0, 0) ==
         CL_INVALID_OPERATION);
  EXPECT(clEnqueueCopyBuffer(queue, part, source, 0, 0, 4, 0, 0, 0) ==
         CL_INVALID_OPERATION);
  EXPECT(clEnqueueCopyBuffer(queue, source, static_buffer, 0, 0, 4, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(copy_count == 0);

  await_count = 0;
  EXPECT(clEnqueueCopyBuffer(queue, source, destination, 8, 4, 4, 0, 0,
                             &copy_event) == CL_SUCCESS);
  EXPECT(copy_count == 1 && await_count == 0);
  EXPECT(clGetEventInfo(copy_event, CL_EVENT_COMMAND_TYPE, sizeof(command),
                        &command, 0) == CL_SUCCESS &&
         command == CL_COMMAND_COPY_BUFFER);
  EXPECT(clReleaseEvent(copy_event) == CL_SUCCESS);
  EXPECT(clEnqueueReadBuffer(queue, destination, CL_TRUE, 0, sizeof(readback),
                             readback, 0, 0, 0) == CL_SUCCESS);
  EXPECT(!memcmp(readback, expected, sizeof(expected)));
  /* A sub-buffer range is addressed from its origin in the parent. */
  EXPECT(clEnqueueCopyBuffer(queue, part, destination, 0, 12, 4, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(clEnqueueReadBuffer(queue, destination, CL_TRUE, 0, sizeof(readback),
                             readback, 0, 0, 0) == CL_SUCCESS);
  EXPECT(!memcmp(readback, expected, 12) &&
         !memcmp(readback + 12, source_bytes + 8, 4));
  EXPECT(clReleaseMemObject(part) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(destination) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(source) == CL_SUCCESS);
  return 0;
}

/* A non-blocking read is submitted at its queue position and awaited only
 * when its event is waited for, or when the next transfer needs the command
 * slot. Device commands issued in between do not wait for it. */
static int test_deferred_read(cl_context context, cl_command_queue queue) {
  uint32_t value = 0x11223344u, other = 0x55667788u, readback = 0;
  uint32_t pattern = 0xCAFEF00Du;
  cl_event read_event = 0, second_event = 0;
  cl_int error, status;
  cl_mem source =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(value), 0, &error);
  cl_mem destination;
  EXPECT(source && error == CL_SUCCESS);
  destination =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(value), 0, &error);
  EXPECT(destination && error == CL_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, source, CL_TRUE, 0, sizeof(value), &value,
                              0, 0, 0) == CL_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, destination, CL_TRUE, 0, sizeof(other),
                              &other, 0, 0, 0) == CL_SUCCESS);

  /* A blocking read waits and needs no separate submission. */
  await_count = flush_count = 0;
  EXPECT(clEnqueueReadBuffer(queue, source, CL_TRUE, 0, sizeof(readback),
                             &readback, 0, 0, 0) == CL_SUCCESS);
  EXPECT(await_count == 1 && flush_count == 0 && readback == value);

  await_count = 0;
  EXPECT(clEnqueueReadBuffer(queue, source, CL_FALSE, 0, sizeof(readback),
                             &readback, 0, 0, &read_event) == CL_SUCCESS);
  EXPECT(flush_count == 1 && await_count == 0 && transfer_outstanding);
  EXPECT(clGetEventInfo(read_event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(status), &status, 0) == CL_SUCCESS &&
         status == CL_SUBMITTED);
  /* Device commands do not wait for the read. */
  EXPECT(clEnqueueCopyBuffer(queue, source, destination, 0, 0, sizeof(value), 0,
                             0, 0) == CL_SUCCESS);
  EXPECT(await_count == 0 && transfer_outstanding);
  EXPECT(clGetEventInfo(read_event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(status), &status, 0) == CL_SUCCESS &&
         status == CL_SUBMITTED);
  EXPECT(clWaitForEvents(1, &read_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 8 && !transfer_outstanding);
  EXPECT(clGetEventInfo(read_event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(status), &status, 0) == CL_SUCCESS &&
         status == CL_COMPLETE);
  /* A completed event is not awaited again. */
  EXPECT(clWaitForEvents(1, &read_event) == CL_SUCCESS && await_count == 1);
  EXPECT(clReleaseEvent(read_event) == CL_SUCCESS);

  /* clFlush issues queued commands without waiting for them: it executes a
   * write still in the command slot and leaves a submitted read in flight. */
  await_count = flush_count = 0;
  EXPECT(clEnqueueWriteBuffer(queue, destination, CL_FALSE, 0, sizeof(value),
                              &value, 0, 0, 0) == CL_SUCCESS);
  EXPECT(transfer_outstanding && await_count == 0);
  EXPECT(clFlush(queue) == CL_SUCCESS);
  EXPECT(await_count == 1 && !transfer_outstanding);
  EXPECT(clEnqueueReadBuffer(queue, source, CL_FALSE, 0, sizeof(readback),
                             &readback, 0, 0, &read_event) == CL_SUCCESS);
  EXPECT(clFlush(queue) == CL_SUCCESS);
  EXPECT(flush_count == 1 && await_count == 1 && transfer_outstanding);
  EXPECT(clGetEventInfo(read_event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(status), &status, 0) == CL_SUCCESS &&
         status == CL_SUBMITTED);
  EXPECT(clWaitForEvents(1, &read_event) == CL_SUCCESS && await_count == 2);
  EXPECT(clReleaseEvent(read_event) == CL_SUCCESS);
  EXPECT(clFlush(0) == CL_INVALID_COMMAND_QUEUE);

  /* The command slot holds one transfer: the next transfer and a fill drain
   * a read nobody has waited for. */
  await_count = flush_count = 0;
  transfer_overlaps = 0;
  EXPECT(clEnqueueReadBuffer(queue, source, CL_FALSE, 0, sizeof(readback),
                             &readback, 0, 0, &read_event) == CL_SUCCESS);
  EXPECT(clEnqueueReadBuffer(queue, destination, CL_FALSE, 0, sizeof(other),
                             &other, 0, 0, &second_event) == CL_SUCCESS);
  EXPECT(flush_count == 2 && await_count == 1 && transfer_overlaps == 0);
  EXPECT(clEnqueueFillBuffer(queue, destination, &pattern, sizeof(pattern), 0,
                             sizeof(pattern), 0, 0, 0) == CL_SUCCESS);
  EXPECT(await_count == 2 && !transfer_outstanding && other == value);
  EXPECT(clReleaseEvent(read_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(second_event) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(destination) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(source) == CL_SUCCESS);
  return 0;
}

int main(void) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_context context_alias;
  cl_context context_alias2;
  cl_command_queue queue;
  cl_command_queue queue_alias;
  cl_command_queue queue_alias2;
  cl_program program;
  cl_kernel kernel;
  cl_event marker_event, write_event, wait_events[2], recycled_event;
  cl_event argmax_reduce_event, argmax_finalize_event, argmax_read_event;
  cl_event read_only_write_event = 0;
  cl_mem first, second, nested_buffer, nested_buffer2, static_buffer;
  cl_mem sub_parent, sub_buffer;
  cl_mem same_physical_buffer;
  cl_mem blas_input, blas_output;
  cl_mem dnn_input, dnn_filter, dnn_output, dnn_other;
  cl_mem batched_a, batched_b, batched_c;
  cl_mem m1_batched_a, m1_batched_b, m1_batched_c;
  cl_mem transpose_input_buffer, transpose_output_buffer, concat_left_buffer;
  cl_mem gather_indices_buffer;
  cl_mem cast_input_buffer, scatter_output_buffer, where_output_buffer;
  cl_mem q4_activation, q4_packed, q4_scales, q4_previous, q4_output;
  cl_mem argmax_input, argmax_partials, argmax_result;
  cl_mem capacity_buffers[32];
  cl_mem resolved;
  udonclBufferHandle_t common_buffer = {0, 0}, stale_buffer, replacement;
  udonclBufferHandle_t host_input_buffer = {0, 0};
  udonclBufferHandle_t host_output_buffer = {0, 0};
  udonclBufferView_t view;
  udonclWorkspace_t workspace;
  udononnxTensor_t tensor;
  udonblasHandle_t blas_handle = 0;
  cl_command_queue blas_stream = 0;
  udondnnHandle_t dnn_handle = 0;
  udondnnTensorDescriptor_t dnn_input_descriptor = 0;
  udondnnTensorDescriptor_t dnn_output_descriptor = 0;
  udondnnFilterDescriptor_t dnn_filter_descriptor = 0;
  udondnnConvolutionDescriptor_t dnn_convolution_descriptor = 0;
  udondnnActivationDescriptor_t dnn_activation_descriptor = 0;
  udondnnPoolingDescriptor_t dnn_pooling_descriptor = 0;
  udondnnOpTensorDescriptor_t dnn_op_tensor_descriptor = 0;
  cl_command_queue dnn_stream = 0;
  cl_int error;
  cl_int event_status;
  cl_int integer_arguments[8] = {1, 10, 16, 0, 0, 1, 16, 1};
  cl_float float_arguments[2] = {1.5f, 0.0f};
  const char *source_text = "known AOT program";
  size_t global[2] = {32, 32}, local[2] = {1, 1};
  cl_uint index, physical_token;
  cl_uint sub_parent_logical_id;
  cl_uint blas_input_token, blas_output_token;
  cl_uint dnn_input_token, dnn_filter_token, dnn_output_token, dnn_other_token;
  cl_uint dnn_temporary_token;
  cl_uint argmax_input_token, argmax_partials_token, argmax_result_token;
  cl_uint unary_input_token, unary_output_token;
  cl_uint transpose_input_token, transpose_output_token, concat_left_token;
  cl_uint gather_indices_token, cast_input_token, scatter_output_token;
  cl_uint argmax_result_value = 0xffffffffu;
  float dnn_alpha = 1.0f, dnn_beta_zero = 0.0f, dnn_beta_one = 1.0f;
  int source = 42;
  int sub_source = 91, sub_result = 0;
  cl_buffer_region sub_region = {4, sizeof(sub_source)};
  size_t bytes, byte_offset, byte_length;
  size_t elements;
  udononnxOperatorHandle_t operator_handle = 0;
  float gather_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float gather_output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int64_t gather_shape[2] = {3, 2};
  int64_t gather_indices[2] = {-1, 0};
  int64_t cast_input[4] = {0, 1, -2, 16777217};
  float cast_output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int64_t expand_input_shape[3] = {1, 2, 1};
  int64_t expand_output_shape[3] = {1, 2, 3};
  float expand_input[2] = {4.0f, 7.0f};
  float expand_output[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int64_t transpose_shape[2] = {2, 3};
  int64_t transpose_permutation[2] = {1, 0};
  int32_t transpose_input[6] = {1, 2, 3, 4, 5, 6};
  int32_t transpose_output[6] = {0, 0, 0, 0, 0, 0};
  int64_t concat_left_shape[2] = {2, 1};
  int64_t concat_right_shape[2] = {2, 2};
  int64_t concat_output_shape[2] = {2, 3};
  int32_t concat_left[2] = {1, 4};
  int32_t concat_right[4] = {2, 3, 5, 6};
  int32_t concat_output[6] = {0, 0, 0, 0, 0, 0};
  float unary_input[2] = {0.0f, 1.5707963267948966f};
  float unary_output[2] = {0.0f, 0.0f};
  int64_t slice_input_shape[2] = {2, 4};
  int64_t slice_output_shape[2] = {2, 2};
  int64_t slice_starts[1] = {1}, slice_ends[1] = {4};
  int64_t slice_axes[1] = {1}, slice_steps[1] = {2};
  int32_t slice_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t slice_output[4] = {0, 0, 0, 0};
  int64_t comparison_shape[2] = {2, 2}, scalar_shape[1] = {1};
  int64_t comparison_left[4] = {1, 2, 1, 3}, comparison_right[1] = {1};
  uint8_t comparison_output[4] = {0, 0, 0, 0};
  int64_t where_true[1] = {9}, where_false[4] = {4, 5, 6, 7};
  int64_t where_output[4] = {0, 0, 0, 0};
  int64_t add_output[4] = {0, 0, 0, 0};
  int64_t scatter_data_shape[1] = {4}, scatter_indices_shape[2] = {2, 1};
  int64_t scatter_indices[2] = {1, 3};
  int32_t scatter_data[4] = {1, 2, 3, 4}, scatter_updates[2] = {9, 8};
  int32_t scatter_output[4] = {0, 0, 0, 0};
  float nbits_a[16], nbits_scales[2] = {0.5f, 1.0f};
  float nbits_bias[2] = {1.0f, -2.0f}, nbits_y[2] = {0.0f, 0.0f};
  uint8_t nbits_b[32], nbits_zero[2] = {0x88, 0};
  uint16_t half_a[16], half_scales[2] = {0x3800, 0x3c00};
  uint16_t half_bias[2] = {0x3c00, 0xc000}, half_y[2] = {0, 0};
  uint16_t bfloat_a[16], bfloat_scales[2] = {0x3f00, 0x3f80};
  uint16_t bfloat_bias[2] = {0x3f80, 0xc000}, bfloat_y[2] = {0, 0};
  int bits;
  int64_t tensor_dimensions[2] = {2, 2};
  EXPECT(clGetPlatformIDs(1, &platform, 0) == CL_SUCCESS);
  EXPECT(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) ==
         CL_SUCCESS);
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  EXPECT(context && error == CL_SUCCESS);
  queue = clCreateCommandQueue(context, device, 0, &error);
  EXPECT(queue && error == CL_SUCCESS);

  /* The AOT runtime represents equivalent OpenCL contexts and queues with
   * singleton objects.  Independent library clients must receive retained
   * aliases so one client's teardown cannot invalidate another's handle. */
  context_alias = clCreateContext(0, 1, &device, 0, 0, &error);
  EXPECT(context_alias == context && error == CL_SUCCESS);
  queue_alias = clCreateCommandQueue(context_alias, device, 0, &error);
  EXPECT(queue_alias == queue && error == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(queue_alias) == CL_SUCCESS);
  EXPECT(clFlush(queue) == CL_SUCCESS);
  EXPECT(clReleaseContext(context_alias) == CL_SUCCESS);
  EXPECT(clFlush(queue) == CL_SUCCESS);

  EXPECT(udonclDeviceBufferCreate(context, CL_MEM_READ_WRITE, 64,
                                  &common_buffer) == UDONCL_STATUS_SUCCESS);
  stale_buffer = common_buffer;
  EXPECT(udonclDeviceBufferGetSize(common_buffer, &bytes) ==
             UDONCL_STATUS_SUCCESS &&
         bytes == 64);
  EXPECT(udonclDeviceBufferView(common_buffer, 8, 4, UDONCL_DATA_FLOAT32,
                                &view) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclBufferViewResolve(view, &resolved, &byte_offset, &byte_length) ==
             UDONCL_STATUS_SUCCESS &&
         resolved && byte_offset == 8 && byte_length == 16);
  EXPECT(udononnxCreateTensor(view, 2, tensor_dimensions, &tensor) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxGetTensorElementCount(tensor, &elements) ==
             UDONONNX_STATUS_SUCCESS &&
         elements == 4);
  EXPECT(udonclDeviceBufferView(common_buffer, 2, 1, UDONCL_DATA_FLOAT32,
                                &view) == UDONCL_STATUS_INVALID_VALUE);
  EXPECT(udonclDeviceBufferView(common_buffer, 60, 2, UDONCL_DATA_FLOAT32,
                                &view) == UDONCL_STATUS_INVALID_VALUE);
  EXPECT(udonclDeviceWorkspace(common_buffer, 24, 32, &workspace) ==
         UDONCL_STATUS_SUCCESS);
  EXPECT(udonclWorkspaceResolve(workspace, &resolved, &byte_offset,
                                &byte_length) == UDONCL_STATUS_SUCCESS &&
         byte_offset == 24 && byte_length == 32);
  EXPECT(udonclDeviceBufferDestroy(&common_buffer) == UDONCL_STATUS_SUCCESS);
  EXPECT(!common_buffer.slot && !common_buffer.generation);
  EXPECT(udonclWorkspaceResolve(workspace, &resolved, &byte_offset,
                                &byte_length) == UDONCL_STATUS_STALE_HANDLE);
  EXPECT(udononnxValidateTensor(tensor) == UDONONNX_STATUS_STALE_HANDLE);
  EXPECT(udonclDeviceBufferCreate(context, CL_MEM_READ_WRITE, 64,
                                  &replacement) == UDONCL_STATUS_SUCCESS);
  EXPECT(replacement.slot == stale_buffer.slot &&
         replacement.generation != stale_buffer.generation);
  EXPECT(udonclDeviceBufferGetRaw(stale_buffer, &resolved) ==
         UDONCL_STATUS_STALE_HANDLE);
  EXPECT(udonclDeviceBufferDestroy(&replacement) == UDONCL_STATUS_SUCCESS);

  first = clCreateBuffer(context, CL_MEM_READ_WRITE, 32, 0, &error);
  EXPECT(first && error == CL_SUCCESS);
  EXPECT(udonclDeviceBufferAdopt(context, first, 32, &common_buffer) ==
         UDONCL_STATUS_SUCCESS);
  stale_buffer = common_buffer;
  EXPECT(udonclDeviceBufferGetRaw(common_buffer, &resolved) ==
             UDONCL_STATUS_SUCCESS &&
         resolved == first);
  EXPECT(udonclDeviceBufferDestroy(&common_buffer) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferGetRaw(stale_buffer, &resolved) ==
         UDONCL_STATUS_STALE_HANDLE);

  /* CL_MEM_READ_ONLY constrains kernel access, not Host transfers. ONNX
   * inputs therefore remain writable through clEnqueueWriteBuffer. */
  first = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(source), 0, &error);
  EXPECT(first && error == CL_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, first, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0,
                              &read_only_write_event) == CL_SUCCESS);
  EXPECT(clWaitForEvents(1, &read_only_write_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(read_only_write_event) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(first) == CL_SUCCESS);

  first = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(source), 0, &error);
  EXPECT(first && error == CL_SUCCESS);
  EXPECT(allocated_logical_id >= 1 && allocated_logical_id <= 32 &&
         allocated_bytes == sizeof(source));
  EXPECT(clRetainMemObject(first) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(first) == CL_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, first, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, &write_event) == CL_SUCCESS);
  EXPECT(write_event);
  EXPECT(clGetEventInfo(write_event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(event_status), &event_status, 0) == CL_SUCCESS &&
         event_status == CL_SUBMITTED);
  EXPECT(clRetainEvent(write_event) == CL_SUCCESS);
  EXPECT(clEnqueueMarkerWithWaitList(queue, 1, &write_event, &marker_event) ==
             CL_SUCCESS &&
         marker_event);
  EXPECT(clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(clGetEventInfo(write_event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(event_status), &event_status, 0) == CL_SUCCESS &&
         event_status == CL_COMPLETE);
  EXPECT(clReleaseEvent(write_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(write_event) == CL_SUCCESS);
  EXPECT(uploaded_token == allocated_logical_id + 100);
  EXPECT(clReleaseMemObject(first) == CL_SUCCESS);
  EXPECT(released_token == allocated_logical_id + 100);

  released_token = 0;
  sub_parent = clCreateBuffer(context, CL_MEM_READ_WRITE, 16, 0, &error);
  EXPECT(sub_parent && error == CL_SUCCESS);
  sub_parent_logical_id = allocated_logical_id;
  sub_buffer = clCreateSubBuffer(sub_parent, 0, CL_BUFFER_CREATE_TYPE_REGION,
                                 &sub_region, &error);
  EXPECT(sub_buffer && error == CL_SUCCESS);
  EXPECT(!clCreateSubBuffer(sub_buffer, 0, CL_BUFFER_CREATE_TYPE_REGION,
                            &sub_region, &error) &&
         error == CL_INVALID_MEM_OBJECT);
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_device_args(
             queue, sub_buffer, sub_buffer, sub_parent, 1, 1, 1, 0, 0, 1.0f, 0,
             0, 0) == CL_INVALID_VALUE);
  EXPECT(clReleaseMemObject(sub_parent) == CL_SUCCESS);
  EXPECT(released_token == 0);
  EXPECT(clEnqueueWriteBuffer(queue, sub_buffer, CL_TRUE, 0, sizeof(sub_source),
                              &sub_source, 0, 0, 0) == CL_SUCCESS);
  EXPECT(clEnqueueReadBuffer(queue, sub_buffer, CL_TRUE, 0, sizeof(sub_result),
                             &sub_result, 0, 0, 0) == CL_SUCCESS);
  EXPECT(sub_result == sub_source);
  EXPECT(clReleaseMemObject(sub_buffer) == CL_SUCCESS);
  EXPECT(released_token == sub_parent_logical_id + 100);

  /* Reusing the logical table slot must request a new physical token rather
   * than retaining the released backing-resource identity. */
  second =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(source), 0, &error);
  EXPECT(second && error == CL_SUCCESS);
  EXPECT(allocated_logical_id == sub_parent_logical_id);

  first = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(source), 0, &error);
  EXPECT(first && error == CL_SUCCESS);
  await_count = 0;
  await_failure_call = 0;
  EXPECT(clEnqueueWriteBuffer(queue, first, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, &wait_events[0]) == CL_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, second, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, &wait_events[1]) == CL_SUCCESS);
  EXPECT(clWaitForEvents(2, wait_events) == CL_SUCCESS);
  EXPECT(await_count == 2 && awaited_events[0] == 7 && awaited_events[1] == 7);
  EXPECT(clReleaseEvent(wait_events[0]) == CL_SUCCESS);
  EXPECT(clReleaseEvent(wait_events[1]) == CL_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, first, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, &recycled_event) == CL_SUCCESS);
  EXPECT(recycled_event == wait_events[1]);
  EXPECT(clWaitForEvents(1, &recycled_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(recycled_event) == CL_SUCCESS);

  /* Consecutive non-blocking writes, as a C Host stages input_ids,
   * attention_mask and position_ids before one run, must each reach the
   * single Udon transfer slot: every later write drains the earlier one. */
  await_count = 0;
  transfer_overlaps = 0;
  EXPECT(clEnqueueWriteBuffer(queue, first, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, 0) == CL_SUCCESS);
  EXPECT(await_count == 0);
  EXPECT(clEnqueueWriteBuffer(queue, second, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, 0) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 7);
  EXPECT(clEnqueueWriteBuffer(queue, first, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, 0) == CL_SUCCESS);
  EXPECT(await_count == 2 && transfer_overlaps == 0);
  EXPECT(clFinish(queue) == CL_SUCCESS);
  EXPECT(await_count == 3 && transfer_outstanding == 0);

  await_count = 0;
  await_failure_call = 2;
  EXPECT(clEnqueueWriteBuffer(queue, first, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, &wait_events[0]) == CL_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, second, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, &wait_events[1]) == CL_SUCCESS);
  EXPECT(clWaitForEvents(2, wait_events) == CL_INVALID_VALUE);
  EXPECT(await_count == 2);
  await_failure_call = 0;
  EXPECT(clReleaseEvent(wait_events[0]) == CL_SUCCESS);
  EXPECT(clReleaseEvent(wait_events[1]) == CL_SUCCESS);

  program = clCreateProgramWithSource(context, 1, &source_text, 0, &error);
  EXPECT(program && error == CL_SUCCESS);
  EXPECT(clBuildProgram(program, 0, 0, 0, 0, 0) == CL_SUCCESS);
  kernel = clCreateKernel(program, "udonblas_sgemm", &error);
  EXPECT(kernel && error == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 0, sizeof(first), &first) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 1, sizeof(second), &second) == CL_SUCCESS);
  for (index = 0; index < 8; ++index)
    EXPECT(clSetKernelArg(kernel, index + 2, sizeof(integer_arguments[index]),
                          &integer_arguments[index]) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 10, sizeof(float_arguments[0]),
                        &float_arguments[0]) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 11, sizeof(float_arguments[1]),
                        &float_arguments[1]) == CL_SUCCESS);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 2, 0, global, local, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(dispatched_input_token == 102 && dispatched_output_token == 101);
  EXPECT(dispatched_m == 1 && dispatched_alpha == 1.5f);

  /* Replacing copied arguments for later K tiles must not consume an append-
   * only byte arena or retain the previous scalar value. */
  integer_arguments[0] = 2;
  float_arguments[0] = 2.5f;
  EXPECT(clSetKernelArg(kernel, 2, sizeof(integer_arguments[0]),
                        &integer_arguments[0]) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 10, sizeof(float_arguments[0]),
                        &float_arguments[0]) == CL_SUCCESS);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 2, 0, global, local, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(dispatched_m == 2 && dispatched_alpha == 2.5f);

  nested_buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(source), 0, &error);
  EXPECT(nested_buffer && error == CL_SUCCESS);
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_args(queue, first, second,
                                                 nested_buffer, 1, 1, 1, 0, 0,
                                                 1.0f) == CL_SUCCESS);
  EXPECT(dispatched_arena_a_token == 102 && dispatched_arena_b_token == 101 &&
         dispatched_arena_output_token == 103);
  EXPECT(clReleaseMemObject(nested_buffer) == CL_SUCCESS);
  EXPECT(clFinish(queue) == CL_SUCCESS);

  /* AOT weights are deterministic tagged cl_mem handles.  They do not consume
   * either mutable allocation slots or a runtime static-handle table. */
  allocated_logical_id = 0;
  released_token = 0;
  static_buffer = udonclAdoptStaticBuffer(
      queue, 7, 5760, __vrc_opencl_static_tensor(7, 5760), &error);
  EXPECT(static_buffer && error == CL_SUCCESS);
  EXPECT(static_resource_id == 7 && static_resource_bytes == 5760);
  EXPECT(allocated_logical_id == 0);
  EXPECT(clEnqueueWriteBuffer(queue, static_buffer, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, 0) == CL_INVALID_VALUE);
  index = (cl_uint)test_fill_buffer(context, queue, static_buffer);
  if (index)
    return (int)index;
  index = (cl_uint)test_copy_buffer(context, queue, static_buffer);
  if (index)
    return (int)index;
  index = (cl_uint)test_deferred_read(context, queue);
  if (index)
    return (int)index;
  nested_buffer = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                             sizeof(source), &error);
  EXPECT(nested_buffer && error == CL_SUCCESS);
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_args(queue, first, static_buffer,
                                                 nested_buffer, 1, 10, 144, 0,
                                                 0, 1.0f) == CL_SUCCESS);
  EXPECT(dispatched_arena_a_token == 102 &&
         dispatched_arena_b_token == 0xfffffff8u &&
         dispatched_arena_output_token == 103);
  await_count = 0;
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_device_args(
             queue, first, static_buffer, nested_buffer, 1, 1, 1, 0, 0, 1.0f, 0,
             0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event);
  EXPECT(clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 11);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_device_args(
             queue, first, static_buffer, first, 1, 1, 1, 0, 0, 1.0f, 0, 0,
             0) == CL_INVALID_OPERATION);

  /* The installed cuBLAS-shaped API receives opaque cl_mem-backed Device
   * pointers.  It binds the caller queue without creating private resources,
   * enqueues asynchronously on that queue's tail, and performs the
   * row-major/column-major operand mapping inside the library. */
  EXPECT(udonblasCreate(&blas_handle) == UDONBLAS_STATUS_SUCCESS);
  EXPECT(udonblasGetStream(blas_handle, &blas_stream) ==
         UDONBLAS_STATUS_NOT_INITIALIZED);
  EXPECT(udonblasSetStream(blas_handle, queue) == UDONBLAS_STATUS_SUCCESS);
  EXPECT(udonblasGetStream(blas_handle, &blas_stream) ==
             UDONBLAS_STATUS_SUCCESS &&
         blas_stream == queue);
  blas_input =
      clCreateBuffer(context, CL_MEM_READ_ONLY, 144 * sizeof(float), 0, &error);
  EXPECT(blas_input && error == CL_SUCCESS);
  blas_input_token = 100 + allocated_logical_id;
  blas_output =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, 10 * sizeof(float), 0, &error);
  EXPECT(blas_output && error == CL_SUCCESS);
  blas_output_token = 100 + allocated_logical_id;
  await_count = 0;
  EXPECT(udonblasSgemm(blas_handle, UDONBLAS_OP_N, UDONBLAS_OP_N, 10, 1, 144,
                       &float_arguments[0], (const float *)static_buffer, 10,
                       (const float *)blas_input, 144, &float_arguments[1],
                       (float *)blas_output, 10) == UDONBLAS_STATUS_SUCCESS);
  EXPECT(await_count == 0);
  EXPECT(dispatched_arena_a_token == blas_input_token &&
         dispatched_arena_b_token == 0xfffffff8u &&
         dispatched_arena_output_token == blas_output_token);
  released_token = 0;
  EXPECT(clReleaseMemObject(blas_input) == CL_SUCCESS);
  EXPECT(released_token == 0);
  EXPECT(clEnqueueMarkerWithWaitList(queue, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && released_token == blas_input_token);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(blas_output) == CL_SUCCESS);
  EXPECT(udonblasDestroy(blas_handle) == UDONBLAS_STATUS_SUCCESS);

  /* The cuDNN-shaped API uses caller-owned cl_mem values as device pointers.
   * beta == 0 dispatches directly into the caller's output.  beta != 0 uses
   * a temporary device buffer and enqueues an asynchronous copy-back so the
   * fragment never binds the same physical resource for read and write. */
  EXPECT(udondnnCreate(&dnn_handle) == UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnGetStream(dnn_handle, &dnn_stream) ==
         UDONDNN_STATUS_NOT_INITIALIZED);
  EXPECT(udondnnSetStream(dnn_handle, queue) == UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnGetStream(dnn_handle, &dnn_stream) == UDONDNN_STATUS_SUCCESS &&
         dnn_stream == queue);
  EXPECT(udondnnCreateTensorDescriptor(&dnn_input_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnCreateTensorDescriptor(&dnn_output_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnSetTensor4dDescriptor(dnn_input_descriptor, UDONDNN_TENSOR_NCHW,
                                      UDONDNN_DATA_FLOAT, 1, 1, 2,
                                      2) == UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnSetTensor4dDescriptor(dnn_output_descriptor,
                                      UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT,
                                      1, 1, 2, 2) == UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnCreateFilterDescriptor(&dnn_filter_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnSetFilter4dDescriptor(dnn_filter_descriptor, UDONDNN_DATA_FLOAT,
                                      UDONDNN_TENSOR_NCHW, 1, 1, 1,
                                      1) == UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnCreateConvolutionDescriptor(&dnn_convolution_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnSetConvolution2dDescriptor(dnn_convolution_descriptor, 0, 0, 1,
                                           1, 1, 1, UDONDNN_CROSS_CORRELATION,
                                           UDONDNN_DATA_FLOAT) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnCreateActivationDescriptor(&dnn_activation_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnSetActivationDescriptor(
             dnn_activation_descriptor, UDONDNN_ACTIVATION_RELU,
             UDONDNN_NOT_PROPAGATE_NAN, 0.0f) == UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnCreatePoolingDescriptor(&dnn_pooling_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnSetPooling2dDescriptor(dnn_pooling_descriptor,
                                       UDONDNN_POOLING_MAX,
                                       UDONDNN_NOT_PROPAGATE_NAN, 1, 1, 0, 0, 1,
                                       1) == UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnCreateOpTensorDescriptor(&dnn_op_tensor_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnSetOpTensorDescriptor(dnn_op_tensor_descriptor,
                                      UDONDNN_OP_TENSOR_ADD, UDONDNN_DATA_FLOAT,
                                      UDONDNN_NOT_PROPAGATE_NAN) ==
         UDONDNN_STATUS_SUCCESS);
  dnn_input =
      clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * sizeof(float), 0, &error);
  EXPECT(dnn_input && error == CL_SUCCESS);
  dnn_input_token = 100 + allocated_logical_id;
  dnn_filter =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), 0, &error);
  EXPECT(dnn_filter && error == CL_SUCCESS);
  dnn_filter_token = 100 + allocated_logical_id;
  dnn_output =
      clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * sizeof(float), 0, &error);
  EXPECT(dnn_output && error == CL_SUCCESS);
  dnn_output_token = 100 + allocated_logical_id;
  dnn_other =
      clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * sizeof(float), 0, &error);
  EXPECT(dnn_other && error == CL_SUCCESS);
  dnn_other_token = 100 + allocated_logical_id;

  await_count = 0;
  EXPECT(udondnnConvolutionForward(
             dnn_handle, &dnn_alpha, dnn_input_descriptor,
             (const float *)dnn_input, dnn_filter_descriptor,
             (const float *)dnn_filter, dnn_convolution_descriptor,
             UDONDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 0, 0, &dnn_beta_zero,
             dnn_output_descriptor,
             (float *)dnn_output) == UDONDNN_STATUS_SUCCESS);
  EXPECT(await_count == 0 && dispatched_dnn_operation == 1);
  EXPECT(dispatched_dnn_input_token == dnn_input_token &&
         dispatched_dnn_other_token == dnn_filter_token &&
         dispatched_dnn_previous_token == dnn_input_token &&
         dispatched_dnn_output_token == dnn_output_token);
  EXPECT(udondnnActivationForward(dnn_handle, dnn_activation_descriptor,
                                  &dnn_alpha, dnn_input_descriptor,
                                  (const float *)dnn_input, &dnn_beta_zero,
                                  dnn_input_descriptor, (float *)dnn_input) ==
         UDONDNN_STATUS_NOT_SUPPORTED);

  released_token = 0;
  EXPECT(udondnnActivationForward(dnn_handle, dnn_activation_descriptor,
                                  &dnn_alpha, dnn_input_descriptor,
                                  (const float *)dnn_input, &dnn_beta_one,
                                  dnn_output_descriptor, (float *)dnn_output) ==
         UDONDNN_STATUS_SUCCESS);
  dnn_temporary_token = dispatched_dnn_output_token;
  EXPECT(await_count == 0 && dispatched_dnn_operation == 2);
  EXPECT(dispatched_dnn_input_token == dnn_input_token &&
         dispatched_dnn_previous_token == dnn_output_token &&
         dnn_temporary_token != dnn_output_token && released_token == 0);
  EXPECT(clEnqueueMarkerWithWaitList(queue, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 19 &&
         released_token == dnn_temporary_token);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);

  EXPECT(udondnnPoolingForward(dnn_handle, dnn_pooling_descriptor, &dnn_alpha,
                               dnn_input_descriptor, (const float *)dnn_input,
                               &dnn_beta_zero, dnn_output_descriptor,
                               (float *)dnn_output) == UDONDNN_STATUS_SUCCESS);
  EXPECT(dispatched_dnn_operation == 3 &&
         dispatched_dnn_input_token == dnn_input_token &&
         dispatched_dnn_previous_token == dnn_input_token &&
         dispatched_dnn_output_token == dnn_output_token);
  EXPECT(udondnnOpTensor(dnn_handle, dnn_op_tensor_descriptor, &dnn_alpha,
                         dnn_input_descriptor, (const float *)dnn_input,
                         &dnn_alpha, dnn_input_descriptor,
                         (const float *)dnn_other, &dnn_beta_zero,
                         dnn_output_descriptor,
                         (float *)dnn_output) == UDONDNN_STATUS_SUCCESS);
  EXPECT(dispatched_dnn_operation == 4 &&
         dispatched_dnn_input_token == dnn_input_token &&
         dispatched_dnn_other_token == dnn_other_token &&
         dispatched_dnn_previous_token == dnn_input_token &&
         dispatched_dnn_output_token == dnn_output_token);
  EXPECT(udondnnSoftmaxForward(
             dnn_handle, UDONDNN_SOFTMAX_ACCURATE, UDONDNN_SOFTMAX_MODE_CHANNEL,
             &dnn_alpha, dnn_input_descriptor, (const float *)dnn_input,
             &dnn_beta_zero, dnn_output_descriptor,
             (float *)dnn_output) == UDONDNN_STATUS_NOT_SUPPORTED);
  EXPECT(udondnnBatchNormalizationForwardInference(
             dnn_handle, UDONDNN_BATCHNORM_PER_ACTIVATION, &dnn_alpha,
             &dnn_beta_zero, dnn_input_descriptor, (const float *)dnn_input,
             dnn_output_descriptor, (float *)dnn_output, dnn_input_descriptor,
             (const float *)dnn_other, (const float *)dnn_other,
             (const float *)dnn_other, (const float *)dnn_other,
             UDONDNN_BN_MIN_EPSILON) == UDONDNN_STATUS_NOT_SUPPORTED);

  EXPECT(udondnnDestroyOpTensorDescriptor(dnn_op_tensor_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnDestroyPoolingDescriptor(dnn_pooling_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnDestroyActivationDescriptor(dnn_activation_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnDestroyConvolutionDescriptor(dnn_convolution_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnDestroyFilterDescriptor(dnn_filter_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnDestroyTensorDescriptor(dnn_output_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnDestroyTensorDescriptor(dnn_input_descriptor) ==
         UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnDestroy(dnn_handle) == UDONDNN_STATUS_SUCCESS);
  EXPECT(clReleaseMemObject(dnn_other) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(dnn_output) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(dnn_filter) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(dnn_input) == CL_SUCCESS);
  EXPECT(clFinish(queue) == CL_SUCCESS);

  /* The tagged representation has no fixed resource-count ceiling. */
  same_physical_buffer = udonclAdoptStaticBuffer(
      queue, 1024, sizeof(source),
      __vrc_opencl_static_tensor(1024, sizeof(source)), &error);
  EXPECT(same_physical_buffer && error == CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(same_physical_buffer, &physical_token) ==
         CL_SUCCESS);
  EXPECT(physical_token == 0xfffffbffu);
  /* Read/write aliasing is still rejected when a tagged static handle is
   * accidentally selected as both an input and an output. */
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_device_args(
             queue, same_physical_buffer, static_buffer, same_physical_buffer,
             1, 1, 1, 0, 0, 1.0f, 0, 0, 0) == CL_INVALID_OPERATION);
  EXPECT(clReleaseMemObject(same_physical_buffer) == CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(first, &unary_input_token) == CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(nested_buffer, &unary_output_token) ==
         CL_SUCCESS);
  ((float *)physical_buffers[unary_input_token - 100])[0] = 0.0f;
  ((float *)physical_buffers[unary_input_token - 100])[1] = 1.57079632679f;
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_unary_device_args(queue, first, nested_buffer,
                                                  2, 2, 0, 0,
                                                  &marker_event) == CL_SUCCESS);
  EXPECT(marker_event);
  EXPECT(clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 17);
  EXPECT(((float *)physical_buffers[unary_output_token - 100])[0] == 0.0f);
  EXPECT(((float *)physical_buffers[unary_output_token - 100])[1] > 0.999999f &&
         ((float *)physical_buffers[unary_output_token - 100])[1] < 1.000001f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  transpose_input_buffer = udonclCreateBufferForQueue(
      queue, CL_MEM_READ_WRITE, 6 * sizeof(float), &error);
  EXPECT(transpose_input_buffer && error == CL_SUCCESS);
  transpose_output_buffer = udonclCreateBufferForQueue(
      queue, CL_MEM_READ_WRITE, 6 * sizeof(float), &error);
  EXPECT(transpose_output_buffer && error == CL_SUCCESS);
  concat_left_buffer = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                                  2 * sizeof(float), &error);
  EXPECT(concat_left_buffer && error == CL_SUCCESS);
  gather_indices_buffer = udonclCreateBufferForQueue(
      queue, CL_MEM_READ_WRITE, 2 * sizeof(cl_uint), &error);
  EXPECT(gather_indices_buffer && error == CL_SUCCESS);
  cast_input_buffer = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                                 4 * sizeof(cl_uint), &error);
  EXPECT(cast_input_buffer && error == CL_SUCCESS);
  scatter_output_buffer = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                                     6 * sizeof(float), &error);
  EXPECT(scatter_output_buffer && error == CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(transpose_input_buffer,
                                     &transpose_input_token) == CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(transpose_output_buffer,
                                     &transpose_output_token) == CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(concat_left_buffer, &concat_left_token) ==
         CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(gather_indices_buffer,
                                     &gather_indices_token) == CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(cast_input_buffer, &cast_input_token) ==
         CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(scatter_output_buffer,
                                     &scatter_output_token) == CL_SUCCESS);
  ((float *)physical_buffers[transpose_input_token - 100])[0] = 1.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[1] = 2.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[2] = 3.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[3] = 4.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[4] = 5.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[5] = 6.0f;
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_transpose_device_args(
             queue, transpose_input_buffer, transpose_output_buffer, 6, 2,
             2 | (3 << 16), 1 | (1 << 16), 1, 1, 0, 0,
             &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 21);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[0] == 1.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[1] == 4.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[2] == 2.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[3] == 5.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[4] == 3.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[5] == 6.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  for (index = 0; index < 6; ++index)
    ((float *)physical_buffers[transpose_input_token - 100])[index] =
        (float)(index + 1);
  ((cl_uint *)physical_buffers[cast_input_token - 100])[0] = 1u;
  ((cl_uint *)physical_buffers[cast_input_token - 100])[1] = 0xffffffffu;
  ((float *)physical_buffers[transpose_output_token - 100])[0] = 10.0f;
  ((float *)physical_buffers[transpose_output_token - 100])[1] = 11.0f;
  ((float *)physical_buffers[transpose_output_token - 100])[2] = 12.0f;
  ((float *)physical_buffers[transpose_output_token - 100])[3] = 20.0f;
  ((float *)physical_buffers[transpose_output_token - 100])[4] = 21.0f;
  ((float *)physical_buffers[transpose_output_token - 100])[5] = 22.0f;
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_scatter_nd_f32_device_args(
             queue, transpose_input_buffer, cast_input_buffer,
             transpose_output_buffer, scatter_output_buffer, 6, 2,
             2 | (3 << 16), 1 | (1 << 16), 1, 2, 3, 1, 0, 0,
             &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 29);
  EXPECT(((float *)physical_buffers[scatter_output_token - 100])[0] == 1.0f);
  EXPECT(((float *)physical_buffers[scatter_output_token - 100])[2] == 3.0f);
  EXPECT(((float *)physical_buffers[scatter_output_token - 100])[3] == 20.0f);
  EXPECT(((float *)physical_buffers[scatter_output_token - 100])[5] == 22.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(scatter_output_buffer) == CL_SUCCESS);
  where_output_buffer = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                                   6 * sizeof(float), &error);
  EXPECT(where_output_buffer && error == CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(where_output_buffer,
                                     &scatter_output_token) == CL_SUCCESS);
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_slice_device_args(
             queue, transpose_input_buffer, transpose_output_buffer, 4, 2,
             2 | (3 << 16), 1 | (1 << 16), 1, 2 | (2 << 16), 1 | (1 << 16), 1,
             0, 0, 0, 1 | (2 << 16), 1 | (1 << 16), 1, 0, 0,
             &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 22);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[0] == 1.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[1] == 3.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[2] == 4.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[3] == 6.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  ((float *)physical_buffers[transpose_input_token - 100])[0] = 4.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[1] = 7.0f;
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_expand_device_args(
             queue, transpose_input_buffer, transpose_output_buffer, 6, 3,
             1 | (2 << 16), 1 | (1 << 16), 1, 3, 1 | (2 << 16), 3 | (1 << 16),
             1, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 23);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[0] == 4.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[1] == 4.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[2] == 4.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[3] == 7.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[4] == 7.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[5] == 7.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  ((float *)physical_buffers[concat_left_token - 100])[0] = 1.0f;
  ((float *)physical_buffers[concat_left_token - 100])[1] = 2.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[0] = 3.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[1] = 4.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[2] = 5.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[3] = 6.0f;
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_concat2_device_args(
             queue, concat_left_buffer, transpose_input_buffer,
             transpose_output_buffer, 6, 1, 2, 4, 0, 0,
             &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 24);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[0] == 1.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[1] == 2.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[2] == 3.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[3] == 4.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[4] == 5.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[5] == 6.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  ((float *)physical_buffers[transpose_input_token - 100])[0] = 10.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[1] = 11.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[2] = 20.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[3] = 21.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[4] = 30.0f;
  ((float *)physical_buffers[transpose_input_token - 100])[5] = 31.0f;
  ((cl_uint *)physical_buffers[gather_indices_token - 100])[0] = 2u;
  ((cl_uint *)physical_buffers[gather_indices_token - 100])[1] = 0u;
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_gather_axis0_device_args(
             queue, transpose_input_buffer, gather_indices_buffer,
             transpose_output_buffer, 4, 2, 3, 1, 0, 0,
             &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 25);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[0] == 30.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[1] == 31.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[2] == 10.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[3] == 11.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  ((cl_uint *)physical_buffers[cast_input_token - 100])[0] = 7u;
  ((cl_uint *)physical_buffers[cast_input_token - 100])[1] = 0u;
  ((cl_uint *)physical_buffers[cast_input_token - 100])[2] = 0xfffffffeu;
  ((cl_uint *)physical_buffers[cast_input_token - 100])[3] = 0xffffffffu;
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_cast_i64_f32_device_args(
             queue, cast_input_buffer, transpose_output_buffer, 2, 0, 0,
             &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 26);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[0] == 7.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[1] == -2.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  ((float *)physical_buffers[concat_left_token - 100])[0] = 0.0f;
  ((float *)physical_buffers[concat_left_token - 100])[1] = 2.0f;
  ((float *)physical_buffers[cast_input_token - 100])[0] = 0.0f;
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_equal_f32_device_args(
             queue, concat_left_buffer, cast_input_buffer,
             transpose_output_buffer, 6, 2 | (2 << 6), 2 | (1 << 16),
             1 | (1 << 16), 1 | (1 << 16), 1 | (1 << 16), 2 | (3 << 16),
             1 | (1 << 16), 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 27);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[0] == 1.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[2] == 1.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[3] == 0.0f);
  EXPECT(((float *)physical_buffers[transpose_output_token - 100])[5] == 0.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  ((float *)physical_buffers[concat_left_token - 100])[0] = 9.0f;
  for (index = 0; index < 6; ++index)
    ((float *)physical_buffers[transpose_input_token - 100])[index] =
        (float)(index + 1);
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_where_f32_device_args(
             queue, transpose_output_buffer, concat_left_buffer,
             transpose_input_buffer, where_output_buffer, 6,
             2 | (2 << 6) | (2 << 9), 2 | (3 << 16), 1 | (1 << 16),
             1 | (1 << 16), 1 | (1 << 16), 2 | (3 << 16), 1 | (1 << 16),
             2 | (3 << 16), 1 | (1 << 16), 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 28);
  EXPECT(((float *)physical_buffers[scatter_output_token - 100])[0] == 9.0f);
  EXPECT(((float *)physical_buffers[scatter_output_token - 100])[2] == 9.0f);
  EXPECT(((float *)physical_buffers[scatter_output_token - 100])[3] == 4.0f);
  EXPECT(((float *)physical_buffers[scatter_output_token - 100])[5] == 6.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(where_output_buffer) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(cast_input_buffer) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(gather_indices_buffer) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(concat_left_buffer) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(transpose_output_buffer) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(transpose_input_buffer) == CL_SUCCESS);

  transpose_input_buffer = udonclCreateBufferForQueue(
      queue, CL_MEM_READ_WRITE, 131072 * sizeof(float), &error);
  EXPECT(transpose_input_buffer && error == CL_SUCCESS);
  transpose_output_buffer = udonclCreateBufferForQueue(
      queue, CL_MEM_READ_WRITE, 131072 * sizeof(float), &error);
  EXPECT(transpose_output_buffer && error == CL_SUCCESS);
  concat_left_buffer = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                                  sizeof(float), &error);
  EXPECT(concat_left_buffer && error == CL_SUCCESS);
  gather_indices_buffer = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                                     sizeof(float), &error);
  EXPECT(gather_indices_buffer && error == CL_SUCCESS);
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_transpose_device_args(
             queue, transpose_input_buffer, transpose_output_buffer, 131072, 4,
             1 | (32 << 16), 64 | (64 << 16), 1, 1224, 0, 0,
             &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 21);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_expand_device_args(
             queue, transpose_input_buffer, transpose_output_buffer, 131072, 5,
             1 | (4 << 16), 1 | (64 << 16), 64, 5, 1 | (4 << 16),
             8 | (64 << 16), 64, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 23);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  await_count = 0;
  EXPECT(__vrc_enqueue_udondnn_large_op_tensor_device_args(
             queue, transpose_input_buffer, concat_left_buffer,
             gather_indices_buffer, transpose_output_buffer, 131072,
             1 | (32 << 16), 64 | (64 << 16), 1 | (1 << 16), 1 | (1 << 16),
             1 | (32 << 16), 64 | (64 << 16), 1, 1.0f, 1.0f, 0.0f, 0, 0,
             &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 33);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(gather_indices_buffer) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(concat_left_buffer) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(transpose_output_buffer) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(transpose_input_buffer) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(nested_buffer) == CL_SUCCESS);

  batched_a = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                         8 * sizeof(float), &error);
  EXPECT(batched_a && error == CL_SUCCESS);
  batched_b = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                         4 * sizeof(float), &error);
  EXPECT(batched_b && error == CL_SUCCESS);
  batched_c = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                         8 * sizeof(float), &error);
  EXPECT(batched_c && error == CL_SUCCESS);
  await_count = 0;
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_strided_batched_device_args(
             queue, batched_a, batched_b, batched_c, 2, 2, 2, 4, 0, 4, 2, 0, 0,
             1.0f, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 20);
  EXPECT(dispatched_arena_stride_a == 4 && dispatched_arena_stride_b == 0 &&
         dispatched_arena_batch_count == 2);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(batched_c) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(batched_b) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(batched_a) == CL_SUCCESS);

  m1_batched_a = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                            2048 * sizeof(float), &error);
  EXPECT(m1_batched_a && error == CL_SUCCESS);
  m1_batched_b = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                            131072 * sizeof(float), &error);
  EXPECT(m1_batched_b && error == CL_SUCCESS);
  m1_batched_c = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                            2048 * sizeof(float), &error);
  EXPECT(m1_batched_c && error == CL_SUCCESS);
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_strided_batched_device_args(
             queue, m1_batched_a, m1_batched_b, m1_batched_c, 1, 64, 64, 64,
             4096, 64, 32, 0, 0, 1.0f, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(dispatched_arena_stride_a == 64 && dispatched_arena_stride_b == 4096 &&
         dispatched_arena_batch_count == 32);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_strided_batched_device_args(
             queue, m1_batched_a, m1_batched_b, m1_batched_c, 1, 64, 64, 64,
             4096, 64, 33, 0, 0, 1.0f, 0, 0, 0) == CL_INVALID_VALUE);
  /* K beyond one 256-wide chunk is accepted up to 8192. */
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_strided_batched_device_args(
             queue, m1_batched_b, m1_batched_b, m1_batched_c, 1, 8, 8192, 0, 0,
             8, 1, 0, 0, 1.0f, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(dispatched_arena_batch_count == 1);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_strided_batched_device_args(
             queue, m1_batched_a, m1_batched_b, m1_batched_c, 1, 1, 8193, 0, 0,
             1, 1, 0, 0, 1.0f, 0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_device_args(
             queue, m1_batched_a, m1_batched_b, m1_batched_c, 1, 1, 2048, 0, 0,
             1.0f, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_device_args(
             queue, m1_batched_a, m1_batched_b, m1_batched_c, 1, 1, 8193, 0, 0,
             1.0f, 0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(clReleaseMemObject(m1_batched_c) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(m1_batched_b) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(m1_batched_a) == CL_SUCCESS);

  q4_activation = udonclAdoptStaticBuffer(
      queue, 8, 90112, __vrc_opencl_static_tensor(8, 90112), &error);
  EXPECT(q4_activation && error == CL_SUCCESS);
  q4_packed = udonclAdoptStaticBuffer(
      queue, 9, 90112000, __vrc_opencl_static_tensor(9, 90112000), &error);
  EXPECT(q4_packed && error == CL_SUCCESS);
  q4_scales = udonclAdoptStaticBuffer(
      queue, 10, 22528000, __vrc_opencl_static_tensor(10, 22528000), &error);
  EXPECT(q4_scales && error == CL_SUCCESS);
  q4_previous =
      udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE, 524288, &error);
  EXPECT(q4_previous && error == CL_SUCCESS);
  q4_output =
      udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE, 524288, &error);
  EXPECT(q4_output && error == CL_SUCCESS);
  /* The whole K (176 blocks) in one dispatch. */
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             4, 32000, 5632, 176, 1, 0, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(dispatched_q4_block_count == 176 && dispatched_q4_splits == 1 &&
         dispatched_q4_previous_partials == 0);
  /* A split-K partial pass: 16 planes of 11 blocks over 2 x 2048 words. */
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             2, 2048, 5632, 11, 16, 0, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(dispatched_q4_block_count == 11 && dispatched_q4_splits == 16 &&
         dispatched_q4_previous_partials == 0);
  /* Its ordered sum pass. */
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             2, 2048, 5632, 0, 1, 16, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(dispatched_q4_block_count == 0 && dispatched_q4_splits == 1 &&
         dispatched_q4_previous_partials == 16);
  /* Splits that leave K uncovered or end empty, slices longer than one
   * fragment covers, too many planes for the surface or the kernel, a sum
   * pass with several planes or nothing to sum, and negative counts are
   * rejected. */
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             4, 32000, 5632, 175, 1, 0, 0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             1, 2048, 5632, 11, 17, 0, 0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             1, 2048, 5632, 257, 1, 0, 0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             4, 32000, 5632, 88, 2, 0, 0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             1, 1, 5632, 2, 88, 0, 0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             1, 1, 5632, 0, 2, 2, 0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             1, 1, 5632, 0, 1, 0, 0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             1, 1, 5632, -1, 1, 0, 0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             1, 1, 5632, 176, 1, -1, 0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(clReleaseMemObject(q4_output) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(q4_previous) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(q4_scales) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(q4_packed) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(q4_activation) == CL_SUCCESS);
  released_token = 0;
  EXPECT(clReleaseMemObject(static_buffer) == CL_SUCCESS);
  EXPECT(released_token == 0);

  EXPECT(clReleaseKernel(kernel) == CL_SUCCESS);
  kernel = clCreateKernel(program, "udonblas_sgemv", &error);
  EXPECT(kernel && error == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 0, sizeof(first), &first) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 1, sizeof(second), &second) == CL_SUCCESS);
  for (index = 0; index < 6; ++index)
    EXPECT(clSetKernelArg(kernel, index + 2, sizeof(integer_arguments[index]),
                          &integer_arguments[index]) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 8, sizeof(float_arguments[0]),
                        &float_arguments[0]) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 9, sizeof(float_arguments[1]),
                        &float_arguments[1]) == CL_SUCCESS);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, 0, global, local, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(dispatched_input_token == 102 && dispatched_output_token == 101);
  EXPECT(dispatched_sgemv_trans == 2 && dispatched_alpha == 2.5f);

  EXPECT(clReleaseKernel(kernel) == CL_SUCCESS);
  kernel = clCreateKernel(program, "mnist_conv", &error);
  EXPECT(kernel && error == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 0, sizeof(first), &first) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 1, sizeof(second), &second) == CL_SUCCESS);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, 0, global, local, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(dispatched_mnist_kernel == 20 && dispatched_input_token == 102 &&
         dispatched_output_token == 101);

  EXPECT(clReleaseKernel(kernel) == CL_SUCCESS);
  kernel = clCreateKernel(program, "mnist_pool", &error);
  EXPECT(kernel && error == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 0, sizeof(first), &first) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 1, sizeof(second), &second) == CL_SUCCESS);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, 0, global, local, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(dispatched_mnist_kernel == 21);

  EXPECT(clReleaseKernel(kernel) == CL_SUCCESS);
  kernel = clCreateKernel(program, "mnist_dense", &error);
  EXPECT(kernel && error == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 0, sizeof(first), &first) == CL_SUCCESS);
  EXPECT(clSetKernelArg(kernel, 1, sizeof(second), &second) == CL_SUCCESS);
  EXPECT(clEnqueueNDRangeKernel(queue, kernel, 1, 0, global, local, 0, 0, 0) ==
         CL_SUCCESS);
  EXPECT(dispatched_mnist_kernel == 22);

  EXPECT(clReleaseKernel(kernel) == CL_SUCCESS);
  EXPECT(clReleaseProgram(program) == CL_SUCCESS);
  released_token = 0;
  EXPECT(clReleaseMemObject(first) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(second) == CL_SUCCESS);
  EXPECT(released_token == 0);
  EXPECT(clFinish(queue) == CL_SUCCESS);
  EXPECT(released_token == 101);
  for (index = 0; index < 16; ++index) {
    nbits_a[index] = 1.0f;
    half_a[index] = 0x3c00;
    bfloat_a[index] = 0x3f80;
  }
  EXPECT(udonclDeviceBufferCreate(context, CL_MEM_READ_ONLY, 64,
                                  &host_input_buffer) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferCreate(context, CL_MEM_WRITE_ONLY, 64,
                                  &host_output_buffer) ==
         UDONCL_STATUS_SUCCESS);
  context_alias = clCreateContext(0, 1, &device, 0, 0, &error);
  EXPECT(context_alias == context && error == CL_SUCCESS);
  queue_alias = clCreateCommandQueue(context_alias, device, 0, &error);
  EXPECT(queue_alias == queue && error == CL_SUCCESS);
  nested_buffer =
      clCreateBuffer(context_alias, CL_MEM_READ_WRITE, 64, 0, &error);
  EXPECT(nested_buffer && error == CL_SUCCESS);
  context_alias2 = clCreateContext(0, 1, &device, 0, 0, &error);
  EXPECT(context_alias2 == context && error == CL_SUCCESS);
  queue_alias2 = clCreateCommandQueue(context_alias2, device, 0, &error);
  EXPECT(queue_alias2 == queue && error == CL_SUCCESS);
  nested_buffer2 =
      clCreateBuffer(context_alias2, CL_MEM_READ_WRITE, 64, 0, &error);
  EXPECT(nested_buffer2 && error == CL_SUCCESS);
  EXPECT(clReleaseMemObject(nested_buffer) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(queue_alias) == CL_SUCCESS);
  EXPECT(clReleaseContext(context_alias) == CL_SUCCESS);
  EXPECT(clFlush(queue) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(nested_buffer2) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(queue_alias2) == CL_SUCCESS);
  EXPECT(clReleaseContext(context_alias2) == CL_SUCCESS);
  EXPECT(clFlush(queue) == CL_SUCCESS);

  argmax_input = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                            32016u * sizeof(float), &error);
  EXPECT(argmax_input && error == CL_SUCCESS);
  argmax_partials = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                               512u * sizeof(cl_uint), &error);
  EXPECT(argmax_partials && error == CL_SUCCESS);
  argmax_result = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                             sizeof(cl_uint), &error);
  EXPECT(argmax_result && error == CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(argmax_input, &argmax_input_token) ==
         CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(argmax_partials, &argmax_partials_token) ==
         CL_SUCCESS);
  EXPECT(udonclResolvePhysicalBuffer(argmax_result, &argmax_result_token) ==
         CL_SUCCESS);
  for (index = 0; index < 32000u; ++index)
    ((float *)physical_buffers[argmax_input_token - 100])[16u + index] =
        -(float)index;
  ((float *)physical_buffers[argmax_input_token - 100])[16u + 29958u] = 4.0f;
  EXPECT(__vrc_enqueue_udonaten_argmax_reduce_device_args(
             queue, argmax_input, argmax_partials, 16, 32000, 0, 0,
             &argmax_reduce_event) == CL_SUCCESS);
  EXPECT(__vrc_enqueue_udonaten_argmax_finalize_device_args(
             queue, argmax_input, argmax_partials, argmax_result, 16, 32000, 1,
             1, &argmax_reduce_event, &argmax_finalize_event) == CL_SUCCESS);
  EXPECT(clEnqueueReadBuffer(queue, argmax_result, CL_FALSE, 0,
                             sizeof(argmax_result_value), &argmax_result_value,
                             1, &argmax_finalize_event,
                             &argmax_read_event) == CL_SUCCESS);
  EXPECT(clWaitForEvents(1, &argmax_read_event) == CL_SUCCESS);
  EXPECT(argmax_result_value == 29958u);
  EXPECT(__vrc_enqueue_udonaten_argmax_reduce_device_args(
             queue, argmax_input, argmax_partials, -1, 32000, 0, 0, 0) ==
         CL_INVALID_VALUE);
  EXPECT(__vrc_enqueue_udonaten_argmax_reduce_device_args(
             queue, argmax_input, argmax_partials, 17, 32000, 0, 0, 0) ==
         CL_INVALID_BUFFER_SIZE);
  EXPECT(__vrc_enqueue_udonaten_argmax_finalize_device_args(
             queue, argmax_input, argmax_partials, argmax_result, 16, 32000, 2,
             0, 0, 0) == CL_INVALID_VALUE);
  EXPECT(__vrc_enqueue_udonaten_argmax_finalize_device_args(
             queue, argmax_input, argmax_partials, argmax_input, 16, 32000, 1,
             0, 0, 0) == CL_INVALID_OPERATION);
  EXPECT(clReleaseEvent(argmax_read_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(argmax_finalize_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(argmax_reduce_event) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(argmax_result) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(argmax_partials) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(argmax_input) == CL_SUCCESS);

  EXPECT(udononnxCreateOperatorHandle(&operator_handle) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxGather(operator_handle, UDONCL_DATA_FLOAT32, gather_data,
                        gather_shape, 2, UDONCL_DATA_INT64, gather_indices, 2,
                        0, gather_output) == UDONONNX_STATUS_SUCCESS);
  EXPECT(gather_output[0] == 5.0f && gather_output[1] == 6.0f &&
         gather_output[2] == 1.0f && gather_output[3] == 2.0f);
  EXPECT(dispatched_gather_words == 2);
  EXPECT(udononnxCast(operator_handle, UDONCL_DATA_INT64, cast_input,
                      UDONCL_DATA_FLOAT32, cast_output,
                      4) == UDONONNX_STATUS_SUCCESS);
  EXPECT(cast_output[0] == 0.0f && cast_output[1] == 1.0f &&
         cast_output[2] == -2.0f && cast_output[3] == 16777216.0f);
  EXPECT(udononnxExpand(operator_handle, UDONCL_DATA_FLOAT32, expand_input,
                        expand_input_shape, 3, expand_output,
                        expand_output_shape, 3) == UDONONNX_STATUS_SUCCESS);
  EXPECT(expand_output[0] == 4.0f && expand_output[1] == 4.0f &&
         expand_output[2] == 4.0f && expand_output[3] == 7.0f &&
         expand_output[4] == 7.0f && expand_output[5] == 7.0f);
  EXPECT(udononnxTranspose(operator_handle, UDONCL_DATA_INT32, transpose_input,
                           transpose_shape, 2, transpose_permutation,
                           transpose_output) == UDONONNX_STATUS_SUCCESS);
  EXPECT(transpose_output[0] == 1 && transpose_output[1] == 4 &&
         transpose_output[2] == 2 && transpose_output[3] == 5 &&
         transpose_output[4] == 3 && transpose_output[5] == 6);
  EXPECT(udononnxConcat(operator_handle, UDONCL_DATA_INT32, concat_left,
                        concat_left_shape, 2, 1, 0, concat_output,
                        concat_output_shape) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxConcat(operator_handle, UDONCL_DATA_INT32, concat_right,
                        concat_right_shape, 2, 1, 1, concat_output,
                        concat_output_shape) == UDONONNX_STATUS_SUCCESS);
  EXPECT(concat_output[0] == 1 && concat_output[1] == 2 &&
         concat_output[2] == 3 && concat_output[3] == 4 &&
         concat_output[4] == 5 && concat_output[5] == 6);
  EXPECT(udononnxCos(operator_handle, unary_input, 2, unary_output) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(unary_output[0] == 1.0f && unary_output[1] > -0.000001f &&
         unary_output[1] < 0.000001f);
  EXPECT(udononnxSin(operator_handle, unary_input, 2, unary_output) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(unary_output[0] == 0.0f && unary_output[1] > 0.999999f &&
         unary_output[1] < 1.000001f);
  EXPECT(udononnxNeg(operator_handle, unary_input, 2, unary_output) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(unary_output[0] == 0.0f && unary_output[1] < -1.570796f &&
         unary_output[1] > -1.570797f);
  EXPECT(udononnxSlice(operator_handle, UDONCL_DATA_INT32, slice_input,
                       slice_input_shape, 2, slice_starts, slice_ends,
                       slice_axes, slice_steps, 1, slice_output,
                       slice_output_shape) == UDONONNX_STATUS_SUCCESS);
  EXPECT(slice_output[0] == 2 && slice_output[1] == 4 && slice_output[2] == 6 &&
         slice_output[3] == 8);
  EXPECT(udononnxEqual(operator_handle, UDONCL_DATA_INT64, comparison_left,
                       comparison_shape, 2, comparison_right, scalar_shape, 1,
                       comparison_output, comparison_shape,
                       2) == UDONONNX_STATUS_SUCCESS);
  EXPECT(comparison_output[0] == 1 && comparison_output[1] == 0 &&
         comparison_output[2] == 1 && comparison_output[3] == 0);
  EXPECT(udononnxWhere(operator_handle, UDONCL_DATA_INT64, comparison_output,
                       comparison_shape, 2, where_true, scalar_shape, 1,
                       where_false, comparison_shape, 2, where_output,
                       comparison_shape, 2) == UDONONNX_STATUS_SUCCESS);
  EXPECT(where_output[0] == 9 && where_output[1] == 5 && where_output[2] == 9 &&
         where_output[3] == 7);
  EXPECT(udononnxAdd(operator_handle, UDONCL_DATA_INT64, comparison_left,
                     comparison_shape, 2, comparison_right, scalar_shape, 1,
                     add_output, comparison_shape,
                     2) == UDONONNX_STATUS_SUCCESS);
  EXPECT(add_output[0] == 2 && add_output[1] == 3 && add_output[2] == 2 &&
         add_output[3] == 4);
  EXPECT(udononnxScatterND(operator_handle, UDONCL_DATA_INT32, scatter_data,
                           scatter_data_shape, 1, UDONCL_DATA_INT64,
                           scatter_indices, scatter_indices_shape, 2,
                           scatter_updates,
                           scatter_output) == UDONONNX_STATUS_SUCCESS);
  EXPECT(scatter_output[0] == 1 && scatter_output[1] == 9 &&
         scatter_output[2] == 3 && scatter_output[3] == 8);
  for (bits = 2; bits <= 8; ++bits) {
    size_t packed_bytes = (size_t)2 * 16u * (size_t)bits / 8u;
    unsigned zero = 1u << ((unsigned)bits - 1u);
    memset(nbits_b, 0, sizeof(nbits_b));
    for (index = 0; index < 16; ++index) {
      pack_nbits(nbits_b, packed_bytes, index, zero + 1u, bits);
      pack_nbits(nbits_b, packed_bytes, 16u + index, zero - 1u, bits);
    }
    EXPECT(udononnxMatMulNBits(operator_handle, UDONCL_DATA_FLOAT32, nbits_a, 1,
                               16, nbits_b, packed_bytes, nbits_scales, 2, 0, 0,
                               UDONONNX_ZERO_POINT_NONE, 0, 0, nbits_bias, 2, 2,
                               16, bits, nbits_y) == UDONONNX_STATUS_SUCCESS);
    EXPECT(nbits_y[0] == 9.0f && nbits_y[1] == -18.0f);
  }
  memset(nbits_b, 0, sizeof(nbits_b));
  for (index = 0; index < 16; ++index) {
    pack_nbits(nbits_b, 16, index, 9, 4);
    pack_nbits(nbits_b, 16, 16u + index, 7, 4);
  }
  EXPECT(udononnxMatMulNBits(operator_handle, UDONCL_DATA_FLOAT16, half_a, 1,
                             16, nbits_b, 16, half_scales, 2, nbits_zero, 1,
                             UDONONNX_ZERO_POINT_PACKED, 0, 0, half_bias, 2, 2,
                             16, 4, half_y) == UDONONNX_STATUS_SUCCESS);
  EXPECT(half_y[0] == 0x4880 && half_y[1] == 0xcc80);
  nbits_zero[0] = 8;
  nbits_zero[1] = 8;
  EXPECT(udononnxMatMulNBits(operator_handle, UDONCL_DATA_BFLOAT16, bfloat_a, 1,
                             16, nbits_b, 16, bfloat_scales, 2, nbits_zero, 2,
                             UDONONNX_ZERO_POINT_UNPACKED, 0, 0, bfloat_bias, 2,
                             2, 16, 4, bfloat_y) == UDONONNX_STATUS_SUCCESS);
  EXPECT(bfloat_y[0] == 0x4110 && bfloat_y[1] == 0xc190);
  EXPECT(dispatched_nbits_tile_n == 2 && dispatched_nbits_finalize == 1);
  EXPECT(udononnxDestroyOperatorHandle(operator_handle) ==
         UDONONNX_STATUS_SUCCESS);
  /* Nested ONNX operator clients repeatedly create and release aliases of the
   * AOT singleton context and queue.  The outer host must still be able to
   * destroy its device buffers and finally release its queue and context. */
  EXPECT(udonclDeviceBufferDestroy(&host_input_buffer) ==
         UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferDestroy(&host_output_buffer) ==
         UDONCL_STATUS_SUCCESS);
  for (index = 0; index < 32; ++index) {
    capacity_buffers[index] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, 4, 0, &error);
    EXPECT(capacity_buffers[index] && error == CL_SUCCESS);
  }
  EXPECT(!clCreateBuffer(context, CL_MEM_READ_WRITE, 4, 0, &error) &&
         error == CL_OUT_OF_RESOURCES);
  EXPECT(clReleaseMemObject(capacity_buffers[11]) == CL_SUCCESS);
  capacity_buffers[11] =
      clCreateBuffer(context, CL_MEM_READ_WRITE, 4, 0, &error);
  EXPECT(capacity_buffers[11] && error == CL_SUCCESS &&
         allocated_logical_id == 12);
  for (index = 0; index < 32; ++index)
    EXPECT(clReleaseMemObject(capacity_buffers[index]) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(queue) == CL_SUCCESS);
  EXPECT(clReleaseContext(context) == CL_SUCCESS);
  EXPECT(transfer_overlaps == 0);
  return 0;
}
