#include <CL/cl.h>
#include <stddef.h>
#include <string.h>
#include <udoncl/device_memory.h>
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
static cl_int dispatched_gather_words;
static cl_uint await_count;
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
  return resource_id + 1000;
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
  memcpy(physical_buffers[physical_token - 100] + offset, source, bytes);
  return 7;
}

cl_int __vrc_opencl_buffer_readback(cl_uint physical_token, size_t offset,
                                    size_t bytes, void *destination) {
  (void)physical_token;
  (void)offset;
  (void)bytes;
  memcpy(destination, physical_buffers[physical_token - 100] + offset, bytes);
  return 8;
}

cl_int __vrc_opencl_buffer_copy(cl_uint source_token, cl_uint destination_token,
                                size_t source_offset, size_t destination_offset,
                                size_t bytes) {
  memmove(physical_buffers[destination_token - 100] + destination_offset,
          physical_buffers[source_token - 100] + source_offset, bytes);
  return 19;
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
extern cl_int __vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
    cl_command_queue queue, cl_mem activation, cl_mem packed_weight,
    cl_mem scales, cl_mem previous, cl_mem output, cl_int m, cl_int n, cl_int k,
    cl_int block_size, cl_int n_base, cl_int k_base, cl_int tile_n,
    cl_int tile_k, cl_int accumulate, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event);
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
    cl_int block_size, cl_int n_base, cl_int k_base, cl_int tile_n,
    cl_int tile_k, cl_int accumulate) {
  (void)activation_token;
  (void)packed_weight_token;
  (void)scales_token;
  (void)previous_token;
  (void)output_token;
  (void)m;
  (void)n;
  (void)k;
  (void)block_size;
  (void)n_base;
  (void)k_base;
  (void)tile_n;
  (void)tile_k;
  (void)accumulate;
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
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < 16384;
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
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < 16384;
       ++udononnx_reindex_cpu_gid)
    udononnx_expand_f32((const float *)physical_buffers[input_token - 100],
                        (float *)physical_buffers[output_token - 100], words,
                        input_rank, input_shape01, input_shape23, input_shape4,
                        output_rank, output_shape01, output_shape23,
                        output_shape4);
  return 23;
}

cl_int __vrc_opencl_await(cl_int event) {
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
  cl_event marker_event, write_event, wait_events[2];
  cl_mem first, second, nested_buffer, nested_buffer2, static_buffer;
  cl_mem batched_a, batched_b, batched_c;
  cl_mem transpose_input_buffer, transpose_output_buffer;
  cl_mem q4_activation, q4_packed, q4_scales, q4_previous, q4_output;
  cl_mem resolved;
  udonclBufferHandle_t common_buffer = {0, 0}, stale_buffer, replacement;
  udonclBufferHandle_t host_input_buffer = {0, 0};
  udonclBufferHandle_t host_output_buffer = {0, 0};
  udonclBufferView_t view;
  udonclWorkspace_t workspace;
  udononnxTensor_t tensor;
  cl_int error;
  cl_int event_status;
  cl_int integer_arguments[8] = {1, 10, 16, 0, 0, 1, 16, 1};
  cl_float float_arguments[2] = {1.5f, 0.0f};
  const char *source_text = "known AOT program";
  size_t global[2] = {32, 32}, local[2] = {1, 1};
  cl_uint index;
  int source = 42;
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

  first = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(source), 0, &error);
  EXPECT(first && error == CL_SUCCESS);
  EXPECT(allocated_logical_id == 1 && allocated_bytes == sizeof(source));
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
  EXPECT(uploaded_token == 101);
  EXPECT(clReleaseMemObject(first) == CL_SUCCESS);
  EXPECT(released_token == 101);

  /* Reusing the logical table slot must request a new physical token rather
   * than retaining the released backing-resource identity. */
  second =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(source), 0, &error);
  EXPECT(second && error == CL_SUCCESS);
  EXPECT(allocated_logical_id == 1);

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

  /* AOT weights have their own immutable table.  They use the same cl_mem
   * ABI as mutable buffers without consuming a mutable allocation slot. */
  allocated_logical_id = 0;
  released_token = 0;
  static_buffer = udonclAdoptStaticBuffer(
      queue, 7, 5760, __vrc_opencl_static_tensor(7, 5760), &error);
  EXPECT(static_buffer && error == CL_SUCCESS);
  EXPECT(static_resource_id == 7 && static_resource_bytes == 5760);
  EXPECT(allocated_logical_id == 0);
  EXPECT(clEnqueueWriteBuffer(queue, static_buffer, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, 0) == CL_INVALID_VALUE);
  nested_buffer = udonclCreateBufferForQueue(queue, CL_MEM_READ_WRITE,
                                             sizeof(source), &error);
  EXPECT(nested_buffer && error == CL_SUCCESS);
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_args(queue, first, static_buffer,
                                                 nested_buffer, 1, 10, 144, 0,
                                                 0, 1.0f) == CL_SUCCESS);
  EXPECT(dispatched_arena_a_token == 102 && dispatched_arena_b_token == 1007 &&
         dispatched_arena_output_token == 103);
  await_count = 0;
  EXPECT(__vrc_enqueue_udonblas_arena_sgemm_device_args(
             queue, first, static_buffer, nested_buffer, 1, 10, 144, 0, 0, 1.0f,
             0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event);
  EXPECT(clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 11);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  ((float *)physical_buffers[2])[0] = 0.0f;
  ((float *)physical_buffers[2])[1] = 1.57079632679f;
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_unary_device_args(queue, first, nested_buffer,
                                                  2, 2, 0, 0,
                                                  &marker_event) == CL_SUCCESS);
  EXPECT(marker_event);
  EXPECT(clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 17);
  EXPECT(((float *)physical_buffers[3])[0] == 0.0f);
  EXPECT(((float *)physical_buffers[3])[1] > 0.999999f &&
         ((float *)physical_buffers[3])[1] < 1.000001f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  transpose_input_buffer = udonclCreateBufferForQueue(
      queue, CL_MEM_READ_WRITE, 6 * sizeof(float), &error);
  EXPECT(transpose_input_buffer && error == CL_SUCCESS);
  transpose_output_buffer = udonclCreateBufferForQueue(
      queue, CL_MEM_READ_WRITE, 6 * sizeof(float), &error);
  EXPECT(transpose_output_buffer && error == CL_SUCCESS);
  ((float *)physical_buffers[4])[0] = 1.0f;
  ((float *)physical_buffers[4])[1] = 2.0f;
  ((float *)physical_buffers[4])[2] = 3.0f;
  ((float *)physical_buffers[4])[3] = 4.0f;
  ((float *)physical_buffers[4])[4] = 5.0f;
  ((float *)physical_buffers[4])[5] = 6.0f;
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_transpose_device_args(
             queue, transpose_input_buffer, transpose_output_buffer, 6, 2,
             2 | (3 << 16), 1 | (1 << 16), 1, 1, 0, 0,
             &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 21);
  EXPECT(((float *)physical_buffers[5])[0] == 1.0f);
  EXPECT(((float *)physical_buffers[5])[1] == 4.0f);
  EXPECT(((float *)physical_buffers[5])[2] == 2.0f);
  EXPECT(((float *)physical_buffers[5])[3] == 5.0f);
  EXPECT(((float *)physical_buffers[5])[4] == 3.0f);
  EXPECT(((float *)physical_buffers[5])[5] == 6.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_slice_device_args(
             queue, transpose_input_buffer, transpose_output_buffer, 4, 2,
             2 | (3 << 16), 1 | (1 << 16), 1, 2 | (2 << 16), 1 | (1 << 16), 1,
             0, 0, 0, 1 | (2 << 16), 1 | (1 << 16), 1, 0, 0,
             &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 22);
  EXPECT(((float *)physical_buffers[5])[0] == 1.0f);
  EXPECT(((float *)physical_buffers[5])[1] == 3.0f);
  EXPECT(((float *)physical_buffers[5])[2] == 4.0f);
  EXPECT(((float *)physical_buffers[5])[3] == 6.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  ((float *)physical_buffers[4])[0] = 4.0f;
  ((float *)physical_buffers[4])[1] = 7.0f;
  await_count = 0;
  EXPECT(__vrc_enqueue_udononnx_expand_device_args(
             queue, transpose_input_buffer, transpose_output_buffer, 6, 3,
             1 | (2 << 16), 1 | (1 << 16), 1, 3, 1 | (2 << 16), 3 | (1 << 16),
             1, 0, 0, &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(await_count == 1 && awaited_events[0] == 23);
  EXPECT(((float *)physical_buffers[5])[0] == 4.0f);
  EXPECT(((float *)physical_buffers[5])[1] == 4.0f);
  EXPECT(((float *)physical_buffers[5])[2] == 4.0f);
  EXPECT(((float *)physical_buffers[5])[3] == 7.0f);
  EXPECT(((float *)physical_buffers[5])[4] == 7.0f);
  EXPECT(((float *)physical_buffers[5])[5] == 7.0f);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
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
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             4, 32000, 5632, 32, 0, 5504, 32000, 128, 1, 0, 0,
             &marker_event) == CL_SUCCESS);
  EXPECT(marker_event && clWaitForEvents(1, &marker_event) == CL_SUCCESS);
  EXPECT(clReleaseEvent(marker_event) == CL_SUCCESS);
  EXPECT(__vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
             queue, q4_activation, q4_packed, q4_scales, q4_previous, q4_output,
             4, 32000, 5632, 32, 1, 5504, 31999, 128, 1, 0, 0,
             0) == CL_INVALID_VALUE);
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
  EXPECT(clReleaseMemObject(first) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(second) == CL_SUCCESS);
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
  EXPECT(clReleaseCommandQueue(queue) == CL_SUCCESS);
  EXPECT(clReleaseContext(context) == CL_SUCCESS);
  return 0;
}
