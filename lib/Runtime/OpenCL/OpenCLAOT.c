/* OpenCL 1.2 Host subset for the Udon AOT backend.
 *
 * This translation unit deliberately has no hosted-libc dependency. Logical
 * OpenCL objects live in fixed per-library-instance storage; only __vrc_*
 * functions cross into the compiler/Unity backend. The native test runtime
 * remains OpenCL.c and implements the same public OpenCL calls independently.
 */
#include <CL/cl.h>

#ifndef UDONCL_AOT_BUFFER_CAPACITY
#define UDONCL_AOT_BUFFER_CAPACITY 32
#endif
#define UDONCL_AOT_KERNEL_CAPACITY 32
#define UDONCL_AOT_EVENT_CAPACITY 32
#define UDONCL_AOT_ARGUMENT_BYTES 8
#define UDONCL_AOT_EVENT_RESOURCE_CAPACITY 5

#ifndef UDONCL_AOT_ENABLE_BLAS
#define UDONCL_AOT_ENABLE_BLAS 1
#endif
#ifndef UDONCL_AOT_ENABLE_MNIST
#define UDONCL_AOT_ENABLE_MNIST 0
#endif
#ifndef UDONCL_AOT_ENABLE_DNN
#define UDONCL_AOT_ENABLE_DNN 0
#endif
#ifndef UDONCL_AOT_ENABLE_ONNX
#define UDONCL_AOT_ENABLE_ONNX 0
#endif

/* The compiler backend returns an opaque physical-buffer token.  It is kept
 * separately from the OpenCL handle-table id: cl_mem identity and lifetime
 * belong to this runtime, while the token only selects a preallocated Unity
 * backing resource.  Zero is reserved for allocation failure. */
extern cl_uint __vrc_opencl_buffer_alloc(cl_uint logical_id, size_t bytes);
extern cl_int __vrc_opencl_buffer_release(cl_uint physical_token);
extern cl_int __vrc_opencl_buffer_upload(cl_uint physical_token, size_t offset,
                                         size_t bytes, const void *source);
extern cl_int __vrc_opencl_buffer_readback(cl_uint physical_token,
                                           size_t offset, size_t bytes,
                                           void *destination);
extern cl_int __vrc_opencl_buffer_copy(cl_uint source_token,
                                       cl_uint destination_token,
                                       size_t source_offset,
                                       size_t destination_offset, size_t bytes);
#if UDONCL_AOT_ENABLE_BLAS
extern cl_int __vrc_opencl_dispatch_udonblas_sgemm(
    cl_uint input_token, cl_uint output_token, cl_int m, cl_int n, cl_int k,
    cl_int transa, cl_int transb, cl_int lda, cl_int ldb, cl_int ldc,
    cl_float alpha, cl_float beta);
extern cl_int __vrc_opencl_dispatch_udonblas_arena_sgemm(
    cl_uint a_token, cl_uint b_token, cl_uint output_token, cl_int m, cl_int n,
    cl_int k, cl_int transa, cl_int transb, cl_float alpha);
extern cl_int __vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched(
    cl_uint a_token, cl_uint b_token, cl_uint output_token, cl_int m, cl_int n,
    cl_int k, cl_int stride_a, cl_int stride_b, cl_int stride_c,
    cl_int batch_count, cl_int transa, cl_int transb, cl_float alpha);
extern cl_int __vrc_opencl_dispatch_udonblas_sgemv(
    cl_uint input_token, cl_uint output_token, cl_int trans, cl_int m, cl_int n,
    cl_int lda, cl_int incx, cl_int incy, cl_float alpha, cl_float beta);
#endif
#if UDONCL_AOT_ENABLE_DNN
extern cl_int __vrc_opencl_dispatch_udondnn_large_convolution(
    cl_uint input_token, cl_uint filter_token, cl_uint previous_token,
    cl_uint output_token, cl_int output_words, cl_int input_ch, cl_int input_w,
    cl_int output_ch, cl_int output_w, cl_int filter_hw, cl_int pad_hw,
    cl_int stride_hw, cl_int dilation_hw, cl_int mode, cl_float alpha,
    cl_float beta);
extern cl_int __vrc_opencl_dispatch_udondnn_large_activation(
    cl_uint input_token, cl_uint previous_token, cl_uint output_token,
    cl_int words, cl_int mode, cl_int nan_propagation, cl_float coefficient,
    cl_float alpha, cl_float beta);
extern cl_int __vrc_opencl_dispatch_udondnn_large_pooling(
    cl_uint input_token, cl_uint previous_token, cl_uint output_token,
    cl_int output_words, cl_int input_c, cl_int input_h, cl_int input_w,
    cl_int output_h, cl_int output_w, cl_int window_shape, cl_int pad_shape,
    cl_int stride_shape, cl_int options, cl_float alpha, cl_float beta);
extern cl_int __vrc_opencl_dispatch_udondnn_large_op_tensor(
    cl_uint a_token, cl_uint b_token, cl_uint previous_token,
    cl_uint output_token, cl_int output_words, cl_int a_nc, cl_int a_hw,
    cl_int b_nc, cl_int b_hw, cl_int output_nc, cl_int output_hw,
    cl_int options, cl_float alpha_a, cl_float alpha_b, cl_float beta);
extern cl_int __vrc_opencl_dispatch_udonaten_embedding(
    cl_uint input_token, cl_uint output_token, cl_int outer, cl_int rows,
    cl_float rows_float, cl_int inner, cl_int indices_words,
    cl_int output_words);
extern cl_int __vrc_opencl_dispatch_udonaten_arena_rms_norm_reduce(
    cl_uint input_token, cl_uint partials_token, cl_int words,
    cl_int normalized_words);
extern cl_int __vrc_opencl_dispatch_udonaten_arena_rms_norm_finalize(
    cl_uint input_token, cl_uint partials_token, cl_uint weight_token,
    cl_uint output_token, cl_int words, cl_int normalized_words,
    cl_float epsilon, cl_int has_weight);
extern cl_int __vrc_opencl_dispatch_udondnn_arena_softmax(
    cl_uint input_token, cl_uint output_token, cl_int input_n, cl_int input_c,
    cl_int input_h, cl_int input_w, cl_int algorithm, cl_int mode);
#endif
#if UDONCL_AOT_ENABLE_ONNX
extern cl_int __vrc_opencl_dispatch_udononnx_matmul_nbits(
    cl_uint input_token, cl_uint output_token, cl_int tile_n, cl_int tile_k,
    cl_int accumulate, cl_int finalize);
extern cl_int __vrc_opencl_dispatch_udononnx_matmul_nbits_q4(
    cl_uint activation_token, cl_uint packed_weight_token, cl_uint scales_token,
    cl_uint previous_token, cl_uint output_token, cl_int m, cl_int n, cl_int k,
    cl_int block_size, cl_int n_base, cl_int k_base, cl_int tile_n,
    cl_int tile_k, cl_int accumulate);
extern cl_int __vrc_opencl_dispatch_udononnx_gather_copy(cl_uint input_token,
                                                         cl_uint output_token,
                                                         cl_int words);
extern cl_int __vrc_opencl_dispatch_udononnx_unary_f32(cl_uint input_token,
                                                       cl_uint output_token,
                                                       cl_int operation,
                                                       cl_int words);
extern cl_int __vrc_opencl_dispatch_udononnx_transpose_f32(
    cl_uint input_token, cl_uint output_token, cl_int words, cl_int rank,
    cl_int shape01, cl_int shape23, cl_int shape4, cl_int permutation);
extern cl_int __vrc_opencl_dispatch_udononnx_slice_f32(
    cl_uint input_token, cl_uint output_token, cl_int words, cl_int rank,
    cl_int input_shape01, cl_int input_shape23, cl_int input_shape4,
    cl_int output_shape01, cl_int output_shape23, cl_int output_shape4,
    cl_int starts01, cl_int starts23, cl_int starts4, cl_int steps01,
    cl_int steps23, cl_int steps4);
extern cl_int __vrc_opencl_dispatch_udononnx_expand_f32(
    cl_uint input_token, cl_uint output_token, cl_int words, cl_int input_rank,
    cl_int input_shape01, cl_int input_shape23, cl_int input_shape4,
    cl_int output_rank, cl_int output_shape01, cl_int output_shape23,
    cl_int output_shape4);
extern cl_int __vrc_opencl_dispatch_udononnx_concat2_f32(
    cl_uint left_token, cl_uint right_token, cl_uint output_token, cl_int words,
    cl_int inner_words, cl_int left_extent, cl_int right_extent);
extern cl_int __vrc_opencl_dispatch_udononnx_gather_axis0_f32(
    cl_uint data_token, cl_uint indices_token, cl_uint output_token,
    cl_int words, cl_int inner_words, cl_int axis_extent, cl_int index_words);
extern cl_int __vrc_opencl_dispatch_udononnx_cast_i64_f32(cl_uint input_token,
                                                          cl_uint output_token,
                                                          cl_int words);
extern cl_int __vrc_opencl_dispatch_udononnx_equal_f32(
    cl_uint left_token, cl_uint right_token, cl_uint output_token, cl_int words,
    cl_int ranks, cl_int left_shape01, cl_int left_shape23,
    cl_int right_shape01, cl_int right_shape23, cl_int output_shape01,
    cl_int output_shape23);
extern cl_int __vrc_opencl_dispatch_udononnx_where_f32(
    cl_uint condition_token, cl_uint true_token, cl_uint false_token,
    cl_uint output_token, cl_int words, cl_int ranks, cl_int condition_shape01,
    cl_int condition_shape23, cl_int true_shape01, cl_int true_shape23,
    cl_int false_shape01, cl_int false_shape23, cl_int output_shape01,
    cl_int output_shape23);
extern cl_int __vrc_opencl_dispatch_udononnx_scatter_nd_f32(
    cl_uint data_token, cl_uint indices_token, cl_uint updates_token,
    cl_uint output_token, cl_int words, cl_int rank, cl_int shape01,
    cl_int shape23, cl_int tuple_rank, cl_int update_count, cl_int slice_words,
    cl_int index_words);
#endif
#if UDONCL_AOT_ENABLE_MNIST
extern cl_int __vrc_opencl_dispatch_mnist_conv(cl_uint input_token,
                                               cl_uint output_token);
extern cl_int __vrc_opencl_dispatch_mnist_pool(cl_uint input_token,
                                               cl_uint output_token);
extern cl_int __vrc_opencl_dispatch_mnist_dense(cl_uint input_token,
                                                cl_uint output_token);
#endif
extern cl_int __vrc_opencl_await(cl_int event);

struct _cl_platform_id {
  cl_uint active;
};
struct _cl_device_id {
  cl_uint active;
};
struct _cl_context {
  cl_uint active;
  cl_uint refs;
};
struct _cl_command_queue {
  cl_uint active;
  cl_uint refs;
  cl_context context;
  cl_ulong sequence;
  cl_event tail;
};
struct _cl_program {
  cl_uint active;
  cl_context context;
  cl_build_status status;
};
struct _cl_mem {
  cl_uint active;
  cl_uint id;
  cl_uint physical_token;
  cl_uint immutable;
  cl_uint refs;
  cl_uint command_refs;
  cl_context context;
  cl_mem_flags flags;
  size_t bytes;
};
struct _cl_kernel {
  cl_uint active;
  cl_uint id;
  cl_program program;
  unsigned char arguments[UDONCL_AOT_KERNEL_CAPACITY]
                         [UDONCL_AOT_ARGUMENT_BYTES];
  cl_uint argument_sizes[UDONCL_AOT_KERNEL_CAPACITY];
};
struct _cl_event {
  cl_uint active;
  cl_uint refs;
  cl_uint internal;
  cl_command_queue queue;
  cl_command_type command_type;
  cl_int pending;
  cl_int execution_status;
  cl_ulong sequence;
  cl_uint resource_count;
  cl_mem resources[UDONCL_AOT_EVENT_RESOURCE_CAPACITY];
};

static struct _cl_platform_id udoncl_platform = {1};
static struct _cl_device_id udoncl_device = {1};
static struct _cl_context udoncl_context;
static struct _cl_command_queue udoncl_queue;
static struct _cl_program udoncl_program;
static struct _cl_mem udoncl_buffers[UDONCL_AOT_BUFFER_CAPACITY];
static struct _cl_kernel udoncl_kernels[UDONCL_AOT_KERNEL_CAPACITY];
static struct _cl_event udoncl_events[UDONCL_AOT_EVENT_CAPACITY];
static cl_int udoncl_dispatch_status;
static cl_int udoncl_dispatch_event;
static cl_uint udoncl_pending_resource_count;
static cl_mem udoncl_pending_resources[UDONCL_AOT_EVENT_RESOURCE_CAPACITY];

static void udoncl_pending_resources_reset(void) {
  udoncl_pending_resource_count = 0;
}

/* Immutable artifact resources use odd integer cl_mem tags, while mutable
 * handles are aligned pointers into udoncl_buffers.  The tag encodes
 * resource_id + 1; its physical token remains -(resource_id + 1).  This avoids
 * both a per-weight _cl_mem table and pointer ordering comparisons, which are
 * not part of the recoverable Udon Host ABI. */
static int udoncl_mem_is_static(cl_mem buffer) {
  return buffer && (((size_t)buffer & 1u) != 0);
}

static int udoncl_mem_is_active(cl_mem buffer) {
  return udoncl_mem_is_static(buffer) ? udoncl_context.active != 0
                                      : buffer && buffer->active != 0;
}

static cl_context udoncl_mem_context(cl_mem buffer) {
  return udoncl_mem_is_static(buffer) ? &udoncl_context : buffer->context;
}

static cl_uint udoncl_mem_physical_token(cl_mem buffer) {
  return udoncl_mem_is_static(buffer) ? 0u - ((cl_uint)(size_t)buffer >> 1u)
                                      : buffer->physical_token;
}

static size_t udoncl_mem_bytes(cl_mem buffer) {
  return udoncl_mem_is_static(buffer) ? (size_t)-1 : buffer->bytes;
}

static cl_mem_flags udoncl_mem_flags(cl_mem buffer) {
  return udoncl_mem_is_static(buffer) ? CL_MEM_READ_ONLY : buffer->flags;
}

static cl_int udoncl_pending_resource_add(cl_mem buffer) {
  cl_uint index;
  if (!buffer)
    return CL_SUCCESS;
  if (udoncl_mem_is_static(buffer))
    return CL_SUCCESS;
  for (index = 0; index < udoncl_pending_resource_count; ++index)
    if (udoncl_pending_resources[index] == buffer)
      return CL_SUCCESS;
  if (udoncl_pending_resource_count == UDONCL_AOT_EVENT_RESOURCE_CAPACITY)
    return CL_OUT_OF_RESOURCES;
  udoncl_pending_resources[udoncl_pending_resource_count++] = buffer;
  return CL_SUCCESS;
}

static int udoncl_mem_is_subbuffer(cl_mem buffer) {
  return buffer && !udoncl_mem_is_static(buffer) && buffer->immutable >= 2u;
}

static cl_mem udoncl_mem_parent(cl_mem buffer) {
  return udoncl_mem_is_subbuffer(buffer)
             ? &udoncl_buffers[buffer->immutable - 2u]
             : 0;
}

static size_t udoncl_mem_origin(cl_mem buffer) {
  return udoncl_mem_is_subbuffer(buffer) ? (size_t)buffer->id : 0u;
}

static cl_int udoncl_prepare_resources(cl_mem output, const cl_mem *reads,
                                       cl_uint read_count) {
  cl_uint index;
  udoncl_pending_resources_reset();
  for (index = 0; index < read_count; ++index)
    if (udoncl_pending_resource_add(reads[index]) != CL_SUCCESS)
      return CL_OUT_OF_RESOURCES;
  return udoncl_pending_resource_add(output);
}

#ifndef UDONCL_AOT_RUNTIME_VALIDATION
#define UDONCL_AOT_RUNTIME_VALIDATION 0
#endif

static cl_int udoncl_validate_read_write_alias(cl_mem output,
                                               const cl_mem *reads,
                                               cl_uint read_count) {
#if UDONCL_AOT_RUNTIME_VALIDATION
  cl_uint index;
  for (index = 0; index < read_count; ++index)
    if (udoncl_mem_physical_token(reads[index]) ==
        udoncl_mem_physical_token(output))
      return CL_INVALID_OPERATION;
#else
  (void)output;
  (void)reads;
  (void)read_count;
#endif
  return CL_SUCCESS;
}
#define UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, ...)                         \
  do {                                                                         \
    cl_mem udoncl_reads[] = {__VA_ARGS__};                                     \
    if (udoncl_prepare_resources(                                              \
            output, udoncl_reads,                                              \
            (cl_uint)(sizeof(udoncl_reads) / sizeof(udoncl_reads[0]))) !=      \
        CL_SUCCESS)                                                            \
      return CL_OUT_OF_RESOURCES;                                              \
    if (udoncl_validate_read_write_alias(                                      \
            output, udoncl_reads,                                              \
            (cl_uint)(sizeof(udoncl_reads) / sizeof(udoncl_reads[0]))) !=      \
        CL_SUCCESS)                                                            \
      return CL_INVALID_OPERATION;                                             \
  } while (0)

static int udoncl_string_equal(const char *left, const char *right) {
  if (!left || !right)
    return 0;
  while (*left && *left == *right) {
    ++left;
    ++right;
  }
  return *left == *right;
}

static cl_uint udoncl_kernel_id(const char *name) {
#if UDONCL_AOT_ENABLE_BLAS
  if (udoncl_string_equal(name, "udonblas_sgemm"))
    return 1;
  if (udoncl_string_equal(name, "udonblas_sgemv"))
    return 2;
  if (udoncl_string_equal(name, "udonblas_saxpy"))
    return 3;
  if (udoncl_string_equal(name, "udonblas_sdot_product"))
    return 4;
  if (udoncl_string_equal(name, "udonblas_sdot_reduce"))
    return 5;
#endif
#if UDONCL_AOT_ENABLE_MNIST
  if (udoncl_string_equal(name, "mnist_conv"))
    return 20;
  if (udoncl_string_equal(name, "mnist_pool"))
    return 21;
  if (udoncl_string_equal(name, "mnist_dense"))
    return 22;
#endif
#if UDONCL_AOT_ENABLE_DNN
  if (udoncl_string_equal(name, "udondnn_convolution_forward"))
    return 200;
  if (udoncl_string_equal(name, "udondnn_activation_forward"))
    return 201;
  if (udoncl_string_equal(name, "udondnn_pooling_forward"))
    return 202;
  if (udoncl_string_equal(name, "udondnn_softmax_forward"))
    return 203;
  if (udoncl_string_equal(name, "udondnn_batchnorm_inference"))
    return 204;
  if (udoncl_string_equal(name, "udondnn_op_tensor"))
    return 205;
  if (udoncl_string_equal(name, "udonaten_rms_norm_small"))
    return 206;
  if (udoncl_string_equal(name, "udonaten_embedding"))
    return 207;
  if (udoncl_string_equal(name, "udonaten_tile"))
    return 209;
  if (udoncl_string_equal(name, "udonaten_rms_norm_reduce"))
    return 210;
  if (udoncl_string_equal(name, "udonaten_rms_norm_finalize"))
    return 211;
  if (udoncl_string_equal(name, "udonaten_arena_pointwise"))
    return 215;
  if (udoncl_string_equal(name, "udonaten_arena_tile"))
    return 216;
  if (udoncl_string_equal(name, "udonaten_arena_rms_norm_reduce"))
    return 217;
  if (udoncl_string_equal(name, "udonaten_arena_rms_norm_finalize"))
    return 218;
  if (udoncl_string_equal(name, "udondnn_large_pooling_forward"))
    return 231;
  if (udoncl_string_equal(name, "udondnn_large_convolution_forward"))
    return 232;
  if (udoncl_string_equal(name, "udondnn_large_activation_forward"))
    return 233;
  if (udoncl_string_equal(name, "udondnn_large_op_tensor"))
    return 234;
#endif
#if UDONCL_AOT_ENABLE_ONNX
  if (udoncl_string_equal(name, "udononnx_matmul_nbits"))
    return 219;
  if (udoncl_string_equal(name, "udononnx_gather_copy"))
    return 220;
  if (udoncl_string_equal(name, "udononnx_unary_f32"))
    return 222;
  if (udoncl_string_equal(name, "udononnx_transpose_f32"))
    return 238;
  if (udoncl_string_equal(name, "udononnx_slice_f32"))
    return 239;
  if (udoncl_string_equal(name, "udononnx_expand_f32"))
    return 240;
  if (udoncl_string_equal(name, "udononnx_concat2_f32"))
    return 241;
  if (udoncl_string_equal(name, "udononnx_gather_axis0_f32"))
    return 242;
  if (udoncl_string_equal(name, "udononnx_cast_i64_f32"))
    return 243;
  if (udoncl_string_equal(name, "udononnx_equal_f32"))
    return 244;
  if (udoncl_string_equal(name, "udononnx_where_f32"))
    return 245;
  if (udoncl_string_equal(name, "udononnx_scatter_nd_f32"))
    return 246;
#endif
  return 0;
}

static void udoncl_error(cl_int *output, cl_int value) {
  if (output)
    *output = value;
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(cl_uint count,
                                                 cl_platform_id *platforms,
                                                 cl_uint *total) {
  if ((!platforms && !total) || (platforms && !count))
    return CL_INVALID_VALUE;
  if (platforms)
    platforms[0] = &udoncl_platform;
  if (total)
    *total = 1;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id platform,
                                               cl_device_type type,
                                               cl_uint count,
                                               cl_device_id *devices,
                                               cl_uint *total) {
  if (platform != &udoncl_platform)
    return CL_INVALID_PLATFORM;
  if (!(type & (CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_GPU)))
    return CL_DEVICE_NOT_FOUND;
  if ((!devices && !total) || (devices && !count))
    return CL_INVALID_VALUE;
  if (devices)
    devices[0] = &udoncl_device;
  if (total)
    *total = 1;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_context CL_API_CALL clCreateContext(
    const cl_context_properties *properties, cl_uint count,
    const cl_device_id *devices,
    void(CL_CALLBACK *notify)(const char *, const void *, size_t, void *),
    void *user, cl_int *error) {
  if (properties || count != 1 || !devices || devices[0] != &udoncl_device ||
      notify || user) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  /* Separate library clients share the AOT singleton, so every successful
   * create owns one reference to the same logical object. */
  if (udoncl_context.active) {
    ++udoncl_context.refs;
  } else {
    udoncl_context.active = 1;
    udoncl_context.refs = 1;
  }
  udoncl_error(error, CL_SUCCESS);
  return &udoncl_context;
}

CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context context, cl_device_id device,
                     cl_command_queue_properties properties, cl_int *error) {
  if (context != &udoncl_context || !context->active ||
      device != &udoncl_device || properties) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  /* Match the context aliasing rule above; resetting refs here lets one
   * library's teardown invalidate every other client's queue. */
  if (udoncl_queue.active) {
    if (udoncl_queue.context != context) {
      udoncl_error(error, CL_INVALID_CONTEXT);
      return 0;
    }
    ++udoncl_queue.refs;
  } else {
    udoncl_queue.active = 1;
    udoncl_queue.refs = 1;
    udoncl_queue.context = context;
    udoncl_queue.sequence = 0;
    udoncl_queue.tail = 0;
  }
  udoncl_error(error, CL_SUCCESS);
  return &udoncl_queue;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetCommandQueueInfo(cl_command_queue queue, cl_command_queue_info param,
                      size_t value_size, void *value, size_t *required_size) {
  if (queue != &udoncl_queue || !queue->active)
    return CL_INVALID_COMMAND_QUEUE;
  if (param == CL_QUEUE_CONTEXT) {
    if (required_size)
      *required_size = sizeof(queue->context);
    if (!value)
      return value_size ? CL_INVALID_VALUE : CL_SUCCESS;
    if (value_size < sizeof(queue->context))
      return CL_INVALID_VALUE;
    *(cl_context *)value = queue->context;
    return CL_SUCCESS;
  }
  if (param == CL_QUEUE_REFERENCE_COUNT) {
    if (required_size)
      *required_size = sizeof(queue->refs);
    if (!value)
      return value_size ? CL_INVALID_VALUE : CL_SUCCESS;
    if (value_size < sizeof(queue->refs))
      return CL_INVALID_VALUE;
    *(cl_uint *)value = queue->refs;
    return CL_SUCCESS;
  }
  return CL_INVALID_VALUE;
}

CL_API_ENTRY cl_int CL_API_CALL clRetainCommandQueue(cl_command_queue queue) {
  if (queue != &udoncl_queue || !queue->active)
    return CL_INVALID_COMMAND_QUEUE;
  ++queue->refs;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context context,
                                               cl_mem_flags flags, size_t bytes,
                                               void *host, cl_int *error) {
  cl_uint index;
  if (context != &udoncl_context || !context->active || !bytes || host) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  for (index = 0; index < UDONCL_AOT_BUFFER_CAPACITY; ++index) {
    cl_mem buffer = &udoncl_buffers[index];
    if (buffer->active || buffer->command_refs || buffer->physical_token)
      continue;
    buffer->active = 1;
    buffer->id = index + 1;
    buffer->refs = 1;
    buffer->command_refs = 0;
    buffer->context = context;
    buffer->flags = flags;
    buffer->immutable = 0;
    buffer->bytes = bytes;
    buffer->physical_token =
        __vrc_opencl_buffer_alloc(buffer->id, buffer->bytes);
    if (!buffer->physical_token) {
      buffer->active = 0;
      udoncl_error(error, CL_OUT_OF_RESOURCES);
      return 0;
    }
    udoncl_error(error, CL_SUCCESS);
    return buffer;
  }
  udoncl_error(error, CL_OUT_OF_RESOURCES);
  return 0;
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer(cl_mem parent,
                                                  cl_mem_flags flags,
                                                  cl_buffer_create_type type,
                                                  const void *creation_info,
                                                  cl_int *error) {
  const cl_buffer_region *region = (const cl_buffer_region *)creation_info;
  cl_mem_flags access =
      flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY);
  cl_mem_flags parent_access;
  cl_uint index;
  cl_uint parent_index;
  for (parent_index = 0; parent_index < UDONCL_AOT_BUFFER_CAPACITY;
       ++parent_index)
    if (parent == &udoncl_buffers[parent_index])
      break;
  if (parent_index == UDONCL_AOT_BUFFER_CAPACITY || !parent->active ||
      parent->immutable) {
    udoncl_error(error, CL_INVALID_MEM_OBJECT);
    return 0;
  }
  if (type != CL_BUFFER_CREATE_TYPE_REGION || !region ||
      (flags & ~(CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY)) ||
      (access != 0 && access != CL_MEM_READ_WRITE &&
       access != CL_MEM_WRITE_ONLY && access != CL_MEM_READ_ONLY)) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  parent_access = parent->flags &
                  (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY);
  if ((parent_access == CL_MEM_WRITE_ONLY && access != 0 &&
       access != CL_MEM_WRITE_ONLY) ||
      (parent_access == CL_MEM_READ_ONLY && access != 0 &&
       access != CL_MEM_READ_ONLY)) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  if (region->origin & 3u) {
    udoncl_error(error, CL_MISALIGNED_SUB_BUFFER_OFFSET);
    return 0;
  }
  if (!region->size || region->origin > parent->bytes ||
      region->size > parent->bytes - region->origin) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  for (index = 0; index < UDONCL_AOT_BUFFER_CAPACITY; ++index) {
    cl_mem buffer = &udoncl_buffers[index];
    if (buffer->active || buffer->command_refs || buffer->physical_token)
      continue;
    buffer->active = 1;
    buffer->id = (cl_uint)region->origin;
    buffer->physical_token = parent->physical_token;
    buffer->immutable = parent_index + 2u;
    buffer->refs = 1;
    buffer->command_refs = 0;
    buffer->context = parent->context;
    buffer->flags = access ? access : parent->flags;
    buffer->bytes = region->size;
    ++parent->refs;
    udoncl_error(error, CL_SUCCESS);
    return buffer;
  }
  udoncl_error(error, CL_OUT_OF_RESOURCES);
  return 0;
}

cl_mem udonclCreateBufferForQueue(cl_command_queue queue, cl_mem_flags flags,
                                  size_t bytes, cl_int *error) {
  if (queue != &udoncl_queue || !queue->active) {
    udoncl_error(error, CL_INVALID_COMMAND_QUEUE);
    return 0;
  }
  return clCreateBuffer(queue->context, flags, bytes, 0, error);
}

cl_int udonclResolvePhysicalBuffer(cl_mem buffer, cl_uint *physical_token) {
  if (!udoncl_mem_is_active(buffer) || !physical_token)
    return CL_INVALID_MEM_OBJECT;
  *physical_token = udoncl_mem_physical_token(buffer);
  return *physical_token ? CL_SUCCESS : CL_INVALID_MEM_OBJECT;
}

cl_mem udonclAdoptStaticBuffer(cl_command_queue queue, cl_uint resource_id,
                               size_t bytes, cl_uint physical_token,
                               cl_int *error) {
  cl_context context;
  cl_uint expected_token;
  if (queue != &udoncl_queue || !queue->active) {
    udoncl_error(error, CL_INVALID_COMMAND_QUEUE);
    return 0;
  }
  context = queue->context;
  expected_token = 0u - resource_id - 1u;
  if (context != &udoncl_context || !context->active || !bytes ||
      resource_id >= 0x7ffffffeu || !(expected_token & 0x80000000u) ||
      physical_token != expected_token) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  udoncl_error(error, CL_SUCCESS);
  return (cl_mem)(size_t)(((resource_id + 1u) << 1u) | 1u);
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(
    cl_context context, cl_uint count, const char **strings,
    const size_t *lengths, cl_int *error) {
  (void)lengths;
  if (context != &udoncl_context || !context->active || count != 1 ||
      !strings || !strings[0]) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  udoncl_program.active = 1;
  udoncl_program.context = context;
  udoncl_program.status = CL_BUILD_NONE;
  udoncl_error(error, CL_SUCCESS);
  return &udoncl_program;
}

CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program program, cl_uint count, const cl_device_id *devices,
               const char *options,
               void(CL_CALLBACK *notify)(cl_program, void *), void *user) {
  (void)options;
  if (program != &udoncl_program || !program->active || count || devices ||
      notify || user)
    return CL_INVALID_VALUE;
  program->status = CL_BUILD_SUCCESS;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program program,
                                                  const char *name,
                                                  cl_int *error) {
  cl_uint id = udoncl_kernel_id(name);
  cl_uint slot;
  cl_kernel kernel = 0;
  if (program != &udoncl_program || !program->active ||
      program->status != CL_BUILD_SUCCESS || !id) {
    udoncl_error(error, id ? CL_INVALID_PROGRAM : CL_INVALID_KERNEL_NAME);
    return 0;
  }
  for (slot = 0; slot < UDONCL_AOT_KERNEL_CAPACITY; ++slot)
    if (!udoncl_kernels[slot].active) {
      kernel = &udoncl_kernels[slot];
      break;
    }
  if (!kernel) {
    udoncl_error(error, CL_OUT_OF_RESOURCES);
    return 0;
  }
  kernel->active = 1;
  kernel->id = id;
  kernel->program = program;
  for (slot = 0; slot < UDONCL_AOT_KERNEL_CAPACITY; ++slot)
    kernel->argument_sizes[slot] = 0;
  udoncl_error(error, CL_SUCCESS);
  return kernel;
}

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel, cl_uint index,
                                               size_t bytes,
                                               const void *value) {
  const unsigned char *source = (const unsigned char *)value;
  cl_uint cursor;
  if (!kernel || !kernel->active || index >= UDONCL_AOT_KERNEL_CAPACITY ||
      !bytes || !value || bytes > UDONCL_AOT_ARGUMENT_BYTES)
    return CL_INVALID_ARG_VALUE;
  kernel->argument_sizes[index] = (cl_uint)bytes;
  for (cursor = 0; cursor < bytes; ++cursor)
    kernel->arguments[index][cursor] = source[cursor];
  return CL_SUCCESS;
}

static cl_int udoncl_i32_argument(cl_kernel kernel, cl_uint index,
                                  cl_int *value) {
  unsigned char *destination = (unsigned char *)value;
  cl_uint cursor;
  if (kernel->argument_sizes[index] != sizeof(*value))
    return CL_INVALID_KERNEL_ARGS;
  for (cursor = 0; cursor < sizeof(*value); ++cursor)
    destination[cursor] = kernel->arguments[index][cursor];
  return CL_SUCCESS;
}

static cl_int udoncl_f32_argument(cl_kernel kernel, cl_uint index,
                                  cl_float *value) {
  unsigned char *destination = (unsigned char *)value;
  cl_uint cursor;
  if (kernel->argument_sizes[index] != sizeof(*value))
    return CL_INVALID_KERNEL_ARGS;
  for (cursor = 0; cursor < sizeof(*value); ++cursor)
    destination[cursor] = kernel->arguments[index][cursor];
  return CL_SUCCESS;
}

static cl_int udoncl_mem_argument(cl_kernel kernel, cl_uint index,
                                  cl_mem *value) {
  unsigned char *destination = (unsigned char *)value;
  cl_uint cursor;
  if (kernel->argument_sizes[index] != sizeof(*value))
    return CL_INVALID_KERNEL_ARGS;
  for (cursor = 0; cursor < sizeof(*value); ++cursor)
    destination[cursor] = kernel->arguments[index][cursor];
  if (!udoncl_mem_is_active(*value) || udoncl_mem_is_subbuffer(*value))
    return CL_INVALID_MEM_OBJECT;
  return udoncl_pending_resource_add(*value);
}

#if UDONCL_AOT_ENABLE_BLAS
static void udoncl_dispatch_sgemm(cl_kernel kernel, cl_uint work_dim,
                                  const size_t *global_offset,
                                  const size_t *global_size,
                                  const size_t *local_size) {
  cl_mem input, output;
  cl_int values[8];
  cl_float alpha, beta;
  cl_uint index;
  if (work_dim != 2) {
    udoncl_dispatch_status = CL_INVALID_WORK_DIMENSION;
    return;
  }
  if (global_offset) {
    udoncl_dispatch_status = CL_INVALID_GLOBAL_OFFSET;
    return;
  }
  // The Udon AOT compiler already pins and verifies global/local sizes in the
  // vrc.dispatch attributes generated from the Unity library configuration.
  // Do not dereference the caller's temporary size arrays after an async Host
  // suspension; they are not part of the runtime dispatch ABI.
  (void)global_size;
  (void)local_size;
  if (udoncl_mem_argument(kernel, 0, &input) != CL_SUCCESS ||
      udoncl_mem_argument(kernel, 1, &output) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_MEM_OBJECT;
    return;
  }
  for (index = 0; index < 8; ++index)
    if (udoncl_i32_argument(kernel, index + 2, &values[index]) != CL_SUCCESS) {
      udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
      return;
    }
  if (udoncl_f32_argument(kernel, 10, &alpha) != CL_SUCCESS ||
      udoncl_f32_argument(kernel, 11, &beta) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
    return;
  }
  udoncl_dispatch_event = __vrc_opencl_dispatch_udonblas_sgemm(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      values[0], values[1], values[2], values[3], values[4], values[5],
      values[6], values[7], alpha, beta);
  return;
}

static void udoncl_dispatch_sgemv(cl_kernel kernel, cl_uint work_dim,
                                  const size_t *global_offset,
                                  const size_t *global_size,
                                  const size_t *local_size) {
  cl_mem input, output;
  cl_int values[6];
  cl_float alpha, beta;
  cl_uint index;
  if (work_dim != 1) {
    udoncl_dispatch_status = CL_INVALID_WORK_DIMENSION;
    return;
  }
  if (global_offset) {
    udoncl_dispatch_status = CL_INVALID_GLOBAL_OFFSET;
    return;
  }
  (void)global_size;
  (void)local_size;
  if (udoncl_mem_argument(kernel, 0, &input) != CL_SUCCESS ||
      udoncl_mem_argument(kernel, 1, &output) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_MEM_OBJECT;
    return;
  }
  for (index = 0; index < 6; ++index)
    if (udoncl_i32_argument(kernel, index + 2, &values[index]) != CL_SUCCESS) {
      udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
      return;
    }
  if (udoncl_f32_argument(kernel, 8, &alpha) != CL_SUCCESS ||
      udoncl_f32_argument(kernel, 9, &beta) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
    return;
  }
  udoncl_dispatch_event = __vrc_opencl_dispatch_udonblas_sgemv(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      values[0], values[1], values[2], values[3], values[4], values[5], alpha,
      beta);
}
#endif

#if UDONCL_AOT_ENABLE_DNN
static int udoncl_large_dispatch_header(cl_kernel kernel, cl_uint work_dim,
                                        const size_t *global_offset,
                                        const size_t *global_size,
                                        const size_t *local_size) {
  if (work_dim != 2) {
    udoncl_dispatch_status = CL_INVALID_WORK_DIMENSION;
    return 0;
  }
  if (global_offset) {
    udoncl_dispatch_status = CL_INVALID_GLOBAL_OFFSET;
    return 0;
  }
  (void)kernel;
  (void)global_size;
  (void)local_size;
  return 1;
}

static void udoncl_dispatch_large_convolution(cl_kernel kernel,
                                              cl_uint work_dim,
                                              const size_t *global_offset,
                                              const size_t *global_size,
                                              const size_t *local_size) {
  cl_mem buffers[4];
  cl_int values[10];
  cl_float alpha, beta;
  cl_uint index;
  if (!udoncl_large_dispatch_header(kernel, work_dim, global_offset,
                                    global_size, local_size))
    return;
  for (index = 0; index < 4; ++index)
    if (udoncl_mem_argument(kernel, index, &buffers[index]) != CL_SUCCESS) {
      udoncl_dispatch_status = CL_INVALID_MEM_OBJECT;
      return;
    }
  for (index = 0; index < 10; ++index)
    if (udoncl_i32_argument(kernel, index + 4, &values[index]) != CL_SUCCESS) {
      udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
      return;
    }
  if (udoncl_f32_argument(kernel, 14, &alpha) != CL_SUCCESS ||
      udoncl_f32_argument(kernel, 15, &beta) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
    return;
  }
  udoncl_dispatch_event = __vrc_opencl_dispatch_udondnn_large_convolution(
      udoncl_mem_physical_token(buffers[0]),
      udoncl_mem_physical_token(buffers[1]),
      udoncl_mem_physical_token(buffers[2]),
      udoncl_mem_physical_token(buffers[3]), values[0], values[1], values[2],
      values[3], values[4], values[5], values[6], values[7], values[8],
      values[9], alpha, beta);
}

static void udoncl_dispatch_large_activation(cl_kernel kernel, cl_uint work_dim,
                                             const size_t *global_offset,
                                             const size_t *global_size,
                                             const size_t *local_size) {
  cl_mem buffers[3];
  cl_int values[3];
  cl_float coefficient, alpha, beta;
  cl_uint index;
  if (!udoncl_large_dispatch_header(kernel, work_dim, global_offset,
                                    global_size, local_size))
    return;
  for (index = 0; index < 3; ++index)
    if (udoncl_mem_argument(kernel, index, &buffers[index]) != CL_SUCCESS) {
      udoncl_dispatch_status = CL_INVALID_MEM_OBJECT;
      return;
    }
  for (index = 0; index < 3; ++index)
    if (udoncl_i32_argument(kernel, index + 3, &values[index]) != CL_SUCCESS) {
      udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
      return;
    }
  if (udoncl_f32_argument(kernel, 6, &coefficient) != CL_SUCCESS ||
      udoncl_f32_argument(kernel, 7, &alpha) != CL_SUCCESS ||
      udoncl_f32_argument(kernel, 8, &beta) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
    return;
  }
  udoncl_dispatch_event = __vrc_opencl_dispatch_udondnn_large_activation(
      udoncl_mem_physical_token(buffers[0]),
      udoncl_mem_physical_token(buffers[1]),
      udoncl_mem_physical_token(buffers[2]), values[0], values[1], values[2],
      coefficient, alpha, beta);
}

static void udoncl_dispatch_large_pooling(cl_kernel kernel, cl_uint work_dim,
                                          const size_t *global_offset,
                                          const size_t *global_size,
                                          const size_t *local_size) {
  cl_mem buffers[3];
  cl_int values[10];
  cl_float alpha, beta;
  cl_uint index;
  if (!udoncl_large_dispatch_header(kernel, work_dim, global_offset,
                                    global_size, local_size))
    return;
  for (index = 0; index < 3; ++index)
    if (udoncl_mem_argument(kernel, index, &buffers[index]) != CL_SUCCESS) {
      udoncl_dispatch_status = CL_INVALID_MEM_OBJECT;
      return;
    }
  for (index = 0; index < 10; ++index)
    if (udoncl_i32_argument(kernel, index + 3, &values[index]) != CL_SUCCESS) {
      udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
      return;
    }
  if (udoncl_f32_argument(kernel, 13, &alpha) != CL_SUCCESS ||
      udoncl_f32_argument(kernel, 14, &beta) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
    return;
  }
  udoncl_dispatch_event = __vrc_opencl_dispatch_udondnn_large_pooling(
      udoncl_mem_physical_token(buffers[0]),
      udoncl_mem_physical_token(buffers[1]),
      udoncl_mem_physical_token(buffers[2]), values[0], values[1], values[2],
      values[3], values[4], values[5], values[6], values[7], values[8],
      values[9], alpha, beta);
}

static void udoncl_dispatch_large_op_tensor(cl_kernel kernel, cl_uint work_dim,
                                            const size_t *global_offset,
                                            const size_t *global_size,
                                            const size_t *local_size) {
  cl_mem buffers[4];
  cl_int values[8];
  cl_float alpha_a, alpha_b, beta;
  cl_uint index;
  if (!udoncl_large_dispatch_header(kernel, work_dim, global_offset,
                                    global_size, local_size))
    return;
  for (index = 0; index < 4; ++index)
    if (udoncl_mem_argument(kernel, index, &buffers[index]) != CL_SUCCESS) {
      udoncl_dispatch_status = CL_INVALID_MEM_OBJECT;
      return;
    }
  for (index = 0; index < 8; ++index)
    if (udoncl_i32_argument(kernel, index + 4, &values[index]) != CL_SUCCESS) {
      udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
      return;
    }
  if (udoncl_f32_argument(kernel, 12, &alpha_a) != CL_SUCCESS ||
      udoncl_f32_argument(kernel, 13, &alpha_b) != CL_SUCCESS ||
      udoncl_f32_argument(kernel, 14, &beta) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
    return;
  }
  udoncl_dispatch_event = __vrc_opencl_dispatch_udondnn_large_op_tensor(
      udoncl_mem_physical_token(buffers[0]),
      udoncl_mem_physical_token(buffers[1]),
      udoncl_mem_physical_token(buffers[2]),
      udoncl_mem_physical_token(buffers[3]), values[0], values[1], values[2],
      values[3], values[4], values[5], values[6], values[7], alpha_a, alpha_b,
      beta);
}

static void udoncl_dispatch_embedding(cl_kernel kernel, cl_uint work_dim,
                                      const size_t *global_offset,
                                      const size_t *global_size,
                                      const size_t *local_size) {
  cl_mem input, output;
  cl_int outer, rows, inner, indices_words, output_words;
  cl_float rows_float;
  (void)global_size;
  (void)local_size;
  if (work_dim != 1) {
    udoncl_dispatch_status = CL_INVALID_WORK_DIMENSION;
    return;
  }
  if (global_offset) {
    udoncl_dispatch_status = CL_INVALID_GLOBAL_OFFSET;
    return;
  }
  if (udoncl_mem_argument(kernel, 0, &input) != CL_SUCCESS ||
      udoncl_mem_argument(kernel, 1, &output) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_MEM_OBJECT;
    return;
  }
  if (udoncl_i32_argument(kernel, 2, &outer) != CL_SUCCESS ||
      udoncl_i32_argument(kernel, 3, &rows) != CL_SUCCESS ||
      udoncl_f32_argument(kernel, 4, &rows_float) != CL_SUCCESS ||
      udoncl_i32_argument(kernel, 5, &inner) != CL_SUCCESS ||
      udoncl_i32_argument(kernel, 6, &indices_words) != CL_SUCCESS ||
      udoncl_i32_argument(kernel, 7, &output_words) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
    return;
  }
  udoncl_dispatch_event = __vrc_opencl_dispatch_udonaten_embedding(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      outer, rows, rows_float, inner, indices_words, output_words);
}
#endif

#if UDONCL_AOT_ENABLE_ONNX
static void udoncl_dispatch_matmul_nbits(cl_kernel kernel, cl_uint work_dim,
                                         const size_t *global_offset) {
  cl_mem input, output;
  cl_int tile_n, tile_k, accumulate, finalize;
  if (work_dim != 1) {
    udoncl_dispatch_status = CL_INVALID_WORK_DIMENSION;
    return;
  }
  if (global_offset) {
    udoncl_dispatch_status = CL_INVALID_GLOBAL_OFFSET;
    return;
  }
  if (udoncl_mem_argument(kernel, 0, &input) != CL_SUCCESS ||
      udoncl_mem_argument(kernel, 1, &output) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_MEM_OBJECT;
    return;
  }
  if (udoncl_i32_argument(kernel, 2, &tile_n) != CL_SUCCESS ||
      udoncl_i32_argument(kernel, 3, &tile_k) != CL_SUCCESS ||
      udoncl_i32_argument(kernel, 4, &accumulate) != CL_SUCCESS ||
      udoncl_i32_argument(kernel, 5, &finalize) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
    return;
  }
  udoncl_dispatch_event = __vrc_opencl_dispatch_udononnx_matmul_nbits(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      tile_n, tile_k, accumulate, finalize);
}

static void udoncl_dispatch_gather(cl_kernel kernel, cl_uint work_dim,
                                   const size_t *global_offset) {
  cl_mem input, output;
  cl_int words;
  if (work_dim != 1) {
    udoncl_dispatch_status = CL_INVALID_WORK_DIMENSION;
    return;
  }
  if (global_offset) {
    udoncl_dispatch_status = CL_INVALID_GLOBAL_OFFSET;
    return;
  }
  if (udoncl_mem_argument(kernel, 0, &input) != CL_SUCCESS ||
      udoncl_mem_argument(kernel, 1, &output) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_MEM_OBJECT;
    return;
  }
  if (udoncl_i32_argument(kernel, 2, &words) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
    return;
  }
  udoncl_dispatch_event = __vrc_opencl_dispatch_udononnx_gather_copy(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      words);
}

static void udoncl_dispatch_unary(cl_kernel kernel, cl_uint work_dim,
                                  const size_t *global_offset) {
  cl_mem input, output;
  cl_int operation, words;
  if (work_dim != 1) {
    udoncl_dispatch_status = CL_INVALID_WORK_DIMENSION;
    return;
  }
  if (global_offset) {
    udoncl_dispatch_status = CL_INVALID_GLOBAL_OFFSET;
    return;
  }
  if (udoncl_mem_argument(kernel, 0, &input) != CL_SUCCESS ||
      udoncl_mem_argument(kernel, 1, &output) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_MEM_OBJECT;
    return;
  }
  if (udoncl_i32_argument(kernel, 2, &operation) != CL_SUCCESS ||
      udoncl_i32_argument(kernel, 3, &words) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_KERNEL_ARGS;
    return;
  }
  udoncl_dispatch_event = __vrc_opencl_dispatch_udononnx_unary_f32(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      operation, words);
}
#endif

#if UDONCL_AOT_ENABLE_MNIST
static void udoncl_dispatch_mnist(cl_kernel kernel, cl_uint work_dim,
                                  const size_t *global_offset) {
  cl_mem input, output;
  if (work_dim != 1) {
    udoncl_dispatch_status = CL_INVALID_WORK_DIMENSION;
    return;
  }
  if (global_offset) {
    udoncl_dispatch_status = CL_INVALID_GLOBAL_OFFSET;
    return;
  }
  if (udoncl_mem_argument(kernel, 0, &input) != CL_SUCCESS ||
      udoncl_mem_argument(kernel, 1, &output) != CL_SUCCESS) {
    udoncl_dispatch_status = CL_INVALID_MEM_OBJECT;
    return;
  }
  if (kernel->id == 20)
    udoncl_dispatch_event = __vrc_opencl_dispatch_mnist_conv(
        udoncl_mem_physical_token(input), udoncl_mem_physical_token(output));
  else if (kernel->id == 21)
    udoncl_dispatch_event = __vrc_opencl_dispatch_mnist_pool(
        udoncl_mem_physical_token(input), udoncl_mem_physical_token(output));
  else
    udoncl_dispatch_event = __vrc_opencl_dispatch_mnist_dense(
        udoncl_mem_physical_token(input), udoncl_mem_physical_token(output));
}
#endif

static cl_event udoncl_event_find(cl_event event) {
  cl_uint index;
  if (!event)
    return 0;
  for (index = 0; index < UDONCL_AOT_EVENT_CAPACITY; ++index)
    if (event == &udoncl_events[index] && event->active)
      return event;
  return 0;
}

static cl_int udoncl_mem_finalize(cl_mem buffer) {
  cl_int status = CL_SUCCESS;
  cl_mem parent;
  if (!buffer || buffer->refs || buffer->command_refs ||
      !buffer->physical_token)
    return CL_SUCCESS;
  parent = udoncl_mem_parent(buffer);
  if (!parent && buffer->immutable == 0)
    status = __vrc_opencl_buffer_release(buffer->physical_token);
  buffer->physical_token = 0;
  buffer->context = 0;
  buffer->flags = 0;
  buffer->bytes = 0;
  buffer->immutable = 0;
  if (parent) {
    if (parent->refs && --parent->refs == 0)
      parent->active = 0;
    if (!parent->refs && !parent->command_refs && parent->physical_token) {
      if (!parent->immutable)
        status = __vrc_opencl_buffer_release(parent->physical_token);
      parent->physical_token = 0;
      parent->context = 0;
      parent->flags = 0;
      parent->bytes = 0;
      parent->immutable = 0;
    }
  }
  return status;
}

static void udoncl_event_destroy(cl_event event) {
  cl_command_queue queue = event->queue;
  if (queue && queue->tail == event)
    queue->tail = 0;
  if (queue && queue->refs && --queue->refs == 0)
    queue->active = 0;
  event->active = 0;
  event->refs = 0;
  event->internal = 0;
  event->queue = 0;
  event->pending = 0;
  event->execution_status = CL_COMPLETE;
  event->sequence = 0;
  event->resource_count = 0;
}

static void udoncl_event_release_resources(cl_event event) {
  cl_uint index;
  for (index = 0; index < event->resource_count; ++index) {
    cl_mem buffer = event->resources[index];
    if (buffer && buffer->command_refs)
      --buffer->command_refs;
    udoncl_mem_finalize(buffer);
    event->resources[index] = 0;
  }
  event->resource_count = 0;
}

static void udoncl_event_complete_through(cl_event target, cl_int status) {
  cl_uint index;
  cl_command_queue queue = target->queue;
  cl_ulong sequence = target->sequence;
  for (index = 0; index < UDONCL_AOT_EVENT_CAPACITY; ++index) {
    cl_event event = &udoncl_events[index];
    if (!event->active || event->queue != queue || event->sequence > sequence ||
        event->execution_status <= CL_COMPLETE)
      continue;
    event->execution_status = event == target ? status : CL_COMPLETE;
    udoncl_event_release_resources(event);
    event->internal = 0;
    if (!event->refs)
      udoncl_event_destroy(event);
  }
}

static cl_int udoncl_event_wait(cl_event event) {
  cl_int status;
  event = udoncl_event_find(event);
  if (!event)
    return CL_INVALID_EVENT;
  if (event->execution_status <= CL_COMPLETE)
    return event->execution_status;
  status = __vrc_opencl_await(event->pending);
  udoncl_event_complete_through(event,
                                status == CL_SUCCESS ? CL_COMPLETE : status);
  return status;
}

static cl_int udoncl_event_create(cl_command_queue queue,
                                  cl_command_type command_type, cl_int pending,
                                  cl_event *output) {
  cl_uint index;
  cl_event event = 0;
  if (pending < CL_SUCCESS) {
    udoncl_pending_resources_reset();
    return pending;
  }
  if (output)
    *output = 0;
  for (index = 0; index < UDONCL_AOT_EVENT_CAPACITY; ++index) {
    if (udoncl_events[index].active)
      continue;
    event = &udoncl_events[index];
    event->active = 1;
    event->refs = output ? 1 : 0;
    event->internal = 1;
    event->queue = queue;
    event->command_type = command_type;
    event->pending = pending;
    event->execution_status =
        pending == CL_SUCCESS ? CL_COMPLETE : CL_SUBMITTED;
    event->sequence = ++queue->sequence;
    event->resource_count = udoncl_pending_resource_count;
    for (cl_uint resource = 0; resource < event->resource_count; ++resource) {
      event->resources[resource] = udoncl_pending_resources[resource];
      ++event->resources[resource]->command_refs;
    }
    udoncl_pending_resources_reset();
    ++queue->refs;
    queue->tail = event;
    if (output)
      *output = event;
    if (pending == CL_SUCCESS) {
      udoncl_event_release_resources(event);
      event->internal = 0;
      if (!event->refs)
        udoncl_event_destroy(event);
    }
    return CL_SUCCESS;
  }
  udoncl_pending_resources_reset();
  return CL_OUT_OF_RESOURCES;
}

static cl_int udoncl_wait_list(cl_command_queue queue, cl_uint count,
                               const cl_event *events) {
  cl_uint index;
  cl_context context = 0;
  if ((!count && events) || (count && !events))
    return CL_INVALID_EVENT_WAIT_LIST;
  for (index = 0; index < count; ++index) {
    cl_event event = udoncl_event_find(events[index]);
    cl_int status;
    if (!event)
      return CL_INVALID_EVENT_WAIT_LIST;
    if (event->queue->context != queue->context)
      return CL_INVALID_CONTEXT;
    if (!context)
      context = event->queue->context;
    else if (context != event->queue->context)
      return CL_INVALID_CONTEXT;
    status = udoncl_event_wait(event);
    if (status != CL_SUCCESS)
      return status;
  }
  return CL_SUCCESS;
}

static cl_int udoncl_finish_enqueue(cl_command_queue queue,
                                    cl_command_type command_type,
                                    cl_int pending, cl_bool blocking,
                                    cl_event *event) {
  cl_int status = udoncl_event_create(queue, command_type, pending, event);
  if (status != CL_SUCCESS)
    return status;
  if (!blocking)
    return CL_SUCCESS;
  return event && *event ? udoncl_event_wait(*event)
         : queue->tail   ? udoncl_event_wait(queue->tail)
                         : CL_SUCCESS;
}

static cl_int udoncl_finish_dispatch(cl_command_queue queue, cl_event *event) {
  if (udoncl_dispatch_status != CL_SUCCESS) {
    udoncl_pending_resources_reset();
    return udoncl_dispatch_status;
  }
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL,
                               udoncl_dispatch_event, CL_FALSE, event);
}

#if UDONCL_AOT_ENABLE_DNN
cl_int __vrc_enqueue_udondnn_large_convolution_device_args(
    cl_command_queue queue, cl_mem input, cl_mem filter, cl_mem previous,
    cl_mem output, cl_int output_words, cl_int input_ch, cl_int input_w,
    cl_int output_ch, cl_int output_w, cl_int filter_hw, cl_int pad_hw,
    cl_int stride_hw, cl_int dilation_hw, cl_int mode, cl_float alpha,
    cl_float beta, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !filter ||
      !udoncl_mem_is_active(filter) || !previous ||
      !udoncl_mem_is_active(previous) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(filter) != queue->context ||
      udoncl_mem_context(previous) != queue->context ||
      udoncl_mem_context(output) != queue->context ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, input, filter, previous);
  pending = __vrc_opencl_dispatch_udondnn_large_convolution(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(filter),
      udoncl_mem_physical_token(previous), udoncl_mem_physical_token(output),
      output_words, input_ch, input_w, output_ch, output_w, filter_hw, pad_hw,
      stride_hw, dilation_hw, mode, alpha, beta);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udondnn_large_convolution_args(
    cl_command_queue queue, cl_mem input, cl_mem filter, cl_mem previous,
    cl_mem output, cl_int output_words, cl_int input_ch, cl_int input_w,
    cl_int output_ch, cl_int output_w, cl_int filter_hw, cl_int pad_hw,
    cl_int stride_hw, cl_int dilation_hw, cl_int mode, cl_float alpha,
    cl_float beta) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !filter ||
      !udoncl_mem_is_active(filter) || !previous ||
      !udoncl_mem_is_active(previous) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(filter) != queue->context ||
      udoncl_mem_context(previous) != queue->context ||
      udoncl_mem_context(output) != queue->context)
    return CL_INVALID_VALUE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, input, filter, previous);
  pending = __vrc_opencl_dispatch_udondnn_large_convolution(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(filter),
      udoncl_mem_physical_token(previous), udoncl_mem_physical_token(output),
      output_words, input_ch, input_w, output_ch, output_w, filter_hw, pad_hw,
      stride_hw, dilation_hw, mode, alpha, beta);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, 0);
}

cl_int __vrc_enqueue_udondnn_large_activation_device_args(
    cl_command_queue queue, cl_mem input, cl_mem previous, cl_mem output,
    cl_int words, cl_int mode, cl_int nan_propagation, cl_float coefficient,
    cl_float alpha, cl_float beta, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !previous ||
      !udoncl_mem_is_active(previous) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(previous) != queue->context ||
      udoncl_mem_context(output) != queue->context ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, input, previous);
  pending = __vrc_opencl_dispatch_udondnn_large_activation(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(previous),
      udoncl_mem_physical_token(output), words, mode, nan_propagation,
      coefficient, alpha, beta);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udondnn_large_pooling_device_args(
    cl_command_queue queue, cl_mem input, cl_mem previous, cl_mem output,
    cl_int output_words, cl_int input_c, cl_int input_h, cl_int input_w,
    cl_int output_h, cl_int output_w, cl_int window_shape, cl_int pad_shape,
    cl_int stride_shape, cl_int options, cl_float alpha, cl_float beta,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !previous ||
      !udoncl_mem_is_active(previous) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(previous) != queue->context ||
      udoncl_mem_context(output) != queue->context ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, input, previous);
  pending = __vrc_opencl_dispatch_udondnn_large_pooling(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(previous),
      udoncl_mem_physical_token(output), output_words, input_c, input_h,
      input_w, output_h, output_w, window_shape, pad_shape, stride_shape,
      options, alpha, beta);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udondnn_large_op_tensor_device_args(
    cl_command_queue queue, cl_mem a, cl_mem b, cl_mem previous, cl_mem output,
    cl_int words, cl_int a_nc, cl_int a_hw, cl_int b_nc, cl_int b_hw,
    cl_int c_nc, cl_int c_hw, cl_int options, cl_float alpha_a,
    cl_float alpha_b, cl_float beta, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !a ||
      !udoncl_mem_is_active(a) || !b || !udoncl_mem_is_active(b) || !previous ||
      !udoncl_mem_is_active(previous) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(a) != queue->context ||
      udoncl_mem_context(b) != queue->context ||
      udoncl_mem_context(previous) != queue->context ||
      udoncl_mem_context(output) != queue->context ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, a, b, previous);
  pending = __vrc_opencl_dispatch_udondnn_large_op_tensor(
      udoncl_mem_physical_token(a), udoncl_mem_physical_token(b),
      udoncl_mem_physical_token(previous), udoncl_mem_physical_token(output),
      words, a_nc, a_hw, b_nc, b_hw, c_nc, c_hw, options, alpha_a, alpha_b,
      beta);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udonaten_arena_rms_norm_reduce_device_args(
    cl_command_queue queue, cl_mem input, cl_mem partials, cl_int words,
    cl_int normalized_words, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event) {
  cl_int pending;
  size_t rows, partials_per_row;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !partials ||
      !udoncl_mem_is_active(partials) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(partials) != queue->context || words < 1 ||
      words > 16384 || normalized_words < 1 || normalized_words > 4096 ||
      words % normalized_words != 0 ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  rows = (size_t)words / (size_t)normalized_words;
  partials_per_row = ((size_t)normalized_words + 63u) / 64u;
  if (rows * partials_per_row > 16384u)
    return CL_INVALID_VALUE;
  if (udoncl_mem_bytes(input) < (size_t)words * 4u ||
      udoncl_mem_bytes(partials) < 65536u)
    return CL_INVALID_BUFFER_SIZE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(partials, input);
  pending = __vrc_opencl_dispatch_udonaten_arena_rms_norm_reduce(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(partials),
      words, normalized_words);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udonaten_arena_rms_norm_finalize_device_args(
    cl_command_queue queue, cl_mem input, cl_mem partials, cl_mem weight,
    cl_mem output, cl_int words, cl_int normalized_words, cl_float epsilon,
    cl_int has_weight, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !partials ||
      !udoncl_mem_is_active(partials) || !weight ||
      !udoncl_mem_is_active(weight) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(partials) != queue->context ||
      udoncl_mem_context(weight) != queue->context ||
      udoncl_mem_context(output) != queue->context || words < 1 ||
      words > 16384 || normalized_words < 1 || normalized_words > 4096 ||
      words % normalized_words != 0 || epsilon < 0.0f || has_weight != 1 ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  if (udoncl_mem_bytes(input) < (size_t)words * 4u ||
      udoncl_mem_bytes(partials) < 65536u ||
      udoncl_mem_bytes(weight) < (size_t)normalized_words * 4u ||
      udoncl_mem_bytes(output) < (size_t)words * 4u)
    return CL_INVALID_BUFFER_SIZE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, input, partials, weight);
  pending = __vrc_opencl_dispatch_udonaten_arena_rms_norm_finalize(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(partials),
      udoncl_mem_physical_token(weight), udoncl_mem_physical_token(output),
      words, normalized_words, epsilon, has_weight);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udondnn_arena_softmax_device_args(
    cl_command_queue queue, cl_mem input, cl_mem output, cl_int input_n,
    cl_int input_c, cl_int input_h, cl_int input_w, cl_int algorithm,
    cl_int mode, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event) {
  cl_int pending;
  size_t words;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(output) != queue->context || input_n < 1 ||
      input_c < 1 || input_c > 512 || input_h < 1 || input_w < 1 ||
      algorithm < 0 || algorithm > 2 || mode < 0 || mode > 1 ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  words = (size_t)input_n * (size_t)input_c;
  if (words > 16384u / (size_t)input_h)
    return CL_INVALID_VALUE;
  words *= (size_t)input_h;
  if (words > 16384u / (size_t)input_w)
    return CL_INVALID_VALUE;
  words *= (size_t)input_w;
  if (udoncl_mem_bytes(input) < words * 4u ||
      udoncl_mem_bytes(output) < words * 4u)
    return CL_INVALID_BUFFER_SIZE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, input);
  pending = __vrc_opencl_dispatch_udondnn_arena_softmax(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      input_n, input_c, input_h, input_w, algorithm, mode);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}
#endif

#if UDONCL_AOT_ENABLE_ONNX
cl_int __vrc_enqueue_udononnx_matmul_nbits_q4_device_args(
    cl_command_queue queue, cl_mem activation, cl_mem packed_weight,
    cl_mem scales, cl_mem previous, cl_mem output, cl_int m, cl_int n, cl_int k,
    cl_int block_size, cl_int n_base, cl_int k_base, cl_int tile_n,
    cl_int tile_k, cl_int accumulate, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  size_t blocks;
  if (queue != &udoncl_queue || !queue->active || !activation ||
      !udoncl_mem_is_active(activation) || !packed_weight ||
      !udoncl_mem_is_active(packed_weight) || !scales ||
      !udoncl_mem_is_active(scales) || !previous ||
      !udoncl_mem_is_active(previous) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(activation) != queue->context ||
      udoncl_mem_context(packed_weight) != queue->context ||
      udoncl_mem_context(scales) != queue->context ||
      udoncl_mem_context(previous) != queue->context ||
      udoncl_mem_context(output) != queue->context || m < 1 || m > 4 || n < 1 ||
      n > 32000 || k < 1 || k > 5632 || block_size != 32 || n_base != 0 ||
      k_base < 0 || k_base > 5504 || tile_n < 1 || tile_n > 32000 ||
      tile_k < 1 || tile_k > 128 || accumulate < 0 || accumulate > 1 ||
      n_base > n - tile_n || k_base > k - tile_k || m > 131072 / n ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  blocks = ((size_t)k + 31u) / 32u;
  if (udoncl_mem_bytes(activation) < (size_t)m * (size_t)k * 4u ||
      udoncl_mem_bytes(packed_weight) < (size_t)n * blocks * 16u ||
      udoncl_mem_bytes(scales) < (size_t)n * blocks * 4u ||
      udoncl_mem_bytes(previous) < 131072u * 4u ||
      udoncl_mem_bytes(output) < 131072u * 4u)
    return CL_INVALID_BUFFER_SIZE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, activation, packed_weight, scales,
                                    previous);
  pending = __vrc_opencl_dispatch_udononnx_matmul_nbits_q4(
      udoncl_mem_physical_token(activation),
      udoncl_mem_physical_token(packed_weight),
      udoncl_mem_physical_token(scales), udoncl_mem_physical_token(previous),
      udoncl_mem_physical_token(output), m, n, k, block_size, n_base, k_base,
      tile_n, tile_k, accumulate);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udononnx_unary_device_args(cl_command_queue queue,
                                                cl_mem input, cl_mem output,
                                                cl_int operation, cl_int words,
                                                cl_uint wait_count,
                                                const cl_event *wait_list,
                                                cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(output) != queue->context || operation < 1 ||
      operation > 3 || words <= 0 || words > 16384 ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, input);
  pending = __vrc_opencl_dispatch_udononnx_unary_f32(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      operation, words);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udononnx_transpose_device_args(
    cl_command_queue queue, cl_mem input, cl_mem output, cl_int words,
    cl_int rank, cl_int shape01, cl_int shape23, cl_int shape4,
    cl_int permutation, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event) {
  cl_int pending;
  size_t elements = 1u;
  cl_uint packed_shapes[3];
  cl_int axis;
  cl_uint seen = 0u;
  packed_shapes[0] = (cl_uint)shape01;
  packed_shapes[1] = (cl_uint)shape23;
  packed_shapes[2] = (cl_uint)shape4;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(output) != queue->context || words < 1 ||
      words > 16384 || rank < 1 || rank > 5 ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  for (axis = 0; axis < rank; ++axis) {
    cl_uint extent =
        (packed_shapes[(cl_uint)axis / 2u] >> ((axis & 1) * 16)) & 65535u;
    cl_uint input_axis = ((cl_uint)permutation >> (axis * 3)) & 7u;
    if (!extent || input_axis >= (cl_uint)rank || (seen & (1u << input_axis)) ||
        elements > 16384u / extent)
      return CL_INVALID_VALUE;
    seen |= 1u << input_axis;
    elements *= extent;
  }
  if (elements != (size_t)words || udoncl_mem_bytes(input) < elements * 4u ||
      udoncl_mem_bytes(output) < elements * 4u)
    return CL_INVALID_BUFFER_SIZE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, input);
  pending = __vrc_opencl_dispatch_udononnx_transpose_f32(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      words, rank, shape01, shape23, shape4, permutation);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udononnx_slice_device_args(
    cl_command_queue queue, cl_mem input, cl_mem output, cl_int words,
    cl_int rank, cl_int input_shape01, cl_int input_shape23,
    cl_int input_shape4, cl_int output_shape01, cl_int output_shape23,
    cl_int output_shape4, cl_int starts01, cl_int starts23, cl_int starts4,
    cl_int steps01, cl_int steps23, cl_int steps4, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  cl_uint input_shapes[3];
  cl_uint output_shapes[3];
  cl_uint packed_starts[3];
  cl_uint packed_steps[3];
  size_t input_elements = 1u;
  size_t output_elements = 1u;
  cl_int axis;
  input_shapes[0] = (cl_uint)input_shape01;
  input_shapes[1] = (cl_uint)input_shape23;
  input_shapes[2] = (cl_uint)input_shape4;
  output_shapes[0] = (cl_uint)output_shape01;
  output_shapes[1] = (cl_uint)output_shape23;
  output_shapes[2] = (cl_uint)output_shape4;
  packed_starts[0] = (cl_uint)starts01;
  packed_starts[1] = (cl_uint)starts23;
  packed_starts[2] = (cl_uint)starts4;
  packed_steps[0] = (cl_uint)steps01;
  packed_steps[1] = (cl_uint)steps23;
  packed_steps[2] = (cl_uint)steps4;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(output) != queue->context || words < 1 ||
      words > 16384 || rank < 1 || rank > 5 ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  for (axis = 0; axis < rank; ++axis) {
    cl_uint shift = (cl_uint)(axis & 1) * 16u;
    cl_uint input_extent = (input_shapes[(cl_uint)axis / 2u] >> shift) & 65535u;
    cl_uint output_extent =
        (output_shapes[(cl_uint)axis / 2u] >> shift) & 65535u;
    cl_uint start = (packed_starts[(cl_uint)axis / 2u] >> shift) & 65535u;
    cl_uint step = (packed_steps[(cl_uint)axis / 2u] >> shift) & 65535u;
    size_t last;
    if (!input_extent || !output_extent || !step ||
        input_elements > 16384u / input_extent ||
        output_elements > 16384u / output_extent)
      return CL_INVALID_VALUE;
    last = (size_t)start + ((size_t)output_extent - 1u) * (size_t)step;
    if (start >= input_extent || last >= input_extent)
      return CL_INVALID_VALUE;
    input_elements *= input_extent;
    output_elements *= output_extent;
  }
  if (output_elements != (size_t)words)
    return CL_INVALID_VALUE;
  if (udoncl_mem_bytes(input) < input_elements * 4u ||
      udoncl_mem_bytes(output) < output_elements * 4u)
    return CL_INVALID_BUFFER_SIZE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, input);
  pending = __vrc_opencl_dispatch_udononnx_slice_f32(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      words, rank, input_shape01, input_shape23, input_shape4, output_shape01,
      output_shape23, output_shape4, starts01, starts23, starts4, steps01,
      steps23, steps4);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udononnx_expand_device_args(
    cl_command_queue queue, cl_mem input, cl_mem output, cl_int words,
    cl_int input_rank, cl_int input_shape01, cl_int input_shape23,
    cl_int input_shape4, cl_int output_rank, cl_int output_shape01,
    cl_int output_shape23, cl_int output_shape4, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  cl_uint input_shapes[3];
  cl_uint output_shapes[3];
  size_t input_elements = 1u;
  size_t output_elements = 1u;
  cl_int output_axis;
  cl_int rank_offset;
  input_shapes[0] = (cl_uint)input_shape01;
  input_shapes[1] = (cl_uint)input_shape23;
  input_shapes[2] = (cl_uint)input_shape4;
  output_shapes[0] = (cl_uint)output_shape01;
  output_shapes[1] = (cl_uint)output_shape23;
  output_shapes[2] = (cl_uint)output_shape4;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(output) != queue->context || words < 1 ||
      words > 16384 || input_rank < 0 || input_rank > 5 ||
      output_rank < input_rank || output_rank < 1 || output_rank > 5 ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  rank_offset = output_rank - input_rank;
  for (output_axis = 0; output_axis < output_rank; ++output_axis) {
    cl_uint shift = (cl_uint)(output_axis & 1) * 16u;
    cl_uint output_extent =
        (output_shapes[(cl_uint)output_axis / 2u] >> shift) & 65535u;
    cl_int input_axis = output_axis - rank_offset;
    if (!output_extent || output_elements > 16384u / output_extent)
      return CL_INVALID_VALUE;
    output_elements *= output_extent;
    if (input_axis >= 0) {
      shift = (cl_uint)(input_axis & 1) * 16u;
      cl_uint input_extent =
          (input_shapes[(cl_uint)input_axis / 2u] >> shift) & 65535u;
      if (!input_extent ||
          (input_extent != 1u && input_extent != output_extent) ||
          input_elements > 16384u / input_extent)
        return CL_INVALID_VALUE;
      input_elements *= input_extent;
    }
  }
  if (output_elements != (size_t)words)
    return CL_INVALID_VALUE;
  if (udoncl_mem_bytes(input) < input_elements * 4u ||
      udoncl_mem_bytes(output) < output_elements * 4u)
    return CL_INVALID_BUFFER_SIZE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, input);
  pending = __vrc_opencl_dispatch_udononnx_expand_f32(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      words, input_rank, input_shape01, input_shape23, input_shape4,
      output_rank, output_shape01, output_shape23, output_shape4);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udononnx_concat2_device_args(
    cl_command_queue queue, cl_mem left, cl_mem right, cl_mem output,
    cl_int words, cl_int inner_words, cl_int left_extent, cl_int right_extent,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  size_t left_block;
  size_t right_block;
  size_t output_block;
  size_t outer;
  if (queue != &udoncl_queue || !queue->active || !left ||
      !udoncl_mem_is_active(left) || !right || !udoncl_mem_is_active(right) ||
      !output || !udoncl_mem_is_active(output) ||
      udoncl_mem_context(left) != queue->context ||
      udoncl_mem_context(right) != queue->context ||
      udoncl_mem_context(output) != queue->context || words < 1 ||
      words > 16384 || inner_words < 1 || left_extent < 1 || right_extent < 1 ||
      (size_t)inner_words > 16384u / (size_t)left_extent ||
      (size_t)inner_words > 16384u / (size_t)right_extent ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  left_block = (size_t)inner_words * (size_t)left_extent;
  right_block = (size_t)inner_words * (size_t)right_extent;
  if (left_block > 16384u - right_block)
    return CL_INVALID_VALUE;
  output_block = left_block + right_block;
  if ((size_t)words % output_block)
    return CL_INVALID_VALUE;
  outer = (size_t)words / output_block;
  if (udoncl_mem_bytes(left) < outer * left_block * 4u ||
      udoncl_mem_bytes(right) < outer * right_block * 4u ||
      udoncl_mem_bytes(output) < (size_t)words * 4u)
    return CL_INVALID_BUFFER_SIZE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, left, right);
  pending = __vrc_opencl_dispatch_udononnx_concat2_f32(
      udoncl_mem_physical_token(left), udoncl_mem_physical_token(right),
      udoncl_mem_physical_token(output), words, inner_words, left_extent,
      right_extent);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udononnx_gather_axis0_device_args(
    cl_command_queue queue, cl_mem data, cl_mem indices, cl_mem output,
    cl_int words, cl_int inner_words, cl_int axis_extent, cl_int index_words,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  size_t index_count;
  size_t data_words;
  if (queue != &udoncl_queue || !queue->active || !data ||
      !udoncl_mem_is_active(data) || !indices ||
      !udoncl_mem_is_active(indices) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(data) != queue->context ||
      udoncl_mem_context(indices) != queue->context ||
      udoncl_mem_context(output) != queue->context || words < 1 ||
      words > 16384 || inner_words < 1 || axis_extent < 1 ||
      (index_words != 1 && index_words != 2) || words % inner_words ||
      (size_t)axis_extent > (size_t)-1 / (size_t)inner_words ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  index_count = (size_t)words / (size_t)inner_words;
  data_words = (size_t)axis_extent * (size_t)inner_words;
  if (data_words > (size_t)-1 / 4u ||
      udoncl_mem_bytes(data) < data_words * 4u ||
      udoncl_mem_bytes(indices) < index_count * (size_t)index_words * 4u ||
      udoncl_mem_bytes(output) < (size_t)words * 4u)
    return CL_INVALID_BUFFER_SIZE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, data, indices);
  pending = __vrc_opencl_dispatch_udononnx_gather_axis0_f32(
      udoncl_mem_physical_token(data), udoncl_mem_physical_token(indices),
      udoncl_mem_physical_token(output), words, inner_words, axis_extent,
      index_words);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udononnx_cast_i64_f32_device_args(
    cl_command_queue queue, cl_mem input, cl_mem output, cl_int words,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !input ||
      !udoncl_mem_is_active(input) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(input) != queue->context ||
      udoncl_mem_context(output) != queue->context || words < 1 ||
      words > 16384 || udoncl_mem_bytes(input) < (size_t)words * 8u ||
      udoncl_mem_bytes(output) < (size_t)words * 4u ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, input);
  pending = __vrc_opencl_dispatch_udononnx_cast_i64_f32(
      udoncl_mem_physical_token(input), udoncl_mem_physical_token(output),
      words);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

static size_t udoncl_broadcast_elements(cl_int input_rank,
                                        const cl_uint input_shapes[3],
                                        cl_int output_rank,
                                        const cl_uint output_shapes[3]) {
  size_t input_elements = 1u;
  if (input_rank < 0 || input_rank > output_rank || output_rank < 0 ||
      output_rank > 5)
    return 0u;
  for (cl_int axis = 0; axis < output_rank; ++axis) {
    cl_uint shift = (cl_uint)(axis & 1) * 16u;
    cl_uint output_extent =
        (output_shapes[(cl_uint)axis / 2u] >> shift) & 65535u;
    cl_int input_axis = axis - (output_rank - input_rank);
    if (!output_extent)
      return 0u;
    if (input_axis >= 0) {
      shift = (cl_uint)(input_axis & 1) * 16u;
      cl_uint input_extent =
          (input_shapes[(cl_uint)input_axis / 2u] >> shift) & 65535u;
      if (!input_extent ||
          (input_extent != 1u && input_extent != output_extent) ||
          input_elements > 16384u / input_extent)
        return 0u;
      input_elements *= input_extent;
    }
  }
  return input_elements;
}

static size_t udoncl_packed_shape_elements(cl_int rank,
                                           const cl_uint shapes[3]) {
  size_t elements = 1u;
  if (rank < 0 || rank > 5)
    return 0u;
  for (cl_int axis = 0; axis < rank; ++axis) {
    cl_uint shift = (cl_uint)(axis & 1) * 16u;
    cl_uint extent = (shapes[(cl_uint)axis / 2u] >> shift) & 65535u;
    if (!extent || elements > 16384u / extent)
      return 0u;
    elements *= extent;
  }
  return elements;
}

cl_int __vrc_enqueue_udononnx_equal_f32_device_args(
    cl_command_queue queue, cl_mem left, cl_mem right, cl_mem output,
    cl_int words, cl_int ranks, cl_int left_shape01, cl_int left_shape23,
    cl_int right_shape01, cl_int right_shape23, cl_int output_shape01,
    cl_int output_shape23, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event) {
  cl_int pending;
  cl_int left_rank = ranks & 7;
  cl_int right_rank = (ranks >> 3) & 7;
  cl_int output_rank = (ranks >> 6) & 7;
  const cl_uint left_shapes[3] = {(cl_uint)left_shape01, (cl_uint)left_shape23,
                                  1u};
  const cl_uint right_shapes[3] = {(cl_uint)right_shape01,
                                   (cl_uint)right_shape23, 1u};
  const cl_uint output_shapes[3] = {(cl_uint)output_shape01,
                                    (cl_uint)output_shape23, 1u};
  size_t left_elements = udoncl_broadcast_elements(left_rank, left_shapes,
                                                   output_rank, output_shapes);
  size_t right_elements = udoncl_broadcast_elements(right_rank, right_shapes,
                                                    output_rank, output_shapes);
  if (queue != &udoncl_queue || !queue->active || !left ||
      !udoncl_mem_is_active(left) || !right || !udoncl_mem_is_active(right) ||
      !output || !udoncl_mem_is_active(output) ||
      udoncl_mem_context(left) != queue->context ||
      udoncl_mem_context(right) != queue->context ||
      udoncl_mem_context(output) != queue->context || words < 1 ||
      words > 16384 || !left_elements || !right_elements ||
      udoncl_packed_shape_elements(output_rank, output_shapes) !=
          (size_t)words ||
      udoncl_mem_bytes(left) < left_elements * 4u ||
      udoncl_mem_bytes(right) < right_elements * 4u ||
      udoncl_mem_bytes(output) < (size_t)words * 4u ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, left, right);
  pending = __vrc_opencl_dispatch_udononnx_equal_f32(
      udoncl_mem_physical_token(left), udoncl_mem_physical_token(right),
      udoncl_mem_physical_token(output), words, ranks, left_shape01,
      left_shape23, right_shape01, right_shape23, output_shape01,
      output_shape23);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udononnx_where_f32_device_args(
    cl_command_queue queue, cl_mem condition, cl_mem when_true,
    cl_mem when_false, cl_mem output, cl_int words, cl_int ranks,
    cl_int condition_shape01, cl_int condition_shape23, cl_int true_shape01,
    cl_int true_shape23, cl_int false_shape01, cl_int false_shape23,
    cl_int output_shape01, cl_int output_shape23, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  cl_int condition_rank = ranks & 7;
  cl_int true_rank = (ranks >> 3) & 7;
  cl_int false_rank = (ranks >> 6) & 7;
  cl_int output_rank = (ranks >> 9) & 7;
  const cl_uint condition_shapes[3] = {(cl_uint)condition_shape01,
                                       (cl_uint)condition_shape23, 1u};
  const cl_uint true_shapes[3] = {(cl_uint)true_shape01, (cl_uint)true_shape23,
                                  1u};
  const cl_uint false_shapes[3] = {(cl_uint)false_shape01,
                                   (cl_uint)false_shape23, 1u};
  const cl_uint output_shapes[3] = {(cl_uint)output_shape01,
                                    (cl_uint)output_shape23, 1u};
  size_t condition_elements = udoncl_broadcast_elements(
      condition_rank, condition_shapes, output_rank, output_shapes);
  size_t true_elements = udoncl_broadcast_elements(true_rank, true_shapes,
                                                   output_rank, output_shapes);
  size_t false_elements = udoncl_broadcast_elements(false_rank, false_shapes,
                                                    output_rank, output_shapes);
  if (queue != &udoncl_queue || !queue->active || !condition ||
      !udoncl_mem_is_active(condition) || !when_true ||
      !udoncl_mem_is_active(when_true) || !when_false ||
      !udoncl_mem_is_active(when_false) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(condition) != queue->context ||
      udoncl_mem_context(when_true) != queue->context ||
      udoncl_mem_context(when_false) != queue->context ||
      udoncl_mem_context(output) != queue->context || words < 1 ||
      words > 16384 || !condition_elements || !true_elements ||
      !false_elements ||
      udoncl_packed_shape_elements(output_rank, output_shapes) !=
          (size_t)words ||
      udoncl_mem_bytes(condition) < condition_elements * 4u ||
      udoncl_mem_bytes(when_true) < true_elements * 4u ||
      udoncl_mem_bytes(when_false) < false_elements * 4u ||
      udoncl_mem_bytes(output) < (size_t)words * 4u ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, condition, when_true, when_false);
  pending = __vrc_opencl_dispatch_udononnx_where_f32(
      udoncl_mem_physical_token(condition),
      udoncl_mem_physical_token(when_true),
      udoncl_mem_physical_token(when_false), udoncl_mem_physical_token(output),
      words, ranks, condition_shape01, condition_shape23, true_shape01,
      true_shape23, false_shape01, false_shape23, output_shape01,
      output_shape23);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udononnx_scatter_nd_f32_device_args(
    cl_command_queue queue, cl_mem data, cl_mem indices, cl_mem updates,
    cl_mem output, cl_int words, cl_int rank, cl_int shape01, cl_int shape23,
    cl_int tuple_rank, cl_int update_count, cl_int slice_words,
    cl_int index_words, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event) {
  cl_int pending;
  const cl_uint shapes[3] = {(cl_uint)shape01, (cl_uint)shape23, 1u};
  size_t index_elements;
  size_t update_elements;
  if (rank < 1 || rank > 4 || tuple_rank < 1 || tuple_rank > rank ||
      update_count < 1 || update_count > 256 || slice_words < 1 ||
      (index_words != 1 && index_words != 2) ||
      (size_t)update_count > 16384u / (size_t)slice_words)
    return CL_INVALID_VALUE;
  index_elements = (size_t)update_count * (size_t)tuple_rank;
  update_elements = (size_t)update_count * (size_t)slice_words;
  if (queue != &udoncl_queue || !queue->active || !data ||
      !udoncl_mem_is_active(data) || !indices ||
      !udoncl_mem_is_active(indices) || !updates ||
      !udoncl_mem_is_active(updates) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(data) != queue->context ||
      udoncl_mem_context(indices) != queue->context ||
      udoncl_mem_context(updates) != queue->context ||
      udoncl_mem_context(output) != queue->context || words < 1 ||
      words > 16384 ||
      udoncl_packed_shape_elements(rank, shapes) != (size_t)words ||
      udoncl_mem_bytes(data) < (size_t)words * 4u ||
      udoncl_mem_bytes(indices) < index_elements * (size_t)index_words * 4u ||
      udoncl_mem_bytes(updates) < update_elements * 4u ||
      udoncl_mem_bytes(output) < (size_t)words * 4u ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, data, indices, updates);
  pending = __vrc_opencl_dispatch_udononnx_scatter_nd_f32(
      udoncl_mem_physical_token(data), udoncl_mem_physical_token(indices),
      udoncl_mem_physical_token(updates), udoncl_mem_physical_token(output),
      words, rank, shape01, shape23, tuple_rank, update_count, slice_words,
      index_words);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

#endif

#define UDONCL_DEFINE_DIRECT_ENQUEUE(function_name, expected_id, dispatcher)   \
  CL_API_ENTRY cl_int CL_API_CALL function_name(                               \
      cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,              \
      const size_t *global_offset, const size_t *global_size,                  \
      const size_t *local_size, cl_uint wait_count, const cl_event *wait_list, \
      cl_event *event) {                                                       \
    if (queue != &udoncl_queue || !queue->active || !kernel ||                 \
        !kernel->active || kernel->id != expected_id || !global_size ||        \
        !local_size ||                                                         \
        udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)          \
      return CL_INVALID_VALUE;                                                 \
    udoncl_pending_resources_reset();                                          \
    udoncl_dispatch_status = CL_SUCCESS;                                       \
    udoncl_dispatch_event = CL_SUCCESS;                                        \
    dispatcher(kernel, work_dim, global_offset, global_size, local_size);      \
    return udoncl_finish_dispatch(queue, event);                               \
  }

#if UDONCL_AOT_ENABLE_BLAS
cl_int __vrc_enqueue_udonblas_arena_sgemm_device_args(
    cl_command_queue queue, cl_mem a, cl_mem b, cl_mem output, cl_int m,
    cl_int n, cl_int k, cl_int transa, cl_int transb, cl_float alpha,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  size_t a_words;
  size_t b_words;
  size_t output_words;
  if (queue != &udoncl_queue || !queue->active || !a ||
      !udoncl_mem_is_active(a) || !b || !udoncl_mem_is_active(b) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(a) != queue->context ||
      udoncl_mem_context(b) != queue->context ||
      udoncl_mem_context(output) != queue->context ||
      udoncl_mem_is_subbuffer(a) || udoncl_mem_is_subbuffer(b) ||
      udoncl_mem_is_subbuffer(output) || m <= 0 || n <= 0 || k <= 0 ||
      k > 256 || udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  a_words = (size_t)m * (size_t)k;
  b_words = (size_t)k * (size_t)n;
  output_words = (size_t)m * (size_t)n;
  if (a_words > 16384u || b_words > 16384u || output_words > 16384u ||
      udoncl_mem_bytes(a) < a_words * 4u ||
      udoncl_mem_bytes(b) < b_words * 4u ||
      udoncl_mem_bytes(output) < output_words * 4u)
    return CL_INVALID_BUFFER_SIZE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, a, b);
  udoncl_pending_resources_reset();
  udoncl_pending_resource_add(a);
  udoncl_pending_resource_add(b);
  udoncl_pending_resource_add(output);
  pending = __vrc_opencl_dispatch_udonblas_arena_sgemm(
      udoncl_mem_physical_token(a), udoncl_mem_physical_token(b),
      udoncl_mem_physical_token(output), m, n, k, transa, transb, alpha);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udonblas_arena_sgemm_strided_batched_device_args(
    cl_command_queue queue, cl_mem a, cl_mem b, cl_mem output, cl_int m,
    cl_int n, cl_int k, cl_int stride_a, cl_int stride_b, cl_int stride_c,
    cl_int batch_count, cl_int transa, cl_int transb, cl_float alpha,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  size_t matrix_a;
  size_t matrix_b;
  size_t matrix_c;
  size_t required_a;
  size_t required_b;
  size_t required_c;
  if (queue != &udoncl_queue || !queue->active || !a ||
      !udoncl_mem_is_active(a) || !b || !udoncl_mem_is_active(b) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(a) != queue->context ||
      udoncl_mem_context(b) != queue->context ||
      udoncl_mem_context(output) != queue->context ||
      udoncl_mem_is_subbuffer(a) || udoncl_mem_is_subbuffer(b) ||
      udoncl_mem_is_subbuffer(output) || m < 1 || n < 1 || k < 1 || k > 256 ||
      stride_a < 0 || stride_b < 0 || stride_c < 1 || batch_count < 1 ||
      transa < 0 || transa > 1 || transb < 0 || transb > 1 ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  matrix_a = (size_t)m * (size_t)k;
  matrix_b = (size_t)k * (size_t)n;
  matrix_c = (size_t)m * (size_t)n;
  if (matrix_a > 131072u || matrix_b > 131072u || matrix_c > 16384u ||
      (size_t)batch_count > 16384u / matrix_c || (size_t)stride_c != matrix_c)
    return CL_INVALID_VALUE;
  required_a =
      matrix_a + (stride_a ? (size_t)(batch_count - 1) * (size_t)stride_a : 0u);
  required_b =
      matrix_b + (stride_b ? (size_t)(batch_count - 1) * (size_t)stride_b : 0u);
  required_c = (size_t)batch_count * matrix_c;
  if (required_a > 131072u || required_b > 131072u || required_c > 16384u)
    return CL_INVALID_VALUE;
  if (udoncl_mem_bytes(a) < required_a * 4u ||
      udoncl_mem_bytes(b) < required_b * 4u ||
      udoncl_mem_bytes(output) < required_c * 4u)
    return CL_INVALID_BUFFER_SIZE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, a, b);
  udoncl_pending_resources_reset();
  udoncl_pending_resource_add(a);
  udoncl_pending_resource_add(b);
  udoncl_pending_resource_add(output);
  pending = __vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched(
      udoncl_mem_physical_token(a), udoncl_mem_physical_token(b),
      udoncl_mem_physical_token(output), m, n, k, stride_a, stride_b, stride_c,
      batch_count, transa, transb, alpha);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, event);
}

cl_int __vrc_enqueue_udonblas_arena_sgemm_args(cl_command_queue queue, cl_mem a,
                                               cl_mem b, cl_mem output,
                                               cl_int m, cl_int n, cl_int k,
                                               cl_int transa, cl_int transb,
                                               cl_float alpha) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !a ||
      !udoncl_mem_is_active(a) || !b || !udoncl_mem_is_active(b) || !output ||
      !udoncl_mem_is_active(output) ||
      udoncl_mem_context(a) != queue->context ||
      udoncl_mem_context(b) != queue->context ||
      udoncl_mem_context(output) != queue->context || m <= 0 || n <= 0 ||
      k <= 0)
    return CL_INVALID_VALUE;
  UDONCL_RETURN_IF_READ_WRITE_ALIAS(output, a, b);
  pending = __vrc_opencl_dispatch_udonblas_arena_sgemm(
      udoncl_mem_physical_token(a), udoncl_mem_physical_token(b),
      udoncl_mem_physical_token(output), m, n, k, transa, transb, alpha);
  return udoncl_finish_enqueue(queue, CL_COMMAND_NDRANGE_KERNEL, pending,
                               CL_FALSE, 0);
}

UDONCL_DEFINE_DIRECT_ENQUEUE(__vrc_enqueue_udonblas_sgemm, 1,
                             udoncl_dispatch_sgemm)
UDONCL_DEFINE_DIRECT_ENQUEUE(__vrc_enqueue_udonblas_sgemv, 2,
                             udoncl_dispatch_sgemv)
#endif
#if UDONCL_AOT_ENABLE_DNN
UDONCL_DEFINE_DIRECT_ENQUEUE(__vrc_enqueue_udondnn_large_pooling, 231,
                             udoncl_dispatch_large_pooling)
UDONCL_DEFINE_DIRECT_ENQUEUE(__vrc_enqueue_udondnn_large_convolution, 232,
                             udoncl_dispatch_large_convolution)
UDONCL_DEFINE_DIRECT_ENQUEUE(__vrc_enqueue_udondnn_large_activation, 233,
                             udoncl_dispatch_large_activation)
UDONCL_DEFINE_DIRECT_ENQUEUE(__vrc_enqueue_udondnn_large_op_tensor, 234,
                             udoncl_dispatch_large_op_tensor)
#endif
#undef UDONCL_DEFINE_DIRECT_ENQUEUE

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(
    cl_command_queue queue, cl_mem buffer, cl_bool blocking, size_t offset,
    size_t bytes, const void *source, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active ||
      !udoncl_mem_is_active(buffer) || udoncl_mem_is_static(buffer) ||
      !source || offset > udoncl_mem_bytes(buffer) ||
      bytes > udoncl_mem_bytes(buffer) - offset ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  udoncl_pending_resources_reset();
  udoncl_pending_resource_add(buffer);
  pending = __vrc_opencl_buffer_upload(udoncl_mem_physical_token(buffer),
                                       udoncl_mem_origin(buffer) + offset,
                                       bytes, source);
  return udoncl_finish_enqueue(queue, CL_COMMAND_WRITE_BUFFER, pending,
                               blocking, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(
    cl_command_queue queue, cl_mem buffer, cl_bool blocking, size_t offset,
    size_t bytes, void *destination, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active ||
      !udoncl_mem_is_active(buffer) || !destination ||
      offset > udoncl_mem_bytes(buffer) ||
      bytes > udoncl_mem_bytes(buffer) - offset ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  udoncl_pending_resources_reset();
  udoncl_pending_resource_add(buffer);
  pending = __vrc_opencl_buffer_readback(udoncl_mem_physical_token(buffer),
                                         udoncl_mem_origin(buffer) + offset,
                                         bytes, destination);
  return udoncl_finish_enqueue(queue, CL_COMMAND_READ_BUFFER, pending, blocking,
                               event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBuffer(
    cl_command_queue queue, cl_mem source, cl_mem destination,
    size_t source_offset, size_t destination_offset, size_t bytes,
    cl_uint wait_count, const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !source ||
      !udoncl_mem_is_active(source) || !destination ||
      !udoncl_mem_is_active(destination) ||
      udoncl_mem_context(source) != queue->context ||
      udoncl_mem_context(destination) != queue->context ||
      (udoncl_mem_flags(destination) & CL_MEM_READ_ONLY) || !bytes ||
      source_offset > udoncl_mem_bytes(source) ||
      bytes > udoncl_mem_bytes(source) - source_offset ||
      destination_offset > udoncl_mem_bytes(destination) ||
      bytes > udoncl_mem_bytes(destination) - destination_offset ||
      (udoncl_mem_physical_token(source) ==
           udoncl_mem_physical_token(destination) &&
       udoncl_mem_origin(source) + source_offset <
           udoncl_mem_origin(destination) + destination_offset + bytes &&
       udoncl_mem_origin(destination) + destination_offset <
           udoncl_mem_origin(source) + source_offset + bytes) ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return source && destination &&
                   udoncl_mem_physical_token(source) ==
                       udoncl_mem_physical_token(destination) &&
                   udoncl_mem_origin(source) + source_offset <
                       udoncl_mem_origin(destination) + destination_offset +
                           bytes &&
                   udoncl_mem_origin(destination) + destination_offset <
                       udoncl_mem_origin(source) + source_offset + bytes
               ? CL_MEM_COPY_OVERLAP
               : CL_INVALID_VALUE;
  if (udoncl_mem_context(source) != udoncl_mem_context(destination))
    return CL_INVALID_CONTEXT;
  udoncl_pending_resources_reset();
  udoncl_pending_resource_add(source);
  udoncl_pending_resource_add(destination);
  pending = __vrc_opencl_buffer_copy(
      udoncl_mem_physical_token(source), udoncl_mem_physical_token(destination),
      udoncl_mem_origin(source) + source_offset,
      udoncl_mem_origin(destination) + destination_offset, bytes);
  return udoncl_finish_enqueue(queue, CL_COMMAND_COPY_BUFFER, pending, CL_FALSE,
                               event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(
    cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_offset, const size_t *global_size,
    const size_t *local_size, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event) {
  if (queue != &udoncl_queue || !queue->active || !kernel || !kernel->active ||
      !global_size || !local_size || work_dim < 1 || work_dim > 3 ||
      udoncl_wait_list(queue, wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  udoncl_pending_resources_reset();
  udoncl_dispatch_event = CL_SUCCESS;
#if UDONCL_AOT_ENABLE_BLAS
  if (kernel->id == 1) {
    udoncl_dispatch_status = CL_SUCCESS;
    udoncl_dispatch_sgemm(kernel, work_dim, global_offset, global_size,
                          local_size);
    return udoncl_finish_dispatch(queue, event);
  }
  if (kernel->id == 2) {
    udoncl_dispatch_status = CL_SUCCESS;
    udoncl_dispatch_sgemv(kernel, work_dim, global_offset, global_size,
                          local_size);
    return udoncl_finish_dispatch(queue, event);
  }
#endif
#if UDONCL_AOT_ENABLE_DNN
  if (kernel->id == 231) {
    udoncl_dispatch_status = CL_SUCCESS;
    udoncl_dispatch_large_pooling(kernel, work_dim, global_offset, global_size,
                                  local_size);
    return udoncl_finish_dispatch(queue, event);
  }
  if (kernel->id == 232) {
    udoncl_dispatch_status = CL_SUCCESS;
    udoncl_dispatch_large_convolution(kernel, work_dim, global_offset,
                                      global_size, local_size);
    return udoncl_finish_dispatch(queue, event);
  }
  if (kernel->id == 233) {
    udoncl_dispatch_status = CL_SUCCESS;
    udoncl_dispatch_large_activation(kernel, work_dim, global_offset,
                                     global_size, local_size);
    return udoncl_finish_dispatch(queue, event);
  }
  if (kernel->id == 234) {
    udoncl_dispatch_status = CL_SUCCESS;
    udoncl_dispatch_large_op_tensor(kernel, work_dim, global_offset,
                                    global_size, local_size);
    return udoncl_finish_dispatch(queue, event);
  }
  if (kernel->id == 207) {
    udoncl_dispatch_status = CL_SUCCESS;
    udoncl_dispatch_embedding(kernel, work_dim, global_offset, global_size,
                              local_size);
    return udoncl_finish_dispatch(queue, event);
  }
#endif
#if UDONCL_AOT_ENABLE_ONNX
  if (kernel->id == 219) {
    udoncl_dispatch_status = CL_SUCCESS;
    udoncl_dispatch_matmul_nbits(kernel, work_dim, global_offset);
    return udoncl_finish_dispatch(queue, event);
  }
  if (kernel->id == 220) {
    udoncl_dispatch_status = CL_SUCCESS;
    udoncl_dispatch_gather(kernel, work_dim, global_offset);
    return udoncl_finish_dispatch(queue, event);
  }
  if (kernel->id == 222) {
    udoncl_dispatch_status = CL_SUCCESS;
    udoncl_dispatch_unary(kernel, work_dim, global_offset);
    return udoncl_finish_dispatch(queue, event);
  }
#endif
#if UDONCL_AOT_ENABLE_MNIST
  if (kernel->id >= 20 && kernel->id <= 22) {
    udoncl_dispatch_status = CL_SUCCESS;
    udoncl_dispatch_mnist(kernel, work_dim, global_offset);
    return udoncl_finish_dispatch(queue, event);
  }
#endif
  return CL_INVALID_KERNEL;
}

CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue queue) {
  if (queue != &udoncl_queue || !queue->active)
    return CL_INVALID_COMMAND_QUEUE;
  return queue->tail ? udoncl_event_wait(queue->tail) : CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clFlush(cl_command_queue queue) {
  return queue == &udoncl_queue && queue->active ? CL_SUCCESS
                                                 : CL_INVALID_COMMAND_QUEUE;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMarkerWithWaitList(cl_command_queue queue, cl_uint wait_count,
                            const cl_event *wait_list, cl_event *event) {
  cl_int status;
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !event)
    return CL_INVALID_VALUE;
  status = udoncl_wait_list(queue, wait_count, wait_list);
  if (status != CL_SUCCESS)
    return status;
  pending = queue->tail && queue->tail->execution_status > CL_COMPLETE
                ? queue->tail->pending
                : CL_SUCCESS;
  udoncl_pending_resources_reset();
  return udoncl_event_create(queue, CL_COMMAND_MARKER, pending, event);
}

CL_API_ENTRY cl_int CL_API_CALL clWaitForEvents(cl_uint count,
                                                const cl_event *events) {
  cl_uint index;
  cl_context context = 0;
  if (!count || !events)
    return CL_INVALID_VALUE;
  for (index = 0; index < count; ++index) {
    cl_event event = udoncl_event_find(events[index]);
    cl_int status;
    if (!event)
      return CL_INVALID_EVENT;
    if (!context)
      context = event->queue->context;
    else if (context != event->queue->context)
      return CL_INVALID_CONTEXT;
    status = udoncl_event_wait(event);
    if (status != CL_SUCCESS)
      return status;
  }
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clRetainEvent(cl_event event) {
  event = udoncl_event_find(event);
  if (!event)
    return CL_INVALID_EVENT;
  ++event->refs;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent(cl_event event) {
  event = udoncl_event_find(event);
  if (!event)
    return CL_INVALID_EVENT;
  if (--event->refs)
    return CL_SUCCESS;
  if (!event->internal)
    udoncl_event_destroy(event);
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clGetEventInfo(cl_event event,
                                               cl_event_info param,
                                               size_t value_size, void *value,
                                               size_t *required_size) {
  size_t bytes;
  const void *source;
  cl_context context;
  size_t index;
  event = udoncl_event_find(event);
  if (!event)
    return CL_INVALID_EVENT;
  if (param == CL_EVENT_COMMAND_QUEUE) {
    source = &event->queue;
    bytes = sizeof(event->queue);
  } else if (param == CL_EVENT_CONTEXT) {
    context = event->queue->context;
    source = &context;
    bytes = sizeof(context);
  } else if (param == CL_EVENT_COMMAND_TYPE) {
    source = &event->command_type;
    bytes = sizeof(event->command_type);
  } else if (param == CL_EVENT_COMMAND_EXECUTION_STATUS) {
    source = &event->execution_status;
    bytes = sizeof(event->execution_status);
  } else if (param == CL_EVENT_REFERENCE_COUNT) {
    source = &event->refs;
    bytes = sizeof(event->refs);
  } else {
    return CL_INVALID_VALUE;
  }
  if (required_size)
    *required_size = bytes;
  if (!value)
    return value_size ? CL_INVALID_VALUE : CL_SUCCESS;
  if (value_size < bytes)
    return CL_INVALID_VALUE;
  for (index = 0; index < bytes; ++index)
    ((unsigned char *)value)[index] = ((const unsigned char *)source)[index];
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(cl_kernel kernel) {
  if (!kernel || !kernel->active)
    return CL_INVALID_KERNEL;
  kernel->active = 0;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(cl_program program) {
  if (program != &udoncl_program || !program->active)
    return CL_INVALID_PROGRAM;
  program->active = 0;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem buffer) {
  if (!udoncl_mem_is_active(buffer))
    return CL_INVALID_MEM_OBJECT;
  if (udoncl_mem_is_static(buffer))
    return CL_SUCCESS;
  if (--buffer->refs == 0)
    buffer->active = 0;
  return udoncl_mem_finalize(buffer);
}

CL_API_ENTRY cl_int CL_API_CALL clRetainMemObject(cl_mem buffer) {
  if (!udoncl_mem_is_active(buffer))
    return CL_INVALID_MEM_OBJECT;
  if (udoncl_mem_is_static(buffer))
    return CL_SUCCESS;
  ++buffer->refs;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue queue) {
  if (queue != &udoncl_queue || !queue->active)
    return CL_INVALID_COMMAND_QUEUE;
  if (--queue->refs == 0)
    queue->active = 0;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseContext(cl_context context) {
  if (context != &udoncl_context || !context->active)
    return CL_INVALID_CONTEXT;
  if (--context->refs == 0)
    context->active = 0;
  return CL_SUCCESS;
}
