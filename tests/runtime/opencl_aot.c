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

static cl_uint allocated_logical_id;
static size_t allocated_bytes;
static cl_uint uploaded_token;
static cl_uint released_token;
static cl_uint dispatched_input_token;
static cl_uint dispatched_output_token;
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
static unsigned char physical_buffers[33][65536];

cl_uint __vrc_opencl_buffer_alloc(cl_uint logical_id, size_t bytes) {
  allocated_logical_id = logical_id;
  allocated_bytes = bytes;
  if (!logical_id || logical_id > 32 || bytes > sizeof(physical_buffers[0]))
    return 0;
  return logical_id + 100;
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
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_event marker_event, write_event, wait_events[2];
  cl_mem first, second;
  cl_mem resolved;
  udonclBufferHandle_t common_buffer = {0, 0}, stale_buffer, replacement;
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
  EXPECT(udononnxCreateOperatorHandle(&operator_handle) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxGather(operator_handle, UDONCL_DATA_FLOAT32, gather_data,
                        gather_shape, 2, UDONCL_DATA_INT64, gather_indices, 2,
                        0, gather_output) == UDONONNX_STATUS_SUCCESS);
  EXPECT(gather_output[0] == 5.0f && gather_output[1] == 6.0f &&
         gather_output[2] == 1.0f && gather_output[3] == 2.0f);
  EXPECT(dispatched_gather_words == 2);
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
  return 0;
}
