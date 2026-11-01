#include "udondnn.h"
#include "udondnn_device_source.h"
#include <CL/cl.h>
#include <float.h>

#define UDONDNN_WORDS 2048
#define UDONDNN_REGION_WORDS 512
#define UDONDNN_INPUT_OFFSET 0
#define UDONDNN_FILTER_OFFSET 512
#define UDONDNN_PREVIOUS_OUTPUT_OFFSET 1024
#define UDONDNN_RESULT_OFFSET 1536
#define UDONDNN_ACTIVATION_WORDS 2048
#define UDONDNN_ACTIVATION_PREVIOUS_OUTPUT_OFFSET 512
#define UDONDNN_ACTIVATION_RESULT_OFFSET 1024
#define UDONDNN_POOLING_WORDS 2048
#define UDONDNN_POOLING_PREVIOUS_OUTPUT_OFFSET 512
#define UDONDNN_POOLING_RESULT_OFFSET 1024
#define UDONDNN_SOFTMAX_WORDS 2048
#define UDONDNN_SOFTMAX_PREVIOUS_OUTPUT_OFFSET 512
#define UDONDNN_SOFTMAX_RESULT_OFFSET 1024
#define UDONDNN_BATCHNORM_WORDS 4096
#define UDONDNN_BATCHNORM_PREVIOUS_OUTPUT_OFFSET 512
#define UDONDNN_BATCHNORM_SCALE_OFFSET 1024
#define UDONDNN_BATCHNORM_BIAS_OFFSET 1536
#define UDONDNN_BATCHNORM_MEAN_OFFSET 2048
#define UDONDNN_BATCHNORM_VARIANCE_OFFSET 2560
#define UDONDNN_BATCHNORM_RESULT_OFFSET 3072
#define UDONDNN_OP_TENSOR_B_OFFSET 512
#define UDONDNN_OP_TENSOR_C_OFFSET 1024
#define UDONDNN_OP_TENSOR_RESULT_OFFSET 1536
#define UDONATEN_RMS_SMALL_WEIGHT_OFFSET 512
#define UDONATEN_RMS_SMALL_RESULT_OFFSET 1536
#define UDONATEN_EMBEDDING_INDICES_OFFSET 512
#define UDONATEN_EMBEDDING_RESULT_OFFSET 1024
#define UDONATEN_INT4_WORDS 8192
#define UDONATEN_INT4_INPUT_OFFSET 6144
#define UDONATEN_INT4_RESULT_OFFSET 7168
#define UDONATEN_INT4_MAX_N 64
#define UDONATEN_INT4_MAX_K 128
#define UDONATEN_TILE_RESULT_OFFSET 4096
#define UDONATEN_RMS_WORDS 8192
#define UDONATEN_RMS_WEIGHT_OFFSET 4096
#define UDONATEN_ARENA_POINTWISE_WORDS 16384
#define UDONDNN_DESCRIPTOR_MAX_WORDS 16777216
#define UDONDNN_HANDLE 1u
#define UDONDNN_DESCRIPTOR_SLOTS 64
#define UDONDNN_SLOT_MASK 255u
#define UDONDNN_GENERATION_MASK 0x00ffffffu

typedef struct TensorDescriptor {
  unsigned generation;
  int used;
  int configured;
  int n, c, h, w;
} TensorDescriptor;

typedef struct FilterDescriptor {
  unsigned generation;
  int used;
  int configured;
  int k, c, h, w;
} FilterDescriptor;

typedef struct ConvolutionDescriptor {
  unsigned generation;
  int used;
  int configured;
  int pad_h, pad_w;
  int stride_h, stride_w;
  int dilation_h, dilation_w;
  udondnnConvolutionMode_t mode;
} ConvolutionDescriptor;

typedef struct ActivationDescriptor {
  unsigned generation;
  int used;
  int configured;
  udondnnActivationMode_t mode;
  udondnnNanPropagation_t nan_propagation;
  float coefficient;
} ActivationDescriptor;

typedef struct PoolingDescriptor {
  unsigned generation;
  int used;
  int configured;
  udondnnPoolingMode_t mode;
  udondnnNanPropagation_t nan_propagation;
  int window_h, window_w;
  int pad_h, pad_w;
  int stride_h, stride_w;
} PoolingDescriptor;

typedef struct OpTensorDescriptor {
  unsigned generation;
  int used;
  int configured;
  udondnnOpTensorOp_t operation;
  udondnnNanPropagation_t nan_propagation;
} OpTensorDescriptor;

static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel convolution_kernel;
static cl_kernel activation_kernel;
static cl_kernel pooling_kernel;
static cl_kernel softmax_kernel;
static cl_kernel batchnorm_kernel;
static cl_kernel op_tensor_kernel;
static cl_kernel rms_norm_small_kernel;
static cl_kernel embedding_kernel;
static cl_kernel weight_int4pack_mm_kernel;
static cl_kernel tile_kernel;
static cl_kernel aten_rms_reduce_kernel;
static cl_kernel aten_rms_finalize_kernel;
static cl_kernel arena_pointwise_kernel;
static cl_kernel arena_tile_kernel;
static cl_kernel arena_rms_reduce_kernel;
static cl_kernel arena_rms_finalize_kernel;
static cl_mem input_buffer;
static cl_mem output_buffer;
static cl_mem batchnorm_input_buffer;
static cl_mem batchnorm_output_buffer;
static cl_mem int4_input_buffer;
static cl_mem int4_output_buffer;
static cl_mem aten_rms_state_buffer;
static cl_mem aten_rms_partial_buffer;
static cl_mem aten_rms_output_buffer;
static cl_mem arena_pointwise_self_buffer;
static cl_mem arena_pointwise_other_buffer;
static cl_mem arena_pointwise_output_buffer;
static cl_mem arena_rms_partial_buffer;
static float packed[UDONDNN_WORDS];
static float batchnorm_packed[UDONDNN_BATCHNORM_WORDS];
static float result[UDONDNN_REGION_WORDS];
static float int4_state[UDONATEN_INT4_WORDS];
static float aten_rms_state[UDONATEN_RMS_WORDS];
static float arena_pointwise_zero[UDONATEN_ARENA_POINTWISE_WORDS];
static TensorDescriptor tensors[UDONDNN_DESCRIPTOR_SLOTS];
static FilterDescriptor filters[UDONDNN_DESCRIPTOR_SLOTS];
static ConvolutionDescriptor convolutions[UDONDNN_DESCRIPTOR_SLOTS];
static ActivationDescriptor activations[UDONDNN_DESCRIPTOR_SLOTS];
static PoolingDescriptor poolings[UDONDNN_DESCRIPTOR_SLOTS];
static OpTensorDescriptor op_tensors[UDONDNN_DESCRIPTOR_SLOTS];

static unsigned next_generation(unsigned generation) {
  generation = (generation + 1u) & UDONDNN_GENERATION_MASK;
  return generation ? generation : 1u;
}

static unsigned make_descriptor(unsigned generation, unsigned slot) {
  return (generation << 8u) | (slot + 1u);
}

static int descriptor_slot(unsigned descriptor, unsigned *generation) {
  unsigned encoded_slot = descriptor & UDONDNN_SLOT_MASK;
  if (!encoded_slot || encoded_slot > UDONDNN_DESCRIPTOR_SLOTS)
    return -1;
  *generation = descriptor >> 8u;
  return (int)encoded_slot - 1;
}

static TensorDescriptor *get_tensor(udondnnTensorDescriptor_t descriptor) {
  unsigned generation;
  int slot = descriptor_slot(descriptor, &generation);
  if (slot < 0 || !tensors[slot].used || tensors[slot].generation != generation)
    return 0;
  return &tensors[slot];
}

static FilterDescriptor *get_filter(udondnnFilterDescriptor_t descriptor) {
  unsigned generation;
  int slot = descriptor_slot(descriptor, &generation);
  if (slot < 0 || !filters[slot].used || filters[slot].generation != generation)
    return 0;
  return &filters[slot];
}

static ConvolutionDescriptor *
get_convolution(udondnnConvolutionDescriptor_t descriptor) {
  unsigned generation;
  int slot = descriptor_slot(descriptor, &generation);
  if (slot < 0 || !convolutions[slot].used ||
      convolutions[slot].generation != generation)
    return 0;
  return &convolutions[slot];
}

static ActivationDescriptor *
get_activation(udondnnActivationDescriptor_t descriptor) {
  unsigned generation;
  int slot = descriptor_slot(descriptor, &generation);
  if (slot < 0 || !activations[slot].used ||
      activations[slot].generation != generation)
    return 0;
  return &activations[slot];
}

static PoolingDescriptor *get_pooling(udondnnPoolingDescriptor_t descriptor) {
  unsigned generation;
  int slot = descriptor_slot(descriptor, &generation);
  if (slot < 0 || !poolings[slot].used ||
      poolings[slot].generation != generation)
    return 0;
  return &poolings[slot];
}

static OpTensorDescriptor *
get_op_tensor(udondnnOpTensorDescriptor_t descriptor) {
  unsigned generation;
  int slot = descriptor_slot(descriptor, &generation);
  if (slot < 0 || !op_tensors[slot].used ||
      op_tensors[slot].generation != generation)
    return 0;
  return &op_tensors[slot];
}

static void clear_objects(void) {
  if (arena_rms_finalize_kernel)
    clReleaseKernel(arena_rms_finalize_kernel);
  if (arena_rms_reduce_kernel)
    clReleaseKernel(arena_rms_reduce_kernel);
  if (arena_tile_kernel)
    clReleaseKernel(arena_tile_kernel);
  if (arena_pointwise_kernel)
    clReleaseKernel(arena_pointwise_kernel);
  if (aten_rms_finalize_kernel)
    clReleaseKernel(aten_rms_finalize_kernel);
  if (aten_rms_reduce_kernel)
    clReleaseKernel(aten_rms_reduce_kernel);
  if (tile_kernel)
    clReleaseKernel(tile_kernel);
  if (weight_int4pack_mm_kernel)
    clReleaseKernel(weight_int4pack_mm_kernel);
  if (embedding_kernel)
    clReleaseKernel(embedding_kernel);
  if (rms_norm_small_kernel)
    clReleaseKernel(rms_norm_small_kernel);
  if (op_tensor_kernel)
    clReleaseKernel(op_tensor_kernel);
  if (batchnorm_kernel)
    clReleaseKernel(batchnorm_kernel);
  if (softmax_kernel)
    clReleaseKernel(softmax_kernel);
  if (pooling_kernel)
    clReleaseKernel(pooling_kernel);
  if (activation_kernel)
    clReleaseKernel(activation_kernel);
  if (convolution_kernel)
    clReleaseKernel(convolution_kernel);
  if (program)
    clReleaseProgram(program);
  if (output_buffer)
    clReleaseMemObject(output_buffer);
  if (input_buffer)
    clReleaseMemObject(input_buffer);
  if (batchnorm_output_buffer)
    clReleaseMemObject(batchnorm_output_buffer);
  if (batchnorm_input_buffer)
    clReleaseMemObject(batchnorm_input_buffer);
  if (int4_output_buffer)
    clReleaseMemObject(int4_output_buffer);
  if (int4_input_buffer)
    clReleaseMemObject(int4_input_buffer);
  if (aten_rms_output_buffer)
    clReleaseMemObject(aten_rms_output_buffer);
  if (aten_rms_partial_buffer)
    clReleaseMemObject(aten_rms_partial_buffer);
  if (aten_rms_state_buffer)
    clReleaseMemObject(aten_rms_state_buffer);
  if (arena_pointwise_output_buffer)
    clReleaseMemObject(arena_pointwise_output_buffer);
  if (arena_pointwise_other_buffer)
    clReleaseMemObject(arena_pointwise_other_buffer);
  if (arena_pointwise_self_buffer)
    clReleaseMemObject(arena_pointwise_self_buffer);
  if (arena_rms_partial_buffer)
    clReleaseMemObject(arena_rms_partial_buffer);
  if (queue)
    clReleaseCommandQueue(queue);
  if (context)
    clReleaseContext(context);
  convolution_kernel = 0;
  activation_kernel = 0;
  pooling_kernel = 0;
  softmax_kernel = 0;
  batchnorm_kernel = 0;
  op_tensor_kernel = 0;
  rms_norm_small_kernel = 0;
  embedding_kernel = 0;
  weight_int4pack_mm_kernel = 0;
  tile_kernel = 0;
  aten_rms_reduce_kernel = 0;
  aten_rms_finalize_kernel = 0;
  arena_pointwise_kernel = 0;
  arena_tile_kernel = 0;
  arena_rms_reduce_kernel = 0;
  arena_rms_finalize_kernel = 0;
  program = 0;
  output_buffer = input_buffer = 0;
  batchnorm_output_buffer = batchnorm_input_buffer = 0;
  int4_output_buffer = int4_input_buffer = 0;
  aten_rms_state_buffer = aten_rms_partial_buffer = aten_rms_output_buffer = 0;
  arena_pointwise_self_buffer = arena_pointwise_other_buffer =
      arena_pointwise_output_buffer = 0;
  arena_rms_partial_buffer = 0;
  queue = 0;
  context = 0;
}

static udondnnStatus_t execution_failure(void) {
  clear_objects();
  return UDONDNN_STATUS_EXECUTION_FAILED;
}

udondnnStatus_t udondnnCreate(udondnnHandle_t *handle) {
  cl_platform_id platform;
  cl_device_id device;
  const char *source = UDONDNN_DEVICE_SOURCE;
  cl_int status;
  if (!handle)
    return UDONDNN_STATUS_BAD_PARAM;
  if (context)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, 0) !=
          CL_SUCCESS)
    return execution_failure();
  context = clCreateContext(0, 1, &device, 0, 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  queue = clCreateCommandQueue(context, device, 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  input_buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(packed), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  output_buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(packed), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  batchnorm_input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(batchnorm_packed), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  batchnorm_output_buffer = clCreateBuffer(
      context, CL_MEM_READ_WRITE, sizeof(batchnorm_packed), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  int4_input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     sizeof(int4_state), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  int4_output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(int4_state), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  aten_rms_state_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         sizeof(aten_rms_state), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  aten_rms_partial_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           sizeof(aten_rms_state), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  aten_rms_output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(aten_rms_state), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  arena_pointwise_self_buffer = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      UDONATEN_ARENA_POINTWISE_WORDS * sizeof(float), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  arena_pointwise_other_buffer = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      UDONATEN_ARENA_POINTWISE_WORDS * sizeof(float), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  arena_pointwise_output_buffer = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      UDONATEN_ARENA_POINTWISE_WORDS * sizeof(float), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  arena_rms_partial_buffer = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      UDONATEN_ARENA_POINTWISE_WORDS * sizeof(float), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  program = clCreateProgramWithSource(context, 1, &source, 0, &status);
  if (status != CL_SUCCESS ||
      clBuildProgram(program, 0, 0, "-cl-std=CL1.2", 0, 0) != CL_SUCCESS)
    return execution_failure();
  convolution_kernel =
      clCreateKernel(program, "udondnn_convolution_forward", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(convolution_kernel, 0, sizeof(input_buffer),
                     &input_buffer) != CL_SUCCESS ||
      clSetKernelArg(convolution_kernel, 1, sizeof(output_buffer),
                     &output_buffer) != CL_SUCCESS)
    return execution_failure();
  activation_kernel =
      clCreateKernel(program, "udondnn_activation_forward", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(activation_kernel, 0, sizeof(input_buffer),
                     &input_buffer) != CL_SUCCESS ||
      clSetKernelArg(activation_kernel, 1, sizeof(output_buffer),
                     &output_buffer) != CL_SUCCESS)
    return execution_failure();
  pooling_kernel = clCreateKernel(program, "udondnn_pooling_forward", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(pooling_kernel, 0, sizeof(input_buffer), &input_buffer) !=
          CL_SUCCESS ||
      clSetKernelArg(pooling_kernel, 1, sizeof(output_buffer),
                     &output_buffer) != CL_SUCCESS)
    return execution_failure();
  softmax_kernel = clCreateKernel(program, "udondnn_softmax_forward", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(softmax_kernel, 0, sizeof(input_buffer), &input_buffer) !=
          CL_SUCCESS ||
      clSetKernelArg(softmax_kernel, 1, sizeof(output_buffer),
                     &output_buffer) != CL_SUCCESS)
    return execution_failure();
  batchnorm_kernel =
      clCreateKernel(program, "udondnn_batchnorm_inference", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(batchnorm_kernel, 0, sizeof(batchnorm_input_buffer),
                     &batchnorm_input_buffer) != CL_SUCCESS ||
      clSetKernelArg(batchnorm_kernel, 1, sizeof(batchnorm_output_buffer),
                     &batchnorm_output_buffer) != CL_SUCCESS)
    return execution_failure();
  op_tensor_kernel = clCreateKernel(program, "udondnn_op_tensor", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(op_tensor_kernel, 0, sizeof(input_buffer),
                     &input_buffer) != CL_SUCCESS ||
      clSetKernelArg(op_tensor_kernel, 1, sizeof(output_buffer),
                     &output_buffer) != CL_SUCCESS)
    return execution_failure();
  rms_norm_small_kernel =
      clCreateKernel(program, "udonaten_rms_norm_small", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(rms_norm_small_kernel, 0, sizeof(input_buffer),
                     &input_buffer) != CL_SUCCESS ||
      clSetKernelArg(rms_norm_small_kernel, 1, sizeof(output_buffer),
                     &output_buffer) != CL_SUCCESS)
    return execution_failure();
  embedding_kernel = clCreateKernel(program, "udonaten_embedding", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(embedding_kernel, 0, sizeof(input_buffer),
                     &input_buffer) != CL_SUCCESS ||
      clSetKernelArg(embedding_kernel, 1, sizeof(output_buffer),
                     &output_buffer) != CL_SUCCESS)
    return execution_failure();
  weight_int4pack_mm_kernel =
      clCreateKernel(program, "udonaten_weight_int4pack_mm", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(weight_int4pack_mm_kernel, 0, sizeof(int4_input_buffer),
                     &int4_input_buffer) != CL_SUCCESS ||
      clSetKernelArg(weight_int4pack_mm_kernel, 1, sizeof(int4_output_buffer),
                     &int4_output_buffer) != CL_SUCCESS)
    return execution_failure();
  tile_kernel = clCreateKernel(program, "udonaten_tile", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(tile_kernel, 0, sizeof(int4_input_buffer),
                     &int4_input_buffer) != CL_SUCCESS ||
      clSetKernelArg(tile_kernel, 1, sizeof(int4_output_buffer),
                     &int4_output_buffer) != CL_SUCCESS)
    return execution_failure();
  aten_rms_reduce_kernel =
      clCreateKernel(program, "udonaten_rms_norm_reduce", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(aten_rms_reduce_kernel, 0, sizeof(aten_rms_state_buffer),
                     &aten_rms_state_buffer) != CL_SUCCESS ||
      clSetKernelArg(aten_rms_reduce_kernel, 1, sizeof(aten_rms_partial_buffer),
                     &aten_rms_partial_buffer) != CL_SUCCESS)
    return execution_failure();
  aten_rms_finalize_kernel =
      clCreateKernel(program, "udonaten_rms_norm_finalize", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(aten_rms_finalize_kernel, 0, sizeof(aten_rms_state_buffer),
                     &aten_rms_state_buffer) != CL_SUCCESS ||
      clSetKernelArg(aten_rms_finalize_kernel, 1,
                     sizeof(aten_rms_partial_buffer),
                     &aten_rms_partial_buffer) != CL_SUCCESS ||
      clSetKernelArg(aten_rms_finalize_kernel, 2,
                     sizeof(aten_rms_output_buffer),
                     &aten_rms_output_buffer) != CL_SUCCESS)
    return execution_failure();
  arena_pointwise_kernel =
      clCreateKernel(program, "udonaten_arena_pointwise", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(arena_pointwise_kernel, 0,
                     sizeof(arena_pointwise_self_buffer),
                     &arena_pointwise_self_buffer) != CL_SUCCESS ||
      clSetKernelArg(arena_pointwise_kernel, 1,
                     sizeof(arena_pointwise_other_buffer),
                     &arena_pointwise_other_buffer) != CL_SUCCESS ||
      clSetKernelArg(arena_pointwise_kernel, 2,
                     sizeof(arena_pointwise_output_buffer),
                     &arena_pointwise_output_buffer) != CL_SUCCESS)
    return execution_failure();
  arena_tile_kernel = clCreateKernel(program, "udonaten_arena_tile", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(arena_tile_kernel, 0, sizeof(arena_pointwise_self_buffer),
                     &arena_pointwise_self_buffer) != CL_SUCCESS ||
      clSetKernelArg(arena_tile_kernel, 1,
                     sizeof(arena_pointwise_output_buffer),
                     &arena_pointwise_output_buffer) != CL_SUCCESS)
    return execution_failure();
  arena_rms_reduce_kernel =
      clCreateKernel(program, "udonaten_arena_rms_norm_reduce", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(arena_rms_reduce_kernel, 0,
                     sizeof(arena_pointwise_self_buffer),
                     &arena_pointwise_self_buffer) != CL_SUCCESS ||
      clSetKernelArg(arena_rms_reduce_kernel, 1,
                     sizeof(arena_rms_partial_buffer),
                     &arena_rms_partial_buffer) != CL_SUCCESS)
    return execution_failure();
  arena_rms_finalize_kernel =
      clCreateKernel(program, "udonaten_arena_rms_norm_finalize", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(arena_rms_finalize_kernel, 0,
                     sizeof(arena_pointwise_self_buffer),
                     &arena_pointwise_self_buffer) != CL_SUCCESS ||
      clSetKernelArg(arena_rms_finalize_kernel, 1,
                     sizeof(arena_rms_partial_buffer),
                     &arena_rms_partial_buffer) != CL_SUCCESS ||
      clSetKernelArg(arena_rms_finalize_kernel, 2,
                     sizeof(arena_pointwise_other_buffer),
                     &arena_pointwise_other_buffer) != CL_SUCCESS ||
      clSetKernelArg(arena_rms_finalize_kernel, 3,
                     sizeof(arena_pointwise_output_buffer),
                     &arena_pointwise_output_buffer) != CL_SUCCESS)
    return execution_failure();
  *handle = UDONDNN_HANDLE;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t udondnnDestroy(udondnnHandle_t handle) {
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  clear_objects();
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnCreateTensorDescriptor(udondnnTensorDescriptor_t *descriptor) {
  int slot;
  if (!descriptor)
    return UDONDNN_STATUS_BAD_PARAM;
  for (slot = 0; slot < UDONDNN_DESCRIPTOR_SLOTS; ++slot)
    if (!tensors[slot].used) {
      tensors[slot].generation = next_generation(tensors[slot].generation);
      tensors[slot].used = 1;
      tensors[slot].configured = 0;
      *descriptor = make_descriptor(tensors[slot].generation, (unsigned)slot);
      return UDONDNN_STATUS_SUCCESS;
    }
  return UDONDNN_STATUS_ALLOC_FAILED;
}

udondnnStatus_t udondnnSetTensor4dDescriptor(
    udondnnTensorDescriptor_t descriptor, udondnnTensorFormat_t format,
    udondnnDataType_t data_type, int n, int c, int h, int w) {
  TensorDescriptor *tensor = get_tensor(descriptor);
  if (!tensor || format != UDONDNN_TENSOR_NCHW ||
      data_type != UDONDNN_DATA_FLOAT || n <= 0 || c <= 0 || h <= 0 || w <= 0 ||
      (size_t)n * (size_t)c * (size_t)h * (size_t)w >
          UDONDNN_DESCRIPTOR_MAX_WORDS)
    return UDONDNN_STATUS_BAD_PARAM;
  tensor->n = n;
  tensor->c = c;
  tensor->h = h;
  tensor->w = w;
  tensor->configured = 1;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnGetTensor4dDescriptor(udondnnTensorDescriptor_t descriptor,
                             udondnnDataType_t *data_type, int *n, int *c,
                             int *h, int *w, int *n_stride, int *c_stride,
                             int *h_stride, int *w_stride) {
  TensorDescriptor *tensor = get_tensor(descriptor);
  if (!tensor || !tensor->configured || !data_type || !n || !c || !h || !w ||
      !n_stride || !c_stride || !h_stride || !w_stride)
    return UDONDNN_STATUS_BAD_PARAM;
  *data_type = UDONDNN_DATA_FLOAT;
  *n = tensor->n;
  *c = tensor->c;
  *h = tensor->h;
  *w = tensor->w;
  *w_stride = 1;
  *h_stride = tensor->w;
  *c_stride = tensor->h * *h_stride;
  *n_stride = tensor->c * *c_stride;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnDestroyTensorDescriptor(udondnnTensorDescriptor_t descriptor) {
  TensorDescriptor *tensor = get_tensor(descriptor);
  if (!tensor)
    return UDONDNN_STATUS_BAD_PARAM;
  tensor->used = 0;
  tensor->configured = 0;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnCreateFilterDescriptor(udondnnFilterDescriptor_t *descriptor) {
  int slot;
  if (!descriptor)
    return UDONDNN_STATUS_BAD_PARAM;
  for (slot = 0; slot < UDONDNN_DESCRIPTOR_SLOTS; ++slot)
    if (!filters[slot].used) {
      filters[slot].generation = next_generation(filters[slot].generation);
      filters[slot].used = 1;
      filters[slot].configured = 0;
      *descriptor = make_descriptor(filters[slot].generation, (unsigned)slot);
      return UDONDNN_STATUS_SUCCESS;
    }
  return UDONDNN_STATUS_ALLOC_FAILED;
}

udondnnStatus_t udondnnSetFilter4dDescriptor(
    udondnnFilterDescriptor_t descriptor, udondnnDataType_t data_type,
    udondnnTensorFormat_t format, int k, int c, int h, int w) {
  FilterDescriptor *filter = get_filter(descriptor);
  if (!filter || data_type != UDONDNN_DATA_FLOAT ||
      format != UDONDNN_TENSOR_NCHW || k <= 0 || c <= 0 || h <= 0 || w <= 0 ||
      k > 15 || c > 8 || h > 5 || w > 5 ||
      k * c * h * w > UDONDNN_CONV_MAX_FILTER_WORDS)
    return UDONDNN_STATUS_BAD_PARAM;
  filter->k = k;
  filter->c = c;
  filter->h = h;
  filter->w = w;
  filter->configured = 1;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnDestroyFilterDescriptor(udondnnFilterDescriptor_t descriptor) {
  FilterDescriptor *filter = get_filter(descriptor);
  if (!filter)
    return UDONDNN_STATUS_BAD_PARAM;
  filter->used = 0;
  filter->configured = 0;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnCreateConvolutionDescriptor(udondnnConvolutionDescriptor_t *descriptor) {
  int slot;
  if (!descriptor)
    return UDONDNN_STATUS_BAD_PARAM;
  for (slot = 0; slot < UDONDNN_DESCRIPTOR_SLOTS; ++slot)
    if (!convolutions[slot].used) {
      convolutions[slot].generation =
          next_generation(convolutions[slot].generation);
      convolutions[slot].used = 1;
      convolutions[slot].configured = 0;
      *descriptor =
          make_descriptor(convolutions[slot].generation, (unsigned)slot);
      return UDONDNN_STATUS_SUCCESS;
    }
  return UDONDNN_STATUS_ALLOC_FAILED;
}

udondnnStatus_t udondnnSetConvolution2dDescriptor(
    udondnnConvolutionDescriptor_t descriptor, int pad_h, int pad_w,
    int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w,
    udondnnConvolutionMode_t mode, udondnnDataType_t compute_type) {
  ConvolutionDescriptor *convolution = get_convolution(descriptor);
  if (!convolution || pad_h < 0 || pad_w < 0 || pad_h > 15 || pad_w > 15 ||
      vertical_stride <= 0 || vertical_stride > 15 || horizontal_stride <= 0 ||
      horizontal_stride > 15 || dilation_h <= 0 || dilation_h > 15 ||
      dilation_w <= 0 || dilation_w > 15 ||
      (mode != UDONDNN_CONVOLUTION && mode != UDONDNN_CROSS_CORRELATION) ||
      compute_type != UDONDNN_DATA_FLOAT)
    return UDONDNN_STATUS_BAD_PARAM;
  convolution->pad_h = pad_h;
  convolution->pad_w = pad_w;
  convolution->stride_h = vertical_stride;
  convolution->stride_w = horizontal_stride;
  convolution->dilation_h = dilation_h;
  convolution->dilation_w = dilation_w;
  convolution->mode = mode;
  convolution->configured = 1;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnDestroyConvolutionDescriptor(udondnnConvolutionDescriptor_t descriptor) {
  ConvolutionDescriptor *convolution = get_convolution(descriptor);
  if (!convolution)
    return UDONDNN_STATUS_BAD_PARAM;
  convolution->used = 0;
  convolution->configured = 0;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnCreateActivationDescriptor(udondnnActivationDescriptor_t *descriptor) {
  int slot;
  if (!descriptor)
    return UDONDNN_STATUS_BAD_PARAM;
  for (slot = 0; slot < UDONDNN_DESCRIPTOR_SLOTS; ++slot)
    if (!activations[slot].used) {
      activations[slot].generation =
          next_generation(activations[slot].generation);
      activations[slot].used = 1;
      activations[slot].configured = 0;
      *descriptor =
          make_descriptor(activations[slot].generation, (unsigned)slot);
      return UDONDNN_STATUS_SUCCESS;
    }
  return UDONDNN_STATUS_ALLOC_FAILED;
}

udondnnStatus_t udondnnSetActivationDescriptor(
    udondnnActivationDescriptor_t descriptor, udondnnActivationMode_t mode,
    udondnnNanPropagation_t nan_propagation, float coefficient) {
  ActivationDescriptor *activation = get_activation(descriptor);
  if (!activation || mode < UDONDNN_ACTIVATION_SIGMOID ||
      mode > UDONDNN_ACTIVATION_IDENTITY ||
      (nan_propagation != UDONDNN_NOT_PROPAGATE_NAN &&
       nan_propagation != UDONDNN_PROPAGATE_NAN) ||
      coefficient != coefficient || coefficient > FLT_MAX ||
      coefficient < -FLT_MAX ||
      ((mode == UDONDNN_ACTIVATION_CLIPPED_RELU ||
        mode == UDONDNN_ACTIVATION_ELU) &&
       coefficient < 0.0f))
    return UDONDNN_STATUS_BAD_PARAM;
  activation->mode = mode;
  activation->nan_propagation = nan_propagation;
  activation->coefficient = coefficient;
  activation->configured = 1;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnDestroyActivationDescriptor(udondnnActivationDescriptor_t descriptor) {
  ActivationDescriptor *activation = get_activation(descriptor);
  if (!activation)
    return UDONDNN_STATUS_BAD_PARAM;
  activation->used = 0;
  activation->configured = 0;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnCreatePoolingDescriptor(udondnnPoolingDescriptor_t *descriptor) {
  int slot;
  if (!descriptor)
    return UDONDNN_STATUS_BAD_PARAM;
  for (slot = 0; slot < UDONDNN_DESCRIPTOR_SLOTS; ++slot)
    if (!poolings[slot].used) {
      poolings[slot].generation = next_generation(poolings[slot].generation);
      poolings[slot].used = 1;
      poolings[slot].configured = 0;
      *descriptor = make_descriptor(poolings[slot].generation, (unsigned)slot);
      return UDONDNN_STATUS_SUCCESS;
    }
  return UDONDNN_STATUS_ALLOC_FAILED;
}

udondnnStatus_t udondnnSetPooling2dDescriptor(
    udondnnPoolingDescriptor_t descriptor, udondnnPoolingMode_t mode,
    udondnnNanPropagation_t nan_propagation, int window_height,
    int window_width, int vertical_padding, int horizontal_padding,
    int vertical_stride, int horizontal_stride) {
  PoolingDescriptor *pooling = get_pooling(descriptor);
  if (!pooling || mode < UDONDNN_POOLING_MAX ||
      mode > UDONDNN_POOLING_MAX_DETERMINISTIC ||
      (nan_propagation != UDONDNN_NOT_PROPAGATE_NAN &&
       nan_propagation != UDONDNN_PROPAGATE_NAN) ||
      window_height <= 0 || window_height > 5 || window_width <= 0 ||
      window_width > 5 || vertical_padding < 0 || horizontal_padding < 0 ||
      vertical_padding * 2 >= window_height ||
      horizontal_padding * 2 >= window_width || vertical_stride <= 0 ||
      vertical_stride > 15 || horizontal_stride <= 0 || horizontal_stride > 15)
    return UDONDNN_STATUS_BAD_PARAM;
  pooling->mode = mode;
  pooling->nan_propagation = nan_propagation;
  pooling->window_h = window_height;
  pooling->window_w = window_width;
  pooling->pad_h = vertical_padding;
  pooling->pad_w = horizontal_padding;
  pooling->stride_h = vertical_stride;
  pooling->stride_w = horizontal_stride;
  pooling->configured = 1;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnDestroyPoolingDescriptor(udondnnPoolingDescriptor_t descriptor) {
  PoolingDescriptor *pooling = get_pooling(descriptor);
  if (!pooling)
    return UDONDNN_STATUS_BAD_PARAM;
  pooling->used = 0;
  pooling->configured = 0;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnCreateOpTensorDescriptor(udondnnOpTensorDescriptor_t *descriptor) {
  int slot;
  if (!descriptor)
    return UDONDNN_STATUS_BAD_PARAM;
  for (slot = 0; slot < UDONDNN_DESCRIPTOR_SLOTS; ++slot)
    if (!op_tensors[slot].used) {
      op_tensors[slot].generation =
          next_generation(op_tensors[slot].generation);
      op_tensors[slot].used = 1;
      op_tensors[slot].configured = 0;
      *descriptor =
          make_descriptor(op_tensors[slot].generation, (unsigned)slot);
      return UDONDNN_STATUS_SUCCESS;
    }
  return UDONDNN_STATUS_ALLOC_FAILED;
}

udondnnStatus_t udondnnSetOpTensorDescriptor(
    udondnnOpTensorDescriptor_t descriptor, udondnnOpTensorOp_t operation,
    udondnnDataType_t compute_type, udondnnNanPropagation_t nan_propagation) {
  OpTensorDescriptor *op_tensor = get_op_tensor(descriptor);
  if (!op_tensor || operation < UDONDNN_OP_TENSOR_ADD ||
      operation > UDONDNN_OP_TENSOR_MAX || compute_type != UDONDNN_DATA_FLOAT ||
      (nan_propagation != UDONDNN_NOT_PROPAGATE_NAN &&
       nan_propagation != UDONDNN_PROPAGATE_NAN))
    return UDONDNN_STATUS_BAD_PARAM;
  op_tensor->operation = operation;
  op_tensor->nan_propagation = nan_propagation;
  op_tensor->configured = 1;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnDestroyOpTensorDescriptor(udondnnOpTensorDescriptor_t descriptor) {
  OpTensorDescriptor *op_tensor = get_op_tensor(descriptor);
  if (!op_tensor)
    return UDONDNN_STATUS_BAD_PARAM;
  op_tensor->used = 0;
  op_tensor->configured = 0;
  return UDONDNN_STATUS_SUCCESS;
}

static udondnnStatus_t
output_dimensions(const ConvolutionDescriptor *convolution,
                  const TensorDescriptor *input, const FilterDescriptor *filter,
                  int *n, int *c, int *h, int *w) {
  int effective_h, effective_w, numerator_h, numerator_w;
  if (!convolution || !input || !filter || !convolution->configured ||
      !input->configured || !filter->configured || filter->c != input->c)
    return UDONDNN_STATUS_BAD_PARAM;
  effective_h = convolution->dilation_h * (filter->h - 1) + 1;
  effective_w = convolution->dilation_w * (filter->w - 1) + 1;
  numerator_h = input->h + 2 * convolution->pad_h - effective_h;
  numerator_w = input->w + 2 * convolution->pad_w - effective_w;
  if (numerator_h < 0 || numerator_w < 0)
    return UDONDNN_STATUS_BAD_PARAM;
  *n = input->n;
  *c = filter->k;
  *h = 1 + numerator_h / convolution->stride_h;
  *w = 1 + numerator_w / convolution->stride_w;
  if (*n > 15 || *c > 15 || *h > 15 || *w > 15 ||
      *n * *c * *h * *w > UDONDNN_CONV_MAX_OUTPUT_WORDS)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t udondnnGetConvolution2dForwardOutputDim(
    udondnnConvolutionDescriptor_t convolution_descriptor,
    udondnnTensorDescriptor_t input_descriptor,
    udondnnFilterDescriptor_t filter_descriptor, int *n, int *c, int *h,
    int *w) {
  if (!n || !c || !h || !w)
    return UDONDNN_STATUS_BAD_PARAM;
  return output_dimensions(get_convolution(convolution_descriptor),
                           get_tensor(input_descriptor),
                           get_filter(filter_descriptor), n, c, h, w);
}

static udondnnStatus_t
validate_operation(udondnnTensorDescriptor_t input_descriptor,
                   udondnnFilterDescriptor_t filter_descriptor,
                   udondnnConvolutionDescriptor_t convolution_descriptor,
                   udondnnTensorDescriptor_t output_descriptor,
                   udondnnConvolutionFwdAlgo_t algorithm, int *n, int *c,
                   int *h, int *w) {
  TensorDescriptor *output = get_tensor(output_descriptor);
  udondnnStatus_t status;
  if (algorithm != UDONDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM || !output ||
      !output->configured)
    return UDONDNN_STATUS_BAD_PARAM;
  status = output_dimensions(get_convolution(convolution_descriptor),
                             get_tensor(input_descriptor),
                             get_filter(filter_descriptor), n, c, h, w);
  if (status != UDONDNN_STATUS_SUCCESS)
    return status;
  if (output->n != *n || output->c != *c || output->h != *h || output->w != *w)
    return UDONDNN_STATUS_BAD_PARAM;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t udondnnGetConvolutionForwardWorkspaceSize(
    udondnnHandle_t handle, udondnnTensorDescriptor_t input,
    udondnnFilterDescriptor_t filter,
    udondnnConvolutionDescriptor_t convolution,
    udondnnTensorDescriptor_t output, udondnnConvolutionFwdAlgo_t algorithm,
    size_t *size_in_bytes) {
  int n, c, h, w;
  udondnnStatus_t status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  if (!size_in_bytes)
    return UDONDNN_STATUS_BAD_PARAM;
  status = validate_operation(input, filter, convolution, output, algorithm, &n,
                              &c, &h, &w);
  if (status != UDONDNN_STATUS_SUCCESS)
    return status;
  *size_in_bytes = 0;
  return UDONDNN_STATUS_SUCCESS;
}

static int pack_shape(int a, int b, int c, int d) {
  return a | (b << 4) | (c << 8) | (d << 12);
}

static int packable_shape(const TensorDescriptor *tensor) {
  return tensor && tensor->configured && tensor->n <= 15 && tensor->c <= 15 &&
         tensor->h <= 15 && tensor->w <= 15;
}

udondnnStatus_t udondnnConvolutionForward(
    udondnnHandle_t handle, const float *alpha,
    udondnnTensorDescriptor_t input_descriptor, const float *input,
    udondnnFilterDescriptor_t filter_descriptor, const float *filter,
    udondnnConvolutionDescriptor_t convolution_descriptor,
    udondnnConvolutionFwdAlgo_t algorithm, void *workspace,
    size_t workspace_size_in_bytes, const float *beta,
    udondnnTensorDescriptor_t output_descriptor, float *output) {
  TensorDescriptor *input_info;
  FilterDescriptor *filter_info;
  ConvolutionDescriptor *convolution_info;
  int n, c, h, w, input_words, filter_words, output_words, index;
  int input_shape, filter_shape, convolution_shape, convolution_mode;
  int output_shape;
  size_t global = UDONDNN_WORDS, local = 1;
  cl_int cl_status;
  udondnnStatus_t status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  if (!alpha || !beta || !input || !filter || !output || workspace ||
      workspace_size_in_bytes)
    return UDONDNN_STATUS_BAD_PARAM;
  status = validate_operation(input_descriptor, filter_descriptor,
                              convolution_descriptor, output_descriptor,
                              algorithm, &n, &c, &h, &w);
  if (status != UDONDNN_STATUS_SUCCESS)
    return status;
  input_info = get_tensor(input_descriptor);
  filter_info = get_filter(filter_descriptor);
  convolution_info = get_convolution(convolution_descriptor);
  input_words = input_info->n * input_info->c * input_info->h * input_info->w;
  filter_words =
      filter_info->k * filter_info->c * filter_info->h * filter_info->w;
  output_words = n * c * h * w;
  if (!packable_shape(input_info) || input_words > UDONDNN_REGION_WORDS ||
      filter_words > UDONDNN_REGION_WORDS ||
      output_words > UDONDNN_REGION_WORDS)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  for (index = 0; index < UDONDNN_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < input_words; ++index)
    packed[UDONDNN_INPUT_OFFSET + index] = input[index];
  for (index = 0; index < filter_words; ++index)
    packed[UDONDNN_FILTER_OFFSET + index] = filter[index];
  for (index = 0; index < output_words; ++index)
    packed[UDONDNN_PREVIOUS_OUTPUT_OFFSET + index] = output[index];

  input_shape =
      pack_shape(input_info->n, input_info->c, input_info->h, input_info->w);
  filter_shape = pack_shape(filter_info->k, filter_info->c, filter_info->h,
                            filter_info->w);
  convolution_shape = convolution_info->pad_h | (convolution_info->pad_w << 4) |
                      (convolution_info->stride_h << 8) |
                      (convolution_info->stride_w << 12) |
                      (convolution_info->dilation_h << 16) |
                      (convolution_info->dilation_w << 20);
  convolution_mode = (int)convolution_info->mode;
  output_shape = pack_shape(n, c, h, w);

#define SET_ARGUMENT(slot, value)                                              \
  do {                                                                         \
    cl_status =                                                                \
        clSetKernelArg(convolution_kernel, slot, sizeof(value), &(value));     \
    if (cl_status != CL_SUCCESS)                                               \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
  SET_ARGUMENT(2, input_shape);
  SET_ARGUMENT(3, filter_shape);
  SET_ARGUMENT(4, convolution_shape);
  SET_ARGUMENT(5, convolution_mode);
  SET_ARGUMENT(6, output_shape);
  SET_ARGUMENT(7, *alpha);
  SET_ARGUMENT(8, *beta);
#undef SET_ARGUMENT

  if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(packed),
                           packed, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, convolution_kernel, 1, 0, &global, &local,
                             0, 0, 0) != CL_SUCCESS ||
      clEnqueueReadBuffer(
          queue, output_buffer, CL_TRUE, UDONDNN_RESULT_OFFSET * sizeof(float),
          output_words * sizeof(float), result, 0, 0, 0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONDNN_STATUS_EXECUTION_FAILED;
  for (index = 0; index < output_words; ++index)
    output[index] = result[index];
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t udondnnActivationForward(
    udondnnHandle_t handle, udondnnActivationDescriptor_t activation_descriptor,
    const float *alpha, udondnnTensorDescriptor_t input_descriptor,
    const float *input, const float *beta,
    udondnnTensorDescriptor_t output_descriptor, float *output) {
  ActivationDescriptor *activation;
  TensorDescriptor *input_info, *output_info;
  int words, index;
  int mode, nan_propagation;
  float coefficient;
  size_t global = UDONDNN_ACTIVATION_WORDS, local = 1;
  cl_int cl_status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  activation = get_activation(activation_descriptor);
  input_info = get_tensor(input_descriptor);
  output_info = get_tensor(output_descriptor);
  if (!activation || !activation->configured || !input_info ||
      !input_info->configured || !output_info || !output_info->configured ||
      !alpha || !input || !beta || !output || input_info->n != output_info->n ||
      input_info->c != output_info->c || input_info->h != output_info->h ||
      input_info->w != output_info->w)
    return UDONDNN_STATUS_BAD_PARAM;
  words = input_info->n * input_info->c * input_info->h * input_info->w;
  if (words > UDONDNN_REGION_WORDS)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  for (index = 0; index < UDONDNN_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < words; ++index) {
    packed[index] = input[index];
    packed[UDONDNN_ACTIVATION_PREVIOUS_OUTPUT_OFFSET + index] = output[index];
  }
  mode = (int)activation->mode;
  nan_propagation = (int)activation->nan_propagation;
  coefficient = activation->coefficient;

#define SET_ACTIVATION_ARGUMENT(slot, value)                                   \
  do {                                                                         \
    cl_status =                                                                \
        clSetKernelArg(activation_kernel, slot, sizeof(value), &(value));      \
    if (cl_status != CL_SUCCESS)                                               \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
  SET_ACTIVATION_ARGUMENT(2, words);
  SET_ACTIVATION_ARGUMENT(3, mode);
  SET_ACTIVATION_ARGUMENT(4, nan_propagation);
  SET_ACTIVATION_ARGUMENT(5, coefficient);
  SET_ACTIVATION_ARGUMENT(6, *alpha);
  SET_ACTIVATION_ARGUMENT(7, *beta);
#undef SET_ACTIVATION_ARGUMENT

  if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(packed),
                           packed, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, activation_kernel, 1, 0, &global, &local, 0,
                             0, 0) != CL_SUCCESS ||
      clEnqueueReadBuffer(queue, output_buffer, CL_TRUE,
                          UDONDNN_ACTIVATION_RESULT_OFFSET * sizeof(float),
                          words * sizeof(float), result, 0, 0,
                          0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONDNN_STATUS_EXECUTION_FAILED;
  for (index = 0; index < words; ++index)
    output[index] = result[index];
  return UDONDNN_STATUS_SUCCESS;
}

static udondnnStatus_t
pooling_output_dimensions(const PoolingDescriptor *pooling,
                          const TensorDescriptor *input, int *n, int *c, int *h,
                          int *w) {
  int numerator_h, numerator_w;
  if (!pooling || !pooling->configured || !input || !input->configured)
    return UDONDNN_STATUS_BAD_PARAM;
  numerator_h = input->h + 2 * pooling->pad_h - pooling->window_h;
  numerator_w = input->w + 2 * pooling->pad_w - pooling->window_w;
  if (numerator_h < 0 || numerator_w < 0)
    return UDONDNN_STATUS_BAD_PARAM;
  *n = input->n;
  *c = input->c;
  *h = 1 + numerator_h / pooling->stride_h;
  *w = 1 + numerator_w / pooling->stride_w;
  if (*n > 15 || *c > 15 || *h > 15 || *w > 15 ||
      *n * *c * *h * *w > UDONDNN_CONV_MAX_OUTPUT_WORDS)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t udondnnGetPooling2dForwardOutputDim(
    udondnnPoolingDescriptor_t pooling_descriptor,
    udondnnTensorDescriptor_t input_descriptor, int *n, int *c, int *h,
    int *w) {
  if (!n || !c || !h || !w)
    return UDONDNN_STATUS_BAD_PARAM;
  return pooling_output_dimensions(get_pooling(pooling_descriptor),
                                   get_tensor(input_descriptor), n, c, h, w);
}

udondnnStatus_t udondnnPoolingForward(
    udondnnHandle_t handle, udondnnPoolingDescriptor_t pooling_descriptor,
    const float *alpha, udondnnTensorDescriptor_t input_descriptor,
    const float *input, const float *beta,
    udondnnTensorDescriptor_t output_descriptor, float *output) {
  PoolingDescriptor *pooling;
  TensorDescriptor *input_info, *output_info;
  int n, c, h, w, input_words, output_words, index;
  int input_shape, pooling_shape, pooling_mode, nan_propagation, output_shape;
  size_t global = UDONDNN_POOLING_WORDS, local = 1;
  cl_int cl_status;
  udondnnStatus_t status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  pooling = get_pooling(pooling_descriptor);
  input_info = get_tensor(input_descriptor);
  output_info = get_tensor(output_descriptor);
  if (!alpha || !input || !beta || !output || !output_info ||
      !output_info->configured)
    return UDONDNN_STATUS_BAD_PARAM;
  status = pooling_output_dimensions(pooling, input_info, &n, &c, &h, &w);
  if (status != UDONDNN_STATUS_SUCCESS)
    return status;
  if (output_info->n != n || output_info->c != c || output_info->h != h ||
      output_info->w != w)
    return UDONDNN_STATUS_BAD_PARAM;
  input_words = input_info->n * input_info->c * input_info->h * input_info->w;
  output_words = n * c * h * w;
  if (!packable_shape(input_info) || input_words > UDONDNN_REGION_WORDS ||
      output_words > UDONDNN_REGION_WORDS)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  for (index = 0; index < UDONDNN_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < input_words; ++index)
    packed[index] = input[index];
  for (index = 0; index < output_words; ++index)
    packed[UDONDNN_POOLING_PREVIOUS_OUTPUT_OFFSET + index] = output[index];

  input_shape =
      pack_shape(input_info->n, input_info->c, input_info->h, input_info->w);
  pooling_shape = pooling->window_h | (pooling->window_w << 4) |
                  (pooling->pad_h << 8) | (pooling->pad_w << 12) |
                  (pooling->stride_h << 16) | (pooling->stride_w << 20);
  pooling_mode = (int)pooling->mode;
  nan_propagation = (int)pooling->nan_propagation;
  output_shape = pack_shape(n, c, h, w);

#define SET_POOLING_ARGUMENT(slot, value)                                      \
  do {                                                                         \
    cl_status = clSetKernelArg(pooling_kernel, slot, sizeof(value), &(value)); \
    if (cl_status != CL_SUCCESS)                                               \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
  SET_POOLING_ARGUMENT(2, input_shape);
  SET_POOLING_ARGUMENT(3, pooling_shape);
  SET_POOLING_ARGUMENT(4, pooling_mode);
  SET_POOLING_ARGUMENT(5, nan_propagation);
  SET_POOLING_ARGUMENT(6, output_shape);
  SET_POOLING_ARGUMENT(7, *alpha);
  SET_POOLING_ARGUMENT(8, *beta);
#undef SET_POOLING_ARGUMENT

  if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(packed),
                           packed, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, pooling_kernel, 1, 0, &global, &local, 0, 0,
                             0) != CL_SUCCESS ||
      clEnqueueReadBuffer(queue, output_buffer, CL_TRUE,
                          UDONDNN_POOLING_RESULT_OFFSET * sizeof(float),
                          output_words * sizeof(float), result, 0, 0,
                          0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONDNN_STATUS_EXECUTION_FAILED;
  for (index = 0; index < output_words; ++index)
    output[index] = result[index];
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t udondnnSoftmaxForward(
    udondnnHandle_t handle, udondnnSoftmaxAlgorithm_t algorithm,
    udondnnSoftmaxMode_t mode, const float *alpha,
    udondnnTensorDescriptor_t input_descriptor, const float *input,
    const float *beta, udondnnTensorDescriptor_t output_descriptor,
    float *output) {
  TensorDescriptor *input_info, *output_info;
  int words, index, input_shape, algorithm_value, mode_value;
  size_t global = UDONDNN_SOFTMAX_WORDS, local = 1;
  cl_int cl_status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  input_info = get_tensor(input_descriptor);
  output_info = get_tensor(output_descriptor);
  if (algorithm < UDONDNN_SOFTMAX_FAST || algorithm > UDONDNN_SOFTMAX_LOG ||
      mode < UDONDNN_SOFTMAX_MODE_INSTANCE ||
      mode > UDONDNN_SOFTMAX_MODE_CHANNEL || !input_info ||
      !input_info->configured || !output_info || !output_info->configured ||
      !alpha || !input || !beta || !output || input_info->n != output_info->n ||
      input_info->c != output_info->c || input_info->h != output_info->h ||
      input_info->w != output_info->w)
    return UDONDNN_STATUS_BAD_PARAM;
  words = input_info->n * input_info->c * input_info->h * input_info->w;
  if (!packable_shape(input_info) || words > UDONDNN_REGION_WORDS)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  for (index = 0; index < UDONDNN_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < words; ++index) {
    packed[index] = input[index];
    packed[UDONDNN_SOFTMAX_PREVIOUS_OUTPUT_OFFSET + index] = output[index];
  }
  input_shape =
      pack_shape(input_info->n, input_info->c, input_info->h, input_info->w);
  algorithm_value = (int)algorithm;
  mode_value = (int)mode;

#define SET_SOFTMAX_ARGUMENT(slot, value)                                      \
  do {                                                                         \
    cl_status = clSetKernelArg(softmax_kernel, slot, sizeof(value), &(value)); \
    if (cl_status != CL_SUCCESS)                                               \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
  SET_SOFTMAX_ARGUMENT(2, input_shape);
  SET_SOFTMAX_ARGUMENT(3, algorithm_value);
  SET_SOFTMAX_ARGUMENT(4, mode_value);
  SET_SOFTMAX_ARGUMENT(5, *alpha);
  SET_SOFTMAX_ARGUMENT(6, *beta);
#undef SET_SOFTMAX_ARGUMENT

  if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(packed),
                           packed, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, softmax_kernel, 1, 0, &global, &local, 0, 0,
                             0) != CL_SUCCESS ||
      clEnqueueReadBuffer(queue, output_buffer, CL_TRUE,
                          UDONDNN_SOFTMAX_RESULT_OFFSET * sizeof(float),
                          words * sizeof(float), result, 0, 0,
                          0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONDNN_STATUS_EXECUTION_FAILED;
  for (index = 0; index < words; ++index)
    output[index] = result[index];
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t
udondnnDeriveBNTensorDescriptor(udondnnTensorDescriptor_t derived_descriptor,
                                udondnnTensorDescriptor_t input_descriptor,
                                udondnnBatchNormMode_t mode) {
  TensorDescriptor *derived = get_tensor(derived_descriptor);
  TensorDescriptor *input = get_tensor(input_descriptor);
  if (!derived || !input || !input->configured ||
      mode < UDONDNN_BATCHNORM_PER_ACTIVATION ||
      mode > UDONDNN_BATCHNORM_SPATIAL_PERSISTENT)
    return UDONDNN_STATUS_BAD_PARAM;
  derived->n = 1;
  derived->c = input->c;
  derived->h = mode == UDONDNN_BATCHNORM_PER_ACTIVATION ? input->h : 1;
  derived->w = mode == UDONDNN_BATCHNORM_PER_ACTIVATION ? input->w : 1;
  derived->configured = 1;
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t udondnnBatchNormalizationForwardInference(
    udondnnHandle_t handle, udondnnBatchNormMode_t mode, const float *alpha,
    const float *beta, udondnnTensorDescriptor_t input_descriptor,
    const float *input, udondnnTensorDescriptor_t output_descriptor,
    float *output, udondnnTensorDescriptor_t parameter_descriptor,
    const float *scale, const float *bias, const float *estimated_mean,
    const float *estimated_variance, float epsilon) {
  TensorDescriptor *input_info, *output_info, *parameter_info;
  int words, parameter_words, index, input_shape, mode_value;
  size_t global = UDONDNN_BATCHNORM_WORDS, local = 1;
  cl_int cl_status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  input_info = get_tensor(input_descriptor);
  output_info = get_tensor(output_descriptor);
  parameter_info = get_tensor(parameter_descriptor);
  if (mode < UDONDNN_BATCHNORM_PER_ACTIVATION ||
      mode > UDONDNN_BATCHNORM_SPATIAL_PERSISTENT || !input_info ||
      !input_info->configured || !output_info || !output_info->configured ||
      !parameter_info || !parameter_info->configured || !alpha || !beta ||
      !input || !output || !scale || !bias || !estimated_mean ||
      !estimated_variance || epsilon < UDONDNN_BN_MIN_EPSILON ||
      epsilon != epsilon || epsilon > FLT_MAX ||
      input_info->n != output_info->n || input_info->c != output_info->c ||
      input_info->h != output_info->h || input_info->w != output_info->w ||
      parameter_info->n != 1 || parameter_info->c != input_info->c ||
      parameter_info->h !=
          (mode == UDONDNN_BATCHNORM_PER_ACTIVATION ? input_info->h : 1) ||
      parameter_info->w !=
          (mode == UDONDNN_BATCHNORM_PER_ACTIVATION ? input_info->w : 1))
    return UDONDNN_STATUS_BAD_PARAM;
  words = input_info->n * input_info->c * input_info->h * input_info->w;
  parameter_words = parameter_info->c * parameter_info->h * parameter_info->w;
  if (!packable_shape(input_info) || words > UDONDNN_REGION_WORDS ||
      parameter_words > UDONDNN_REGION_WORDS)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  for (index = 0; index < parameter_words; ++index)
    if (estimated_variance[index] < 0.0f ||
        estimated_variance[index] != estimated_variance[index] ||
        estimated_variance[index] > FLT_MAX)
      return UDONDNN_STATUS_BAD_PARAM;
  for (index = 0; index < UDONDNN_BATCHNORM_WORDS; ++index)
    batchnorm_packed[index] = 0.0f;
  for (index = 0; index < words; ++index) {
    batchnorm_packed[index] = input[index];
    batchnorm_packed[UDONDNN_BATCHNORM_PREVIOUS_OUTPUT_OFFSET + index] =
        output[index];
  }
  for (index = 0; index < parameter_words; ++index) {
    batchnorm_packed[UDONDNN_BATCHNORM_SCALE_OFFSET + index] = scale[index];
    batchnorm_packed[UDONDNN_BATCHNORM_BIAS_OFFSET + index] = bias[index];
    batchnorm_packed[UDONDNN_BATCHNORM_MEAN_OFFSET + index] =
        estimated_mean[index];
    batchnorm_packed[UDONDNN_BATCHNORM_VARIANCE_OFFSET + index] =
        estimated_variance[index];
  }
  input_shape =
      pack_shape(input_info->n, input_info->c, input_info->h, input_info->w);
  mode_value = (int)mode;

#define SET_BATCHNORM_ARGUMENT(slot, value)                                    \
  do {                                                                         \
    cl_status =                                                                \
        clSetKernelArg(batchnorm_kernel, slot, sizeof(value), &(value));       \
    if (cl_status != CL_SUCCESS)                                               \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
  SET_BATCHNORM_ARGUMENT(2, input_shape);
  SET_BATCHNORM_ARGUMENT(3, mode_value);
  SET_BATCHNORM_ARGUMENT(4, epsilon);
  SET_BATCHNORM_ARGUMENT(5, *alpha);
  SET_BATCHNORM_ARGUMENT(6, *beta);
#undef SET_BATCHNORM_ARGUMENT

  if (clEnqueueWriteBuffer(queue, batchnorm_input_buffer, CL_TRUE, 0,
                           sizeof(batchnorm_packed), batchnorm_packed, 0, 0,
                           0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, batchnorm_kernel, 1, 0, &global, &local, 0,
                             0, 0) != CL_SUCCESS ||
      clEnqueueReadBuffer(queue, batchnorm_output_buffer, CL_TRUE,
                          UDONDNN_BATCHNORM_RESULT_OFFSET * sizeof(float),
                          words * sizeof(float), result, 0, 0,
                          0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONDNN_STATUS_EXECUTION_FAILED;
  for (index = 0; index < words; ++index)
    output[index] = result[index];
  return UDONDNN_STATUS_SUCCESS;
}

static int broadcasts_to(const TensorDescriptor *source,
                         const TensorDescriptor *output) {
  return (source->n == 1 || source->n == output->n) &&
         (source->c == 1 || source->c == output->c) &&
         (source->h == 1 || source->h == output->h) &&
         (source->w == 1 || source->w == output->w);
}

udondnnStatus_t udondnnOpTensor(
    udondnnHandle_t handle, udondnnOpTensorDescriptor_t operation_descriptor,
    const float *alpha_a, udondnnTensorDescriptor_t a_descriptor,
    const float *a, const float *alpha_b,
    udondnnTensorDescriptor_t b_descriptor, const float *b, const float *beta,
    udondnnTensorDescriptor_t c_descriptor, float *c) {
  OpTensorDescriptor *operation_info;
  TensorDescriptor *a_info, *b_info, *c_info;
  int a_words, b_words, c_words, index;
  int a_shape, b_shape, c_shape, operation, nan_propagation;
  size_t global = UDONDNN_WORDS, local = 1;
  cl_int cl_status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  operation_info = get_op_tensor(operation_descriptor);
  a_info = get_tensor(a_descriptor);
  b_info = get_tensor(b_descriptor);
  c_info = get_tensor(c_descriptor);
  if (!operation_info || !operation_info->configured || !a_info ||
      !a_info->configured || !b_info || !b_info->configured || !c_info ||
      !c_info->configured || !alpha_a || !a || !alpha_b || !b || !beta || !c ||
      !broadcasts_to(a_info, c_info) || !broadcasts_to(b_info, c_info))
    return UDONDNN_STATUS_BAD_PARAM;
  a_words = a_info->n * a_info->c * a_info->h * a_info->w;
  b_words = b_info->n * b_info->c * b_info->h * b_info->w;
  c_words = c_info->n * c_info->c * c_info->h * c_info->w;
  if (!packable_shape(a_info) || !packable_shape(b_info) ||
      !packable_shape(c_info) || a_words > UDONDNN_REGION_WORDS ||
      b_words > UDONDNN_REGION_WORDS || c_words > UDONDNN_REGION_WORDS)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  for (index = 0; index < UDONDNN_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < a_words; ++index)
    packed[index] = a[index];
  for (index = 0; index < b_words; ++index)
    packed[UDONDNN_OP_TENSOR_B_OFFSET + index] = b[index];
  for (index = 0; index < c_words; ++index)
    packed[UDONDNN_OP_TENSOR_C_OFFSET + index] = c[index];
  a_shape = pack_shape(a_info->n, a_info->c, a_info->h, a_info->w);
  b_shape = pack_shape(b_info->n, b_info->c, b_info->h, b_info->w);
  c_shape = pack_shape(c_info->n, c_info->c, c_info->h, c_info->w);
  operation = (int)operation_info->operation;
  nan_propagation = (int)operation_info->nan_propagation;

#define SET_OP_TENSOR_ARGUMENT(slot, value)                                    \
  do {                                                                         \
    cl_status =                                                                \
        clSetKernelArg(op_tensor_kernel, slot, sizeof(value), &(value));       \
    if (cl_status != CL_SUCCESS)                                               \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
  SET_OP_TENSOR_ARGUMENT(2, a_shape);
  SET_OP_TENSOR_ARGUMENT(3, b_shape);
  SET_OP_TENSOR_ARGUMENT(4, c_shape);
  SET_OP_TENSOR_ARGUMENT(5, operation);
  SET_OP_TENSOR_ARGUMENT(6, nan_propagation);
  SET_OP_TENSOR_ARGUMENT(7, *alpha_a);
  SET_OP_TENSOR_ARGUMENT(8, *alpha_b);
  SET_OP_TENSOR_ARGUMENT(9, *beta);
#undef SET_OP_TENSOR_ARGUMENT

  if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(packed),
                           packed, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, op_tensor_kernel, 1, 0, &global, &local, 0,
                             0, 0) != CL_SUCCESS ||
      clEnqueueReadBuffer(queue, output_buffer, CL_TRUE,
                          UDONDNN_OP_TENSOR_RESULT_OFFSET * sizeof(float),
                          c_words * sizeof(float), result, 0, 0,
                          0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONDNN_STATUS_EXECUTION_FAILED;
  for (index = 0; index < c_words; ++index)
    c[index] = result[index];
  return UDONDNN_STATUS_SUCCESS;
}

udondnnStatus_t udonatenBackendRmsNorm(udondnnHandle_t handle, int words,
                                       int normalized_words, const float *input,
                                       const float *weight, float epsilon,
                                       float *output) {
  int index;
  int has_weight = weight != 0;
  float alpha = 1.0f, beta = 0.0f;
  size_t global = UDONDNN_WORDS, local = 1;
  cl_int status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  if (!input || !output || words <= 0 ||
      words > UDONATEN_ARENA_POINTWISE_WORDS || normalized_words <= 0 ||
      normalized_words > 4096 || words % normalized_words != 0 ||
      epsilon < 0.0f)
    return UDONDNN_STATUS_BAD_PARAM;
  if (words > UDONDNN_REGION_WORDS) {
    const size_t arena_global[2] = {8192, 2};
    const size_t arena_local[2] = {1, 1};
#define SET_LARGE_RMS_ARGUMENT(kernel, slot, value)                            \
  do {                                                                         \
    status = clSetKernelArg((kernel), (slot), sizeof(value), &(value));        \
    if (status != CL_SUCCESS)                                                  \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
    SET_LARGE_RMS_ARGUMENT(arena_rms_reduce_kernel, 2, words);
    SET_LARGE_RMS_ARGUMENT(arena_rms_reduce_kernel, 3, normalized_words);
    SET_LARGE_RMS_ARGUMENT(arena_rms_finalize_kernel, 4, words);
    SET_LARGE_RMS_ARGUMENT(arena_rms_finalize_kernel, 5, normalized_words);
    SET_LARGE_RMS_ARGUMENT(arena_rms_finalize_kernel, 6, epsilon);
    SET_LARGE_RMS_ARGUMENT(arena_rms_finalize_kernel, 7, has_weight);
#undef SET_LARGE_RMS_ARGUMENT
    if (clEnqueueWriteBuffer(queue, arena_pointwise_self_buffer, CL_TRUE, 0,
                             sizeof(arena_pointwise_zero), arena_pointwise_zero,
                             0, 0, 0) != CL_SUCCESS ||
        clEnqueueWriteBuffer(queue, arena_pointwise_other_buffer, CL_TRUE, 0,
                             sizeof(arena_pointwise_zero), arena_pointwise_zero,
                             0, 0, 0) != CL_SUCCESS ||
        clEnqueueWriteBuffer(queue, arena_pointwise_self_buffer, CL_TRUE, 0,
                             (size_t)words * sizeof(float), input, 0, 0,
                             0) != CL_SUCCESS ||
        (weight &&
         clEnqueueWriteBuffer(queue, arena_pointwise_other_buffer, CL_TRUE, 0,
                              (size_t)normalized_words * sizeof(float), weight,
                              0, 0, 0) != CL_SUCCESS) ||
        clEnqueueNDRangeKernel(queue, arena_rms_reduce_kernel, 2, 0,
                               arena_global, arena_local, 0, 0,
                               0) != CL_SUCCESS ||
        clEnqueueNDRangeKernel(queue, arena_rms_finalize_kernel, 2, 0,
                               arena_global, arena_local, 0, 0,
                               0) != CL_SUCCESS ||
        clEnqueueReadBuffer(queue, arena_pointwise_output_buffer, CL_TRUE, 0,
                            (size_t)words * sizeof(float), output, 0, 0,
                            0) != CL_SUCCESS ||
        clFinish(queue) != CL_SUCCESS)
      return UDONDNN_STATUS_EXECUTION_FAILED;
    return UDONDNN_STATUS_SUCCESS;
  }
  for (index = 0; index < UDONDNN_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < words; ++index)
    packed[index] = input[index];
  for (index = 0; index < normalized_words; ++index)
    packed[UDONATEN_RMS_SMALL_WEIGHT_OFFSET + index] =
        weight ? weight[index] : 1.0f;

#define SET_ATEN_RMS_ARGUMENT(slot, value)                                     \
  do {                                                                         \
    status =                                                                   \
        clSetKernelArg(rms_norm_small_kernel, slot, sizeof(value), &(value));  \
    if (status != CL_SUCCESS)                                                  \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
  SET_ATEN_RMS_ARGUMENT(2, words);
  SET_ATEN_RMS_ARGUMENT(3, normalized_words);
  SET_ATEN_RMS_ARGUMENT(4, epsilon);
  SET_ATEN_RMS_ARGUMENT(5, alpha);
  SET_ATEN_RMS_ARGUMENT(6, beta);
#undef SET_ATEN_RMS_ARGUMENT

  if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(packed),
                           packed, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, rms_norm_small_kernel, 1, 0, &global,
                             &local, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueReadBuffer(queue, output_buffer, CL_TRUE,
                          UDONATEN_RMS_SMALL_RESULT_OFFSET * sizeof(float),
                          (size_t)words * sizeof(float), result, 0, 0,
                          0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONDNN_STATUS_EXECUTION_FAILED;
  for (index = 0; index < words; ++index)
    output[index] = result[index];
  return UDONDNN_STATUS_SUCCESS;
}

/* Private UdonATen AOT backend. Add, Mul, and Sigmoid remain public
 * udonaten* operations; this entry only validates their arena-native shader. */
udondnnStatus_t udonatenBackendArenaPointwise(udondnnHandle_t handle, int words,
                                              const float *self,
                                              const float *other, int operation,
                                              float *output) {
  const size_t global[2] = {8192, 2};
  const size_t local[2] = {1, 1};
  cl_int status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  if (!self || !other || !output || words <= 0 ||
      words > UDONATEN_ARENA_POINTWISE_WORDS || operation < 0 || operation > 2)
    return UDONDNN_STATUS_BAD_PARAM;
#define SET_ARENA_POINTWISE_ARGUMENT(slot, value)                              \
  do {                                                                         \
    status =                                                                   \
        clSetKernelArg(arena_pointwise_kernel, slot, sizeof(value), &(value)); \
    if (status != CL_SUCCESS)                                                  \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
  SET_ARENA_POINTWISE_ARGUMENT(3, words);
  SET_ARENA_POINTWISE_ARGUMENT(4, operation);
#undef SET_ARENA_POINTWISE_ARGUMENT
  if (clEnqueueWriteBuffer(queue, arena_pointwise_self_buffer, CL_TRUE, 0,
                           sizeof(arena_pointwise_zero), arena_pointwise_zero,
                           0, 0, 0) != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, arena_pointwise_other_buffer, CL_TRUE, 0,
                           sizeof(arena_pointwise_zero), arena_pointwise_zero,
                           0, 0, 0) != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, arena_pointwise_self_buffer, CL_TRUE, 0,
                           (size_t)words * sizeof(float), self, 0, 0,
                           0) != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, arena_pointwise_other_buffer, CL_TRUE, 0,
                           (size_t)words * sizeof(float), other, 0, 0,
                           0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, arena_pointwise_kernel, 2, 0, global, local,
                             0, 0, 0) != CL_SUCCESS ||
      clEnqueueReadBuffer(queue, arena_pointwise_output_buffer, CL_TRUE, 0,
                          (size_t)words * sizeof(float), output, 0, 0,
                          0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONDNN_STATUS_EXECUTION_FAILED;
  return UDONDNN_STATUS_SUCCESS;
}

/* Private ATen embedding backend. The public operation is udonatenEmbedding;
 * this implementation intentionally does not appear in udondnn.h. */
udondnnStatus_t udonatenBackendEmbedding(
    udondnnHandle_t handle, udondnnTensorDescriptor_t weight_descriptor,
    const float *weight, const int32_t *indices, int indices_count,
    udondnnTensorDescriptor_t output_descriptor, float *output) {
  TensorDescriptor *data_info, *output_info;
  int dimensions[4], data_words, output_words, outer = 1, inner = 1;
  int index, selected, axis = 0;
  size_t global = UDONDNN_WORDS, local = 1;
  cl_int status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  data_info = get_tensor(weight_descriptor);
  output_info = get_tensor(output_descriptor);
  if (!data_info || !data_info->configured || !output_info ||
      !output_info->configured || !weight || !indices || indices_count <= 0 ||
      !output)
    return UDONDNN_STATUS_BAD_PARAM;
  dimensions[0] = data_info->n;
  dimensions[1] = data_info->c;
  dimensions[2] = data_info->h;
  dimensions[3] = data_info->w;
  for (index = 0; index < axis; ++index)
    outer *= dimensions[index];
  for (index = axis + 1; index < 4; ++index)
    inner *= dimensions[index];
  data_words = outer * dimensions[axis] * inner;
  output_words =
      output_info->n * output_info->c * output_info->h * output_info->w;
  if (data_words > UDONDNN_REGION_WORDS ||
      indices_count > UDONDNN_REGION_WORDS ||
      output_words > UDONDNN_REGION_WORDS ||
      output_words != outer * indices_count * inner)
    return UDONDNN_STATUS_BAD_PARAM;
  for (index = 0; index < indices_count; ++index) {
    selected = indices[index];
    if (selected < 0)
      selected += dimensions[axis];
    if (selected < 0 || selected >= dimensions[axis])
      return UDONDNN_STATUS_BAD_PARAM;
  }
  for (index = 0; index < UDONDNN_WORDS; ++index)
    packed[index] = 0.0f;
  for (index = 0; index < data_words; ++index)
    packed[index] = weight[index];
  for (index = 0; index < indices_count; ++index)
    packed[UDONATEN_EMBEDDING_INDICES_OFFSET + index] = (float)indices[index];

#define SET_EMBEDDING_ARGUMENT(slot, value)                                    \
  do {                                                                         \
    status = clSetKernelArg(embedding_kernel, slot, sizeof(value), &(value));  \
    if (status != CL_SUCCESS)                                                  \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
  SET_EMBEDDING_ARGUMENT(2, outer);
  SET_EMBEDDING_ARGUMENT(3, dimensions[axis]);
  {
    float axis_words_float = (float)dimensions[axis];
    SET_EMBEDDING_ARGUMENT(4, axis_words_float);
  }
  SET_EMBEDDING_ARGUMENT(5, inner);
  SET_EMBEDDING_ARGUMENT(6, indices_count);
  SET_EMBEDDING_ARGUMENT(7, output_words);
#undef SET_EMBEDDING_ARGUMENT

  if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(packed),
                           packed, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, embedding_kernel, 1, 0, &global, &local, 0,
                             0, 0) != CL_SUCCESS ||
      clEnqueueReadBuffer(queue, output_buffer, CL_TRUE,
                          UDONATEN_EMBEDDING_RESULT_OFFSET * sizeof(float),
                          output_words * sizeof(float), result, 0, 0,
                          0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONDNN_STATUS_EXECUTION_FAILED;
  for (index = 0; index < output_words; ++index)
    output[index] = result[index];
  return UDONDNN_STATUS_SUCCESS;
}

/* Private UdonATen backend. This symbol is intentionally absent from the
 * UdonDNN public header: ownership and validation remain in UdonATen. */
udondnnStatus_t
udonatenBackendWeightInt4packMm(udondnnHandle_t handle, int m, int n, int k,
                                const float *input,
                                const uint8_t *packed_weight, int group_size,
                                const float *scales_and_zeros, float *output) {
  int m_offset, n_offset, k_offset, tile_n, tile_k, blocks, row_stride, row,
      block;
  int byte_index, index, accumulate;
  size_t global = UDONATEN_INT4_WORDS, local = 1;
  cl_int status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  if (m <= 0 || n <= 0 || k <= 0 || group_size != 32 || k % group_size ||
      !input || !packed_weight || !scales_and_zeros || !output)
    return UDONDNN_STATUS_BAD_PARAM;
#define SET_INT4_ARGUMENT(slot, value)                                         \
  do {                                                                         \
    status = clSetKernelArg(weight_int4pack_mm_kernel, slot, sizeof(value),    \
                            &(value));                                         \
    if (status != CL_SUCCESS)                                                  \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
  for (m_offset = 0; m_offset < m; ++m_offset)
    for (n_offset = 0; n_offset < n; n_offset += UDONATEN_INT4_MAX_N) {
      tile_n = n - n_offset;
      if (tile_n > UDONATEN_INT4_MAX_N)
        tile_n = UDONATEN_INT4_MAX_N;
      for (k_offset = 0; k_offset < k; k_offset += UDONATEN_INT4_MAX_K) {
        tile_k = k - k_offset;
        if (tile_k > UDONATEN_INT4_MAX_K)
          tile_k = UDONATEN_INT4_MAX_K;
        blocks = tile_k / group_size;
        row_stride = blocks * 18;
        accumulate = k_offset != 0;
        for (index = 0; index < UDONATEN_INT4_WORDS; ++index)
          int4_state[index] = 0.0f;
        for (row = 0; row < tile_n; ++row) {
          int source_row = n_offset + row;
          for (block = 0; block < blocks; ++block) {
            int source_block = k_offset / group_size + block;
            int destination = row * row_stride + block * 18;
            int source = source_row * (k / 2) + k_offset / 2 + block * 16;
            int qparams = (source_block * n + source_row) * 2;
            for (byte_index = 0; byte_index < 16; ++byte_index)
              int4_state[destination + byte_index] =
                  (float)packed_weight[source + byte_index];
            int4_state[destination + 16] = scales_and_zeros[qparams];
            int4_state[destination + 17] = scales_and_zeros[qparams + 1];
          }
          if (accumulate)
            int4_state[UDONATEN_INT4_RESULT_OFFSET + row] =
                output[m_offset * n + n_offset + row];
        }
        for (index = 0; index < tile_k; ++index)
          int4_state[UDONATEN_INT4_INPUT_OFFSET + index] =
              input[m_offset * k + k_offset + index];
        SET_INT4_ARGUMENT(2, tile_n);
        SET_INT4_ARGUMENT(3, tile_k);
        SET_INT4_ARGUMENT(4, row_stride);
        SET_INT4_ARGUMENT(5, accumulate);
        if (clEnqueueWriteBuffer(queue, int4_input_buffer, CL_TRUE, 0,
                                 sizeof(int4_state), int4_state, 0, 0,
                                 0) != CL_SUCCESS ||
            clEnqueueNDRangeKernel(queue, weight_int4pack_mm_kernel, 1, 0,
                                   &global, &local, 0, 0, 0) != CL_SUCCESS ||
            clEnqueueReadBuffer(queue, int4_output_buffer, CL_TRUE,
                                UDONATEN_INT4_RESULT_OFFSET * sizeof(float),
                                tile_n * sizeof(float),
                                output + m_offset * n + n_offset, 0, 0,
                                0) != CL_SUCCESS ||
            clFinish(queue) != CL_SUCCESS)
          return UDONDNN_STATUS_EXECUTION_FAILED;
      }
    }
#undef SET_INT4_ARGUMENT
  return UDONDNN_STATUS_SUCCESS;
}

/* Private UdonATen backend. ATen owns tile; UdonDNN only supplies the shared
 * OpenCL runtime and descriptor implementation. */
udondnnStatus_t udonatenBackendTile(udondnnHandle_t handle, int input_n,
                                    int input_c, int input_h, int input_w,
                                    const float *input, int repeat_n,
                                    int repeat_c, int repeat_h, int repeat_w,
                                    int output_words, float *output) {
  int input_words = input_n * input_c * input_h * input_w;
  const size_t global[2] = {8192, 2};
  const size_t local[2] = {1, 1};
  cl_int status;
  if (handle != UDONDNN_HANDLE || !context)
    return UDONDNN_STATUS_NOT_INITIALIZED;
  if (!input || !output || input_n <= 0 || input_c <= 0 || input_h <= 0 ||
      input_w <= 0 || repeat_n <= 0 || repeat_c <= 0 || repeat_h <= 0 ||
      repeat_w <= 0 || input_words > UDONATEN_ARENA_POINTWISE_WORDS ||
      output_words <= 0 || output_words > UDONATEN_ARENA_POINTWISE_WORDS)
    return UDONDNN_STATUS_BAD_PARAM;
#define SET_TILE_ARGUMENT(slot, value)                                         \
  do {                                                                         \
    status = clSetKernelArg(arena_tile_kernel, slot, sizeof(value), &(value)); \
    if (status != CL_SUCCESS)                                                  \
      return UDONDNN_STATUS_EXECUTION_FAILED;                                  \
  } while (0)
  SET_TILE_ARGUMENT(2, input_n);
  SET_TILE_ARGUMENT(3, input_c);
  SET_TILE_ARGUMENT(4, input_h);
  SET_TILE_ARGUMENT(5, input_w);
  SET_TILE_ARGUMENT(6, repeat_n);
  SET_TILE_ARGUMENT(7, repeat_c);
  SET_TILE_ARGUMENT(8, repeat_h);
  SET_TILE_ARGUMENT(9, repeat_w);
  SET_TILE_ARGUMENT(10, output_words);
#undef SET_TILE_ARGUMENT
  if (clEnqueueWriteBuffer(queue, arena_pointwise_self_buffer, CL_TRUE, 0,
                           sizeof(arena_pointwise_zero), arena_pointwise_zero,
                           0, 0, 0) != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, arena_pointwise_self_buffer, CL_TRUE, 0,
                           (size_t)input_words * sizeof(float), input, 0, 0,
                           0) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, arena_tile_kernel, 2, 0, global, local, 0,
                             0, 0) != CL_SUCCESS ||
      clEnqueueReadBuffer(queue, arena_pointwise_output_buffer, CL_TRUE, 0,
                          (size_t)output_words * sizeof(float), output, 0, 0,
                          0) != CL_SUCCESS ||
      clFinish(queue) != CL_SUCCESS)
    return UDONDNN_STATUS_EXECUTION_FAILED;
  return UDONDNN_STATUS_SUCCESS;
}
