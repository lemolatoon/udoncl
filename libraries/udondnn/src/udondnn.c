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
#define UDONDNN_HANDLE 1u
#define UDONDNN_DESCRIPTOR_SLOTS 8
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

static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel convolution_kernel;
static cl_kernel activation_kernel;
static cl_kernel pooling_kernel;
static cl_kernel softmax_kernel;
static cl_mem input_buffer;
static cl_mem output_buffer;
static float packed[UDONDNN_WORDS];
static float result[UDONDNN_REGION_WORDS];
static TensorDescriptor tensors[UDONDNN_DESCRIPTOR_SLOTS];
static FilterDescriptor filters[UDONDNN_DESCRIPTOR_SLOTS];
static ConvolutionDescriptor convolutions[UDONDNN_DESCRIPTOR_SLOTS];
static ActivationDescriptor activations[UDONDNN_DESCRIPTOR_SLOTS];
static PoolingDescriptor poolings[UDONDNN_DESCRIPTOR_SLOTS];

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

static void clear_objects(void) {
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
  if (queue)
    clReleaseCommandQueue(queue);
  if (context)
    clReleaseContext(context);
  convolution_kernel = 0;
  activation_kernel = 0;
  pooling_kernel = 0;
  softmax_kernel = 0;
  program = 0;
  output_buffer = input_buffer = 0;
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
      n > 15 || c > 15 || h > 15 || w > 15 ||
      n * c * h * w > UDONDNN_CONV_MAX_TENSOR_WORDS)
    return UDONDNN_STATUS_BAD_PARAM;
  tensor->n = n;
  tensor->c = c;
  tensor->h = h;
  tensor->w = w;
  tensor->configured = 1;
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
