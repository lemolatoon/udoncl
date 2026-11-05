#include "udononnxoperators.h"
#include <udononnxoperators_device_source.h>
#include <CL/cl.h>
#include <limits.h>

#define UDONONNX_OPERATOR_HANDLE 1u
#define UDONONNX_NBITS_WORDS 16384
#define UDONONNX_NBITS_MAX_N 32
#define UDONONNX_NBITS_MAX_K 128
#define UDONONNX_NBITS_SCALE_OFFSET 4096
#define UDONONNX_NBITS_ZERO_OFFSET 8192
#define UDONONNX_NBITS_INPUT_OFFSET 12288
#define UDONONNX_NBITS_BIAS_OFFSET 12416
#define UDONONNX_NBITS_RESULT_OFFSET 12544

static cl_context operator_context;
static cl_command_queue operator_queue;
static cl_program operator_program;
static cl_kernel matmul_nbits_kernel;
static cl_kernel gather_kernel;
static cl_mem state_buffer;
static cl_mem result_buffer;
static float state[UDONONNX_NBITS_WORDS];
static float result[UDONONNX_NBITS_MAX_N];

static void clear_operator(void) {
  if (gather_kernel)
    clReleaseKernel(gather_kernel);
  if (matmul_nbits_kernel)
    clReleaseKernel(matmul_nbits_kernel);
  if (operator_program)
    clReleaseProgram(operator_program);
  if (result_buffer)
    clReleaseMemObject(result_buffer);
  if (state_buffer)
    clReleaseMemObject(state_buffer);
  if (operator_queue)
    clReleaseCommandQueue(operator_queue);
  if (operator_context)
    clReleaseContext(operator_context);
  matmul_nbits_kernel = 0;
  gather_kernel = 0;
  operator_program = 0;
  result_buffer = state_buffer = 0;
  operator_queue = 0;
  operator_context = 0;
}

static udononnxStatus_t execution_failure(void) {
  clear_operator();
  return UDONONNX_STATUS_EXECUTION_FAILURE;
}

udononnxStatus_t
udononnxCreateOperatorHandle(udononnxOperatorHandle_t *handle) {
  cl_platform_id platform;
  cl_device_id device;
  const char *source = UDONONNX_DEVICE_SOURCE;
  cl_int status;
  if (!handle)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (operator_context)
    return UDONONNX_STATUS_BUSY;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, 0) !=
          CL_SUCCESS)
    return execution_failure();
  operator_context = clCreateContext(0, 1, &device, 0, 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  operator_queue = clCreateCommandQueue(operator_context, device, 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  state_buffer = clCreateBuffer(operator_context, CL_MEM_READ_WRITE,
                                sizeof(state), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  result_buffer = clCreateBuffer(operator_context, CL_MEM_READ_WRITE,
                                 sizeof(state), 0, &status);
  if (status != CL_SUCCESS)
    return execution_failure();
  operator_program =
      clCreateProgramWithSource(operator_context, 1, &source, 0, &status);
  if (status != CL_SUCCESS ||
      clBuildProgram(operator_program, 0, 0, "-cl-std=CL1.2", 0, 0) !=
          CL_SUCCESS)
    return execution_failure();
  matmul_nbits_kernel =
      clCreateKernel(operator_program, "udononnx_matmul_nbits", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(matmul_nbits_kernel, 0, sizeof(state_buffer),
                     &state_buffer) != CL_SUCCESS ||
      clSetKernelArg(matmul_nbits_kernel, 1, sizeof(result_buffer),
                     &result_buffer) != CL_SUCCESS)
    return execution_failure();
  gather_kernel =
      clCreateKernel(operator_program, "udononnx_gather_copy", &status);
  if (status != CL_SUCCESS ||
      clSetKernelArg(gather_kernel, 0, sizeof(state_buffer), &state_buffer) !=
          CL_SUCCESS ||
      clSetKernelArg(gather_kernel, 1, sizeof(result_buffer), &result_buffer) !=
          CL_SUCCESS)
    return execution_failure();
  *handle = UDONONNX_OPERATOR_HANDLE;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t
udononnxDestroyOperatorHandle(udononnxOperatorHandle_t handle) {
  if (handle != UDONONNX_OPERATOR_HANDLE || !operator_context)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  clear_operator();
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxGather(udononnxOperatorHandle_t handle,
                                udonclDataType_t data_type, const void *data,
                                const int64_t *data_shape, int32_t data_rank,
                                udonclDataType_t indices_type,
                                const void *indices, int32_t indices_count,
                                int64_t axis, void *output) {
  int64_t element_bytes, outer = 1, inner = 1, axis_elements;
  int64_t outer_index, index_position, index_count;
  int32_t dimension;
  if (handle != UDONONNX_OPERATOR_HANDLE || !operator_context)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  element_bytes = (int32_t)udonclDataTypeSize(data_type);
  if (!data || !data_shape || !data_rank ||
      data_rank > UDONONNX_MAX_TENSOR_RANK || !indices || !indices_count ||
      !output || !element_bytes ||
      (indices_type != UDONCL_DATA_INT32 && indices_type != UDONCL_DATA_INT64))
    return UDONONNX_STATUS_INVALID_VALUE;
  index_count = (int64_t)indices_count;
  if (axis < 0)
    axis += data_rank;
  if (axis < 0 || axis >= data_rank)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (dimension = 0; dimension < data_rank; ++dimension) {
    int64_t extent;
    if (data_shape[dimension] <= 0 || data_shape[dimension] > INT32_MAX)
      return UDONONNX_STATUS_INVALID_VALUE;
    extent = data_shape[dimension];
    if (dimension < axis) {
      if (outer * extent > INT32_MAX)
        return UDONONNX_STATUS_INVALID_VALUE;
      outer *= extent;
    } else if (dimension > axis) {
      if (inner * extent > INT32_MAX)
        return UDONONNX_STATUS_INVALID_VALUE;
      inner *= extent;
    }
  }
  axis_elements = data_shape[axis];
  if (inner * element_bytes > INT32_MAX)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (outer_index = 0; outer_index < outer; ++outer_index)
    for (index_position = 0; index_position < index_count; ++index_position) {
      int64_t selected =
          indices_type == UDONCL_DATA_INT32
              ? (int64_t)((const int32_t *)indices)[index_position]
              : ((const int64_t *)indices)[index_position];
      int64_t source_offset, output_offset, bytes = inner * element_bytes;
      size_t global = UDONONNX_NBITS_WORDS, local = 1;
      int words;
      cl_int cl_status;
      if (selected < 0)
        selected += axis_elements;
      if (selected < 0 || selected >= axis_elements)
        return UDONONNX_STATUS_INVALID_VALUE;
      source_offset =
          (outer_index * axis_elements + selected) * inner * element_bytes;
      output_offset =
          (outer_index * index_count + index_position) * inner * element_bytes;
      if (bytes > (int64_t)sizeof(state))
        return UDONONNX_STATUS_INVALID_VALUE;
      words = ((int)bytes + 3) >> 2;
      cl_status = clSetKernelArg(gather_kernel, 2, sizeof(words), &words);
      if (cl_status != CL_SUCCESS ||
          clEnqueueWriteBuffer(operator_queue, state_buffer, CL_TRUE, 0, bytes,
                               (const uint8_t *)data + source_offset, 0, 0,
                               0) != CL_SUCCESS ||
          clEnqueueNDRangeKernel(operator_queue, gather_kernel, 1, 0, &global,
                                 &local, 0, 0, 0) != CL_SUCCESS ||
          clEnqueueReadBuffer(operator_queue, result_buffer, CL_TRUE, 0, bytes,
                              (uint8_t *)output + output_offset, 0, 0,
                              0) != CL_SUCCESS ||
          clFinish(operator_queue) != CL_SUCCESS)
        return execution_failure();
    }
  return UDONONNX_STATUS_SUCCESS;
}

static float half_to_float(uint16_t value) {
  unsigned sign = (unsigned)(value >> 15);
  int exponent = (int)((unsigned)(value >> 10) & 31u);
  int fraction = (int)((unsigned)value & 1023u);
  float magnitude;
  if (!exponent) {
    magnitude = (float)fraction / 16777216.0f;
  } else if (exponent == 31) {
    union {
      uint32_t bits;
      float value;
    } special;
    special.bits = fraction ? 0x7fc00000u : 0x7f800000u;
    magnitude = special.value;
  } else {
    int power = exponent - 15;
    magnitude = 1.0f + (float)fraction / 1024.0f;
    while (power > 0) {
      magnitude *= 2.0f;
      --power;
    }
    while (power < 0) {
      magnitude *= 0.5f;
      ++power;
    }
  }
  return sign ? 0.0f - magnitude : magnitude;
}

static float load_value(const void *data, size_t index, udonclDataType_t type) {
  if (type == UDONCL_DATA_FLOAT32)
    return ((const float *)data)[index];
  if (type == UDONCL_DATA_FLOAT16)
    return half_to_float(((const uint16_t *)data)[index]);
  if (type == UDONCL_DATA_BFLOAT16) {
    union {
      uint32_t bits;
      float value;
    } converted;
    converted.bits = (uint32_t)((const uint16_t *)data)[index] << 16u;
    return converted.value;
  }
  return 0.0f;
}

static uint16_t float_to_bfloat16(float value) {
  union {
    float value;
    uint32_t bits;
  } converted;
  uint32_t rounding;
  converted.value = value;
  rounding = 0x7fffu + ((converted.bits >> 16u) & 1u);
  return (uint16_t)((converted.bits + rounding) >> 16u);
}

static uint16_t float_to_half(float value) {
  union {
    float value;
    uint32_t bits;
  } converted;
  unsigned sign, exponent, fraction;
  int half_exponent;
  converted.value = value;
  sign = converted.bits >> 31u;
  exponent = (converted.bits >> 23u) & 255u;
  fraction = converted.bits & 0x7fffffu;
  if (exponent == 255u)
    return (uint16_t)((sign << 15u) | 0x7c00u | (fraction ? 0x0200u : 0u));
  half_exponent = (int)exponent - 127 + 15;
  if (half_exponent >= 31)
    return (uint16_t)((sign << 15u) | 0x7c00u);
  if (half_exponent <= 0) {
    if (half_exponent < -10)
      return (uint16_t)(sign << 15u);
    fraction = (fraction | 0x800000u) >> (unsigned)(1 - half_exponent);
    return (uint16_t)((sign << 15u) | ((fraction + 0x1000u) >> 13u));
  }
  return (uint16_t)((sign << 15u) | ((unsigned)half_exponent << 10u) |
                    ((fraction + 0x1000u) >> 13u));
}

static void store_value(void *data, size_t index, udonclDataType_t type,
                        float value) {
  if (type == UDONCL_DATA_FLOAT32)
    ((float *)data)[index] = value;
  else if (type == UDONCL_DATA_FLOAT16)
    ((uint16_t *)data)[index] = float_to_half(value);
  else
    ((uint16_t *)data)[index] = float_to_bfloat16(value);
}

static unsigned packed_value(const uint8_t *data, size_t bytes,
                             size_t bit_offset, int bits, int *valid) {
  size_t byte = bit_offset >> 3u;
  unsigned shift = (unsigned)(bit_offset & 7u);
  unsigned value;
  if (byte >= bytes) {
    *valid = 0;
    return 0;
  }
  value = data[byte];
  if (shift + (unsigned)bits > 8u) {
    if (byte + 1u >= bytes) {
      *valid = 0;
      return 0;
    }
    value |= (unsigned)data[byte + 1u] << 8u;
  }
  return (value >> shift) & ((1u << (unsigned)bits) - 1u);
}

udononnxStatus_t udononnxMatMulNBits(
    udononnxOperatorHandle_t handle, udonclDataType_t data_type, const void *a,
    int64_t m, int64_t k, const uint8_t *b, size_t b_bytes, const void *scales,
    size_t scale_count, const uint8_t *zero_points, size_t zero_point_bytes,
    udononnxZeroPointFormat_t zero_point_format, const int32_t *g_idx,
    size_t g_idx_count, const void *bias, size_t bias_count, int64_t n,
    int block_size, int bits, void *y) {
  int row, column_offset, inner_offset;
  int blocks, packed_block_bytes;
  size_t required_b;
  unsigned mask = (1u << (unsigned)bits) - 1u;
  if (handle != UDONONNX_OPERATOR_HANDLE || !operator_context)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if ((data_type != UDONCL_DATA_FLOAT32 && data_type != UDONCL_DATA_FLOAT16 &&
       data_type != UDONCL_DATA_BFLOAT16) ||
      !a || !b || !scales || !y || m <= 0 || n <= 0 || k <= 0 || bits < 2 ||
      bits > 8 || block_size < 16 ||
      ((unsigned)block_size & ((unsigned)block_size - 1u)) != 0u ||
      zero_point_format < UDONONNX_ZERO_POINT_NONE ||
      zero_point_format > UDONONNX_ZERO_POINT_UNPACKED || m > INT_MAX ||
      n > INT_MAX || k > INT_MAX ||
      (zero_point_format != UDONONNX_ZERO_POINT_NONE && !zero_points) ||
      (g_idx && g_idx_count != (size_t)k) || (!g_idx && g_idx_count) ||
      (bias && bias_count != (size_t)n) || (!bias && bias_count))
    return UDONONNX_STATUS_INVALID_VALUE;
  blocks = ((int)k + block_size - 1) / block_size;
  packed_block_bytes = block_size * bits / 8;
  required_b = (size_t)n * (size_t)blocks * (size_t)packed_block_bytes;
  if (b_bytes < required_b)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (scale_count < (size_t)n * (size_t)blocks)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (zero_point_format == UDONONNX_ZERO_POINT_PACKED &&
      zero_point_bytes * 8u < (size_t)n * (size_t)blocks * (size_t)bits)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (zero_point_format == UDONONNX_ZERO_POINT_UNPACKED &&
      zero_point_bytes < (size_t)n * (size_t)blocks)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (row = 0; row < m; ++row)
    for (column_offset = 0; column_offset < n;
         column_offset += UDONONNX_NBITS_MAX_N) {
      int tile_n = (int)(n - column_offset);
      int accumulate = 0;
      if (tile_n > UDONONNX_NBITS_MAX_N)
        tile_n = UDONONNX_NBITS_MAX_N;
      for (inner_offset = 0; inner_offset < k;
           inner_offset += UDONONNX_NBITS_MAX_K) {
        int tile_k = (int)(k - inner_offset);
        int finalize;
        int local_n, local_k;
        size_t global = UDONONNX_NBITS_WORDS, local = 1;
        cl_int cl_status;
        if (tile_k > UDONONNX_NBITS_MAX_K)
          tile_k = UDONONNX_NBITS_MAX_K;
        finalize = inner_offset + tile_k == k;
        for (local_n = 0; local_n < tile_n; ++local_n) {
          int output_column = column_offset + local_n;
          for (local_k = 0; local_k < tile_k; ++local_k) {
            int source_k = inner_offset + local_k;
            int block = g_idx ? g_idx[source_k] : source_k / block_size;
            int in_block = source_k % block_size;
            int state_index = local_n * UDONONNX_NBITS_MAX_K + local_k;
            size_t row_bit =
                (size_t)((output_column * blocks + source_k / block_size) *
                             packed_block_bytes * 8 +
                         in_block * bits);
            int scale_index;
            int quantized, zero = 1 << (bits - 1);
            int valid = 1;
            if (block < 0 || block >= blocks)
              return UDONONNX_STATUS_INVALID_VALUE;
            scale_index = output_column * blocks + block;
            quantized = (int)packed_value(b, b_bytes, row_bit, bits, &valid);
            if (zero_point_format == UDONONNX_ZERO_POINT_PACKED)
              zero = (int)packed_value(zero_points, zero_point_bytes,
                                       (size_t)scale_index * (size_t)bits, bits,
                                       &valid);
            else if (zero_point_format == UDONONNX_ZERO_POINT_UNPACKED)
              zero = zero_points[scale_index] & mask;
            if (!valid)
              return UDONONNX_STATUS_INVALID_VALUE;
            state[state_index] = (float)quantized;
            state[UDONONNX_NBITS_SCALE_OFFSET + state_index] =
                load_value(scales, scale_index, data_type);
            state[UDONONNX_NBITS_ZERO_OFFSET + state_index] = (float)zero;
          }
          state[UDONONNX_NBITS_BIAS_OFFSET + (size_t)local_n] =
              bias ? load_value(bias, output_column, data_type) : 0.0f;
          if (accumulate)
            state[UDONONNX_NBITS_RESULT_OFFSET + (size_t)local_n] = load_value(
                y, (size_t)row * (size_t)n + output_column, data_type);
        }
        for (local_k = 0; local_k < tile_k; ++local_k)
          state[UDONONNX_NBITS_INPUT_OFFSET + (size_t)local_k] = load_value(
              a,
              (size_t)row * (size_t)k + (size_t)inner_offset + (size_t)local_k,
              data_type);
#define SET_ARGUMENT(slot, value)                                              \
  do {                                                                         \
    cl_status =                                                                \
        clSetKernelArg(matmul_nbits_kernel, slot, sizeof(value), &(value));    \
    if (cl_status != CL_SUCCESS)                                               \
      return execution_failure();                                              \
  } while (0)
        SET_ARGUMENT(2, tile_n);
        SET_ARGUMENT(3, tile_k);
        SET_ARGUMENT(4, accumulate);
        SET_ARGUMENT(5, finalize);
#undef SET_ARGUMENT
        if (clEnqueueWriteBuffer(operator_queue, state_buffer, CL_TRUE, 0,
                                 sizeof(state), state, 0, 0, 0) != CL_SUCCESS ||
            clEnqueueNDRangeKernel(operator_queue, matmul_nbits_kernel, 1, 0,
                                   &global, &local, 0, 0, 0) != CL_SUCCESS ||
            clEnqueueReadBuffer(operator_queue, result_buffer, CL_TRUE,
                                UDONONNX_NBITS_RESULT_OFFSET * sizeof(float),
                                (size_t)tile_n * sizeof(float), result, 0, 0,
                                0) != CL_SUCCESS ||
            clFinish(operator_queue) != CL_SUCCESS)
          return execution_failure();
        for (local_n = 0; local_n < tile_n; ++local_n)
          store_value(y,
                      (size_t)row * (size_t)n + (size_t)column_offset +
                          (size_t)local_n,
                      data_type, result[local_n]);
        accumulate = 1;
      }
    }
  return UDONONNX_STATUS_SUCCESS;
}
