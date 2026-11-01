#include "udonaten.h"
#include <float.h>
#include <math.h>

#define UDONATEN_POINTWISE_TILE_WORDS 512
#define UDONATEN_ARENA_WORDS 16384

static int udonatenPointwiseTileShape(size_t remaining, int *n, int *c, int *h,
                                      int *w) {
  int candidate = remaining > UDONATEN_POINTWISE_TILE_WORDS
                      ? UDONATEN_POINTWISE_TILE_WORDS
                      : (int)remaining;
  for (; candidate > 0; --candidate) {
    int candidate_n, candidate_c, candidate_h;
    for (candidate_n = 1; candidate_n <= 15; ++candidate_n)
      for (candidate_c = 1; candidate_c <= 15; ++candidate_c)
        for (candidate_h = 1; candidate_h <= 15; ++candidate_h) {
          int divisor = candidate_n * candidate_c * candidate_h;
          int candidate_w;
          if (candidate % divisor)
            continue;
          candidate_w = candidate / divisor;
          if (candidate_w > 0 && candidate_w <= 15) {
            *n = candidate_n;
            *c = candidate_c;
            *h = candidate_h;
            *w = candidate_w;
            return candidate;
          }
        }
  }
  return 0;
}

udonatenStatus_t
udonatenBackendWeightInt4packMm(udonatenHandle_t handle, int m, int n, int k,
                                const float *input,
                                const uint8_t *packed_weight, int group_size,
                                const float *scales_and_zeros, float *output);

udonatenStatus_t udonatenBackendTile(udonatenHandle_t handle, int input_n,
                                     int input_c, int input_h, int input_w,
                                     const float *input, int repeat_n,
                                     int repeat_c, int repeat_h, int repeat_w,
                                     int output_words, float *output);
udonatenStatus_t udonatenBackendRmsNorm(udonatenHandle_t handle, int words,
                                        int normalized_words,
                                        const float *input, const float *weight,
                                        float epsilon, float *output);
udonatenStatus_t udonatenBackendEmbedding(
    udonatenHandle_t handle, udonatenTensorDescriptor_t weight_descriptor,
    const float *weight, const int32_t *indices, int indices_count,
    udonatenTensorDescriptor_t output_descriptor, float *output);

static udonatenStatus_t
udonatenBinary(udonatenHandle_t handle, udondnnOpTensorOp_t operation,
               udonatenTensorDescriptor_t self_descriptor, const float *self,
               udonatenTensorDescriptor_t other_descriptor, const float *other,
               float other_alpha, udonatenTensorDescriptor_t output_descriptor,
               float *output) {
  const float one = 1.0f, zero = 0.0f;
  udondnnOpTensorDescriptor_t descriptor = 0;
  udondnnTensorDescriptor_t tile_descriptor = 0;
  udondnnDataType_t self_type, other_type, output_type;
  int self_n, self_c, self_h, self_w;
  int other_n, other_c, other_h, other_w;
  int output_n, output_c, output_h, output_w;
  int n_stride, c_stride, h_stride, w_stride;
  size_t words = 0, offset;
  udonatenStatus_t status = udondnnCreateOpTensorDescriptor(&descriptor);
  if (status != UDONDNN_STATUS_SUCCESS)
    return status;
  status = udondnnSetOpTensorDescriptor(
      descriptor, operation, UDONDNN_DATA_FLOAT, UDONDNN_PROPAGATE_NAN);
  if (status == UDONDNN_STATUS_SUCCESS)
    status = udondnnGetTensor4dDescriptor(self_descriptor, &self_type, &self_n,
                                          &self_c, &self_h, &self_w, &n_stride,
                                          &c_stride, &h_stride, &w_stride);
  if (status == UDONDNN_STATUS_SUCCESS)
    status = udondnnGetTensor4dDescriptor(
        other_descriptor, &other_type, &other_n, &other_c, &other_h, &other_w,
        &n_stride, &c_stride, &h_stride, &w_stride);
  if (status == UDONDNN_STATUS_SUCCESS)
    status = udondnnGetTensor4dDescriptor(
        output_descriptor, &output_type, &output_n, &output_c, &output_h,
        &output_w, &n_stride, &c_stride, &h_stride, &w_stride);
  if (status == UDONDNN_STATUS_SUCCESS)
    words = (size_t)output_n * (size_t)output_c * (size_t)output_h *
            (size_t)output_w;
  if (status == UDONDNN_STATUS_SUCCESS &&
      words > UDONATEN_POINTWISE_TILE_WORDS) {
    if (self_type != UDONDNN_DATA_FLOAT || other_type != UDONDNN_DATA_FLOAT ||
        output_type != UDONDNN_DATA_FLOAT || self_n != output_n ||
        self_c != output_c || self_h != output_h || self_w != output_w ||
        other_n != output_n || other_c != output_c || other_h != output_h ||
        other_w != output_w)
      status = UDONDNN_STATUS_NOT_SUPPORTED;
    else
      status = udondnnCreateTensorDescriptor(&tile_descriptor);
    for (offset = 0; status == UDONDNN_STATUS_SUCCESS && offset < words;) {
      int tile_n, tile_c, tile_h, tile_w;
      int tile_words = udonatenPointwiseTileShape(words - offset, &tile_n,
                                                  &tile_c, &tile_h, &tile_w);
      status = udondnnSetTensor4dDescriptor(
          tile_descriptor, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT, tile_n,
          tile_c, tile_h, tile_w);
      if (status == UDONDNN_STATUS_SUCCESS)
        status = udondnnOpTensor(handle, descriptor, &one, tile_descriptor,
                                 self + offset, &other_alpha, tile_descriptor,
                                 other + offset, &zero, tile_descriptor,
                                 output + offset);
      offset += (size_t)tile_words;
    }
  } else if (status == UDONDNN_STATUS_SUCCESS)
    status = udondnnOpTensor(handle, descriptor, &one, self_descriptor, self,
                             &other_alpha, other_descriptor, other, &zero,
                             output_descriptor, output);
  {
    if (tile_descriptor) {
      udonatenStatus_t tile_destroy_status =
          udondnnDestroyTensorDescriptor(tile_descriptor);
      if (status == UDONDNN_STATUS_SUCCESS)
        status = tile_destroy_status;
    }
    udonatenStatus_t destroy_status =
        udondnnDestroyOpTensorDescriptor(descriptor);
    return status == UDONDNN_STATUS_SUCCESS ? destroy_status : status;
  }
}

udonatenStatus_t
udonatenAdd(udonatenHandle_t handle, udonatenTensorDescriptor_t self_descriptor,
            const float *self, udonatenTensorDescriptor_t other_descriptor,
            const float *other, float alpha,
            udonatenTensorDescriptor_t output_descriptor, float *output) {
  return udonatenBinary(handle, UDONDNN_OP_TENSOR_ADD, self_descriptor, self,
                        other_descriptor, other, alpha, output_descriptor,
                        output);
}

udonatenStatus_t
udonatenSub(udonatenHandle_t handle, udonatenTensorDescriptor_t self_descriptor,
            const float *self, udonatenTensorDescriptor_t other_descriptor,
            const float *other, float alpha,
            udonatenTensorDescriptor_t output_descriptor, float *output) {
  return udonatenBinary(handle, UDONDNN_OP_TENSOR_ADD, self_descriptor, self,
                        other_descriptor, other, -alpha, output_descriptor,
                        output);
}

udonatenStatus_t
udonatenMul(udonatenHandle_t handle, udonatenTensorDescriptor_t self_descriptor,
            const float *self, udonatenTensorDescriptor_t other_descriptor,
            const float *other, udonatenTensorDescriptor_t output_descriptor,
            float *output) {
  return udonatenBinary(handle, UDONDNN_OP_TENSOR_MUL, self_descriptor, self,
                        other_descriptor, other, 1.0f, output_descriptor,
                        output);
}

static udonatenStatus_t udonatenActivation(
    udonatenHandle_t handle, udondnnActivationMode_t mode,
    udonatenTensorDescriptor_t input_descriptor, const float *input,
    udonatenTensorDescriptor_t output_descriptor, float *output) {
  const float one = 1.0f, zero = 0.0f;
  udondnnActivationDescriptor_t descriptor = 0;
  udondnnTensorDescriptor_t tile_descriptor = 0;
  udondnnDataType_t input_type, output_type;
  int input_n, input_c, input_h, input_w;
  int output_n, output_c, output_h, output_w;
  int n_stride, c_stride, h_stride, w_stride;
  size_t words = 0, offset;
  udonatenStatus_t status = udondnnCreateActivationDescriptor(&descriptor);
  if (status != UDONDNN_STATUS_SUCCESS)
    return status;
  status = udondnnSetActivationDescriptor(descriptor, mode,
                                          UDONDNN_PROPAGATE_NAN, 0.0f);
  if (status == UDONDNN_STATUS_SUCCESS)
    status = udondnnGetTensor4dDescriptor(
        input_descriptor, &input_type, &input_n, &input_c, &input_h, &input_w,
        &n_stride, &c_stride, &h_stride, &w_stride);
  if (status == UDONDNN_STATUS_SUCCESS)
    status = udondnnGetTensor4dDescriptor(
        output_descriptor, &output_type, &output_n, &output_c, &output_h,
        &output_w, &n_stride, &c_stride, &h_stride, &w_stride);
  if (status == UDONDNN_STATUS_SUCCESS)
    words = (size_t)output_n * (size_t)output_c * (size_t)output_h *
            (size_t)output_w;
  if (status == UDONDNN_STATUS_SUCCESS &&
      words > UDONATEN_POINTWISE_TILE_WORDS) {
    if (input_type != UDONDNN_DATA_FLOAT || output_type != UDONDNN_DATA_FLOAT ||
        input_n != output_n || input_c != output_c || input_h != output_h ||
        input_w != output_w)
      status = UDONDNN_STATUS_BAD_PARAM;
    else
      status = udondnnCreateTensorDescriptor(&tile_descriptor);
    for (offset = 0; status == UDONDNN_STATUS_SUCCESS && offset < words;) {
      int tile_n, tile_c, tile_h, tile_w;
      int tile_words = udonatenPointwiseTileShape(words - offset, &tile_n,
                                                  &tile_c, &tile_h, &tile_w);
      status = udondnnSetTensor4dDescriptor(
          tile_descriptor, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT, tile_n,
          tile_c, tile_h, tile_w);
      if (status == UDONDNN_STATUS_SUCCESS)
        status = udondnnActivationForward(
            handle, descriptor, &one, tile_descriptor, input + offset, &zero,
            tile_descriptor, output + offset);
      offset += (size_t)tile_words;
    }
  } else if (status == UDONDNN_STATUS_SUCCESS)
    status =
        udondnnActivationForward(handle, descriptor, &one, input_descriptor,
                                 input, &zero, output_descriptor, output);
  {
    if (tile_descriptor) {
      udonatenStatus_t tile_destroy_status =
          udondnnDestroyTensorDescriptor(tile_descriptor);
      if (status == UDONDNN_STATUS_SUCCESS)
        status = tile_destroy_status;
    }
    udonatenStatus_t destroy_status =
        udondnnDestroyActivationDescriptor(descriptor);
    return status == UDONDNN_STATUS_SUCCESS ? destroy_status : status;
  }
}

udonatenStatus_t udonatenSigmoid(udonatenHandle_t handle,
                                 udonatenTensorDescriptor_t input_descriptor,
                                 const float *input,
                                 udonatenTensorDescriptor_t output_descriptor,
                                 float *output) {
  return udonatenActivation(handle, UDONDNN_ACTIVATION_SIGMOID,
                            input_descriptor, input, output_descriptor, output);
}

udonatenStatus_t udonatenRelu(udonatenHandle_t handle,
                              udonatenTensorDescriptor_t input_descriptor,
                              const float *input,
                              udonatenTensorDescriptor_t output_descriptor,
                              float *output) {
  return udonatenActivation(handle, UDONDNN_ACTIVATION_RELU, input_descriptor,
                            input, output_descriptor, output);
}

udonatenStatus_t udonatenTanh(udonatenHandle_t handle,
                              udonatenTensorDescriptor_t input_descriptor,
                              const float *input,
                              udonatenTensorDescriptor_t output_descriptor,
                              float *output) {
  return udonatenActivation(handle, UDONDNN_ACTIVATION_TANH, input_descriptor,
                            input, output_descriptor, output);
}

udonatenStatus_t udonatenEmbedding(udonatenHandle_t handle,
                                   udonatenTensorDescriptor_t weight_descriptor,
                                   const float *weight, const int32_t *indices,
                                   int indices_count, int64_t padding_idx,
                                   int scale_grad_by_freq, int sparse,
                                   udonatenTensorDescriptor_t output_descriptor,
                                   float *output) {
  if (padding_idx != -1 || scale_grad_by_freq || sparse)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  return udonatenBackendEmbedding(handle, weight_descriptor, weight, indices,
                                  indices_count, output_descriptor, output);
}

udonatenStatus_t
udonatenRmsNorm(udonatenHandle_t handle,
                udonatenTensorDescriptor_t input_descriptor, const float *input,
                const int64_t *normalized_shape, size_t normalized_dimensions,
                udonatenTensorDescriptor_t weight_descriptor,
                const float *weight, const float *epsilon,
                udonatenTensorDescriptor_t output_descriptor, float *output) {
  float effective_epsilon = epsilon ? *epsilon : FLT_EPSILON;
  udondnnDataType_t input_type, weight_type, output_type;
  int input_n, input_c, input_h, input_w;
  int weight_n, weight_c, weight_h, weight_w;
  int output_n, output_c, output_h, output_w;
  int n_stride, c_stride, h_stride, w_stride;
  int input_shape[4], normalized_words, words;
  int64_t normalized_product = 1;
  udonatenStatus_t status;
  size_t index;
  if (!normalized_shape || normalized_dimensions == 0 ||
      normalized_dimensions > 4 || !input || !output ||
      !isfinite(effective_epsilon) || effective_epsilon < 0.0f)
    return UDONDNN_STATUS_BAD_PARAM;
  status = udondnnGetTensor4dDescriptor(input_descriptor, &input_type, &input_n,
                                        &input_c, &input_h, &input_w, &n_stride,
                                        &c_stride, &h_stride, &w_stride);
  if (status == UDONDNN_STATUS_SUCCESS)
    status = udondnnGetTensor4dDescriptor(
        output_descriptor, &output_type, &output_n, &output_c, &output_h,
        &output_w, &n_stride, &c_stride, &h_stride, &w_stride);
  if (status != UDONDNN_STATUS_SUCCESS)
    return status;
  if (input_type != UDONDNN_DATA_FLOAT || output_type != UDONDNN_DATA_FLOAT ||
      input_n != output_n || input_c != output_c || input_h != output_h ||
      input_w != output_w)
    return UDONDNN_STATUS_BAD_PARAM;
  input_shape[0] = input_n;
  input_shape[1] = input_c;
  input_shape[2] = input_h;
  input_shape[3] = input_w;
  for (index = 0; index < normalized_dimensions; ++index) {
    if (normalized_shape[index] <= 0 ||
        normalized_shape[index] !=
            input_shape[4 - normalized_dimensions + index])
      return UDONDNN_STATUS_BAD_PARAM;
    normalized_product *= normalized_shape[index];
  }
  if (normalized_product > 4096)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  normalized_words = (int)normalized_product;
  words = input_n * input_c * input_h * input_w;
  if (weight) {
    status = udondnnGetTensor4dDescriptor(
        weight_descriptor, &weight_type, &weight_n, &weight_c, &weight_h,
        &weight_w, &n_stride, &c_stride, &h_stride, &w_stride);
    if (status != UDONDNN_STATUS_SUCCESS)
      return status;
    if (weight_type != UDONDNN_DATA_FLOAT ||
        weight_n * weight_c * weight_h * weight_w != normalized_words)
      return UDONDNN_STATUS_BAD_PARAM;
  }
  if (words > UDONATEN_ARENA_WORDS || normalized_words > 4096)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  return udonatenBackendRmsNorm(handle, words, normalized_words, input, weight,
                                effective_epsilon, output);
}

udonatenStatus_t udonatenTile(udonatenHandle_t handle,
                              udonatenTensorDescriptor_t input_descriptor,
                              const float *input, const int64_t *dims,
                              size_t dimensions,
                              udonatenTensorDescriptor_t output_descriptor,
                              float *output) {
  udondnnDataType_t input_type, output_type;
  int input_n, input_c, input_h, input_w;
  int output_n, output_c, output_h, output_w;
  int n_stride, c_stride, h_stride, w_stride;
  int repeats[4] = {1, 1, 1, 1};
  size_t index;
  if (dimensions > 4)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  if (dimensions && !dims)
    return UDONDNN_STATUS_BAD_PARAM;
  for (index = 0; index < dimensions; ++index) {
    int64_t repeat = dims[index];
    if (repeat < 0)
      return UDONDNN_STATUS_BAD_PARAM;
    if (repeat == 0 || repeat > 0x7fffffff)
      return UDONDNN_STATUS_NOT_SUPPORTED;
    repeats[4 - dimensions + index] = (int)repeat;
  }
  if (udondnnGetTensor4dDescriptor(input_descriptor, &input_type, &input_n,
                                   &input_c, &input_h, &input_w, &n_stride,
                                   &c_stride, &h_stride,
                                   &w_stride) != UDONDNN_STATUS_SUCCESS ||
      udondnnGetTensor4dDescriptor(output_descriptor, &output_type, &output_n,
                                   &output_c, &output_h, &output_w, &n_stride,
                                   &c_stride, &h_stride,
                                   &w_stride) != UDONDNN_STATUS_SUCCESS ||
      input_type != UDONDNN_DATA_FLOAT || output_type != UDONDNN_DATA_FLOAT ||
      (int64_t)output_n != (int64_t)input_n * repeats[0] ||
      (int64_t)output_c != (int64_t)input_c * repeats[1] ||
      (int64_t)output_h != (int64_t)input_h * repeats[2] ||
      (int64_t)output_w != (int64_t)input_w * repeats[3] || !input || !output)
    return UDONDNN_STATUS_BAD_PARAM;
  return udonatenBackendTile(handle, input_n, input_c, input_h, input_w, input,
                             repeats[0], repeats[1], repeats[2], repeats[3],
                             output_n * output_c * output_h * output_w, output);
}

udonatenStatus_t udonatenWeightInt4packMm(
    udonatenHandle_t handle, udonatenTensorDescriptor_t input_descriptor,
    const float *input, udonatenTensorDescriptor_t packed_weight_descriptor,
    const uint8_t *packed_weight, int64_t group_size,
    udonatenTensorDescriptor_t scales_and_zeros_descriptor,
    const float *scales_and_zeros, udonatenTensorDescriptor_t output_descriptor,
    float *output) {
  udondnnDataType_t data_type;
  int input_n, input_c, input_h, input_w;
  int packed_n, packed_c, packed_h, packed_w;
  int qparams_n, qparams_c, qparams_h, qparams_w;
  int output_n, output_c, output_h, output_w;
  int n_stride, c_stride, h_stride, w_stride;
#define GET_TENSOR(descriptor, n, c, h, w)                                     \
  do {                                                                         \
    udonatenStatus_t status = udondnnGetTensor4dDescriptor(                    \
        descriptor, &data_type, &n, &c, &h, &w, &n_stride, &c_stride,          \
        &h_stride, &w_stride);                                                 \
    if (status != UDONDNN_STATUS_SUCCESS)                                      \
      return status;                                                           \
  } while (0)
  GET_TENSOR(input_descriptor, input_n, input_c, input_h, input_w);
  GET_TENSOR(packed_weight_descriptor, packed_n, packed_c, packed_h, packed_w);
  GET_TENSOR(scales_and_zeros_descriptor, qparams_n, qparams_c, qparams_h,
             qparams_w);
  GET_TENSOR(output_descriptor, output_n, output_c, output_h, output_w);
#undef GET_TENSOR
  if (group_size != 32 && group_size != 64 && group_size != 128 &&
      group_size != 256)
    return UDONDNN_STATUS_BAD_PARAM;
  if (group_size != 32)
    return UDONDNN_STATUS_NOT_SUPPORTED;
  if (data_type != UDONDNN_DATA_FLOAT || input_n <= 0 || input_c % group_size ||
      input_h != 1 || input_w != 1 || packed_h != 1 || packed_w != 1 ||
      packed_c != input_c / 2 || qparams_n != input_c / group_size ||
      qparams_c != packed_n || qparams_h != 2 || qparams_w != 1 ||
      output_n != input_n || output_c != packed_n || output_h != 1 ||
      output_w != 1)
    return UDONDNN_STATUS_BAD_PARAM;
  return udonatenBackendWeightInt4packMm(handle, input_n, packed_n, input_c,
                                         input, packed_weight, (int)group_size,
                                         scales_and_zeros, output);
}
