#include "udonaten.h"
#include <float.h>

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

static udonatenStatus_t
udonatenBinary(udonatenHandle_t handle, udondnnOpTensorOp_t operation,
               udonatenTensorDescriptor_t self_descriptor, const float *self,
               udonatenTensorDescriptor_t other_descriptor, const float *other,
               float other_alpha, udonatenTensorDescriptor_t output_descriptor,
               float *output) {
  const float one = 1.0f, zero = 0.0f;
  udondnnOpTensorDescriptor_t descriptor = 0;
  udonatenStatus_t status = udondnnCreateOpTensorDescriptor(&descriptor);
  if (status != UDONDNN_STATUS_SUCCESS)
    return status;
  status = udondnnSetOpTensorDescriptor(
      descriptor, operation, UDONDNN_DATA_FLOAT, UDONDNN_PROPAGATE_NAN);
  if (status == UDONDNN_STATUS_SUCCESS)
    status = udondnnOpTensor(handle, descriptor, &one, self_descriptor, self,
                             &other_alpha, other_descriptor, other, &zero,
                             output_descriptor, output);
  {
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
udonatenMul(udonatenHandle_t handle, udonatenTensorDescriptor_t self_descriptor,
            const float *self, udonatenTensorDescriptor_t other_descriptor,
            const float *other, udonatenTensorDescriptor_t output_descriptor,
            float *output) {
  return udonatenBinary(handle, UDONDNN_OP_TENSOR_MUL, self_descriptor, self,
                        other_descriptor, other, 1.0f, output_descriptor,
                        output);
}

udonatenStatus_t udonatenSigmoid(udonatenHandle_t handle,
                                 udonatenTensorDescriptor_t input_descriptor,
                                 const float *input,
                                 udonatenTensorDescriptor_t output_descriptor,
                                 float *output) {
  const float one = 1.0f, zero = 0.0f;
  udondnnActivationDescriptor_t descriptor = 0;
  udonatenStatus_t status = udondnnCreateActivationDescriptor(&descriptor);
  if (status != UDONDNN_STATUS_SUCCESS)
    return status;
  status = udondnnSetActivationDescriptor(
      descriptor, UDONDNN_ACTIVATION_SIGMOID, UDONDNN_PROPAGATE_NAN, 0.0f);
  if (status == UDONDNN_STATUS_SUCCESS)
    status =
        udondnnActivationForward(handle, descriptor, &one, input_descriptor,
                                 input, &zero, output_descriptor, output);
  {
    udonatenStatus_t destroy_status =
        udondnnDestroyActivationDescriptor(descriptor);
    return status == UDONDNN_STATUS_SUCCESS ? destroy_status : status;
  }
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
  return udondnnExtGatherForward(handle, weight_descriptor, weight, indices,
                                 indices_count, 0, output_descriptor, output);
}

udonatenStatus_t
udonatenRmsNorm(udonatenHandle_t handle,
                udonatenTensorDescriptor_t input_descriptor, const float *input,
                const int64_t *normalized_shape, size_t normalized_dimensions,
                udonatenTensorDescriptor_t weight_descriptor,
                const float *weight, const float *epsilon,
                udonatenTensorDescriptor_t output_descriptor, float *output) {
  const float alpha = 1.0f, beta = 0.0f;
  float effective_epsilon = epsilon ? *epsilon : FLT_EPSILON;
  size_t index;
  if (!normalized_shape || normalized_dimensions == 0)
    return UDONDNN_STATUS_BAD_PARAM;
  for (index = 0; index < normalized_dimensions; ++index)
    if (normalized_shape[index] <= 0)
      return UDONDNN_STATUS_BAD_PARAM;
  return udondnnExtRmsNormalizationForward(
      handle, &alpha, input_descriptor, input, weight_descriptor, weight,
      effective_epsilon, &beta, output_descriptor, output);
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
