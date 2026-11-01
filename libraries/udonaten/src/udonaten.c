#include "udonaten.h"
#include <float.h>

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
