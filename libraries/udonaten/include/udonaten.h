#ifndef UDONATEN_H
#define UDONATEN_H

#include "udondnn.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef udondnnHandle_t udonatenHandle_t;
typedef udondnnTensorDescriptor_t udonatenTensorDescriptor_t;
typedef udondnnStatus_t udonatenStatus_t;

/* ATen: embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse). */
udonatenStatus_t udonatenEmbedding(udonatenHandle_t handle,
                                   udonatenTensorDescriptor_t weight_descriptor,
                                   const float *weight, const int32_t *indices,
                                   int indices_count, int64_t padding_idx,
                                   int scale_grad_by_freq, int sparse,
                                   udonatenTensorDescriptor_t output_descriptor,
                                   float *output);

/* ATen: rms_norm(input, normalized_shape, weight=None, eps=None). */
udonatenStatus_t
udonatenRmsNorm(udonatenHandle_t handle,
                udonatenTensorDescriptor_t input_descriptor, const float *input,
                const int64_t *normalized_shape, size_t normalized_dimensions,
                udonatenTensorDescriptor_t weight_descriptor,
                const float *weight, const float *epsilon,
                udonatenTensorDescriptor_t output_descriptor, float *output);

#ifdef __cplusplus
}
#endif

#endif
