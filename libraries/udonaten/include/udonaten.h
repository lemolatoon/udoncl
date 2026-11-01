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

/* ATen: add.Tensor(self, other, *, alpha=1). */
udonatenStatus_t
udonatenAdd(udonatenHandle_t handle, udonatenTensorDescriptor_t self_descriptor,
            const float *self, udonatenTensorDescriptor_t other_descriptor,
            const float *other, float alpha,
            udonatenTensorDescriptor_t output_descriptor, float *output);

/* ATen: mul.Tensor(self, other). */
udonatenStatus_t
udonatenMul(udonatenHandle_t handle, udonatenTensorDescriptor_t self_descriptor,
            const float *self, udonatenTensorDescriptor_t other_descriptor,
            const float *other, udonatenTensorDescriptor_t output_descriptor,
            float *output);

/* ATen: sigmoid(self). */
udonatenStatus_t udonatenSigmoid(udonatenHandle_t handle,
                                 udonatenTensorDescriptor_t input_descriptor,
                                 const float *input,
                                 udonatenTensorDescriptor_t output_descriptor,
                                 float *output);

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

/* ATen: tile(input, dims). The dimensions are aligned to the trailing input
 * dimensions; omitted leading repeats are one, as in ATen tile_symint. */
udonatenStatus_t udonatenTile(udonatenHandle_t handle,
                              udonatenTensorDescriptor_t input_descriptor,
                              const float *input, const int64_t *dims,
                              size_t dimensions,
                              udonatenTensorDescriptor_t output_descriptor,
                              float *output);

/* ATen: _weight_int4pack_mm(input, packed_weight, group_size,
 *                           scales_and_zeros).
 *
 * input is a contiguous logical [M,K] matrix and output is [M,N].
 * packed_weight is the backend-specific signed-INT4 packed layout.
 * scales_and_zeros has logical shape [K / group_size, N, 2] and applies
 * dequantized = signed_int4 * scale + zero. Group sizes 32, 64, 128, and 256
 * belong to the ATen contract; unsupported valid sizes return NOT_SUPPORTED. */
udonatenStatus_t udonatenWeightInt4packMm(
    udonatenHandle_t handle, udonatenTensorDescriptor_t input_descriptor,
    const float *input, udonatenTensorDescriptor_t packed_weight_descriptor,
    const uint8_t *packed_weight, int64_t group_size,
    udonatenTensorDescriptor_t scales_and_zeros_descriptor,
    const float *scales_and_zeros, udonatenTensorDescriptor_t output_descriptor,
    float *output);

#ifdef __cplusplus
}
#endif

#endif
