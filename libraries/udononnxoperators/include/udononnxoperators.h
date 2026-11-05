#ifndef UDONONNXOPERATORS_H
#define UDONONNXOPERATORS_H

#include <udoncl/onnx_types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t udononnxOperatorHandle_t;

typedef enum udononnxZeroPointFormat {
  UDONONNX_ZERO_POINT_NONE = 0,
  UDONONNX_ZERO_POINT_PACKED = 1,
  UDONONNX_ZERO_POINT_UNPACKED = 2
} udononnxZeroPointFormat_t;

udononnxStatus_t udononnxCreateOperatorHandle(udononnxOperatorHandle_t *handle);
udononnxStatus_t udononnxDestroyOperatorHandle(udononnxOperatorHandle_t handle);
/* Implements com.microsoft::MatMulNBits. B and packed zero points use the
 * ONNX Runtime block bitstream without requantization. A, scales, bias, and Y
 * use data_type and may be float32, float16, or bfloat16. */
udononnxStatus_t udononnxMatMulNBits(
    udononnxOperatorHandle_t handle, udonclDataType_t data_type, const void *a,
    int64_t m, int64_t k, const uint8_t *b, size_t b_bytes, const void *scales,
    size_t scale_count, const uint8_t *zero_points, size_t zero_point_bytes,
    udononnxZeroPointFormat_t zero_point_format, const int32_t *g_idx,
    size_t g_idx_count, const void *bias, size_t bias_count, int64_t n,
    int block_size, int bits, void *y);
/* Implements ONNX Gather for all UdonCL scalar storage types. The output
 * shape is data_shape[:axis] + indices_shape + data_shape[axis + 1:]. */
udononnxStatus_t udononnxGather(udononnxOperatorHandle_t handle,
                                udonclDataType_t data_type, const void *data,
                                const int64_t *data_shape, int32_t data_rank,
                                udonclDataType_t indices_type,
                                const void *indices, int32_t indices_count,
                                int64_t axis, void *output);
#ifdef __cplusplus
}
#endif

#endif
