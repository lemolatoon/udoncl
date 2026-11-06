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
/* Converts a contiguous tensor between UdonCL scalar storage types. Boolean
 * ONNX tensors use UDONCL_DATA_UINT8 with zero and one values. */
udononnxStatus_t udononnxCast(udononnxOperatorHandle_t handle,
                              udonclDataType_t input_type, const void *input,
                              udonclDataType_t output_type, void *output,
                              size_t element_count);
/* Implements ONNX right-aligned multidirectional broadcasting for one
 * contiguous tensor. */
udononnxStatus_t udononnxExpand(udononnxOperatorHandle_t handle,
                                udonclDataType_t data_type, const void *input,
                                const int64_t *input_shape, int32_t input_rank,
                                void *output, const int64_t *output_shape,
                                int32_t output_rank);
udononnxStatus_t udononnxTranspose(udononnxOperatorHandle_t handle,
                                   udonclDataType_t data_type,
                                   const void *input,
                                   const int64_t *input_shape, int32_t rank,
                                   const int64_t *permutation, void *output);
/* Copies one input of an ONNX Concat into its axis range in output. Calling
 * this once per input with cumulative axis_offset implements the full op. */
udononnxStatus_t udononnxConcat(udononnxOperatorHandle_t handle,
                                udonclDataType_t data_type, const void *input,
                                const int64_t *input_shape, int32_t rank,
                                int32_t axis, int64_t axis_offset, void *output,
                                const int64_t *output_shape);
udononnxStatus_t udononnxCos(udononnxOperatorHandle_t handle,
                             const float *input, size_t element_count,
                             float *output);
udononnxStatus_t udononnxSin(udononnxOperatorHandle_t handle,
                             const float *input, size_t element_count,
                             float *output);
udononnxStatus_t udononnxNeg(udononnxOperatorHandle_t handle,
                             const float *input, size_t element_count,
                             float *output);
udononnxStatus_t udononnxSlice(udononnxOperatorHandle_t handle,
                               udonclDataType_t data_type, const void *input,
                               const int64_t *input_shape, int32_t rank,
                               const int64_t *starts, const int64_t *ends,
                               const int64_t *axes, const int64_t *steps,
                               int32_t slice_rank, void *output,
                               const int64_t *output_shape);
udononnxStatus_t udononnxEqual(udononnxOperatorHandle_t handle,
                               udonclDataType_t data_type, const void *left,
                               const int64_t *left_shape, int32_t left_rank,
                               const void *right, const int64_t *right_shape,
                               int32_t right_rank, uint8_t *output,
                               const int64_t *output_shape,
                               int32_t output_rank);
udononnxStatus_t
udononnxWhere(udononnxOperatorHandle_t handle, udonclDataType_t data_type,
              const uint8_t *condition, const int64_t *condition_shape,
              int32_t condition_rank, const void *when_true,
              const int64_t *true_shape, int32_t true_rank,
              const void *when_false, const int64_t *false_shape,
              int32_t false_rank, void *output, const int64_t *output_shape,
              int32_t output_rank);
udononnxStatus_t udononnxAdd(udononnxOperatorHandle_t handle,
                             udonclDataType_t data_type, const void *left,
                             const int64_t *left_shape, int32_t left_rank,
                             const void *right, const int64_t *right_shape,
                             int32_t right_rank, void *output,
                             const int64_t *output_shape, int32_t output_rank);
udononnxStatus_t
udononnxScatterND(udononnxOperatorHandle_t handle, udonclDataType_t data_type,
                  const void *data, const int64_t *data_shape,
                  int32_t data_rank, udonclDataType_t indices_type,
                  const void *indices, const int64_t *indices_shape,
                  int32_t indices_rank, const void *updates, void *output);
#ifdef __cplusplus
}
#endif

#endif
