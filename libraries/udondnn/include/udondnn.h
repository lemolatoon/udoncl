#ifndef UDONDNN_H
#define UDONDNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t udondnnHandle_t;
typedef uint32_t udondnnTensorDescriptor_t;
typedef uint32_t udondnnFilterDescriptor_t;
typedef uint32_t udondnnConvolutionDescriptor_t;
typedef uint32_t udondnnActivationDescriptor_t;
typedef uint32_t udondnnPoolingDescriptor_t;

typedef enum udondnnStatus_t {
  UDONDNN_STATUS_SUCCESS = 0,
  UDONDNN_STATUS_NOT_INITIALIZED = 1,
  UDONDNN_STATUS_ALLOC_FAILED = 2,
  UDONDNN_STATUS_BAD_PARAM = 3,
  UDONDNN_STATUS_NOT_SUPPORTED = 9,
  UDONDNN_STATUS_EXECUTION_FAILED = 13
} udondnnStatus_t;

typedef enum udondnnDataType_t { UDONDNN_DATA_FLOAT = 0 } udondnnDataType_t;
typedef enum udondnnTensorFormat_t {
  UDONDNN_TENSOR_NCHW = 0
} udondnnTensorFormat_t;
typedef enum udondnnConvolutionMode_t {
  UDONDNN_CONVOLUTION = 0,
  UDONDNN_CROSS_CORRELATION = 1
} udondnnConvolutionMode_t;
typedef enum udondnnConvolutionFwdAlgo_t {
  UDONDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
} udondnnConvolutionFwdAlgo_t;
typedef enum udondnnNanPropagation_t {
  UDONDNN_NOT_PROPAGATE_NAN = 0,
  UDONDNN_PROPAGATE_NAN = 1
} udondnnNanPropagation_t;
typedef enum udondnnActivationMode_t {
  UDONDNN_ACTIVATION_SIGMOID = 0,
  UDONDNN_ACTIVATION_RELU = 1,
  UDONDNN_ACTIVATION_TANH = 2,
  UDONDNN_ACTIVATION_CLIPPED_RELU = 3,
  UDONDNN_ACTIVATION_ELU = 4,
  UDONDNN_ACTIVATION_IDENTITY = 5
} udondnnActivationMode_t;
typedef enum udondnnPoolingMode_t {
  UDONDNN_POOLING_MAX = 0,
  UDONDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
  UDONDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
  UDONDNN_POOLING_MAX_DETERMINISTIC = 3
} udondnnPoolingMode_t;
typedef enum udondnnSoftmaxAlgorithm_t {
  UDONDNN_SOFTMAX_FAST = 0,
  UDONDNN_SOFTMAX_ACCURATE = 1,
  UDONDNN_SOFTMAX_LOG = 2
} udondnnSoftmaxAlgorithm_t;
typedef enum udondnnSoftmaxMode_t {
  UDONDNN_SOFTMAX_MODE_INSTANCE = 0,
  UDONDNN_SOFTMAX_MODE_CHANNEL = 1
} udondnnSoftmaxMode_t;
typedef enum udondnnBatchNormMode_t {
  UDONDNN_BATCHNORM_PER_ACTIVATION = 0,
  UDONDNN_BATCHNORM_SPATIAL = 1,
  UDONDNN_BATCHNORM_SPATIAL_PERSISTENT = 2
} udondnnBatchNormMode_t;

#define UDONDNN_BN_MIN_EPSILON 1.0e-5f

enum {
  UDONDNN_CONV_MAX_TENSOR_WORDS = 512,
  UDONDNN_CONV_MAX_FILTER_WORDS = 512,
  UDONDNN_CONV_MAX_OUTPUT_WORDS = 512
};

udondnnStatus_t udondnnCreate(udondnnHandle_t *handle);
udondnnStatus_t udondnnDestroy(udondnnHandle_t handle);

udondnnStatus_t
udondnnCreateTensorDescriptor(udondnnTensorDescriptor_t *descriptor);
udondnnStatus_t udondnnSetTensor4dDescriptor(
    udondnnTensorDescriptor_t descriptor, udondnnTensorFormat_t format,
    udondnnDataType_t data_type, int n, int c, int h, int w);
udondnnStatus_t
udondnnDestroyTensorDescriptor(udondnnTensorDescriptor_t descriptor);

udondnnStatus_t
udondnnCreateFilterDescriptor(udondnnFilterDescriptor_t *descriptor);
udondnnStatus_t udondnnSetFilter4dDescriptor(
    udondnnFilterDescriptor_t descriptor, udondnnDataType_t data_type,
    udondnnTensorFormat_t format, int k, int c, int h, int w);
udondnnStatus_t
udondnnDestroyFilterDescriptor(udondnnFilterDescriptor_t descriptor);

udondnnStatus_t
udondnnCreateConvolutionDescriptor(udondnnConvolutionDescriptor_t *descriptor);
udondnnStatus_t udondnnSetConvolution2dDescriptor(
    udondnnConvolutionDescriptor_t descriptor, int pad_h, int pad_w,
    int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w,
    udondnnConvolutionMode_t mode, udondnnDataType_t compute_type);
udondnnStatus_t
udondnnDestroyConvolutionDescriptor(udondnnConvolutionDescriptor_t descriptor);

udondnnStatus_t udondnnGetConvolution2dForwardOutputDim(
    udondnnConvolutionDescriptor_t convolution, udondnnTensorDescriptor_t input,
    udondnnFilterDescriptor_t filter, int *n, int *c, int *h, int *w);
udondnnStatus_t udondnnGetConvolutionForwardWorkspaceSize(
    udondnnHandle_t handle, udondnnTensorDescriptor_t input,
    udondnnFilterDescriptor_t filter,
    udondnnConvolutionDescriptor_t convolution,
    udondnnTensorDescriptor_t output, udondnnConvolutionFwdAlgo_t algorithm,
    size_t *size_in_bytes);
udondnnStatus_t udondnnConvolutionForward(
    udondnnHandle_t handle, const float *alpha,
    udondnnTensorDescriptor_t input_descriptor, const float *input,
    udondnnFilterDescriptor_t filter_descriptor, const float *filter,
    udondnnConvolutionDescriptor_t convolution_descriptor,
    udondnnConvolutionFwdAlgo_t algorithm, void *workspace,
    size_t workspace_size_in_bytes, const float *beta,
    udondnnTensorDescriptor_t output_descriptor, float *output);

udondnnStatus_t
udondnnCreateActivationDescriptor(udondnnActivationDescriptor_t *descriptor);
udondnnStatus_t udondnnSetActivationDescriptor(
    udondnnActivationDescriptor_t descriptor, udondnnActivationMode_t mode,
    udondnnNanPropagation_t nan_propagation, float coefficient);
udondnnStatus_t
udondnnDestroyActivationDescriptor(udondnnActivationDescriptor_t descriptor);
udondnnStatus_t udondnnActivationForward(
    udondnnHandle_t handle, udondnnActivationDescriptor_t activation_descriptor,
    const float *alpha, udondnnTensorDescriptor_t input_descriptor,
    const float *input, const float *beta,
    udondnnTensorDescriptor_t output_descriptor, float *output);

udondnnStatus_t
udondnnCreatePoolingDescriptor(udondnnPoolingDescriptor_t *descriptor);
udondnnStatus_t udondnnSetPooling2dDescriptor(
    udondnnPoolingDescriptor_t descriptor, udondnnPoolingMode_t mode,
    udondnnNanPropagation_t nan_propagation, int window_height,
    int window_width, int vertical_padding, int horizontal_padding,
    int vertical_stride, int horizontal_stride);
udondnnStatus_t
udondnnDestroyPoolingDescriptor(udondnnPoolingDescriptor_t descriptor);
udondnnStatus_t udondnnGetPooling2dForwardOutputDim(
    udondnnPoolingDescriptor_t pooling_descriptor,
    udondnnTensorDescriptor_t input_descriptor, int *n, int *c, int *h, int *w);
udondnnStatus_t udondnnPoolingForward(
    udondnnHandle_t handle, udondnnPoolingDescriptor_t pooling_descriptor,
    const float *alpha, udondnnTensorDescriptor_t input_descriptor,
    const float *input, const float *beta,
    udondnnTensorDescriptor_t output_descriptor, float *output);

udondnnStatus_t udondnnSoftmaxForward(
    udondnnHandle_t handle, udondnnSoftmaxAlgorithm_t algorithm,
    udondnnSoftmaxMode_t mode, const float *alpha,
    udondnnTensorDescriptor_t input_descriptor, const float *input,
    const float *beta, udondnnTensorDescriptor_t output_descriptor,
    float *output);

udondnnStatus_t
udondnnDeriveBNTensorDescriptor(udondnnTensorDescriptor_t derived_descriptor,
                                udondnnTensorDescriptor_t input_descriptor,
                                udondnnBatchNormMode_t mode);
udondnnStatus_t udondnnBatchNormalizationForwardInference(
    udondnnHandle_t handle, udondnnBatchNormMode_t mode, const float *alpha,
    const float *beta, udondnnTensorDescriptor_t input_descriptor,
    const float *input, udondnnTensorDescriptor_t output_descriptor,
    float *output, udondnnTensorDescriptor_t parameter_descriptor,
    const float *scale, const float *bias, const float *estimated_mean,
    const float *estimated_variance, float epsilon);

#ifdef __cplusplus
}
#endif

#endif
