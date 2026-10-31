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

#ifdef __cplusplus
}
#endif

#endif
