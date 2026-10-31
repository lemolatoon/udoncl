#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "udondnn.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Bindings.h"

extern "C" size_t udondnn_cpu_gid;
size_t udondnn_cpu_gid;

namespace {
struct Case {
  int n, inputChannels, inputHeight, inputWidth;
  int outputChannels, filterHeight, filterWidth;
  int padHeight, padWidth, strideHeight, strideWidth;
  int dilationHeight, dilationWidth;
  udondnnConvolutionMode_t mode;
  float alpha, beta;
};

float value(int index, int salt) {
  int bits = ((index + 5) * (salt * 11 + 7)) % 31;
  return (float(bits) - 15.0f) * 0.0625f;
}

void require(udondnnStatus_t actual, udondnnStatus_t expected,
             const char *operation) {
  if (actual != expected)
    throw std::runtime_error(std::string(operation) + " returned " +
                             std::to_string(actual) + ", expected " +
                             std::to_string(expected));
}

std::array<int, 4> outputShape(const Case &test) {
  int effectiveHeight = test.dilationHeight * (test.filterHeight - 1) + 1;
  int effectiveWidth = test.dilationWidth * (test.filterWidth - 1) + 1;
  return {test.n, test.outputChannels,
          1 + (test.inputHeight + 2 * test.padHeight - effectiveHeight) /
                  test.strideHeight,
          1 + (test.inputWidth + 2 * test.padWidth - effectiveWidth) /
                  test.strideWidth};
}

void reference(const Case &test, const std::vector<float> &input,
               const std::vector<float> &filter, std::vector<float> &output) {
  auto shape = outputShape(test);
  for (int batch = 0; batch < shape[0]; ++batch)
    for (int outputChannel = 0; outputChannel < shape[1]; ++outputChannel)
      for (int outputY = 0; outputY < shape[2]; ++outputY)
        for (int outputX = 0; outputX < shape[3]; ++outputX) {
          float sum = 0.0f;
          for (int inputChannel = 0; inputChannel < test.inputChannels;
               ++inputChannel)
            for (int kernelY = 0; kernelY < test.filterHeight; ++kernelY)
              for (int kernelX = 0; kernelX < test.filterWidth; ++kernelX) {
                int inputY = outputY * test.strideHeight +
                             kernelY * test.dilationHeight - test.padHeight;
                int inputX = outputX * test.strideWidth +
                             kernelX * test.dilationWidth - test.padWidth;
                if (inputY < 0 || inputX < 0 || inputY >= test.inputHeight ||
                    inputX >= test.inputWidth)
                  continue;
                int filterY = test.mode == UDONDNN_CROSS_CORRELATION
                                  ? kernelY
                                  : test.filterHeight - 1 - kernelY;
                int filterX = test.mode == UDONDNN_CROSS_CORRELATION
                                  ? kernelX
                                  : test.filterWidth - 1 - kernelX;
                int inputIndex = ((batch * test.inputChannels + inputChannel) *
                                      test.inputHeight +
                                  inputY) *
                                     test.inputWidth +
                                 inputX;
                int filterIndex =
                    ((outputChannel * test.inputChannels + inputChannel) *
                         test.filterHeight +
                     filterY) *
                        test.filterWidth +
                    filterX;
                sum += input[inputIndex] * filter[filterIndex];
              }
          int outputIndex =
              ((batch * shape[1] + outputChannel) * shape[2] + outputY) *
                  shape[3] +
              outputX;
          output[outputIndex] =
              test.alpha * sum + test.beta * output[outputIndex];
        }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: udondnn-device convolution-fragment vertex\n";
    return 2;
  }
  try {
    native_opencl::Session session(deviceSource(), manifestBindings(argv[1]),
                                   argv[2]);
    udondnnHandle_t handle = 0;
    require(udondnnCreate(&handle), UDONDNN_STATUS_SUCCESS, "udondnnCreate");
    const std::array<Case, 5> cases{{
        {1, 1, 4, 5, 2, 2, 3, 0, 0, 1, 1, 1, 1, UDONDNN_CROSS_CORRELATION, 1.0f,
         0.0f},
        {1, 2, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, UDONDNN_CROSS_CORRELATION,
         0.75f, -0.25f},
        {2, 2, 5, 6, 2, 2, 2, 0, 0, 2, 1, 1, 1, UDONDNN_CROSS_CORRELATION,
         -1.0f, 0.5f},
        {1, 1, 7, 7, 1, 3, 2, 0, 0, 1, 1, 2, 2, UDONDNN_CROSS_CORRELATION,
         1.25f, 0.125f},
        {1, 2, 5, 5, 2, 3, 3, 1, 1, 1, 1, 1, 1, UDONDNN_CONVOLUTION, 0.5f,
         -0.75f},
    }};
    float maximumError = 0.0f;
    int outputs = 0;
    for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
      const Case &test = cases[caseIndex];
      auto shape = outputShape(test);
      int inputWords =
          test.n * test.inputChannels * test.inputHeight * test.inputWidth;
      int filterWords = test.outputChannels * test.inputChannels *
                        test.filterHeight * test.filterWidth;
      int outputWords = shape[0] * shape[1] * shape[2] * shape[3];
      std::vector<float> input(inputWords), filter(filterWords),
          actual(outputWords), expected;
      for (int index = 0; index < inputWords; ++index)
        input[index] = value(index, int(caseIndex) + 1);
      for (int index = 0; index < filterWords; ++index)
        filter[index] = value(index, int(caseIndex) + 7);
      for (int index = 0; index < outputWords; ++index)
        actual[index] = value(index, int(caseIndex) + 13);
      expected = actual;
      reference(test, input, filter, expected);

      udondnnTensorDescriptor_t inputDescriptor = 0, outputDescriptor = 0;
      udondnnFilterDescriptor_t filterDescriptor = 0;
      udondnnConvolutionDescriptor_t convolutionDescriptor = 0;
      require(udondnnCreateTensorDescriptor(&inputDescriptor),
              UDONDNN_STATUS_SUCCESS, "create input descriptor");
      require(udondnnCreateTensorDescriptor(&outputDescriptor),
              UDONDNN_STATUS_SUCCESS, "create output descriptor");
      require(udondnnCreateFilterDescriptor(&filterDescriptor),
              UDONDNN_STATUS_SUCCESS, "create filter descriptor");
      require(udondnnCreateConvolutionDescriptor(&convolutionDescriptor),
              UDONDNN_STATUS_SUCCESS, "create convolution descriptor");
      require(udondnnSetTensor4dDescriptor(inputDescriptor, UDONDNN_TENSOR_NCHW,
                                           UDONDNN_DATA_FLOAT, test.n,
                                           test.inputChannels, test.inputHeight,
                                           test.inputWidth),
              UDONDNN_STATUS_SUCCESS, "set input descriptor");
      require(udondnnSetFilter4dDescriptor(
                  filterDescriptor, UDONDNN_DATA_FLOAT, UDONDNN_TENSOR_NCHW,
                  test.outputChannels, test.inputChannels, test.filterHeight,
                  test.filterWidth),
              UDONDNN_STATUS_SUCCESS, "set filter descriptor");
      require(udondnnSetConvolution2dDescriptor(
                  convolutionDescriptor, test.padHeight, test.padWidth,
                  test.strideHeight, test.strideWidth, test.dilationHeight,
                  test.dilationWidth, test.mode, UDONDNN_DATA_FLOAT),
              UDONDNN_STATUS_SUCCESS, "set convolution descriptor");
      int actualN = 0, actualC = 0, actualH = 0, actualW = 0;
      require(udondnnGetConvolution2dForwardOutputDim(
                  convolutionDescriptor, inputDescriptor, filterDescriptor,
                  &actualN, &actualC, &actualH, &actualW),
              UDONDNN_STATUS_SUCCESS, "get output dimensions");
      if (std::array<int, 4>{actualN, actualC, actualH, actualW} != shape)
        throw std::runtime_error("incorrect convolution output dimensions");
      require(udondnnSetTensor4dDescriptor(
                  outputDescriptor, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT,
                  shape[0], shape[1], shape[2], shape[3]),
              UDONDNN_STATUS_SUCCESS, "set output descriptor");
      size_t workspaceSize = 99;
      require(udondnnGetConvolutionForwardWorkspaceSize(
                  handle, inputDescriptor, filterDescriptor,
                  convolutionDescriptor, outputDescriptor,
                  UDONDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspaceSize),
              UDONDNN_STATUS_SUCCESS, "get workspace size");
      if (workspaceSize != 0)
        throw std::runtime_error("implicit GEMM unexpectedly needs workspace");
      require(udondnnConvolutionForward(
                  handle, &test.alpha, inputDescriptor, input.data(),
                  filterDescriptor, filter.data(), convolutionDescriptor,
                  UDONDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0,
                  &test.beta, outputDescriptor, actual.data()),
              UDONDNN_STATUS_SUCCESS, "udondnnConvolutionForward");
      for (int index = 0; index < outputWords; ++index) {
        maximumError =
            std::max(maximumError, std::abs(actual[index] - expected[index]));
        ++outputs;
      }
      require(udondnnDestroyTensorDescriptor(inputDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy input descriptor");
      require(udondnnDestroyTensorDescriptor(outputDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy output descriptor");
      require(udondnnDestroyFilterDescriptor(filterDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy filter descriptor");
      require(udondnnDestroyConvolutionDescriptor(convolutionDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy convolution descriptor");
    }
    if (maximumError > 0.0005f)
      throw std::runtime_error(
          "UdonDNN convolution CPU/Device results diverged");

    udondnnTensorDescriptor_t stale = 0, replacement = 0;
    require(udondnnCreateTensorDescriptor(&stale), UDONDNN_STATUS_SUCCESS,
            "create stale descriptor");
    require(udondnnDestroyTensorDescriptor(stale), UDONDNN_STATUS_SUCCESS,
            "destroy stale descriptor");
    require(udondnnCreateTensorDescriptor(&replacement), UDONDNN_STATUS_SUCCESS,
            "create replacement descriptor");
    require(udondnnSetTensor4dDescriptor(stale, UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 1, 1, 1),
            UDONDNN_STATUS_BAD_PARAM, "reject stale descriptor");
    require(udondnnSetTensor4dDescriptor(replacement, UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 9, 9, 9),
            UDONDNN_STATUS_BAD_PARAM, "reject oversized tensor");
    require(udondnnDestroyTensorDescriptor(replacement), UDONDNN_STATUS_SUCCESS,
            "destroy replacement descriptor");
    require(udondnnDestroy(handle), UDONDNN_STATUS_SUCCESS, "udondnnDestroy");
    if (udoncl_cl_live_objects())
      throw std::runtime_error("OpenCL object leak");
    std::cout << "udondnn_convolution_outputs=" << outputs
              << " cases=" << cases.size()
              << " max_cpu_gpu_error=" << maximumError
              << " renderer=" << session.deviceName()
              << " draws=" << session.drawCount() << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
