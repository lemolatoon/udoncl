#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "udondnn.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Bindings.h"

extern "C" size_t udondnn_cpu_gid;
size_t udondnn_cpu_gid;
extern "C" size_t udondnn_activation_cpu_gid;
size_t udondnn_activation_cpu_gid;
extern "C" size_t udondnn_pooling_cpu_gid;
size_t udondnn_pooling_cpu_gid;
extern "C" size_t udondnn_softmax_cpu_gid;
size_t udondnn_softmax_cpu_gid;

namespace {
struct Case {
  int n, inputChannels, inputHeight, inputWidth;
  int outputChannels, filterHeight, filterWidth;
  int padHeight, padWidth, strideHeight, strideWidth;
  int dilationHeight, dilationWidth;
  udondnnConvolutionMode_t mode;
  float alpha, beta;
};

struct ActivationCase {
  udondnnActivationMode_t mode;
  udondnnNanPropagation_t nanPropagation;
  float coefficient, alpha, beta;
  bool injectNan;
};

struct PoolingCase {
  int n, channels, inputHeight, inputWidth;
  int windowHeight, windowWidth;
  int padHeight, padWidth;
  int strideHeight, strideWidth;
  udondnnPoolingMode_t mode;
  udondnnNanPropagation_t nanPropagation;
  float alpha, beta;
  bool injectNan;
};

struct SoftmaxCase {
  int n, channels, height, width;
  udondnnSoftmaxAlgorithm_t algorithm;
  udondnnSoftmaxMode_t mode;
  float alpha, beta;
  bool injectNan;
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

float activationReference(const ActivationCase &test, float input) {
  if (std::isnan(input)) {
    if (test.nanPropagation == UDONDNN_PROPAGATE_NAN)
      return input;
    input = 0.0f;
  }
  switch (test.mode) {
  case UDONDNN_ACTIVATION_SIGMOID:
    return 1.0f / (1.0f + std::exp(-input));
  case UDONDNN_ACTIVATION_RELU:
    return input > 0.0f ? input : 0.0f;
  case UDONDNN_ACTIVATION_TANH:
    return 2.0f / (1.0f + std::exp(-2.0f * input)) - 1.0f;
  case UDONDNN_ACTIVATION_CLIPPED_RELU:
    return std::min(test.coefficient, input > 0.0f ? input : 0.0f);
  case UDONDNN_ACTIVATION_ELU:
    return input > 0.0f ? input : test.coefficient * (std::exp(input) - 1.0f);
  case UDONDNN_ACTIVATION_IDENTITY:
    return input;
  }
  throw std::runtime_error("unknown activation mode");
}

std::array<int, 4> poolingOutputShape(const PoolingCase &test) {
  return {test.n, test.channels,
          1 + (test.inputHeight + 2 * test.padHeight - test.windowHeight) /
                  test.strideHeight,
          1 + (test.inputWidth + 2 * test.padWidth - test.windowWidth) /
                  test.strideWidth};
}

void poolingReference(const PoolingCase &test, const std::vector<float> &input,
                      std::vector<float> &output) {
  auto shape = poolingOutputShape(test);
  for (int batch = 0; batch < shape[0]; ++batch)
    for (int channel = 0; channel < shape[1]; ++channel)
      for (int outputY = 0; outputY < shape[2]; ++outputY)
        for (int outputX = 0; outputX < shape[3]; ++outputX) {
          float sum = 0.0f;
          float maximum = -std::numeric_limits<float>::max();
          float nanValue = 0.0f;
          int valid = 0, numeric = 0;
          bool hasNan = false;
          for (int kernelY = 0; kernelY < test.windowHeight; ++kernelY)
            for (int kernelX = 0; kernelX < test.windowWidth; ++kernelX) {
              int inputY =
                  outputY * test.strideHeight + kernelY - test.padHeight;
              int inputX = outputX * test.strideWidth + kernelX - test.padWidth;
              if (inputY < 0 || inputX < 0 || inputY >= test.inputHeight ||
                  inputX >= test.inputWidth)
                continue;
              int inputIndex =
                  ((batch * test.channels + channel) * test.inputHeight +
                   inputY) *
                      test.inputWidth +
                  inputX;
              float sample = input[inputIndex];
              ++valid;
              if (std::isnan(sample)) {
                if (test.nanPropagation == UDONDNN_PROPAGATE_NAN) {
                  hasNan = true;
                  nanValue = sample;
                }
                sample = 0.0f;
              } else {
                ++numeric;
              }
              sum += sample;
              maximum = std::max(maximum, sample);
            }
          float pooled = maximum;
          if (test.mode == UDONDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
            pooled = sum / float(test.windowHeight * test.windowWidth);
          else if (test.mode == UDONDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
            pooled = sum / float(valid);
          else if (!numeric)
            pooled = 0.0f;
          if (hasNan)
            pooled = nanValue;
          int outputIndex =
              ((batch * shape[1] + channel) * shape[2] + outputY) * shape[3] +
              outputX;
          output[outputIndex] =
              test.alpha * pooled + test.beta * output[outputIndex];
        }
}

void softmaxReference(const SoftmaxCase &test, const std::vector<float> &input,
                      std::vector<float> &output) {
  int sampleWords = test.channels * test.height * test.width;
  int words = test.n * sampleWords;
  for (int element = 0; element < words; ++element) {
    int x = element % test.width;
    int y = (element / test.width) % test.height;
    int batch = element / sampleWords;
    float maximum = -std::numeric_limits<float>::max();
    if (test.mode == UDONDNN_SOFTMAX_MODE_INSTANCE) {
      for (int candidate = 0; candidate < sampleWords; ++candidate)
        maximum = std::max(maximum, input[batch * sampleWords + candidate]);
    } else {
      for (int channel = 0; channel < test.channels; ++channel) {
        int candidate =
            ((batch * test.channels + channel) * test.height + y) * test.width +
            x;
        maximum = std::max(maximum, input[candidate]);
      }
    }
    float denominator = 0.0f;
    if (test.mode == UDONDNN_SOFTMAX_MODE_INSTANCE) {
      for (int candidate = 0; candidate < sampleWords; ++candidate)
        denominator +=
            std::exp(input[batch * sampleWords + candidate] - maximum);
    } else {
      for (int channel = 0; channel < test.channels; ++channel) {
        int candidate =
            ((batch * test.channels + channel) * test.height + y) * test.width +
            x;
        denominator += std::exp(input[candidate] - maximum);
      }
    }
    float normalized = std::exp(input[element] - maximum) / denominator;
    if (test.algorithm == UDONDNN_SOFTMAX_LOG)
      normalized = input[element] - maximum - std::log(denominator);
    output[element] = test.alpha * normalized + test.beta * output[element];
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: udondnn-device convolution-fragment "
                 "activation-fragment pooling-fragment softmax-fragment "
                 "vertex\n";
    return 2;
  }
  try {
    native_opencl::Session session(
        deviceSource(), manifestBindings(argv[1], argv[2], argv[3], argv[4]),
        argv[5]);
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

    const std::array<ActivationCase, 8> activationCases{{
        {UDONDNN_ACTIVATION_SIGMOID, UDONDNN_NOT_PROPAGATE_NAN, 0.0f, 1.0f,
         0.0f, false},
        {UDONDNN_ACTIVATION_RELU, UDONDNN_NOT_PROPAGATE_NAN, 0.0f, 0.75f,
         -0.25f, false},
        {UDONDNN_ACTIVATION_TANH, UDONDNN_NOT_PROPAGATE_NAN, 0.0f, -1.25f, 0.5f,
         false},
        {UDONDNN_ACTIVATION_CLIPPED_RELU, UDONDNN_NOT_PROPAGATE_NAN, 0.8f, 1.0f,
         0.125f, false},
        {UDONDNN_ACTIVATION_ELU, UDONDNN_NOT_PROPAGATE_NAN, 1.2f, 0.5f, -0.5f,
         false},
        {UDONDNN_ACTIVATION_IDENTITY, UDONDNN_NOT_PROPAGATE_NAN, 0.0f, -0.75f,
         1.0f, false},
        {UDONDNN_ACTIVATION_RELU, UDONDNN_NOT_PROPAGATE_NAN, 0.0f, 1.0f, 0.0f,
         true},
        {UDONDNN_ACTIVATION_RELU, UDONDNN_PROPAGATE_NAN, 0.0f, 1.0f, 0.0f,
         true},
    }};
    constexpr int activationWords = 64;
    udondnnTensorDescriptor_t activationInputDescriptor = 0;
    udondnnTensorDescriptor_t activationOutputDescriptor = 0;
    require(udondnnCreateTensorDescriptor(&activationInputDescriptor),
            UDONDNN_STATUS_SUCCESS, "create activation input descriptor");
    require(udondnnCreateTensorDescriptor(&activationOutputDescriptor),
            UDONDNN_STATUS_SUCCESS, "create activation output descriptor");
    require(udondnnSetTensor4dDescriptor(activationInputDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 2, 4, 8),
            UDONDNN_STATUS_SUCCESS, "set activation input descriptor");
    require(udondnnSetTensor4dDescriptor(activationOutputDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 2, 4, 8),
            UDONDNN_STATUS_SUCCESS, "set activation output descriptor");
    float activationMaximumError = 0.0f;
    int activationOutputs = 0;
    for (size_t caseIndex = 0; caseIndex < activationCases.size();
         ++caseIndex) {
      const ActivationCase &test = activationCases[caseIndex];
      std::vector<float> input(activationWords), actual(activationWords),
          expected(activationWords);
      for (int index = 0; index < activationWords; ++index) {
        input[index] = value(index, int(caseIndex) + 19) * 4.0f;
        actual[index] = value(index, int(caseIndex) + 31);
      }
      if (test.injectNan)
        input[7] = std::numeric_limits<float>::quiet_NaN();
      for (int index = 0; index < activationWords; ++index)
        expected[index] = test.alpha * activationReference(test, input[index]) +
                          test.beta * actual[index];
      udondnnActivationDescriptor_t activationDescriptor = 0;
      require(udondnnCreateActivationDescriptor(&activationDescriptor),
              UDONDNN_STATUS_SUCCESS, "create activation descriptor");
      require(udondnnSetActivationDescriptor(activationDescriptor, test.mode,
                                             test.nanPropagation,
                                             test.coefficient),
              UDONDNN_STATUS_SUCCESS, "set activation descriptor");
      require(udondnnActivationForward(
                  handle, activationDescriptor, &test.alpha,
                  activationInputDescriptor, input.data(), &test.beta,
                  activationOutputDescriptor, actual.data()),
              UDONDNN_STATUS_SUCCESS, "udondnnActivationForward");
      for (int index = 0; index < activationWords; ++index) {
        if (std::isnan(expected[index]) && std::isnan(actual[index])) {
          ++activationOutputs;
          continue;
        }
        activationMaximumError = std::max(
            activationMaximumError, std::abs(actual[index] - expected[index]));
        ++activationOutputs;
      }
      require(udondnnDestroyActivationDescriptor(activationDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy activation descriptor");
    }
    if (activationMaximumError > 0.0005f)
      throw std::runtime_error(
          "UdonDNN activation CPU/Device results diverged");
    require(udondnnDestroyTensorDescriptor(activationInputDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy activation input descriptor");
    require(udondnnDestroyTensorDescriptor(activationOutputDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy activation output descriptor");

    const std::array<PoolingCase, 6> poolingCases{{
        {1, 2, 4, 5, 2, 3, 0, 0, 2, 2, UDONDNN_POOLING_MAX,
         UDONDNN_NOT_PROPAGATE_NAN, 1.0f, 0.0f, false},
        {1, 1, 5, 5, 3, 3, 1, 1, 2, 2, UDONDNN_POOLING_MAX_DETERMINISTIC,
         UDONDNN_NOT_PROPAGATE_NAN, 0.75f, -0.25f, false},
        {1, 2, 4, 4, 3, 3, 1, 1, 2, 2,
         UDONDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
         UDONDNN_NOT_PROPAGATE_NAN, -1.0f, 0.5f, false},
        {1, 2, 4, 4, 3, 3, 1, 1, 2, 2,
         UDONDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
         UDONDNN_NOT_PROPAGATE_NAN, 1.25f, 0.125f, false},
        {1, 1, 4, 4, 2, 2, 0, 0, 2, 2, UDONDNN_POOLING_MAX,
         UDONDNN_PROPAGATE_NAN, 1.0f, 0.0f, true},
        {1, 1, 4, 4, 2, 2, 0, 0, 2, 2, UDONDNN_POOLING_MAX,
         UDONDNN_NOT_PROPAGATE_NAN, 1.0f, 0.0f, true},
    }};
    float poolingMaximumError = 0.0f;
    int poolingOutputs = 0;
    for (size_t caseIndex = 0; caseIndex < poolingCases.size(); ++caseIndex) {
      const PoolingCase &test = poolingCases[caseIndex];
      auto shape = poolingOutputShape(test);
      int inputWords =
          test.n * test.channels * test.inputHeight * test.inputWidth;
      int outputWords = shape[0] * shape[1] * shape[2] * shape[3];
      std::vector<float> input(inputWords), actual(outputWords), expected;
      for (int index = 0; index < inputWords; ++index)
        input[index] = value(index, int(caseIndex) + 43) * 2.0f;
      for (int index = 0; index < outputWords; ++index)
        actual[index] = value(index, int(caseIndex) + 53);
      if (test.injectNan)
        input[7] = std::numeric_limits<float>::quiet_NaN();
      expected = actual;
      poolingReference(test, input, expected);

      udondnnTensorDescriptor_t inputDescriptor = 0, outputDescriptor = 0;
      udondnnPoolingDescriptor_t poolingDescriptor = 0;
      require(udondnnCreateTensorDescriptor(&inputDescriptor),
              UDONDNN_STATUS_SUCCESS, "create pooling input descriptor");
      require(udondnnCreateTensorDescriptor(&outputDescriptor),
              UDONDNN_STATUS_SUCCESS, "create pooling output descriptor");
      require(udondnnCreatePoolingDescriptor(&poolingDescriptor),
              UDONDNN_STATUS_SUCCESS, "create pooling descriptor");
      require(udondnnSetTensor4dDescriptor(
                  inputDescriptor, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT,
                  test.n, test.channels, test.inputHeight, test.inputWidth),
              UDONDNN_STATUS_SUCCESS, "set pooling input descriptor");
      require(udondnnSetPooling2dDescriptor(
                  poolingDescriptor, test.mode, test.nanPropagation,
                  test.windowHeight, test.windowWidth, test.padHeight,
                  test.padWidth, test.strideHeight, test.strideWidth),
              UDONDNN_STATUS_SUCCESS, "set pooling descriptor");
      int actualN = 0, actualC = 0, actualH = 0, actualW = 0;
      require(udondnnGetPooling2dForwardOutputDim(poolingDescriptor,
                                                  inputDescriptor, &actualN,
                                                  &actualC, &actualH, &actualW),
              UDONDNN_STATUS_SUCCESS, "get pooling output dimensions");
      if (std::array<int, 4>{actualN, actualC, actualH, actualW} != shape)
        throw std::runtime_error("incorrect pooling output dimensions");
      require(udondnnSetTensor4dDescriptor(
                  outputDescriptor, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT,
                  shape[0], shape[1], shape[2], shape[3]),
              UDONDNN_STATUS_SUCCESS, "set pooling output descriptor");
      require(udondnnPoolingForward(handle, poolingDescriptor, &test.alpha,
                                    inputDescriptor, input.data(), &test.beta,
                                    outputDescriptor, actual.data()),
              UDONDNN_STATUS_SUCCESS, "udondnnPoolingForward");
      for (int index = 0; index < outputWords; ++index) {
        if (std::isnan(expected[index]) && std::isnan(actual[index])) {
          ++poolingOutputs;
          continue;
        }
        poolingMaximumError = std::max(
            poolingMaximumError, std::abs(actual[index] - expected[index]));
        ++poolingOutputs;
      }
      require(udondnnDestroyPoolingDescriptor(poolingDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy pooling descriptor");
      require(udondnnDestroyTensorDescriptor(inputDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy pooling input descriptor");
      require(udondnnDestroyTensorDescriptor(outputDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy pooling output descriptor");
    }
    if (poolingMaximumError > 0.0005f)
      throw std::runtime_error("UdonDNN pooling CPU/Device results diverged");

    const std::array<SoftmaxCase, 6> softmaxCases{{
        {1, 3, 2, 2, UDONDNN_SOFTMAX_FAST, UDONDNN_SOFTMAX_MODE_INSTANCE, 1.0f,
         0.0f, false},
        {2, 2, 2, 2, UDONDNN_SOFTMAX_ACCURATE, UDONDNN_SOFTMAX_MODE_INSTANCE,
         0.75f, -0.25f, false},
        {1, 2, 2, 3, UDONDNN_SOFTMAX_LOG, UDONDNN_SOFTMAX_MODE_INSTANCE, -1.0f,
         0.5f, false},
        {1, 4, 2, 2, UDONDNN_SOFTMAX_FAST, UDONDNN_SOFTMAX_MODE_CHANNEL, 1.25f,
         0.125f, false},
        {2, 3, 1, 2, UDONDNN_SOFTMAX_ACCURATE, UDONDNN_SOFTMAX_MODE_CHANNEL,
         0.5f, -0.5f, false},
        {1, 3, 2, 2, UDONDNN_SOFTMAX_LOG, UDONDNN_SOFTMAX_MODE_CHANNEL, -0.75f,
         1.0f, true},
    }};
    float softmaxMaximumError = 0.0f;
    int softmaxOutputs = 0;
    for (size_t caseIndex = 0; caseIndex < softmaxCases.size(); ++caseIndex) {
      const SoftmaxCase &test = softmaxCases[caseIndex];
      int words = test.n * test.channels * test.height * test.width;
      std::vector<float> input(words), actual(words), expected;
      for (int index = 0; index < words; ++index) {
        input[index] = value(index, int(caseIndex) + 71) * 8.0f;
        actual[index] = value(index, int(caseIndex) + 79);
      }
      if (test.injectNan)
        input[7] = std::numeric_limits<float>::quiet_NaN();
      expected = actual;
      softmaxReference(test, input, expected);

      udondnnTensorDescriptor_t inputDescriptor = 0, outputDescriptor = 0;
      require(udondnnCreateTensorDescriptor(&inputDescriptor),
              UDONDNN_STATUS_SUCCESS, "create softmax input descriptor");
      require(udondnnCreateTensorDescriptor(&outputDescriptor),
              UDONDNN_STATUS_SUCCESS, "create softmax output descriptor");
      require(udondnnSetTensor4dDescriptor(
                  inputDescriptor, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT,
                  test.n, test.channels, test.height, test.width),
              UDONDNN_STATUS_SUCCESS, "set softmax input descriptor");
      require(udondnnSetTensor4dDescriptor(
                  outputDescriptor, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT,
                  test.n, test.channels, test.height, test.width),
              UDONDNN_STATUS_SUCCESS, "set softmax output descriptor");
      require(udondnnSoftmaxForward(handle, test.algorithm, test.mode,
                                    &test.alpha, inputDescriptor, input.data(),
                                    &test.beta, outputDescriptor,
                                    actual.data()),
              UDONDNN_STATUS_SUCCESS, "udondnnSoftmaxForward");
      for (int index = 0; index < words; ++index) {
        if (std::isnan(expected[index]) && std::isnan(actual[index])) {
          ++softmaxOutputs;
          continue;
        }
        softmaxMaximumError = std::max(
            softmaxMaximumError, std::abs(actual[index] - expected[index]));
        ++softmaxOutputs;
      }
      require(udondnnDestroyTensorDescriptor(inputDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy softmax input descriptor");
      require(udondnnDestroyTensorDescriptor(outputDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy softmax output descriptor");
    }
    if (softmaxMaximumError > 0.0005f)
      throw std::runtime_error("UdonDNN softmax CPU/Device results diverged");

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
    udondnnActivationDescriptor_t staleActivation = 0;
    require(udondnnCreateActivationDescriptor(&staleActivation),
            UDONDNN_STATUS_SUCCESS, "create stale activation descriptor");
    require(udondnnDestroyActivationDescriptor(staleActivation),
            UDONDNN_STATUS_SUCCESS, "destroy stale activation descriptor");
    require(udondnnSetActivationDescriptor(staleActivation,
                                           UDONDNN_ACTIVATION_RELU,
                                           UDONDNN_NOT_PROPAGATE_NAN, 0.0f),
            UDONDNN_STATUS_BAD_PARAM, "reject stale activation descriptor");
    udondnnActivationDescriptor_t invalidActivation = 0;
    require(udondnnCreateActivationDescriptor(&invalidActivation),
            UDONDNN_STATUS_SUCCESS, "create invalid activation descriptor");
    require(udondnnSetActivationDescriptor(invalidActivation,
                                           UDONDNN_ACTIVATION_CLIPPED_RELU,
                                           UDONDNN_NOT_PROPAGATE_NAN, -1.0f),
            UDONDNN_STATUS_BAD_PARAM, "reject negative activation ceiling");
    require(udondnnSetActivationDescriptor(invalidActivation,
                                           udondnnActivationMode_t(99),
                                           UDONDNN_NOT_PROPAGATE_NAN, 0.0f),
            UDONDNN_STATUS_BAD_PARAM, "reject invalid activation mode");
    require(udondnnSetActivationDescriptor(
                invalidActivation, UDONDNN_ACTIVATION_ELU,
                UDONDNN_NOT_PROPAGATE_NAN,
                std::numeric_limits<float>::infinity()),
            UDONDNN_STATUS_BAD_PARAM, "reject infinite activation coefficient");
    require(udondnnDestroyActivationDescriptor(invalidActivation),
            UDONDNN_STATUS_SUCCESS, "destroy invalid activation descriptor");
    udondnnPoolingDescriptor_t stalePooling = 0;
    require(udondnnCreatePoolingDescriptor(&stalePooling),
            UDONDNN_STATUS_SUCCESS, "create stale pooling descriptor");
    require(udondnnDestroyPoolingDescriptor(stalePooling),
            UDONDNN_STATUS_SUCCESS, "destroy stale pooling descriptor");
    require(udondnnSetPooling2dDescriptor(stalePooling, UDONDNN_POOLING_MAX,
                                          UDONDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0,
                                          2, 2),
            UDONDNN_STATUS_BAD_PARAM, "reject stale pooling descriptor");
    udondnnPoolingDescriptor_t invalidPooling = 0;
    require(udondnnCreatePoolingDescriptor(&invalidPooling),
            UDONDNN_STATUS_SUCCESS, "create invalid pooling descriptor");
    require(udondnnSetPooling2dDescriptor(invalidPooling, UDONDNN_POOLING_MAX,
                                          UDONDNN_NOT_PROPAGATE_NAN, 6, 2, 0, 0,
                                          1, 1),
            UDONDNN_STATUS_BAD_PARAM, "reject oversized pooling window");
    require(udondnnSetPooling2dDescriptor(
                invalidPooling, udondnnPoolingMode_t(99),
                UDONDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 1, 1),
            UDONDNN_STATUS_BAD_PARAM, "reject invalid pooling mode");
    require(udondnnDestroyPoolingDescriptor(invalidPooling),
            UDONDNN_STATUS_SUCCESS, "destroy invalid pooling descriptor");
    udondnnTensorDescriptor_t softmaxInput = 0, softmaxOutput = 0;
    float softmaxInputValues[2] = {0.0f, 1.0f};
    float softmaxOutputValues[2] = {0.0f, 0.0f};
    const float one = 1.0f, zero = 0.0f;
    require(udondnnCreateTensorDescriptor(&softmaxInput),
            UDONDNN_STATUS_SUCCESS, "create invalid softmax input descriptor");
    require(udondnnCreateTensorDescriptor(&softmaxOutput),
            UDONDNN_STATUS_SUCCESS, "create invalid softmax output descriptor");
    require(udondnnSetTensor4dDescriptor(softmaxInput, UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 1, 1, 2),
            UDONDNN_STATUS_SUCCESS, "set invalid softmax input descriptor");
    require(udondnnSetTensor4dDescriptor(softmaxOutput, UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 1, 1, 2),
            UDONDNN_STATUS_SUCCESS, "set invalid softmax output descriptor");
    require(udondnnSoftmaxForward(handle, udondnnSoftmaxAlgorithm_t(99),
                                  UDONDNN_SOFTMAX_MODE_INSTANCE, &one,
                                  softmaxInput, softmaxInputValues, &zero,
                                  softmaxOutput, softmaxOutputValues),
            UDONDNN_STATUS_BAD_PARAM, "reject invalid softmax algorithm");
    require(udondnnSoftmaxForward(handle, UDONDNN_SOFTMAX_ACCURATE,
                                  udondnnSoftmaxMode_t(99), &one, softmaxInput,
                                  softmaxInputValues, &zero, softmaxOutput,
                                  softmaxOutputValues),
            UDONDNN_STATUS_BAD_PARAM, "reject invalid softmax mode");
    require(udondnnDestroyTensorDescriptor(softmaxInput),
            UDONDNN_STATUS_SUCCESS, "destroy invalid softmax input descriptor");
    require(udondnnDestroyTensorDescriptor(softmaxOutput),
            UDONDNN_STATUS_SUCCESS,
            "destroy invalid softmax output descriptor");
    require(udondnnDestroy(handle), UDONDNN_STATUS_SUCCESS, "udondnnDestroy");
    if (udoncl_cl_live_objects())
      throw std::runtime_error("OpenCL object leak");
    std::cout << "udondnn_convolution_outputs=" << outputs
              << " convolution_cases=" << cases.size()
              << " convolution_max_error=" << maximumError
              << " activation_outputs=" << activationOutputs
              << " activation_cases=" << activationCases.size()
              << " activation_max_error=" << activationMaximumError
              << " pooling_outputs=" << poolingOutputs
              << " pooling_cases=" << poolingCases.size()
              << " pooling_max_error=" << poolingMaximumError
              << " softmax_outputs=" << softmaxOutputs
              << " softmax_cases=" << softmaxCases.size()
              << " softmax_max_error=" << softmaxMaximumError
              << " renderer=" << session.deviceName()
              << " draws=" << session.drawCount() << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
