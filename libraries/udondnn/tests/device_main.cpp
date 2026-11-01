#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "udonaten.h"
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
extern "C" size_t udondnn_batchnorm_cpu_gid;
size_t udondnn_batchnorm_cpu_gid;
extern "C" size_t udondnn_op_tensor_cpu_gid;
size_t udondnn_op_tensor_cpu_gid;
extern "C" size_t udondnn_rms_norm_cpu_gid;
size_t udondnn_rms_norm_cpu_gid;
extern "C" size_t udondnn_gather_cpu_gid;
size_t udondnn_gather_cpu_gid;
extern "C" size_t udonaten_int4_cpu_gid;
size_t udonaten_int4_cpu_gid;
extern "C" size_t udonaten_tile_cpu_gid;
size_t udonaten_tile_cpu_gid;

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

struct BatchNormCase {
  int n, channels, height, width;
  udondnnBatchNormMode_t mode;
  float epsilon, alpha, beta;
  bool injectNan;
};

struct OpTensorCase {
  std::array<int, 4> aShape, bShape, cShape;
  udondnnOpTensorOp_t operation;
  udondnnNanPropagation_t nanPropagation;
  float alphaA, alphaB, beta;
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

void batchNormReference(const BatchNormCase &test,
                        const std::vector<float> &input,
                        const std::vector<float> &scale,
                        const std::vector<float> &bias,
                        const std::vector<float> &mean,
                        const std::vector<float> &variance,
                        std::vector<float> &output) {
  int sampleWords = test.channels * test.height * test.width;
  int words = test.n * sampleWords;
  for (int element = 0; element < words; ++element) {
    int parameter = element % sampleWords;
    if (test.mode != UDONDNN_BATCHNORM_PER_ACTIVATION)
      parameter = (element / (test.height * test.width)) % test.channels;
    float normalized = (input[element] - mean[parameter]) /
                       std::sqrt(variance[parameter] + test.epsilon);
    normalized = normalized * scale[parameter] + bias[parameter];
    output[element] = test.alpha * normalized + test.beta * output[element];
  }
}

int tensorWords(const std::array<int, 4> &shape) {
  return shape[0] * shape[1] * shape[2] * shape[3];
}

int broadcastIndex(int element, const std::array<int, 4> &source,
                   const std::array<int, 4> &output) {
  int x = element % output[3];
  int y = (element / output[3]) % output[2];
  int channel = (element / (output[3] * output[2])) % output[1];
  int batch = element / (output[3] * output[2] * output[1]);
  if (source[0] == 1)
    batch = 0;
  if (source[1] == 1)
    channel = 0;
  if (source[2] == 1)
    y = 0;
  if (source[3] == 1)
    x = 0;
  return ((batch * source[1] + channel) * source[2] + y) * source[3] + x;
}

void opTensorReference(const OpTensorCase &test, const std::vector<float> &a,
                       const std::vector<float> &b, std::vector<float> &c) {
  for (int element = 0; element < tensorWords(test.cShape); ++element) {
    float left =
        test.alphaA * a[broadcastIndex(element, test.aShape, test.cShape)];
    float right =
        test.alphaB * b[broadcastIndex(element, test.bShape, test.cShape)];
    float combined = left + right;
    if (test.operation == UDONDNN_OP_TENSOR_MUL)
      combined = left * right;
    else if (test.operation == UDONDNN_OP_TENSOR_MIN ||
             test.operation == UDONDNN_OP_TENSOR_MAX) {
      bool leftNan = std::isnan(left), rightNan = std::isnan(right);
      if (test.nanPropagation == UDONDNN_PROPAGATE_NAN && (leftNan || rightNan))
        combined = left + right;
      else if (leftNan)
        combined = rightNan ? 0.0f : right;
      else if (rightNan)
        combined = left;
      else if (test.operation == UDONDNN_OP_TENSOR_MIN)
        combined = std::min(left, right);
      else
        combined = std::max(left, right);
    }
    c[element] = combined + test.beta * c[element];
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 12) {
    std::cerr << "usage: udondnn-device convolution-fragment "
                 "activation-fragment pooling-fragment softmax-fragment "
                 "batchnorm-fragment op-tensor-fragment rmsnorm-fragment "
                 "gather-fragment int4-fragment tile-fragment vertex\n";
    return 2;
  }
  try {
    native_opencl::Session session(
        deviceSource(),
        manifestBindings(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6],
                         argv[7], argv[8], argv[9], argv[10]),
        argv[11]);
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

    const std::array<BatchNormCase, 4> batchNormCases{{
        {1, 2, 2, 3, UDONDNN_BATCHNORM_PER_ACTIVATION, 1.0e-5f, 1.0f, 0.0f,
         false},
        {2, 3, 2, 2, UDONDNN_BATCHNORM_SPATIAL, 1.0e-4f, 0.75f, -0.25f, false},
        {1, 4, 3, 2, UDONDNN_BATCHNORM_SPATIAL_PERSISTENT, 1.0e-3f, -1.25f,
         0.5f, false},
        {1, 2, 2, 2, UDONDNN_BATCHNORM_SPATIAL, 1.0e-5f, 0.5f, 0.125f, true},
    }};
    float batchNormMaximumError = 0.0f;
    int batchNormOutputs = 0;
    for (size_t caseIndex = 0; caseIndex < batchNormCases.size(); ++caseIndex) {
      const BatchNormCase &test = batchNormCases[caseIndex];
      int words = test.n * test.channels * test.height * test.width;
      int parameterWords = test.mode == UDONDNN_BATCHNORM_PER_ACTIVATION
                               ? test.channels * test.height * test.width
                               : test.channels;
      std::vector<float> input(words), actual(words), expected;
      std::vector<float> scale(parameterWords), bias(parameterWords),
          mean(parameterWords), variance(parameterWords);
      for (int index = 0; index < words; ++index) {
        input[index] = value(index, int(caseIndex) + 97) * 4.0f;
        actual[index] = value(index, int(caseIndex) + 103);
      }
      for (int index = 0; index < parameterWords; ++index) {
        scale[index] = 0.5f + 0.125f * float(index % 5);
        bias[index] = value(index, int(caseIndex) + 107);
        mean[index] = value(index, int(caseIndex) + 109);
        variance[index] = 0.25f + 0.0625f * float((index * 3) % 7);
      }
      if (test.injectNan)
        input[7] = std::numeric_limits<float>::quiet_NaN();
      expected = actual;
      batchNormReference(test, input, scale, bias, mean, variance, expected);

      udondnnTensorDescriptor_t inputDescriptor = 0, outputDescriptor = 0,
                                parameterDescriptor = 0;
      require(udondnnCreateTensorDescriptor(&inputDescriptor),
              UDONDNN_STATUS_SUCCESS, "create batchnorm input descriptor");
      require(udondnnCreateTensorDescriptor(&outputDescriptor),
              UDONDNN_STATUS_SUCCESS, "create batchnorm output descriptor");
      require(udondnnCreateTensorDescriptor(&parameterDescriptor),
              UDONDNN_STATUS_SUCCESS, "create batchnorm parameter descriptor");
      require(udondnnSetTensor4dDescriptor(
                  inputDescriptor, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT,
                  test.n, test.channels, test.height, test.width),
              UDONDNN_STATUS_SUCCESS, "set batchnorm input descriptor");
      require(udondnnSetTensor4dDescriptor(
                  outputDescriptor, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT,
                  test.n, test.channels, test.height, test.width),
              UDONDNN_STATUS_SUCCESS, "set batchnorm output descriptor");
      require(udondnnDeriveBNTensorDescriptor(parameterDescriptor,
                                              inputDescriptor, test.mode),
              UDONDNN_STATUS_SUCCESS, "derive batchnorm parameter descriptor");
      require(udondnnBatchNormalizationForwardInference(
                  handle, test.mode, &test.alpha, &test.beta, inputDescriptor,
                  input.data(), outputDescriptor, actual.data(),
                  parameterDescriptor, scale.data(), bias.data(), mean.data(),
                  variance.data(), test.epsilon),
              UDONDNN_STATUS_SUCCESS,
              "udondnnBatchNormalizationForwardInference");
      for (int index = 0; index < words; ++index) {
        if (std::isnan(expected[index]) && std::isnan(actual[index])) {
          ++batchNormOutputs;
          continue;
        }
        batchNormMaximumError = std::max(
            batchNormMaximumError, std::abs(actual[index] - expected[index]));
        ++batchNormOutputs;
      }
      require(udondnnDestroyTensorDescriptor(inputDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy batchnorm input descriptor");
      require(udondnnDestroyTensorDescriptor(outputDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy batchnorm output descriptor");
      require(udondnnDestroyTensorDescriptor(parameterDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy batchnorm parameter descriptor");
    }
    if (batchNormMaximumError > 0.0005f)
      throw std::runtime_error("UdonDNN batchnorm CPU/Device results diverged");

    const std::array<OpTensorCase, 4> opTensorCases{{
        OpTensorCase{{1, 2, 2, 3},
                     {1, 2, 1, 1},
                     {1, 2, 2, 3},
                     UDONDNN_OP_TENSOR_ADD,
                     UDONDNN_NOT_PROPAGATE_NAN,
                     1.0f,
                     0.75f,
                     -0.25f,
                     false},
        OpTensorCase{{2, 1, 2, 2},
                     {1, 3, 1, 2},
                     {2, 3, 2, 2},
                     UDONDNN_OP_TENSOR_MUL,
                     UDONDNN_NOT_PROPAGATE_NAN,
                     -1.0f,
                     0.5f,
                     0.25f,
                     false},
        OpTensorCase{{1, 2, 1, 3},
                     {1, 1, 2, 1},
                     {1, 2, 2, 3},
                     UDONDNN_OP_TENSOR_MIN,
                     UDONDNN_NOT_PROPAGATE_NAN,
                     1.25f,
                     -0.75f,
                     0.0f,
                     true},
        OpTensorCase{{1, 2, 1, 3},
                     {1, 1, 2, 1},
                     {1, 2, 2, 3},
                     UDONDNN_OP_TENSOR_MAX,
                     UDONDNN_PROPAGATE_NAN,
                     0.5f,
                     1.5f,
                     -0.5f,
                     true},
    }};
    float opTensorMaximumError = 0.0f;
    int opTensorOutputs = 0;
    for (size_t caseIndex = 0; caseIndex < opTensorCases.size(); ++caseIndex) {
      const OpTensorCase &test = opTensorCases[caseIndex];
      std::vector<float> a(tensorWords(test.aShape));
      std::vector<float> b(tensorWords(test.bShape));
      std::vector<float> actual(tensorWords(test.cShape)), expected;
      for (int index = 0; index < int(a.size()); ++index)
        a[index] = value(index, int(caseIndex) + 127) * 2.0f;
      for (int index = 0; index < int(b.size()); ++index)
        b[index] = value(index, int(caseIndex) + 131) * 3.0f;
      for (int index = 0; index < int(actual.size()); ++index)
        actual[index] = value(index, int(caseIndex) + 137);
      if (test.injectNan)
        a[1] = std::numeric_limits<float>::quiet_NaN();
      expected = actual;
      opTensorReference(test, a, b, expected);

      udondnnOpTensorDescriptor_t operationDescriptor = 0;
      udondnnTensorDescriptor_t aDescriptor = 0, bDescriptor = 0,
                                cDescriptor = 0;
      require(udondnnCreateOpTensorDescriptor(&operationDescriptor),
              UDONDNN_STATUS_SUCCESS, "create OpTensor descriptor");
      require(udondnnSetOpTensorDescriptor(operationDescriptor, test.operation,
                                           UDONDNN_DATA_FLOAT,
                                           test.nanPropagation),
              UDONDNN_STATUS_SUCCESS, "set OpTensor descriptor");
      require(udondnnCreateTensorDescriptor(&aDescriptor),
              UDONDNN_STATUS_SUCCESS, "create OpTensor A descriptor");
      require(udondnnCreateTensorDescriptor(&bDescriptor),
              UDONDNN_STATUS_SUCCESS, "create OpTensor B descriptor");
      require(udondnnCreateTensorDescriptor(&cDescriptor),
              UDONDNN_STATUS_SUCCESS, "create OpTensor C descriptor");
      require(udondnnSetTensor4dDescriptor(aDescriptor, UDONDNN_TENSOR_NCHW,
                                           UDONDNN_DATA_FLOAT, test.aShape[0],
                                           test.aShape[1], test.aShape[2],
                                           test.aShape[3]),
              UDONDNN_STATUS_SUCCESS, "set OpTensor A descriptor");
      require(udondnnSetTensor4dDescriptor(bDescriptor, UDONDNN_TENSOR_NCHW,
                                           UDONDNN_DATA_FLOAT, test.bShape[0],
                                           test.bShape[1], test.bShape[2],
                                           test.bShape[3]),
              UDONDNN_STATUS_SUCCESS, "set OpTensor B descriptor");
      require(udondnnSetTensor4dDescriptor(cDescriptor, UDONDNN_TENSOR_NCHW,
                                           UDONDNN_DATA_FLOAT, test.cShape[0],
                                           test.cShape[1], test.cShape[2],
                                           test.cShape[3]),
              UDONDNN_STATUS_SUCCESS, "set OpTensor C descriptor");
      require(udondnnOpTensor(handle, operationDescriptor, &test.alphaA,
                              aDescriptor, a.data(), &test.alphaB, bDescriptor,
                              b.data(), &test.beta, cDescriptor, actual.data()),
              UDONDNN_STATUS_SUCCESS, "udondnnOpTensor");
      for (int index = 0; index < int(actual.size()); ++index) {
        if (std::isnan(expected[index]) && std::isnan(actual[index])) {
          ++opTensorOutputs;
          continue;
        }
        opTensorMaximumError = std::max(
            opTensorMaximumError, std::abs(actual[index] - expected[index]));
        ++opTensorOutputs;
      }
      require(udondnnDestroyOpTensorDescriptor(operationDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy OpTensor descriptor");
      require(udondnnDestroyTensorDescriptor(aDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy OpTensor A descriptor");
      require(udondnnDestroyTensorDescriptor(bDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy OpTensor B descriptor");
      require(udondnnDestroyTensorDescriptor(cDescriptor),
              UDONDNN_STATUS_SUCCESS, "destroy OpTensor C descriptor");
    }
    if (opTensorMaximumError > 0.0005f)
      throw std::runtime_error("UdonDNN OpTensor CPU/Device results diverged");

    udondnnTensorDescriptor_t atenLeftDescriptor = 0, atenRightDescriptor = 0,
                              atenOutputDescriptor = 0;
    const std::array<float, 6> atenLeft{{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f}};
    const std::array<float, 3> atenRight{{4.0f, -2.0f, 0.5f}};
    std::array<float, 6> atenOutput{};
    require(udondnnCreateTensorDescriptor(&atenLeftDescriptor),
            UDONDNN_STATUS_SUCCESS, "create ATen left descriptor");
    require(udondnnCreateTensorDescriptor(&atenRightDescriptor),
            UDONDNN_STATUS_SUCCESS, "create ATen right descriptor");
    require(udondnnCreateTensorDescriptor(&atenOutputDescriptor),
            UDONDNN_STATUS_SUCCESS, "create ATen output descriptor");
    require(udondnnSetTensor4dDescriptor(atenLeftDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 2, 1, 3),
            UDONDNN_STATUS_SUCCESS, "set ATen left descriptor");
    require(udondnnSetTensor4dDescriptor(atenRightDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 1, 1, 3),
            UDONDNN_STATUS_SUCCESS, "set ATen right descriptor");
    require(udondnnSetTensor4dDescriptor(atenOutputDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 2, 1, 3),
            UDONDNN_STATUS_SUCCESS, "set ATen output descriptor");
    require(udonatenAdd(handle, atenLeftDescriptor, atenLeft.data(),
                        atenRightDescriptor, atenRight.data(), 0.25f,
                        atenOutputDescriptor, atenOutput.data()),
            UDONDNN_STATUS_SUCCESS, "udonatenAdd");
    for (size_t index = 0; index < atenOutput.size(); ++index)
      if (std::abs(atenOutput[index] -
                   (atenLeft[index] + 0.25f * atenRight[index % 3])) > 0.0005f)
        throw std::runtime_error("UdonATen add results diverged");
    require(udonatenSub(handle, atenLeftDescriptor, atenLeft.data(),
                        atenRightDescriptor, atenRight.data(), 0.25f,
                        atenOutputDescriptor, atenOutput.data()),
            UDONDNN_STATUS_SUCCESS, "udonatenSub");
    for (size_t index = 0; index < atenOutput.size(); ++index)
      if (std::abs(atenOutput[index] -
                   (atenLeft[index] - 0.25f * atenRight[index % 3])) > 0.0005f)
        throw std::runtime_error("UdonATen sub results diverged");
    require(udonatenMul(handle, atenLeftDescriptor, atenLeft.data(),
                        atenRightDescriptor, atenRight.data(),
                        atenOutputDescriptor, atenOutput.data()),
            UDONDNN_STATUS_SUCCESS, "udonatenMul");
    for (size_t index = 0; index < atenOutput.size(); ++index)
      if (std::abs(atenOutput[index] - atenLeft[index] * atenRight[index % 3]) >
          0.0005f)
        throw std::runtime_error("UdonATen mul results diverged");
    require(udonatenSigmoid(handle, atenLeftDescriptor, atenLeft.data(),
                            atenOutputDescriptor, atenOutput.data()),
            UDONDNN_STATUS_SUCCESS, "udonatenSigmoid");
    for (size_t index = 0; index < atenOutput.size(); ++index)
      if (std::abs(atenOutput[index] -
                   1.0f / (1.0f + std::exp(-atenLeft[index]))) > 0.0005f)
        throw std::runtime_error("UdonATen sigmoid results diverged");
    require(udonatenRelu(handle, atenLeftDescriptor, atenLeft.data(),
                         atenOutputDescriptor, atenOutput.data()),
            UDONDNN_STATUS_SUCCESS, "udonatenRelu");
    for (size_t index = 0; index < atenOutput.size(); ++index)
      if (std::abs(atenOutput[index] - std::max(atenLeft[index], 0.0f)) >
          0.0005f)
        throw std::runtime_error("UdonATen relu results diverged");
    require(udonatenTanh(handle, atenLeftDescriptor, atenLeft.data(),
                         atenOutputDescriptor, atenOutput.data()),
            UDONDNN_STATUS_SUCCESS, "udonatenTanh");
    for (size_t index = 0; index < atenOutput.size(); ++index)
      if (std::abs(atenOutput[index] - std::tanh(atenLeft[index])) > 0.0005f)
        throw std::runtime_error("UdonATen tanh results diverged");
    require(udondnnDestroyTensorDescriptor(atenLeftDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy ATen left descriptor");
    require(udondnnDestroyTensorDescriptor(atenRightDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy ATen right descriptor");
    require(udondnnDestroyTensorDescriptor(atenOutputDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy ATen output descriptor");

    udondnnTensorDescriptor_t rmsInputDescriptor = 0, rmsScaleDescriptor = 0,
                              rmsOutputDescriptor = 0;
    std::vector<float> rmsInput(16), rmsScale(8), rmsActual(16),
        rmsExpected(16);
    const float rmsAlpha = 1.0f, rmsBeta = 0.0f, rmsEpsilon = 1.0e-5f;
    float rmsMaximumError = 0.0f;
    for (int index = 0; index < 16; ++index) {
      rmsInput[index] = value(index, 149) * 3.0f;
      rmsActual[index] = value(index, 151);
      rmsExpected[index] = rmsActual[index];
    }
    for (int index = 0; index < 8; ++index)
      rmsScale[index] = 0.5f + 0.125f * float(index);
    for (int row = 0; row < 2; ++row) {
      float squareSum = 0.0f;
      for (int column = 0; column < 8; ++column) {
        float sample = rmsInput[row * 8 + column];
        squareSum += sample * sample;
      }
      float inverseRms = 1.0f / std::sqrt(squareSum / 8.0f + rmsEpsilon);
      for (int column = 0; column < 8; ++column) {
        int index = row * 8 + column;
        rmsExpected[index] =
            rmsAlpha * rmsInput[index] * inverseRms * rmsScale[column] +
            rmsBeta * rmsExpected[index];
      }
    }
    require(udondnnCreateTensorDescriptor(&rmsInputDescriptor),
            UDONDNN_STATUS_SUCCESS, "create RMS input descriptor");
    require(udondnnCreateTensorDescriptor(&rmsScaleDescriptor),
            UDONDNN_STATUS_SUCCESS, "create RMS scale descriptor");
    require(udondnnCreateTensorDescriptor(&rmsOutputDescriptor),
            UDONDNN_STATUS_SUCCESS, "create RMS output descriptor");
    require(udondnnSetTensor4dDescriptor(rmsInputDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 2, 1, 1, 8),
            UDONDNN_STATUS_SUCCESS, "set RMS input descriptor");
    require(udondnnSetTensor4dDescriptor(rmsScaleDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 1, 1, 8),
            UDONDNN_STATUS_SUCCESS, "set RMS scale descriptor");
    require(udondnnSetTensor4dDescriptor(rmsOutputDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 2, 1, 1, 8),
            UDONDNN_STATUS_SUCCESS, "set RMS output descriptor");
    const int64_t rmsNormalizedShape[1] = {8};
    require(udonatenRmsNorm(handle, rmsInputDescriptor, rmsInput.data(),
                            rmsNormalizedShape, 1, rmsScaleDescriptor,
                            rmsScale.data(), &rmsEpsilon, rmsOutputDescriptor,
                            rmsActual.data()),
            UDONDNN_STATUS_SUCCESS, "udonatenRmsNorm");
    for (int index = 0; index < 16; ++index)
      rmsMaximumError = std::max(
          rmsMaximumError, std::abs(rmsActual[index] - rmsExpected[index]));
    if (rmsMaximumError > 0.0005f)
      throw std::runtime_error("UdonDNN RMS normalization results diverged");
    require(udondnnDestroyTensorDescriptor(rmsInputDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy RMS input descriptor");
    require(udondnnDestroyTensorDescriptor(rmsScaleDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy RMS scale descriptor");
    require(udondnnDestroyTensorDescriptor(rmsOutputDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy RMS output descriptor");

    udondnnTensorDescriptor_t gatherDataDescriptor = 0,
                              gatherOutputDescriptor = 0;
    std::vector<float> gatherData(48), gatherActual(24), gatherExpected(24);
    const std::array<int32_t, 3> gatherIndices{{4, 1, -1}};
    for (int index = 0; index < int(gatherData.size()); ++index)
      gatherData[index] = value(index, 157);
    for (int selected = 0; selected < 3; ++selected) {
      int row = gatherIndices[selected] < 0 ? gatherIndices[selected] + 6
                                            : gatherIndices[selected];
      for (int column = 0; column < 8; ++column)
        gatherExpected[selected * 8 + column] = gatherData[row * 8 + column];
    }
    require(udondnnCreateTensorDescriptor(&gatherDataDescriptor),
            UDONDNN_STATUS_SUCCESS, "create Gather data descriptor");
    require(udondnnCreateTensorDescriptor(&gatherOutputDescriptor),
            UDONDNN_STATUS_SUCCESS, "create Gather output descriptor");
    require(udondnnSetTensor4dDescriptor(gatherDataDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 6, 8, 1, 1),
            UDONDNN_STATUS_SUCCESS, "set Gather data descriptor");
    require(udondnnSetTensor4dDescriptor(gatherOutputDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 3, 8, 1, 1),
            UDONDNN_STATUS_SUCCESS, "set Gather output descriptor");
    require(udonatenEmbedding(handle, gatherDataDescriptor, gatherData.data(),
                              gatherIndices.data(), int(gatherIndices.size()),
                              -1, 0, 0, gatherOutputDescriptor,
                              gatherActual.data()),
            UDONDNN_STATUS_SUCCESS, "udonatenEmbedding");
    require(
        udonatenEmbedding(handle, gatherDataDescriptor, gatherData.data(),
                          gatherIndices.data(), int(gatherIndices.size()), 0, 0,
                          0, gatherOutputDescriptor, gatherActual.data()),
        UDONDNN_STATUS_NOT_SUPPORTED, "reject unsupported embedding padding");
    float gatherMaximumError = 0.0f;
    for (int index = 0; index < int(gatherActual.size()); ++index)
      gatherMaximumError =
          std::max(gatherMaximumError,
                   std::abs(gatherActual[index] - gatherExpected[index]));
    if (gatherMaximumError > 0.0f)
      throw std::runtime_error("UdonDNN Gather results diverged");
    require(udonatenEmbedding(handle, gatherDataDescriptor, gatherData.data(),
                              std::array<int32_t, 1>{{6}}.data(), 1, -1, 0, 0,
                              gatherOutputDescriptor, gatherActual.data()),
            UDONDNN_STATUS_BAD_PARAM, "reject out-of-range embedding index");
    require(udondnnDestroyTensorDescriptor(gatherDataDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy Gather data descriptor");
    require(udondnnDestroyTensorDescriptor(gatherOutputDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy Gather output descriptor");

    udondnnTensorDescriptor_t tileInputDescriptor = 0, tileOutputDescriptor = 0;
    std::vector<float> tileInput(12), tileActual(72), tileExpected(72);
    const int64_t tileRepeats[2] = {2, 3};
    for (int index = 0; index < int(tileInput.size()); ++index)
      tileInput[index] = value(index, 181);
    for (int channel = 0; channel < 2; ++channel)
      for (int height = 0; height < 4; ++height)
        for (int width = 0; width < 9; ++width) {
          int outputIndex = (channel * 4 + height) * 9 + width;
          int inputIndex = (channel * 2 + height % 2) * 3 + width % 3;
          tileExpected[outputIndex] = tileInput[inputIndex];
        }
    require(udondnnCreateTensorDescriptor(&tileInputDescriptor),
            UDONDNN_STATUS_SUCCESS, "create Tile input descriptor");
    require(udondnnCreateTensorDescriptor(&tileOutputDescriptor),
            UDONDNN_STATUS_SUCCESS, "create Tile output descriptor");
    require(udondnnSetTensor4dDescriptor(tileInputDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 2, 2, 3),
            UDONDNN_STATUS_SUCCESS, "set Tile input descriptor");
    require(udondnnSetTensor4dDescriptor(tileOutputDescriptor,
                                         UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 2, 4, 9),
            UDONDNN_STATUS_SUCCESS, "set Tile output descriptor");
    require(udonatenTile(handle, tileInputDescriptor, tileInput.data(),
                         tileRepeats, 2, tileOutputDescriptor,
                         tileActual.data()),
            UDONDNN_STATUS_SUCCESS, "udonatenTile");
    float tileMaximumError = 0.0f;
    for (size_t index = 0; index < tileActual.size(); ++index)
      tileMaximumError = std::max(
          tileMaximumError, std::abs(tileActual[index] - tileExpected[index]));
    if (tileMaximumError > 0.0f)
      throw std::runtime_error("UdonATen Tile results diverged");
    const int64_t tooManyTileDimensions[5] = {1, 1, 1, 1, 1};
    require(udonatenTile(handle, tileInputDescriptor, tileInput.data(),
                         tooManyTileDimensions, 5, tileOutputDescriptor,
                         tileActual.data()),
            UDONDNN_STATUS_NOT_SUPPORTED, "reject Tile rank above four");
    const int64_t zeroTileRepeat[2] = {0, 3};
    require(udonatenTile(handle, tileInputDescriptor, tileInput.data(),
                         zeroTileRepeat, 2, tileOutputDescriptor,
                         tileActual.data()),
            UDONDNN_STATUS_NOT_SUPPORTED, "report unsupported empty Tile");
    const int64_t negativeTileRepeat[2] = {-1, 3};
    require(udonatenTile(handle, tileInputDescriptor, tileInput.data(),
                         negativeTileRepeat, 2, tileOutputDescriptor,
                         tileActual.data()),
            UDONDNN_STATUS_BAD_PARAM, "reject negative Tile repeat");
    require(udondnnDestroyTensorDescriptor(tileInputDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy Tile input descriptor");
    require(udondnnDestroyTensorDescriptor(tileOutputDescriptor),
            UDONDNN_STATUS_SUCCESS, "destroy Tile output descriptor");

    /* K=256 forces two kernel-208 pages while the public ATen call remains a
     * single whole-matrix operation. */
    constexpr int int4M = 3, int4N = 32, int4K = 256, int4Group = 32;
    std::vector<float> int4Input(int4M * int4K),
        int4Qparams((int4K / int4Group) * int4N * 2), int4Actual(int4M * int4N),
        int4Expected(int4M * int4N);
    std::vector<uint8_t> int4Packed(int4N * int4K / 2);
    for (int matrixRow = 0; matrixRow < int4M; ++matrixRow)
      for (int inner = 0; inner < int4K; ++inner)
        int4Input[matrixRow * int4K + inner] =
            value(inner + matrixRow * 19, 173 + matrixRow * 7);
    for (int block = 0; block < int4K / int4Group; ++block)
      for (int row = 0; row < int4N; ++row) {
        int qparams = (block * int4N + row) * 2;
        int4Qparams[qparams] = 0.03125f * float(row + block + 1);
        int4Qparams[qparams + 1] = 0.015625f * float(row - block);
      }
    for (int row = 0; row < int4N; ++row)
      for (int inner = 0; inner < int4K; inner += 2) {
        int low = (row * 5 + inner * 3 + 1) % 16 - 8;
        int high = (row * 7 + (inner + 1) * 5 + 3) % 16 - 8;
        int4Packed[row * (int4K / 2) + inner / 2] =
            uint8_t((low & 15) | ((high & 15) << 4));
      }
    for (int matrixRow = 0; matrixRow < int4M; ++matrixRow)
      for (int row = 0; row < int4N; ++row)
        for (int inner = 0; inner < int4K; ++inner) {
          uint8_t packed = int4Packed[row * (int4K / 2) + inner / 2];
          int nibble = inner % 2 ? packed >> 4 : packed & 15;
          int quantized = nibble >= 8 ? nibble - 16 : nibble;
          int block = inner / int4Group;
          int qparams = (block * int4N + row) * 2;
          int4Expected[matrixRow * int4N + row] +=
              (float(quantized) * int4Qparams[qparams] +
               int4Qparams[qparams + 1]) *
              int4Input[matrixRow * int4K + inner];
        }
    udondnnTensorDescriptor_t int4InputDescriptor = 0, int4PackedDescriptor = 0,
                              int4QparamsDescriptor = 0,
                              int4OutputDescriptor = 0;
    require(udondnnCreateTensorDescriptor(&int4InputDescriptor),
            UDONDNN_STATUS_SUCCESS, "create INT4 input descriptor");
    require(udondnnCreateTensorDescriptor(&int4PackedDescriptor),
            UDONDNN_STATUS_SUCCESS, "create INT4 packed descriptor");
    require(udondnnCreateTensorDescriptor(&int4QparamsDescriptor),
            UDONDNN_STATUS_SUCCESS, "create INT4 qparams descriptor");
    require(udondnnCreateTensorDescriptor(&int4OutputDescriptor),
            UDONDNN_STATUS_SUCCESS, "create INT4 output descriptor");
    require(
        udondnnSetTensor4dDescriptor(int4InputDescriptor, UDONDNN_TENSOR_NCHW,
                                     UDONDNN_DATA_FLOAT, int4M, int4K, 1, 1),
        UDONDNN_STATUS_SUCCESS, "set INT4 input descriptor");
    require(udondnnSetTensor4dDescriptor(
                int4PackedDescriptor, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT,
                int4N, int4K / 2, 1, 1),
            UDONDNN_STATUS_SUCCESS, "set INT4 packed descriptor");
    udondnnDataType_t queriedType = UDONDNN_DATA_FLOAT;
    int queriedN = 0, queriedC = 0, queriedH = 0, queriedW = 0;
    int queriedNStride = 0, queriedCStride = 0, queriedHStride = 0,
        queriedWStride = 0;
    require(udondnnGetTensor4dDescriptor(
                int4PackedDescriptor, &queriedType, &queriedN, &queriedC,
                &queriedH, &queriedW, &queriedNStride, &queriedCStride,
                &queriedHStride, &queriedWStride),
            UDONDNN_STATUS_SUCCESS, "get INT4 packed descriptor");
    if (queriedType != UDONDNN_DATA_FLOAT || queriedN != int4N ||
        queriedC != int4K / 2 || queriedH != 1 || queriedW != 1 ||
        queriedNStride != int4K / 2 || queriedCStride != 1 ||
        queriedHStride != 1 || queriedWStride != 1)
      throw std::runtime_error("UdonDNN tensor descriptor strides diverged");
    require(udondnnSetTensor4dDescriptor(
                int4QparamsDescriptor, UDONDNN_TENSOR_NCHW, UDONDNN_DATA_FLOAT,
                int4K / int4Group, int4N, 2, 1),
            UDONDNN_STATUS_SUCCESS, "set INT4 qparams descriptor");
    require(
        udondnnSetTensor4dDescriptor(int4OutputDescriptor, UDONDNN_TENSOR_NCHW,
                                     UDONDNN_DATA_FLOAT, int4M, int4N, 1, 1),
        UDONDNN_STATUS_SUCCESS, "set INT4 output descriptor");
    require(udonatenWeightInt4packMm(handle, int4InputDescriptor,
                                     int4Input.data(), int4PackedDescriptor,
                                     int4Packed.data(), int4Group,
                                     int4QparamsDescriptor, int4Qparams.data(),
                                     int4OutputDescriptor, int4Actual.data()),
            UDONDNN_STATUS_SUCCESS, "udonatenWeightInt4packMm");
    float int4MaximumError = 0.0f;
    for (size_t index = 0; index < int4Actual.size(); ++index)
      int4MaximumError = std::max(
          int4MaximumError, std::abs(int4Actual[index] - int4Expected[index]));
    if (int4MaximumError > 0.0005f)
      throw std::runtime_error("UdonATen INT4 results diverged");
    require(udonatenWeightInt4packMm(handle, int4InputDescriptor,
                                     int4Input.data(), int4PackedDescriptor,
                                     int4Packed.data(), 16,
                                     int4QparamsDescriptor, int4Qparams.data(),
                                     int4OutputDescriptor, int4Actual.data()),
            UDONDNN_STATUS_BAD_PARAM, "reject unsupported INT4 group size");
    require(udonatenWeightInt4packMm(handle, int4InputDescriptor,
                                     int4Input.data(), int4PackedDescriptor,
                                     int4Packed.data(), 64,
                                     int4QparamsDescriptor, int4Qparams.data(),
                                     int4OutputDescriptor, int4Actual.data()),
            UDONDNN_STATUS_NOT_SUPPORTED,
            "report recognized but unsupported INT4 group size");
    for (udondnnTensorDescriptor_t descriptor :
         {int4InputDescriptor, int4PackedDescriptor, int4QparamsDescriptor,
          int4OutputDescriptor})
      require(udondnnDestroyTensorDescriptor(descriptor),
              UDONDNN_STATUS_SUCCESS, "destroy INT4 descriptor");

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
                                         UDONDNN_DATA_FLOAT, 1, 17, 17, 17),
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
    udondnnTensorDescriptor_t batchNormInput = 0, batchNormOutput = 0,
                              batchNormParameter = 0;
    float batchNormInputValues[2] = {0.0f, 1.0f};
    float batchNormOutputValues[2] = {0.0f, 0.0f};
    float batchNormScale[1] = {1.0f}, batchNormBias[1] = {0.0f};
    float batchNormMean[1] = {0.0f}, batchNormVariance[1] = {1.0f};
    require(udondnnCreateTensorDescriptor(&batchNormInput),
            UDONDNN_STATUS_SUCCESS,
            "create invalid batchnorm input descriptor");
    require(udondnnCreateTensorDescriptor(&batchNormOutput),
            UDONDNN_STATUS_SUCCESS,
            "create invalid batchnorm output descriptor");
    require(udondnnCreateTensorDescriptor(&batchNormParameter),
            UDONDNN_STATUS_SUCCESS,
            "create invalid batchnorm parameter descriptor");
    require(udondnnSetTensor4dDescriptor(batchNormInput, UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 1, 1, 2),
            UDONDNN_STATUS_SUCCESS, "set invalid batchnorm input descriptor");
    require(udondnnSetTensor4dDescriptor(batchNormOutput, UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 1, 1, 2),
            UDONDNN_STATUS_SUCCESS, "set invalid batchnorm output descriptor");
    require(udondnnDeriveBNTensorDescriptor(batchNormParameter, batchNormInput,
                                            UDONDNN_BATCHNORM_SPATIAL),
            UDONDNN_STATUS_SUCCESS,
            "derive invalid batchnorm parameter descriptor");
    require(udondnnBatchNormalizationForwardInference(
                handle, UDONDNN_BATCHNORM_SPATIAL, &one, &zero, batchNormInput,
                batchNormInputValues, batchNormOutput, batchNormOutputValues,
                batchNormParameter, batchNormScale, batchNormBias,
                batchNormMean, batchNormVariance, 0.0f),
            UDONDNN_STATUS_BAD_PARAM, "reject batchnorm epsilon below minimum");
    require(udondnnDeriveBNTensorDescriptor(batchNormParameter, batchNormInput,
                                            udondnnBatchNormMode_t(99)),
            UDONDNN_STATUS_BAD_PARAM, "reject invalid batchnorm mode");
    require(udondnnDestroyTensorDescriptor(batchNormInput),
            UDONDNN_STATUS_SUCCESS,
            "destroy invalid batchnorm input descriptor");
    require(udondnnDestroyTensorDescriptor(batchNormOutput),
            UDONDNN_STATUS_SUCCESS,
            "destroy invalid batchnorm output descriptor");
    require(udondnnDestroyTensorDescriptor(batchNormParameter),
            UDONDNN_STATUS_SUCCESS,
            "destroy invalid batchnorm parameter descriptor");
    udondnnOpTensorDescriptor_t invalidOpTensor = 0;
    require(udondnnCreateOpTensorDescriptor(&invalidOpTensor),
            UDONDNN_STATUS_SUCCESS, "create invalid OpTensor descriptor");
    require(udondnnSetOpTensorDescriptor(
                invalidOpTensor, udondnnOpTensorOp_t(99), UDONDNN_DATA_FLOAT,
                UDONDNN_NOT_PROPAGATE_NAN),
            UDONDNN_STATUS_BAD_PARAM, "reject invalid OpTensor operation");
    require(udondnnDestroyOpTensorDescriptor(invalidOpTensor),
            UDONDNN_STATUS_SUCCESS, "destroy invalid OpTensor descriptor");
    require(udondnnSetOpTensorDescriptor(invalidOpTensor, UDONDNN_OP_TENSOR_ADD,
                                         UDONDNN_DATA_FLOAT,
                                         UDONDNN_NOT_PROPAGATE_NAN),
            UDONDNN_STATUS_BAD_PARAM, "reject stale OpTensor descriptor");
    udondnnOpTensorDescriptor_t broadcastOpTensor = 0;
    udondnnTensorDescriptor_t broadcastA = 0, broadcastB = 0, broadcastC = 0;
    float broadcastAValues[2] = {1.0f, 2.0f};
    float broadcastBValues[3] = {1.0f, 2.0f, 3.0f};
    float broadcastCValues[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    require(udondnnCreateOpTensorDescriptor(&broadcastOpTensor),
            UDONDNN_STATUS_SUCCESS, "create broadcast OpTensor descriptor");
    require(udondnnSetOpTensorDescriptor(
                broadcastOpTensor, UDONDNN_OP_TENSOR_ADD, UDONDNN_DATA_FLOAT,
                UDONDNN_NOT_PROPAGATE_NAN),
            UDONDNN_STATUS_SUCCESS, "set broadcast OpTensor descriptor");
    require(udondnnCreateTensorDescriptor(&broadcastA), UDONDNN_STATUS_SUCCESS,
            "create incompatible broadcast A");
    require(udondnnCreateTensorDescriptor(&broadcastB), UDONDNN_STATUS_SUCCESS,
            "create incompatible broadcast B");
    require(udondnnCreateTensorDescriptor(&broadcastC), UDONDNN_STATUS_SUCCESS,
            "create incompatible broadcast C");
    require(udondnnSetTensor4dDescriptor(broadcastA, UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 1, 1, 2),
            UDONDNN_STATUS_SUCCESS, "set incompatible broadcast A");
    require(udondnnSetTensor4dDescriptor(broadcastB, UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 1, 1, 3),
            UDONDNN_STATUS_SUCCESS, "set incompatible broadcast B");
    require(udondnnSetTensor4dDescriptor(broadcastC, UDONDNN_TENSOR_NCHW,
                                         UDONDNN_DATA_FLOAT, 1, 1, 1, 4),
            UDONDNN_STATUS_SUCCESS, "set incompatible broadcast C");
    require(udondnnOpTensor(handle, broadcastOpTensor, &one, broadcastA,
                            broadcastAValues, &one, broadcastB,
                            broadcastBValues, &zero, broadcastC,
                            broadcastCValues),
            UDONDNN_STATUS_BAD_PARAM, "reject incompatible OpTensor broadcast");
    require(udondnnDestroyOpTensorDescriptor(broadcastOpTensor),
            UDONDNN_STATUS_SUCCESS, "destroy broadcast OpTensor descriptor");
    require(udondnnDestroyTensorDescriptor(broadcastA), UDONDNN_STATUS_SUCCESS,
            "destroy incompatible broadcast A");
    require(udondnnDestroyTensorDescriptor(broadcastB), UDONDNN_STATUS_SUCCESS,
            "destroy incompatible broadcast B");
    require(udondnnDestroyTensorDescriptor(broadcastC), UDONDNN_STATUS_SUCCESS,
            "destroy incompatible broadcast C");
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
              << " batchnorm_outputs=" << batchNormOutputs
              << " batchnorm_cases=" << batchNormCases.size()
              << " batchnorm_max_error=" << batchNormMaximumError
              << " optensor_outputs=" << opTensorOutputs
              << " optensor_cases=" << opTensorCases.size()
              << " optensor_max_error=" << opTensorMaximumError
              << " rmsnorm_outputs=" << rmsActual.size()
              << " rmsnorm_max_error=" << rmsMaximumError
              << " gather_outputs=" << gatherActual.size()
              << " gather_max_error=" << gatherMaximumError
              << " int4_outputs=" << int4Actual.size()
              << " int4_max_error=" << int4MaximumError
              << " tile_outputs=" << tileActual.size()
              << " tile_max_error=" << tileMaximumError
              << " renderer=" << session.deviceName()
              << " draws=" << session.drawCount() << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
