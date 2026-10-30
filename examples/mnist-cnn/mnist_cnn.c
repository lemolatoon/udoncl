#include "mnist_cnn.h"
#include "weights.h"

enum {
  kConvChannels = 16,
  kConvWidth = 24,
  kPoolWidth = 3,
  kPoolStride = 8,
  kFeatures = kPoolWidth * kPoolWidth * kConvChannels,
};

void mnist_cnn_infer(const float *input, float *logits) {
  float convolution[kConvWidth * kConvWidth * kConvChannels];
  float features[kFeatures];

  for (int y = 0; y < kConvWidth; ++y) {
    for (int x = 0; x < kConvWidth; ++x) {
      for (int channel = 0; channel < kConvChannels; ++channel) {
        float sum = kMnistConvBias[channel];
        for (int kernel_y = 0; kernel_y < 5; ++kernel_y) {
          for (int kernel_x = 0; kernel_x < 5; ++kernel_x) {
            int input_index =
                (y + kernel_y) * MNIST_CNN_INPUT_WIDTH + x + kernel_x;
            int weight_index = (channel * 5 + kernel_y) * 5 + kernel_x;
            sum += input[input_index] * kMnistConvWeight[weight_index];
          }
        }
        int output_index = (y * kConvWidth + x) * kConvChannels + channel;
        convolution[output_index] = sum > 0.0f ? sum : 0.0f;
      }
    }
  }

  for (int pool_y = 0; pool_y < kPoolWidth; ++pool_y) {
    for (int pool_x = 0; pool_x < kPoolWidth; ++pool_x) {
      for (int channel = 0; channel < kConvChannels; ++channel) {
        float sum = 0.0f;
        for (int y = 0; y < kPoolStride; ++y) {
          for (int x = 0; x < kPoolStride; ++x) {
            int conv_y = pool_y * kPoolStride + y;
            int conv_x = pool_x * kPoolStride + x;
            int index =
                (conv_y * kConvWidth + conv_x) * kConvChannels + channel;
            sum += convolution[index];
          }
        }
        int feature_index =
            (pool_y * kPoolWidth + pool_x) * kConvChannels + channel;
        features[feature_index] = sum * (1.0f / 64.0f);
      }
    }
  }

  for (int digit = 0; digit < MNIST_CNN_CLASSES; ++digit) {
    float sum = kMnistDenseBias[digit];
    for (int feature = 0; feature < kFeatures; ++feature) {
      sum += features[feature] * kMnistDenseWeight[digit * kFeatures + feature];
    }
    logits[digit] = sum;
  }
}

int mnist_cnn_predict(const float *input) {
  float logits[MNIST_CNN_CLASSES];
  mnist_cnn_infer(input, logits);
  int prediction = 0;
  for (int digit = 1; digit < MNIST_CNN_CLASSES; ++digit) {
    if (logits[digit] > logits[prediction])
      prediction = digit;
  }
  return prediction;
}
