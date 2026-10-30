#pragma once

#define MNIST_CNN_INPUT_WIDTH 28
#define MNIST_CNN_INPUT_WORDS 784
#define MNIST_CNN_CLASSES 10

#ifdef __cplusplus
extern "C" {
#endif
void mnist_cnn_infer(const float *input, float *logits);
int mnist_cnn_predict(const float *input);
#ifdef __cplusplus
}
#endif
